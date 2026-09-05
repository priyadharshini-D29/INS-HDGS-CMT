"""
================================================================
Grounding the soft-rule premises in electrodes and frequency bands
(Reviewer 2, comment 7 — replaces the latent z_k indices of Fig. 8D)
================================================================
The rule premises q_r live in a 64-d latent key space, so decoding them
by their largest latent coordinates (z46, z22, ...) is not interpretable.
Instead, this script projects every rule back onto the PHYSICAL input
space by attribution: for rule r, the activation a_r (Eq. 15) is
differentiated with respect to the model inputs

  eeg_windows (B, W, C, 5)  : band power per window x electrode x band
                              (delta, theta, alpha, beta, gamma)
  et_seq      (B, T, 3)     : gaze x, gaze y, pupil
  roi_vector  (B, N_rois)   : ROI dwell histogram

and the signed attribution  input * d a_r / d input  is aggregated over
windows / time and averaged over held-out epochs. The result is, per
rule, a 19 x 5 electrode-by-band map (plus the three ET streams and the
ROI cells) whose largest |entries| are the grounded premises, with the
sign giving the direction (↑ = higher power raises the rule activation).
The rule conclusion (HIGH/LOW) comes from the rule head l_r evaluated at
the epochs on which the rule dominates.

Integrated gradients (zero baseline, --ig-steps) is used by default so
the attribution satisfies completeness; --method gradxinput is faster.

Outputs
-------
  results/explainability/rule_grounding_<label>.json   full maps
  results/explainability/rule_grounding_<label>.md     top-k premises per rule
  results/explainability/fig_rule_grounding_<label>.pdf/.png
      (8 electrode x band heatmaps + grounded IF-THEN text panel)

Usage
-----
  cd src/model
  CUDA_VISIBLE_DEVICES="" python ../../scripts/analysis/ground_rules_to_electrodes.py \
      --ckpt output/checkpoints/<label>/<label>_fold01_e0.pt --subjects S01 --label <label>
  (use --subjects all for population-level maps; several --ckpt may be given
   to average over ensemble members / folds)
================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "src" / "model"
for p in (str(MODEL), str(MODEL.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config.settings import EEG_BANDS, SUBJECT_IDS                       # noqa: E402
from data.channel_harmonizer import CANONICAL_CHANNELS                   # noqa: E402
from data.dataset import NeumaGraphDataset, build_dataloaders            # noqa: E402
from evaluation.losocv import _make_model                                # noqa: E402
from models.ins_hdgs_cmt import AblationConfig                           # noqa: E402

OUT = ROOT / "results" / "explainability"
BANDS = list(EEG_BANDS.keys())
ET_NAMES = ["gaze_x", "gaze_y", "pupil"]
CLASS = ["LOW", "HIGH"]


def load_state(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    return {k.replace("module.", ""): v for k, v in sd.items()}


def attribute(model, batch, n_rules, method="ig", ig_steps=16):
    """Return dict of signed attributions per rule:
       eeg (R, W, C, 5), et (R, T, 3), roi (R, N)   for this batch (summed over epochs),
       plus rule activations (B, R), rule margins (B, R) and final margins (B,)."""
    x_eeg = batch["eeg_windows"].float()
    x_et = batch["et_seq"].float()
    x_roi = batch["roi_vector"].float()
    adj = batch["adj_matrices"].float(); wadj = batch["weighted_adjs"].float()
    B = x_eeg.shape[0]
    alphas = np.linspace(0, 1, ig_steps + 1)[1:] if method == "ig" else np.array([1.0])
    acc = {"eeg": torch.zeros(n_rules, *x_eeg.shape[1:]), "et": torch.zeros(n_rules, *x_et.shape[1:]),
           "roi": torch.zeros(n_rules, *x_roi.shape[1:])}
    for a in alphas:
        xe = (a * x_eeg).clone().requires_grad_(True)
        xt = (a * x_et).clone().requires_grad_(True)
        xr = (a * x_roi).clone().requires_grad_(True)
        out = model(eeg_windows=xe, adj_matrices=adj, et_seq=xt, roi_vector=xr, weighted_adjs=wadj)
        act = out["rule_act"]                                   # (B, R)
        for r in range(n_rules):
            g = torch.autograd.grad(act[:, r].sum(), [xe, xt, xr], retain_graph=True, allow_unused=True)
            for key, x, gi in zip(["eeg", "et", "roi"], [x_eeg, x_et, x_roi], g):
                if gi is not None:
                    acc[key][r] += (gi.detach() * x).sum(0) / len(alphas)   # IG ≈ x * mean_a grad
    with torch.no_grad():
        out = model(eeg_windows=x_eeg, adj_matrices=adj, et_seq=x_et, roi_vector=x_roi, weighted_adjs=wadj)
    act = out["rule_act"].numpy()
    # per-rule class conclusion on these epochs: l_r evaluated on the keys
    rl = model.rule_layer
    keys = rl.key_proj(out["fused"])
    rule_margin = torch.stack([h(keys)[:, 1] - h(keys)[:, 0] for h in rl.rule_heads], 1).detach().numpy()   # (B, R)
    final_margin = (out["logits"][:, 1] - out["logits"][:, 0]).numpy()
    return {k: v.numpy() for k, v in acc.items()}, act, rule_margin, final_margin, B


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--subjects", default="S01", help="comma list or 'all'")
    ap.add_argument("--label", default="model")
    ap.add_argument("--method", choices=["ig", "gradxinput"], default="ig")
    ap.add_argument("--ig-steps", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    subjects = SUBJECT_IDS if args.subjects == "all" else args.subjects.split(",")
    ds = NeumaGraphDataset(subject_ids=subjects, precompute_graphs=True, augment=False)
    _, loader = build_dataloaders(ds, ds, batch_size=16, num_workers=0)
    chan = list(CANONICAL_CHANNELS)[:ds.n_eeg_ch]

    n_rules = None
    tot = None; acts = []; rmarg = []; fmarg = []; n_epochs = 0
    for ck in args.ckpt:
        model = _make_model(ds.n_eeg_ch, ds.n_classes, AblationConfig.full(), ds.n_et_ch).eval()
        missing, _ = model.load_state_dict(load_state(Path(ck)), strict=False)
        if missing:
            print(f"[warn] {Path(ck).name}: {len(missing)} missing keys")
        n_rules = model.rule_layer.n_rules
        for b in loader:
            attr, act, rm, fm, B = attribute(model, b, n_rules, args.method, args.ig_steps)
            if tot is None:
                tot = {k: np.zeros_like(v) for k, v in attr.items()}
            for k in tot:
                tot[k] += attr[k]
            acts.append(act); rmarg.append(rm); fmarg.append(fm); n_epochs += B
    n_norm = n_epochs   # attributions were summed over epochs and checkpoints
    eeg = tot["eeg"].sum(axis=1) / n_norm            # (R, C, 5): summed over windows, mean over epochs
    et = tot["et"].sum(axis=1) / n_norm              # (R, 3)
    roi = tot["roi"] / n_norm                        # (R, N)
    A = np.concatenate(acts); RM = np.concatenate(rmarg); FM = np.concatenate(fmarg)
    dom = A.argmax(1)

    rules = []
    for r in range(n_rules):
        flat = eeg[r].ravel()
        order = np.argsort(-np.abs(flat))[: args.top_k]
        prem = [dict(electrode=chan[i // 5], band=BANDS[i % 5], attribution=float(flat[i]),
                     direction="up" if flat[i] > 0 else "down") for i in order]
        et_top = int(np.argmax(np.abs(et[r])))
        m = RM[dom == r]
        concl = CLASS[int(m.mean() > 0)] if len(m) else CLASS[int(RM[:, r].mean() > 0)]
        rules.append(dict(rule=r + 1, conclusion=concl, dominant_frac=float((dom == r).mean()),
                          mean_activation=float(A[:, r].mean()), premises=prem,
                          et_premise=dict(stream=ET_NAMES[et_top], attribution=float(et[r][et_top]),
                                          direction="up" if et[r][et_top] > 0 else "down"),
                          roi_top_cell=int(np.argmax(np.abs(roi[r]))),
                          eeg_share=float(np.abs(eeg[r]).sum() / (np.abs(eeg[r]).sum() + np.abs(et[r]).sum() + np.abs(roi[r]).sum() + 1e-12))))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = args.label
    json.dump(dict(label=tag, checkpoints=args.ckpt, subjects=subjects, n_epochs=int(n_epochs), method=args.method,
                   electrodes=chan, bands=BANDS, eeg_attr=eeg.tolist(), et_attr=et.tolist(), roi_attr=roi.tolist(),
                   rules=rules), open(out / f"rule_grounding_{tag}.json", "w"), indent=1)

    arrow = {"up": "↑", "down": "↓"}
    lines = [f"# Grounded soft rules — {tag} ({n_epochs} epochs, {len(args.ckpt)} checkpoint(s), {args.method})", "",
             "Premise terms are (electrode, band) node features ranked by |integrated-gradient attribution| of the rule "
             "activation a_r; ↑ means higher band power at that electrode increases the rule's activation. "
             "The conclusion is the sign of the rule head l_r on the epochs where the rule dominates.", "",
             "| rule | conclusion | dominant in | EEG share | grounded premise (top-k electrode–band) | ET term |", "|---|---|---|---|---|---|"]
    text_panel = []
    for r in rules:
        prem = " ∧ ".join(f"{p['electrode']}-{p['band']}{arrow[p['direction']]}" for p in r["premises"])
        etp = f"{r['et_premise']['stream']}{arrow[r['et_premise']['direction']]}"
        lines.append(f"| {r['rule']} | {r['conclusion']} | {r['dominant_frac']*100:.0f}% | {r['eeg_share']*100:.0f}% | {prem} | {etp} |")
        text_panel.append(f"R{r['rule']}: IF {prem} ∧ {etp} → {r['conclusion']}")
    (out / f"rule_grounding_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # figure: 8 heatmaps (electrode x band) + text panel
    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        R = n_rules
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, R // 2 + 2, width_ratios=[1] * (R // 2) + [0.08, 1.6], wspace=0.35, hspace=0.45)
        vmax = np.abs(eeg).max()
        for r in range(R):
            ax = fig.add_subplot(gs[r % 2, r // 2])
            im = ax.imshow(eeg[r], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"Rule {r+1} → {rules[r]['conclusion']}", fontsize=10, fontweight="bold")
            ax.set_xticks(range(5)); ax.set_xticklabels([b[:1].upper() + b[1:] for b in BANDS], rotation=45, fontsize=7)
            ax.set_yticks(range(len(chan))); ax.set_yticklabels(chan, fontsize=6)
        cax = fig.add_subplot(gs[:, R // 2]); fig.colorbar(im, cax=cax, label="attribution of a_r (IG, signed)")
        axT = fig.add_subplot(gs[:, R // 2 + 1]); axT.axis("off")
        axT.set_title("Grounded IF–THEN summaries", fontweight="bold", loc="left")
        axT.text(0, 0.98, "\n".join(text_panel), va="top", ha="left", fontsize=10.5, family="monospace", linespacing=1.9,
                 transform=axT.transAxes, wrap=True)
        fig.suptitle(f"Soft-rule premises projected onto scalp electrodes × frequency bands ({tag})", fontweight="bold")
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_rule_grounding_{tag}.{ext}", dpi=300, bbox_inches="tight")
        print(f"[grounding] figure → {out / f'fig_rule_grounding_{tag}.pdf'}")
    except Exception as e:  # pragma: no cover
        print(f"[grounding] figure skipped: {e}")


if __name__ == "__main__":
    main()
