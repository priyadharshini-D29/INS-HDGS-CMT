"""
================================================================
INS-HDGS-CMT — HIGH vs LOW engagement case study
================================================================
Two representative held-out epochs, traced through the SAME trained model:

  • HIGH case : S24 (fold-22) — most-confident correctly-classified HIGH epoch
  • LOW  case : S30 (fold-28) — most-confident correctly-classified LOW  epoch

Both use the canonical production ensemble (repro_focal_g3p0_effective_num_37)
and the model's actual inputs from NeumaGraphDataset (no synthetic data):

  eeg_windows (10,24,5)  band-power  · adj/weighted_adjs (10,24,24) connectivity
  et_seq (600,3) gaze    · roi_vector (10) attended-region saliency

Outputs
-------
  figures/case_study_high_low/fig_regions.{png,pdf}   region × band, connectivity
  figures/case_study_high_low/fig_gaze_pred.{png,pdf} gaze, ROI, prediction, IG
  results/case_study/high_low_comparison.{csv,md,tex} side-by-side metrics table
  results/case_study/high_low_provenance.json

Run:  python analysis/case_study_high_low.py
================================================================
"""
import os
import sys
import json
from pathlib import Path

PHASE8 = Path(__file__).resolve().parents[1]
os.chdir(PHASE8)
sys.path.insert(0, str(PHASE8))
sys.path.insert(0, str(PHASE8.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# Publication clarity: larger, fully-black text on every panel + HD raster.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.labelweight": "bold",
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10,
    "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.titlecolor": "black",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Standard numeric precision for ALL figures: 3 decimal places on every axis
# and in every in-figure number, so panels are visually consistent (3 chosen
# over 2 so the small within-subject deltas (~0.005-0.016) stay legible).
DECIMALS = 3
FMT = mticker.FormatStrFormatter(f"%.{DECIMALS}f")


def _std_axis(a, xlabel=None, ylabel=None, fmt_x=False, fmt_y=True):
    """Apply the shared figure standard: axis labels + 3-decimal tick formatting."""
    if xlabel is not None:
        a.set_xlabel(xlabel)
    if ylabel is not None:
        a.set_ylabel(ylabel)
    if fmt_y:
        a.yaxis.set_major_formatter(FMT)
    if fmt_x:
        a.xaxis.set_major_formatter(FMT)

from data.dataset import NeumaGraphDataset
from models.ins_hdgs_cmt import AblationConfig
import main as neuma_main
from data.channel_harmonizer import CANONICAL_CHANNELS

DEVICE = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
FOLD_TAG = "repro_focal_g3p0_effective_num_37"
CKPT_DIR = PHASE8 / "output" / "checkpoints" / FOLD_TAG

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
FRONTAL = ["Fz", "F3", "F4", "FC1", "FC2"]
POSTERIOR = ["Pz", "Oz", "O1", "O2", "P3", "P4", "P7", "P8"]
THETA, ALPHA = 1, 2

OUT_FIG = PHASE8 / "figures" / "case_study_high_low"; OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_RES = PHASE8 / "results" / "case_study"; OUT_RES.mkdir(parents=True, exist_ok=True)

CASES = {  # name -> (subject, fold_number, opt_threshold, target_label)
    "HIGH": dict(subj="S24", fold=22, thr=0.38, label=1),
    "LOW":  dict(subj="S30", fold=28, thr=0.44, label=0),
}


def log(m): print(m, flush=True)


def load_ensemble(fold, n_eeg_ch):
    members = []
    for ck in sorted(CKPT_DIR.glob(f"{FOLD_TAG}_fold{fold:02d}_e*.pt")):
        m = neuma_main._make_model(n_eeg_ch=n_eeg_ch, n_classes=2,
                                   ablation=AblationConfig.full()).to(DEVICE).eval()
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        m.load_state_dict(sd["model_state_dict"], strict=True)
        members.append(m)
    return members


def collate(sample):
    def t(x):
        x = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x, np.float32))
        return x.float().unsqueeze(0).to(DEVICE)
    return dict(eeg_windows=t(sample["eeg_windows"]), adj_matrices=t(sample["adj_matrices"]),
                et_seq=t(sample["et_seq"]), roi_vector=t(sample["roi_vector"]),
                weighted_adjs=t(sample["weighted_adjs"]))


def integrated_gradients(members, sample, baseline_inputs, steps=24):
    """Mean |IG| per modality for the HIGH logit, averaged over ensemble."""
    keys = ["eeg_windows", "et_seq", "roi_vector", "weighted_adjs"]
    kw = collate(sample)
    base = {k: torch.as_tensor(baseline_inputs[k], dtype=torch.float32).unsqueeze(0) for k in keys}
    fixed = {"adj_matrices": kw["adj_matrices"]}
    out = {k: 0.0 for k in keys}
    for m in members:
        accum = {k: torch.zeros_like(kw[k]) for k in keys}
        for a in torch.linspace(0, 1, steps):
            xs = {k: (base[k] + a * (kw[k] - base[k])).requires_grad_(True) for k in keys}
            o = m(eeg_windows=xs["eeg_windows"], adj_matrices=fixed["adj_matrices"],
                  et_seq=xs["et_seq"], roi_vector=xs["roi_vector"], weighted_adjs=xs["weighted_adjs"])
            tgt = o["logits"][:, 1].sum()
            grads = torch.autograd.grad(tgt, list(xs.values()), allow_unused=True)
            for k, g in zip(keys, grads):
                if g is not None and torch.isfinite(g).all():
                    accum[k] = accum[k] + g.detach()
        for k in keys:
            ig = ((kw[k] - base[k]) * accum[k] / steps).abs().mean().item()
            out[k] += ig / len(members)
    return out


def analyse_case(name, cfg):
    log(f"\n{'='*60}\n[{name}] {cfg['subj']} fold-{cfg['fold']:02d}\n{'='*60}")
    ds = NeumaGraphDataset(subject_ids=[cfg["subj"]], norm_mode="zscore",
                           label_mode="global", precompute_graphs=True, use_disk_cache=True)
    n_eeg_ch = np.asarray(ds[0]["eeg_windows"]).shape[1]
    members = load_ensemble(cfg["fold"], n_eeg_ch)
    labels = np.array([int(ds[i]["label"]) for i in range(len(ds))])

    probs = np.zeros((len(ds), len(members)))
    with torch.no_grad():
        for i in range(len(ds)):
            kw = collate(ds[i])
            for j, m in enumerate(members):
                probs[i, j] = torch.softmax(m(**kw)["logits"], -1)[0, 1].item()
    mean_p = probs.mean(1)
    pred = (mean_p >= cfg["thr"]).astype(int)

    # representative = most-confident correctly classified epoch of the target class
    conf = np.abs(mean_p - 0.5)
    cand = [i for i in range(len(ds)) if pred[i] == labels[i] == cfg["label"]]
    rep = max(cand, key=lambda i: conf[i]) if cand else int(np.argmax(conf if cfg["label"] else -conf))
    log(f"    rep epoch={rep}  true={labels[rep]} pred={pred[rep]}  p(HIGH)={mean_p[rep]:.3f}")

    s = ds[rep]
    eeg = np.asarray(s["eeg_windows"], np.float32)       # (10,24,5)
    adj = np.asarray(s["weighted_adjs"], np.float32).mean(0)  # (24,24)
    roi = np.asarray(s["roi_vector"], np.float32)        # (10,)
    et = np.asarray(s["et_seq"], np.float32)             # (600,3)

    ch = CANONICAL_CHANNELS[:n_eeg_ch]
    fidx = [ch.index(c) for c in FRONTAL if c in ch]
    pidx = [ch.index(c) for c in POSTERIOR if c in ch]
    bandpow = eeg.mean(0)                                # (24,5) avg over windows
    region_band = {
        "frontal": bandpow[fidx].mean(0),                # (5,)
        "posterior": bandpow[pidx].mean(0),
    }
    # gaze dispersion / entropy on (x,y); pupil mean
    xy = et[:, :2]
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=8)
    p = H / max(H.sum(), 1); p = p[p > 0]
    gaze_entropy = float(-(p * np.log(p)).sum())
    gaze_disp = float(np.sqrt(xy.var(0).sum()))
    pupil = float(et[:, 2].mean())

    # connectivity summaries
    def block(a, ii, jj): return float(a[np.ix_(ii, jj)].mean())
    conn = dict(frontal=block(adj, fidx, fidx), posterior=block(adj, pidx, pidx),
                fronto_posterior=block(adj, fidx, pidx))

    # IG: baseline = per-feature mean over this subject's epochs
    keys = ["eeg_windows", "et_seq", "roi_vector", "weighted_adjs"]
    base = {k: np.mean([np.asarray(ds[i][k], np.float32) for i in range(len(ds))], 0) for k in keys}
    ig = integrated_gradients(members, s, base)

    # within-subject HIGH vs LOW contrast (removes cross-subject z-score confound)
    def class_marker(lbl, fb_idx, band):
        idxs = [i for i in range(len(ds)) if labels[i] == lbl]
        if not idxs:
            return float("nan")
        bp = np.mean([np.asarray(ds[i]["eeg_windows"], np.float32).mean(0) for i in idxs], 0)
        return float(bp[fb_idx, band].mean())
    within = dict(
        ftheta_high=class_marker(1, fidx, THETA), ftheta_low=class_marker(0, fidx, THETA),
        palpha_high=class_marker(1, pidx, ALPHA), palpha_low=class_marker(0, pidx, ALPHA),
    )
    within["d_ftheta"] = within["ftheta_high"] - within["ftheta_low"]
    within["d_palpha"] = within["palpha_high"] - within["palpha_low"]

    return dict(name=name, subj=cfg["subj"], fold=cfg["fold"], rep=int(rep),
                true=int(labels[rep]), pred=int(pred[rep]), p_high=float(mean_p[rep]),
                eeg=eeg, bandpow=bandpow, region_band=region_band, adj=adj, roi=roi, et=et,
                ch=ch, fidx=fidx, pidx=pidx, conn=conn,
                gaze_entropy=gaze_entropy, gaze_disp=gaze_disp, pupil=pupil,
                frontal_theta=float(region_band["frontal"][THETA]),
                posterior_alpha=float(region_band["posterior"][ALPHA]),
                within=within,
                ig=ig, n_low=int((labels == 0).sum()), n_high=int((labels == 1).sum()))


# ── run both cases ───────────────────────────────────────────────────────────
R = {name: analyse_case(name, cfg) for name, cfg in CASES.items()}
H, L = R["HIGH"], R["LOW"]

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — regional EEG signatures + connectivity
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for col, (C, ttl) in enumerate([(H, "HIGH engagement (S24)"), (L, "LOW engagement (S30)")]):
    # region × band grouped bars
    x = np.arange(len(BANDS)); w = 0.38
    ax[col, 0].bar(x - w/2, C["region_band"]["frontal"], w, label="Frontal", color="#d1495b")
    ax[col, 0].bar(x + w/2, C["region_band"]["posterior"], w, label="Posterior", color="#30638e")
    ax[col, 0].set_xticks(x); ax[col, 0].set_xticklabels(BANDS, rotation=30)
    ax[col, 0].set_title(f"{ttl}\nBand power by region (z-scored)"); ax[col, 0].legend(fontsize=8)
    ax[col, 0].axhline(0, color="k", lw=0.5)
    _std_axis(ax[col, 0], xlabel="Frequency band", ylabel="Band power (z-scored)")
    # connectivity heatmap (mask self-edges so off-diagonal structure is visible)
    A = C["adj"].copy(); np.fill_diagonal(A, np.nan)
    im = ax[col, 1].imshow(A, cmap="magma", aspect="auto")
    ax[col, 1].set_title("Functional connectivity (off-diagonal)")
    _std_axis(ax[col, 1], xlabel="EEG channel", ylabel="EEG channel", fmt_y=False)
    cb = plt.colorbar(im, ax=ax[col, 1], fraction=0.046)
    cb.ax.yaxis.set_major_formatter(FMT)
    cb.set_label("Edge weight")
    # within-subject HIGH-minus-LOW contrast (absolute z-scored values are per
    # subject, so the meaningful, comparable quantity is each subject's own
    # HIGH vs LOW difference). Positive frontal-theta delta = engagement signature.
    ax[col, 2].bar(["$\\Delta$ Frontal $\\theta$", "$\\Delta$ Posterior $\\alpha$"],
                   [C["within"]["d_ftheta"], C["within"]["d_palpha"]],
                   color=["#d1495b", "#30638e"])
    ax[col, 2].axhline(0, color="k", lw=0.8)
    _std_axis(ax[col, 2], xlabel="Marker", ylabel="$\\Delta$ band power (HIGH$-$LOW)")
    ax[col, 2].set_title(f"Within-subject $\\Delta$ (HIGH$-$LOW)\n"
                         f"model p(HIGH)={C['p_high']:.3f}")
fig.suptitle("INS-HDGS-CMT — regional EEG signatures: HIGH vs LOW engagement", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"): fig.savefig(OUT_FIG / f"fig_regions.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"\n[fig] → {OUT_FIG}/fig_regions.png/pdf")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — gaze, ROI, prediction, IG attribution
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(2, 4, figsize=(18, 8))
IGN = {"eeg_windows": "EEG", "et_seq": "Gaze", "roi_vector": "ROI", "weighted_adjs": "Conn."}
for col, (C, ttl) in enumerate([(H, "HIGH (S24)"), (L, "LOW (S30)")]):
    xy = C["et"][:, :2]
    ax[col, 0].plot(xy[:, 0], xy[:, 1], lw=0.6, color="#444"); ax[col, 0].scatter(xy[:, 0], xy[:, 1], s=4, c=np.arange(len(xy)), cmap="viridis")
    ax[col, 0].set_title(f"{ttl}\nGaze trajectory (entropy={C['gaze_entropy']:.3f})")
    _std_axis(ax[col, 0], xlabel="Gaze x (norm.)", ylabel="Gaze y (norm.)", fmt_x=True, fmt_y=True)
    ax[col, 1].bar(np.arange(len(C["roi"])), C["roi"], color="#3f7d20")
    ax[col, 1].set_title("ROI saliency (10 windows)")
    _std_axis(ax[col, 1], xlabel="Window index", ylabel="ROI saliency")
    ax[col, 2].bar(["LOW", "HIGH"], [1 - C["p_high"], C["p_high"]], color=["#30638e", "#d1495b"])
    ax[col, 2].set_ylim(0, 1); ax[col, 2].set_title(f"Prediction  (true={'HIGH' if C['true'] else 'LOW'})")
    _std_axis(ax[col, 2], xlabel="Class", ylabel="Probability")
    tot = sum(C["ig"].values()) or 1
    vals = [C["ig"][k] / tot for k in C["ig"]]
    ax[col, 3].bar([IGN[k] for k in C["ig"]], vals, color="#9b5de5")
    ax[col, 3].set_title("IG attribution (norm.)")
    _std_axis(ax[col, 3], xlabel="Modality", ylabel="Attribution (norm.)")
fig.suptitle("INS-HDGS-CMT — gaze, ROI, decision & attribution: HIGH vs LOW", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"): fig.savefig(OUT_FIG / f"fig_gaze_pred.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"[fig] → {OUT_FIG}/fig_gaze_pred.png/pdf")

# ════════════════════════════════════════════════════════════════════════════
# TABLE — side-by-side comparison
# ════════════════════════════════════════════════════════════════════════════
def ign(C): tot = sum(C["ig"].values()) or 1; return {k: C["ig"][k]/tot for k in C["ig"]}
igh, igl = ign(H), ign(L)
rows = [
    ("Subject / fold", H["subj"]+f" / {H['fold']}", L["subj"]+f" / {L['fold']}", "—"),
    ("Reference label (ET-derived)", "HIGH", "LOW", "both correctly classified"),
    ("Model p(HIGH)", f"{H['p_high']:.3f}", f"{L['p_high']:.3f}", "confident, opposite poles"),
    ("Within-subj Δ frontal θ (HIGH−LOW)", f"{H['within']['d_ftheta']:+.3f}", f"{L['within']['d_ftheta']:+.3f}", ">0 ⇒ frontal theta higher in HIGH (attentional control)"),
    ("Within-subj Δ posterior α (HIGH−LOW)", f"{H['within']['d_palpha']:+.3f}", f"{L['within']['d_palpha']:+.3f}", "<0 ⇒ posterior alpha suppressed in HIGH (engagement)"),
    ("Frontal connectivity (this epoch)", f"{H['conn']['frontal']:.3f}", f"{L['conn']['frontal']:.3f}", "fronto-frontal integration"),
    ("Fronto-posterior conn. (this epoch)", f"{H['conn']['fronto_posterior']:.3f}", f"{L['conn']['fronto_posterior']:.3f}", "long-range coupling"),
    ("Gaze entropy (this epoch)", f"{H['gaze_entropy']:.3f}", f"{L['gaze_entropy']:.3f}", "scattered looking (single-epoch, noisy)"),
    ("Pupil mean (this epoch)", f"{H['pupil']:.3f}", f"{L['pupil']:.3f}", "arousal proxy"),
    ("IG — ROI", f"{igh['roi_vector']:.2f}", f"{igl['roi_vector']:.2f}", "attended stimulus region"),
    ("IG — EEG", f"{igh['eeg_windows']:.2f}", f"{igl['eeg_windows']:.2f}", "band power"),
    ("IG — Gaze", f"{igh['et_seq']:.2f}", f"{igl['et_seq']:.2f}", "fixation location"),
    ("IG — Connectivity", f"{igh['weighted_adjs']:.2f}", f"{igl['weighted_adjs']:.2f}", "graph edges"),
]
tab = pd.DataFrame(rows, columns=["Metric", "HIGH (S24)", "LOW (S30)", "Interpretation"])
tab.to_csv(OUT_RES / "high_low_comparison.csv", index=False)

md = ["# HIGH vs LOW engagement — case-study comparison\n",
      f"HIGH: {H['subj']} fold-{H['fold']} epoch {H['rep']} (p(HIGH)={H['p_high']:.3f}).  "
      f"LOW: {L['subj']} fold-{L['fold']} epoch {L['rep']} (p(HIGH)={L['p_high']:.3f}).\n",
      "| " + " | ".join(tab.columns) + " |", "|" + "---|" * len(tab.columns)]
for _, r in tab.iterrows():
    md.append("| " + " | ".join(str(r[c]) for c in tab.columns) + " |")
(OUT_RES / "high_low_comparison.md").write_text("\n".join(md) + "\n")

tex = [r"\begin{table}[t]", r"\centering",
       r"\caption{HIGH vs LOW engagement case study (representative held-out epochs, canonical ensemble).}",
       r"\label{tab:case_study}", r"\begin{tabular}{lccl}", r"\toprule",
       r"Metric & HIGH (S24) & LOW (S30) & Interpretation \\", r"\midrule"]
for _, r in tab.iterrows():
    tex.append(" & ".join(str(r[c]).replace("θ", r"$\theta$").replace("α", r"$\alpha$").replace("↑", r"$\uparrow$") for c in tab.columns) + r" \\")
tex += [r"\botrule", r"\end{tabular}", r"\end{table}"]
(OUT_RES / "high_low_comparison.tex").write_text("\n".join(tex) + "\n")

prov = {name: {k: R[name][k] for k in ("subj", "fold", "rep", "true", "pred", "p_high",
        "frontal_theta", "posterior_alpha", "gaze_entropy", "pupil", "conn", "ig", "n_low", "n_high")}
        for name in R}
(OUT_RES / "high_low_provenance.json").write_text(json.dumps(prov, indent=2))

log("\n" + tab.to_string(index=False))
log(f"\n[table] → {OUT_RES}/high_low_comparison.csv/.md/.tex")
log(f"[prov]  → {OUT_RES}/high_low_provenance.json")
