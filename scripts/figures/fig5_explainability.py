"""
Manuscript Fig. 8 (file fig5_explainability.{pdf,png}) - explainability of INS-HDGS-CMT (CPU only).

  (A) electrode attention topomap    mean GAT L2 attention received per node     - production (gated) model
  (B) GAT electrode attention matrix (C x C)                                      - production (gated) model
  (C) neuro-symbolic rule activations, mean over the subject's epochs            - rule-only model (alpha == 0)
  (D) grounded IF-THEN rules from scripts/analysis/ground_rules_to_electrodes.py - rule-only model

Why two checkpoints: in the production model the bypass gate never leaves its initial value
(alpha = 0.570) and the rule attention stays uniform (results/explainability/rule_fidelity.md),
so its rule traces do not explain its decision.  Panels C and D are therefore drawn from the
model trained with the gate closed (AblationConfig.ns_rule_only), whose rules decide, and panel D
uses the integrated-gradient grounding of that model (electrode x band premises + gaze term).

Usage (any cwd; src/model is put on sys.path):
  CUDA_VISIBLE_DEVICES="" python scripts/figures/fig5_explainability.py \
      --ckpt      src/model/output/checkpoints/abl_full/abl_full_fold01_e0.pt \
      --rule-ckpt src/model/output/checkpoints/abl_ns_rule_only/abl_ns_rule_only_fold01_e0.pt \
      --grounding results/explainability/rule_grounding_abl_ns_rule_only.json --subject S01
Writes paper/figures/fig5_explainability.{pdf,png}.  Driver: scripts/revision/run_revision.sh figs
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]           # <repo>
MODEL_DIR = ROOT / "src" / "model"
for p in (str(MODEL_DIR), str(MODEL_DIR.parent), str(ROOT)):   # src/model, src (package "model"), repo
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # force CPU
import torch  # noqa: E402

OUT_DIR = MODEL_DIR / os.environ.get("NEUMA_OUTPUT_DIR", "output")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default=str(OUT_DIR / "checkpoints/abl_full/abl_full_fold01_e0.pt"),
                help="production (gated) fold checkpoint: panels A and B")
ap.add_argument("--rule-ckpt", default=str(OUT_DIR / "checkpoints/abl_ns_rule_only/abl_ns_rule_only_fold01_e0.pt"),
                help="rule-only fold checkpoint (trained with alpha == 0): panel C")
ap.add_argument("--grounding", default=str(ROOT / "results/explainability/rule_grounding_abl_ns_rule_only.json"),
                help="JSON written by ground_rules_to_electrodes.py for the rule-only model: panel D")
ap.add_argument("--subject", default="S01")
ap.add_argument("--n-epochs", type=int, default=12, help="epochs of the subject passed through the models")
ap.add_argument("--out", default=str(ROOT / "paper/figures/fig5_explainability"), help="output stem (.pdf/.png added)")
args = ap.parse_args()

from config.settings import EMBED_DIM                      # noqa: E402
from models.ins_hdgs_cmt import INS_HDGS_CMT, AblationConfig  # noqa: E402
from data.dataset import NeumaGraphDataset                 # noqa: E402

DEV = torch.device("cpu")
CHAN = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]   # 19 canonical (matches data order)
GREEK = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ"}
ARROW = {"up": "↑", "down": "↓"}

print(f"[fig8] loading subject {args.subject} ...", flush=True)
ds = NeumaGraphDataset(subject_ids=[args.subject], augment=False)
n_ch, n_et = ds.n_eeg_ch, ds.n_et_ch
batch = [ds[i] for i in range(min(len(ds), args.n_epochs))]
collate = lambda key: torch.stack([b[key] for b in batch]).to(DEV)  # noqa: E731


def load_model(path: str, ablation: AblationConfig, tag: str) -> INS_HDGS_CMT:
    model = INS_HDGS_CMT(n_eeg_ch=n_ch, n_et_ch=n_et, n_classes=ds.n_classes, embed_dim=EMBED_DIM,
                         ablation=ablation, feature_names=CHAN[:n_ch]).to(DEV).eval()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[fig8] {tag}: loaded {Path(path).name} (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    return model


def forward(model: INS_HDGS_CMT):
    with torch.no_grad():
        return model(eeg_windows=collate("eeg_windows"), adj_matrices=collate("adj_matrices"),
                     et_seq=collate("et_seq"), roi_vector=collate("roi_vector"),
                     weighted_adjs=collate("weighted_adjs"))


# -- panels A, B: graph attention of the production model ---------------------------------------
out_full = forward(load_model(args.ckpt, AblationConfig.full(), "gated model"))
gat = out_full["gat_attn"]
attn = gat.get("l2_attn", gat.get("l1_attn")).detach().cpu().numpy()
attn_mat = attn.mean(axis=0)                       # (C, C)
node_imp = attn_mat.mean(axis=0)                   # attention received per electrode

# -- panel C: rule activations of the rule-only model -------------------------------------------
rule_model = load_model(args.rule_ckpt, AblationConfig.ns_rule_only(), "rule-only model")
rule_act = forward(rule_model)["rule_act"].detach().cpu().numpy().mean(axis=0)   # (n_rules,)

# -- panel D: grounded premises (integrated gradients over the held-out epochs of every fold) ----
gpath = Path(args.grounding)
if gpath.exists():
    g = json.load(open(gpath, encoding="utf-8"))
    lines = []
    for r in g["rules"]:
        prem = " ∧ ".join(f"{p['electrode']}-{GREEK.get(p['band'], p['band'])}{ARROW[p['direction']]}"
                                for p in r["premises"])
        etp = f"{r['et_premise']['stream']}{ARROW[r['et_premise']['direction']]}"
        lines.append(f"R{r['rule']}: {prem} ∧ {etp} → {r['conclusion']}  ({r['dominant_frac']*100:.0f}%)")
    d_title = "(D) Grounded rules (rule-only model; IG over held-out epochs)"
    d_note = (f"{g['n_epochs']} epochs, {len(g['checkpoints'])} fold checkpoint(s); "
              "(%) = share of epochs in which the rule dominates")
else:
    print(f"[fig8] WARNING: grounding file {gpath} not found; panel D falls back to latent-key rule text",
          flush=True)
    raw = rule_model.rule_layer.explain_rules(class_names=["LOW", "HIGH"])
    lines = [" ".join(t.replace("feature_", "z").replace("Rule ", "R").replace(": IF ", ": ")
                      .replace(" AND ", " ").replace("THEN", "→").split()) for t in raw]
    d_title = "(D) Decoded rules (latent keys; grounding file missing)"
    d_note = "run scripts/analysis/ground_rules_to_electrodes.py on the rule-only checkpoints"

# -- compose figure (2 x 2 equal panels) --------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig = plt.figure(figsize=(14.0, 11.0))
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)

axA = fig.add_subplot(gs[0, 0])
try:
    from explainability.topomap import plot_topomap
    plot_topomap(node_imp, CHAN[:n_ch], "(A) Electrode attention (GAT, gated model)", axA)
except Exception:
    sc = axA.scatter(range(n_ch), node_imp, c=node_imp, cmap="inferno", s=90)
    axA.set_xticks(range(n_ch)); axA.set_xticklabels(CHAN[:n_ch], rotation=90, fontsize=9)
    axA.set_ylabel("Mean attention received")
    axA.set_title("(A) Electrode attention (GAT, gated model)", fontweight="bold")
    fig.colorbar(sc, ax=axA, fraction=.046)

axB = fig.add_subplot(gs[0, 1])
im = axB.imshow(attn_mat, cmap="viridis", aspect="auto")
axB.set_title("(B) GAT electrode attention matrix (gated model)", fontweight="bold")
axB.set_xticks(range(n_ch)); axB.set_xticklabels(CHAN[:n_ch], rotation=90, fontsize=9)
axB.set_yticks(range(n_ch)); axB.set_yticklabels(CHAN[:n_ch], fontsize=9)
fig.colorbar(im, ax=axB, fraction=.046)

axC = fig.add_subplot(gs[1, 0])
axC.bar(range(1, len(rule_act) + 1), rule_act, color="#7C5CBF", alpha=.85)
axC.set_xlabel("Rule"); axC.set_ylabel("Mean activation")
axC.set_title("(C) Rule activations (rule-only model, α ≡ 0)", fontweight="bold")
axC.set_xticks(range(1, len(rule_act) + 1)); axC.grid(axis="y", alpha=.25)

axD = fig.add_subplot(gs[1, 1]); axD.axis("off")
axD.set_title(d_title, fontweight="bold", loc="left")
axD.text(0.0, 0.95, "\n".join(lines), va="top", ha="left", fontsize=13, family="monospace",
         linespacing=1.9, transform=axD.transAxes)
axD.text(0.0, 0.02, d_note, va="bottom", ha="left", fontsize=9, color="0.35", transform=axD.transAxes)

fig.suptitle(f"INS-HDGS-CMT explainability ({args.subject}, fold-01; A,B gated model; C,D rule-only model)",
             fontsize=15, fontweight="bold", y=0.995)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
print(f"[fig8] wrote {args.out}.pdf and .png", flush=True)
