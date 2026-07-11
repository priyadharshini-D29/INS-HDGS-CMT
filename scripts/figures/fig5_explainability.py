"""
Figure 5 — Explainability of INS-HDGS-CMT (runs on CPU; no GPU contention).
Loads a production fold checkpoint, runs a forward pass on one subject's epochs,
and composites:
  (A) electrode attention topomap   (mean GAT L2 attention received per node)
  (B) GAT electrode attention matrix (C×C)
  (C) neuro-symbolic rule activations (mean over samples, 8 rules)
  (D) decoded human-readable IF–THEN rules (rule_layer.explain_rules)

Usage:
  CUDA_VISIBLE_DEVICES="" python figures/fig5_explainability.py \
      --ckpt output/checkpoints/repro_focal_g3p0_effective_num_37/repro_focal_g3p0_effective_num_37_fold01_e0.pt \
      --subject S01
Renders figures/fig5_explainability.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch, argparse, sys
from pathlib import Path

PH8 = Path(__file__).resolve().parents[1]
for p in (str(PH8), str(PH8.parent)):
    if p not in sys.path: sys.path.insert(0, p)

ap = argparse.ArgumentParser()
# Default to the 19-channel (clean19) checkpoint so the figure matches the
# manuscript's 19-channel pipeline. The legacy 24-channel checkpoint
# (repro_focal_g3p0_effective_num_37) produces inconsistent electrode names
# and rules and must NOT be used for the 19-channel manuscript figures.
ap.add_argument("--ckpt", default=str(PH8 / "output/checkpoints/eeg_clean19_full/"
                                    "eeg_clean19_full_fold01_e0.pt"))
ap.add_argument("--subject", default="S01")
args = ap.parse_args()

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # force CPU
from config.settings import EMBED_DIM
from models.ins_hdgs_cmt import INS_HDGS_CMT, AblationConfig
from data.dataset import NeumaGraphDataset

DEV = torch.device("cpu")
CHAN = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
        "F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"]  # 19 canonical (matches data order)

print(f"[fig5] loading subject {args.subject} …", flush=True)
ds = NeumaGraphDataset(subject_ids=[args.subject], augment=False)
n_ch, n_et = ds.n_eeg_ch, ds.n_et_ch
batch = [ds[i] for i in range(min(len(ds), 12))]
collate = lambda key: torch.stack([b[key] for b in batch]).to(DEV)

print(f"[fig5] building model (C={n_ch}, ET={n_et}) and loading {Path(args.ckpt).name} …", flush=True)
model = INS_HDGS_CMT(n_eeg_ch=n_ch, n_et_ch=n_et, n_classes=ds.n_classes,
                     embed_dim=EMBED_DIM, ablation=AblationConfig.full(),
                     feature_names=CHAN[:n_ch]).to(DEV).eval()
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
sd = {k.replace("module.", ""): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[fig5] loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

with torch.no_grad():
    out = model(eeg_windows=collate("eeg_windows"), adj_matrices=collate("adj_matrices"),
                et_seq=collate("et_seq"), roi_vector=collate("roi_vector"),
                weighted_adjs=collate("weighted_adjs"))

# GAT attention: l2_attn (B*W, C, C) → mean over batch·windows → attention received per node
gat = out["gat_attn"]; attn = gat.get("l2_attn", gat.get("l1_attn"))
attn = attn.detach().cpu().numpy()
attn_mat = attn.mean(axis=0)                      # (C, C)
node_imp = attn_mat.mean(axis=0)                  # attention received per electrode
rule_act = out["rule_act"].detach().cpu().numpy().mean(axis=0)   # (n_rules,)
rules_text = model.rule_layer.explain_rules(class_names=["LOW", "HIGH"])

# ── compose figure (2x2 EQUAL panels: A,B top; C,D bottom) ─────────────────────
# Panel D shows the decoded rules in compact form at a large font; the full,
# precise rule text is also typeset in the manuscript (Table tab:rules).
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig = plt.figure(figsize=(14.0, 11.0))
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)

# (A) per-electrode attention
axA = fig.add_subplot(gs[0, 0])
try:
    from explainability.topomap import plot_topomap
    plot_topomap(node_imp, CHAN[:n_ch], "(A) Electrode attention (GAT)", axA)
except Exception:
    sc = axA.scatter(range(n_ch), node_imp, c=node_imp, cmap="inferno", s=90)
    axA.set_xticks(range(n_ch)); axA.set_xticklabels(CHAN[:n_ch], rotation=90, fontsize=9)
    axA.set_ylabel("Mean attention received")
    axA.set_title("(A) Electrode attention (GAT)", fontweight="bold")
    fig.colorbar(sc, ax=axA, fraction=.046)

# (B) electrode-by-electrode attention matrix
axB = fig.add_subplot(gs[0, 1])
im = axB.imshow(attn_mat, cmap="viridis", aspect="auto")
axB.set_title("(B) GAT electrode attention matrix", fontweight="bold")
axB.set_xticks(range(n_ch)); axB.set_xticklabels(CHAN[:n_ch], rotation=90, fontsize=9)
axB.set_yticks(range(n_ch)); axB.set_yticklabels(CHAN[:n_ch], fontsize=9)
fig.colorbar(im, ax=axB, fraction=.046)

# (C) neuro-symbolic rule activations
axC = fig.add_subplot(gs[1, 0])
axC.bar(range(1, len(rule_act)+1), rule_act, color="#7C5CBF", alpha=.85)
axC.set_xlabel("Rule"); axC.set_ylabel("Mean activation")
axC.set_title("(C) Neuro-symbolic rule activations", fontweight="bold")
axC.set_xticks(range(1, len(rule_act)+1)); axC.grid(axis="y", alpha=.25)

# (D) decoded interpretable rules — compact, large font; full text in the table
def _compact(txt):
    s = txt.replace("feature_", "z").replace("Rule ", "R").replace(": IF ", ": ")
    s = s.replace(" AND ", " ").replace("THEN", "→")
    s = s.replace(" ↑", "↑").replace(" ↓", "↓")
    return " ".join(s.split())
axD = fig.add_subplot(gs[1, 1]); axD.axis("off")
axD.set_title("(D) Decoded interpretable rules", fontweight="bold", loc="left")
lines = [_compact(rules_text[i]) for i in range(len(rules_text))]
axD.text(0.0, 0.95, "\n".join(lines), va="top", ha="left",
         fontsize=19, family="monospace", linespacing=2.0, transform=axD.transAxes)

fig.suptitle(f"INS-HDGS-CMT explainability  ({args.subject}, fold-01, 19-channel EEG branch)",
             fontsize=15, fontweight="bold", y=0.995)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig5_explainability.{ext}", dpi=300, bbox_inches="tight")
print("[fig5] wrote figures/fig5_explainability.pdf and .png (2x2 equal; panel D + table)", flush=True)
