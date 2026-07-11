"""
Figure 2 — INS-HDGS-CMT architecture (publication, layered left→right dataflow).
Dims/components verified against models/ins_hdgs_cmt.py and config/settings.py.
Renders figures/fig2_architecture.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

C_IN   = "#E7ECF5"   # inputs grey-blue
C_EEG  = "#5B8DEF"   # EEG/graph blue
C_SNN  = "#7FB2F0"   # snn light blue
C_ET   = "#E8A33D"   # ET amber
C_ROI  = "#46B39D"   # ROI teal
C_FUS  = "#2EA37A"   # fusion green
C_HEAD = "#7C5CBF"   # head purple
C_LOSS = "#F4F6FA"
C_EDGE = "#2F3645"
C_DANN = "#B2592E"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, ax = plt.subplots(figsize=(13.5, 7.6))
ax.set_xlim(0, 135); ax.set_ylim(0, 76); ax.axis("off")

def box(x, y, w, h, fc, ec=C_EDGE, lw=1.3, r=1.4, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.1,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, alpha=alpha))
def t(x, y, s, fs=8.5, w="normal", c=C_EDGE, ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=fs, fontweight=w, color=c, ha=ha, va=va, style=style)
def arr(x1, y1, x2, y2, c=C_EDGE, lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=13, color=c, lw=lw, ls=ls, shrinkA=1, shrinkB=1))

t(67, 73.6, "INS-HDGS-CMT", 17, "bold")
t(67, 70.2, "Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking Cognitive Multimodal Transformer "
            "· embedding dim D=128", 9.5, c="#555")

# ── COLUMN 1 — INPUTS ─────────────────────────────────────────────────────────
t(13, 64.5, "Inputs", 10, "bold")
box(2, 55, 22, 7.5, C_IN); t(13, 58.8, "EEG band-power\n(W=10, C, 5 bands)", 8.4, "bold")
box(2, 45, 22, 7.5, C_IN); t(13, 48.8, "Functional adjacency\n(W, C, C) · Pearson", 8.4, "bold")
box(2, 30, 22, 7.5, C_IN); t(13, 33.8, "Eye tracking\n(600, 3) gaze_x,y,pupil", 8.4, "bold")
box(2, 20, 22, 7.5, C_IN); t(13, 23.8, "ROI dwell vector\n(10,)  5×2 grid", 8.4, "bold")

# ── COLUMN 2 — ENCODERS ───────────────────────────────────────────────────────
t(50, 64.5, "Modality encoders", 10, "bold")
# ROI graph modulation (between adj and GAT)
box(30, 44.0, 17, 5.2, C_ROI, alpha=.9); t(38.5, 46.6, "ROI graph\nmodulation", 7.8, "bold", c="white")
# Dynamic GAT
box(30, 54.0, 17, 8.5, C_EEG); t(38.5, 59.6, "Dynamic GAT", 9, "bold", c="white")
t(38.5, 56.4, "2-layer (4×32→128)\n+ temporal Transformer", 7.2, c="white")
# SNN
box(30, 30.5, 17, 8.0, C_SNN); t(38.5, 35.8, "Spiking EEG enc.", 8.8, "bold", c=C_EDGE)
t(38.5, 32.9, "LIF · 10 steps · θ/α\nband-weighted proxy", 7.0)
# ET attention encoder
box(30, 19.5, 17, 7.5, C_ET); t(38.5, 24.6, "ET attention enc.", 8.8, "bold", c="#3a2e10")
t(38.5, 21.8, "MLP+CLS Transformer\n→ et_emb, et_roi_attn", 7.0, c="#3a2e10")

# eeg_merge
box(54, 42.5, 13, 6.5, "#cdddf7"); t(60.5, 45.75, "EEG merge\nLin256→128", 7.8, "bold")
# ROI attention gate
box(54, 51.5, 13, 6.0, C_ROI, alpha=.9); t(60.5, 54.5, "ROI attention\ngate", 7.6, "bold", c="white")

# input→encoder arrows
arr(24, 58.7, 30, 58.5, C_EEG)                  # eeg→GAT
arr(24, 48.7, 30, 47.0, C_ROI)                  # adj→ROImod
arr(38.5, 49.2, 38.5, 54.0, C_ROI)             # ROImod→GAT (modulated adj)
arr(24, 47.5, 30, 33.8, C_SNN, ls=(0,(3,2)))   # eeg→SNN (proxy)
arr(24, 33.7, 30, 23.0, C_ET)                   # ET→ETenc
arr(24, 22.0, 30, 45.0, C_ROI, ls=(0,(2,2)))   # roi→ROImod
# encoder→merge
arr(47, 58.0, 54, 47.5, C_EEG)                  # graph_emb→merge
arr(47, 34.0, 54, 44.0, C_SNN)                  # snn_emb→merge
arr(60.5, 49.0, 60.5, 51.5, C_EEG)             # merge→ROIattn
arr(24, 21.5, 54, 52.5, C_ROI, ls=(0,(2,2)))   # roi→ROIattn

# ── COLUMN 3 — FUSION ─────────────────────────────────────────────────────────
t(86, 64.5, "Cross-modal fusion", 10, "bold")
box(75, 30, 23, 28, C_FUS, alpha=.12, ec=C_FUS, lw=1.8)
t(86.5, 55.5, "NeuroFusion Transformer", 9.2, "bold", c=C_FUS)
for i, s in enumerate([
    "1 · modality-type tokens [eeg|graph|et]",
    "2 · self-attention over 3 tokens",
    "3 · directed cross-attn  EEG←Graph, EEG←ET",
    "4 · gated residual fusion → fused (128)"]):
    box(77, 49.5 - i*5.0, 19, 4.0, "white", ec=C_FUS, lw=1.0)
    t(86.5, 51.5 - i*5.0, s, 6.9)
# encoder→fusion arrows
arr(67, 54.5, 75, 50.0, C_EEG)                  # eeg_emb (gated)→fusion
arr(47, 57.5, 75, 47.0, C_EEG, ls=(0,(4,2)))   # graph window seq→fusion
arr(47, 21.5, 75, 36.0, C_ET)                   # et_emb / et seq→fusion

# ── COLUMN 4 — HEAD / OUTPUT ──────────────────────────────────────────────────
t(116, 64.5, "Interpretable head", 10, "bold")
box(106, 44, 24, 11, C_HEAD); t(118, 51.5, "Neuro-Symbolic\nRule Layer", 9.4, "bold", c="white")
t(118, 47.0, "8 differentiable IF–THEN rules\n+ learned bypass mix", 7.2, c="white")
arr(98, 44.0, 106, 49.0, C_FUS)                 # fused→head
box(108, 33, 20, 6.5, "#efe9fb"); t(118, 36.25, "logits → softmax\nHIGH / LOW (2)", 8.4, "bold", c=C_HEAD)
arr(118, 44.0, 118, 39.5, C_HEAD)

# DANN (training only)
box(106, 22, 24, 6.5, "white", ec=C_DANN, lw=1.3, r=1.2)
t(118, 25.2, "DANN subject head (GRL)\ntraining only · subject-invariance", 6.9, "bold", c=C_DANN)
arr(98, 33.0, 112, 28.5, C_DANN, ls=(0,(3,2)))

# ── LOSS STRIP (bottom) ───────────────────────────────────────────────────────
box(2, 4, 128, 9.5, C_LOSS, ec=C_EDGE, lw=1.2)
t(6, 11.2, "Multi-task loss", 9.2, "bold", ha="left")
losses = [
    ("L_focal  (γ=3, eff-num)", "λ=5.0", C_HEAD),
    ("L_contrast  InfoNCE", "λ=0.05", C_ET),
    ("L_ROI  KL", "λ=0.05", C_ROI),
    ("L_conn  sparsity", "λ=0.02", C_EEG),
    ("L_MMD  RBF", "λ=0.10", C_FUS),
    ("L_rules  div+sparse", "λ=0.02", C_HEAD),
]
x = 8
for name, lam, c in losses:
    box(x, 5.2, 19.5, 5.0, "white", ec=c, lw=1.4)
    t(x+9.75, 7.9, name, 6.9, "bold", c=C_EDGE); t(x+9.75, 6.1, lam, 6.6, c=c)
    x += 20.5
t(67, 1.6, "Per-fold ensemble (×5–15) · AdamW 5e-4 · cosine warm-restarts · 250 ep · early-stop on val balanced-acc · "
           "AMP · temperature calibration on held-out val subject", 7.2, c="#555")

fig.tight_layout(pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig2_architecture.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig2_architecture.pdf and .png")
