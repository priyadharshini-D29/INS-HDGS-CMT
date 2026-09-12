"""
Figure 4 — Cross-modal fusion and neuro-symbolic reasoning mechanism (fig:fusion).
Schematic companion to fig2_architecture / fig2b_encoders, in the same clean style:
light fills, uniform dark text, accent-coloured borders, HD 300 dpi.
Flow: EEG embedding + ET embedding -> cross-modal Transformer (bidirectional
cross-attention) -> joint representation -> neuro-symbolic reasoning -> engagement
prediction.  Verified against models/ins_hdgs_cmt.py (cross-modal transformer +
differentiable rule layer).  Renders figures/fig_fusion.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# accent colours (borders + headers) — shared with the other schematics
C_EEG = "#2F5FC0"; C_ET = "#B97D17"; C_FUS = "#6C4FB0"; C_SYM = "#1F8A63"
C_OUT = "#C0392B"; C_EDGE = "#2F3645"
# light fills so one dark text colour reads on every block
F_EEG = "#D9E4FB"; F_ET = "#FBE7C6"; F_FUS = "#E4DBF4"; F_SYM = "#CFEBDC"
F_OUT = "#F6D9D5"; F_IN = "#E7ECF5"
DK = "#1B2540"; SUB = "#39435A"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(12.5, 8.4))
ax.set_xlim(0, 120); ax.set_ylim(0, 84); ax.axis("off")


def box(x, y, w, h, fc, ec=C_EDGE, lw=1.7, r=1.3, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.1,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, alpha=alpha))


def t(x, y, s, fs=9, w="normal", c=DK, ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=fs, fontweight=w, color=c, ha=ha, va=va, style=style)


def arr(x1, y1, x2, y2, c=C_EDGE, lw=1.9, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=15, color=c, lw=lw, ls=ls, shrinkA=2, shrinkB=2))


t(60, 81.5, "Cross-Modal Fusion & Neuro-Symbolic Reasoning", 15, "bold")
t(60, 78.0, "bidirectional cross-attention  ·  learned multimodal fusion  ·  interpretable rule-based reasoning",
  9.5, c="#555")

# ── encoder embeddings (inputs) ───────────────────────────────────────────────
box(10, 67, 40, 7.5, F_EEG, ec=C_EEG)
t(30, 71.6, "EEG embedding", 9.5, "bold")
t(30, 68.7, "dynamic functional-graph (GAT) + spiking encoder", 7.4, c=SUB)

box(70, 67, 40, 7.5, F_ET, ec=C_ET)
t(90, 71.6, "ET embedding", 9.5, "bold")
t(90, 68.7, "eye-tracking (gaze + pupil) attention encoder", 7.4, c=SUB)

# ── cross-modal transformer block ─────────────────────────────────────────────
box(8, 43, 104, 20, "#F4F1FB", ec=C_FUS, lw=2.0, r=1.6)
t(11, 61.2, "Cross-Modal Transformer  (× L layers)", 11, "bold", c=C_FUS, ha="left")

# two directional cross-attention sub-blocks
box(13, 48, 44, 9.0, F_FUS, ec=C_FUS)
t(35, 54.4, "EEG → ET cross-attention", 9, "bold")
t(35, 51.4, "neural features attend to gaze evidence", 7.2, c=SUB)
t(35, 49.2, "(EEG asks where the eyes attend)", 6.9, c=SUB)

box(63, 48, 44, 9.0, F_FUS, ec=C_FUS)
t(85, 54.4, "ET → EEG cross-attention", 9, "bold")
t(85, 51.4, "gaze features attend to neural activity", 7.2, c=SUB)
t(85, 49.2, "(gaze grounded in cortical engagement)", 6.9, c=SUB)

# inputs into the two cross-attention blocks (crossed Q vs K,V)
arr(30, 67, 32, 57.2, C_EEG)                 # EEG -> EEG→ET (as Q)
arr(34, 67, 80, 57.2, C_EEG, ls=(0, (4, 2)))  # EEG -> ET→EEG (as K,V)
arr(90, 67, 88, 57.2, C_ET)                  # ET  -> ET→EEG (as Q)
arr(86, 67, 40, 57.2, C_ET, ls=(0, (4, 2)))   # ET  -> EEG→ET (as K,V)
t(60, 45.7, "queries (solid) ·  keys / values (dashed) exchanged across modalities",
  7.6, "normal", c="#666", style="italic")

# ── joint representation ──────────────────────────────────────────────────────
arr(35, 48, 50, 39.6, C_FUS)
arr(85, 48, 70, 39.6, C_FUS)
box(34, 32, 52, 7.5, F_FUS, ec=C_FUS, lw=2.0)
t(60, 36.6, "Joint multimodal representation", 9.5, "bold")
t(60, 33.7, "fused EEG + eye-tracking summary", 7.6, c=SUB)

# ── neuro-symbolic reasoning ──────────────────────────────────────────────────
arr(60, 32, 60, 26.1, C_SYM)
box(14, 13, 92, 13, "#EAF6F0", ec=C_SYM, lw=2.0, r=1.6)
t(60, 23.6, "Neuro-Symbolic Reasoning Module", 11, "bold", c=C_SYM)

box(18, 14.5, 26, 6.6, F_SYM, ec=C_SYM)
t(31, 18.7, "Fuzzy membership", 8.2, "bold")
t(31, 16.2, "soft, graded concept activations", 6.4, c=SUB)

box(47, 14.5, 26, 6.6, F_SYM, ec=C_SYM)
t(60, 18.7, "Differentiable rules", 8.2, "bold")
t(60, 16.2, "learnable human-readable rules", 6.4, c=SUB)

box(76, 14.5, 26, 6.6, F_SYM, ec=C_SYM)
t(89, 18.7, "Constraint aggregation", 8.2, "bold")
t(89, 16.2, "combines fired rules into a decision", 6.2, c=SUB)
arr(44, 17.8, 47, 17.8, C_SYM)
arr(73, 17.8, 76, 17.8, C_SYM)

# ── prediction ────────────────────────────────────────────────────────────────
arr(60, 13, 60, 7.6, C_OUT)
box(36, 0.5, 48, 7.0, F_OUT, ec=C_OUT, lw=2.0)
t(60, 4.8, "Consumer engagement prediction", 9.5, "bold")
t(60, 1.9, "HIGH vs LOW engagement probability", 7.4, c=SUB)

fig.tight_layout(pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_fusion.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig_fusion.pdf and .png")
