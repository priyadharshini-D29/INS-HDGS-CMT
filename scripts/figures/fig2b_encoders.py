"""
Figure 2b — Modality encoder internals (companion to fig2_architecture).
Three panels: EEG spiking (LIF) encoder, dynamic graph (GAT) encoder, ET attention encoder.
Dims/components verified against models/spiking_encoder.py, models/gat_encoder.py,
models/et_encoder.py (et_encoder returns emb, roi_attn, window_seq).
Renders figures/fig2b_encoders.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# palette shared with fig2_architecture.py — strong accents used for borders + headers
C_EEG  = "#2F5FC0"   # EEG temporal blue (darkened for header/border)
C_GR   = "#2C4F9E"   # graph deeper blue
C_ET   = "#B97D17"   # ET amber (darkened)
C_OUT  = "#1F8A63"   # embedding-out green
C_EDGE = "#2F3645"

# light fills so a single dark text colour reads uniformly on every block
F_IN   = "#E7ECF5"
F_EEG  = "#D9E4FB"
F_GR   = "#CFDCF2"
F_ET   = "#FBE7C6"
F_OUT  = "#CFEBDC"

DK  = "#1B2540"   # uniform dark text for every block (main)
SUB = "#39435A"   # uniform dark text for sub-lines

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, ax = plt.subplots(figsize=(13.5, 8.2))
ax.set_xlim(0, 135); ax.set_ylim(0, 80); ax.axis("off")


def box(x, y, w, h, fc, ec=C_EDGE, lw=1.6, r=1.2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.1,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, alpha=alpha))


def t(x, y, s, fs=8.5, w="normal", c=DK, ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=fs, fontweight=w, color=c, ha=ha, va=va, style=style)


def arr(x1, y1, x2, y2, c=C_EDGE, lw=1.7, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=13, color=c, lw=lw, ls=ls, shrinkA=1, shrinkB=1))


def curve(x1, y1, x2, y2, rad, c, lw=1.7, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=13, color=c, lw=lw, ls=ls, shrinkA=2, shrinkB=2,
                 connectionstyle=f"arc3,rad={rad}"))


def vstack(cx, top, w, rows, ec):
    """rows: list of (main, sub, fc). Draw a top→down stack with arrows. Return bottom y."""
    h, gap = 6.1, 2.6
    y = top
    centers = []
    for main, sub, fc in rows:
        box(cx - w / 2, y - h, w, h, fc, ec=ec)
        if sub:
            t(cx, y - h * 0.36, main, 8.0, "bold")
            t(cx, y - h * 0.70, sub, 6.7, c=SUB)
        else:
            t(cx, y - h * 0.5, main, 8.0, "bold")
        centers.append(y - h)
        y -= h + gap
    for i in range(len(centers) - 1):
        arr(cx, centers[i], cx, centers[i] + gap + 0.05, ec)
    return y + gap


t(67, 77.8, "Modality Encoders", 16, "bold")
t(67, 74.4, "EEG spiking (LIF) · dynamic graph (GAT) · eye-tracking attention   ·   embedding dim D = 128",
  9.5, c="#444")

# ── PANEL 1 — EEG SPIKING ENCODER (LIF) ───────────────────────────────────────
cx = 22
t(cx, 70.5, "EEG spiking encoder (LIF)", 10, "bold", c=C_EEG)
vstack(cx, 67.5, 30, [
    ("Band-power windows (B,10,24,5)", "5 bands × 24 ch × 10 windows", F_IN),
    ("Learnable band weighting", "softmax over δ,θ,α,β,γ → (B,24,10)", F_EEG),
    ("Synaptic projection → 128", "Linear input current per step", F_EEG),
    ("LIF layer 1 (decay 0.9, V_th 1.0)", "integrate-and-fire · surrogate grad", F_EEG),
    ("LIF layer 2 → 128", "spike train · LayerNorm · dropout", F_EEG),
    ("Temporal pooling (T = 10 steps)", "mean over membrane states", F_EEG),
    ("EEG spiking embedding (B, 128)", "≈10.6% firing · event-driven", F_OUT),
], C_EEG)

# ── PANEL 2 — DYNAMIC GRAPH (GAT) ENCODER ─────────────────────────────────────
cx = 67
t(cx, 70.5, "Dynamic graph (GAT) encoder", 10, "bold", c=C_GR)
g_in_y = 67.5          # top of graph stack (input box top edge)
vstack(cx, g_in_y, 30, [
    ("Node feats (B, N, 5) + adj (B,N,N)", "Pearson functional connectivity", F_IN),
    ("GAT layer 1 · 4 heads × 32", "concat → 128 · LayerNorm · ELU", F_GR),
    ("GAT layer 2 · 1 head → 128", "+ residual projection", F_GR),
    ("LayerNorm", "", F_GR),
    ("Node embeddings  (B, N, 128)", "+ attention maps α (l1,l2) — explainability", F_OUT),
], C_GR)
t(cx, 23.5, "eᵢⱼ = LeakyReLU(aᵀ[Whᵢ ‖ Whⱼ])   αᵢⱼ = softmaxⱼ(eᵢⱼ)",
  8.0, c="#555", style="italic")
t(cx, 20.1, "(Veličković et al., 2018)", 7.2, c="#888")

# ── PANEL 3 — ET ATTENTION ENCODER ────────────────────────────────────────────
cx = 112
t(cx, 70.5, "ET attention encoder", 10, "bold", c=C_ET)
bot = vstack(cx, 67.5, 30, [
    ("ET sequence  (B, 600, 3)", "gaze_x, gaze_y, pupil", F_IN),
    ("Per-timestep MLP → 64", "GELU · LayerNorm", F_ET),
    ("+ CLS token · sinusoidal pos-enc", "", F_ET),
    ("Transformer × 2 · 4 heads", "pre-norm · FF 256 · self-attn", F_ET),
    ("CLS → projection → 128", "", F_ET),
], C_ET)

# ── ET encoder produces THREE parallel outputs (emb, ROI attn, window seq) ─────
ow, oh = 30, 5.4
y_emb = bot - 3.5            # box top edges
y_roi = y_emb - 6.7
y_win = y_roi - 6.7

# branch from the last encoder box down into the parallel-output group
arr(cx, bot, cx, y_emb + 0.05, C_ET)
t(cx + ow/2 + 0.5, y_emb - oh/2, "3 parallel outputs", 6.6, "bold", c=C_ET,
  ha="left", style="italic")

box(cx - ow/2, y_emb - oh, ow, oh, F_OUT, ec=C_OUT)
t(cx, y_emb - oh*0.36, "ET embedding (B, 128)", 8.0, "bold")
t(cx, y_emb - oh*0.72, "→ NeuroFusion cross-modal Transformer", 6.7, c=SUB)

box(cx - ow/2, y_roi - oh, ow, oh, F_ET, ec=C_ET, lw=2.0)
t(cx, y_roi - oh*0.36, "ROI attention (B, 10)", 8.0, "bold", c=C_ET)
t(cx, y_roi - oh*0.72, "softmax over 10 ROIs → graph modulation", 6.7, c=SUB)

box(cx - ow/2, y_win - oh, ow, oh, F_ET, ec=C_ET)
t(cx, y_win - oh*0.36, "Window sequence (B, 10, 128)", 8.0, "bold")
t(cx, y_win - oh*0.72, "per-window temporal features", 6.7, c=SUB)

# left-side brace grouping the three parallel outputs
br_x = cx - ow/2 - 1.4
ax.plot([br_x, br_x], [y_win - oh, y_emb], color=C_ET, lw=1.4)

# ── ROI attention modulates the electrode graph (cross-panel flow) ────────────
# route up the clear gap between the graph (x≤82) and ET (x≥97) panels
roi_left = (cx - ow/2, y_roi - oh/2)                 # left edge of ROI box
g_in_right = (67 + 30/2, g_in_y - 3.0)               # right edge of graph input box
curve(roi_left[0], roi_left[1], g_in_right[0], g_in_right[1],
      rad=-0.32, c=C_ET, lw=1.8, ls=(0, (5, 3)))
t(90, 47, "ROI modulation", 7.6, "bold", c=C_ET, style="italic")
t(90, 44.4, "of electrode graph", 7.6, "bold", c=C_ET, style="italic")

t(67, 2.6, "Each encoder outputs a 128-d embedding combined by the NeuroFusion cross-modal Transformer (Fig. 2). "
           "The eye-tracking encoder additionally emits an ROI attention vector that modulates the electrode graph "
           "(dashed) and a per-window sequence.", 7.6, c="#555")

fig.tight_layout(pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig2b_encoders.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig2b_encoders.pdf and .png")
