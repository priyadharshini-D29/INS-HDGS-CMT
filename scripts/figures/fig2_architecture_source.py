"""
INS-HDGS-CMT architecture figure (paper Fig. 2) — publication-quality vector
diagram. Writes fig2_Architecture.pdf (vector, used by the manuscript) and
assets/architecture.png (raster preview, used by the README).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType (editable, journal-safe)
plt.rcParams["ps.fonttype"] = 42

# ---------------- palette ----------------
PAL = {
    "eeg":   dict(cfill="#EAF0F9", cedge="#2F5597", head="#1F3A66",
                  kfill="#FFFFFF", kedge="#A9C0E0", tcol="#1F3A66"),
    "et":    dict(cfill="#E9F3EC", cedge="#2E7D46", head="#1E5631",
                  kfill="#FFFFFF", kedge="#A9D3B4", tcol="#1E5631"),
    "fus":   dict(cfill="#F0EAF7", cedge="#6A4C93", head="#4A2E6B",
                  kfill="#FFFFFF", kedge="#C9B8E0", tcol="#4A2E6B"),
    "dec":   dict(cfill="#FBEEDD", cedge="#C77F2E", head="#8A5314",
                  kfill="#FFFFFF", kedge="#E9C79A", tcol="#8A5314"),
    "pred":  dict(cfill="#F8E1E3", cedge="#B23A48", head="#7A1E28",
                  kfill="#F8E1E3", kedge="#B23A48", tcol="#7A1E28"),
}
SUB = "#555555"
ARR = "#444444"

# ---------------- helpers ----------------
def container(ax, x0, y0, x1, y1, fc, ec):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=1))

def head(ax, cx, y, text, color, size=12.5):
    ax.text(cx, y, text, ha="center", va="center",
            fontsize=size, fontweight="bold", color=color, zorder=5)

def card(ax, x0, y0, x1, y1, title, subtitle, p,
         tsize=9.0, ssize=7.2):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=1.1, edgecolor=p["kedge"], facecolor=p["kfill"], zorder=3))
    cx = (x0 + x1) / 2
    if subtitle:
        ax.text(cx, (y0 + y1) / 2 + 0.16, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=p["tcol"], zorder=5)
        ax.text(cx, (y0 + y1) / 2 - 0.20, subtitle, ha="center", va="center",
                fontsize=ssize, style="italic", color=SUB, zorder=5)
    else:
        ax.text(cx, (y0 + y1) / 2, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=p["tcol"], zorder=5)

def arrow(ax, xy0, xy1, color=ARR, ls="-", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        xy0, xy1, arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, color=color, linestyle=ls, zorder=4,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1, shrinkB=1))

# ---------------- canvas ----------------
fig, ax = plt.subplots(figsize=(12.4, 10.4))
ax.set_xlim(0, 16)
ax.set_ylim(0, 13.4)
ax.set_aspect("equal")
ax.axis("off")

ax.text(8, 13.02, "INS-HDGS-CMT Architecture", ha="center", va="center",
        fontsize=17, fontweight="bold", color="#222222")

# ---- EEG pathway container ----
p = PAL["eeg"]
container(ax, 0.5, 6.9, 7.5, 12.45, p["cfill"], p["cedge"])
head(ax, 4.0, 12.05, "EEG Graph–Spiking Pathway", p["head"])
card(ax, 0.8, 11.05, 7.2, 11.80, "EEG Band-Power Features",
     "Activity across canonical EEG bands", p)
card(ax, 0.8, 9.95, 7.2, 10.70, "Functional-Connectivity Graph",
     "Window-wise electrode interactions form dynamic brain graphs", p, ssize=6.8)
card(ax, 0.8, 8.55, 3.90, 9.55, "DynamicGAT",
     "Spatio-temporal attention", p, ssize=6.8)
card(ax, 4.10, 8.55, 7.20, 9.55, "LIF Spiking Encoder",
     "Theta–alpha spiking dynamics", p, ssize=6.8)
card(ax, 0.8, 7.35, 7.2, 8.10, "EEG Representation",
     "graph + spiking embeddings", p)

# ---- ET pathway container ----
q = PAL["et"]
container(ax, 8.5, 6.9, 15.5, 12.45, q["cfill"], q["cedge"])
head(ax, 12.0, 12.05, "Eye-Tracking Attention Pathway", q["head"])
card(ax, 8.8, 11.05, 15.2, 11.80, "Gaze and Pupil Sequence",
     "Visual behaviour during viewing", q)
card(ax, 8.8, 9.95, 15.2, 10.70, "Transformer Attention Encoder",
     "Attention over gaze / pupil dynamics", q)
card(ax, 8.8, 8.70, 15.2, 9.45, "ROI Attention Distribution",
     "Region importance derived from gaze (ET pathway only)", q, ssize=6.8)
card(ax, 8.8, 7.35, 15.2, 8.10, "Eye-Tracking Representation",
     "embedding of visual-attention behaviour", q)

# vertical flow arrows (EEG)
arrow(ax, (4.0, 11.05), (4.0, 10.70))
arrow(ax, (4.0, 9.95), (4.0, 9.55))
arrow(ax, (4.0, 8.55), (4.0, 8.10))
# vertical flow arrows (ET)
arrow(ax, (12.0, 11.05), (12.0, 10.70))
arrow(ax, (12.0, 9.95), (12.0, 9.45))
arrow(ax, (12.0, 8.70), (12.0, 8.10))

# dashed ROI-modulation link (ET ROI -> EEG graph, fused model only)
arrow(ax, (8.8, 9.10), (7.2, 10.25), color=PAL["fus"]["cedge"], ls=(0, (5, 3)), lw=1.4, rad=0.18)
ax.text(7.98, 9.35, "ROI modulates graph\n(fused model only)", ha="center", va="center",
        fontsize=6.6, style="italic", color=PAL["fus"]["cedge"], zorder=6)

# ---- NeuroFusion Transformer ----
f = PAL["fus"]
container(ax, 0.5, 4.35, 15.5, 6.25, f["cfill"], f["cedge"])
head(ax, 8.0, 5.95, "NeuroFusion Transformer", f["head"])
fx = [0.70, 4.425, 8.15, 11.875]
fw = 3.425
flabels = [("Modality Tokens", "EEG, graph & ET embeddings"),
           ("Self-Attention", "cross-modality interaction"),
           ("Directed Cross-Attention", "EEG attends to graph / ET"),
           ("Gated Residual Fusion", "gated, collapse-safe fusion")]
for x0, (t, s) in zip(fx, flabels):
    card(ax, x0, 4.55, x0 + fw, 5.45, t, s, f, tsize=8.6, ssize=6.6)

# ---- Neuro-Symbolic Decision Layer ----
d = PAL["dec"]
container(ax, 2.5, 2.05, 13.5, 3.95, d["cfill"], d["cedge"])
head(ax, 8.0, 3.65, "Neuro-Symbolic Decision Layer", d["head"])
dx = [2.70, 6.233, 9.766]
dw = 3.033
dlabels = [("Fused Representation", "multimodal engagement embedding"),
           ("Rule Activations", "eight interpretable rules"),
           ("Constraint Aggregation", "rule activations → logits")]
for x0, (t, s) in zip(dx, dlabels):
    card(ax, x0, 2.25, x0 + dw, 3.15, t, s, d, tsize=8.6, ssize=6.6)

# ---- Prediction ----
pr = PAL["pred"]
card(ax, 5.5, 0.80, 10.5, 1.70, "Consumer Engagement Prediction",
     "LOW  vs  HIGH  engagement", pr, tsize=9.6, ssize=7.4)

# stage-to-stage arrows
arrow(ax, (4.0, 7.35), (4.0, 6.25), lw=1.7)     # EEG rep -> fusion
arrow(ax, (12.0, 7.35), (12.0, 6.25), lw=1.7)   # ET rep -> fusion
arrow(ax, (8.0, 4.35), (8.0, 3.95), lw=1.7)     # fusion -> decision
arrow(ax, (8.0, 2.05), (8.0, 1.70), lw=1.7)     # decision -> prediction

# footnote
ax.text(8.0, 0.28,
        "Note: the EEG-only branch (Sec. 2.5.2 / Table 3, leakage-free ranking) disables ROI modulation — "
        "the EEG pathway then receives no eye-tracking-derived input.",
        ha="center", va="center", fontsize=6.8, style="italic", color=SUB)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

# Repo-relative outputs: the manuscript reads the PDF, the README the PNG.
ROOT = Path(__file__).resolve().parents[2]
pdf_out = ROOT / "paper" / "figures" / "fig2_Architecture.pdf"
png_out = ROOT / "assets" / "architecture.png"
pdf_out.parent.mkdir(parents=True, exist_ok=True)
png_out.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(pdf_out)
fig.savefig(png_out, dpi=200)
print("saved", pdf_out)
print("saved", png_out)
