#!/usr/bin/env python3
"""
build_architecture_figure_S01.py
================================
Assemble the publication-quality INS-HDGS-CMT architecture figure from the REAL
case-study panels produced by ``figure_panels_S01.py`` (subject S01, fold-01,
epoch 6 / ImagePage_5, ground-truth HIGH_ENGAGEMENT, ensemble p(HIGH)=0.884).

This script does NO inference — it only composes already-generated, model-derived
evidence panels into the journal layout:

    Multimodal Input  ->  Preprocessing & Epoching
        |                         |
   [EEG Graph-Spiking]      [Eye-Tracking Attention]      (two top columns)
        \\          ROI-Guided Modulation          /
                  NeuroFusion Transformer                 (merge, middle)
                  Neuro-Symbolic Decision  ->  Prediction (bottom)

Output: figures/architecture/INS_HDGS_CMT_architecture_S01.{png,svg}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

PHASE8 = Path("/home/nvidia/24PHD1314/Neuma_Model/NEUMA_PHASE8")
PAN    = PHASE8 / "figures" / "case_study_S01"
OUT    = PHASE8 / "figures" / "architecture"
OUT.mkdir(parents=True, exist_ok=True)

FW, FH = 18.0, 26.0          # canvas inches (portrait, full-page journal)

# ── pastel palette ───────────────────────────────────────────────────────────
C = dict(
    neutral=("#EFF1F3", "#9AA0A6"),
    eeg    =("#DCE7F5", "#5C84BE"),   # soft blue
    et     =("#DDEEDD", "#5FA05F"),   # soft green
    roi    =("#FCF3CF", "#C9A227"),   # soft yellow
    fusion =("#E8E1F4", "#8A77BC"),   # soft lavender
    nsym   =("#FCE4D6", "#D98E5A"),   # soft peach
    pred   =("#F8D7DA", "#C0504D"),   # soft red
)

# ── per-panel top-crop fraction (removes the panel's own internal title) ──────
CROP = {"A1": 0.07, "A2": 0.10, "A3": 0.055, "A4": 0.10,
        "B1": 0.065, "B2": 0.065, "B3": 0.065,
        "C1": 0.135, "C2": 0.135, "C3": 0.085,
        "D1": 0.085, "D2": 0.085}

PATHS = {
    "A1": PAN / "A/A1_raw_eeg.png",         "A2": PAN / "A/A2_connectivity_matrix.png",
    "A3": PAN / "A/A3_brain_graph.png",     "A4": PAN / "A/A4_spike_raster.png",
    "B1": PAN / "B/B1_gaze_trajectory.png", "B2": PAN / "B/B2_fixation_map.png",
    "B3": PAN / "B/B3_roi_attention.png",
    "C1": PAN / "C/C1_xattn_eeg_graph.png", "C2": PAN / "C/C2_xattn_eeg_et.png",
    "C3": PAN / "C/C3_fused_representation.png",
    "D1": PAN / "D/D1_rule_activations.png","D2": PAN / "D/D2_prediction.png",
}
CAP = {
    "A1": "Preprocessed EEG (24 ch)", "A2": "Functional connectivity",
    "A3": "Dynamic brain graph",      "A4": "LIF spike raster",
    "B1": "Gaze trajectory",          "B2": "Fixation map",
    "B3": "ROI attention heatmap",
    "C1": "EEG ← Graph cross-attn", "C2": "EEG ← ET cross-attn",
    "C3": "Fused representation",
    "D1": "8 neuro-symbolic rule activations", "D2": "Prediction output",
}

_imgcache = {}
def load(key):
    if key not in _imgcache:
        im = plt.imread(PATHS[key])
        h = im.shape[0]
        im = im[int(CROP[key] * h):]                    # drop internal title band
        _imgcache[key] = im
    return _imgcache[key]


# ── canvas + background overlay (boxes / arrows / text live here) ─────────────
fig = plt.figure(figsize=(FW, FH), facecolor="white")
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
bg.axis("off"); bg.set_zorder(0)


def card(x0, y0, x1, y1, color, title=None, title_size=15, z=1):
    fill, edge = C[color]
    bg.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        fc=fill, ec=edge, lw=2.2, zorder=z, mutation_aspect=FH / FW))
    if title:
        bg.text(x0 + 0.012, y1 - 0.011, title, ha="left", va="top",
                fontsize=title_size, fontweight="bold", color=edge, zorder=z + 1)


def chip(cx, cy, w, h, text, color, fs=10.5):
    fill, edge = C[color]
    bg.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.003,rounding_size=0.008",
        fc="white", ec=edge, lw=1.6, zorder=4, mutation_aspect=FH / FW))
    bg.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color="#222", zorder=5, wrap=True)
    return (cx, cy)


def chain(x0, x1, y, labels, color, h=0.026, fs=10):
    """Equally-spaced chip chain left->right with arrows between."""
    n = len(labels)
    w = (x1 - x0) / n
    cw = w * 0.86
    centers = []
    for i, lab in enumerate(labels):
        cx = x0 + w * (i + 0.5)
        chip(cx, y, cw, h, lab, color, fs=fs)
        centers.append(cx)
    for i in range(n - 1):
        bg.add_patch(FancyArrowPatch((centers[i] + cw / 2, y),
            (centers[i + 1] - cw / 2, y), arrowstyle="-|>",
            mutation_scale=14, color=C[color][1], lw=1.6, zorder=4))


def arrow(p0, p1, color="#6b6b6b", lw=2.4, rad=0.0, ms=24):
    bg.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
        color=color, lw=lw, zorder=3,
        connectionstyle=f"arc3,rad={rad}"))


def fit(x0, y0, x1, y1, ar):
    bw, bh = (x1 - x0) * FW, (y1 - y0) * FH
    if ar > bw / bh:
        nbw, nbh = bw, bw / ar
    else:
        nbh, nbw = bh, bh * ar
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    fw, fh = nbw / FW, nbh / FH
    return [cx - fw / 2, cy - fh / 2, fw, fh]


def panel(key, x0, y0, x1, y1, cap_size=10, frame=True):
    """Place an evidence panel (aspect-preserved) with a short caption above."""
    im = load(key)
    ar = im.shape[1] / im.shape[0]
    cap_h = 0.011
    rect = fit(x0, y0, x1 - 1e-9, y1 - cap_h, ar)
    ax = fig.add_axes(rect, zorder=6)
    ax.imshow(im, aspect="auto"); ax.axis("off")
    if frame:
        for s in ax.spines.values():
            s.set_visible(True); s.set_edgecolor("#bbb"); s.set_linewidth(0.8)
        ax.set_frame_on(True)
    bg.text((x0 + x1) / 2, rect[1] + rect[3] + 0.0035, CAP[key],
            ha="center", va="bottom", fontsize=cap_size, fontweight="bold",
            color="#333", zorder=7)
    return rect


def badge(cx, cy, num, text, color="#B5371F"):
    # aspect-corrected radius so the number sits in a round disc on a 18x26 canvas
    bg.add_patch(matplotlib.patches.Ellipse((cx, cy), 0.020, 0.020 * FW / FH,
                 fc=color, ec="white", lw=2, zorder=10))
    bg.text(cx, cy, num, ha="center", va="center", color="white",
            fontsize=12, fontweight="bold", zorder=11)
    bg.text(cx + 0.016, cy, text, ha="left", va="center", fontsize=11,
            fontweight="bold", color=color, zorder=11)


# ════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════
bg.text(0.5, 0.985, "INS-HDGS-CMT — Interpretable Neuro-Symbolic Hierarchical "
        "Dynamic Graph-Spiking Cross-Modal Transformer",
        ha="center", va="top", fontsize=20, fontweight="bold", color="#1b1b1b")
bg.text(0.5, 0.971, "Architecture with real case-study outputs  ·  Subject S01, "
        "fold-01, epoch 6 (ImagePage_5)  ·  ground truth: HIGH_ENGAGEMENT",
        ha="center", va="top", fontsize=12.5, color="#555")

# ════════════════════════════════════════════════════════════════════════════
# INPUT + PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
card(0.04, 0.928, 0.96, 0.962, "neutral", "1 · Multimodal Input")
for cx, lab in zip((0.34, 0.52, 0.70, 0.86),
                   ("EEG signal", "Eye-tracking sequence", "ROI information",
                    "Engagement label")):
    chip(cx, 0.940, 0.165, 0.020, lab, "neutral", fs=10.5)

card(0.04, 0.878, 0.96, 0.918, "neutral", "2 · Preprocessing & Epoching")
for cx, lab in zip((0.30, 0.475, 0.65, 0.84),
                   ("EEG cleaning +\nband-power", "Eye-tracking\npreprocessing",
                    "ROI dwell /\nattention", "Dynamic graph\nconstruction")):
    chip(cx, 0.892, 0.158, 0.026, lab, "neutral", fs=9.5)

arrow((0.5, 0.928), (0.5, 0.918), rad=0)

# ════════════════════════════════════════════════════════════════════════════
# TOP COLUMNS  (EEG left, ET right)
# ════════════════════════════════════════════════════════════════════════════
EEG = (0.040, 0.560, 0.487, 0.868)
ET  = (0.513, 0.560, 0.960, 0.868)
card(*EEG, "eeg", "3 · EEG Graph-Spiking Pathway")
card(*ET,  "et",  "4 · Eye-Tracking Attention Pathway")

# EEG evidence 2x2  (real outputs)
ex0, ex1 = 0.052, 0.475
panel("A2", ex0, 0.728, 0.262, 0.838)
panel("A3", 0.265, 0.728, ex1, 0.838)
panel("A1", ex0, 0.622, 0.262, 0.726)
panel("A4", 0.265, 0.622, ex1, 0.726)
# EEG processing chain
chain(0.052, 0.475, 0.592,
      ["ROI graph\nmodulation", "DynamicGAT", "LIF Spiking\nEEG encoder",
       "EEG\nrepresentation"], "eeg", h=0.030, fs=8.6)

# ET evidence row  (real outputs)
tx0, tx1 = 0.525, 0.948
panel("B1", tx0, 0.728, 0.690, 0.838)
panel("B2", 0.690, 0.728, tx1, 0.838)
panel("B3", 0.600, 0.620, 0.880, 0.726)
# ET processing chain
chain(0.525, 0.948, 0.592,
      ["ET Transformer\nattention encoder", "ET\nrepresentation",
       "ROI attention\ndistribution"], "et", h=0.030, fs=8.8)

arrow((0.265, 0.878), (0.20, 0.868), rad=0.1)     # prep -> EEG
arrow((0.735, 0.878), (0.80, 0.868), rad=-0.1)    # prep -> ET

# ════════════════════════════════════════════════════════════════════════════
# ROI-GUIDED MODULATION  (yellow, bridges the two columns)
# ════════════════════════════════════════════════════════════════════════════
ROI = (0.150, 0.486, 0.850, 0.548)
card(*ROI, "roi", "5 · ROI-Guided Modulation")
chip(0.32, 0.508, 0.27, 0.026, "EEG graph modulation", "roi", fs=10)
chip(0.66, 0.508, 0.27, 0.026, "EEG representation gating", "roi", fs=10)
# ET ROI attention feeds the modulation, which feeds back into the EEG branch
arrow((0.74, 0.560), (0.66, 0.548), color=C["roi"][1], rad=-0.1)   # ET ROI -> ROI card
arrow((0.32, 0.548), (0.22, 0.560), color=C["roi"][1], rad=-0.1)   # ROI -> EEG branch

# ════════════════════════════════════════════════════════════════════════════
# NEUROFUSION TRANSFORMER  (lavender, merges both columns)
# ════════════════════════════════════════════════════════════════════════════
FUS = (0.040, 0.272, 0.960, 0.476)
card(*FUS, "fusion", "6 · ROI-Guided NeuroFusion Transformer")
# real cross-attention + fused-embedding evidence
panel("C1", 0.060, 0.300, 0.380, 0.388)
panel("C2", 0.060, 0.392, 0.380, 0.452)
panel("C3", 0.395, 0.300, 0.560, 0.452)
# fusion processing chain (right block)
chain(0.585, 0.945, 0.430,
      ["Modality\ntokens", "Self-\nattention", "Directed\ncross-attention",
       "Gated residual\nfusion"], "fusion", h=0.032, fs=8.8)
chip(0.765, 0.345, 0.30, 0.030, "Fused multimodal representation", "fusion", fs=10.5)
arrow((0.585, 0.388), (0.765, 0.362), color=C["fusion"][1], rad=-0.15)

arrow((0.10, 0.560), (0.10, 0.476), rad=0)        # EEG representation -> fusion
arrow((0.90, 0.560), (0.90, 0.476), rad=0)        # ET representation  -> fusion
arrow((0.50, 0.486), (0.50, 0.476), rad=0)        # ROI -> fusion

# ════════════════════════════════════════════════════════════════════════════
# NEURO-SYMBOLIC DECISION  +  PREDICTION  (bottom)
# ════════════════════════════════════════════════════════════════════════════
NS  = (0.040, 0.030, 0.575, 0.256)
PR  = (0.600, 0.030, 0.960, 0.256)
card(*NS, "nsym", "7 · Neuro-Symbolic Decision Layer")
panel("D1", 0.055, 0.060, 0.560, 0.190)
chip(0.190, 0.215, 0.21, 0.026, "Rule ensemble", "nsym", fs=10)
chip(0.430, 0.215, 0.21, 0.026, "Bypass classifier", "nsym", fs=10)

card(*PR, "pred", "Final Prediction")
panel("D2", 0.612, 0.045, 0.952, 0.205)

arrow((0.30, 0.272), (0.30, 0.256), rad=0)        # fusion -> neuro-symbolic
arrow((0.575, 0.150), (0.600, 0.150), rad=0)      # neuro-symbolic -> prediction

# ════════════════════════════════════════════════════════════════════════════
# CONTRIBUTION BADGES
# ════════════════════════════════════════════════════════════════════════════
# numbered badges placed inline on each section's header row (right side, free space)
badge(0.250, 0.862, "1", "DynamicGAT + LIF Spiking Encoder")
badge(0.470, 0.470, "2", "ROI-Guided NeuroFusion Transformer")
badge(0.250, 0.250, "3", "Neuro-Symbolic Rule Layer")

# ════════════════════════════════════════════════════════════════════════════
for ext in ("png", "svg"):
    f = OUT / f"INS_HDGS_CMT_architecture_S01.{ext}"
    fig.savefig(f, dpi=300 if ext == "png" else None,
                facecolor="white", bbox_inches="tight")
    print(f"✓ wrote {f}")
plt.close(fig)
