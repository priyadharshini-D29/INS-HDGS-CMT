#!/usr/bin/env python3
"""
build_flows_S24.py
==================
Five complementary, publication-quality (Q1) architecture-FLOW figures for the
INS-HDGS-CMT model, every panel sourced from REAL model intermediates of the
best-performing LOSOCV subject:

    Subject S24  ·  fold-22  ·  repro_focal_g3p0_effective_num_37  (γ=3, effective-num)
    LOSOCV (real):  Accuracy 0.923 · Balanced-acc 0.938 · MCC 0.854 · ROC-AUC 1.000
    Case sample:    epoch 11 (ImagePage_6) · GT HIGH_ENGAGEMENT · ensemble p(HIGH)=0.698 · CORRECT

The flows (each a stand-alone journal figure):
    Flow A — Full end-to-end system pipeline (multimodal input → prediction)
    Flow B — EEG dynamic graph-spiking pathway
    Flow C — Eye-tracking + ROI-attention pathway
    Flow D — ROI-guided NeuroFusion cross-modal transformer
    Flow E — Neuro-symbolic decision layer

No inference is performed here — this script only COMPOSES the real evidence
panels produced by ``figure_panels_S24.py`` into the journal layouts, annotated
with the real tensor shapes verified from the trained network.

Outputs: figures/flows_S24/Flow{A..E}_*.{png,pdf}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PHASE8 = Path("/home/nvidia/24PHD1314/Neuma_Model/NEUMA_PHASE8")
PAN    = PHASE8 / "figures" / "case_study_S24"
OUT    = PHASE8 / "figures" / "flows_S24"
OUT.mkdir(parents=True, exist_ok=True)

# ── real headline / provenance (verified from CSV + provenance.json) ──────────
SUBJECT   = "S24"
HEADLINE  = "Accuracy 0.923   ·   Balanced-acc 0.938   ·   MCC 0.854   ·   ROC-AUC 1.000"
POOLED    = "Pooled LOSOCV (n=37):  acc 0.75   ·   bal-acc 0.74   ·   MCC 0.46   ·   AUC 0.90"
PROV      = ("Subject S24 · fold-22 · focal γ=3 (effective-num) · epoch 11 (ImagePage_6) · "
            "ground truth HIGH_ENGAGEMENT · ensemble p(HIGH)=0.698 · CORRECT")

# ── pastel palette (shared with the master architecture figure) ───────────────
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
CROP = {"A1": 0.085, "A2": 0.10, "A3": 0.055, "A4": 0.10,
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
_imgcache = {}
def load(key):
    if key not in _imgcache:
        im = plt.imread(PATHS[key]); h = im.shape[0]
        _imgcache[key] = im[int(CROP[key] * h):]
    return _imgcache[key]


# ════════════════════════════════════════════════════════════════════════════
#  Drawing helpers — every figure makes its own (fig, bg) canvas
# ════════════════════════════════════════════════════════════════════════════
class Canvas:
    def __init__(self, fw, fh):
        self.FW, self.FH = fw, fh
        self.fig = plt.figure(figsize=(fw, fh), facecolor="white")
        self.bg  = self.fig.add_axes([0, 0, 1, 1])
        self.bg.set_xlim(0, 1); self.bg.set_ylim(0, 1)
        self.bg.axis("off"); self.bg.set_zorder(0)

    # rounded section card with a coloured header tab
    def card(self, x0, y0, x1, y1, color, title=None, ts=15, z=1):
        fill, edge = C[color]
        self.bg.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            fc=fill, ec=edge, lw=2.2, zorder=z, mutation_aspect=self.FH / self.FW))
        if title:
            self.bg.text(x0 + 0.010, y1 - 0.009 * 26 / self.FH, title, ha="left",
                         va="top", fontsize=ts, fontweight="bold", color=edge, zorder=z + 1)

    # white pill / node
    def chip(self, cx, cy, w, h, text, color, fs=10.5, bold=False):
        fill, edge = C[color]
        self.bg.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            fc="white", ec=edge, lw=1.7, zorder=4, mutation_aspect=self.FH / self.FW))
        self.bg.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                     color="#1d1d1d", zorder=5, fontweight="bold" if bold else "normal")
        return (cx, cy)

    def arrow(self, p0, p1, color="#6b6b6b", lw=2.6, rad=0.0, ms=22, ls="-"):
        self.bg.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
            color=color, lw=lw, zorder=3, linestyle=ls,
            connectionstyle=f"arc3,rad={rad}"))

    # tensor-shape label that rides on an arrow
    def shape(self, cx, cy, text, color="#444"):
        self.bg.text(cx, cy, text, ha="center", va="center", fontsize=8.6,
                     style="italic", color=color, zorder=6,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.8))

    # aspect-preserving rectangle inside a box
    def _fit(self, x0, y0, x1, y1, ar):
        bw, bh = (x1 - x0) * self.FW, (y1 - y0) * self.FH
        if ar > bw / bh: nbw, nbh = bw, bw / ar
        else:            nbh, nbw = bh, bh * ar
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        fw, fh = nbw / self.FW, nbh / self.FH
        return [cx - fw / 2, cy - fh / 2, fw, fh]

    def panel(self, key, x0, y0, x1, y1, cap=None, cap_size=10.5, frame=True):
        im = load(key); ar = im.shape[1] / im.shape[0]
        cap_h = 0.012 * 26 / self.FH if cap else 0
        rect = self._fit(x0, y0, x1 - 1e-9, y1 - cap_h, ar)
        ax = self.fig.add_axes(rect, zorder=6)
        ax.imshow(im, aspect="auto"); ax.axis("off")
        if frame:
            for s in ax.spines.values():
                s.set_visible(True); s.set_edgecolor("#bbb"); s.set_linewidth(0.8)
            ax.set_frame_on(True)
        if cap:
            self.bg.text((x0 + x1) / 2, rect[1] + rect[3] + 0.004 * 26 / self.FH, cap,
                         ha="center", va="bottom", fontsize=cap_size, fontweight="bold",
                         color="#333", zorder=7)
        return rect

    def title(self, main, sub):
        self.bg.text(0.5, 1 - 0.018 * 9 / self.FH * (self.FH / 9), main,
                     ha="center", va="top", fontsize=20, fontweight="bold", color="#1b1b1b")
        # robust top placement
    def header(self, tag, main, sub1, sub2=None):
        self.bg.text(0.012, 0.985, tag, ha="left", va="top", fontsize=22,
                     fontweight="bold", color="#B5371F")
        self.bg.text(0.5, 0.985, main, ha="center", va="top", fontsize=18.5,
                     fontweight="bold", color="#1b1b1b")
        self.bg.text(0.5, 0.945, sub1, ha="center", va="top", fontsize=11.5, color="#555")
        if sub2:
            self.bg.text(0.5, 0.915, sub2, ha="center", va="top", fontsize=10.5,
                         color="#888", style="italic")

    def metric_badge(self, x, y, text=HEADLINE):
        self.bg.text(x, y, text, ha="center", va="center", fontsize=11.5,
                     fontweight="bold", color="#15633b",
                     bbox=dict(boxstyle="round,pad=0.5", fc="#E7F4EC", ec="#5FA05F", lw=1.6),
                     zorder=12)

    def save(self, name):
        for ext in ("png", "pdf"):
            f = OUT / f"{name}.{ext}"
            self.fig.savefig(f, dpi=300 if ext == "png" else None,
                             facecolor="white", bbox_inches="tight")
            print(f"  ✓ {f.name}")
        plt.close(self.fig)


# ════════════════════════════════════════════════════════════════════════════
#  FLOW A — FULL END-TO-END SYSTEM PIPELINE
# ════════════════════════════════════════════════════════════════════════════
def flow_A():
    print("[Flow A] full pipeline …")
    cv = Canvas(18, 11)
    cv.header("A", "INS-HDGS-CMT — End-to-End Multimodal Engagement-Decoding Pipeline",
              "Interpretable Neuro-Symbolic Hierarchical Dynamic Graph-Spiking Cross-Modal Transformer",
              PROV)
    cv.metric_badge(0.5, 0.875, HEADLINE)

    # row of 8 stages, left→right
    stages = [
        ("neutral", "1 · Multimodal\nInput",
         "EEG  (1500×24)\nEye-track (600×3)\nROI / label"),
        ("neutral", "2 · Preprocess\n& Epoch",
         "band-power\n10 windows ×\n24 ch × 5 bands"),
        ("eeg", "3 · EEG Graph-\nSpiking", "DynamicGAT +\nLIF spiking\n→ 128-d"),
        ("et",  "4 · Eye-Track\nAttention", "ET transformer\n+ ROI attn\n→ 128-d"),
        ("roi", "5 · ROI-Guided\nModulation", "10 ROIs gate\nEEG graph &\nrepresentation"),
        ("fusion", "6 · NeuroFusion\nTransformer", "directed cross-\nattention +\ngated fusion → 128"),
        ("nsym", "7 · Neuro-\nSymbolic", "8 rules +\nbypass MLP\n→ logits"),
        ("pred", "8 · Prediction", "softmax\np(HIGH)=0.698\nHIGH ✓"),
    ]
    n = len(stages)
    x0, x1, y = 0.030, 0.970, 0.66
    w = (x1 - x0) / n
    cw = w * 0.88
    bh = 0.135
    centers = []
    for i, (col, head, body) in enumerate(stages):
        cx = x0 + w * (i + 0.5)
        cv.card(cx - cw / 2, y - bh / 2, cx + cw / 2, y + bh / 2, col)
        cv.bg.text(cx, y + bh / 2 - 0.012, head, ha="center", va="top",
                   fontsize=11, fontweight="bold", color=C[col][1], zorder=5)
        cv.bg.text(cx, y - bh / 2 + 0.052, body, ha="center", va="center",
                   fontsize=8.8, color="#333", zorder=5)
        centers.append(cx)
    for i in range(n - 1):
        cv.arrow((centers[i] + cw / 2, y), (centers[i + 1] - cw / 2, y), ms=18, lw=2.2)

    # real evidence thumbnails below each relevant stage
    cv.panel("A3", centers[2] - 0.075, 0.30, centers[2] + 0.075, 0.50,
             cap="EEG brain graph")
    cv.panel("B3", centers[3] - 0.075, 0.30, centers[3] + 0.075, 0.50,
             cap="ROI attention")
    cv.panel("C3", centers[5] - 0.072, 0.30, centers[5] + 0.072, 0.50,
             cap="Fused 128-d")
    cv.panel("D1", centers[6] - 0.085, 0.30, centers[6] + 0.085, 0.50,
             cap="8 rule activations")
    cv.panel("D2", centers[7] - 0.105, 0.165, centers[7] + 0.105, 0.50,
             cap="Prediction")
    for i in (2, 3, 5, 6, 7):
        cv.arrow((centers[i], y - bh / 2), (centers[i], 0.505), color="#aaaaaa", lw=1.4, ms=12)

    cv.bg.text(0.5, 0.045, POOLED, ha="center", va="center", fontsize=10.5,
               color="#666", style="italic")
    cv.save("FlowA_full_pipeline_S24")


# ════════════════════════════════════════════════════════════════════════════
#  FLOW B — EEG DYNAMIC GRAPH-SPIKING PATHWAY
# ════════════════════════════════════════════════════════════════════════════
def flow_B():
    print("[Flow B] EEG graph-spiking pathway …")
    cv = Canvas(18, 10)
    cv.header("B", "EEG Dynamic Graph-Spiking Pathway",
              "Functional connectivity → DynamicGAT → LIF spiking encoder",
              PROV)

    # top: processing chain of nodes with tensor shapes
    chain = [
        ("Raw EEG\nepoch", "1500 × 24", "eeg"),
        ("Sliding\nwindows", "10 × 24 × 5\n(band-power)", "eeg"),
        ("Pearson\nconnectivity", "10 × 24 × 24", "eeg"),
        ("Dynamic\nbrain graph", "10 graphs", "eeg"),
        ("DynamicGAT\n(graph attn)", "→ 128-d", "eeg"),
        ("LIF Spiking\nencoder", "128 neurons\n× 20 steps", "eeg"),
        ("EEG\nrepresentation", "128-d", "eeg"),
    ]
    n = len(chain); x0, x1, y = 0.035, 0.965, 0.78
    w = (x1 - x0) / n; cw = w * 0.80; ch = 0.085
    centers = []
    for i, (head, shp, col) in enumerate(chain):
        cx = x0 + w * (i + 0.5)
        cv.chip(cx, y, cw, ch, head, col, fs=9.6, bold=True)
        centers.append((cx, cw))
    for i in range(n - 1):
        x_a = centers[i][0] + centers[i][1] / 2
        x_b = centers[i + 1][0] - centers[i + 1][1] / 2
        cv.arrow((x_a, y), (x_b, y), ms=16, lw=2.0)
        cv.shape((x_a + x_b) / 2, y - 0.052, chain[i + 1][1])

    # bottom: real evidence panels aligned under their stage
    cv.panel("A1", 0.030, 0.075, 0.270, 0.60, cap="A1 · Preprocessed EEG (24 ch, 2 s)")
    cv.panel("A2", 0.290, 0.205, 0.520, 0.60, cap="A2 · Functional connectivity (DynamicGAT input)")
    cv.panel("A3", 0.540, 0.205, 0.760, 0.60, cap="A3 · Dynamic brain graph (top-15% edges)")
    cv.panel("A4", 0.775, 0.205, 0.985, 0.60, cap="A4 · LIF spike raster (128 × 20)")
    # connector arrows panel→chain
    for (cx, _), px in zip([centers[0], centers[2], centers[3], centers[5]],
                           (0.150, 0.405, 0.650, 0.880)):
        cv.arrow((px, 0.60), (cx, y - ch / 2), color="#aaaaaa", lw=1.3, ms=11, rad=0.05)

    cv.bg.text(0.5, 0.03, "Spiking encoder yields sparse, energy-efficient temporal codes; "
               "graph attention captures dynamic inter-channel connectivity.",
               ha="center", va="center", fontsize=10, color="#666", style="italic")
    cv.save("FlowB_eeg_graph_spiking_S24")


# ════════════════════════════════════════════════════════════════════════════
#  FLOW C — EYE-TRACKING + ROI-ATTENTION PATHWAY
# ════════════════════════════════════════════════════════════════════════════
def flow_C():
    print("[Flow C] eye-tracking pathway …")
    cv = Canvas(18, 10)
    cv.header("C", "Eye-Tracking + ROI-Attention Pathway",
              "Gaze dynamics → transformer encoder → learned region-of-interest attention",
              PROV)

    chain = [
        ("Raw gaze\nsequence", "600 × 3\n(x, y, pupil)", "et"),
        ("Fixation /\nsaccade", "dispersion\nsegmentation", "et"),
        ("ET Transformer\nencoder", "self-attention\n→ 128-d", "et"),
        ("ROI attention\nhead", "10 ROIs", "roi"),
        ("ET\nrepresentation", "128-d", "et"),
        ("ROI modulation\n→ EEG branch", "gate signal", "roi"),
    ]
    n = len(chain); x0, x1, y = 0.045, 0.955, 0.78
    w = (x1 - x0) / n; cw = w * 0.82; ch = 0.085
    centers = []
    for i, (head, shp, col) in enumerate(chain):
        cx = x0 + w * (i + 0.5)
        cv.chip(cx, y, cw, ch, head, col, fs=9.6, bold=True)
        centers.append((cx, cw))
    for i in range(n - 1):
        x_a = centers[i][0] + centers[i][1] / 2
        x_b = centers[i + 1][0] - centers[i + 1][1] / 2
        cv.arrow((x_a, y), (x_b, y), ms=16, lw=2.0)
        cv.shape((x_a + x_b) / 2, y - 0.052, chain[i + 1][1])

    cv.panel("B1", 0.045, 0.08, 0.355, 0.60, cap="B1 · Gaze trajectory (time-coded)")
    cv.panel("B2", 0.360, 0.08, 0.670, 0.60, cap="B2 · Fixation map (circle ∝ dwell)")
    cv.panel("B3", 0.675, 0.08, 0.985, 0.60, cap="B3 · Learned ROI attention (10 ROIs)")
    for px, (cx, _) in zip((0.20, 0.515, 0.83), [centers[0], centers[1], centers[3]]):
        cv.arrow((px, 0.60), (cx, y - ch / 2), color="#aaaaaa", lw=1.3, ms=11, rad=0.05)

    cv.bg.text(0.5, 0.03, "ROI attention is the bridge: it both forms the ET representation and "
               "modulates the EEG graph-spiking branch (ROI-guided fusion).",
               ha="center", va="center", fontsize=10, color="#666", style="italic")
    cv.save("FlowC_eyetracking_roi_S24")


# ════════════════════════════════════════════════════════════════════════════
#  FLOW D — ROI-GUIDED NEUROFUSION CROSS-MODAL TRANSFORMER
# ════════════════════════════════════════════════════════════════════════════
def flow_D():
    print("[Flow D] NeuroFusion transformer …")
    cv = Canvas(16, 11)
    cv.header("D", "ROI-Guided NeuroFusion Cross-Modal Transformer",
              "Four modality tokens → self-attention → directed cross-attention → gated residual fusion",
              PROV)

    # four input modality tokens (left)
    toks = [("Graph\nembedding", "eeg"), ("SNN\nembedding", "eeg"),
            ("EEG\nembedding", "eeg"), ("ET\nembedding", "et")]
    tx, ty0, ty1 = 0.12, 0.30, 0.78
    ys = [ty1 - (ty1 - ty0) * i / 3 for i in range(4)]
    for (lab, col), yy in zip(toks, ys):
        cv.chip(tx, yy, 0.15, 0.07, lab + "\n(128-d)", col, fs=9.4, bold=True)
    cv.bg.text(tx, ty1 + 0.055, "Modality tokens", ha="center", fontsize=11,
               fontweight="bold", color="#444")

    # self-attention block
    sa_x = 0.36
    cv.card(sa_x - 0.075, 0.36, sa_x + 0.075, 0.72, "fusion", "Self-\nattention", ts=12)
    for yy in ys:
        cv.arrow((tx + 0.075, yy), (sa_x - 0.078, 0.54), color="#8A77BC", lw=1.6,
                 ms=13, rad=0.06)

    # directed cross-attention block (with real attn panels)
    cv.card(0.50, 0.10, 0.80, 0.80, "fusion", "Directed cross-attention")
    cv.panel("C1", 0.515, 0.46, 0.785, 0.66, cap="EEG ← Graph attention")
    cv.panel("C2", 0.515, 0.16, 0.785, 0.42, cap="EEG ← ET attention")
    cv.arrow((sa_x + 0.078, 0.54), (0.50, 0.55), color="#8A77BC", lw=2.2, ms=16)

    # gated residual fusion → fused representation
    cv.chip(0.90, 0.62, 0.13, 0.07, "Gated residual\nfusion\n(gate 128-d)", "fusion",
            fs=9.0, bold=True)
    cv.arrow((0.80, 0.55), (0.90, 0.605), color="#8A77BC", lw=2.2, ms=16, rad=-0.1)
    cv.panel("C3", 0.835, 0.16, 0.985, 0.46, cap="Fused 128-d")
    cv.arrow((0.90, 0.585), (0.90, 0.475), color="#8A77BC", lw=2.0, ms=14)

    cv.bg.text(0.5, 0.045, "Cross-attention lets the EEG query selectively read graph- and "
               "eye-tracking key tokens; a learned 128-d gate fuses them with a residual path.",
               ha="center", va="center", fontsize=10, color="#666", style="italic")
    cv.save("FlowD_neurofusion_transformer_S24")


# ════════════════════════════════════════════════════════════════════════════
#  FLOW E — NEURO-SYMBOLIC DECISION LAYER
# ════════════════════════════════════════════════════════════════════════════
def flow_E():
    print("[Flow E] neuro-symbolic decision …")
    cv = Canvas(16, 10)
    cv.header("E", "Neuro-Symbolic Decision Layer",
              "Differentiable rule ensemble + bypass classifier → gated, interpretable prediction",
              PROV)

    # fused input
    cv.chip(0.12, 0.55, 0.15, 0.08, "Fused multimodal\nrepresentation\n(128-d)", "fusion",
            fs=9.4, bold=True)

    # two parallel branches
    cv.card(0.30, 0.56, 0.66, 0.80, "nsym", "Differentiable rule ensemble (8 rules)")
    cv.panel("D1", 0.315, 0.585, 0.645, 0.73, cap=None)
    cv.card(0.30, 0.30, 0.66, 0.50, "neutral", "Bypass classifier (MLP)")
    cv.bg.text(0.48, 0.40, "direct logit path\n(skips symbolic rules)", ha="center",
               va="center", fontsize=10, color="#555")

    cv.arrow((0.195, 0.57), (0.30, 0.66), color="#D98E5A", lw=2.0, ms=15, rad=0.08)
    cv.arrow((0.195, 0.53), (0.30, 0.40), color="#9AA0A6", lw=2.0, ms=15, rad=-0.08)

    # gated combination
    cv.chip(0.76, 0.55, 0.12, 0.09, "Learned gate\ncombination", "nsym", fs=9.4, bold=True)
    cv.arrow((0.66, 0.68), (0.76, 0.585), color="#D98E5A", lw=2.0, ms=14, rad=-0.08)
    cv.arrow((0.66, 0.40), (0.76, 0.515), color="#9AA0A6", lw=2.0, ms=14, rad=0.08)

    # prediction panel
    cv.panel("D2", 0.66, 0.05, 0.985, 0.45, cap="Engagement prediction")
    cv.arrow((0.76, 0.505), (0.82, 0.455), color="#C0504D", lw=2.2, ms=15)

    cv.bg.text(0.5, 0.025, "Rule activations make each decision auditable; the bypass path preserves "
               "accuracy when no rule fires. Output: HIGH_ENGAGEMENT, p=0.698 (correct).",
               ha="center", va="center", fontsize=10, color="#666", style="italic")
    cv.save("FlowE_neuro_symbolic_S24")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 64)
    print(" Building 5 INS-HDGS-CMT architecture flows (real S24 outputs)")
    print("=" * 64)
    flow_A(); flow_B(); flow_C(); flow_D(); flow_E()
    print(f"\n✓ all flows written to {OUT}")
