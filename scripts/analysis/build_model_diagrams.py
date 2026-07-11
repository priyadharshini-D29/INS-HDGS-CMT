#!/usr/bin/env python3
"""
build_model_diagrams.py
=======================
Two clean, publication-grade (Q1) SCHEMATIC architecture diagrams of INS-HDGS-CMT
— pure vector schematics (no raster output panels), faithful to the source code:

    OUTER MODEL  →  high-level two-stream → fusion → decision pipeline
    INNER MODEL  →  in-depth components, sub-layers and tensor shapes

Architecture facts are read directly from models/ins_hdgs_cmt.py (defaults):
    embed_dim 128 everywhere · DynamicGAT (4 heads × 32-d + 3-layer temporal
    transformer, ff 512) · SpikingEEGEncoder (LIF, 2 layers, T=10, band-weighted
    θ/α proxy) · eeg_merge Linear(256→128) · ET attention encoder (2 layers,
    4 heads, hidden 64) → et_emb + 10-ROI attention · ROI graph modulation gates
    the adjacency · ROI attention gates eeg_emb · NeuroFusion transformer
    (3 modalities, 4 heads, 3 layers, ff 512, directed cross-attention) ·
    8-rule neuro-symbolic layer → 2 logits.

Performance strip uses the REAL best LOSOCV subject S24 (focal γ=3, fold-22):
    Accuracy 0.923 · Balanced-acc 0.938 · MCC 0.854 · ROC-AUC 1.000
    (pooled LOSOCV n=37: acc 0.75 · bal-acc 0.74 · MCC 0.46 · AUC 0.90)

Outputs: figures/architecture_v2/{OuterModel,InnerModel}_INS_HDGS_CMT.{png,pdf}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path("/home/nvidia/24PHD1314/Neuma_Model/NEUMA_PHASE8/figures/architecture_v2")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
})

# ── refined palette: (header/border, soft body tint) ─────────────────────────
PAL = {
    "input" : ("#5B6B7B", "#EEF1F4"),   # slate grey
    "eeg"   : ("#3B6FB6", "#E4EDF8"),   # blue
    "snn"   : ("#7A4FB0", "#ECE5F6"),   # violet
    "et"    : ("#2E8B6B", "#E2F1EB"),   # green
    "roi"   : ("#C79A1E", "#FBF3D7"),   # gold
    "fusion": ("#6A5ACD", "#E9E6F8"),   # indigo
    "nsym"  : ("#C8722E", "#FBE9DA"),   # orange
    "pred"  : ("#B23B3B", "#F7DEDE"),   # red
    "ink"   : ("#1F2A37", "#FFFFFF"),
}
EDGE = "#566270"
INK  = "#1F2A37"

HEAD = ("Accuracy 0.923    ·    Balanced-acc 0.938    ·    MCC 0.854    ·    ROC-AUC 1.000")
POOL = ("Best LOSOCV subject S24 (focal γ=3, fold-22) shown   |   "
        "pooled LOSOCV n=37:  acc 0.75 · bal-acc 0.74 · MCC 0.46 · AUC 0.90")


class Diagram:
    def __init__(self, fw, fh):
        self.FW, self.FH = fw, fh
        self.fig = plt.figure(figsize=(fw, fh), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1); self.ax.axis("off")
        self.asp = fh / fw

    # ── container with drop shadow + coloured header strip ────────────────────
    def container(self, x0, y0, x1, y1, key, header=None, hfs=12.5, z=1, lw=2.0):
        edge, body = PAL[key]
        # soft shadow
        sh = 0.004
        self.ax.add_patch(FancyBboxPatch((x0 + sh, y0 - sh), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.002,rounding_size=0.010", fc="#00000018",
            ec="none", zorder=z, mutation_aspect=self.asp))
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.002,rounding_size=0.010", fc=body, ec=edge,
            lw=lw, zorder=z + 1, mutation_aspect=self.asp))
        if header:
            self.ax.text(x0 + 0.012, y1 - 0.006 / self.asp * 0.6, header,
                         ha="left", va="top", fontsize=hfs, fontweight="bold",
                         color=edge, zorder=z + 2)
        return (x0, y0, x1, y1)

    # ── solid node (white card with accent border) ────────────────────────────
    def node(self, cx, cy, w, h, text, key="ink", fs=10, z=5, sub=None, bold=True):
        edge, _ = PAL[key]
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.002,rounding_size=0.006", fc="white", ec=edge,
            lw=1.7, zorder=z, mutation_aspect=self.asp))
        if sub:
            self.ax.text(cx, cy + h * 0.16, text, ha="center", va="center",
                         fontsize=fs, fontweight="bold" if bold else "normal",
                         color=INK, zorder=z + 1)
            self.ax.text(cx, cy - h * 0.24, sub, ha="center", va="center",
                         fontsize=fs - 1.6, color="#5b6470", zorder=z + 1)
        else:
            self.ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                         fontweight="bold" if bold else "normal", color=INK,
                         zorder=z + 1)
        return (cx, cy, w, h)

    # ── filled accent chip (for inputs / small labels) ────────────────────────
    def chip(self, cx, cy, w, h, text, key, fs=10, z=5):
        edge, body = PAL[key]
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.002,rounding_size=0.006", fc=body, ec=edge,
            lw=1.7, zorder=z, mutation_aspect=self.asp))
        self.ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                     color=INK, zorder=z + 1, fontweight="bold")
        return (cx, cy, w, h)

    # ── connector (straight or orthogonal) with optional tensor-shape label ───
    def edge_(self, p0, p1, label=None, color=EDGE, lw=2.2, ms=16, style="arc3",
              rad=0.0, ldx=0.0, ldy=0.012, ls="-", z=4):
        if style == "ortho":
            cs = f"angle,angleA=0,angleB=90,rad=6"
        else:
            cs = f"arc3,rad={rad}"
        self.ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
            mutation_scale=ms, color=color, lw=lw, zorder=z, linestyle=ls,
            connectionstyle=cs, shrinkA=1, shrinkB=1))
        if label:
            mx, my = (p0[0] + p1[0]) / 2 + ldx, (p0[1] + p1[1]) / 2 + ldy
            self.ax.text(mx, my, label, ha="center", va="center", fontsize=8.4,
                         style="italic", color="#39424d", zorder=z + 1,
                         bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                   ec="#cfd6dd", lw=0.8))

    def title(self, tag, main, sub):
        self.ax.text(0.012, 0.975, tag, ha="left", va="top", fontsize=24,
                     fontweight="bold", color=PAL["pred"][0])
        self.ax.text(0.5, 0.978, main, ha="center", va="top", fontsize=19,
                     fontweight="bold", color=INK)
        self.ax.text(0.5, 0.940, sub, ha="center", va="top", fontsize=11.5,
                     color="#5b6470")
        self.ax.add_line(plt.Line2D([0.04, 0.96], [0.915, 0.915],
                         color="#d7dde3", lw=1.2))

    def perf_strip(self, y=0.052):
        self.ax.add_patch(FancyBboxPatch((0.20, y - 0.026), 0.60, 0.052,
            boxstyle="round,pad=0.002,rounding_size=0.010", fc="#E9F5EE",
            ec="#2E8B6B", lw=1.6, mutation_aspect=self.asp, zorder=8))
        self.ax.text(0.5, y + 0.004, HEAD, ha="center", va="center",
                     fontsize=11.5, fontweight="bold", color="#1c6a4a", zorder=9)
        self.ax.text(0.5, y - 0.016, POOL, ha="center", va="center",
                     fontsize=8.6, color="#6b7280", zorder=9, style="italic")

    def legend(self, items, x=0.04, y=0.045, dx=0.0, dy=0.0):
        # items: list of (key, label)
        cx = x
        for key, lab in items:
            edge, body = PAL[key]
            self.ax.add_patch(Rectangle((cx, y), 0.016, 0.016 / self.asp * 0.55,
                fc=body, ec=edge, lw=1.4, zorder=8))
            self.ax.text(cx + 0.022, y + 0.005, lab, ha="left", va="center",
                         fontsize=9, color="#39424d", zorder=8)
            cx += 0.022 + 0.013 * len(lab)

    def save(self, name):
        for ext in ("png", "pdf"):
            f = OUT / f"{name}.{ext}"
            self.fig.savefig(f, dpi=300 if ext == "png" else None,
                             facecolor="white", bbox_inches="tight")
            print(f"  ✓ {f.name}")
        plt.close(self.fig)


# ════════════════════════════════════════════════════════════════════════════
#  OUTER MODEL — high-level two-stream → fusion → decision pipeline
# ════════════════════════════════════════════════════════════════════════════
def outer():
    print("[Outer] high-level pipeline …")
    d = Diagram(16, 9)
    d.title("", "INS-HDGS-CMT — Model Overview",
            "Two-stream multimodal encoder  →  ROI-guided NeuroFusion transformer  →  neuro-symbolic decision")

    # ── inputs (left, stacked) ────────────────────────────────────────────────
    d.chip(0.105, 0.700, 0.150, 0.072, "EEG signal", "eeg", fs=11)
    d.ax.text(0.105, 0.700 - 0.050, "24 ch · 1500 samples", ha="center",
              va="center", fontsize=8.4, color="#5b6470", style="italic")
    d.chip(0.105, 0.470, 0.150, 0.072, "Eye-tracking", "et", fs=11)
    d.ax.text(0.105, 0.470 - 0.050, "600 × 3 (x, y, pupil)", ha="center",
              va="center", fontsize=8.4, color="#5b6470", style="italic")
    d.chip(0.105, 0.275, 0.150, 0.062, "ROI prior", "roi", fs=10.5)
    d.ax.text(0.04, 0.825, "INPUTS", fontsize=11, fontweight="bold", color="#8a939c")

    # ── encoders (two lanes) ──────────────────────────────────────────────────
    d.container(0.260, 0.620, 0.470, 0.790, "eeg",
                "EEG Graph-Spiking Encoder")
    d.ax.text(0.365, 0.690, "DynamicGAT  +  LIF spiking\nfunctional-connectivity graph",
              ha="center", va="center", fontsize=9.6, color=INK)
    d.ax.text(0.365, 0.645, "→ 128-d EEG embedding", ha="center", va="center",
              fontsize=8.8, color="#5b6470", style="italic")

    d.container(0.260, 0.392, 0.470, 0.548, "et",
                "Eye-Tracking Attention Encoder")
    d.ax.text(0.365, 0.462, "Transformer self-attention\n+ learned ROI attention",
              ha="center", va="center", fontsize=9.6, color=INK)
    d.ax.text(0.365, 0.420, "→ 128-d ET embedding", ha="center", va="center",
              fontsize=8.8, color="#5b6470", style="italic")

    # ROI-guided modulation (bridge between lanes)
    d.container(0.260, 0.250, 0.470, 0.330, "roi", "ROI-Guided Modulation")
    d.ax.text(0.365, 0.285, "gates EEG adjacency & embedding\nfrom 10-ROI attention",
              ha="center", va="center", fontsize=9.0, color=INK)

    # ── fusion ────────────────────────────────────────────────────────────────
    d.container(0.540, 0.470, 0.720, 0.700, "fusion", "NeuroFusion")
    d.ax.text(0.630, 0.585, "Cross-Modal\nTransformer", ha="center", va="center",
              fontsize=12, fontweight="bold", color=PAL["fusion"][0])
    d.ax.text(0.630, 0.520, "directed cross-attention\n+ gated residual fusion",
              ha="center", va="center", fontsize=9.0, color=INK)

    # ── neuro-symbolic ────────────────────────────────────────────────────────
    d.container(0.760, 0.470, 0.910, 0.700, "nsym", "Neuro-Symbolic")
    d.ax.text(0.835, 0.585, "8 differentiable\nrules", ha="center", va="center",
              fontsize=11, fontweight="bold", color=PAL["nsym"][0])
    d.ax.text(0.835, 0.522, "+ bypass classifier\n→ logits", ha="center",
              va="center", fontsize=9.0, color=INK)

    # ── prediction ────────────────────────────────────────────────────────────
    d.chip(0.958, 0.585, 0.070, 0.150, "HIGH /\nLOW\nengage-\nment", "pred", fs=10)

    # ── edges ─────────────────────────────────────────────────────────────────
    d.edge_((0.180, 0.700), (0.260, 0.705), label="10×24×5")
    d.edge_((0.180, 0.470), (0.260, 0.470), label="600×3")
    d.edge_((0.180, 0.275), (0.260, 0.290), label=None, color="#C79A1E")
    # ROI lane interactions
    d.edge_((0.365, 0.392), (0.365, 0.330), color="#C79A1E", lw=1.8, ms=13,
            label=None)                                   # ET ROI attn → modulation
    d.edge_((0.365, 0.330), (0.365, 0.392), color="#C79A1E", lw=0)            # spacer
    d.edge_((0.330, 0.330), (0.300, 0.620), color="#C79A1E", lw=1.8, ms=13,
            rad=0.25)                                     # modulation → EEG encoder
    # encoders → fusion
    d.edge_((0.470, 0.700), (0.540, 0.640), label="128-d", rad=-0.05)
    d.edge_((0.470, 0.470), (0.540, 0.540), label="128-d", rad=0.05)
    d.edge_((0.470, 0.290), (0.560, 0.470), color="#C79A1E", lw=1.8, ms=13,
            rad=-0.15)                                    # ROI → fusion
    # fusion → ns → pred
    d.edge_((0.720, 0.585), (0.760, 0.585), label="fused 128-d")
    d.edge_((0.910, 0.585), (0.923, 0.585), label="softmax")

    d.legend([("eeg", "EEG stream"), ("et", "Eye-tracking stream"),
              ("roi", "ROI guidance"), ("fusion", "Fusion"),
              ("nsym", "Neuro-symbolic"), ("pred", "Output")],
             x=0.04, y=0.150)
    d.perf_strip(y=0.075)
    d.save("OuterModel_INS_HDGS_CMT")


# ════════════════════════════════════════════════════════════════════════════
#  INNER MODEL — in-depth components, sub-layers, tensor shapes
# ════════════════════════════════════════════════════════════════════════════
def inner():
    print("[Inner] in-depth schematic …")
    d = Diagram(17, 12)
    d.title("", "INS-HDGS-CMT — Detailed Architecture",
            "Component-level data flow with sub-layers and tensor shapes  (embed dim = 128)")

    # ── INPUTS row ────────────────────────────────────────────────────────────
    d.chip(0.135, 0.870, 0.150, 0.046, "EEG band-power  (10×24×5)", "eeg", fs=9.2)
    d.chip(0.135, 0.815, 0.150, 0.040, "Dynamic adj  (10×24×24)", "eeg", fs=8.6)
    d.chip(0.500, 0.870, 0.140, 0.046, "ROI prior  (10)", "roi", fs=9.2)
    d.chip(0.820, 0.870, 0.150, 0.046, "Eye-tracking  (600×3)", "et", fs=9.2)

    # ════ EEG PATHWAY container (left) ════════════════════════════════════════
    d.container(0.040, 0.300, 0.470, 0.770, "eeg", "EEG Dynamic Graph-Spiking Pathway")
    # band-power layernorm
    d.node(0.140, 0.720, 0.150, 0.046, "Band-power LayerNorm", "eeg", fs=8.8)
    # --- Graph lane ---
    d.node(0.140, 0.628, 0.160, 0.066, "DynamicGAT", "eeg", fs=10,
           sub="per-window GAT · 4 heads × 32-d")
    d.node(0.140, 0.520, 0.160, 0.066, "Temporal Transformer", "eeg", fs=9.4,
           sub="3 layers · 4 heads · ff 512")
    d.ax.text(0.140, 0.452, "graph_emb (128)", ha="center", fontsize=8.4,
              style="italic", color="#39424d")
    # --- Spiking lane ---
    d.node(0.350, 0.700, 0.160, 0.060, "Band weighting", "snn", fs=9.4,
           sub="softmax · favours θ/α")
    d.node(0.350, 0.590, 0.160, 0.072, "LIF Spiking Encoder", "snn", fs=10,
           sub="2 layers · T = 10 steps · 128")
    d.ax.text(0.350, 0.522, "snn_emb (128)", ha="center", fontsize=8.4,
              style="italic", color="#39424d")
    # merge
    d.node(0.245, 0.380, 0.190, 0.060, "EEG merge", "eeg", fs=10,
           sub="concat → Linear(256→128) · GELU · LN")
    # EEG internal edges
    d.edge_((0.140, 0.870 - 0.024 - 0.030), (0.140, 0.743), color=PAL["eeg"][0], ms=12, lw=1.6)
    d.edge_((0.140, 0.697), (0.140, 0.661), color=PAL["eeg"][0], ms=12, lw=1.6)
    d.edge_((0.140, 0.595), (0.140, 0.553), color=PAL["eeg"][0], ms=12, lw=1.6)
    d.edge_((0.140, 0.487), (0.190, 0.410), color=PAL["eeg"][0], ms=12, lw=1.6, rad=-0.1)
    d.edge_((0.350, 0.670), (0.350, 0.626), color=PAL["snn"][0], ms=12, lw=1.6)
    d.edge_((0.350, 0.554), (0.300, 0.410), color=PAL["snn"][0], ms=12, lw=1.6, rad=0.1)
    # band-power feeds spiking lane too
    d.edge_((0.215, 0.720), (0.300, 0.712), color=PAL["snn"][0], ms=11, lw=1.4, rad=-0.2, ls=(0,(4,2)))

    # ════ ET PATHWAY container (right) ════════════════════════════════════════
    d.container(0.690, 0.300, 0.965, 0.770, "et", "Eye-Tracking Attention Pathway")
    d.node(0.825, 0.660, 0.180, 0.072, "ET Transformer Encoder", "et", fs=9.8,
           sub="2 layers · 4 heads · hidden 64")
    d.node(0.825, 0.530, 0.180, 0.060, "ROI Attention Head", "roi", fs=9.8,
           sub="→ 10 ROIs")
    d.ax.text(0.825, 0.470, "et_emb (128)  ·  et_roi_attn (10)", ha="center",
              fontsize=8.4, style="italic", color="#39424d")
    d.edge_((0.825, 0.815 - 0.030 + 0.005), (0.825, 0.697), color=PAL["et"][0], ms=12, lw=1.6)
    d.edge_((0.825, 0.624), (0.825, 0.561), color=PAL["et"][0], ms=12, lw=1.6)

    # ════ ROI modulation + gating (center bridge) ═════════════════════════════
    d.container(0.500, 0.560, 0.660, 0.770, "roi", "ROI-Guided Gating")
    d.node(0.580, 0.700, 0.140, 0.056, "ROI Graph Modulation", "roi", fs=8.8,
           sub="gates adjacency")
    d.node(0.580, 0.610, 0.140, 0.056, "ROI Attention Gate", "roi", fs=8.8,
           sub="gates eeg_emb · gate (128)")
    # ET ROI attn → modulation/gate
    d.edge_((0.735, 0.530), (0.652, 0.690), color=PAL["roi"][0], ms=12, lw=1.6, rad=-0.2)
    d.edge_((0.510, 0.700), (0.222, 0.730), color=PAL["roi"][0], ms=12, lw=1.5,
            rad=0.18)                                          # modulation → adjacency/GAT
    d.edge_((0.510, 0.610), (0.340, 0.392), color=PAL["roi"][0], ms=12, lw=1.5,
            rad=0.12)                                          # gate → eeg merge
    # ROI prior → gating
    d.edge_((0.500, 0.847), (0.560, 0.728), color=PAL["roi"][0], ms=11, lw=1.4, rad=0.1)

    # ════ NEUROFUSION container (center) ══════════════════════════════════════
    d.container(0.250, 0.140, 0.760, 0.270, "fusion",
                "ROI-Guided NeuroFusion Cross-Modal Transformer  (3 modalities · 4 heads · 3 layers · ff 512)")
    d.node(0.330, 0.195, 0.110, 0.055, "Modality tokens", "fusion", fs=8.8,
           sub="graph · eeg · et")
    d.node(0.470, 0.195, 0.105, 0.055, "Self-attention", "fusion", fs=9.0)
    d.node(0.610, 0.195, 0.130, 0.055, "Directed cross-attn", "fusion", fs=8.8,
           sub="eeg←graph · eeg←et")
    d.node(0.720, 0.195, 0.060, 0.055, "Gated\nfusion", "fusion", fs=8.2)
    d.edge_((0.385, 0.195), (0.418, 0.195), ms=12, lw=1.6, color=PAL["fusion"][0])
    d.edge_((0.523, 0.195), (0.545, 0.195), ms=12, lw=1.6, color=PAL["fusion"][0])
    d.edge_((0.675, 0.195), (0.690, 0.195), ms=12, lw=1.6, color=PAL["fusion"][0])
    # feed embeddings into fusion
    d.edge_((0.245, 0.350), (0.300, 0.225), color=PAL["eeg"][0], ms=12, lw=1.7,
            label="eeg / graph 128", ldx=-0.045, ldy=0.0, rad=0.05)
    d.edge_((0.760, 0.470), (0.388, 0.213), color=PAL["et"][0], ms=12, lw=1.7,
            label="et_emb 128", ldx=0.05, ldy=0.014, rad=0.0)

    # ════ NEURO-SYMBOLIC + PREDICTION ═════════════════════════════════════════
    d.container(0.250, 0.040, 0.640, 0.110, "nsym",
                "Neuro-Symbolic Decision Layer")
    d.node(0.360, 0.075, 0.150, 0.044, "8 differentiable rules", "nsym", fs=9.0)
    d.node(0.530, 0.075, 0.130, 0.044, "Bypass classifier", "nsym", fs=9.0)
    d.chip(0.730, 0.075, 0.150, 0.058, "Prediction\nHIGH / LOW  (2 logits)", "pred", fs=9.4)
    d.edge_((0.500, 0.165), (0.430, 0.097), color=PAL["fusion"][0], ms=13, lw=1.8,
            label="fused 128", rad=0.05)
    d.edge_((0.640, 0.075), (0.655, 0.075), color=PAL["nsym"][0], ms=13, lw=1.8)

    # ════ training-only auxiliary (dashed, right margin next to fusion) ════════
    d.ax.add_patch(FancyBboxPatch((0.772, 0.145), 0.193, 0.120,
        boxstyle="round,pad=0.004,rounding_size=0.008", fc="#F4F5F7",
        ec="#9AA0A6", lw=1.4, ls=(0, (5, 3)), mutation_aspect=d.asp, zorder=2))
    d.ax.text(0.781, 0.255, "Training-only objectives", fontsize=9.4,
              fontweight="bold", color="#6b7280", va="top")
    d.ax.text(0.781, 0.232,
              "• GRL → subject classifier (DANN)\n"
              "• Supervised contrastive (EEG ↔ ET)\n"
              "• MMD + rule diversity / sparsity",
              fontsize=8.0, color="#6b7280", va="top")

    d.legend([("eeg", "EEG graph"), ("snn", "Spiking"), ("et", "Eye-tracking"),
              ("roi", "ROI guidance"), ("fusion", "Fusion"), ("nsym", "Neuro-symbolic"),
              ("pred", "Output")], x=0.040, y=0.012)
    d.save("InnerModel_INS_HDGS_CMT")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 64)
    print(" Building OUTER + INNER INS-HDGS-CMT schematic diagrams")
    print("=" * 64)
    outer()
    inner()
    print(f"\n✓ written to {OUT}")
