"""
INS-HDGS-CMT — Visual Pipeline Diagram
Subject S01, Epoch 47 — HIGH ENGAGEMENT

Run:  python visualization/pipeline_visual.py
Output: pipeline_visual.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "eeg"    : "#4A90D9",
    "et"     : "#E8A838",
    "roi"    : "#8E44AD",
    "snn"    : "#27AE60",
    "gat"    : "#2980B9",
    "merge"  : "#16A085",
    "fusion" : "#7D3C98",
    "rule"   : "#C0392B",
    "decide" : "#1E8449",
    "hdr"    : "#2C3E50",
    "bg"     : "#F4F6F7",
    "wh"     : "#FFFFFF",
}


def make_fig():
    fig = plt.figure(figsize=(26, 34), facecolor=C["bg"])
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 34)
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc, title, sub="", shape="",
         tf=12, sf=9, shf=8, alpha=0.93, ec="#333"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0,rounding_size=0.25",
                       lw=1.6, edgecolor=ec, facecolor=fc, alpha=alpha, zorder=3)
    ax.add_patch(p)
    ty = y + h/2
    if sub:  ty += 0.22
    if shape: ty += 0.1
    ax.text(x+w/2, ty, title, ha="center", va="center",
            fontsize=tf, fontweight="bold", color=C["hdr"], zorder=4)
    if sub:
        sy = y + h/2
        if shape: sy += 0.0
        ax.text(x+w/2, sy - (0.22 if sub else 0), sub,
                ha="center", va="center",
                fontsize=sf, color="#444", style="italic", zorder=4)
    if shape:
        ax.text(x+w/2, y+0.17, shape, ha="center", va="bottom",
                fontsize=shf, color="#666", family="monospace", zorder=4)


def arr(ax, x1, y1, x2, y2, col=C["hdr"], lw=2.0, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                connectionstyle=f"arc3,rad={rad}",
                                mutation_scale=14),
                zorder=5)


def hdr(ax, x, y, w, txt, col):
    p = FancyBboxPatch((x, y), w, 0.5,
                       boxstyle="round,pad=0,rounding_size=0.15",
                       lw=1, edgecolor=col, facecolor=col, alpha=0.15, zorder=2)
    ax.add_patch(p)
    ax.text(x+w/2, y+0.25, txt, ha="center", va="center",
            fontsize=10, fontweight="bold", color=col, zorder=3)


def tensor_tag(ax, x, y, txt, col):
    ax.text(x, y, txt, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=col, zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc=C["wh"],
                      ec=col, lw=1.2, alpha=0.95))


# ─────────────────────────────────────────────────────────────────────────────
fig, ax = make_fig()

# ══ TITLE ════════════════════════════════════════════════════════════════════
ax.text(13, 33.4, "INS-HDGS-CMT  ·  Data Flow  (Subject S01, Epoch 47)",
        ha="center", va="center", fontsize=18, fontweight="bold", color=C["hdr"])
ax.text(13, 32.9,
        "True Label: HIGH_ENGAGEMENT (1)   ·   Model Prediction: HIGH_ENGAGEMENT   ·   P(HIGH) = 0.903",
        ha="center", va="center", fontsize=12, color=C["decide"], fontweight="bold")
ax.plot([0.4, 25.6], [32.62, 32.62], color="#BDC3C7", lw=1.2)

# ══ STAGE 0 — RAW INPUTS ════════════════════════════════════════════════════
hdr(ax, 0.4, 32.18, 25.2, "STAGE 0  —  Raw Inputs from DataLoader", C["hdr"])

rbox(ax, 0.5, 30.5, 5.5, 1.45, C["eeg"],
     "EEG Windows",
     "δ  θ  α  β  γ  band powers",
     "shape: (1, 10 windows, 20 ch, 5 bands)")

rbox(ax, 6.5, 30.5, 5.5, 1.45, C["gat"],
     "Adjacency Matrix",
     "Pearson r > 0.30  (per window)",
     "shape: (1, 10, 20 electrodes, 20 electrodes)")

rbox(ax, 12.5, 30.5, 6.0, 1.45, C["et"],
     "Eye Tracking",
     "gaze_x,  gaze_y,  pupil_mm",
     "shape: (1, 600 samples @ 120 Hz, 3 ch)")

rbox(ax, 19.2, 30.5, 6.0, 1.45, C["roi"],
     "ROI Vector",
     "Dwell time per screen grid cell",
     "shape: (1, 10 regions)  e.g. [0.12, 0.28, ...]")

# ══ STAGE 1 — ET ENCODER ════════════════════════════════════════════════════
hdr(ax, 12.0, 29.68, 13.2, "STAGE 1  —  ET Attention Encoder", C["et"])

arr(ax, 15.5, 30.5, 15.5, 30.22, col=C["et"])

rbox(ax, 12.1, 28.1, 13.0, 1.35, C["et"],
     "BiLSTM  (hidden=64, bidirectional)  +  Self-Attention",
     "Reads 600 gaze frames; fixation events get 3× higher attention weight",
     "shape: (1,600,3)  →  (1,600,128)  →  3 outputs below")

# Three output tags
arr(ax, 13.5, 28.1, 13.0, 27.35, col=C["et"])
tensor_tag(ax, 12.5, 27.2, "et_emb\n(1, 128)", C["et"])

arr(ax, 15.5, 28.1, 15.5, 27.35, col=C["et"])
tensor_tag(ax, 15.5, 27.2, "et_window_seq\n(1, 10, 128)", C["et"])

arr(ax, 17.5, 28.1, 18.0, 27.35, col=C["roi"])
tensor_tag(ax, 18.5, 27.2, "et_roi_attn\n(1, 10)", C["roi"])

# ══ STAGE 2 — ROI GRAPH MODULATION ═════════════════════════════════════════
hdr(ax, 0.4, 29.68, 11.2, "STAGE 2  —  ROI Graph Modulation", C["roi"])

arr(ax, 3.25, 30.5, 3.25, 30.22, col=C["gat"])

rbox(ax, 0.5, 28.1, 11.0, 1.35, C["roi"],
     "ROI Graph Modulation",
     "et_roi_attn → Linear(10→20) → sigmoid → electrode scale factors\n"
     "adj_matrices × scale  →  O2,T6 edges boosted ×1.4 (subject looked left)",
     "(1,10,20,20) + (1,10)  →  adj_modulated  (1,10,20,20)")

# curved arrow from et_roi_attn into roi modulation
ax.annotate("", xy=(8.0, 28.8), xytext=(18.0, 27.2),
            arrowprops=dict(arrowstyle="-|>", color=C["roi"], lw=1.6,
                            connectionstyle="arc3,rad=-0.22", mutation_scale=12),
            zorder=5)
ax.text(13.5, 28.45, "gaze told graph\nwhich electrodes matter",
        ha="center", fontsize=8, color=C["roi"],
        bbox=dict(boxstyle="round,pad=0.15", fc=C["wh"], ec=C["roi"], lw=0.8))

arr(ax, 5.5, 28.1, 5.5, 27.35, col=C["roi"])
tensor_tag(ax, 5.5, 27.2, "adj_modulated\n(1, 10, 20, 20)", C["roi"])

# ══ STAGE 3a — DYNAMIC GAT ══════════════════════════════════════════════════
hdr(ax, 0.4, 26.85, 11.2, "STAGE 3a  —  Dynamic GAT  (Graph Branch)", C["gat"])

arr(ax, 5.5, 27.2, 4.0, 26.88, col=C["roi"])
arr(ax, 1.5, 30.5, 1.5, 29.0, col=C["eeg"])

rbox(ax, 0.5, 25.45, 5.0, 1.2, C["gat"],
     "GAT  Layer 1 + 2",
     "4-head attention (head_dim=32)\nElectrode attends neighbors per 500ms window",
     "(10,20,5) + adj  →  (10,20,128)")

arr(ax, 3.0, 25.45, 3.0, 24.72, col=C["gat"])
ax.text(3.0, 24.58, "mean over\n20 electrodes", ha="center", fontsize=8,
        color="#555",
        bbox=dict(boxstyle="round,pad=0.12", fc=C["wh"], ec="#CCC", lw=0.8))

rbox(ax, 0.5, 23.3, 5.0, 1.05, C["gat"],
     "Temporal Transformer  (3 layers, 4 heads)",
     "10 windows attend each other  →  window_seq",
     "(1,10,128)  →  (1,10,128)")

arr(ax, 3.0, 23.3, 3.0, 22.6, col=C["gat"])
ax.text(3.0, 22.47, "mean-pool + LN", ha="center", fontsize=8, color="#555",
        bbox=dict(boxstyle="round,pad=0.12", fc=C["wh"], ec="#CCC", lw=0.8))

rbox(ax, 0.5, 21.6, 5.0, 0.68, C["gat"],
     "graph_emb   (1, 128)",
     "Spatial-temporal connectivity summary")

# window_seq side tag
arr(ax, 5.5, 23.85, 6.3, 23.85, col=C["gat"])
tensor_tag(ax, 7.5, 23.85, "window_seq\n(1,10,128)", C["gat"])

# ══ STAGE 3b — SNN ══════════════════════════════════════════════════════════
hdr(ax, 12.0, 26.85, 13.2, "STAGE 3b  —  Spiking EEG Encoder  (SNN)", C["snn"])

arr(ax, 2.5, 30.5, 15.0, 26.88, col=C["eeg"])

rbox(ax, 12.1, 25.45, 6.0, 1.2, C["snn"],
     "Band Weighting  (θ=37%  α=33%  learned)",
     "softmax([0.05,0.40,0.35,0.10,0.10]) × band powers",
     "(1,10,20,5)  →  raw_proxy  (1,20,10)")

arr(ax, 15.1, 25.45, 15.1, 24.72, col=C["snn"])

rbox(ax, 12.1, 23.3, 6.0, 1.05, C["snn"],
     "LIF Neurons  (decay=0.90, threshold=1.0)",
     "Membrane V builds each step → spike → reset",
     "spike trains  (1,10,128)   fire_rate=0.2/step")

arr(ax, 15.1, 23.3, 15.1, 22.6, col=C["snn"])
ax.text(15.1, 22.47, "mean_rate ‖ max_burst\nLinear(256→128)", ha="center", fontsize=8,
        color="#555",
        bbox=dict(boxstyle="round,pad=0.12", fc=C["wh"], ec="#CCC", lw=0.8))

rbox(ax, 12.1, 21.6, 6.0, 0.68, C["snn"],
     "snn_emb   (1, 128)",
     "Spike timing + fire rate summary")

# Spike train mini-chart
ax_snn = fig.add_axes([0.705, 0.627, 0.07, 0.04])
spike_y = np.array([0,0,0,0,0,1,0,0,0,1])
ax_snn.bar(range(10), spike_y, color=C["snn"], width=0.7)
ax_snn.set_xlim(-0.5, 9.5); ax_snn.set_ylim(0, 1.5)
ax_snn.set_xticks([]); ax_snn.set_yticks([0,1])
ax_snn.set_yticklabels(["","↑spike"], fontsize=6)
ax_snn.set_facecolor("#EAFAF1")
ax_snn.spines["top"].set_visible(False)
ax_snn.spines["right"].set_visible(False)
ax_snn.set_title("spike train [0,0,0,0,0,1,0,0,0,1]", fontsize=6.5, pad=1.5)

# ══ STAGE 3c — MERGE ════════════════════════════════════════════════════════
hdr(ax, 0.4, 21.18, 18.0, "STAGE 3c  —  Merge  (SNN + GAT  →  eeg_emb)", C["merge"])

arr(ax, 3.0, 21.6, 5.5, 21.0, col=C["gat"])
arr(ax, 15.1, 21.6, 12.5, 21.0, col=C["snn"])

rbox(ax, 5.6, 19.6, 6.8, 1.2, C["merge"],
     "EEG Merge  :  cat  →  Linear(256→128)  +  GELU  +  LN",
     "cat(graph_emb, snn_emb)  →  (1,256)  →  (1,128)\n"
     "cosine_sim(graph_emb, snn_emb) = 0.61  →  complementary",
     "eeg_emb   (1, 128)")

arr(ax, 9.0, 19.6, 9.0, 18.88, col=C["merge"])
tensor_tag(ax, 9.0, 18.75, "eeg_emb   (1, 128)", C["merge"])

# ══ STAGE 4a — ROI GATE ═════════════════════════════════════════════════════
hdr(ax, 0.4, 18.35, 18.0, "STAGE 4a  —  ROI Attention Gate", C["roi"])

arr(ax, 9.0, 18.75, 9.0, 18.38, col=C["merge"])
arr(ax, 22.2, 30.5, 16.5, 18.38, col=C["roi"])   # roi_vector long arrow

rbox(ax, 5.6, 16.9, 6.8, 1.2, C["roi"],
     "ROI Attention Gate",
     "gate = sigmoid(Linear(roi→128))  |  eeg_emb × gate + additive bias\n"
     "Suppresses dims 23,47,91 (unseen regions) → 0.32  |  boosts dim 12,88 → 0.82",
     "eeg_emb (gated)   (1, 128)")

arr(ax, 9.0, 16.9, 9.0, 16.18, col=C["roi"])
tensor_tag(ax, 9.0, 16.05, "eeg_emb (gated)   (1, 128)", C["roi"])

# ══ STAGE 4b — FUSION TRANSFORMER ══════════════════════════════════════════
hdr(ax, 0.4, 15.62, 25.2,
    "STAGE 4b  —  Neuro Fusion Transformer  (4-Stage Cross-Modal)", C["fusion"])

bw, bh, by = 5.8, 2.6, 12.65

# ① EEG self-attn
rbox(ax, 0.5, by, bw, bh, "#D6EAF8",
     "① EEG Self-Attention",
     "EEG windows attend each other\nwindow 5 attends windows 3,4,6 most\n(peak engagement moment identified)",
     "window_seq (1,10,128)\n→ (1,10,128)")

# ② EEG→ET cross-attn
rbox(ax, 0.5+bw+0.3, by, bw, bh, "#FEF9E7",
     "② EEG → ET  Cross-Attention",
     "Q = eeg_emb  (1,1,128)\nK,V = et_window_seq  (1,10,128)\nHighest weight: ET windows 3,4,5\n(t=1.5–2.5s) = spike burst ↔ fixation",
     "eeg_cross   (1,128)")

# ③ ET→EEG cross-attn
rbox(ax, 0.5+2*(bw+0.3), by, bw, bh, "#FEF9E7",
     "③ ET → EEG  Cross-Attention",
     "Q = et_emb  (1,1,128)\nK,V = window_seq  (1,10,128)\nHighest weight: EEG windows 5,6,7\n(t=2.5–3.5s) = fixation→neural 0.5s lag",
     "et_cross   (1,128)")

# ④ Gated fusion
rbox(ax, 0.5+3*(bw+0.3), by, bw, bh, "#E8DAEF",
     "④ Gated Fusion",
     "fused = LayerNorm(\n  eeg_emb + eeg_cross + et_cross)\n\n★ 150ms visual→neural lag\n   learned automatically",
     "fused   (1, 128)")

# arrows between sub-boxes
for i in range(3):
    arr(ax, 0.5+(i+1)*(bw+0.3)-0.3, by+bh/2,
            0.5+(i+1)*(bw+0.3),     by+bh/2,
            col="#888", lw=1.8)

# input arrows into fusion
arr(ax, 9.0, 16.05, 3.4, 15.65, col=C["merge"])          # eeg_emb
arr(ax, 15.5, 27.2, 7.5, 15.65, col=C["et"])              # et_window_seq
arr(ax, 7.5, 23.85, 3.4, 15.65, col=C["gat"])             # window_seq (curved)
arr(ax, 12.5, 27.2, 10.0, 15.65, col=C["et"])             # et_emb

# output of fusion
arr(ax, 21.5+bw/2-0.5, by, 22.4, 12.3, col=C["fusion"])
tensor_tag(ax, 23.5, 12.18, "fused\n(1, 128)", C["fusion"])

# ══ STAGE 5 — RULE LAYER ═══════════════════════════════════════════════════
hdr(ax, 0.4, 11.8, 25.2,
    "STAGE 5  —  Neuro-Symbolic Rule Layer  (8 differentiable rules)", C["rule"])

arr(ax, 22.4, 12.18, 20.0, 11.83, col=C["fusion"])

# Rule activation mini bar chart
ax_r = fig.add_axes([0.024, 0.298, 0.19, 0.07])
acts   = [0.38, 0.11, 0.16, 0.05, 0.09, 0.14, 0.07, 0.00]
labels = ["R0: θ+pupil→HIGH","R1: mixed","R2: β+fixation",
          "R3: α→LOW","R4: other","R5: Fp↔Pz","R6: gaze","R7: none"]
cols_r = [C["merge"],"#BDC3C7","#BDC3C7",C["rule"],
          "#BDC3C7","#BDC3C7","#BDC3C7","#BDC3C7"]
ax_r.barh(range(8), acts, color=cols_r, height=0.65)
ax_r.set_yticks(range(8)); ax_r.set_yticklabels(labels, fontsize=8)
ax_r.set_xlabel("Activation (softmax)", fontsize=8.5)
ax_r.set_xlim(0, 0.5)
ax_r.set_title("Rule Activations — S01, Epoch 47", fontsize=9, fontweight="bold")
ax_r.axvline(0.38, color=C["merge"], lw=1.2, ls="--", alpha=0.6)
ax_r.annotate("dominant → HIGH\n(0.38)", xy=(0.38,7),
              xytext=(0.41,6.4), fontsize=7.5, color=C["merge"],
              arrowprops=dict(arrowstyle="-", color=C["merge"], lw=0.9))
ax_r.set_facecolor("#FAFAFA")
ax_r.spines["top"].set_visible(False); ax_r.spines["right"].set_visible(False)

rbox(ax, 5.5, 10.15, 13.5, 1.4, "#FDEDEC",
     "Weighted Vote:  Σ  activation_r  ×  rule_classifier_r ( fused )",
     "Rule 0 fires (0.38) → HIGH vote weight +0.82\n"
     "Rule 3 fires weakly (0.05) → LOW vote, barely counts",
     "logits  =  [ −0.89 ,  +1.43 ]",
     tf=11)

arr(ax, 12.25, 10.15, 12.25, 9.45, col=C["rule"])
tensor_tag(ax, 12.25, 9.32, "logits  [ −0.89,  +1.43 ]", C["rule"])

# ══ STAGE 6 — DECISION ═════════════════════════════════════════════════════
hdr(ax, 0.4, 8.9, 25.2,
    "STAGE 6  —  Temperature Scaling  +  Ensemble (5 models)  +  Decision", C["rule"])

arr(ax, 12.25, 9.32, 12.25, 8.93, col=C["rule"])

# Temperature box
rbox(ax, 5.0, 7.5, 6.5, 1.15, "#F9EBEA",
     "Temperature Scaling  (T = 0.91)",
     "[ −0.89, +1.43 ]  ÷  0.91  =  [ −0.98, +1.57 ]\n"
     "softmax  →  P(LOW)=0.09   P(HIGH)=0.91")

# Ensemble box
rbox(ax, 12.5, 7.5, 6.5, 1.15, "#F9EBEA",
     "Ensemble  (5 models, average)",
     "0.910  0.881  0.923  0.897  0.905\n"
     "avg  P(HIGH) = 0.903  >  threshold 0.48")

arr(ax, 8.25, 7.5, 9.8, 7.1, col=C["rule"])
arr(ax, 15.75, 7.5, 14.2, 7.1, col=C["rule"])

# Ensemble bar chart
ax_e = fig.add_axes([0.024, 0.19, 0.19, 0.07])
models = ["M0","M1","M2","M3","M4"]
probs  = [0.910, 0.881, 0.923, 0.897, 0.905]
ax_e.bar(models, probs, color=C["merge"], alpha=0.85, width=0.55)
ax_e.axhline(0.903, color=C["rule"],  lw=1.5, ls="--", label="avg 0.903")
ax_e.axhline(0.480, color="#E74C3C", lw=1.2, ls=":",  label="threshold 0.48")
ax_e.set_ylim(0.4, 1.0); ax_e.set_ylabel("P(HIGH)", fontsize=8.5)
ax_e.set_title("Ensemble probabilities", fontsize=9, fontweight="bold")
ax_e.legend(fontsize=7.5, loc="lower right")
ax_e.set_facecolor("#FAFAFA")
ax_e.spines["top"].set_visible(False); ax_e.spines["right"].set_visible(False)

# ── FINAL DECISION BOX ───────────────────────────────────────────────────────
rbox(ax, 8.5, 5.2, 7.0, 1.75, "#1E8449",
     "ŷ  =  HIGH_ENGAGEMENT  ✓",
     "0.903  >  threshold 0.48  →  CORRECT prediction",
     tf=15, sf=10, alpha=0.90, ec="#145A32")

arr(ax, 12.0, 7.5, 12.0, 6.95, col=C["decide"], lw=2.5)

# ══ LEGEND ══════════════════════════════════════════════════════════════════
legend = [
    (C["eeg"],    "EEG input"),
    (C["gat"],    "Graph / GAT branch"),
    (C["snn"],    "Spiking (SNN) branch"),
    (C["et"],     "Eye tracking branch"),
    (C["roi"],    "ROI operations"),
    (C["merge"],  "Merged EEG embedding"),
    (C["fusion"], "Fusion Transformer"),
    (C["rule"],   "Rule layer / Decision"),
]
lx0, ly0 = 19.5, 9.5
ax.text(22.0, 10.1, "Colour Legend", ha="center", fontsize=10,
        fontweight="bold", color=C["hdr"])
for i, (col, lbl) in enumerate(legend):
    cx = lx0 + (i % 2) * 2.55
    cy = ly0 - (i // 2) * 0.6
    ax.add_patch(plt.Rectangle((cx, cy+0.06), 0.32, 0.32,
                                color=col, zorder=5))
    ax.text(cx + 0.44, cy + 0.22, lbl, va="center",
            fontsize=9, color=C["hdr"])

# ── Save ─────────────────────────────────────────────────────────────────────
out = "src/model/visualization/pipeline_visual.png"
plt.savefig(out, dpi=160, bbox_inches="tight",
            facecolor=C["bg"], edgecolor="none")
print(f"Saved → {out}")
plt.close()
