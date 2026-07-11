"""
Figure 1 (improvised, real data) — NeuMa: stimulus, synchronized EEG/ET, and the
engagement labelling used here. Emulates the same-dataset Kalaganis et al. (2025) Fig 1
but for the engagement task: real brochure pages + real EEG + real eye-tracking traces,
with a HIGH/LOW engagement ribbon.

Data:
  brochure : DataSource/Dependencies/Brochure_Pages/ImagePage_{1,6}.tif
  EEG/ET   : NEUMA_PHASE3/S01/output/epochs/{eeg,et}_epochs_engagement.npy
  labels   : NEUMA_PHASE3/S01/output/engagement/engagement_labels.npy
Renders figures/fig1_dataset.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRO  = ROOT / "DataSource/Dependencies/Brochure_Pages"
EP   = ROOT / "NEUMA_PHASE3/S01/output/epochs"
LAB  = ROOT / "NEUMA_PHASE3/S01/output/engagement/engagement_labels.npy"

C_EEG = "#2f4b8f"; C_HIGH = "#2EA37A"; C_LOW = "#B9BFc9"
C_GX = "#E8A33D"; C_GY = "#C0504D"; C_PUP = "#7C5CBF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

# ── load real signals ─────────────────────────────────────────────────────────
eeg = np.load(EP / "eeg_epochs_engagement.npy", allow_pickle=True).astype(np.float32)  # (E,1500,20)
et  = np.load(EP / "et_epochs_engagement.npy",  allow_pickle=True).astype(np.float32)  # (E,600,6)
lab = np.asarray(np.load(LAB, allow_pickle=True)).ravel().astype(int)
NEP = 4                                   # show first 4 consecutive epochs (=20 s)
eeg_c = np.concatenate(eeg[:NEP], axis=0)          # (6000,20)
et_c  = np.concatenate(et[:NEP], axis=0)           # (2400,6)
labs  = lab[:NEP]
t_eeg = np.arange(eeg_c.shape[0]) / 300.0
t_et  = np.arange(et_c.shape[0]) / 120.0
T     = NEP * 5.0

CH = ["Fp1","Fp2","F3","F4","C3","C4","O1","O2"]   # 8 representative electrodes
ch_idx = [0,1,3,4,8,9,17,18]

fig = plt.figure(figsize=(12, 9.2))
gs = fig.add_gridspec(4, 1, height_ratios=[3.0, 0.42, 2.6, 1.7], hspace=0.32)

fig.suptitle("NeuMa: synchronized multimodal acquisition and engagement labelling",
             fontsize=14, fontweight="bold", y=0.98)
fig.text(0.5, 0.945,
         "42 participants · EEG (DSI-24, 19 scalp channels, 300 Hz) + eye tracking "
         "(Tobii Pro Fusion, 120 Hz) · digital-brochure browsing · subject S01 shown",
         ha="center", fontsize=9, color="#555")

# ── Row 1: real brochure pages ────────────────────────────────────────────────
gs_b = gs[0].subgridspec(1, 2, wspace=0.06)
for j, (pg, name) in enumerate([(1, "Brochure page 1 (dairy)"), (6, "Brochure page 6")]):
    axb = fig.add_subplot(gs_b[j])
    img = Image.open(BRO / f"ImagePage_{pg}.tif").convert("RGB")
    img.thumbnail((900, 900))
    axb.imshow(np.asarray(img)); axb.set_xticks([]); axb.set_yticks([])
    for s in axb.spines.values(): s.set_edgecolor("#333"); s.set_linewidth(1.2)
    axb.set_title(name, fontsize=9.5, fontweight="bold")
fig.text(0.5, 0.605, "6 pages × 24 products  →  gaze-defined product epochs",
         ha="center", fontsize=8.5, style="italic", color="#444")

# ── Row 2: engagement ribbon (real per-epoch labels) ──────────────────────────
axr = fig.add_subplot(gs[1])
for i, lb in enumerate(labs):
    axr.add_patch(FancyBboxPatch((i*5+0.06, 0.12), 5-0.12, 0.76,
                  boxstyle="round,pad=0.0,rounding_size=0.06",
                  fc=C_HIGH if lb == 1 else C_LOW, ec="white", lw=1.5))
    axr.text(i*5+2.5, 0.5, "HIGH" if lb == 1 else "LOW", ha="center", va="center",
             fontsize=9, fontweight="bold", color="white")
axr.set_xlim(0, T); axr.set_ylim(0, 1); axr.axis("off")
axr.text(-0.2, 0.5, "Engagement\nlabel", ha="right", va="center", fontsize=8.5,
         fontweight="bold")

# ── Row 3: real EEG (8 representative channels, stacked) ───────────────────────
axe = fig.add_subplot(gs[2])
off = 130
for k, ci in enumerate(ch_idx):
    sig = eeg_c[:, ci]
    axe.plot(t_eeg, sig + k*off, color=C_EEG, lw=0.5)
axe.set_yticks([k*off for k in range(len(CH))]); axe.set_yticklabels(CH, fontsize=8)
axe.set_xlim(0, T); axe.set_ylim(-off, len(CH)*off)
for b in range(1, NEP):
    axe.axvline(b*5, color="#aaa", ls="--", lw=0.8)
axe.set_ylabel("EEG  (µV, 300 Hz)", fontsize=9, fontweight="bold")
axe.set_xticklabels([]); axe.spines[["top","right"]].set_visible(False)
axe.text(0.01, 0.97, "19 channels (10–20) · A1/A2 reference",
         transform=axe.transAxes, fontsize=7.5, color="#555", va="top")

# ── Row 4: real eye tracking (gaze x, gaze y, pupil) ──────────────────────────
axt = fig.add_subplot(gs[3])
def nz(x): return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
axt.plot(t_et, nz(et_c[:, 0]) + 2.2, color=C_GX, lw=0.7, label="gaze$_x$")
axt.plot(t_et, nz(et_c[:, 1]) + 1.1, color=C_GY, lw=0.7, label="gaze$_y$")
axt.plot(t_et, nz(et_c[:, 2]) + 0.0, color=C_PUP, lw=0.7, label="pupil")
axt.set_yticks([2.6, 1.5, 0.4]); axt.set_yticklabels(["gaze$_x$","gaze$_y$","pupil"], fontsize=8)
axt.set_xlim(0, T); axt.set_ylim(-0.2, 3.4)
for b in range(1, NEP):
    axt.axvline(b*5, color="#aaa", ls="--", lw=0.8)
axt.set_xlabel("Time (s)  —  successive 5 s product epochs", fontsize=9)
axt.set_ylabel("Eye tracking  (120 Hz)", fontsize=9, fontweight="bold")
axt.spines[["top","right"]].set_visible(False)

for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"NEUMA_PHASE8/figures/fig1_dataset.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig1_dataset.pdf and .png  (real brochure + EEG + ET, S01)")
