import os
from pathlib import Path

# ── Subject ID (set via env var for multi-subject runs) ──────
#   default: S01  |  override: $env:NEUMA_SUBJECT = "S02"
SUBJECT_ID = os.environ.get("NEUMA_SUBJECT", "S01")

# ── Phase 3 outputs (subject-scoped, matching 04_segmentation's actual
#    output layout: 04_segmentation/<SUBJECT_ID>/output/) ────
PHASE3_DIR = Path(f"../04_segmentation/{SUBJECT_ID}/output")

EEG_EPOCHS_PATH = PHASE3_DIR / "epochs/eeg_epochs.npy"
ET_EPOCHS_PATH  = PHASE3_DIR / "epochs/et_epochs.npy"
LABELS_PATH     = PHASE3_DIR / "epochs/labels.npy"

# ── Output (per-subject subdirectory) ────────────────────────
OUTPUT_FEATURES = Path("output") / SUBJECT_ID / "features"
OUTPUT_PLOTS    = Path("output") / SUBJECT_ID / "plots"
OUTPUT_LOGS     = Path("output") / SUBJECT_ID / "logs"

# ── Sampling rates ───────────────────────────────────────────
EEG_SR = 300
ET_SR  = 120

# ── EEG frequency bands ──────────────────────────────────────
EEG_BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45)
}
