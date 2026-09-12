import os
from pathlib import Path
from types import SimpleNamespace

# ── Sampling rates ────────────────────────────────────────────
EEG_SR = 300
ET_SR  = 120

# ── Epoch window ─────────────────────────────────────────────
PRE_TIME  = 0.0
POST_TIME = 5.0

# ── Minimum samples per epoch ────────────────────────────────
MIN_EEG_SAMPLES = int(EEG_SR * 2)
MIN_ET_SAMPLES  = int(ET_SR  * 2)

# ── Fixation detector ────────────────────────────────────────
FIXATION_VELOCITY_THRESHOLD   = 80
FIXATION_MIN_DURATION_SAMPLES = 12

# ── Root paths (absolute — safe regardless of cwd) ───────────
_PHASE3_DIR = Path(__file__).resolve().parent.parent   # …/src/data_pipeline/04_segmentation
# _PHASE3_DIR is now nested two levels deeper than before (src/data_pipeline/
# instead of directly under the repo root), so three .parent hops are
# needed here instead of one.
_ROOT       = _PHASE3_DIR.parent.parent.parent         # …/Neuma_Model
XDF_DIR     = _ROOT / "DataSource"


# ── Per-subject path factory ──────────────────────────────────
def get_paths(subject_id: str) -> SimpleNamespace:
    """Return all subject-specific paths for one LOSOCV subject."""
    p2 = _ROOT / "src" / "data_pipeline" / "03_preprocessing" / subject_id / "output" / "clean_data"
    p3 = _PHASE3_DIR / subject_id / "output"
    return SimpleNamespace(
        # Phase 2 inputs
        phase2_dir     = p2,
        eeg_clean      = p2 / "eeg_clean.npy",
        et_clean       = p2 / "et_clean.npy",
        eeg_timestamps = p2 / "eeg_timestamps.npy",
        et_timestamps  = p2 / "et_timestamps.npy",
        # Phase 3 outputs
        epochs_dir     = p3 / "epochs",
        plots_dir      = p3 / "plots",
        metadata_dir   = p3 / "metadata",
        logs_dir       = p3 / "logs",
        events_dir     = p3 / "events",
        markers_csv    = p3 / "events" / "markers.csv",
        roi_boxes      = Path("output/roi/roi_boxes.npy"),
    )


# ── Module-level names — resolved from NEUMA_SUBJECT env var ─────────────────
# IMPORTANT: set os.environ["NEUMA_SUBJECT"] BEFORE importing this module.
#            main_phase3.py does this via argparse before any project imports.
_SUBJECT_ID = os.environ.get("NEUMA_SUBJECT", "S01")
_p = get_paths(_SUBJECT_ID)

PHASE2_DIR          = _p.phase2_dir
EEG_CLEAN_PATH      = _p.eeg_clean
ET_CLEAN_PATH       = _p.et_clean
EEG_TS_PATH         = _p.eeg_timestamps
ET_TS_PATH          = _p.et_timestamps
OUTPUT_EPOCHS_DIR   = _p.epochs_dir
OUTPUT_PLOTS_DIR    = _p.plots_dir
OUTPUT_METADATA_DIR = _p.metadata_dir
OUTPUT_LOGS_DIR     = _p.logs_dir
MARKERS_CSV         = _p.markers_csv
ROI_BOXES_PATH      = _p.roi_boxes
