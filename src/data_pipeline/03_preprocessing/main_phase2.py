import sys
import os
import argparse
from pathlib import Path

# ── Parse subject FIRST — before any path-sensitive code ─────────────────────
_parser = argparse.ArgumentParser(description="NEUMA Phase 2 — EEG/ET Preprocessing")
_parser.add_argument(
    "--subject",
    type=str,
    default=os.environ.get("NEUMA_SUBJECT", "S01"),
    help="Subject ID (e.g. S01)",
)
_parser.add_argument(
    "--skip-plots",
    action="store_true",
    help="Skip visualisation plots (useful for batch runs)",
)
_args      = _parser.parse_args()
SUBJECT    = _args.subject
SKIP_PLOTS = _args.skip_plots or (os.environ.get("NEUMA_SKIP_PLOTS", "0") == "1")

os.environ["NEUMA_SUBJECT"] = SUBJECT
print(f"[INFO] Processing subject: {SUBJECT}")

# ── Absolute paths (safe regardless of cwd) ───────────────────────────────────
_HERE      = Path(__file__).resolve().parent          # src/data_pipeline/03_preprocessing/
_ROOT      = _HERE.parent.parent.parent               # Neuma_Model/ (repo root -- DataSource/ lives here,
                                                       # two levels above src/data_pipeline/)

XDF_PATH   = str(_ROOT / "DataSource" / f"{SUBJECT}.xdf")
OUTPUT_DIR = str(_HERE / SUBJECT / "output" / "clean_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, str(_HERE))

# ── Universal UTF-8 / Unicode-safe I/O ────────────────────────────────────────
import sys as _sys, os as _os
_os.environ.setdefault("PYTHONUTF8", "1")
import io as _io
try:
    if isinstance(_sys.stdout, _io.TextIOWrapper):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(_sys.stderr, _io.TextIOWrapper):
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_REPL = {
    "→":"->", "←":"<-", "✔":"[OK]", "✘":"[X]",
    "─":"-",  "│":"|",  "═":"=",    "┌":"+",
    "┐":"+",  "└":"+",  "┘":"+",    "├":"+",
    "┤":"+",  "█":"#",  "░":".",     "►":">",
    "±":"+/-","∞":"inf",
}

def sanitize_text(t):
    t = str(t)
    for o, n in _REPL.items():
        t = t.replace(o, n)
    return t

def safe_print(*args, **kwargs):
    sep_c = kwargs.pop("sep", " ")
    text  = sanitize_text(sep_c.join(str(a) for a in args))
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        print(text.encode("ascii", "ignore").decode(), **kwargs)
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")

import numpy as np

# Backward-compat alias used by preprocessing helpers
SUBJECT_ID = SUBJECT

from loaders.load_xdf import load_xdf_file

from preprocessing.eeg_bad_channel_removal import remove_bad_channels
from preprocessing.eeg_rereference        import common_average_reference
from preprocessing.eeg_filters            import bandpass_filter, notch_filter
from preprocessing.eeg_ica               import apply_ica
from preprocessing.et_interpolation      import interpolate_et
from preprocessing.et_smoothing          import smooth_et
from preprocessing.synchronize_streams   import synchronize_et_to_eeg

from visualization.plot_filter_comparison import plot_filter_comparison
from visualization.plot_clean_eeg         import plot_clean_eeg
from visualization.plot_et_cleaning       import plot_gaze_cleaning


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================")
    print(f" PHASE 2 — EEG/ET PREPROCESSING ({SUBJECT})")
    print("==============================")

    # --------------------------------------------------------
    # LOAD STREAMS
    # --------------------------------------------------------

    streams    = load_xdf_file(XDF_PATH)
    eeg_stream = streams["EEG"]
    et_stream  = streams["ET"]

    eeg    = np.array(eeg_stream["time_series"])
    et     = np.array(et_stream["time_series"])
    eeg_ts = np.array(eeg_stream["time_stamps"])
    et_ts  = np.array(et_stream["time_stamps"])

    print("\n===== RAW DATA =====")
    print(f"EEG : {eeg.shape} | timestamps={eeg_ts.shape}")
    print(f"ET  : {et.shape}  | timestamps={et_ts.shape}")

    eeg_raw = eeg.copy()
    et_raw  = et.copy()

    # --------------------------------------------------------
    # EEG PREPROCESSING
    # --------------------------------------------------------

    eeg, kept_channels = remove_bad_channels(eeg, SUBJECT_ID)
    print(f"\nRemaining EEG channels: {len(kept_channels)}")

    eeg = common_average_reference(eeg)
    print("Common average reference applied")

    eeg_filtered = bandpass_filter(eeg)
    print("Bandpass filter applied (1-45 Hz)")

    eeg_filtered = notch_filter(eeg_filtered)
    print("Notch filter applied (50 Hz)")

    eeg_clean = apply_ica(eeg_filtered)
    print("ICA complete")

    # --------------------------------------------------------
    # ET PREPROCESSING
    # --------------------------------------------------------

    et_interp = interpolate_et(et)
    et_clean  = smooth_et(et_interp)
    print("ET smoothing complete")

    # ET resampled to EEG grid (visualisation / debug only)
    et_synced = synchronize_et_to_eeg(et_clean, et_ts, eeg_ts)

    # --------------------------------------------------------
    # FINAL SHAPES
    # --------------------------------------------------------

    print("\n===== FINAL OUTPUT =====")
    print(f"EEG Clean Shape : {eeg_clean.shape}")
    print(f"EEG TS Shape    : {eeg_ts.shape}")
    print(f"ET Clean Shape  : {et_clean.shape}")
    print(f"ET TS Shape     : {et_ts.shape}")
    print(f"ET Synced Shape : {et_synced.shape}")

    # --------------------------------------------------------
    # VISUALIZATIONS (optional)
    # --------------------------------------------------------

    if not SKIP_PLOTS:
        plot_filter_comparison(eeg_raw[:, :eeg_clean.shape[1]], eeg_clean)
        plot_clean_eeg(eeg_raw[:, :eeg_clean.shape[1]], eeg_clean, subject_id=SUBJECT_ID)
        plot_gaze_cleaning(et_raw, et_clean, subject=SUBJECT_ID)

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    np.save(os.path.join(OUTPUT_DIR, "eeg_clean.npy"),      eeg_clean)
    np.save(os.path.join(OUTPUT_DIR, "eeg_timestamps.npy"), eeg_ts)
    np.save(os.path.join(OUTPUT_DIR, "et_clean.npy"),       et_clean)
    np.save(os.path.join(OUTPUT_DIR, "et_timestamps.npy"),  et_ts)
    np.save(os.path.join(OUTPUT_DIR, "et_synced.npy"),      et_synced)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print("  eeg_clean.npy  eeg_timestamps.npy")
    print("  et_clean.npy   et_timestamps.npy   et_synced.npy")

    print("\n==============================")
    print(f" PHASE 2 COMPLETE — {SUBJECT}")
    print("==============================")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
