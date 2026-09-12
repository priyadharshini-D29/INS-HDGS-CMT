"""
==============================================================
NEUMA data pipeline orchestration - Automated Full-Subject Processing
==============================================================
For each subject S01-S34:
  1. Run 03_preprocessing (EEG + ET preprocessing)
  2. Run 04_segmentation (event segmentation + epoching)
  3. Run 05_feature_extraction (feature extraction)

Each phase is executed as a subprocess in its own working
directory with NEUMA_SUBJECT set as an environment variable.
Visualizations are suppressed (NEUMA_SKIP_PLOTS=1) for speed.

After all subjects, the feature-extraction stage's per-subject
feature files are aggregated into a single global dataset.

Run from: src/data_pipeline/orchestration/
  python run_all_subjects.py
  python run_all_subjects.py S01 S05      # specific range
  python run_all_subjects.py --resume     # skip already-done
==============================================================
"""

import os
import sys
import io
import subprocess
import traceback
import shutil
from pathlib import Path
from collections import Counter
from datetime import datetime

# ── Force UTF-8 stdout/stderr ──────────────────────────────────────────────────
os.environ.setdefault("PYTHONUTF8", "1")
try:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd

# ── PATHS ─────────────────────────────────────────────────────────────────────

# This file lives at src/data_pipeline/orchestration/, now nested two
# levels deeper than before (orchestration/ used to be a direct sibling
# of the phase folders under the repo root), so reaching the repo root
# needs four .parent hops instead of two.
BASE_DIR   = Path(__file__).resolve().parent.parent.parent.parent

PHASE2_DIR = BASE_DIR / "src" / "data_pipeline" / "03_preprocessing"
PHASE3_DIR = BASE_DIR / "src" / "data_pipeline" / "04_segmentation"
PHASE4_DIR = BASE_DIR / "src" / "data_pipeline" / "05_feature_extraction"
DATA_DIR   = BASE_DIR / "DataSource"

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "global_dataset"
LOG_DIR    = Path(__file__).resolve().parent / "output" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────

ALL_SUBJECTS = [f"S{i:02d}" for i in range(1, 35)]

args         = sys.argv[1:]
RESUME       = "--resume" in args
subject_args = [a for a in args if a.startswith("S") and len(a) == 3]
SUBJECTS     = subject_args if subject_args else ALL_SUBJECTS

_LABEL_MAP = {
    "FYLLADIO_1.tif": "ImagePage_1",
    "FYLLADIO_2.tif": "ImagePage_2",
    "FYLLADIO_3.tif": "ImagePage_3",
    "FYLLADIO_4.tif": "ImagePage_4",
    "FYLLADIO_5.tif": "ImagePage_5",
    "FYLLADIO_6.tif": "ImagePage_6",
}
_INVALID_LABELS = {"fixation_cross", "EOE"}


def _normalize_label(label: str) -> str:
    for raw, clean in _LABEL_MAP.items():
        if raw in label:
            return clean
    return label


# ── SAFE OUTPUT UTILITIES ─────────────────────────────────────────────────────

def safe_print(*args, **kwargs) -> None:
    """Unicode-safe drop-in for print(); supports end=, flush=, sep=, file=."""
    text = " ".join(str(a) for a in args)

    _replacements = {
        "─": "-", "│": "|", "┌": "+", "┐": "+",
        "└": "+", "┘": "+", "├": "+", "┤": "+",
        "═": "=", "║": "|", "╔": "+", "╗": "+",
        "╚": "+", "╝": "+", "╠": "+", "╣": "+",
        "╦": "+", "╩": "+", "╬": "+",
        "█": "#", "░": ".",
        "✔": "[OK]", "✘": "[X]",
        "►": ">", "→": "->", "←": "<-",
        "±": "+/-", "∞": "inf",
    }
    for old, new in _replacements.items():
        text = text.replace(old, new)

    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        safe_text = text.encode("ascii", errors="ignore").decode()
        print(safe_text, **kwargs)


def sep(title: str = "") -> None:
    """Print an ASCII-safe separator line."""
    if title:
        pad = max(0, 52 - len(title) - 4)
        line = "=== " + title + " " + "=" * pad
    else:
        line = "=" * 52
    safe_print(line)


def sep_minor(title: str = "") -> None:
    """Print a minor ASCII-safe separator."""
    if title:
        pad = max(0, 52 - len(title) - 4)
        line = "--- " + title + " " + "-" * pad
    else:
        line = "-" * 52
    safe_print(line)


def safe_log(log_file, text: str) -> None:
    """Write text to log file, replacing unencodable characters."""
    try:
        log_file.write(text + "\n")
    except UnicodeEncodeError:
        safe = text.encode("utf-8", errors="replace").decode("utf-8")
        log_file.write(safe + "\n")
    log_file.flush()


# ── SESSION LOG ───────────────────────────────────────────────────────────────

_SESSION_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_SESSION_LOG_PATH = LOG_DIR / f"session_{_SESSION_STAMP}.log"
_SESSION_LOG = open(_SESSION_LOG_PATH, "w", encoding="utf-8")

def session_log(text: str) -> None:
    """Write a line to the session-wide log."""
    try:
        _SESSION_LOG.write(text + "\n")
        _SESSION_LOG.flush()
    except Exception:
        pass


# ── SUBPROCESS RUNNER ─────────────────────────────────────────────────────────

def run_phase(phase_dir: Path, script: str, subject: str, log_file) -> bool:
    """Run a phase script in its directory. Returns True on success."""
    env                    = os.environ.copy()
    env["NEUMA_SUBJECT"]   = subject
    env["NEUMA_SKIP_PLOTS"] = "1"

    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd=str(phase_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        msg = f"[TIMEOUT] {script} exceeded 600s for subject {subject}"
        safe_log(log_file, msg)
        session_log(f"  {msg}")
        return False
    except Exception as exc:
        msg = f"[EXCEPTION] running {script}: {exc}\n{traceback.format_exc()}"
        safe_log(log_file, msg)
        session_log(f"  [EXCEPTION] {script}: {exc}")
        return False

    safe_log(log_file, "\n" + "-" * 40)
    safe_log(log_file, script)
    safe_log(log_file, "-" * 40)
    safe_log(log_file, result.stdout or "")
    if result.stderr:
        safe_log(log_file, "\n[STDERR]\n" + result.stderr)

    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip().splitlines()
        for line in stderr_tail[-30:]:
            safe_print(f"    {line}")
        session_log(f"  [FAIL] {script} exit code {result.returncode}")
        return False

    session_log(f"  [OK] {script}")
    return True


# ── TRACKING ──────────────────────────────────────────────────────────────────

SUCCESS_SUBJECTS = []
FAILED_SUBJECTS  = []

all_eeg      = []
all_et       = []
all_fusion   = []
all_labels   = []
all_subjects = []

summary_rows = []
skipped      = []
processed    = 0

# ── HEADER ────────────────────────────────────────────────────────────────────

sep()
safe_print("  NEUMA - AUTOMATED FULL-SUBJECT PIPELINE")
sep()
safe_print(f"  Subjects : {SUBJECTS[0]} -> {SUBJECTS[-1]}  ({len(SUBJECTS)} total)")
safe_print(f"  Resume   : {RESUME}")
safe_print(f"  Logs     : {LOG_DIR}")
safe_print(f"  Session  : {_SESSION_LOG_PATH.name}")
session_log("=" * 52)
session_log("  NEUMA PIPELINE SESSION")
session_log(f"  Started  : {datetime.now().isoformat()}")
session_log(f"  Subjects : {SUBJECTS}")
session_log("=" * 52)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

for subject in SUBJECTS:

    sep_minor()
    safe_print(f"  SUBJECT : {subject}")
    sep_minor()
    session_log(f"\n--- {subject} ---")

    # ── Pre-flight ────────────────────────────────────────
    xdf_path = DATA_DIR / f"{subject}.xdf"
    if not xdf_path.exists():
        msg = f"  [SKIP] {subject}.xdf not found in {DATA_DIR}"
        safe_print(msg)
        session_log(msg)
        skipped.append({"subject": subject, "reason": "xdf_missing"})
        FAILED_SUBJECTS.append(subject)
        continue

    feat_dir = PHASE4_DIR / "output" / subject / "features"

    if RESUME and (feat_dir / "eeg_features.npy").exists():
        safe_print("  [RESUME] features already exist - skipping processing")
        session_log(f"  [RESUME] {subject} - skipping phases")
    else:
        subject_ok = True

        # ── Phase 2 ──────────────────────────────────────
        log_path = LOG_DIR / f"{subject}_phase2.log"
        safe_print("  [P2] EEG + ET preprocessing ...", end="", flush=True)
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                ok = run_phase(PHASE2_DIR, "main_phase2.py", subject, lf)
        except Exception as exc:
            safe_print(f" ERROR ({exc})")
            session_log(f"  [ERROR] opening log {log_path}: {exc}")
            ok = False

        if not ok:
            safe_print(f" FAILED  (see {log_path.name})")
            skipped.append({"subject": subject, "reason": "phase2_failed"})
            summary_rows.append({"subject": subject, "epochs": 0, "status": "phase2_failed"})
            FAILED_SUBJECTS.append(subject)
            continue
        safe_print(" OK")

        # ── Phase 3 ──────────────────────────────────────
        stale_markers = PHASE3_DIR / "output" / "events" / "markers.csv"
        if stale_markers.exists():
            stale_markers.unlink()

        log_path = LOG_DIR / f"{subject}_phase3.log"
        safe_print("  [P3] Event segmentation + epoching ...", end="", flush=True)
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                ok = run_phase(PHASE3_DIR, "main_phase3.py", subject, lf)
        except Exception as exc:
            safe_print(f" ERROR ({exc})")
            session_log(f"  [ERROR] opening log {log_path}: {exc}")
            ok = False

        if not ok:
            safe_print(f" FAILED  (see {log_path.name})")
            skipped.append({"subject": subject, "reason": "phase3_failed"})
            summary_rows.append({"subject": subject, "epochs": 0, "status": "phase3_failed"})
            FAILED_SUBJECTS.append(subject)
            continue
        safe_print(" OK")

        # ── Phase 4 ──────────────────────────────────────
        log_path = LOG_DIR / f"{subject}_phase4.log"
        safe_print("  [P4] Feature extraction ...", end="", flush=True)
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                ok = run_phase(PHASE4_DIR, "main_phase4.py", subject, lf)
        except Exception as exc:
            safe_print(f" ERROR ({exc})")
            session_log(f"  [ERROR] opening log {log_path}: {exc}")
            ok = False

        if not ok:
            safe_print(f" FAILED  (see {log_path.name})")
            skipped.append({"subject": subject, "reason": "phase4_failed"})
            summary_rows.append({"subject": subject, "epochs": 0, "status": "phase4_failed"})
            FAILED_SUBJECTS.append(subject)
            continue
        safe_print(" OK")

    # ── Read Phase 4 features ─────────────────────────────
    try:
        eeg    = np.load(feat_dir / "eeg_features.npy",    allow_pickle=True).astype(np.float32)
        et     = np.load(feat_dir / "et_features.npy",     allow_pickle=True).astype(np.float32)
        fusion = np.load(feat_dir / "fusion_features.npy", allow_pickle=True).astype(np.float32)
        labels = np.load(feat_dir / "labels.npy",          allow_pickle=True)
    except Exception as exc:
        msg = f"  [ERROR] reading features: {exc}"
        safe_print(msg)
        session_log(msg)
        session_log(traceback.format_exc())
        skipped.append({"subject": subject, "reason": f"read_error: {exc}"})
        summary_rows.append({"subject": subject, "epochs": 0, "status": f"read_error: {exc}"})
        FAILED_SUBJECTS.append(subject)
        continue

    # ── Label normalisation + invalid-label filtering ─────
    labels_norm = np.array([_normalize_label(str(l)) for l in labels])
    mask        = np.array([l not in _INVALID_LABELS for l in labels_norm])

    eeg, et, fusion, labels_norm = (
        eeg[mask], et[mask], fusion[mask], labels_norm[mask]
    )
    n = len(labels_norm)

    if n == 0:
        msg = "  [SKIP] 0 valid epochs after label filtering"
        safe_print(msg)
        session_log(msg)
        skipped.append({"subject": subject, "reason": "no_valid_epochs"})
        summary_rows.append({"subject": subject, "epochs": 0, "status": "no_valid_epochs"})
        FAILED_SUBJECTS.append(subject)
        continue

    all_eeg.append(eeg)
    all_et.append(et)
    all_fusion.append(fusion)
    all_labels.append(labels_norm)
    all_subjects.append(np.array([subject] * n))

    processed += 1
    SUCCESS_SUBJECTS.append(subject)
    info = f"  Epochs={n}  EEG={eeg.shape[1]}  ET={et.shape[1]}"
    safe_print(info)
    session_log(info)

    summary_rows.append({
        "subject":      subject,
        "epochs":       n,
        "eeg_features": eeg.shape[1],
        "et_features":  et.shape[1],
        "status":       "ok",
        "label_dist":   str(dict(Counter(labels_norm))),
    })


# ── CONCATENATE ───────────────────────────────────────────────────────────────

sep()
safe_print("  CONCATENATING GLOBAL DATASET")
sep()
session_log("\n" + "=" * 52)
session_log("  CONCATENATING GLOBAL DATASET")

if not all_eeg:
    safe_print("\n[STOP] No subjects processed successfully.")
    session_log("[STOP] No subjects processed successfully.")
    _SESSION_LOG.close()
    sys.exit(1)

all_eeg      = np.concatenate(all_eeg,      axis=0)
all_et       = np.concatenate(all_et,       axis=0)
all_fusion   = np.concatenate(all_fusion,   axis=0)
all_labels   = np.concatenate(all_labels,   axis=0)
all_subjects = np.concatenate(all_subjects, axis=0)

safe_print(f"  EEG     : {all_eeg.shape}")
safe_print(f"  ET      : {all_et.shape}")
safe_print(f"  Fusion  : {all_fusion.shape}")
safe_print(f"  Labels  : {all_labels.shape}")
safe_print(f"  Subjects: {all_subjects.shape}")


# ── SAVE ──────────────────────────────────────────────────────────────────────

safe_print("\n===== SAVING GLOBAL DATASET =====")
session_log("Saving global dataset...")

saves = {
    "all_eeg_features.npy":    all_eeg,
    "all_et_features.npy":     all_et,
    "all_fusion_features.npy": all_fusion,
    "all_labels.npy":          all_labels,
    "all_subjects.npy":        all_subjects,
}
for fname, arr in saves.items():
    np.save(OUTPUT_DIR / fname, arr)
    safe_print(f"  {fname}")

pd.DataFrame(all_eeg).to_csv(
    OUTPUT_DIR / "all_eeg_features.csv",    index=False)
pd.DataFrame(all_et).to_csv(
    OUTPUT_DIR / "all_et_features.csv",     index=False)
pd.DataFrame(all_fusion).to_csv(
    OUTPUT_DIR / "all_fusion_features.csv", index=False)
pd.DataFrame({"subject": all_subjects, "label": all_labels}).to_csv(
    OUTPUT_DIR / "metadata.csv", index=False)

pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "summary.csv",  index=False)
pd.DataFrame(skipped).to_csv(     OUTPUT_DIR / "skipped.csv",  index=False)

safe_print("  metadata.csv  |  summary.csv  |  skipped.csv")


# ── FINAL REPORT ──────────────────────────────────────────────────────────────

sep()
safe_print("  FINAL DATASET REPORT")
sep()

safe_print(f"  Processed subjects : {processed} / {len(SUBJECTS)}")
safe_print(f"  Succeeded          : {len(SUCCESS_SUBJECTS)}  {SUCCESS_SUBJECTS}")
safe_print(f"  Failed/Skipped     : {len(FAILED_SUBJECTS)}   {FAILED_SUBJECTS}")
safe_print(f"  Total epochs       : {len(all_labels)}")
safe_print(f"  EEG features       : {all_eeg.shape[1]}")
safe_print(f"  ET  features       : {all_et.shape[1]}")
safe_print(f"  Fusion features    : {all_fusion.shape[1]}")

safe_print("\n  Class distribution:")
for label, count in Counter(all_labels).most_common():
    safe_print(f"    {label:<20s}: {count}")

safe_print(f"\n  Saved to : {OUTPUT_DIR}")
safe_print(f"  Logs     : {LOG_DIR}")

session_log("\n" + "=" * 52)
session_log("  FINAL REPORT")
session_log(f"  Processed : {processed} / {len(SUBJECTS)}")
session_log(f"  Succeeded : {SUCCESS_SUBJECTS}")
session_log(f"  Failed    : {FAILED_SUBJECTS}")
session_log(f"  Completed : {datetime.now().isoformat()}")
session_log("=" * 52)

_SESSION_LOG.close()

sep()
safe_print("  PIPELINE COMPLETE")
sep()
