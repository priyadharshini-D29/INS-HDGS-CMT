"""
==============================================================
NEUMA PHASE 6 — MULTI-SUBJECT MULTIMODAL DATASET AGGREGATION
==============================================================
Iterates S01 → S34, loads Phase 4 feature outputs for each
subject, validates shapes, normalises labels, concatenates into
one global dataset, then generates summary statistics and plots.
==============================================================
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Universal UTF-8 / Unicode-safe I/O ────────────────────────────────────────
import sys as _sys, os as _os, io as _io
_os.environ.setdefault("PYTHONUTF8", "1")
try:
    if isinstance(_sys.stdout, _io.TextIOWrapper):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(_sys.stderr, _io.TextIOWrapper):
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_REPL = {"→":"->","←":"<-","✔":"[OK]","✘":"[X]","─":"-","│":"|","═":"=",
         "┌":"+","┐":"+","└":"+","┘":"+","├":"+","┤":"+","█":"#","░":".","►":">","±":"+/-","∞":"inf"}

def sanitize_text(t):
    t = str(t)
    for o, n in _REPL.items(): t = t.replace(o, n)
    return t

def safe_print(*args, **kwargs):
    sep_c = kwargs.pop("sep", " ")
    text  = sanitize_text(sep_c.join(str(a) for a in args))
    try:    print(text, **kwargs)
    except UnicodeEncodeError: print(text.encode("ascii","ignore").decode(), **kwargs)
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from utils.loader    import load_subject
from utils.validator import validate_subject, infer_expected_dims
from utils.plotting  import (
    plot_class_distribution,
    plot_subject_distribution,
    plot_correlation_heatmap,
    plot_modality_stats,
)

# ============================================================
# CONFIG
# ============================================================

# Was a hardcoded absolute Windows path (E:\Nuema_model\Neuma_Model) --
# already broken on any machine other than the original dev box. Fixed
# opportunistically to a portable, file-relative path while touching this
# file for the src/data_pipeline/ rename. This file lives at
# src/data_pipeline/06_dataset_aggregation/, four .parent hops from the
# repo root.
ROOT_DIR      = str(Path(__file__).resolve().parent.parent.parent.parent)
SUBJECTS      = [f"S{i:02d}" for i in range(1, 35)]
PHASE4_FOLDER = "src/data_pipeline/05_feature_extraction"

OUTPUT_DIR  = Path(__file__).resolve().parent / "output"
GLOBAL_DIR  = OUTPUT_DIR / "global"
PLOT_DIR    = OUTPUT_DIR / "plots"

for d in [GLOBAL_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 52)
    print("  PHASE 6 → MULTI-SUBJECT DATA AGGREGATION")
    print("=" * 52)

    # ── Pass 1: load all subjects ────────────────────────────
    raw_data     = []
    skipped      = []

    for subject in SUBJECTS:
        print(f"\n===== PROCESSING {subject} =====")
        data = load_subject(subject, ROOT_DIR, PHASE4_FOLDER)
        if data is None:
            skipped.append(subject)
            continue
        raw_data.append((subject, data))
        print(f"  EEG    : {data['eeg'].shape}")
        print(f"  ET     : {data['et'].shape}")
        print(f"  Fusion : {data['fusion'].shape}")
        print(f"  Labels : {data['n_samples']}  {list(np.unique(data['labels']))}")

    if not raw_data:
        print("\n[STOP] No subjects loaded — check ROOT_DIR and path structure.")
        print(f"Expected: {ROOT_DIR}/<SUBJECT>/{PHASE4_FOLDER}/output/features/")
        return

    # ── Infer expected feature dimensions from first valid subject ──
    just_data = [d for _, d in raw_data]
    exp_eeg, exp_et = infer_expected_dims(just_data)
    print(f"\n  Expected EEG features : {exp_eeg}")
    print(f"  Expected ET  features : {exp_et}")

    # ── Pass 2: validate and filter ──────────────────────────
    valid_data = []
    for subject, data in raw_data:
        if validate_subject(data, subject, exp_eeg, exp_et):
            valid_data.append(data)
        else:
            skipped.append(subject)

    print(f"\n  Loaded  : {len(valid_data)} subjects")
    print(f"  Skipped : {len(skipped)}  {skipped}")

    if not valid_data:
        print("\n[STOP] No subjects passed validation.")
        return

    # ── Concatenate ──────────────────────────────────────────
    print("\n===== CONCATENATING DATA =====")

    all_eeg      = np.concatenate([d["eeg"]         for d in valid_data], axis=0)
    all_et       = np.concatenate([d["et"]          for d in valid_data], axis=0)
    all_fusion   = np.concatenate([d["fusion"]      for d in valid_data], axis=0)
    all_labels   = np.concatenate([d["labels"]      for d in valid_data], axis=0)
    all_subjects = np.concatenate([d["subject_ids"] for d in valid_data], axis=0)

    print(f"  EEG     : {all_eeg.shape}")
    print(f"  ET      : {all_et.shape}")
    print(f"  Fusion  : {all_fusion.shape}")
    print(f"  Labels  : {all_labels.shape}   unique={list(np.unique(all_labels))}")
    print(f"  Subjects: {all_subjects.shape}  unique={list(np.unique(all_subjects))}")

    # ── NaN / Inf audit on global arrays ────────────────────
    print("\n===== QUALITY AUDIT =====")
    for name, arr in [("EEG", all_eeg), ("ET", all_et), ("Fusion", all_fusion)]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        print(f"  {name:8s}  NaN={n_nan}  Inf={n_inf}  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}")

    # ── Save .npy ────────────────────────────────────────────
    print("\n===== SAVING GLOBAL DATASETS =====")

    saves = {
        "all_eeg_features.npy":    all_eeg,
        "all_et_features.npy":     all_et,
        "all_fusion_features.npy": all_fusion,
        "all_labels.npy":          all_labels,
        "all_subjects.npy":        all_subjects,
    }
    for fname, arr in saves.items():
        np.save(GLOBAL_DIR / fname, arr)
        print(f"  {fname}")

    # ── Save .csv ────────────────────────────────────────────
    pd.DataFrame(all_eeg).to_csv(
        GLOBAL_DIR / "all_eeg_features.csv",    index=False)
    pd.DataFrame(all_et).to_csv(
        GLOBAL_DIR / "all_et_features.csv",     index=False)
    pd.DataFrame(all_fusion).to_csv(
        GLOBAL_DIR / "all_fusion_features.csv", index=False)

    pd.DataFrame({
        "subject": all_subjects,
        "label":   all_labels,
    }).to_csv(GLOBAL_DIR / "metadata.csv", index=False)

    # ── Dataset statistics ───────────────────────────────────
    print("\n===== DATASET STATISTICS =====")

    label_series   = pd.Series(all_labels)
    subject_series = pd.Series(all_subjects)

    stats = {
        "total_samples":      len(all_labels),
        "valid_subjects":     int(subject_series.nunique()),
        "n_classes":          int(label_series.nunique()),
        "eeg_features":       all_eeg.shape[1],
        "et_features":        all_et.shape[1],
        "fusion_features":    all_fusion.shape[1],
        "mean_samples_per_subject": float(
            subject_series.value_counts().mean()),
        "std_samples_per_subject":  float(
            subject_series.value_counts().std()),
    }

    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(GLOBAL_DIR / "dataset_stats.csv", index=False)

    for k, v in stats.items():
        print(f"  {k:<30s} : {v}")

    print("\nClass distribution:")
    for lbl, cnt in label_series.value_counts().sort_index().items():
        print(f"  {lbl:<20s} : {cnt}")

    # ── Plots ────────────────────────────────────────────────
    print("\n===== GENERATING PLOTS =====")

    plot_class_distribution(
        all_labels, all_subjects,
        save_path=PLOT_DIR / "class_distribution.png",
    )

    plot_subject_distribution(
        all_subjects,
        save_path=PLOT_DIR / "subject_distribution.png",
    )

    plot_correlation_heatmap(
        all_fusion,
        save_path=PLOT_DIR / "feature_correlation_heatmap.png",
        n_features=50,
    )

    plot_modality_stats(
        all_eeg, all_et, all_fusion, all_labels,
        save_path=PLOT_DIR / "modality_stats_by_class.png",
    )

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  PHASE 6 COMPLETE")
    print("=" * 52)
    print(f"  Valid subjects : {stats['valid_subjects']}")
    print(f"  Total samples  : {stats['total_samples']}")
    print(f"  Global arrays  : {GLOBAL_DIR}")
    print(f"  Plots          : {PLOT_DIR}")


if __name__ == "__main__":
    main()
