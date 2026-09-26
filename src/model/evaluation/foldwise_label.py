"""
Fold-wise engagement label for LOSOCV (opt-in, NEUMA_LABEL_FOLDWISE=1).

The production label (engagement_phase3d.py) min-max scales every feature over
the epochs of ALL subjects and thresholds the weighted score at the pooled
median, so each held-out subject contributes to the rule that labels it.  With
NEUMA_LABEL_FOLDWISE=1 every fold re-derives the label with the scaler and the
median fitted on that fold's TRAINING subjects only (not the validation
subject, not the test subject) and applies it to the training, validation and
test epochs of the fold.  The rule itself lives in
scripts/analysis/label_fold_sensitivity.py (foldwise_labels); this module only
aligns its per-row output with the epochs of a NeumaGraphDataset.

Epoch identity.  The feature table (multimodal_features.csv) is written in
pipeline order: the k-th row of subject S is S's k-th epoch in
eeg_epochs_phase3d.npy / engagement_labels.npy, which is exactly the order the
dataset loads.  NeumaGraphDataset records, per epoch, the subject directory
(_subj_dirs) and the position within the subject (_subj_local_idx); augmented
copies (local index -1) are appended after all originals in the same order.
The key is therefore (subject id, position within subject); the number of
epochs per subject must agree between table and dataset, and the table's
stored label column must reproduce the dataset's stored labels, otherwise a
RuntimeError is raised.

Environment
-----------
NEUMA_LABEL_FOLDWISE   1 -> enabled (default: off, labels untouched)
NEUMA_LABEL_TABLE      optional path of the feature table (default: the
                       pipeline output, see label_fold_sensitivity.DEFAULT_FEATURES)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]            # <repo>
_SCRIPTS = _ROOT / "scripts" / "analysis"

_HIGH = "HIGH_ENGAGEMENT"
_TABLE: Optional[pd.DataFrame] = None


def enabled() -> bool:
    """True when NEUMA_LABEL_FOLDWISE requests the fold-wise label."""
    return os.environ.get("NEUMA_LABEL_FOLDWISE", "").strip().lower() in ("1", "true", "yes", "on")


def _rule_module():
    """scripts/analysis/label_fold_sensitivity.py (single source of the rule)."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import label_fold_sensitivity as lfs
    return lfs


def load_label_table() -> pd.DataFrame:
    """The per-epoch label feature table (cached per process)."""
    global _TABLE
    if _TABLE is None:
        src = os.environ.get("NEUMA_LABEL_SOURCE", "phase3d").strip().lower()
        if src != "phase3d":
            raise RuntimeError(
                f"NEUMA_LABEL_FOLDWISE=1 re-derives the phase-3D rule label, but "
                f"NEUMA_LABEL_SOURCE={src!r} has no feature table")
        path = os.environ.get("NEUMA_LABEL_TABLE", "").strip() or None
        _TABLE = _rule_module().load_label_feature_table(path)
    return _TABLE


def _by_subject(values: np.ndarray, table: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Split a per-row vector into per-subject arrays in pipeline (row) order."""
    subj = table["subject_id"].astype(str).to_numpy()
    return {s: np.asarray(values)[subj == s] for s in dict.fromkeys(subj)}


def labels_by_subject(table: pd.DataFrame, train_subjects: Iterable[str]) -> Dict[str, np.ndarray]:
    """Fold-wise label of every subject's epochs, rule fitted on train_subjects only."""
    y = _rule_module().foldwise_labels(table, list(train_subjects)).to_numpy()
    return _by_subject(y, table)


def stored_labels_by_subject(table: pd.DataFrame) -> Dict[str, np.ndarray]:
    """The label column written by the pipeline (what engagement_labels.npy holds)."""
    if "engagement_label" not in table.columns:
        raise RuntimeError("label feature table has no engagement_label column; cannot verify alignment")
    y = (table["engagement_label"].astype(str) == _HIGH).astype(int).to_numpy()
    return _by_subject(y, table)


def aligned_vector(ds, by_subject: Dict[str, np.ndarray]) -> np.ndarray:
    """Map per-subject label arrays onto the epoch order of a NeumaGraphDataset.

    Key: (subject id from the epoch's subject directory, position within the
    subject).  Augmented copies (local index -1) receive the label of the
    original they duplicate.  Raises RuntimeError on any count mismatch.
    """
    from data.dataset import _normalize_subject_id

    dirs = getattr(ds, "_subj_dirs", None)
    loc = getattr(ds, "_subj_local_idx", None)
    if dirs is None or loc is None:
        raise RuntimeError("dataset carries no epoch identity (_subj_dirs / _subj_local_idx)")
    loc = np.asarray(loc)
    n = len(ds.labels)
    n_orig = int((loc >= 0).sum())
    if not np.all(loc[:n_orig] >= 0) or not np.all(loc[n_orig:] < 0) or (n - n_orig) not in (0, n_orig):
        raise RuntimeError(
            f"unexpected epoch layout: {n} epochs, {n_orig} originals; augmented copies "
            f"must follow the originals in the same order")
    new = np.empty(n, dtype=np.int64)
    seen: Dict[str, int] = {}
    for i in range(n_orig):
        d = dirs[i]
        if d is None:
            raise RuntimeError(f"epoch {i}: no subject directory recorded")
        sid = _normalize_subject_id(Path(d).name)
        if sid not in by_subject:
            raise RuntimeError(f"{sid}: no rows in the label feature table")
        arr = by_subject[sid]
        if loc[i] >= len(arr):
            raise RuntimeError(
                f"{sid}: dataset epoch {loc[i]} beyond the {len(arr)} rows of the label feature table")
        new[i] = arr[loc[i]]
        seen[sid] = seen.get(sid, 0) + 1
    for sid, c in seen.items():
        if c != len(by_subject[sid]):
            raise RuntimeError(
                f"{sid}: dataset has {c} epochs but the label feature table has {len(by_subject[sid])} rows")
    if n > n_orig:
        new[n_orig:] = new[:n_orig]      # circular-shift augmentation: one copy per original, same order
    return new


def verify_stored_labels(ds, table: pd.DataFrame) -> None:
    """The dataset's stored labels must equal the table's label column, epoch by epoch."""
    ref = aligned_vector(ds, stored_labels_by_subject(table))
    bad = int((ref != np.asarray(ds.labels)).sum())
    if bad:
        raise RuntimeError(
            f"{bad} of {len(ref)} stored labels differ from the label feature table for subjects "
            f"{list(ds.unique_subjects)}: table and epochs are out of sync, cannot re-derive the label")


def fold_vector(ds, train_subjects: Iterable[str], table: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Fold-wise labels aligned to the epoch order of ds (ds is not modified)."""
    table = load_label_table() if table is None else table
    verify_stored_labels(ds, table)
    return aligned_vector(ds, labels_by_subject(table, train_subjects))


def relabel_fold(train_ds, val_ds, test_ds, train_subjects, fold_no: int,
                 table: Optional[pd.DataFrame] = None, verbose: bool = True) -> Dict[str, int]:
    """Overwrite the labels of the fold's three datasets with the fold-wise label.

    The rule is fitted on train_subjects only.  Returns the number of labels
    that changed relative to the stored labels, per dataset.
    """
    table = load_label_table() if table is None else table
    train_subjects = list(train_subjects)
    by = None
    changed: Dict[str, int] = {}
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        verify_stored_labels(ds, table)
        if by is None:
            by = labels_by_subject(table, train_subjects)
        new = aligned_vector(ds, by)
        changed[name] = int((new != np.asarray(ds.labels)).sum())
        ds.labels = new
    if verbose:
        print(f"  [Fold {fold_no:02d}] fold-wise label (scaler + median fitted on "
              f"{len(train_subjects)} training subjects): labels changed "
              f"train={changed['train']}/{len(train_ds.labels)}  "
              f"val={changed['val']}/{len(val_ds.labels)}  "
              f"test={changed['test']}/{len(test_ds.labels)}", flush=True)
    return changed
