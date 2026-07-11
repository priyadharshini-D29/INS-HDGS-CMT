"""
Per-subject feature loader for Phase 6 aggregation.
Applies label normalisation so raw marker strings become clean page names.
"""

import numpy as np
from pathlib import Path

# ── Label normalisation (mirrors 04_segmentation/utils/label_mapper.py) ─────
_LABEL_MAP = {
    "FYLLADIO_1.tif": "ImagePage_1",
    "FYLLADIO_2.tif": "ImagePage_2",
    "FYLLADIO_3.tif": "ImagePage_3",
    "FYLLADIO_4.tif": "ImagePage_4",
    "FYLLADIO_5.tif": "ImagePage_5",
    "FYLLADIO_6.tif": "ImagePage_6",
}

_INVALID_LABELS = {"fixation_cross", "EOE"}


def _normalize(label: str) -> str:
    for raw, clean in _LABEL_MAP.items():
        if raw in label:
            return clean
    return label


def _clean_labels(labels):
    return np.array([_normalize(str(lbl)) for lbl in labels])


def _filter_invalid(eeg, et, fusion, labels):
    """Remove fixation_cross / EOE epochs after normalisation."""
    mask = np.array([lbl not in _INVALID_LABELS for lbl in labels])
    return eeg[mask], et[mask], fusion[mask], labels[mask], int(mask.sum())


# ── Main loader ───────────────────────────────────────────────────────────────

def load_subject(subject: str, root_dir: str, phase4_folder: str) -> dict | None:
    """
    Load Phase 4 feature files for one subject.

    Returns a dict with keys: eeg, et, fusion, labels, subject_ids, n_samples
    Returns None if any required file is missing or unreadable.
    """
    feat_dir = (
        Path(root_dir) / phase4_folder / "output" / subject / "features"
    )

    paths = {
        "eeg":    feat_dir / "eeg_features.npy",
        "et":     feat_dir / "et_features.npy",
        "fusion": feat_dir / "fusion_features.npy",
        "labels": feat_dir / "labels.npy",
    }

    # Check all files present
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        print(f"  [SKIP] {subject} — missing files: {missing}")
        return None

    try:
        eeg    = np.load(paths["eeg"],    allow_pickle=True).astype(np.float32)
        et     = np.load(paths["et"],     allow_pickle=True).astype(np.float32)
        fusion = np.load(paths["fusion"], allow_pickle=True).astype(np.float32)
        labels = np.load(paths["labels"], allow_pickle=True)
    except Exception as exc:
        print(f"  [ERROR] {subject} — load failed: {exc}")
        return None

    labels = _clean_labels(labels)
    eeg, et, fusion, labels, n = _filter_invalid(eeg, et, fusion, labels)

    subject_ids = np.array([subject] * n)

    return {
        "eeg":         eeg,
        "et":          et,
        "fusion":      fusion,
        "labels":      labels,
        "subject_ids": subject_ids,
        "n_samples":   n,
    }
