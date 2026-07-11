"""
Shape and quality checks for per-subject feature arrays.
"""

import numpy as np


def validate_subject(data: dict, subject: str,
                     expected_eeg_feats: int | None = None,
                     expected_et_feats:  int | None = None) -> bool:
    """
    Return True if the subject's data passes all checks.
    Logs a descriptive warning for every failure.
    """
    eeg, et, fusion, labels = (
        data["eeg"], data["et"], data["fusion"], data["labels"]
    )
    n = data["n_samples"]

    if n == 0:
        print(f"  [SKIP] {subject} — 0 samples after label filtering")
        return False

    # Row-count consistency
    if not (len(eeg) == len(et) == len(fusion) == len(labels) == n):
        print(f"  [SKIP] {subject} — row counts inconsistent "
              f"EEG={len(eeg)} ET={len(et)} Fusion={len(fusion)} Labels={len(labels)}")
        return False

    # Feature dimension consistency
    if expected_eeg_feats is not None and eeg.shape[1] != expected_eeg_feats:
        print(f"  [SKIP] {subject} — EEG feature dim {eeg.shape[1]} "
              f"!= expected {expected_eeg_feats}")
        return False

    if expected_et_feats is not None and et.shape[1] != expected_et_feats:
        print(f"  [SKIP] {subject} — ET feature dim {et.shape[1]} "
              f"!= expected {expected_et_feats}")
        return False

    # NaN / Inf warnings (don't skip, just warn)
    for name, arr in [("EEG", eeg), ("ET", et), ("Fusion", fusion)]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan:
            print(f"  [WARN] {subject} — {name} has {n_nan} NaN values")
        if n_inf:
            print(f"  [WARN] {subject} — {name} has {n_inf} Inf values")

    return True


def infer_expected_dims(all_data: list[dict]) -> tuple[int | None, int | None]:
    """
    Determine the most common EEG and ET feature dimensions
    across already-loaded subjects (majority vote).
    """
    from collections import Counter

    eeg_dims = Counter(d["eeg"].shape[1] for d in all_data)
    et_dims  = Counter(d["et"].shape[1]  for d in all_data)

    exp_eeg = eeg_dims.most_common(1)[0][0] if eeg_dims else None
    exp_et  = et_dims.most_common(1)[0][0]  if et_dims  else None
    return exp_eeg, exp_et
