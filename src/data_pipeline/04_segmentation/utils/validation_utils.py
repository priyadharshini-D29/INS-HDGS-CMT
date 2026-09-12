import numpy as np


def check_shape(array, expected_ndim, label="array"):
    if array.ndim != expected_ndim:
        raise ValueError(f"{label}: expected {expected_ndim}D, got {array.ndim}D")


def check_no_all_nan(array, label="array"):
    if np.isnan(array).all():
        raise ValueError(f"{label}: all values are NaN")


def check_min_length(array, min_len, label="array"):
    if len(array) < min_len:
        raise ValueError(f"{label}: length {len(array)} < minimum {min_len}")


def epoch_nan_report(epoch, label="epoch"):
    nan_ratio = np.isnan(epoch).mean()
    print(f"[{label}] NaN ratio: {nan_ratio:.4f} ({np.isnan(epoch).sum()} / {epoch.size})")
    return nan_ratio
