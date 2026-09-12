import numpy as np


def check_nan_inf(data, label="data"):
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    if nan_count > 0:
        print(f"[WARN] {label}: {nan_count} NaN values")
    if inf_count > 0:
        print(f"[WARN] {label}: {inf_count} Inf values")
    return nan_count, inf_count


def check_timestamps_monotonic(timestamps, label="stream"):
    diffs = np.diff(timestamps)
    bad = (diffs <= 0).sum()
    if bad > 0:
        print(f"[WARN] {label}: {bad} non-monotonic timestamp steps")
    return bad
