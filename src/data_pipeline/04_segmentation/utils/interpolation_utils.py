import numpy as np
import pandas as pd


def fill_short_gaps(signal, max_gap_samples=12):
    """
    Linear interpolation for NaN gaps shorter than max_gap_samples.
    Longer gaps are left as NaN (blinks / loss of tracking).
    """
    s = pd.Series(signal)
    s = s.interpolate(method="linear", limit=max_gap_samples, limit_direction="both")
    return s.values


def fill_short_gaps_2d(et, max_gap_samples=12):
    out = np.copy(et)
    for ch in range(et.shape[1]):
        out[:, ch] = fill_short_gaps(et[:, ch], max_gap_samples)
    return out
