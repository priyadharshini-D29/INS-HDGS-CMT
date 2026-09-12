import numpy as np
import pandas as pd


def smooth_et(et, window=5):

    smoothed = np.empty_like(et, dtype=float)

    for ch in range(et.shape[1]):
        s = pd.Series(et[:, ch])
        # min_periods=1 averages only valid samples inside each window,
        # so existing NaNs are preserved rather than spreading to neighbors
        smoothed[:, ch] = s.rolling(
            window,
            center=True,
            min_periods=1
        ).mean().values

    nan_after = np.isnan(smoothed).sum()
    print(f"\nET smoothing complete — window={window}, NaNs after: {nan_after}")

    return smoothed
