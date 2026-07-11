import pandas as pd
import numpy as np


def interpolate_et(
    et,
    max_gap=12
):
    """
    max_gap=12 samples
    at 120 Hz ≈ 100 ms
    """

    et_interp = np.copy(et)

    for ch in range(et.shape[1]):

        s = pd.Series(et[:, ch])

        s = s.interpolate(
            method="linear",
            limit=max_gap,
            limit_direction="both"
        )

        et_interp[:, ch] = s.values

    remaining_nan = np.isnan(et_interp).sum()

    print(
        f"\nET interpolation complete "
        f"— Remaining NaNs: {remaining_nan}"
    )

    return et_interp