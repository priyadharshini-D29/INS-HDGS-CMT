import numpy as np
from scipy.interpolate import interp1d


def synchronize_et_to_eeg(
    et_data,
    et_ts,
    eeg_ts
):

    synced = np.zeros(
        (len(eeg_ts), et_data.shape[1])
    )

    synced[:] = np.nan

    for ch in range(et_data.shape[1]):

        signal = et_data[:, ch]

        valid = ~np.isnan(signal)

        # need enough valid samples
        if np.sum(valid) < 2:
            continue

        f = interp1d(
            et_ts[valid],
            signal[valid],
            kind="nearest",
            bounds_error=False,
            fill_value=np.nan
        )

        synced[:, ch] = f(eeg_ts)

    return synced