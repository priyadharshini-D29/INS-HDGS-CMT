import numpy as np

from configs.thresholds import EEG_MIN_AMPLITUDE, EEG_MAX_AMPLITUDE, EEG_FLAT_STD_THRESHOLD


def validate_eeg_signal(eeg_stream):

    eeg = np.array(eeg_stream["time_series"])

    print("\n===== EEG SIGNAL VALIDATION =====")

    print(f"Shape : {eeg.shape}")

    nan_count = np.isnan(eeg).sum()

    inf_count = np.isinf(eeg).sum()

    print(f"NaN Count : {nan_count}")
    print(f"Inf Count : {inf_count}")

    ch_std = np.std(eeg, axis=0)

    flat_channels = np.where(ch_std < EEG_FLAT_STD_THRESHOLD)[0]

    print(f"Flat channels : {flat_channels}")

    ch_min = eeg.min(axis=0)
    ch_max = eeg.max(axis=0)

    for i in range(eeg.shape[1]):

        amplitude_ok = (ch_min[i] >= EEG_MIN_AMPLITUDE) and (ch_max[i] <= EEG_MAX_AMPLITUDE)
        status = "[PASS]" if amplitude_ok else "[WARN]"

        print(
            f"{status} Ch{i:02d} | "
            f"min={ch_min[i]:.2f} "
            f"max={ch_max[i]:.2f} "
            f"std={ch_std[i]:.2f}"
        )
