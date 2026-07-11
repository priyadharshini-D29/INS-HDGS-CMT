import numpy as np


def validate_artifacts(eeg_stream, et_stream):

    eeg = np.array(eeg_stream["time_series"])
    et = np.array(et_stream["time_series"])

    print("\n===== ARTIFACT VALIDATION =====")

    eeg_std = np.std(eeg, axis=1)

    spikes = np.sum(eeg_std > np.percentile(eeg_std, 99))

    print(f"High-amplitude EEG spikes : {spikes}")

    et_nan = np.isnan(et).any(axis=1)

    blink_count = np.sum(et_nan)

    print(f"Possible blink samples    : {blink_count}")

    overlap = min(len(eeg_std), len(et_nan))

    artifact_overlap = np.sum(
        (eeg_std[:overlap] > np.percentile(eeg_std, 99))
        &
        (et_nan[:overlap])
    )

    print(f"EEG-ET overlap artifacts  : {artifact_overlap}")