import numpy as np


def inspect_eeg_stream(eeg_stream):
    print("\n===== EEG STREAM INSPECTION =====")

    data = np.array(eeg_stream["time_series"])
    timestamps = np.array(eeg_stream["time_stamps"])

    print(f"Shape              : {data.shape}")
    print(f"Total Samples      : {data.shape[0]}")
    print(f"Total Channels     : {data.shape[1]}")
    print(f"Timestamp Start    : {timestamps[0]:.4f}")
    print(f"Timestamp End      : {timestamps[-1]:.4f}")
    print(f"Duration (s)       : {timestamps[-1] - timestamps[0]:.2f}")
    print(f"NaN Count          : {np.isnan(data).sum()}")
    print(f"Inf Count          : {np.isinf(data).sum()}")
    print(f"Amplitude Min      : {np.nanmin(data):.4f}")
    print(f"Amplitude Max      : {np.nanmax(data):.4f}")
