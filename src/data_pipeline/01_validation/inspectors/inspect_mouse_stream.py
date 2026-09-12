import numpy as np


def inspect_mouse_stream(mouse_stream):
    print("\n===== MOUSE STREAM INSPECTION =====")

    data = np.array(mouse_stream["time_series"])
    timestamps = np.array(mouse_stream["time_stamps"])

    print(f"Shape              : {data.shape}")
    print(f"Timestamp Start    : {timestamps[0]:.4f}")
    print(f"Timestamp End      : {timestamps[-1]:.4f}")
    print(f"Duration (s)       : {timestamps[-1] - timestamps[0]:.2f}")
    print(f"NaN Count          : {np.isnan(data).sum()}")
