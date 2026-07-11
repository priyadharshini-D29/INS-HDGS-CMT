import numpy as np


def inspect_et_stream(et_stream):
    print("\n===== ET STREAM INSPECTION =====")

    data = np.array(et_stream["time_series"])
    timestamps = np.array(et_stream["time_stamps"])

    print(f"Shape              : {data.shape}")
    print(f"Timestamp Start    : {timestamps[0]:.4f}")
    print(f"Timestamp End      : {timestamps[-1]:.4f}")
    print(f"Duration (s)       : {timestamps[-1] - timestamps[0]:.2f}")
    print(f"NaN Count          : {np.isnan(data).sum()}")
    print(f"Inf Count          : {np.isinf(data).sum()}")
    print(f"Min Value          : {np.nanmin(data):.4f}")
    print(f"Max Value          : {np.nanmax(data):.4f}")
