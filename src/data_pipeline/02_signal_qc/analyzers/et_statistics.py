import numpy as np


def compute_et_statistics(et_stream):

    et = np.array(et_stream["time_series"])
    timestamps = np.array(et_stream["time_stamps"])

    duration = timestamps[-1] - timestamps[0]
    effective_sr = (len(timestamps) - 1) / duration

    stats = {
        "shape": et.shape,
        "duration_sec": duration,
        "effective_sr": effective_sr,
        "nan_ratio_per_channel": np.isnan(et).mean(axis=0).tolist(),
        "channel_min": np.nanmin(et, axis=0).tolist(),
        "channel_max": np.nanmax(et, axis=0).tolist(),
        "channel_mean": np.nanmean(et, axis=0).tolist(),
    }

    print("\n===== ET STATISTICS =====")
    print(f"Shape          : {stats['shape']}")
    print(f"Duration (s)   : {stats['duration_sec']:.2f}")
    print(f"Effective SR   : {stats['effective_sr']:.2f} Hz")

    for i, nr in enumerate(stats["nan_ratio_per_channel"]):
        print(f"Ch{i:02d} NaN ratio : {nr:.4f}")

    return stats
