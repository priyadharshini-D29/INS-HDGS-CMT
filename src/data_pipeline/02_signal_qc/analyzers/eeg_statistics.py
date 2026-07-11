import numpy as np


def compute_eeg_statistics(eeg_stream):

    eeg = np.array(eeg_stream["time_series"])
    timestamps = np.array(eeg_stream["time_stamps"])

    duration = timestamps[-1] - timestamps[0]
    effective_sr = (len(timestamps) - 1) / duration

    stats = {
        "shape": eeg.shape,
        "duration_sec": duration,
        "effective_sr": effective_sr,
        "nan_count": int(np.isnan(eeg).sum()),
        "inf_count": int(np.isinf(eeg).sum()),
        "channel_std": np.std(eeg, axis=0).tolist(),
        "channel_min": np.nanmin(eeg, axis=0).tolist(),
        "channel_max": np.nanmax(eeg, axis=0).tolist(),
        "channel_mean": np.nanmean(eeg, axis=0).tolist(),
    }

    print("\n===== EEG STATISTICS =====")
    print(f"Shape          : {stats['shape']}")
    print(f"Duration (s)   : {stats['duration_sec']:.2f}")
    print(f"Effective SR   : {stats['effective_sr']:.2f} Hz")
    print(f"NaN count      : {stats['nan_count']}")
    print(f"Inf count      : {stats['inf_count']}")

    return stats
