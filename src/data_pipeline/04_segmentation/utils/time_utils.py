import numpy as np


def find_nearest_index(timestamps, target_time):
    """Return the index of the timestamp closest to target_time."""
    return int(np.argmin(np.abs(timestamps - target_time)))


def time_to_sample(time_sec, fs):
    return int(round(time_sec * fs))


def sample_to_time(sample_idx, fs):
    return sample_idx / fs


def get_time_mask(timestamps, t_start, t_end):
    return (timestamps >= t_start) & (timestamps <= t_end)


def compute_effective_sr(timestamps):
    duration = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / duration if duration > 0 else 0.0
