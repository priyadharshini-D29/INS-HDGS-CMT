import numpy as np


def normalise_epoch(epoch):
    """Z-score normalise each channel independently."""
    mean = np.nanmean(epoch, axis=0, keepdims=True)
    std  = np.nanstd(epoch,  axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (epoch - mean) / std


def pad_or_trim(epoch, target_len):
    """Pad with zeros or trim so every epoch has exactly target_len samples."""
    n = len(epoch)
    if n >= target_len:
        return epoch[:target_len]
    pad_width = [(0, target_len - n)] + [(0, 0)] * (epoch.ndim - 1)
    return np.pad(epoch, pad_width, mode="constant", constant_values=0)


def count_nan_channels(epoch, threshold=0.5):
    """Return indices of channels where NaN ratio exceeds threshold."""
    nan_ratio = np.isnan(epoch).mean(axis=0)
    return np.where(nan_ratio > threshold)[0]
