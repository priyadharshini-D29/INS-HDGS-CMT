import numpy as np


def zscore_normalize(X):
    """Z-score each feature column independently."""
    mean = np.mean(X, axis=0, keepdims=True)
    std  = np.std(X,  axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X - mean) / std


def minmax_normalize(X):
    xmin = X.min(axis=0, keepdims=True)
    xmax = X.max(axis=0, keepdims=True)
    rng  = xmax - xmin
    rng[rng < 1e-8] = 1.0
    return (X - xmin) / rng
