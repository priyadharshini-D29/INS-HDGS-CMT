import numpy as np


def common_average_reference(eeg):

    mean_signal = np.mean(eeg, axis=1, keepdims=True)

    reref = eeg - mean_signal

    return reref
