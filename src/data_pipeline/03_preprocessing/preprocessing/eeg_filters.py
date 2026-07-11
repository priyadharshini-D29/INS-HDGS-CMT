from scipy.signal import butter, filtfilt, iirnotch
import numpy as np


def bandpass_filter(
    eeg,
    fs=300,
    lowcut=1,
    highcut=45,
    order=4
):

    nyq = 0.5 * fs

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")

    filtered = filtfilt(b, a, eeg, axis=0)

    return filtered


def notch_filter(
    eeg,
    fs=300,
    freq=50,
    q=30
):

    b, a = iirnotch(freq, q, fs)

    filtered = filtfilt(b, a, eeg, axis=0)

    return filtered
