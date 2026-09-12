import numpy as np
from scipy.signal import welch


def compute_bandpower(signal, sf, band):

    low, high = band

    freqs, psd = welch(signal, sf, nperseg=sf * 2)

    idx = np.logical_and(freqs >= low, freqs <= high)

    bp = np.trapezoid(psd[idx], freqs[idx])

    return bp


def compute_all_bandpowers(signal, sf, bands):
    return {
        name: compute_bandpower(signal, sf, rng)
        for name, rng in bands.items()
    }


def relative_bandpower(signal, sf, band, total_band=(1, 45)):
    bp       = compute_bandpower(signal, sf, band)
    bp_total = compute_bandpower(signal, sf, total_band)
    return bp / bp_total if bp_total > 0 else 0.0
