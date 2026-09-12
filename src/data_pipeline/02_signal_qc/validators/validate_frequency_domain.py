import numpy as np
from scipy.signal import welch


def validate_frequency_domain(eeg_stream):

    eeg = np.array(eeg_stream["time_series"])
    fs = 300

    print("\n===== EEG FREQUENCY DOMAIN VALIDATION =====")

    for ch in range(min(5, eeg.shape[1])):

        signal = eeg[:, ch]

        freqs, psd = welch(
            signal,
            fs=fs,
            nperseg=2048
        )

        peak_freq = freqs[np.argmax(psd)]

        delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
        theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
        alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
        beta = np.mean(psd[(freqs >= 13) & (freqs < 30)])

        print(
            f"\nCh{ch:02d}"
            f"\nPeak Frequency : {peak_freq:.2f} Hz"
            f"\nDelta Power    : {delta:.4f}"
            f"\nTheta Power    : {theta:.4f}"
            f"\nAlpha Power    : {alpha:.4f}"
            f"\nBeta Power     : {beta:.4f}"
        )

        if peak_freq >= 45 and peak_freq <= 55:
            print("[WARN] Possible line noise contamination")