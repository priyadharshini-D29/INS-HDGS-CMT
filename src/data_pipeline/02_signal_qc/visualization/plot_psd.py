import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def plot_psd(eeg_stream):

    eeg = np.array(eeg_stream["time_series"])

    fs = 300

    plt.figure(figsize=(12, 6))

    for ch in range(min(5, eeg.shape[1])):

        freqs, psd = welch(
            eeg[:, ch],
            fs=fs,
            nperseg=2048
        )

        plt.semilogy(freqs, psd, label=f"Ch{ch}")

    plt.title("EEG Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")

    plt.xlim(0, 60)

    plt.legend()

    plt.tight_layout()

    plt.show()