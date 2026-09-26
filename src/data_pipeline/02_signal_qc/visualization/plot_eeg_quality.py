import numpy as np
import matplotlib.pyplot as plt


def plot_eeg_quality(eeg_stream, subject_id=""):

    eeg = np.array(eeg_stream["time_series"])
    timestamps = np.array(eeg_stream["time_stamps"])

    ch_std = np.std(eeg, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(timestamps, eeg[:, :8], linewidth=0.4, alpha=0.7)
    axes[0].set_title(f"EEG Raw Signal (first 8 channels) — {subject_id}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (µV)")

    axes[1].bar(range(len(ch_std)), ch_std)
    axes[1].set_title("EEG Channel Std Dev")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Std Dev (µV)")

    plt.tight_layout()
    plt.show()
