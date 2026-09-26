import numpy as np
import matplotlib.pyplot as plt


def plot_clean_eeg(eeg_raw, eeg_clean, subject_id=""):

    n_ch = min(8, eeg_raw.shape[1], eeg_clean.shape[1])

    fig, axes = plt.subplots(n_ch, 2, figsize=(16, 2 * n_ch), sharex=True)

    for i in range(n_ch):
        axes[i, 0].plot(eeg_raw[:3000, i], linewidth=0.4, color="steelblue")
        axes[i, 0].set_ylabel(f"Ch{i}")

        axes[i, 1].plot(eeg_clean[:3000, i], linewidth=0.4, color="green")

    axes[0, 0].set_title(f"Raw EEG — {subject_id}")
    axes[0, 1].set_title(f"Clean EEG — {subject_id}")

    plt.tight_layout()
    plt.show()
