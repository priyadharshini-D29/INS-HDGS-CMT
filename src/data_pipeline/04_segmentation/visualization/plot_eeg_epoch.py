import numpy as np
import matplotlib.pyplot as plt

from config.settings import EEG_SR


def plot_eeg_epoch(epoch, event_id=None, n_channels=8, offset_uv=40, save_path=None):
    """
    Waterfall plot of EEG epoch channels.
    epoch : (T, C) array.
    """
    n_ch  = min(n_channels, epoch.shape[1])
    times = np.arange(len(epoch)) / EEG_SR

    fig, ax = plt.subplots(figsize=(14, 6))

    for ch in range(n_ch):
        ax.plot(times, epoch[:, ch] + ch * offset_uv, linewidth=0.6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels (offset)")
    ax.set_title(f"EEG Epoch{f' — event {event_id}' if event_id is not None else ''}")
    ax.set_yticks(np.arange(n_ch) * offset_uv)
    ax.set_yticklabels([f"Ch{i}" for i in range(n_ch)])

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
