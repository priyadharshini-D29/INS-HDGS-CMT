import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_epoch_summary(metadata_df, save_path=None):
    """
    Bar chart of epoch counts per label + EEG/ET sample-count distributions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Epoch count per label ─────────────────────────────────
    label_counts = metadata_df["label"].value_counts()
    axes[0].bar(label_counts.index, label_counts.values, color="steelblue")
    axes[0].set_title("Epochs per Label")
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    # ── EEG sample count distribution ────────────────────────
    axes[1].hist(metadata_df["eeg_samples"], bins=20, color="green", alpha=0.7)
    axes[1].set_title("EEG Samples per Epoch")
    axes[1].set_xlabel("Samples")
    axes[1].set_ylabel("Count")

    # ── ET sample count distribution ──────────────────────────
    axes[2].hist(metadata_df["et_samples"], bins=20, color="tomato", alpha=0.7)
    axes[2].set_title("ET Samples per Epoch")
    axes[2].set_xlabel("Samples")
    axes[2].set_ylabel("Count")

    plt.suptitle("Phase 3 — Epoch Summary", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_fixation_density(fixation_list_per_epoch, save_path=None):
    """
    Histogram of fixation counts across all epochs.
    fixation_list_per_epoch : list of lists (one per epoch).
    """
    counts = [len(fx) for fx in fixation_list_per_epoch]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=20, color="purple", alpha=0.7)
    ax.set_title("Fixation Count Distribution Across Epochs")
    ax.set_xlabel("Fixations per epoch")
    ax.set_ylabel("Epoch count")

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
