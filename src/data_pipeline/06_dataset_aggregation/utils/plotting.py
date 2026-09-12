"""
Publication-quality plots for Phase 6 dataset summary.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.1)
_PALETTE = "tab10"


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Class distribution ────────────────────────────────────────────────────────

def plot_class_distribution(labels, subjects, save_path):
    counts = Counter(labels)
    sorted_items = sorted(counts.items(), key=lambda x: x[0])
    names  = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Also show per-subject breakdown
    df = pd.DataFrame({"label": labels, "subject": subjects})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left — global bar
    colors = sns.color_palette(_PALETTE, len(names))
    axes[0].bar(range(len(names)), values, color=colors, edgecolor="white")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    axes[0].set_ylabel("Sample count")
    axes[0].set_title("Global Class Distribution", fontweight="bold")
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.3, str(v), ha="center", fontsize=8)

    # Right — stacked by subject
    pivot = df.groupby(["subject", "label"]).size().unstack(fill_value=0)
    pivot.plot(kind="bar", stacked=True, ax=axes[1],
               colormap=_PALETTE, edgecolor="white", legend=True)
    axes[1].set_xlabel("Subject")
    axes[1].set_ylabel("Sample count")
    axes[1].set_title("Class Distribution per Subject", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.suptitle("Label Distribution — NeuMa Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)


# ── Subject distribution ──────────────────────────────────────────────────────

def plot_subject_distribution(subjects, save_path):
    counts = Counter(subjects)
    sorted_items = sorted(counts.items())
    names  = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.5), 5))
    colors = sns.color_palette("Blues_d", len(names))
    ax.bar(names, values, color=colors, edgecolor="white")
    ax.axhline(np.mean(values), color="red", linestyle="--",
               linewidth=1.2, label=f"Mean = {np.mean(values):.1f}")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Epoch count")
    ax.set_title("Samples per Subject", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9)
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=8)
    plt.tight_layout()
    _save(fig, save_path)


# ── Feature correlation heatmap ───────────────────────────────────────────────

def plot_correlation_heatmap(fusion, save_path, n_features=50):
    n_cols = min(n_features, fusion.shape[1])
    df     = pd.DataFrame(fusion[:, :n_cols])
    corr   = df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle only

    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", center=0,
        vmin=-1, vmax=1,
        annot=False, linewidths=0,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title(
        f"Fusion Feature Correlation (first {n_cols} features)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, save_path)


# ── Per-modality statistics table ─────────────────────────────────────────────

def plot_modality_stats(eeg, et, fusion, labels, save_path):
    """Bar chart of per-modality mean absolute feature value by class."""
    classes = np.unique(labels)
    modalities = {"EEG": eeg, "ET": et, "Fusion": fusion}

    rows = []
    for mod_name, X in modalities.items():
        for cls in classes:
            mask = labels == cls
            mean_abs = float(np.abs(X[mask]).mean())
            rows.append({"Modality": mod_name, "Class": str(cls),
                         "Mean |feature|": mean_abs})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = df.pivot(index="Class", columns="Modality", values="Mean |feature|")
    pivot.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white")
    ax.set_title("Mean Absolute Feature Magnitude by Class & Modality",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Mean |feature value|")
    ax.tick_params(axis="x", rotation=40)
    ax.legend(title="Modality", fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)
