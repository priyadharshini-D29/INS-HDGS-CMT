"""
================================================================
NEUMA Phase 8 — Results Visualisation
================================================================
Plots LOSOCV per-fold performance, ablation comparisons,
confusion matrices, and ROC curves.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


# ── LOSOCV per-fold lines ────────────────────────────────────────────────────

def plot_losocv_results(
    results_df  : pd.DataFrame,
    metrics     = ("accuracy", "f1", "roc_auc"),
    title       : str  = "LOSOCV Results",
    save_path   : Path = None,
) -> plt.Figure:
    """
    Line chart of per-fold metric values.

    Parameters
    ----------
    results_df : DataFrame with columns: fold, accuracy, f1, …
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric not in results_df.columns:
            continue
        vals  = results_df[metric].values
        folds = results_df["fold"].values if "fold" in results_df.columns \
                else range(1, len(vals) + 1)
        ax.plot(folds, vals, "o-", lw=2, color="#4C72B0", ms=7)
        ax.axhline(np.mean(vals), ls="--", color="#C44E52", lw=1.2,
                   label=f"Mean={np.mean(vals):.3f}")
        ax.fill_between(folds, np.mean(vals) - np.std(vals),
                        np.mean(vals) + np.std(vals),
                        alpha=0.15, color="#4C72B0")
        ax.set_xlabel("Fold (left-out subject)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── Ablation comparison bar chart ────────────────────────────────────────────

def plot_ablation_comparison(
    summary_df  : pd.DataFrame,
    metric      : str  = "accuracy",
    title       : str  = "Ablation Study",
    save_path   : Path = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing ablation configurations.

    Parameters
    ----------
    summary_df : from run_ablation() — columns: ablation, metric, mean, std
    """
    sub = summary_df[summary_df["metric"] == metric].copy()
    if sub.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for metric '{metric}'", ha="center")
        return fig

    sub = sub.sort_values("mean", ascending=False)
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(sub))]

    fig, ax = plt.subplots(figsize=(max(6, len(sub) * 1.2), 4))
    bars = ax.bar(sub["ablation"], sub["mean"], yerr=sub["std"],
                  color=colors, capsize=4, alpha=0.85, edgecolor="white")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticklabels(sub["ablation"], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(sub["mean"].max(), ls="--", color="gray", lw=0.8, alpha=0.5)

    # Value labels
    for bar, val in zip(bars, sub["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── Confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix_grid(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    class_names = None,
    title       : str  = "Confusion Matrix",
    save_path   : Path = None,
) -> plt.Figure:
    """Standard confusion matrix heatmap."""
    n = len(set(y_true))
    class_names = class_names or [str(i) for i in range(n)]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=class_names).plot(
        ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── ROC curves ───────────────────────────────────────────────────────────────

def plot_roc_multiclass(
    y_true      : np.ndarray,
    y_prob      : np.ndarray,
    n_classes   : int,
    class_names = None,
    title       : str  = "ROC Curves",
    save_path   : Path = None,
) -> plt.Figure:
    """One-vs-rest ROC curves for all classes."""
    class_names = class_names or [str(i) for i in range(n_classes)]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    # sklearn label_binarize returns (N, 1) for binary — expand to (N, 2)
    # so that y_bin[:, i] aligns with y_prob[:, i] for i in {0, 1}.
    if n_classes == 2 and y_bin.ndim == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors  = plt.cm.tab10(np.linspace(0, 0.9, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        if y_bin.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, lw=1.8, color=color,
                label=f"{name}  AUC={auc(fpr, tpr):.3f}")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
