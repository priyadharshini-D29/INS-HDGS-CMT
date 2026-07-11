"""
================================================================
NEUMA Phase 8 — Attention Visualisation
================================================================
Plots fusion attention weights, temporal attention, and ROI
gate visualisations.
================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_attention_heatmap(
    attn_matrix : np.ndarray,
    row_labels  = None,
    col_labels  = None,
    title       : str  = "Cross-Modal Attention",
    save_path   : Path = None,
) -> plt.Figure:
    """
    Render a 2-D attention weight matrix as a heatmap.

    Parameters
    ----------
    attn_matrix : (Q, K) attention weights
    """
    Q, K = attn_matrix.shape
    row_labels = row_labels or [f"Q{i}" for i in range(Q)]
    col_labels = col_labels or [f"K{i}" for i in range(K)]

    fig, ax = plt.subplots(figsize=(max(4, K * 0.6), max(3, Q * 0.5)))
    im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Attention weight")

    ax.set_xticks(range(K))
    ax.set_yticks(range(Q))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_temporal_attention(
    temporal_emb   : np.ndarray,
    window_labels  = None,
    title          : str  = "Temporal EEG Attention (Window Embeddings)",
    save_path      : Path = None,
) -> plt.Figure:
    """
    Plot per-window EEG embedding norms to show temporal attention.

    Parameters
    ----------
    temporal_emb : (W, D)  temporal transformer output
    """
    W, D = temporal_emb.shape
    window_norms = np.linalg.norm(temporal_emb, axis=1)          # (W,)
    window_norms = (window_norms - window_norms.min()) / \
                   (window_norms.max() - window_norms.min() + 1e-8)

    labels = window_labels or [f"W{i+1}" for i in range(W)]

    fig, ax = plt.subplots(figsize=(max(6, W), 3))
    bars = ax.bar(labels, window_norms, color=plt.cm.Blues(window_norms + 0.2))
    ax.set_ylabel("Normalised embedding norm")
    ax.set_xlabel("Temporal window (0.5 s each)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=window_norms.mean(), ls="--", color="red", alpha=0.6,
               label=f"Mean = {window_norms.mean():.3f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_roi_gate(
    gate     : np.ndarray,
    roi_vec  : np.ndarray,
    title    : str  = "ROI-Guided Modulation Gate",
    save_path: Path = None,
) -> plt.Figure:
    """
    Side-by-side bar plots of ROI dwell vector and gate activation.

    Parameters
    ----------
    gate    : (D,) sigmoid gate values (will be summarised)
    roi_vec : (N_rois,) dwell-time vector
    """
    n_rois = len(roi_vec)
    gate_summary = gate[:n_rois] if len(gate) >= n_rois else gate

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].bar(range(n_rois), roi_vec, color="#4C72B0", alpha=0.8)
    axes[0].set_title("ROI Dwell-Time Distribution", fontsize=10)
    axes[0].set_xlabel("ROI index")
    axes[0].set_ylabel("Proportion of gaze")

    axes[1].bar(range(len(gate_summary)), gate_summary,
                color="#DD8452", alpha=0.8)
    axes[1].set_title("Sigmoid Gate Activation", fontsize=10)
    axes[1].set_xlabel("Feature dim (first N_rois shown)")
    axes[1].set_ylabel("Gate value")
    axes[1].set_ylim(0, 1.05)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
