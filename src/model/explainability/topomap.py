"""
================================================================
NEUMA Phase 8 — EEG Topographic Maps
================================================================
Renders electrode importance values onto a 2-D scalp topography
using MNE's plot_topomap.

Requires MNE (pip install mne).  Falls back to a scatter plot
on a standard 10-20 layout if MNE is unavailable.
================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import mne
    _HAS_MNE = True
except ImportError:
    _HAS_MNE = False


# Standard 10-20 positions (x, y in normalised head coords).
# Covers the 19-channel real cortical montage (see
# data.channel_harmonizer.CANONICAL_CHANNELS) — T3/T4/T5/T6 naming
# (not T7/T8/P7/P8), no Oz/FC1/FC2/FC5/FC6 (not part of the montage).
_DEFAULT_10_20 = {
    "Fp1": (-0.31,  0.95), "Fp2": ( 0.31,  0.95),
    "F7" : (-0.71,  0.71), "F3" : (-0.40,  0.69),
    "Fz" : ( 0.00,  0.72), "F4" : ( 0.40,  0.69),
    "F8" : ( 0.71,  0.71),
    "T3" : (-1.00,  0.00), "C3" : (-0.60,  0.00),
    "Cz" : ( 0.00,  0.00), "C4" : ( 0.60,  0.00),
    "T4" : ( 1.00,  0.00),
    "T5" : (-0.71, -0.71), "P3" : (-0.40, -0.69),
    "Pz" : ( 0.00, -0.72), "P4" : ( 0.40, -0.69),
    "T6" : ( 0.71, -0.71),
    "O1" : (-0.31, -0.95),
    "O2" : ( 0.31, -0.95),
}


def _fallback_topomap(values, channel_names, title, ax):
    """Simple scatter-based topomap when MNE is unavailable."""
    pos_map = _DEFAULT_10_20
    matched_names, matched_vals, coords = [], [], []

    for i, ch in enumerate(channel_names):
        key = ch.strip()
        if key in pos_map:
            coords.append(pos_map[key])
            matched_vals.append(values[i])
            matched_names.append(key)

    if not coords:
        ax.text(0.5, 0.5, "No channel positions", ha="center", va="center")
        return

    xs, ys = zip(*coords)
    sc = ax.scatter(xs, ys, c=matched_vals, s=200, cmap="RdBu_r",
                    vmin=0, vmax=1, zorder=3)
    for x, y, name in zip(xs, ys, matched_names):
        ax.text(x, y + 0.08, name, ha="center", fontsize=7)

    # Draw head circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1.5)
    ax.plot([0.0, 0.0], [1.0, 1.15], "k-", lw=1.5)   # nose
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.colorbar(sc, ax=ax, shrink=0.7)


def plot_topomap(
    values       : np.ndarray,
    channel_names,
    title        : str  = "Electrode Importance",
    mne_info     = None,
    ax           = None,
) -> plt.Figure:
    """
    Plot electrode importance values as a topographic map.

    Parameters
    ----------
    values        : (C,) importance scores normalised to [0, 1]
    channel_names : list of channel name strings
    title         : plot title
    mne_info      : mne.Info object (if available; improves layout)
    ax            : existing matplotlib Axes (None → create new)

    Returns
    -------
    fig : matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    if _HAS_MNE and mne_info is not None:
        mne.viz.plot_topomap(
            values,
            mne_info,
            axes    = ax,
            show    = False,
            cmap    = "RdBu_r",
            vlim    = (0, 1),
            contours= 4,
        )
        ax.set_title(title, fontsize=10, fontweight="bold")
    else:
        _fallback_topomap(values, channel_names, title, ax)

    plt.tight_layout()
    return fig


def save_topomap(
    values       : np.ndarray,
    channel_names,
    save_path    : Path,
    title        : str  = "Electrode Importance",
    mne_info     = None,
):
    """Save topomap to file."""
    fig = plot_topomap(values, channel_names, title, mne_info)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
