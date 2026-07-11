"""
================================================================
NEUMA Phase 8 — Connectivity Visualisation
================================================================
Plots dynamic EEG connectivity matrices and electrode graphs.
================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


def plot_connectivity_matrix(
    adj       : np.ndarray,
    channel_names = None,
    title     : str  = "EEG Connectivity (Pearson |r|)",
    save_path : Path = None,
    cmap      : str  = "RdYlBu_r",
) -> plt.Figure:
    """
    Render a single (C, C) adjacency matrix as a heatmap.

    Parameters
    ----------
    adj           : (C, C) float adjacency
    channel_names : list of C channel labels
    """
    C = adj.shape[0]
    ticks = channel_names or [str(i) for i in range(C)]

    fig, ax = plt.subplots(figsize=(max(6, C * 0.35), max(5, C * 0.32)))
    im = ax.imshow(adj, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Connectivity strength")

    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(ticks, rotation=90, fontsize=7)
    ax.set_yticklabels(ticks, fontsize=7)
    ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_dynamic_graphs(
    weighted_adjs : np.ndarray,
    channel_names = None,
    title         : str  = "Dynamic EEG Connectivity",
    save_path     : Path = None,
    n_show        : int  = 5,
) -> plt.Figure:
    """
    Plot a row of W connectivity matrices (one per temporal window).

    Parameters
    ----------
    weighted_adjs : (W, C, C)
    n_show        : max windows to display
    """
    W, C, _ = weighted_adjs.shape
    n_show = min(n_show, W)
    ticks  = channel_names or [str(i) for i in range(C)]

    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 3.2, 3.5))
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        im = ax.imshow(weighted_adjs[i], cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_title(f"Window {i+1}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0 and len(ticks) <= 16:
            ax.set_xticks(range(C))
            ax.set_xticklabels(ticks, rotation=90, fontsize=6)
            ax.set_yticks(range(C))
            ax.set_yticklabels(ticks, fontsize=6)

    fig.colorbar(im, ax=axes[-1], shrink=0.8, label="|r|")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_electrode_graph(
    adj           : np.ndarray,
    cam           : np.ndarray = None,
    channel_names = None,
    threshold     : float = 0.30,
    title         : str  = "Electrode Connectivity Graph",
    save_path     : Path = None,
) -> plt.Figure:
    """
    Render electrode connectivity as a NetworkX graph.

    Node colour = Grad-CAM importance (if provided).
    Edge width   = connectivity strength.

    Requires networkx.
    """
    if not _HAS_NX:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "networkx not installed", ha="center")
        if save_path:
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
        return fig

    C     = adj.shape[0]
    names = channel_names or [str(i) for i in range(C)]

    G = nx.Graph()
    for i in range(C):
        G.add_node(i, label=names[i])

    edge_weights = []
    for i in range(C):
        for j in range(i + 1, C):
            w = float(adj[i, j])
            if w >= threshold:
                G.add_edge(i, j, weight=w)
                edge_weights.append(w)

    if len(G.edges) == 0:
        # Add all edges with weight < threshold for visualisation
        for i in range(C):
            for j in range(i + 1, C):
                G.add_edge(i, j, weight=float(adj[i, j]))
                edge_weights.append(float(adj[i, j]))

    node_colors = cam if cam is not None else np.full(C, 0.5)
    widths = [G[u][v]["weight"] * 3 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        cmap=plt.cm.RdYlBu_r,
        node_size=600,
        vmin=0, vmax=1,
    )
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: names[i] for i in range(C)},
                            font_size=7)
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.6,
                           edge_color="gray")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
