"""
==============================================================
NEUMA Phase 8A — EEG Connectivity Graph Construction
ROI-Guided Dynamic Graph Attention NeuroFusion Network
==============================================================
Converts EEG epochs into dynamic functional brain connectivity
graphs for Graph Attention Network (GAT) learning.

Connectivity methods:
  Pearson correlation : A_ij = corr(x_i, x_j)
  Spectral coherence  : A_ij = mean_f Cxy(x_i, x_j)
  Phase Locking Value : A_ij = |mean_t e^{i(phi_i - phi_j)}|

Dynamic graphs:
  Each epoch is split into N_WINDOWS temporal windows.
  A separate adjacency matrix is computed per window,
  yielding a time-varying connectivity sequence:

    G_t = (V, A_t),  t = 1 ... N_WINDOWS

Outputs:
  adjacency.npy           (N, C, C)       mean Pearson adjacency
  adjacency_windowed.npy  (N, W, C, C)    dynamic window adjacency
  adjacency_coherence.npy (N, C, C)       mean coherence adjacency
  adjacency_plv.npy       (N, C, C)       mean PLV adjacency
  edge_indices.npy        (N,) object     COO edge_index per epoch
  edge_attrs.npy          (N,) object     edge weights per epoch
  graph_metrics.csv                        per-epoch graph statistics

Visualisations:
  connectivity_heatmap.png
  connectivity_graph.png
  dynamic_connectivity.png
  graph_metrics_boxplot.png
  method_comparison.png
  degree_distribution.png
==============================================================
"""

import os
import sys
import warnings
import io as _io

# ── UTF-8 / Unicode-safe I/O ────────────────────────────────────────────────
os.environ.setdefault("PYTHONUTF8", "1")
try:
    if isinstance(sys.stdout, _io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, _io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_REPL = {
    "→": "->", "←": "<-", "✔": "[OK]", "✘": "[X]",
    "─": "-",  "│": "|",  "═": "=",   "├": "+",
    "┤": "+",  "└": "+",  "┐": "+",   "┌": "+",
    "┘": "+",  "█": "#",  "░": ".",   "±": "+/-",
}

def sanitize(t):
    t = str(t)
    for o, n in _REPL.items():
        t = t.replace(o, n)
    return t

def sp(*args, **kw):
    sep = kw.pop("sep", " ")
    text = sanitize(sep.join(str(a) for a in args))
    try:
        print(text, **kw)
    except UnicodeEncodeError:
        print(text.encode("ascii", "ignore").decode(), **kw)

warnings.filterwarnings("ignore")

# ── Imports ──────────────────────────────────────────────────────────────────
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    sp("[WARN] networkx not installed — graph plots skipped")

from scipy.signal import coherence, welch
from scipy.signal import hilbert

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

def _progress(iterable, desc=""):
    if _HAS_TQDM:
        return tqdm(iterable, desc=desc, ncols=70)
    return iterable

# ============================================================
# CONFIG
# ============================================================

# Paths
# _HERE.parent (src/) is the shared ancestor of both src/model/ and
# src/data_pipeline/ -- still correct after the reorg, just the sibling
# folder name changed (NEUMA_PHASE3 -> data_pipeline/04_segmentation).
_HERE       = Path(__file__).resolve().parent
PHASE3_DIR  = _HERE.parent / "data_pipeline" / "04_segmentation" / "output"

# Try multiple candidate epoch paths (local first, then Phase 3)
_CANDIDATES = [
    Path("output/epochs/eeg_epochs.npy"),         # run from 04_segmentation
    PHASE3_DIR / "epochs" / "eeg_epochs.npy",     # sibling from src/model
    _HERE / "output" / "epochs" / "eeg_epochs.npy",
]

LABELS_CANDIDATES = [
    Path("output/epochs/labels.npy"),
    PHASE3_DIR / "epochs" / "labels.npy",
    _HERE / "output" / "epochs" / "labels.npy",
]

OUT_DIR = _HERE / "output" / "connectivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Connectivity parameters ──────────────────────────────────
EEG_SR       = 300          # Hz
N_WINDOWS    = 10           # dynamic windows per epoch
THRESHOLD    = 0.30         # edge-inclusion absolute threshold
THRESHOLD_VIS= 0.50         # stricter threshold for network visualisation
CONN_METHODS = ["pearson", "coherence", "plv"]   # methods to compute

# ── 10-20 channel positions (for topographic reference) ──────
_CH_POSITIONS = {
    "Fp1":(-0.31, 0.95), "Fp2":( 0.31, 0.95),
    "F7" :(-0.71, 0.71), "F3" :(-0.40, 0.69),
    "Fz" :( 0.00, 0.72), "F4" :( 0.40, 0.69),
    "F8" :( 0.71, 0.71),
    "T7" :(-1.00, 0.00), "C3" :(-0.60, 0.00),
    "Cz" :( 0.00, 0.00), "C4" :( 0.60, 0.00),
    "T8" :( 1.00, 0.00),
    "P7" :(-0.71,-0.71), "P3" :(-0.40,-0.69),
    "Pz" :( 0.00,-0.72), "P4" :( 0.40,-0.69),
    "P8" :( 0.71,-0.71),
    "O1" :(-0.31,-0.95), "Oz" :( 0.00,-0.98),
    "O2" :( 0.31,-0.95),
    "FC1":(-0.20, 0.40), "FC2":( 0.20, 0.40),
    "FC5":(-0.72, 0.42), "FC6":( 0.72, 0.42),
    "CP1":(-0.20,-0.40), "CP2":( 0.20,-0.40),
    "CP5":(-0.72,-0.42), "CP6":( 0.72,-0.42),
    "AF3":(-0.20, 0.83), "AF4":( 0.20, 0.83),
    "PO3":(-0.40,-0.83), "PO4":( 0.40,-0.83),
    "F1" :(-0.20, 0.69), "F2" :( 0.20, 0.69),
}

STANDARD_20 = [
    "Fp1","Fp2","F3","F4","C3","C4",
    "P3","P4","O1","O2","F7","F8",
    "T7","T8","P7","P8","Fz","Cz",
    "Pz","Oz"
]

# ============================================================
# LOAD EEG EPOCHS
# ============================================================

sp("\n" + "=" * 56)
sp("  RD-GANet Phase 8A -- EEG Connectivity Graph Construction")
sp("=" * 56)

EEG_PATH = None
for cand in _CANDIDATES:
    if cand.exists():
        EEG_PATH = cand
        break

if EEG_PATH is None:
    sp("\n[ERROR] EEG epoch file not found. Tried:")
    for c in _CANDIDATES:
        sp(f"  {c}")
    sp("\nPlease run Phase 3 first to generate epoch data.")
    sys.exit(1)

sp(f"\n  Loading: {EEG_PATH}")
eeg_raw = np.load(EEG_PATH, allow_pickle=True)

# Handle object arrays (ragged) and regular 3-D arrays
if eeg_raw.ndim == 1 or eeg_raw.dtype == object:
    eeg_list = list(eeg_raw)
else:
    eeg_list = [eeg_raw[i] for i in range(len(eeg_raw))]

N = len(eeg_list)
T = eeg_list[0].shape[0]
C = eeg_list[0].shape[1]

sp(f"\n  EEG epochs  : {N}")
sp(f"  Samples/ep  : {T}  ({T/EEG_SR:.1f} s)")
sp(f"  Channels    : {C}")

# Load labels
LABELS = None
for cand in LABELS_CANDIDATES:
    if cand.exists():
        LABELS = np.load(cand, allow_pickle=True)
        break

# Assign channel names
if C <= 20:
    CHANNEL_NAMES = STANDARD_20[:C]
elif C <= 32:
    CHANNEL_NAMES = (STANDARD_20 + ["FC1","FC2","FC5","FC6","CP1","CP2",
                                     "CP5","CP6","AF3","AF4","PO3","PO4"])[:C]
else:
    CHANNEL_NAMES = [f"Ch{i:02d}" for i in range(C)]

sp(f"\n  Channels    : {CHANNEL_NAMES}")

WINDOW_SAMPLES = T // N_WINDOWS
sp(f"  N_WINDOWS   : {N_WINDOWS}  ({WINDOW_SAMPLES} samples each, "
   f"{WINDOW_SAMPLES/EEG_SR*1000:.0f} ms)")


# ============================================================
# CONNECTIVITY FUNCTIONS
# ============================================================

def pearson_matrix(epoch: np.ndarray) -> np.ndarray:
    """
    Pearson correlation adjacency matrix.

    A_ij = corr(x_i, x_j)

    Parameters
    ----------
    epoch : (T, C)

    Returns
    -------
    adj : (C, C)  float32 in [-1, 1]
    """
    return np.corrcoef(epoch.T).astype(np.float32)


def coherence_matrix(epoch: np.ndarray, fs: float = 300.0) -> np.ndarray:
    """
    Mean spectral coherence adjacency matrix.

    C_ij = mean_f [ |P_ij(f)|^2 / (P_ii(f) * P_jj(f)) ]

    Parameters
    ----------
    epoch : (T, C)
    fs    : sampling frequency

    Returns
    -------
    adj : (C, C)  float32 in [0, 1]
    """
    C_size = epoch.shape[1]
    adj = np.eye(C_size, dtype=np.float32)
    nperseg = min(256, epoch.shape[0] // 2)

    for i in range(C_size):
        for j in range(i + 1, C_size):
            _, cxy = coherence(epoch[:, i], epoch[:, j], fs=fs, nperseg=nperseg)
            val = float(np.mean(cxy))
            adj[i, j] = val
            adj[j, i] = val

    return adj


def plv_matrix(epoch: np.ndarray) -> np.ndarray:
    """
    Phase Locking Value adjacency matrix.

    PLV_ij = |1/T * sum_t e^{i(phi_i(t) - phi_j(t))}|

    Parameters
    ----------
    epoch : (T, C)

    Returns
    -------
    adj : (C, C)  float32 in [0, 1]
    """
    phases = np.angle(hilbert(epoch, axis=0))       # (T, C)
    C_size = epoch.shape[1]
    adj    = np.eye(C_size, dtype=np.float32)

    for i in range(C_size):
        for j in range(i + 1, C_size):
            delta = phases[:, i] - phases[:, j]
            val   = float(np.abs(np.mean(np.exp(1j * delta))))
            adj[i, j] = val
            adj[j, i] = val

    return adj


def threshold_adjacency(adj: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Apply absolute threshold and preserve self-loops."""
    binary = (np.abs(adj) >= threshold).astype(np.float32)
    np.fill_diagonal(binary, 1.0)
    return binary


# ============================================================
# GRAPH METRICS
# ============================================================

def compute_graph_metrics(adj: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """
    Compute graph-theoretic metrics from a weighted adjacency matrix.

    Parameters
    ----------
    adj : (C, C)  raw float adjacency

    Returns
    -------
    dict of scalar metrics
    """
    if not _HAS_NX:
        return {
            "mean_degree": float(np.sum(np.abs(adj) > threshold, axis=1).mean() - 1),
            "density":     float(np.sum(np.abs(adj) > threshold) / (adj.shape[0] ** 2)),
        }

    C_size = adj.shape[0]
    G = nx.Graph()
    for i in range(C_size):
        G.add_node(i)

    for i in range(C_size):
        for j in range(i + 1, C_size):
            w = float(adj[i, j])
            if abs(w) >= threshold:
                G.add_edge(i, j, weight=abs(w))

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    degrees = [d for _, d in G.degree()]

    clustering = nx.average_clustering(G, weight="weight") if n_edges > 0 else 0.0
    density    = nx.density(G)

    betweenness = list(nx.betweenness_centrality(G).values())
    mean_between = float(np.mean(betweenness)) if betweenness else 0.0

    # Global efficiency
    if n_nodes > 1 and nx.is_connected(G):
        spl = nx.average_shortest_path_length(G)
        global_eff = 1.0 / spl if spl > 0 else 0.0
    else:
        try:
            # Use largest connected component
            lcc = max(nx.connected_components(G), key=len)
            H   = G.subgraph(lcc)
            spl = nx.average_shortest_path_length(H) if len(lcc) > 1 else 1.0
            global_eff = 1.0 / spl if spl > 0 else 0.0
        except Exception:
            global_eff = 0.0

    return {
        "mean_degree"   : float(np.mean(degrees)) if degrees else 0.0,
        "std_degree"    : float(np.std(degrees))  if degrees else 0.0,
        "clustering"    : float(clustering),
        "density"       : float(density),
        "betweenness"   : float(mean_between),
        "global_eff"    : float(global_eff),
        "n_edges"       : int(n_edges),
    }


# ============================================================
# EDGE INDEX FOR PYTORCH GEOMETRIC / GAT
# ============================================================

def adjacency_to_edge_index(adj: np.ndarray, threshold: float = THRESHOLD):
    """
    Convert adjacency matrix to COO edge_index + edge_attr.

    Parameters
    ----------
    adj : (C, C)  float adjacency

    Returns
    -------
    edge_index : (2, E)  int64 source/destination arrays
    edge_attr  : (E,)    float32 edge weights
    """
    rows, cols = np.where(np.abs(adj) >= threshold)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_attr  = np.abs(adj[rows, cols]).astype(np.float32)
    return edge_index, edge_attr


# ============================================================
# MAIN COMPUTATION LOOP
# ============================================================

sp("\n" + "─" * 56)
sp("  Computing connectivity matrices...")
sp("─" * 56)

# Storage arrays
all_adj_pearson   = np.zeros((N, C, C), dtype=np.float32)
all_adj_coherence = np.zeros((N, C, C), dtype=np.float32)
all_adj_plv       = np.zeros((N, C, C), dtype=np.float32)

# Dynamic windowed Pearson adjacency
all_adj_windowed  = np.zeros((N, N_WINDOWS, C, C), dtype=np.float32)

# Edge indices (one per epoch)
edge_indices = np.empty(N, dtype=object)
edge_attrs   = np.empty(N, dtype=object)

# Graph metrics
metrics_rows = []

for idx in _progress(range(N), desc="Connectivity"):
    epoch = eeg_epoch = np.array(eeg_list[idx], dtype=np.float32)  # (T, C)

    # ── Whole-epoch connectivity ────────────────────────────
    adj_p = pearson_matrix(epoch)
    adj_c = coherence_matrix(epoch, fs=EEG_SR)
    adj_v = plv_matrix(epoch)

    all_adj_pearson[idx]   = adj_p
    all_adj_coherence[idx] = adj_c
    all_adj_plv[idx]       = adj_v

    # ── Windowed Pearson connectivity ─────────────────────
    for w in range(N_WINDOWS):
        s = w * WINDOW_SAMPLES
        e = s + WINDOW_SAMPLES
        win = epoch[s:e]
        all_adj_windowed[idx, w] = pearson_matrix(win)

    # ── Edge index from whole-epoch adjacency ─────────────
    ei, ea = adjacency_to_edge_index(np.abs(adj_p), THRESHOLD)
    edge_indices[idx] = ei
    edge_attrs[idx]   = ea

    # ── Graph metrics ─────────────────────────────────────
    m = compute_graph_metrics(adj_p, THRESHOLD)
    m["epoch"] = idx
    if LABELS is not None:
        m["label"] = str(LABELS[idx])
    metrics_rows.append(m)

sp("  Done.")

# ============================================================
# SAVE ARRAYS
# ============================================================

sp("\n  Saving outputs...")

np.save(OUT_DIR / "adjacency.npy",           all_adj_pearson)
np.save(OUT_DIR / "adjacency_windowed.npy",  all_adj_windowed)
np.save(OUT_DIR / "adjacency_coherence.npy", all_adj_coherence)
np.save(OUT_DIR / "adjacency_plv.npy",       all_adj_plv)
np.save(OUT_DIR / "edge_indices.npy",        edge_indices, allow_pickle=True)
np.save(OUT_DIR / "edge_attrs.npy",          edge_attrs,   allow_pickle=True)

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT_DIR / "graph_metrics.csv", index=False)

sp(f"  adjacency.npy           shape: {all_adj_pearson.shape}")
sp(f"  adjacency_windowed.npy  shape: {all_adj_windowed.shape}")
sp(f"  adjacency_coherence.npy shape: {all_adj_coherence.shape}")
sp(f"  adjacency_plv.npy       shape: {all_adj_plv.shape}")
sp(f"  edge_indices.npy        N={N} object arrays")
sp(f"  graph_metrics.csv       {len(metrics_df)} rows x {len(metrics_df.columns)} cols")


# ============================================================
# VISUALISATIONS
# ============================================================

sp("\n" + "─" * 56)
sp("  Generating visualisations...")
sp("─" * 56)

PALETTE = sns.color_palette("Blues", as_cmap=False)

# ── 1. Mean Pearson connectivity heatmap ─────────────────────
mean_adj = np.mean(np.abs(all_adj_pearson), axis=0)

fig, ax = plt.subplots(figsize=(max(8, C * 0.4), max(7, C * 0.38)))
mask_diag = np.eye(C, dtype=bool)
sns.heatmap(
    mean_adj,
    mask          = mask_diag,
    cmap          = "coolwarm",
    center        = 0,
    vmin          = 0, vmax = 1,
    xticklabels   = CHANNEL_NAMES,
    yticklabels   = CHANNEL_NAMES,
    linewidths    = 0.2,
    ax            = ax,
    cbar_kws      = {"label": "Mean |Pearson r|", "shrink": 0.8},
)
ax.set_title("Mean EEG Functional Connectivity — Pearson |r|",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=7)
plt.tight_layout()
fig.savefig(OUT_DIR / "connectivity_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)
sp("  [OK] connectivity_heatmap.png")

# ── 2. NetworkX connectivity graph ───────────────────────────
if _HAS_NX:
    G_vis = nx.Graph()
    for i, ch in enumerate(CHANNEL_NAMES):
        G_vis.add_node(i, label=ch)

    edge_wts = []
    for i in range(C):
        for j in range(i + 1, C):
            w = float(mean_adj[i, j])
            if w >= THRESHOLD_VIS:
                G_vis.add_edge(i, j, weight=w)
                edge_wts.append(w)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Prefer anatomical positions; fall back to circular
    node_pos = {}
    for i, ch in enumerate(CHANNEL_NAMES):
        if ch in _CH_POSITIONS:
            node_pos[i] = _CH_POSITIONS[ch]
    if len(node_pos) < C:
        node_pos = nx.circular_layout(G_vis)

    node_deg = np.array([G_vis.degree(n) for n in range(C)], dtype=float)
    node_clr = plt.cm.RdYlBu_r(node_deg / (node_deg.max() + 1e-8))

    wt_arr  = np.array(edge_wts) if edge_wts else np.array([1.0])
    wt_norm = (wt_arr - wt_arr.min()) / (wt_arr.max() - wt_arr.min() + 1e-8)
    widths  = 1.0 + 4.0 * wt_norm

    nx.draw_networkx_nodes(G_vis, node_pos, ax=ax,
                           node_color=node_clr, node_size=700, alpha=0.9)
    nx.draw_networkx_labels(G_vis, node_pos, ax=ax,
                            labels={i: CHANNEL_NAMES[i] for i in range(C)},
                            font_size=8, font_weight="bold")
    if edge_wts:
        nx.draw_networkx_edges(G_vis, node_pos, ax=ax,
                               width=widths, alpha=0.55, edge_color="steelblue")

    ax.set_title(
        f"EEG Functional Connectivity Graph (threshold = {THRESHOLD_VIS})\n"
        f"Node colour = degree  |  Edge width = connectivity strength",
        fontsize=11, fontweight="bold"
    )
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r,
                                norm=plt.Normalize(0, int(node_deg.max())))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, label="Node degree")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "connectivity_graph.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    sp("  [OK] connectivity_graph.png")
else:
    sp("  [SKIP] connectivity_graph.png (networkx not installed)")

# ── 3. Dynamic connectivity across windows (first epoch) ─────
fig = plt.figure(figsize=(N_WINDOWS * 2.2, 3.5))
gs  = gridspec.GridSpec(1, N_WINDOWS, wspace=0.08)

vmin, vmax = 0.0, 1.0
for w in range(N_WINDOWS):
    ax = fig.add_subplot(gs[0, w])
    im = ax.imshow(
        np.abs(all_adj_windowed[0, w]),
        cmap="coolwarm", aspect="auto",
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(f"W{w+1}\n{w*WINDOW_SAMPLES/EEG_SR:.2f}s",
                 fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

cb_ax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
plt.colorbar(im, cax=cb_ax, label="|r|")
fig.suptitle(
    f"Dynamic EEG Connectivity — Epoch 0 ({N_WINDOWS} windows x "
    f"{WINDOW_SAMPLES/EEG_SR*1000:.0f} ms)",
    fontsize=11, fontweight="bold", y=1.02,
)
fig.savefig(OUT_DIR / "dynamic_connectivity.png", dpi=300, bbox_inches="tight")
plt.close(fig)
sp("  [OK] dynamic_connectivity.png")

# ── 4. Graph metrics boxplots ─────────────────────────────────
metric_cols = ["mean_degree", "clustering", "density", "betweenness", "global_eff"]
metric_cols = [c for c in metric_cols if c in metrics_df.columns]

if metric_cols:
    fig, axes = plt.subplots(1, len(metric_cols),
                             figsize=(len(metric_cols) * 2.8, 4))
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, metric_cols):
        vals = metrics_df[col].dropna()
        ax.boxplot(vals, patch_artist=True,
                   boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                   medianprops=dict(color="white", lw=2),
                   whiskerprops=dict(color="#4C72B0"),
                   capprops=dict(color="#4C72B0"),
                   flierprops=dict(marker="o", markersize=3, alpha=0.4))
        ax.set_title(col.replace("_", "\n"), fontsize=9, fontweight="bold")
        ax.set_xticklabels([])
        ax.text(1, vals.median(), f" {vals.median():.3f}",
                va="center", fontsize=8, color="navy")

    fig.suptitle("Graph-Theoretic Metrics across Epochs",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "graph_metrics_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    sp("  [OK] graph_metrics_boxplot.png")

# ── 5. Connectivity method comparison ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
method_data = [
    (np.mean(np.abs(all_adj_pearson),   axis=0), "Pearson |r|",  "coolwarm"),
    (np.mean(all_adj_coherence,          axis=0), "Coherence",    "viridis"),
    (np.mean(all_adj_plv,                axis=0), "PLV",          "plasma"),
]
for ax, (data, title, cmap) in zip(axes, method_data):
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    if C <= 20:
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=6)
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8, label="Strength")

fig.suptitle("EEG Connectivity Methods — Mean across Epochs",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "method_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
sp("  [OK] method_comparison.png")

# ── 6. Degree distribution ────────────────────────────────────
if "mean_degree" in metrics_df.columns:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(metrics_df["mean_degree"].dropna(), bins=20,
            color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(metrics_df["mean_degree"].mean(), color="#C44E52", lw=2,
               ls="--", label=f"Mean = {metrics_df['mean_degree'].mean():.2f}")
    ax.set_xlabel("Mean Node Degree")
    ax.set_ylabel("Number of Epochs")
    ax.set_title("Electrode Degree Distribution (Pearson, "
                 f"threshold = {THRESHOLD})", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "degree_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    sp("  [OK] degree_distribution.png")

# ── 7. Connectivity temporal evolution ───────────────────────
#    Show mean |r| across all windows and epochs
mean_window_conn = np.abs(all_adj_windowed).mean(axis=(0, 2, 3))  # (W,)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(range(1, N_WINDOWS + 1), mean_window_conn, "o-",
        lw=2, ms=8, color="#4C72B0")
ax.fill_between(range(1, N_WINDOWS + 1),
                mean_window_conn - all_adj_windowed.mean(axis=0).std(axis=(1, 2)),
                mean_window_conn + all_adj_windowed.mean(axis=0).std(axis=(1, 2)),
                alpha=0.15, color="#4C72B0")
ax.set_xlabel("Temporal Window (0.5 s each)")
ax.set_ylabel("Mean |Pearson r|  (all electrode pairs)")
ax.set_title("Dynamic Connectivity Strength over Time",
             fontweight="bold")
ax.set_xticks(range(1, N_WINDOWS + 1))
ax.set_xticklabels([f"{w * WINDOW_SAMPLES / EEG_SR:.1f}s"
                    for w in range(N_WINDOWS)], rotation=45)
ax.set_ylim(0, 1.05)
plt.tight_layout()
fig.savefig(OUT_DIR / "temporal_evolution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
sp("  [OK] temporal_evolution.png")


# ============================================================
# SUMMARY
# ============================================================

sp("\n" + "=" * 56)
sp("  Phase 8A COMPLETE -- EEG Connectivity Graph Construction")
sp("=" * 56)

sp(f"\n  Input")
sp(f"    Epochs    : {N}")
sp(f"    Channels  : {C}")
sp(f"    Duration  : {T/EEG_SR:.1f} s  ({T} samples @ {EEG_SR} Hz)")

sp(f"\n  Connectivity")
sp(f"    Methods   : Pearson, Coherence, PLV")
sp(f"    Threshold : |r| >= {THRESHOLD} (GAT edges)")
sp(f"    N_WINDOWS : {N_WINDOWS}  ({WINDOW_SAMPLES} samples / {WINDOW_SAMPLES/EEG_SR*1000:.0f} ms)")

mean_edges = np.mean([edge_indices[i].shape[1] for i in range(N)
                      if edge_indices[i] is not None])
sp(f"    Mean edges/epoch : {mean_edges:.1f}")

if "density" in metrics_df.columns:
    sp(f"    Mean density     : {metrics_df['density'].mean():.3f}")
if "clustering" in metrics_df.columns:
    sp(f"    Mean clustering  : {metrics_df['clustering'].mean():.3f}")
if "global_eff" in metrics_df.columns:
    sp(f"    Mean global eff  : {metrics_df['global_eff'].mean():.3f}")

sp(f"\n  Saved to: {OUT_DIR}/")
sp(f"    adjacency.npy           {all_adj_pearson.shape}")
sp(f"    adjacency_windowed.npy  {all_adj_windowed.shape}")
sp(f"    adjacency_coherence.npy {all_adj_coherence.shape}")
sp(f"    adjacency_plv.npy       {all_adj_plv.shape}")
sp(f"    edge_indices.npy        [{N} variable-length arrays]")
sp(f"    graph_metrics.csv       {metrics_df.shape}")

sp(f"\n  Visualisations:")
for fname in [
    "connectivity_heatmap.png",
    "connectivity_graph.png",
    "dynamic_connectivity.png",
    "graph_metrics_boxplot.png",
    "method_comparison.png",
    "degree_distribution.png",
    "temporal_evolution.png",
]:
    exists = "[OK]" if (OUT_DIR / fname).exists() else "[--]"
    sp(f"    {exists} {fname}")

sp(f"\n  Next: Run Phase 8B (GAT training) using the saved arrays.")
sp(f"        python main.py\n")
