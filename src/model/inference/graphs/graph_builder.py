"""
================================================================
NEUMA Phase 8 — Dynamic Graph Construction
================================================================
Converts per-window EEG data into graph representations:

  G_t = (V, A_t)

  V   = electrodes (nodes)
  A_t = time-varying adjacency (computed per window)

Node features: band-power vector per electrode (5 dimensions).
================================================================
"""

import numpy as np
from scipy.signal import welch

from .connectivity import compute_adjacency

# NumPy 2.x removed np.trapz(); use np.trapezoid() with backward-compat fallback
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ── Symmetric graph normalisation: D^{-1/2} A D^{-1/2} ─────────────────────

def _sym_normalize(adj: np.ndarray) -> np.ndarray:
    """
    Symmetric graph normalisation for a single (C, C) adjacency matrix.

    A_norm = D^{-1/2} A D^{-1/2}

    Diagonal self-loops are preserved at 1.0 and the result is guaranteed
    free of NaN / Inf.
    """
    deg          = adj.sum(axis=1)                          # (C,)
    deg_inv_sqrt = np.power(deg + 1e-6, -0.5)              # (C,)
    D            = np.diag(deg_inv_sqrt)
    adj_norm     = D @ adj @ D
    adj_norm     = np.nan_to_num(adj_norm, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(adj_norm, 1.0)
    return adj_norm.astype(np.float32)

# Optional PyTorch Geometric
try:
    from torch_geometric.data import Data as PyGData
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


# ── Band Power Extraction ────────────────────────────────────────────────────

_BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Pre-compute band boundary list once (avoids dict iteration overhead per call)
_BAND_RANGES = list(_BANDS.values())   # [(fmin, fmax), ...]
_N_BANDS     = len(_BAND_RANGES)


def _channel_band_powers(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Compute all 5 relative band powers for one channel with a single welch() call.

    Returns (5,) float32 array — [delta, theta, alpha, beta, gamma].
    Calling welch once and integrating each band range is 5× faster than
    calling welch separately per band.
    """
    try:
        nperseg = min(len(signal), 128)
        f, psd  = welch(signal, fs=fs, nperseg=nperseg)
        total   = _trapz(psd, f) + 1e-10
        out     = np.empty(_N_BANDS, dtype=np.float32)
        for b, (fmin, fmax) in enumerate(_BAND_RANGES):
            mask    = (f >= fmin) & (f <= fmax)
            out[b]  = float(_trapz(psd[mask], f[mask]) / total) if mask.any() else 0.0
        return out
    except Exception:
        return np.zeros(_N_BANDS, dtype=np.float32)


def compute_node_features(
    window: np.ndarray,
    fs: float = 300.0,
) -> np.ndarray:
    """
    Compute band-power node feature matrix for one EEG window.

    Parameters
    ----------
    window : (T_window, C)  — raw EEG window
    fs     : sampling frequency

    Returns
    -------
    X : (C, 5)  — [delta, theta, alpha, beta, gamma] power per channel

    One welch() call per channel (not per band) — 5× faster than the
    previous per-band loop.
    """
    C = window.shape[1]
    X = np.empty((C, _N_BANDS), dtype=np.float32)
    for c in range(C):
        X[c] = _channel_band_powers(window[:, c], fs)
    return X


# ── Edge Construction ────────────────────────────────────────────────────────

def adjacency_to_edge_index(adj: np.ndarray, threshold: float = 0.30):
    """
    Convert dense adjacency matrix to COO edge_index for PyG.

    Parameters
    ----------
    adj       : (C, C)  float or binary adjacency
    threshold : edge-inclusion cutoff

    Returns
    -------
    edge_index : (2, E)  numpy int64 array
    edge_attr  : (E,)    float32 weights
    """
    mask = np.abs(adj) >= threshold
    rows, cols = np.where(mask)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_attr  = adj[rows, cols].astype(np.float32)
    return edge_index, edge_attr


# ── Per-Epoch Graph Sequences ────────────────────────────────────────────────

def compute_epoch_graphs(
    eeg_epoch: np.ndarray,
    n_windows: int = 10,
    fs: float = 300.0,
    conn_method: str = "pearson",
    threshold: float = 0.30,
):
    """
    Decompose one EEG epoch into a sequence of dynamic graphs.

    Parameters
    ----------
    eeg_epoch  : (T, C)  full EEG epoch
    n_windows  : number of temporal windows
    fs         : sampling frequency
    conn_method: connectivity method
    threshold  : adjacency threshold

    Returns
    -------
    node_features : (W, C, 5)     band-power features per window
    adj_matrices  : (W, C, C)     binary adjacency per window
    weighted_adjs : (W, C, C)     float Pearson/coherence per window
    """
    # ── Epoch-level validation ───────────────────────────────────────────────
    eeg_epoch = np.asarray(eeg_epoch)
    if eeg_epoch.ndim != 2 or eeg_epoch.shape[0] < 2 or eeg_epoch.shape[1] < 2:
        raise ValueError(
            f"[GraphBuilder] eeg_epoch must be 2-D (T≥2, C≥2), "
            f"got shape={eeg_epoch.shape}"
        )
    eeg_epoch = np.nan_to_num(
        eeg_epoch.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    # ── Per-channel z-score normalisation + clip ─────────────────────────────
    # Applied per epoch (axis=0 = time) so each channel has zero mean and unit
    # variance before band-power and connectivity computation.
    ch_mean   = eeg_epoch.mean(axis=0, keepdims=True)           # (1, C)
    ch_std    = eeg_epoch.std(axis=0,  keepdims=True) + 1e-6    # (1, C)
    eeg_epoch = (eeg_epoch - ch_mean) / ch_std
    eeg_epoch = np.clip(eeg_epoch, -5.0, 5.0).astype(np.float32)
    assert not np.isnan(eeg_epoch).any(), "[GraphBuilder] NaN after z-score"
    assert not np.isinf(eeg_epoch).any(), "[GraphBuilder] Inf after z-score"

    T, C = eeg_epoch.shape
    w_size = T // n_windows

    node_features = np.zeros((n_windows, C, len(_BANDS)), dtype=np.float32)
    adj_matrices  = np.zeros((n_windows, C, C),           dtype=np.float32)
    weighted_adjs = np.zeros((n_windows, C, C),           dtype=np.float32)
    # Pre-fill diagonal (self-loops) so skipped windows still have valid adj
    for w in range(n_windows):
        np.fill_diagonal(adj_matrices[w],  1.0)
        np.fill_diagonal(weighted_adjs[w], 1.0)

    for w in range(n_windows):
        start = w * w_size
        end   = start + w_size
        window = eeg_epoch[start:end]

        # ── Window validation ────────────────────────────────────────────────
        window = np.asarray(window)

        if window.ndim != 2:
            print(f"[GraphBuilder] Window {w}: invalid ndim={window.ndim} — skipping")
            continue

        if window.shape[0] <= 1:
            print(f"[GraphBuilder] Window {w}: too few samples {window.shape} — skipping")
            continue

        if window.shape[1] <= 1:
            print(f"[GraphBuilder] Window {w}: too few channels {window.shape} — skipping")
            continue

        window = np.nan_to_num(
            window.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

        # ── Node features ────────────────────────────────────────────────────
        node_features[w] = compute_node_features(window, fs=fs)

        # ── Weighted adjacency ───────────────────────────────────────────────
        try:
            from .connectivity import parse_conn_method
            base_method, band = parse_conn_method(conn_method)
            if base_method == "pearson":
                from .connectivity import pearson_connectivity
                raw = np.abs(pearson_connectivity(window))
            elif base_method == "coherence":
                from .connectivity import coherence_connectivity
                raw = coherence_connectivity(window, fs=fs)
            else:
                from .connectivity import plv_connectivity
                raw = plv_connectivity(window, band=band, fs=fs)
        except Exception as e:
            print(f"[GraphBuilder] Window {w}: connectivity failed ({e}) — using zeros")
            raw = np.zeros((C, C), dtype=np.float32)

        if raw.shape != (C, C):
            print(
                f"[GraphBuilder] Window {w}: unexpected adj shape {raw.shape}, "
                f"expected ({C},{C}) — using zeros"
            )
            raw = np.zeros((C, C), dtype=np.float32)

        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        np.fill_diagonal(raw, 1.0)

        # ── Binary adjacency for GAT (threshold on raw values) ───────────────
        adj_matrices[w] = (raw >= threshold).astype(np.float32)
        np.fill_diagonal(adj_matrices[w], 1.0)

        # ── Symmetric normalisation of weighted adjacency ────────────────────
        # A_norm = D^{-1/2} A D^{-1/2}  (preserves diagonal = 1)
        weighted_adjs[w] = _sym_normalize(raw)

    return node_features, adj_matrices, weighted_adjs


def build_pyg_graph(node_features: np.ndarray, adj: np.ndarray, threshold: float = 0.30):
    """
    Build a PyTorch Geometric Data object for one window.

    Requires torch_geometric. If not installed, raises ImportError.
    """
    if not _HAS_PYG:
        raise ImportError(
            "torch_geometric is not installed. "
            "Install with: pip install torch_geometric"
        )

    import torch
    edge_index, edge_attr = adjacency_to_edge_index(adj, threshold)
    return PyGData(
        x          = torch.tensor(node_features, dtype=torch.float),
        edge_index = torch.tensor(edge_index,    dtype=torch.long),
        edge_attr  = torch.tensor(edge_attr,     dtype=torch.float),
    )
