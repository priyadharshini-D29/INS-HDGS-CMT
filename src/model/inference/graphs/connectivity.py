"""
================================================================
NEUMA Phase 8 — EEG Functional Connectivity
================================================================
Implements adjacency matrix computation for dynamic electrode
graphs using three complementary measures:

  A_ij = corr(x_i, x_j)          Pearson correlation
  A_ij = mean(Cxy(x_i, x_j))     Coherence (frequency domain)
  A_ij = |mean(e^{iΔφ_ij})|      Phase Locking Value (PLV)

All matrices are symmetric and normalised to [0, 1].
================================================================
"""

from typing import Optional, Tuple

import numpy as np
from scipy.signal import coherence, hilbert, butter, filtfilt


# ── Canonical EEG bands (must match config.settings.EEG_BANDS) ──────────────
CANONICAL_BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _bandpass(epoch: np.ndarray, fs: float, band: Tuple[float, float],
              order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass filter applied along the time axis.

    A broadband (e.g. 1-45 Hz) signal has no single well-defined
    instantaneous phase — the Hilbert transform's phase is only
    physiologically meaningful once the signal has been narrowed to a
    band with a clear centre frequency (Le Van Quyen et al., 2001).
    This is applied before PLV so the resulting phase-locking values
    reflect within-band synchrony rather than broadband artefacts.
    """
    lo, hi = band
    nyq    = fs / 2.0
    lo_n   = max(lo / nyq, 1e-4)
    hi_n   = min(hi / nyq, 0.999)
    b, a   = butter(order, [lo_n, hi_n], btype="band")
    # filtfilt needs enough samples relative to filter order; pad-aware default
    padlen = 3 * (max(len(a), len(b)) - 1)
    if epoch.shape[0] <= padlen:
        # Too few samples for zero-phase filtering at this order — fall back
        # to a lower-order filter rather than raising.
        order  = max(1, epoch.shape[0] // 4)
        b, a   = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, epoch, axis=0).astype(np.float32)


# ── Input Validation Helper ──────────────────────────────────────────────────

def _validate_epoch(epoch, caller: str):
    """
    Validate that *epoch* is a 2-D (T, C) float array with T > 1 and C > 1.

    Raises TypeError / ValueError with a descriptive message so the caller
    can decide whether to skip or crash.
    """
    if not isinstance(epoch, np.ndarray):
        raise TypeError(
            f"[{caller}] Expected ndarray, got {type(epoch).__name__}"
        )
    if epoch.ndim != 2:
        raise ValueError(
            f"[{caller}] Expected 2-D array (T, C), got ndim={epoch.ndim} "
            f"shape={epoch.shape}"
        )
    T, C = epoch.shape
    if T <= 1:
        raise ValueError(
            f"[{caller}] Too few time samples: shape={epoch.shape}"
        )
    if C <= 1:
        raise ValueError(
            f"[{caller}] Too few channels: shape={epoch.shape}"
        )


def _clean_epoch(epoch: np.ndarray) -> np.ndarray:
    """Cast to float32 and replace NaN / ±Inf with 0."""
    return np.nan_to_num(
        np.asarray(epoch, dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


# ── Pearson Correlation ──────────────────────────────────────────────────────

def pearson_connectivity(
    epoch       : np.ndarray,
    eps         : float = 1e-8,
    channel_names = None,
) -> np.ndarray:
    """
    Compute electrode-wise Pearson correlation adjacency matrix.

    A_ij = corr(x_i, x_j)

    Zero-variance channels (e.g. harmonization-filled zero columns) are
    handled without RuntimeWarning: their rows/columns are set to 0 and
    the diagonal is restored to 1 so self-loops are preserved for the GAT.

    Parameters
    ----------
    epoch         : (T, C)  — time × channels
    eps           : variance threshold below which a channel is "invalid"
    channel_names : optional list[str] of length C for logging

    Returns
    -------
    adj : (C, C)  values in [-1, 1]; diagonal always 1; no NaN or Inf
    """
    _validate_epoch(epoch, "Pearson")
    epoch = _clean_epoch(epoch)
    T, C = epoch.shape

    # ── Detect zero-variance channels ────────────────────────────────────────
    std      = epoch.std(axis=0)           # (C,)
    bad_mask = std < eps                   # True for harmonized-zero channels

    # zero-variance channels handled silently below (self-loop=1, corr=0)

    # ── Numerically stable correlation ───────────────────────────────────────
    # np.errstate suppresses RuntimeWarning from 0/0 on zero-variance channels.
    # nan_to_num maps the resulting NaN values to 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        adj = np.corrcoef(epoch.T)

    adj = np.nan_to_num(
        adj.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    # ── Restore self-loops (diagonal must be 1 for all nodes) ────────────────
    # nan_to_num zeroes NaN diagonal entries for zero-variance channels;
    # fill_diagonal restores them so every node has a self-loop.
    np.fill_diagonal(adj, 1.0)

    assert not np.isnan(adj).any(),  "[Pearson] NaN in output after fix"
    assert not np.isinf(adj).any(),  "[Pearson] Inf in output after fix"

    return adj


# ── Spectral Coherence ───────────────────────────────────────────────────────

def coherence_connectivity(
    epoch : np.ndarray,
    fs    : float = 300.0,
    eps   : float = 1e-8,
) -> np.ndarray:
    """
    Compute mean spectral coherence between every pair of EEG channels.

    Zero-variance channels (e.g. harmonization-filled zeros) are skipped:
    their off-diagonal entries remain 0 and their self-loop is set to 1.

    Parameters
    ----------
    epoch : (T, C)
    fs    : sampling frequency (Hz)
    eps   : variance threshold below which a channel is treated as invalid

    Returns
    -------
    adj : (C, C)  values in [0, 1]; diagonal always 1; no NaN or Inf
    """
    _validate_epoch(epoch, "Coherence")
    epoch = _clean_epoch(epoch)
    C = epoch.shape[1]

    # Pre-compute valid channel mask (skip zero-variance channels entirely)
    std      = epoch.std(axis=0)
    valid    = std >= eps           # (C,) bool
    adj      = np.zeros((C, C), dtype=np.float32)
    np.fill_diagonal(adj, 1.0)     # self-loops for all nodes

    for i in range(C):
        if not valid[i]:
            continue
        for j in range(i + 1, C):
            if not valid[j]:
                continue
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    _, cxy = coherence(
                        epoch[:, i], epoch[:, j],
                        fs=fs, nperseg=min(256, len(epoch)),
                    )
                val = float(np.nan_to_num(np.mean(cxy), nan=0.0))
            except Exception:
                val = 0.0
            adj[i, j] = val
            adj[j, i] = val

    assert not np.isnan(adj).any(),  "[Coherence] NaN in output after fix"
    assert not np.isinf(adj).any(),  "[Coherence] Inf in output after fix"
    return adj


# ── Phase Locking Value ──────────────────────────────────────────────────────

def plv_connectivity(
    epoch : np.ndarray,
    eps   : float = 1e-8,
    band  : Optional[Tuple[float, float]] = None,
    fs    : float = 300.0,
) -> np.ndarray:
    """
    Compute Phase Locking Value matrix.

    PLV_ij = |1/T * sum_t e^{i(φ_i(t) - φ_j(t))}|

    Zero-variance channels (e.g. harmonization-filled zeros) produce a
    constant-zero analytic signal whose Hilbert phase is 0 everywhere.
    This gives a non-NaN but meaningless PLV against any real channel
    (it equals the PLV of the real channel against a zero-phase reference).
    These pairs are explicitly zeroed; self-loops are preserved.

    Parameters
    ----------
    epoch : (T, C)
    eps   : variance threshold below which a channel is treated as invalid
    band  : optional (f_lo, f_hi) in Hz. When given, the signal is
            band-pass filtered *before* the Hilbert transform, so the
            instantaneous phase is computed on a narrowband signal (the
            regime in which it is physiologically well-defined) rather
            than on the raw 1-45 Hz broadband recording. None reproduces
            the original broadband behaviour for backward compatibility.
    fs    : sampling frequency (Hz), only used when band is not None

    Returns
    -------
    adj : (C, C)  values in [0, 1]; diagonal always 1; no NaN or Inf
    """
    _validate_epoch(epoch, "PLV")
    epoch = _clean_epoch(epoch)
    C = epoch.shape[1]

    # Pre-compute valid channel mask
    std   = epoch.std(axis=0)
    valid = std >= eps    # (C,) bool

    if band is not None:
        epoch = _bandpass(epoch, fs, band)

    with np.errstate(divide="ignore", invalid="ignore"):
        phases = np.angle(hilbert(epoch, axis=0))   # (T, C)

    adj = np.zeros((C, C), dtype=np.float32)
    np.fill_diagonal(adj, 1.0)

    for i in range(C):
        if not valid[i]:
            continue
        for j in range(i + 1, C):
            if not valid[j]:
                continue
            delta_phi = phases[:, i] - phases[:, j]
            val = float(np.abs(np.mean(np.exp(1j * delta_phi))))
            val = float(np.nan_to_num(val, nan=0.0))
            adj[i, j] = val
            adj[j, i] = val

    assert not np.isnan(adj).any(),  "[PLV] NaN in output after fix"
    assert not np.isinf(adj).any(),  "[PLV] Inf in output after fix"
    return adj


def plv_connectivity_bands(
    epoch : np.ndarray,
    fs    : float = 300.0,
    bands : Optional[dict] = None,
    eps   : float = 1e-8,
) -> dict:
    """
    Compute narrowband PLV matrices for each canonical EEG band.

    Useful as a robustness check against the broadband-Hilbert PLV
    (band=None): if downstream classification accuracy is qualitatively
    similar whether PLV is computed broadband or per-band, the broadband
    shortcut is at least empirically defensible for this dataset.

    Returns
    -------
    dict[band_name -> (C, C) adjacency]
    """
    bands = bands or CANONICAL_BANDS
    return {
        name: plv_connectivity(epoch, eps=eps, band=rng, fs=fs)
        for name, rng in bands.items()
    }


def parse_conn_method(method: str) -> Tuple[str, Optional[Tuple[float, float]]]:
    """
    Parse a connectivity method string into (base_method, band).

    Supports "plv_<band>" (e.g. "plv_theta") for narrowband PLV in
    addition to the plain "pearson" | "coherence" | "plv" methods.
    Unknown "plv_<x>" band names raise ValueError to fail fast rather
    than silently falling back to broadband.
    """
    if method.startswith("plv_"):
        band_name = method[len("plv_"):]
        if band_name not in CANONICAL_BANDS:
            raise ValueError(
                f"Unknown PLV band {band_name!r}; expected one of "
                f"{list(CANONICAL_BANDS)}"
            )
        return "plv", CANONICAL_BANDS[band_name]
    return method, None


# ── Unified Interface ────────────────────────────────────────────────────────

def compute_adjacency(
    epoch: np.ndarray,
    method: str = "pearson",
    fs: float = 300.0,
    threshold: float = 0.30,
    abs_values: bool = True,
) -> np.ndarray:
    """
    Compute thresholded adjacency matrix from an EEG window.

    The returned matrix is binary (0/1) after threshold application,
    suitable for use as a graph edge mask in the GAT.

    Parameters
    ----------
    epoch     : (T, C)  raw EEG window
    method    : "pearson" | "coherence" | "plv" | "plv_<band>"
                (band in {delta, theta, alpha, beta, gamma})
    fs        : sampling frequency
    threshold : edge inclusion threshold on |A_ij|
    abs_values: if True, take absolute values before thresholding

    Returns
    -------
    adj : (C, C)  binary adjacency (self-loops included on diagonal)
    """
    _validate_epoch(epoch, f"compute_adjacency[{method}]")
    base_method, band = parse_conn_method(method)

    if base_method == "pearson":
        raw = pearson_connectivity(epoch)
    elif base_method == "coherence":
        raw = coherence_connectivity(epoch, fs=fs)
    elif base_method == "plv":
        raw = plv_connectivity(epoch, band=band, fs=fs)
    else:
        raise ValueError(f"Unknown connectivity method: {method!r}")

    raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)

    if abs_values:
        raw = np.abs(raw)

    adj = (raw >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    return adj


def compute_weighted_adjacency(
    epoch: np.ndarray,
    method: str = "pearson",
    fs: float = 300.0,
) -> np.ndarray:
    """
    Return weighted (non-thresholded) adjacency for visualisation.

    Returns
    -------
    adj : (C, C)  float32 values in [0, 1]
    """
    _validate_epoch(epoch, f"compute_weighted_adjacency[{method}]")
    base_method, band = parse_conn_method(method)

    if base_method == "pearson":
        raw = np.abs(pearson_connectivity(epoch))
    elif base_method == "coherence":
        raw = coherence_connectivity(epoch, fs=fs)
    elif base_method == "plv":
        raw = plv_connectivity(epoch, band=band, fs=fs)
    else:
        raise ValueError(f"Unknown connectivity method: {method!r}")

    raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(raw, 1.0)
    return raw
