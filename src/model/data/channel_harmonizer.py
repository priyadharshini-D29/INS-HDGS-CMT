"""
================================================================
NEUMA Phase 8 — EEG Channel Harmonizer
================================================================
Ensures every subject's EEG epoch reaches the canonical 19-channel
cortical montage before graph construction, so collate_fn can stack
tensors of identical shape across subjects.

Problem
-------
Bad-channel removal in Phase 2 produced (T, 20) arrays for S01
(channels [15, 16, 19, 23] removed) while all other subjects have
(T, 24) raw-montage arrays. The raw 24-channel hardware montage
also includes 5 non-cortical channels (X1, X2, X3 = aux/EOG,
A2 = mastoid reference, TRG = trigger pulse) that were previously
being treated as real electrodes — full graph nodes with band-power
features and correlation-based adjacency edges — for every subject
except S01 (whose bad_channels.json entry happens to zero-fill 4 of
the 5, but under the montage order documented below, not the wrong
names this module used to assume).

Fix
---
Two resolution strategies (tried in order) reconstruct the raw
24-channel layout, then a final step drops the 5 non-cortical
channels to produce the model-facing 19-channel montage:

  1. Name-based (precise, zero-pads missing channels by name)
     Requires a ``channel_names.npy`` alongside eeg_epochs.npy.

  2. Index-based (uses data_pipeline/03_preprocessing/metadata/bad_channels.json)
     Reconstructs raw-montage positions from the removed-index list;
     removed positions are zero-filled.

  3. Fast-path: if C_subj == RAW_MONTAGE_N, use as-is.

  4. Last resort: if neither source is available, pad with zeros
     at the END and print a loud warning.

  Every path above then drops the 5 non-cortical channels
  (see NON_CORTICAL_CHANNELS) before returning, so every caller
  always receives (T, 19) — see CANONICAL_CHANNELS below.

Raw hardware montage (24 channels)
-----------------------------------
P3, C3, F3, Fz, F4, C4, P4, Cz,
Pz, Fp1, Fp2, T3, T5, O1, O2, X3,
X2, F7, F8, X1, A2, T6, T4, TRG

This is the real DSI-24 recording montage (verified via pyxdf
against several subjects' raw XDF streams, and independently
confirmed against the hardcoded EEG_CHANNEL_NAMES list in
data_pipeline/03_preprocessing/visualization/plot_sync_overview.py).
Channel index → name mapping:
   0 P3    1 C3    2 F3    3 Fz    4 F4    5 C4
   6 P4    7 Cz    8 Pz    9 Fp1  10 Fp2  11 T3
  12 T5   13 O1   14 O2   15 X3   16 X2   17 F7
  18 F8   19 X1   20 A2   21 T6   22 T4   23 TRG

Of these, X1/X2/X3 (aux/EOG), A2 (mastoid reference), and TRG
(trigger pulse) are not cortical EEG and are dropped before the
model ever sees them — see NON_CORTICAL_CHANNELS / CANONICAL_CHANNELS.
================================================================
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Raw hardware montage ──────────────────────────────────────────────────────

RAW_MONTAGE_CHANNELS: List[str] = [
    "P3",  "C3",  "F3",  "Fz",  "F4",  "C4",  "P4",  "Cz",
    "Pz",  "Fp1", "Fp2", "T3",  "T5",  "O1",  "O2",  "X3",
    "X2",  "F7",  "F8",  "X1",  "A2",  "T6",  "T4",  "TRG",
]
RAW_MONTAGE_N: int = len(RAW_MONTAGE_CHANNELS)  # 24

# Non-cortical channels present in the raw montage: X1/X2/X3 = aux/EOG,
# A2 = mastoid reference, TRG = trigger pulse. These previously got full
# graph-node status (band power + adjacency edges) with no cortical meaning.
NON_CORTICAL_CHANNELS: List[str] = ["X1", "X2", "X3", "A2", "TRG"]

# ── Canonical (model-facing) montage: 19 real cortical channels ─────────────

CANONICAL_CHANNELS: List[str] = [
    ch for ch in RAW_MONTAGE_CHANNELS if ch not in NON_CORTICAL_CHANNELS
]
CANONICAL_N: int = len(CANONICAL_CHANNELS)  # 19

# Reverse lookup: name → raw-montage index (used to locate channels in the
# still-24-wide array before the non-cortical drop step is applied)
_RAW_CH_TO_IDX = {ch: i for i, ch in enumerate(RAW_MONTAGE_CHANNELS)}

# Boolean mask over the 24 raw-montage positions selecting the 19 kept
# (cortical) channels, in original relative order.
_KEEP_MASK = np.array(
    [ch not in NON_CORTICAL_CHANNELS for ch in RAW_MONTAGE_CHANNELS], dtype=bool
)

# Bump whenever the montage list or drop strategy changes; folded into the
# on-disk graph-cache key (see data/dataset.py::_graph_cache_key) so a stale
# pre-fix cache is never silently reused.
MONTAGE_SCHEMA_VERSION = "v2_19ch"

# ── Bad-channel manifest path (Phase 2 metadata) ─────────────────────────────
# parents[2] of this file (data/channel_harmonizer.py -> data/ -> model/ ->
# src/) is src/, which is the common ancestor of both src/model/ and
# src/data_pipeline/ -- still correct after the reorg, just the sibling
# folder name changed.

_BAD_CH_JSON = (
    Path(__file__).resolve().parents[2]
    / "data_pipeline" / "03_preprocessing" / "metadata" / "bad_channels.json"
)


@lru_cache(maxsize=None)
def _load_bad_channel_map() -> dict:
    """Load and cache the Phase 2 bad_channels.json manifest."""
    if not _BAD_CH_JSON.exists():
        return {}
    with _BAD_CH_JSON.open() as f:
        return json.load(f)


def get_bad_channel_indices(subject_id: str) -> List[int]:
    """
    Return the canonical channel indices removed for *subject_id* in Phase 2.

    Example
    -------
    >>> get_bad_channel_indices("S01")
    [15, 16, 19, 23]     # X3, X2, X1, TRG
    """
    bad_map = _load_bad_channel_map()
    return list(bad_map.get(subject_id, []))


# ── Core harmonization ────────────────────────────────────────────────────────

def _drop_non_cortical_channels(raw24: np.ndarray) -> np.ndarray:
    """
    (T, RAW_MONTAGE_N) raw hardware montage → (T, CANONICAL_N) real cortical
    channels, dropping X1/X2/X3 (aux/EOG), A2 (mastoid reference), and TRG
    (trigger pulse). Applied identically regardless of which upstream
    strategy reconstructed the raw-24 layout, so e.g. S01's real (non-zeroed)
    A2 column is dropped the same as everyone else's.
    """
    return raw24[:, _KEEP_MASK].astype(np.float32)


def harmonize_eeg_channels(
    eeg_epoch    : np.ndarray,
    subject_id   : str,
    channel_names: Optional[List[str]] = None,
    *,
    verbose      : bool = True,
) -> np.ndarray:
    """
    Project ``eeg_epoch`` from (T, C_subj) → (T, 19) canonical cortical layout.

    Internally reconstructs the 24-channel raw hardware montage (so
    collate_fn can stack tensors of identical shape across subjects), then
    drops the 5 non-cortical channels before returning.

    Parameters
    ----------
    eeg_epoch     : (T, C_subj)  raw EEG epoch for one trial
    subject_id    : canonical subject string, e.g. "S01"
    channel_names : optional list of channel names recorded for this subject
                    (length must equal C_subj).  If provided, strategy 1 is used.
    verbose       : print harmonization log lines

    Returns
    -------
    np.ndarray of shape (T, 19), dtype float32
    """
    T, C_subj = eeg_epoch.shape

    # ── Fast path: already raw-montage width ─────────────────────────────────
    if C_subj == RAW_MONTAGE_N:
        return _drop_non_cortical_channels(eeg_epoch.astype(np.float32))

    if C_subj > RAW_MONTAGE_N:
        raise ValueError(
            f"[ChannelHarmonizer] {subject_id}: C_subj={C_subj} > RAW_MONTAGE_N={RAW_MONTAGE_N}. "
            "Cannot harmonize — data wider than the raw hardware montage."
        )

    out = np.zeros((T, RAW_MONTAGE_N), dtype=np.float32)

    # ── Strategy 1: name-based mapping ───────────────────────────────────────
    if channel_names is not None:
        if len(channel_names) != C_subj:
            raise ValueError(
                f"[ChannelHarmonizer] {subject_id}: channel_names length "
                f"({len(channel_names)}) != C_subj ({C_subj})."
            )
        missing, placed = [], 0
        for in_idx, ch in enumerate(channel_names):
            if ch in _RAW_CH_TO_IDX:
                out[:, _RAW_CH_TO_IDX[ch]] = eeg_epoch[:, in_idx]
                placed += 1
            else:
                missing.append(ch)

        absent = [c for c in RAW_MONTAGE_CHANNELS if c not in channel_names]
        if verbose:
            print(f"[ChannelHarmonizer] {subject_id}  "
                  f"name-based  {C_subj}→{RAW_MONTAGE_N}  "
                  f"placed={placed}  zero-filled={absent}")
        return _drop_non_cortical_channels(out)

    # ── Strategy 2: index-based mapping via bad_channels.json ────────────────
    bad_idx = get_bad_channel_indices(subject_id)
    if bad_idx:
        good_idx = [i for i in range(RAW_MONTAGE_N) if i not in bad_idx]
        if len(good_idx) != C_subj:
            warnings.warn(
                f"[ChannelHarmonizer] {subject_id}: bad_channels.json says "
                f"{len(bad_idx)} bad channels ({len(good_idx)} good), "
                f"but array has {C_subj} columns. "
                "Falling back to zero-padding at end.",
                stacklevel=2,
            )
        else:
            for in_idx, canon_idx in enumerate(good_idx):
                out[:, canon_idx] = eeg_epoch[:, in_idx]

            absent_names = [RAW_MONTAGE_CHANNELS[i] for i in sorted(bad_idx)]
            if verbose:
                print(
                    f"\n[Channel Harmonization]\n"
                    f"  Subject   : {subject_id}\n"
                    f"  Strategy  : index-based (bad_channels.json)\n"
                    f"  C_subj    : {C_subj}  →  RAW_MONTAGE_N : {RAW_MONTAGE_N}\n"
                    f"  Missing   : {absent_names}\n"
                    f"  Filled with zeros.\n"
                )
            return _drop_non_cortical_channels(out)

    # ── Strategy 3 (last resort): zero-pad at end ────────────────────────────
    warnings.warn(
        f"[ChannelHarmonizer] {subject_id}: no channel name / index info "
        f"available for {C_subj}-channel array. Padding {RAW_MONTAGE_N - C_subj} "
        "zeros at the end. Results may be unreliable.",
        stacklevel=2,
    )
    out[:, :C_subj] = eeg_epoch.astype(np.float32)
    if verbose:
        print(
            f"\n[Channel Harmonization]\n"
            f"  Subject   : {subject_id}\n"
            f"  Strategy  : fallback zero-pad (last resort)\n"
            f"  C_subj    : {C_subj}  →  RAW_MONTAGE_N : {RAW_MONTAGE_N}\n"
            f"  WARNING   : unknown channel layout — zeros appended at end.\n"
        )
    return _drop_non_cortical_channels(out)


# ── Batch validation ──────────────────────────────────────────────────────────

def validate_channel_counts(eeg_lists_by_subject: dict) -> None:
    """
    Audit channel counts across subjects and print a harmonization report.

    Parameters
    ----------
    eeg_lists_by_subject : dict[subject_id, list[ndarray(T, C)]]
    """
    counts = {
        sid: epochs[0].shape[1]
        for sid, epochs in eeg_lists_by_subject.items()
        if epochs
    }
    unique = set(counts.values())

    print("\n[ChannelHarmonizer] Channel count audit:")
    for sid, n in sorted(counts.items()):
        flag = " ← MISMATCH" if n != CANONICAL_N else ""
        print(f"  {sid}: {n} channels{flag}")

    if len(unique) > 1:
        print(
            f"\n  Multiple channel counts detected: {unique}.\n"
            f"  Harmonizing all subjects to {CANONICAL_N} channels.\n"
        )
    else:
        print(f"\n  All subjects have {CANONICAL_N} channels. No harmonization needed.\n")


# ── Phase 3 companion: save channel names alongside epochs ────────────────────

def save_channel_names(
    epoch_dir     : str | Path,
    channel_names : List[str],
) -> None:
    """
    Save a channel_names.npy alongside eeg_epochs.npy in *epoch_dir*.

    Call this from Phase 3 preprocessing to enable name-based harmonization
    in Phase 8.  Saves a (C,) object array of channel name strings.
    """
    path = Path(epoch_dir) / "channel_names.npy"
    np.save(path, np.array(channel_names, dtype=object))
    print(f"[ChannelHarmonizer] Saved channel names → {path}")


def load_channel_names(epoch_dir: str | Path) -> Optional[List[str]]:
    """
    Load channel_names.npy from *epoch_dir* if it exists.

    Returns list of strings or None.
    """
    path = Path(epoch_dir) / "channel_names.npy"
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)
    return [str(c) for c in arr]
