"""
================================================================
INS-HDGS-CMT — Phase 8 Graph Dataset
Cognitive State Decoding (HIGH/LOW Engagement)
================================================================
Loads Phase-3 EEG + ET epoch arrays per subject and computes:

  1. Dynamic graph representations (band-power + adjacency)
  2. ET engagement labels   (HIGH/LOW, from EngagementLabeler)
  3. ROI dwell vectors      (spatial attention distribution)

Each __getitem__ returns:
  eeg_windows  : (W, C, 5)    band-power node features per window
  adj_matrices : (W, C, C)    binary adjacency per window
  weighted_adjs: (W, C, C)    weighted adjacency (for loss/vis)
  et_seq       : (T_et, C_et) raw ET sequence
  roi_vector   : (N_rois,)    normalised ROI dwell time
  label        : scalar int   0=LOW_ENGAGEMENT / 1=HIGH_ENGAGEMENT
  subject_id   : scalar int   (for LOSOCV / MMD)

Leakage prevention
------------------
• Per-subject subfolder isolation: data_pipeline/04_segmentation/SXX/output/epochs/
• Falls back to pooled directory only when subject_ids.npy exists
• Prints loud warning + sets _has_leakage_risk if isolation fails
• LOSOCV enforces strict train/test subject disjointness
================================================================
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    PHASE3_DIR, SUBJECT_IDS,
    N_WINDOWS, EEG_SR, ET_SR,
    N_ROIS, IMAGE_W, IMAGE_H, GRID_COLS, GRID_ROWS,
    CONN_METHOD, CONN_THRESHOLD, ET_INPUT_DIM,
    ENGAGEMENT_BALANCE_RATIO,
    LABELS_DIR,
    EPOCH_OVERLAP,
)
from model.inference.graphs.graph_builder import compute_epoch_graphs
from data.channel_harmonizer import (
    harmonize_eeg_channels,
    load_channel_names,
    CANONICAL_N,
    MONTAGE_SCHEMA_VERSION,
)
from labeling.engagement_labeler import EngagementLabeler


# ── Disk Graph Cache ──────────────────────────────────────────────────────────

def _graph_cache_key(n_windows: int, conn_method: str, threshold: float,
                     fs: float) -> str:
    """Short hash of graph-construction parameters — used as cache subdirectory.

    Includes MONTAGE_SCHEMA_VERSION so a channel-montage fix (e.g. changing
    which/how-many electrodes are dropped) always misses the old cache
    instead of silently reusing graphs built from the previous montage.
    """
    tag = f"w{n_windows}_m{conn_method}_t{threshold:.3f}_fs{int(fs)}_{MONTAGE_SCHEMA_VERSION}"
    return hashlib.md5(tag.encode()).hexdigest()[:8]


def _graph_cache_path(subject_dir: Path, epoch_idx: int,
                      cache_key: str) -> Path:
    return subject_dir / "output" / "graph_cache" / cache_key / f"ep{epoch_idx:04d}.pt"


def load_cached_graph(subject_dir: Path, epoch_idx: int,
                      cache_key: str) -> Optional[dict]:
    """Load cached graph tensors; return as numpy dict or None if missing."""
    p = _graph_cache_path(subject_dir, epoch_idx, cache_key)
    if not p.exists():
        return None
    # saved as pure torch tensors — weights_only=True is safe
    t = torch.load(p, map_location="cpu", weights_only=True)
    return {k: v.numpy() for k, v in t.items()}


def save_cached_graph(subject_dir: Path, epoch_idx: int,
                      cache_key: str, data: dict) -> None:
    """Convert numpy arrays to tensors and save atomically."""
    p = _graph_cache_path(subject_dir, epoch_idx, cache_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: torch.from_numpy(np.asarray(v, dtype=np.float32))
               for k, v in data.items()}
    tmp = p.with_suffix(".tmp")
    torch.save(tensors, tmp)
    tmp.rename(p)

# ── Pre-computed Label Loader ─────────────────────────────────────────────────

def _load_precomputed_engagement(subject_id: str, epochs_dir: Path):
    """
    Load pre-computed engagement labels for one subject.

    Priority (highest first):
      1. Phase 3D multimodal (EEG + ET, global threshold) — engagement_phase3d/
      2. Phase 3  ET-only (global threshold)              — engagement/
      3. V2 fallback (per-subject median, last resort)    — engagement_v2/

    Global threshold is preferred for cross-subject label consistency.
    Per-subject median (v2) destroys label coherence across folds and
    produces near-random pooled metrics despite valid per-fold metrics.

    For each candidate, checks:
      {subject_dir}/output/<eng_subdir>/engagement_labels.npy
      {subject_dir}/output/epochs/<eeg_file>
      {subject_dir}/output/epochs/<et_file>

    Applies channel harmonization so EEG arrays match CANONICAL_N.
    Returns (eeg_list, et_list, labels_ndarray) or None if no candidate found.
    """
    candidates = [
        # (label_subdir, eeg_npy, et_npy, source_tag)
        # Priority: global-threshold labels FIRST (cross-subject consistency).
        # engagement_v2 (per-subject median) is kept only as last fallback —
        # it fixes class balance but destroys cross-subject label coherence,
        # causing near-random pooled metrics (MCC ≈ 0, AUC ≈ 0.5).
        ("engagement_phase3d",
         "eeg_epochs_phase3d.npy", "et_epochs_phase3d.npy",
         "multimodal (EEG+ET, global threshold)"),
        ("engagement",
         "eeg_epochs_engagement.npy", "et_epochs_engagement.npy",
         "ET-only (global threshold)"),
        ("engagement_v2",
         "eeg_epochs_phase3d.npy", "et_epochs_phase3d.npy",
         "v2-labelled (subject-median threshold, fallback only)"),
        ("engagement_purchase",
         "eeg_epochs_purchase.npy", "et_epochs_purchase.npy",
         "purchase intent (behavioural: dominant fixated product bought, Q77)"),
        ("engagement_product",
         "eeg_epochs_product.npy", "et_epochs_product.npy",
         "product-level purchase (behavioural: one 5-s epoch per viewed product, Q77)"),
        ("engagement_control",
         "eeg_epochs_control.npy", "et_epochs_control.npy",
         "positive control: brochure browsing (1) vs resting state (0)"),
    ]
    # NEUMA_LABEL_SOURCE selects the label:  phase3d (default, production
    # rule-based index) | purchase (behavioural, 04_segmentation/purchase_labeling.py)
    # | engagement | v2.  A non-default source never falls back silently.
    import os as _os
    _src = _os.environ.get("NEUMA_LABEL_SOURCE", "phase3d").strip().lower()
    _want = {"phase3d": "engagement_phase3d", "purchase": "engagement_purchase",
             "product": "engagement_product", "control": "engagement_control",
             "engagement": "engagement", "v2": "engagement_v2"}.get(_src)
    if _want is None:
        raise ValueError(f"NEUMA_LABEL_SOURCE={_src!r}; expected phase3d|purchase|product|control|engagement|v2")
    if _src != "phase3d":
        candidates = [c for c in candidates if c[0] == _want]
        if not (epochs_dir.parent / _want / "engagement_labels.npy").exists():
            raise FileNotFoundError(
                f"{subject_id}: NEUMA_LABEL_SOURCE={_src} but {epochs_dir.parent / _want} has no labels — "
                f"run src/data_pipeline/04_segmentation/purchase_labeling.py, product_epoching.py or control_epoching.py first")
    else:
        candidates = [c for c in candidates if c[0] not in ("engagement_purchase", "engagement_product", "engagement_control")]

    for eng_sub, eeg_file, et_file, tag in candidates:
        eng_dir  = epochs_dir.parent / eng_sub
        lbl_path = eng_dir / "engagement_labels.npy"
        eeg_eng  = epochs_dir / eeg_file
        et_eng   = epochs_dir / et_file

        if not (lbl_path.exists() and eeg_eng.exists() and et_eng.exists()):
            continue

        labels  = np.load(lbl_path, allow_pickle=True).astype(np.int64)
        eeg_arr = np.load(eeg_eng,  allow_pickle=True)
        et_arr  = np.load(et_eng,   allow_pickle=True)

        eeg_list = [np.array(eeg_arr[i], dtype=np.float32) for i in range(len(eeg_arr))]
        et_list  = [et_arr[i] for i in range(len(et_arr))]

        if len(labels) != len(eeg_list):
            continue   # length mismatch — try next candidate

        # Channel harmonization (same logic as _load_subject_raw) — always
        # applied, since harmonize_eeg_channels() also drops the 5
        # non-cortical channels even when the raw array is already
        # RAW_MONTAGE_N (24) wide.
        ch_names = load_channel_names(epochs_dir)
        eeg_list = [
            harmonize_eeg_channels(e, subject_id,
                                   channel_names=ch_names, verbose=(i == 0))
            for i, e in enumerate(eeg_list)
        ]

        print(f"  [Dataset] {subject_id!r}: pre-computed labels loaded "
              f"[{tag}]  {len(labels)} epochs  EEG={eeg_list[0].shape}")
        return eeg_list, et_list, labels

    return None


# ── Engagement Class Map ──────────────────────────────────────────────────────

LABEL_NAMES   = {0: "LOW_ENGAGEMENT", 1: "HIGH_ENGAGEMENT"}
N_ENG_CLASSES = 2


# ── ROI Dwell Vector ─────────────────────────────────────────────────────────

def _roi_dwell_vector(et_epoch, n_rois: int = N_ROIS) -> np.ndarray:
    """
    Normalised ROI spatial-attention vector from raw ET epoch.
    Robust to object arrays, NaN, mixed dtypes.
    """
    if not isinstance(et_epoch, np.ndarray):
        et_epoch = np.asarray(et_epoch)
    if et_epoch.ndim != 2 or et_epoch.shape[1] < 2:
        return np.zeros(n_rois, dtype=np.float32)
    if et_epoch.dtype.kind not in ("f", "i", "u"):
        rows = []
        for row in et_epoch:
            safe = []
            for val in row:
                try:
                    safe.append(float(val))
                except (TypeError, ValueError):
                    safe.append(0.0)
            rows.append(safe)
        et_epoch = np.array(rows, dtype=np.float32)
    else:
        et_epoch = et_epoch.astype(np.float32)
    et_epoch = np.nan_to_num(et_epoch, nan=0.0, posinf=0.0, neginf=0.0)

    gx, gy  = et_epoch[:, 0], et_epoch[:, 1]
    valid   = np.isfinite(gx) & np.isfinite(gy)
    gx, gy  = gx[valid], gy[valid]
    if len(gx) == 0:
        return np.zeros(n_rois, dtype=np.float32)

    if gx.max() > 2.0:
        gx = np.clip(gx / IMAGE_W, 0.0, 1.0)
    if gy.max() > 2.0:
        gy = np.clip(gy / IMAGE_H, 0.0, 1.0)

    col_idx = np.clip((gx * GRID_COLS).astype(np.int32), 0, GRID_COLS - 1)
    row_idx = np.clip((gy * GRID_ROWS).astype(np.int32), 0, GRID_ROWS - 1)
    cell    = row_idx * GRID_COLS + col_idx
    n_cells = GRID_COLS * GRID_ROWS
    counts  = np.bincount(cell, minlength=n_cells).astype(np.float32)
    r_t     = counts / (counts.sum() + 1e-10)

    if n_cells < n_rois:
        r_t = np.pad(r_t, (0, n_rois - n_cells))
    elif n_cells > n_rois:
        r_t = r_t[:n_rois]

    return r_t.astype(np.float32)


# ── Subject ID Normaliser ────────────────────────────────────────────────────

def _normalize_subject_id(subject_id) -> str:
    if isinstance(subject_id, (int, np.integer)):
        n = int(subject_id)
        if n <= 0:
            raise ValueError(f"Invalid subject ID {subject_id!r}: must be ≥1.")
        return f"S{n:02d}"
    sid = str(subject_id).strip()
    if sid.isdigit():
        n = int(sid)
        if n <= 0:
            raise ValueError(f"Invalid subject ID {subject_id!r}: must be ≥1.")
        return f"S{n:02d}"
    if sid == "S00":
        raise ValueError("Invalid subject ID 'S00': subjects start at S01.")
    return sid


# ── Subject Data Loader ──────────────────────────────────────────────────────

def _load_subject_raw(subject_id):
    """
    Load Phase-3 epoch arrays for one subject.

    Returns (eeg_list, et_list, chosen_dir)  or  None if not found.

    NOTE: labels are no longer loaded from disk.
          Engagement labels are derived from ET data by EngagementLabeler.
    """
    subject_id = _normalize_subject_id(subject_id)
    per_subj   = PHASE3_DIR.parent / subject_id / "output" / "epochs"
    default    = PHASE3_DIR / "epochs"

    chosen_dir = None
    for cand in [per_subj, default]:
        if (cand / "eeg_epochs.npy").exists():
            chosen_dir = cand
            break

    if chosen_dir is None:
        print(f"  [Dataset] No data for {subject_id!r} — skipping")
        return None

    eeg_arr = np.load(chosen_dir / "eeg_epochs.npy", allow_pickle=True)
    et_path  = chosen_dir / "et_epochs.npy"
    sid_path = chosen_dir / "subject_ids.npy"

    et_arr  = np.load(et_path, allow_pickle=True) if et_path.exists() else None
    sid_arr = np.load(sid_path, allow_pickle=True) if sid_path.exists() else None

    if eeg_arr.ndim == 3:
        eeg_list = [eeg_arr[i] for i in range(len(eeg_arr))]
        et_list  = ([et_arr[i] for i in range(len(et_arr))]
                    if et_arr is not None else None)
    else:
        eeg_list = list(eeg_arr)
        et_list  = list(et_arr) if et_arr is not None else None

    n = len(eeg_list)

    if et_list is None:
        T_et    = int(ET_SR * 5.0)
        et_list = [np.zeros((T_et, ET_INPUT_DIM), dtype=np.float32)] * n

    # ── Subject filtering when using pooled dir ───────────────────────────
    if chosen_dir == default:
        if sid_arr is not None:
            mask = np.array([
                _normalize_subject_id(str(s).strip()) == subject_id
                for s in sid_arr
            ], dtype=bool)
            if not mask.any():
                print(f"  [Dataset] {subject_id!r} not in subject_ids.npy — skip")
                return None
            idx = np.where(mask)[0]
            eeg_list = [eeg_list[i] for i in idx]
            et_list  = [et_list[i]  for i in idx]
        else:
            print(
                f"\n  [Dataset] *** LEAKAGE WARNING *** {subject_id!r} "
                f"using shared dir without subject_ids.npy — LOSOCV INVALID\n"
            )

    print(f"  [Dataset] {subject_id!r}: {len(eeg_list)} epochs  "
          f"EEG={np.array(eeg_list[0]).shape}  "
          f"ET={np.array(et_list[0]).shape}")

    # ── Channel harmonization ─────────────────────────────────────────────
    # Always applied (not gated on n_ch != CANONICAL_N): harmonize_eeg_channels()
    # also drops the 5 non-cortical channels even when the raw array is
    # already RAW_MONTAGE_N (24) wide, so every subject must pass through it.
    ch_names = load_channel_names(chosen_dir)
    eeg_list = [
        harmonize_eeg_channels(
            np.array(e, dtype=np.float32), subject_id,
            channel_names=ch_names, verbose=(i == 0),
        )
        for i, e in enumerate(eeg_list)
    ]

    return eeg_list, et_list, chosen_dir


# ── Dataset Class ────────────────────────────────────────────────────────────

class NeumaGraphDataset(Dataset):
    """
    Multi-subject EEG/ET graph dataset for INS-HDGS-CMT training.

    Labels: HIGH_ENGAGEMENT (1) vs LOW_ENGAGEMENT (0)
    Derived from ET metrics via EngagementLabeler (median-threshold, dataset-level).

    Supports:
      - Single-subject mode
      - Multi-subject mode
      - LOSOCV fold specification via exclude_subjects
      - Leakage detection and reporting
    """

    def __init__(
        self,
        subject_ids       : Optional[List] = None,
        exclude_subjects  : Optional[List] = None,
        n_windows         : int   = N_WINDOWS,
        fs_eeg            : float = EEG_SR,
        conn_method       : str   = CONN_METHOD,
        threshold         : float = CONN_THRESHOLD,
        n_rois            : int   = N_ROIS,
        et_input_dim      : int   = ET_INPUT_DIM,
        precompute_graphs : bool  = True,
        save_labels       : bool  = False,
        label_out_dir     : Optional[Path] = None,
        augment           : bool  = False,
        use_disk_cache    : bool  = True,
        norm_mode         : str   = "zscore",
    ):
        # Subject-aware feature normalization applied before graph construction
        # (Phase-3 ablation). Disk graph cache is keyed per norm_mode to avoid
        # cross-contamination between normalization schemes.
        self.norm_mode     = norm_mode
        self.n_windows     = n_windows
        self.fs_eeg        = fs_eeg
        self.conn_method   = conn_method
        self.threshold     = threshold
        self.n_rois        = n_rois
        self.et_input_dim  = et_input_dim
        self.use_disk_cache = use_disk_cache
        self._augment      = augment       # stored for per-item stochastic augmentation
        self._cache_key    = _graph_cache_key(n_windows, conn_method, threshold, fs_eeg)
        # Namespace the graph cache per normalization scheme so Phase-3 variants
        # never reuse graphs built under a different norm_mode. "zscore" keeps
        # the original key (back-compat with existing caches).
        if getattr(self, "norm_mode", "zscore") != "zscore":
            self._cache_key = f"{self._cache_key}_{self.norm_mode}"

        if subject_ids is None:
            subject_ids = SUBJECT_IDS
        subject_ids = [_normalize_subject_id(s) for s in subject_ids]
        exclude     = {_normalize_subject_id(s) for s in (exclude_subjects or [])}
        subject_ids = [s for s in subject_ids if s not in exclude]

        all_eeg, all_et, all_labels, all_subj_ids = [], [], [], []
        all_subj_dirs:      List[Path] = []   # per-epoch subject root (disk cache)
        all_subj_local_idx: List[int]  = []   # per-epoch index within its subject
        loaded_subjects : List[str] = []
        self._source_dirs: set = set()
        _default_dir_users: List[str] = []

        for s_idx, sid in enumerate(subject_ids):
            result = _load_subject_raw(sid)
            if result is None:
                continue
            eeg_list, et_list, chosen_dir = result
            self._source_dirs.add(str(chosen_dir))
            if chosen_dir == PHASE3_DIR / "epochs":
                _default_dir_users.append(sid)

            # Try pre-computed engagement labels from Phase 3 engagement_labeling.py
            precomp = _load_precomputed_engagement(sid, chosen_dir)
            if precomp is not None:
                eeg_list, et_list, eng_labels = precomp
            else:
                # Fall back: compute engagement labels from ET epochs on-the-fly
                labeler    = EngagementLabeler(
                    balance_ratio = ENGAGEMENT_BALANCE_RATIO,
                    et_sr         = ET_SR,
                    grid_cols     = GRID_COLS,
                    grid_rows     = GRID_ROWS,
                    image_w       = IMAGE_W,
                    image_h       = IMAGE_H,
                    verbose       = True,
                )
                eng_labels = labeler.fit_transform(et_list)

                if save_labels:
                    out = Path(label_out_dir or LABELS_DIR)
                    labeler.save_labels(out, subject_id=sid)
                    labeler.plot_distribution(out / f"engagement_dist_{sid}.png")

            # ── Subject-aware normalization (Phase-3) ────────────────────────
            # All schemes remove subject-specific amplitude distributions
            # before graph construction; they differ in statistic & scope.
            _nm = getattr(self, "norm_mode", "zscore")
            _eeg = [np.array(e, dtype=np.float32) for e in eeg_list]
            if _nm in ("zscore", "robust"):
                concat = np.concatenate(_eeg, axis=0)          # (ΣT, C)
                if _nm == "zscore":
                    # per-subject, per-channel mean/std (existing default)
                    ctr = concat.mean(axis=0, keepdims=True)
                    scl = concat.std(axis=0,  keepdims=True) + 1e-8
                else:
                    # robust: per-subject, per-channel median / MAD
                    ctr = np.median(concat, axis=0, keepdims=True)
                    mad = np.median(np.abs(concat - ctr), axis=0, keepdims=True)
                    scl = mad * 1.4826 + 1e-8                   # MAD→σ consistency
                eeg_list = [((e - ctr) / scl) for e in _eeg]
            elif _nm == "instance":
                # InstanceNorm: per-epoch, per-channel z-score (no cross-epoch stat)
                eeg_list = [((e - e.mean(axis=0, keepdims=True))
                             / (e.std(axis=0, keepdims=True) + 1e-8)) for e in _eeg]
            elif _nm == "layer":
                # LayerNorm: per-epoch, over all channels+time jointly
                eeg_list = [((e - e.mean()) / (e.std() + 1e-8)) for e in _eeg]
            else:
                raise ValueError(f"Unknown norm_mode={_nm!r}")

            # subject root for disk-cache writes (parent of output/epochs/)
            subj_root  = chosen_dir.parents[1]   # data_pipeline/04_segmentation/S##
            n_ep       = len(eeg_list)

            all_eeg.extend(eeg_list)
            all_et.extend(et_list)
            all_labels.extend(eng_labels.tolist())
            all_subj_ids.extend([s_idx] * n_ep)
            all_subj_dirs.extend([subj_root] * n_ep)
            all_subj_local_idx.extend(range(n_ep))   # 0,1,2,… per subject
            loaded_subjects.append(sid)

        if not all_eeg:
            raise FileNotFoundError(
                "No Phase-3 epoch data found. Run Phase 3 pipeline first."
            )

        # Channel count validation
        unique_nch = {np.array(e).shape[1] for e in all_eeg}
        if len(unique_nch) > 1:
            raise RuntimeError(
                f"[Dataset] Channel mismatch post-harmonization: {unique_nch}"
            )

        self._has_leakage_risk = len(_default_dir_users) > 1
        if self._has_leakage_risk:
            print(
                f"\n  [Dataset] *** DATA ISOLATION FAILURE ***\n"
                f"  {len(_default_dir_users)} subjects share one dir without "
                f"subject_ids.npy — LOSOCV results INVALID.\n"
            )

        self.labels            = np.array(all_labels, dtype=np.int64)
        self.label_map         = {v: k for k, v in LABEL_NAMES.items()}
        self.raw_eeg           = list(all_eeg)
        self.raw_et            = list(all_et)
        self.subject_ids       = np.array(all_subj_ids, dtype=np.int64)
        self._subj_dirs        = all_subj_dirs        # per-epoch subject root
        self._subj_local_idx   = all_subj_local_idx   # per-epoch local index
        self.n_classes       = N_ENG_CLASSES
        self.unique_subjects = loaded_subjects
        self.n_eeg_ch        = CANONICAL_N
        self.n_et_ch         = min(np.array(all_et[0]).shape[1], et_input_dim)

        assert all_eeg[0].shape[1] == CANONICAL_N

        # Class distribution report
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\n  [Dataset] Engagement distribution ({len(self.labels)} epochs):")
        for cls, cnt in zip(unique, counts):
            pct = 100.0 * cnt / len(self.labels)
            print(f"    {LABEL_NAMES[cls]}: {cnt}  ({pct:.1f}%)")

        # Circular time-shift augmentation (training only).
        # Each original epoch is duplicated with a half-epoch circular shift,
        # ~doubling training data without collecting new recordings.
        if augment and EPOCH_OVERLAP > 0.0:
            eeg_len = all_eeg[0].shape[0]
            et_len  = all_et[0].shape[0] if hasattr(all_et[0], 'shape') else len(all_et[0])
            shift_eeg = int(eeg_len * (1.0 - EPOCH_OVERLAP))
            shift_et  = int(et_len  * (1.0 - EPOCH_OVERLAP))
            aug_eeg, aug_et, aug_lbl, aug_sid = [], [], [], []
            for eeg_ep, et_ep, lbl, sid in zip(all_eeg, all_et, all_labels, all_subj_ids):
                eeg_ep = np.array(eeg_ep, dtype=np.float32)
                et_ep  = np.array(et_ep,  dtype=np.float32)
                aug_eeg.append(np.roll(eeg_ep, shift_eeg, axis=0))
                aug_et.append( np.roll(et_ep,  shift_et,  axis=0))
                aug_lbl.append(lbl)
                aug_sid.append(sid)
            all_eeg           = all_eeg           + aug_eeg
            all_et            = all_et            + aug_et
            all_labels        = all_labels        + aug_lbl
            all_subj_ids      = all_subj_ids      + aug_sid
            # Augmented epochs: no disk cache (None sentinels)
            all_subj_dirs     = all_subj_dirs     + [None] * len(aug_eeg)
            all_subj_local_idx= all_subj_local_idx+ [-1]  * len(aug_eeg)
            self.labels           = np.array(all_labels, dtype=np.int64)
            self.raw_eeg          = list(all_eeg)
            self.raw_et           = list(all_et)
            self.subject_ids      = np.array(all_subj_ids, dtype=np.int64)
            self._subj_dirs       = all_subj_dirs
            self._subj_local_idx  = all_subj_local_idx
            print(f"  [Dataset] Augmented: {len(all_eeg)} epochs "
                  f"(shift={shift_eeg} samples, overlap={EPOCH_OVERLAP})", flush=True)

        self._graph_cache = {}
        if precompute_graphs:
            self._precompute_all()

    # ── Precomputation ───────────────────────────────────────────────────────

    def _precompute_all(self):
        n = len(self)
        print(f"  [Dataset] Pre-computing graphs for {n} epochs …", flush=True)
        for i in range(n):
            g = self._compute_graph(i)
            # Store as float32 tensors so __getitem__ avoids numpy→tensor copies
            self._graph_cache[i] = {
                k: torch.from_numpy(np.asarray(v, dtype=np.float32))
                for k, v in g.items()
            }
        print("  [Dataset] Done.")

    def _compute_graph(self, idx: int) -> dict:
        # ── Try disk cache first (only for non-augmented epochs) ─────────────
        subj_dir   = self._subj_dirs[idx]      if hasattr(self, "_subj_dirs")       else None
        local_idx  = self._subj_local_idx[idx] if hasattr(self, "_subj_local_idx")  else idx
        # local_idx is the epoch's position within its own subject (stable across
        # all folds), unlike idx which changes when the training set composition changes.
        if self.use_disk_cache and subj_dir is not None and local_idx >= 0:
            cached = load_cached_graph(subj_dir, local_idx, self._cache_key)
            if cached is not None:
                return cached

        # ── Compute from raw data ─────────────────────────────────────────────
        eeg = self.raw_eeg[idx]
        et  = self.raw_et[idx]
        if not isinstance(et, np.ndarray):
            et = np.asarray(et)
        et = et[:, :self.n_et_ch]

        node_feats, adj_bin, adj_wt = compute_epoch_graphs(
            eeg_epoch   = eeg,
            n_windows   = self.n_windows,
            fs          = self.fs_eeg,
            conn_method = self.conn_method,
            threshold   = self.threshold,
        )
        roi_vec = _roi_dwell_vector(et, self.n_rois)

        # Pad / truncate ET
        T_et   = et.shape[0]
        target = int(ET_SR * 5.0)   # 600
        if T_et < target:
            pad = np.zeros((target - T_et, self.n_et_ch), dtype=np.float32)
            et  = np.concatenate([et, pad], axis=0)
        else:
            et = et[:target]
        et = np.nan_to_num(et.astype(np.float32), nan=0.0)

        result = {
            "eeg_windows"  : node_feats.astype(np.float32),
            "adj_matrices" : adj_bin.astype(np.float32),
            "weighted_adjs": adj_wt.astype(np.float32),
            "et_seq"       : et,
            "roi_vector"   : roi_vec,
        }

        # ── Persist to disk (non-augmented epochs only) ───────────────────────
        if self.use_disk_cache and subj_dir is not None and local_idx >= 0:
            save_cached_graph(subj_dir, local_idx, self._cache_key, result)

        return result

    # ── Dataset Protocol ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        g = self._graph_cache.get(idx)
        if g is None:
            raw = self._compute_graph(idx)
            g   = {k: torch.from_numpy(np.asarray(v, dtype=np.float32))
                   for k, v in raw.items()}

        def _as_tensor(v):
            if isinstance(v, torch.Tensor):
                return v.clone()
            return torch.from_numpy(np.asarray(v, dtype=np.float32))

        eeg_win = _as_tensor(g["eeg_windows"])

        # Stochastic augmentation at item-level (training only).
        # Applied on the already-normalised tensor so amplitude scaling and
        # additive noise are signal-relative.
        if self._augment:
            # Amplitude scaling: uniform in [0.85, 1.15] per sample
            scale = 0.85 + 0.30 * float(torch.rand(1))
            eeg_win = eeg_win * scale
            # Gaussian noise: std = 2% of per-channel std
            ch_std  = eeg_win.std(dim=-1, keepdim=True).clamp(min=1e-6)
            noise   = torch.randn_like(eeg_win) * ch_std * 0.02
            eeg_win = eeg_win + noise
            # Channel dropout: zero 1-2 random electrodes (30% chance)
            # eeg_win shape: (N_windows, n_channels, n_samples)
            if torch.rand(1).item() < 0.30:
                n_ch   = eeg_win.shape[1]
                n_drop = 1 + int(torch.rand(1).item() < 0.3)   # 1 or 2
                drop   = torch.randperm(n_ch)[:n_drop]
                eeg_win[:, drop, :] = 0.0
            # Temporal masking: zero a 15% segment of each window (30% chance)
            if torch.rand(1).item() < 0.30:
                T        = eeg_win.shape[2]
                mask_len = max(1, int(T * 0.15))
                start    = int(torch.randint(0, T - mask_len + 1, (1,)).item())
                eeg_win[:, :, start : start + mask_len] = 0.0

        return {
            "eeg_windows"  : eeg_win,
            "adj_matrices" : _as_tensor(g["adj_matrices"]),
            "weighted_adjs": _as_tensor(g["weighted_adjs"]),
            "et_seq"       : _as_tensor(g["et_seq"]),
            "roi_vector"   : _as_tensor(g["roi_vector"]),
            "label"        : torch.tensor(self.labels[idx],      dtype=torch.long),
            "subject_id"   : torch.tensor(self.subject_ids[idx], dtype=torch.long),
        }


# ── Collate & Dataloader ─────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def build_dataloaders(
    train_dataset : NeumaGraphDataset,
    test_dataset  : NeumaGraphDataset,
    batch_size    : int = 32,
    num_workers   : int = 4,
):
    use_cuda = torch.cuda.is_available()
    # Large training sets (product-level epochs: ~2,300 pre-computed graph
    # tensors per fold) exhaust the per-process file-descriptor limit under the
    # default "file_descriptor" tensor-sharing strategy ("Too many open files").
    # The file_system strategy shares via named shm files instead.
    import os as _os
    num_workers = int(_os.environ.get("NEUMA_NUM_WORKERS", num_workers))
    if num_workers > 0:
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except Exception:
            pass
    # persistent_workers keeps worker processes alive between epochs (avoids
    # the fork/import overhead of ~0.5s per epoch with num_workers > 0).
    # prefetch_factor=2 keeps 2 batches ready in worker RAM at all times.
    # pin_memory enables async CPU→GPU DMA transfers.
    persistent = num_workers > 0
    prefetch   = 2 if num_workers > 0 else None

    def _make(ds, shuffle, drop_last):
        n  = len(ds)
        bs = min(batch_size, n)
        return DataLoader(
            ds,
            batch_size         = bs,
            shuffle            = shuffle,
            drop_last          = drop_last and n > bs,
            collate_fn         = collate_fn,
            num_workers        = num_workers,
            pin_memory         = use_cuda,
            persistent_workers = persistent,
            prefetch_factor    = prefetch,
        )
    return (
        _make(train_dataset, shuffle=True,  drop_last=True),
        _make(test_dataset,  shuffle=False, drop_last=False),
    )
