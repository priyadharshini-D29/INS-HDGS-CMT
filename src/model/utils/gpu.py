"""
================================================================
NEUMA Phase 8 — Multi-GPU Utilities
================================================================
Centralised helpers for:
  - GPU detection and environment reporting
  - DataParallel / DDP model wrapping / unwrapping
  - Per-GPU memory monitoring
  - Fold-to-GPU assignment for fold-parallel LOSOCV
  - Benchmark summary generation
  - DDP process-group setup / teardown stubs

Usage
-----
    from utils.gpu import (
        print_gpu_summary, wrap_model_dp, unwrap_model,
        log_gpu_memory, get_fold_gpu_map, benchmark_summary,
    )

CUDA_VISIBLE_DEVICES must be set *before* torch is imported if you
want to restrict which physical GPUs are visible.  Recommended:

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py
================================================================
"""

from __future__ import annotations

import os
import time
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ── GPU Environment ──────────────────────────────────────────────────────────

def get_gpu_info() -> dict:
    """Return a dict describing all visible GPUs."""
    n = torch.cuda.device_count()
    info: dict = {
        "num_gpus"       : n,
        "cuda_available" : torch.cuda.is_available(),
        "cuda_version"   : torch.version.cuda or "N/A",
        "pytorch_version": torch.__version__,
        "gpu_names"      : [],
        "total_memory_gb": [],
        "compute_cap"    : [],
    }
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        info["gpu_names"].append(p.name)
        info["total_memory_gb"].append(round(p.total_memory / 1e9, 2))
        info["compute_cap"].append(f"{p.major}.{p.minor}")
    return info


def print_gpu_summary() -> dict:
    """Print a formatted GPU inventory table and return the info dict."""
    info = get_gpu_info()
    w = 60
    print(f"\n  {'─'*w}")
    print(f"  Multi-GPU Environment")
    print(f"  {'─'*w}")
    print(f"  PyTorch        : {info['pytorch_version']}")
    print(f"  CUDA           : {info['cuda_version']}")
    print(f"  GPUs visible   : {info['num_gpus']}")
    for i in range(info["num_gpus"]):
        name = info["gpu_names"][i]
        mem  = info["total_memory_gb"][i]
        cap  = info["compute_cap"][i]
        print(f"    GPU {i}        : {name}  {mem:.1f} GB  cc={cap}")
    if info["num_gpus"] == 0:
        print("    [WARNING] No GPUs detected — running on CPU")
    print(f"  {'─'*w}\n")
    return info


# ── DataParallel Helpers ─────────────────────────────────────────────────────

def wrap_model_dp(model: nn.Module, device: torch.device) -> tuple[nn.Module, bool]:
    """
    Wrap *model* in DataParallel when multiple GPUs are visible.

    Returns (wrapped_model, is_dp).  The model is moved to *device* before
    wrapping so all replicas share weights from the primary device.

    Parameters
    ----------
    model  : bare nn.Module (must NOT already be DataParallel)
    device : primary device (cuda:0 or cpu)

    Returns
    -------
    model  : DataParallel-wrapped or bare model, already on device
    is_dp  : True if DataParallel was applied
    """
    n = torch.cuda.device_count()
    if n > 1 and device.type == "cuda":
        print(f"[INFO] Using {n} GPUs via DataParallel")
        model = nn.DataParallel(model)
        model = model.to(device)
        return model, True
    model = model.to(device)
    return model, False


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Return the raw model, unwrapping DataParallel / DDP if present.

    Safe to call whether or not wrapping was applied.
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    # Handle DistributedDataParallel too
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(model, DDP):
            return model.module
    except ImportError:
        pass
    return model


# ── GPU Memory Monitoring ────────────────────────────────────────────────────

def gpu_memory_stats(device: Optional[torch.device] = None) -> dict:
    """
    Return allocated and reserved memory in GB for *device*.

    If *device* is None, query the current default CUDA device.
    Returns an empty dict when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return {}
    idx = 0
    if device is not None and device.index is not None:
        idx = device.index
    elif torch.cuda.current_device() >= 0:
        idx = torch.cuda.current_device()

    return {
        "allocated_gb" : round(torch.cuda.memory_allocated(idx)  / 1e9, 3),
        "reserved_gb"  : round(torch.cuda.memory_reserved(idx)   / 1e9, 3),
        "max_alloc_gb" : round(torch.cuda.max_memory_allocated(idx) / 1e9, 3),
    }


def log_gpu_memory(
    device   : Optional[torch.device] = None,
    prefix   : str = "",
    reset_peak: bool = False,
) -> dict:
    """Print and return GPU memory stats. Optionally reset peak tracker."""
    stats = gpu_memory_stats(device)
    if stats:
        tag = f"  {prefix} | " if prefix else "  "
        idx = 0 if device is None or device.index is None else device.index
        print(
            f"{tag}GPU{idx} mem:"
            f"  alloc={stats['allocated_gb']:.2f} GB"
            f"  peak={stats['max_alloc_gb']:.2f} GB"
        )
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(idx)
    return stats


def reset_peak_memory(device: Optional[torch.device] = None) -> None:
    """Reset the CUDA peak memory tracker for *device*."""
    if not torch.cuda.is_available():
        return
    idx = 0
    if device is not None and device.index is not None:
        idx = device.index
    torch.cuda.reset_peak_memory_stats(idx)


# ── Fold-to-GPU Assignment ───────────────────────────────────────────────────

def get_fold_gpu_map(n_folds: int, n_gpus: int) -> Dict[int, int]:
    """
    Build a mapping from fold number (1-based) to GPU index.

    Layout (with 8 GPUs, 34 folds):
      GPU 0 → folds  1– 5
      GPU 1 → folds  6–10
      GPU 2 → folds 11–14
      ...

    Parameters
    ----------
    n_folds : total number of LOSOCV folds
    n_gpus  : number of GPUs available for fold-parallel execution

    Returns
    -------
    fold_map : dict  {fold_no (int) → gpu_id (int)}
    """
    n_gpus = max(1, min(n_gpus, n_folds))
    folds_per_gpu = int(math.ceil(n_folds / n_gpus))
    fold_map: Dict[int, int] = {}
    for fold_idx in range(n_folds):
        gpu_id = fold_idx // folds_per_gpu
        gpu_id = min(gpu_id, n_gpus - 1)   # clamp tail folds to last GPU
        fold_map[fold_idx + 1] = gpu_id
    return fold_map


def print_fold_gpu_map(fold_map: Dict[int, int]) -> None:
    """Pretty-print the fold → GPU assignment."""
    n_gpus = max(fold_map.values()) + 1
    print(f"\n  Fold-parallel GPU assignment ({n_gpus} GPUs):")
    for gpu_id in range(n_gpus):
        folds = [f for f, g in fold_map.items() if g == gpu_id]
        print(f"    GPU {gpu_id} → folds {folds}")
    print()


# ── Post-hoc Temperature Calibration ─────────────────────────────────────────

def _posthoc_calibrate(
    logits_2d : np.ndarray,   # (N, 2) averaged raw ensemble logits
    labels    : np.ndarray,   # (N,)   integer labels
    T_min     : float = 0.05,
    T_max     : float = 5.0,
):
    """Fit a scalar temperature T on averaged logits by minimising NLL.

    Returns (T_post, p_cal) where p_cal is calibrated P(HIGH).
    Falls back to T=1 when fewer than 4 samples or single-class labels.
    """
    from scipy.special import softmax as _sp_softmax
    from scipy.optimize import minimize_scalar as _min_scalar

    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2 or len(labels) < 4:
        return 1.0, _sp_softmax(logits_2d, axis=1)[:, 1]

    def _nll(T):
        p  = _sp_softmax(logits_2d / max(float(T), 1e-6), axis=1)
        p1 = np.clip(p[:, 1], 1e-7, 1.0 - 1e-7)
        return -np.mean(labels * np.log(p1) + (1 - labels) * np.log(1 - p1))

    res    = _min_scalar(_nll, bounds=(T_min, T_max), method="bounded")
    T_post = float(np.clip(res.x, T_min, T_max))
    p_cal  = _sp_softmax(logits_2d / T_post, axis=1)[:, 1]
    return T_post, p_cal


def _youden_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Youden's J threshold clamped to [0.30, 0.70]; fallback 0.5."""
    from sklearn.metrics import roc_curve as _roc_curve
    labels = np.asarray(labels, dtype=int)
    if len(labels) < 4 or len(np.unique(labels)) < 2 or len(labels) < 20:
        return 0.5
    fpr, tpr, thrs = _roc_curve(labels, probs)
    idx = int(np.argmax(tpr - fpr))
    return float(np.clip(thrs[idx], 0.30, 0.70))


# ── Fold-parallel Worker (subprocess entry-point) ───────────────────────────

def run_fold_on_device(
    gpu_id     : int,
    test_subj  : str,
    train_subs : List[str],
    fold_no    : int,
    *,
    ablation,
    epochs     : int,
    batch_size : int,
    label      : str,
    ckpt_dir   : str,
    val_subj          : Optional[str]   = None,
    random_labels     : bool  = False,
    zero_roi          : bool  = False,
    zero_graph        : bool  = False,
    zero_et           : bool  = False,
    identity_graph    : bool  = False,
    random_seed       : int   = 42,
    verbose           : bool  = False,
    resume            : Optional[str]   = None,
    alpha_strategy    : str   = "balanced",
    focal_gamma_override: Optional[float] = None,
    n_ensemble_override : Optional[int]   = None,
    lambda_dann_override: Optional[float] = None,
    lambda_mmd_override : Optional[float] = None,
    mmd_mode            : str   = "marginal",
    norm_mode           : str   = "zscore",
    pretrained_eeg      : Optional[str]   = None,
) -> Optional[dict]:
    """
    Standalone fold worker executed in a subprocess for fold-parallel mode.

    Each subprocess owns exactly one GPU (cuda:0 within its process because
    CUDA_VISIBLE_DEVICES restricts visibility).  The function re-imports all
    necessary modules so it is self-contained.

    Returns a result dict (one LOSOCV row) or None on failure.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    # CUDA_VISIBLE_DEVICES was set in the parent before spawning and also in
    # run_gpu_worker before any torch import, so cuda:0 here is physical GPU
    # gpu_id.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Seed reproducibility
    torch.manual_seed(random_seed)
    import numpy as _np
    _np.random.seed(random_seed)

    import numpy as _np2
    from config.settings import (
        LR, WEIGHT_DECAY, PATIENCE,
        LAMBDA_CLS, LAMBDA_CONTRAST, LAMBDA_ROI, LAMBDA_CONNECTIVITY, LAMBDA_MMD,
        FOCAL_ALPHA, FOCAL_GAMMA, N_ENSEMBLE,
    )
    from data.dataset    import NeumaGraphDataset, build_dataloaders
    from training.losses import MultiTaskLoss, compute_alpha_weights as _caw

    _N_ENS_EFF   = n_ensemble_override if n_ensemble_override is not None else N_ENSEMBLE
    _GAMMA_EFF   = focal_gamma_override if focal_gamma_override is not None else FOCAL_GAMMA
    _LDANN_EFF   = lambda_dann_override if lambda_dann_override is not None else 0.10
    _LMMD_EFF    = lambda_mmd_override  if lambda_mmd_override  is not None else LAMBDA_MMD
    from training.trainer import Trainer
    from training.metrics import compute_metrics

    try:
        train_ds = NeumaGraphDataset(subject_ids=train_subs,  precompute_graphs=True,
                                     augment=True, norm_mode=norm_mode)
        val_ds   = NeumaGraphDataset(
            subject_ids=[val_subj] if val_subj else [train_subs[-1]],
            precompute_graphs=True, norm_mode=norm_mode,
        )
        test_ds  = NeumaGraphDataset(subject_ids=[test_subj], precompute_graphs=True,
                                     norm_mode=norm_mode)
    except FileNotFoundError as exc:
        print(f"  [SKIP] Fold {fold_no}: {exc}")
        return None

    if len(train_ds) < 2 or len(val_ds) < 1 or len(test_ds) < 2:
        print(f"  [SKIP] Fold {fold_no}: insufficient data "
              f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})")
        return None

    # Skip single-class test subjects (e.g. S16, S31, S33 under global threshold)
    import numpy as _np_sc
    te_unique = _np_sc.unique(test_ds.labels)
    if len(te_unique) < 2:
        cls_name = "ALL-HIGH" if int(te_unique[0]) == 1 else "ALL-LOW"
        print(f"  [SKIP] Fold {fold_no} — test subject {test_subj} "
              f"has single-class labels ({cls_name}) under global threshold.")
        return None

    if random_labels:
        rng = _np.random.default_rng(random_seed)
        train_ds.labels = rng.permutation(train_ds.labels)

    # Cap batch_size to training set size so we always get at least one batch
    effective_bs = min(batch_size, len(train_ds))
    train_loader, val_loader = build_dataloaders(
        train_ds, val_ds, batch_size=effective_bs
    )
    _, test_loader = build_dataloaders(
        train_ds, test_ds, batch_size=effective_bs
    )

    _any = zero_roi or zero_graph or zero_et or identity_graph

    def _transform(loader):
        for batch in loader:
            if zero_roi:
                batch["roi_vector"]   = torch.zeros_like(batch["roi_vector"])
            if zero_graph:
                batch["adj_matrices"] = torch.zeros_like(batch["adj_matrices"])
            if zero_et:
                batch["et_seq"]       = torch.zeros_like(batch["et_seq"])
            if identity_graph:
                B, W, N, _ = batch["adj_matrices"].shape
                eye = torch.eye(N, device=batch["adj_matrices"].device)
                batch["adj_matrices"] = (
                    eye.unsqueeze(0).unsqueeze(0).expand(B, W, N, N).clone()
                )
            yield batch

    # ── Fast resume: skip training if all ensemble checkpoints already exist ──
    from config.settings import N_ENSEMBLE as _N_ENS
    _ckpt_dir_p = Path(ckpt_dir)
    _ckpt_paths = [_ckpt_dir_p / f"{label}_fold{fold_no:02d}_e{ei}.pt"
                   for ei in range(_N_ENS)]
    if all(p.exists() for p in _ckpt_paths) and not random_labels:
        print(f"  [RESUME] Fold {fold_no:02d} ({test_subj}): "
              f"all {_N_ENS} checkpoints found — inference only.", flush=True)
        _ensemble_probs     : list = []
        _ensemble_val_logits: list = []
        _ensemble_tst_logits: list = []
        _val_probs_list     : list = []
        _val_labels               = None
        _y_true_final             = None
        for _ei, _cp in enumerate(_ckpt_paths):
            _raw = _build_raw_model(train_ds, ablation).to(device)
            try:
                _sd = torch.load(_cp, map_location=device, weights_only=False)
            except Exception:
                _sd = torch.load(_cp, map_location=device)
            _raw.load_state_dict(
                _sd["model_state_dict"] if isinstance(_sd, dict) and "model_state_dict" in _sd
                else _sd
            )
            # Re-calibrate temperature on val set (forward pass only, no backprop)
            _trainer = Trainer(
                model=_raw, device=device, loss_fn=None,
                lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE,
                ckpt_path=_cp, use_dp=False,
            )
            _trainer.calibrate_temperature(
                _transform(val_loader) if _any else val_loader
            )
            _raw.eval()
            _probs_i  : list = []
            _logits_i : list = []
            _ytrue_i  : list = []
            with torch.no_grad():
                for _batch in (_transform(test_loader) if _any else test_loader):
                    _bd = {k: v.to(device) for k, v in _batch.items()}
                    _out = _raw(
                        eeg_windows  =_bd["eeg_windows"],
                        adj_matrices =_bd["adj_matrices"],
                        et_seq       =_bd["et_seq"],
                        roi_vector   =_bd["roi_vector"],
                        weighted_adjs=_bd["weighted_adjs"],
                    )
                    _p = torch.softmax(_out["logits"] / _trainer.temperature, dim=1)[:, 1]
                    _probs_i.extend(_p.cpu().numpy().tolist())
                    _logits_i.extend(_out["logits"].cpu().numpy().tolist())
                    _ytrue_i.extend(_bd["label"].cpu().numpy().tolist())
            _ensemble_probs.append(_np2.array(_probs_i))
            _ensemble_tst_logits.append(_np2.array(_logits_i))
            if _y_true_final is None:
                _y_true_final = _np2.array(_ytrue_i)
            # Collect val probs + raw logits for threshold calibration
            _vprobs_i : list = []
            _vlogits_i: list = []
            _vlabels_i: list = []
            with torch.no_grad():
                for _vb in (_transform(val_loader) if _any else val_loader):
                    _vbd = {k: v.to(device) for k, v in _vb.items()}
                    _vout = _raw(
                        eeg_windows  =_vbd["eeg_windows"],
                        adj_matrices =_vbd["adj_matrices"],
                        et_seq       =_vbd["et_seq"],
                        roi_vector   =_vbd["roi_vector"],
                        weighted_adjs=_vbd["weighted_adjs"],
                    )
                    _vp = torch.softmax(_vout["logits"] / _trainer.temperature, dim=1)[:, 1]
                    _vprobs_i.extend(_vp.cpu().numpy().tolist())
                    _vlogits_i.extend(_vout["logits"].cpu().numpy().tolist())
                    _vlabels_i.extend(_vbd["label"].cpu().numpy().tolist())
            _val_probs_list.append(_np2.array(_vprobs_i))
            _ensemble_val_logits.append(_np2.array(_vlogits_i))
            if _val_labels is None:
                _val_labels = _np2.array(_vlabels_i)
            if verbose:
                print(f"  [GPU{gpu_id}] Fold {fold_no:02d} ens[{_ei}]  "
                      f"T={_trainer.temperature:.3f}", flush=True)

        _yp_avg  = _np2.stack(_ensemble_probs).mean(axis=0)
        _yp_2d   = _np2.stack([1 - _yp_avg, _yp_avg], axis=1)

        # Threshold sweep on val (original path)
        if _val_probs_list and _val_labels is not None:
            from sklearn.metrics import balanced_accuracy_score as _bas2
            _yv_avg   = _np2.stack(_val_probs_list).mean(axis=0)
            _best_thr, _best_bal = 0.5, -1.0
            for _thr in _np2.linspace(0.05, 0.95, 91):
                _bal = _bas2(_val_labels, (_yv_avg >= _thr).astype(int))
                if _bal > _best_bal:
                    _best_bal, _best_thr = _bal, float(_thr)
            if verbose:
                print(f"  Fold {fold_no:02d} — optimal threshold: {_best_thr:.2f}"
                      f"  (val n={len(_val_labels)})", flush=True)
        else:
            _best_thr = 0.5

        _ypred   = (_yp_avg >= _best_thr).astype(int)
        _metrics = compute_metrics(
            y_true=_y_true_final, y_pred=_ypred,
            y_prob=_yp_2d, n_classes=train_ds.n_classes,
        )

        # Post-hoc temperature scaling on averaged raw logits
        _avg_val_logits  = _np2.stack(_ensemble_val_logits).mean(axis=0)
        _avg_tst_logits  = _np2.stack(_ensemble_tst_logits).mean(axis=0)
        _T_post, _vp_cal = _posthoc_calibrate(_avg_val_logits, _val_labels)
        from scipy.special import softmax as _sp_sm
        _tp_cal      = _sp_sm(_avg_tst_logits / _T_post, axis=1)[:, 1]
        _thr_cal     = _youden_threshold(_vp_cal, _val_labels)
        _ypred_cal   = (_tp_cal >= _thr_cal).astype(int)
        _yp_2d_cal   = _np2.stack([1 - _tp_cal, _tp_cal], axis=1)
        _cal_metrics = compute_metrics(
            y_true=_y_true_final, y_pred=_ypred_cal,
            y_prob=_yp_2d_cal, n_classes=train_ds.n_classes,
        )

        _row = {
            "fold": fold_no, "test_subject": test_subj,
            "train_n": len(train_ds), "test_n": len(test_ds),
            "experiment": label, "gpu_id": gpu_id,
            "duration_s": 0.0, "peak_mem_gb": 0.0, "n_ensemble": _N_ENS,
            "opt_threshold"    : _best_thr,
            "T_post"           : round(_T_post, 4),
            "opt_threshold_cal": round(_thr_cal, 4),
        }
        _row.update(_metrics)
        for _k, _v in _cal_metrics.items():
            _row[f"{_k}_cal"] = _v

        if verbose:
            print(
                f"  [GPU{gpu_id}] Fold {fold_no:02d} → "
                f"Acc={_metrics['accuracy']:.4f}→{_cal_metrics['accuracy']:.4f}  "
                f"BalAcc={_metrics['balanced_acc']:.4f}→{_cal_metrics['balanced_acc']:.4f}  "
                f"ECE={_metrics.get('ece',0):.4f}→{_cal_metrics.get('ece',0):.4f}  "
                f"T_post={_T_post:.3f}  thr={_best_thr:.2f}→{_thr_cal:.2f}",
                flush=True,
            )
        return _row
    # ── End fast resume ───────────────────────────────────────────────────────

    # Per-fold class weights via selected alpha strategy
    n_cls  = train_ds.n_classes
    cw_np  = _caw(train_ds.labels, strategy=alpha_strategy, n_classes=n_cls)
    cw     = torch.tensor(cw_np, dtype=torch.float32)

    if verbose:
        print(f"  [Fold {fold_no:02d} cfg] lambda_dann={_LDANN_EFF}  "
              f"lambda_mmd={_LMMD_EFF}  mmd_mode={mmd_mode}  "
              f"norm_mode={norm_mode}  gamma={_GAMMA_EFF}  "
              f"N_ens={_N_ENS_EFF}", flush=True)

    # ── Ensemble loop ─────────────────────────────────────────────────────────
    # Train N_ENS models with distinct seeds; average softmax probabilities
    # before thresholding.  Temperature scaling is fitted per member on val_ds.
    # Raw logits are also collected for post-hoc ensemble-level calibration.
    ensemble_probs      : list = []
    ensemble_val_logits : list = []   # raw (N_val, 2) per member
    ensemble_tst_logits : list = []   # raw (N_test, 2) per member
    val_probs_list      : list = []
    val_labels                = None
    y_true_final              = None
    total_dur_s               = 0.0
    peak_mem_max              = 0.0

    for ens_idx in range(_N_ENS_EFF):
        seed_i = random_seed + ens_idx * 997
        torch.manual_seed(seed_i)
        _np.random.seed(seed_i)

        raw_model = _build_raw_model(train_ds, ablation, pretrained_eeg=pretrained_eeg)
        loss_fn   = MultiTaskLoss(
            lambda_cls          = LAMBDA_CLS,
            lambda_contrast     = LAMBDA_CONTRAST,
            lambda_roi          = LAMBDA_ROI,
            lambda_connectivity = LAMBDA_CONNECTIVITY,
            lambda_mmd          = _LMMD_EFF,
            lambda_dann         = _LDANN_EFF,
            mmd_mode            = mmd_mode,
            class_weights       = cw,
            focal_alpha         = FOCAL_ALPHA,
            focal_gamma         = _GAMMA_EFF,
        )
        trainer = Trainer(
            model        = raw_model,
            device       = device,
            loss_fn      = loss_fn,
            lr           = LR,
            weight_decay = WEIGHT_DECAY,
            patience     = PATIENCE,
            ckpt_path    = Path(ckpt_dir) / f"{label}_fold{fold_no:02d}_e{ens_idx}.pt",
            use_dp       = False,
        )

        # Match the resume checkpoint to this fold + ensemble member.
        fold_resume = None
        if resume:
            import re as _re
            _fold_m = _re.search(r'_fold(\d+)', Path(resume).stem)
            _ens_m  = _re.search(r'_e(\d+)$',  Path(resume).stem)
            _r_fold = int(_fold_m.group(1)) if _fold_m else None
            _r_ens  = int(_ens_m.group(1))  if _ens_m  else None
            if _r_fold == fold_no and (_r_ens is None or _r_ens == ens_idx):
                fold_resume = resume

        trainer.fit(
            _transform(train_loader) if _any else train_loader,
            _transform(val_loader)   if _any else val_loader,
            epochs      = epochs,
            verbose     = verbose and ens_idx == 0,
            resume_path = fold_resume,
        )

        trainer.calibrate_temperature(
            _transform(val_loader) if _any else val_loader
        )

        # Collect calibrated test probabilities P(HIGH) + raw logits
        raw_model.eval()
        probs_i   : list = []
        logits_i  : list = []
        y_true_i  : list = []
        with torch.no_grad():
            for batch in (_transform(test_loader) if _any else test_loader):
                bd = {k: v.to(device) for k, v in batch.items()}
                out = raw_model(
                    eeg_windows   = bd["eeg_windows"],
                    adj_matrices  = bd["adj_matrices"],
                    et_seq        = bd["et_seq"],
                    roi_vector    = bd["roi_vector"],
                    weighted_adjs = bd["weighted_adjs"],
                )
                p = torch.softmax(out["logits"] / trainer.temperature, dim=1)[:, 1]
                probs_i.extend(p.cpu().numpy().tolist())
                logits_i.extend(out["logits"].cpu().numpy().tolist())
                y_true_i.extend(bd["label"].cpu().numpy().tolist())

        ensemble_probs.append(_np2.array(probs_i))
        ensemble_tst_logits.append(_np2.array(logits_i))
        if y_true_final is None:
            y_true_final = _np2.array(y_true_i)

        # Collect val-set probs + raw logits for calibration
        vprobs_i  : list = []
        vlogits_i : list = []
        vlabels_i : list = []
        raw_model.eval()
        with torch.no_grad():
            for vbatch in (_transform(val_loader) if _any else val_loader):
                vbd = {k: v.to(device) for k, v in vbatch.items()}
                vout = raw_model(
                    eeg_windows   = vbd["eeg_windows"],
                    adj_matrices  = vbd["adj_matrices"],
                    et_seq        = vbd["et_seq"],
                    roi_vector    = vbd["roi_vector"],
                    weighted_adjs = vbd["weighted_adjs"],
                )
                vp = torch.softmax(vout["logits"] / trainer.temperature, dim=1)[:, 1]
                vprobs_i.extend(vp.cpu().numpy().tolist())
                vlogits_i.extend(vout["logits"].cpu().numpy().tolist())
                vlabels_i.extend(vbd["label"].cpu().numpy().tolist())
        val_probs_list.append(_np2.array(vprobs_i))
        ensemble_val_logits.append(_np2.array(vlogits_i))
        if val_labels is None:
            val_labels = _np2.array(vlabels_i)

        total_dur_s  += trainer.perf_log[-1]["cumulative_s"] if trainer.perf_log else 0.0
        peak_mem_max  = max(peak_mem_max,
                            max((e.get("peak_mem_gb", 0) for e in trainer.perf_log),
                                default=0.0))

        if verbose:
            print(f"  [GPU{gpu_id}] Fold {fold_no:02d} ens[{ens_idx}]  "
                  f"T={trainer.temperature:.3f}")

    # Average probabilities across ensemble members (original path)
    y_prob_avg   = _np2.stack(ensemble_probs).mean(axis=0)          # (N_test,)
    y_prob_2d    = _np2.stack([1 - y_prob_avg, y_prob_avg], axis=1) # (N_test, 2)

    # Per-fold threshold optimisation on val-set averaged probs (original)
    if val_probs_list and val_labels is not None:
        y_val_avg  = _np2.stack(val_probs_list).mean(axis=0)
        candidates = _np2.linspace(0.05, 0.95, 91)
        best_thr, best_bal = 0.5, -1.0
        from sklearn.metrics import balanced_accuracy_score as _bas
        for thr in candidates:
            bal = _bas(val_labels, (y_val_avg >= thr).astype(int))
            if bal > best_bal:
                best_bal, best_thr = bal, thr
        if verbose:
            print(f"  Fold {fold_no:02d} — optimal threshold: {best_thr:.2f}"
                  f"  (val n={len(val_labels)})", flush=True)
    else:
        best_thr = 0.5

    y_pred_avg    = (y_prob_avg >= best_thr).astype(int)
    final_metrics = compute_metrics(
        y_true    = y_true_final,
        y_pred    = y_pred_avg,
        y_prob    = y_prob_2d,
        n_classes = train_ds.n_classes,
    )

    # ── Post-hoc temperature scaling on averaged raw logits ──────────────────
    # Average raw (un-scaled) logits across ensemble members, then fit a single
    # T_post by minimising NLL on the validation subject.  This calibrates the
    # ensemble mixture distribution directly rather than averaging per-member
    # calibrated probabilities.
    avg_val_logits = _np2.stack(ensemble_val_logits).mean(axis=0)  # (N_val, 2)
    avg_tst_logits = _np2.stack(ensemble_tst_logits).mean(axis=0)  # (N_test, 2)

    T_post, val_prob_cal = _posthoc_calibrate(avg_val_logits, val_labels)
    from scipy.special import softmax as _sp_sm2
    tst_prob_cal   = _sp_sm2(avg_tst_logits / T_post, axis=1)[:, 1]
    thr_cal        = _youden_threshold(val_prob_cal, val_labels)
    y_pred_cal     = (tst_prob_cal >= thr_cal).astype(int)
    y_prob_2d_cal  = _np2.stack([1 - tst_prob_cal, tst_prob_cal], axis=1)
    cal_metrics    = compute_metrics(
        y_true    = y_true_final,
        y_pred    = y_pred_cal,
        y_prob    = y_prob_2d_cal,
        n_classes = train_ds.n_classes,
    )

    row = {
        "fold"              : fold_no,
        "test_subject"      : test_subj,
        "train_n"           : len(train_ds),
        "test_n"            : len(test_ds),
        "experiment"        : label,
        "gpu_id"            : gpu_id,
        "duration_s"        : round(total_dur_s, 2),
        "peak_mem_gb"       : round(peak_mem_max, 3),
        "n_ensemble"        : _N_ENS_EFF,
        "opt_threshold"     : round(float(best_thr), 4),
        "T_post"            : round(T_post, 4),
        "opt_threshold_cal" : round(thr_cal, 4),
        "alpha_strategy"    : alpha_strategy,
        "focal_gamma"       : _GAMMA_EFF,
        "lambda_dann"       : _LDANN_EFF,
        "lambda_mmd"        : _LMMD_EFF,
        "mmd_mode"          : mmd_mode,
        "norm_mode"         : norm_mode,
        "y_true"            : y_true_final.tolist(),
        "y_prob"            : y_prob_avg.tolist(),
    }
    row.update(final_metrics)
    for k, v in cal_metrics.items():
        row[f"{k}_cal"] = v

    if verbose:
        print(
            f"  [GPU{gpu_id}] Fold {fold_no:02d} → "
            f"Acc={final_metrics['accuracy']:.4f}→{cal_metrics['accuracy']:.4f}  "
            f"BalAcc={final_metrics['balanced_acc']:.4f}→{cal_metrics['balanced_acc']:.4f}  "
            f"ECE={final_metrics.get('ece',0):.4f}→{cal_metrics.get('ece',0):.4f}  "
            f"T_post={T_post:.3f}  thr={best_thr:.2f}→{thr_cal:.2f}",
            flush=True,
        )
    return row


def _build_raw_model(ds, ablation, pretrained_eeg=None):
    """Helper used inside the fold worker — avoids circular imports.

    When ``pretrained_eeg`` is given (a self-supervised EEG-branch checkpoint),
    its weights are transferred into the fresh model before fine-tuning. Only
    pass it on the TRAINING construction path — never when about to load a
    trained checkpoint for inference (it would be immediately overwritten).
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from config.settings import (
        EMBED_DIM, GAT_L1_HEAD_DIM, GAT_L1_HEADS,
        T_NHEAD, T_LAYERS, T_FF_DIM,
        ET_LSTM_HIDDEN, ET_LSTM_LAYERS,
        ROI_HIDDEN_DIM, FUSION_HEADS, CLS_HIDDEN,
        N_ROIS, N_WINDOWS, DROPOUT, TEMPERATURE,
        SNN_TIME_STEPS, SNN_HIDDEN_DIM, NS_N_RULES, NS_HIDDEN_DIM,
    )
    from models.ins_hdgs_cmt import INS_HDGS_CMT, AblationConfig, load_pretrained_eeg

    model = INS_HDGS_CMT(
        n_eeg_ch       = ds.n_eeg_ch,
        n_et_ch        = ds.n_et_ch,
        n_rois         = N_ROIS,
        n_windows      = N_WINDOWS,
        n_classes      = ds.n_classes,
        embed_dim      = EMBED_DIM,
        snn_time_steps = SNN_TIME_STEPS,
        snn_hidden_dim = SNN_HIDDEN_DIM,
        gat_head_dim   = GAT_L1_HEAD_DIM,
        gat_heads      = GAT_L1_HEADS,
        t_nhead        = T_NHEAD,
        t_layers       = T_LAYERS,
        t_ff_dim       = T_FF_DIM,
        et_lstm_hidden = ET_LSTM_HIDDEN,
        et_lstm_layers = ET_LSTM_LAYERS,
        roi_hidden     = ROI_HIDDEN_DIM,
        fusion_heads   = FUSION_HEADS,
        ns_n_rules     = NS_N_RULES,
        ns_hidden_dim  = NS_HIDDEN_DIM,
        cls_hidden     = CLS_HIDDEN,
        dropout        = DROPOUT,
        temperature    = TEMPERATURE,
        ablation       = ablation or AblationConfig.full(),
    )
    if pretrained_eeg:
        load_pretrained_eeg(model, pretrained_eeg)
    return model


# ── Benchmark Summary ────────────────────────────────────────────────────────

def benchmark_summary(
    fold_results : List[dict],
    n_gpus       : int,
    mode         : str = "DataParallel",
) -> dict:
    """
    Compute and print a benchmark summary for the completed LOSOCV run.

    Parameters
    ----------
    fold_results : list of fold result dicts (each must have 'duration_s'
                   and optionally 'peak_mem_gb', 'accuracy', 'train_n')
    n_gpus       : number of GPUs actually used
    mode         : "DataParallel" | "FoldParallel" | "SingleGPU"

    Returns
    -------
    summary : dict with speedup, utilization, memory stats, etc.
    """
    if not fold_results:
        return {}

    durations  = [r["duration_s"]   for r in fold_results if "duration_s"   in r]
    peak_mems  = [r["peak_mem_gb"]  for r in fold_results if "peak_mem_gb"  in r]
    train_sizes= [r.get("train_n", 0) for r in fold_results]

    total_wall = sum(durations)
    mean_fold  = float(np.mean(durations))  if durations else 0.0
    std_fold   = float(np.std(durations))   if durations else 0.0

    # Estimated single-GPU time: DataParallel gives near-linear speedup for
    # large batches; for fold-parallel the folds ran concurrently so total
    # elapsed ≈ max(per-fold time).  Both relative to sequential single-GPU.
    if mode == "DataParallel" and n_gpus > 1:
        est_single = total_wall * n_gpus
        speedup    = est_single / max(total_wall, 1e-9)
    elif mode == "FoldParallel" and n_gpus > 1:
        # Folds ran in parallel; sequential single-GPU ≈ sum of fold times
        est_single = total_wall           # total_wall already reflects parallelism
        # sequential time = sum / n_parallel_batches… approximate here
        speedup    = float(n_gpus)        # ideal parallel speedup
    else:
        est_single = total_wall
        speedup    = 1.0

    throughput = sum(train_sizes) / max(total_wall, 1e-9)   # samples/s

    summary = {
        "mode"               : mode,
        "n_folds"            : len(fold_results),
        "n_gpus"             : n_gpus,
        "total_wall_time_s"  : round(total_wall, 2),
        "mean_fold_time_s"   : round(mean_fold,  2),
        "std_fold_time_s"    : round(std_fold,   2),
        "est_single_gpu_s"   : round(est_single, 2),
        "speedup_vs_1gpu"    : round(speedup, 2),
        "throughput_samp_s"  : round(throughput, 1),
        "mean_peak_mem_gb"   : round(float(np.mean(peak_mems)),  3) if peak_mems else 0.0,
        "max_peak_mem_gb"    : round(float(np.max(peak_mems)),   3) if peak_mems else 0.0,
        "min_fold_time_s"    : round(float(np.min(durations)),   2) if durations else 0.0,
        "max_fold_time_s"    : round(float(np.max(durations)),   2) if durations else 0.0,
    }

    # ── Print ────────────────────────────────────────────────
    w = 58
    print(f"\n  {'═'*w}")
    print(f"  Benchmark Summary — {mode}  ({n_gpus} GPU{'s' if n_gpus>1 else ''})")
    print(f"  {'═'*w}")
    print(f"  Folds completed      : {summary['n_folds']}")
    print(f"  Total wall time      : {summary['total_wall_time_s']:.1f} s")
    print(f"  Mean fold time       : {summary['mean_fold_time_s']:.1f} ± "
          f"{summary['std_fold_time_s']:.1f} s")
    print(f"  Fastest / slowest    : {summary['min_fold_time_s']:.1f} s  /  "
          f"{summary['max_fold_time_s']:.1f} s")
    print(f"  Est. single-GPU time : {summary['est_single_gpu_s']:.1f} s")
    print(f"  Speedup vs 1 GPU     : {summary['speedup_vs_1gpu']:.2f}×")
    print(f"  Throughput           : {summary['throughput_samp_s']:.0f} samples/s")
    if peak_mems:
        print(f"  Peak GPU mem (mean)  : {summary['mean_peak_mem_gb']:.2f} GB")
        print(f"  Peak GPU mem (max)   : {summary['max_peak_mem_gb']:.2f} GB")
    print(f"  {'═'*w}\n")

    return summary


# ── DistributedDataParallel stubs ────────────────────────────────────────────

def ddp_init(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Initialise the default process group for DDP.

    Call at the start of each worker spawned by torchrun or
    torch.multiprocessing.spawn.

    Example launch command:
        torchrun --nproc_per_node=8 main.py --ddp
    """
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def ddp_cleanup() -> None:
    """Destroy the default process group after DDP training."""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model: nn.Module, rank: int) -> nn.Module:
    """
    Wrap *model* in DistributedDataParallel on *rank*.

    Prerequisites: ddp_init() must have been called in this process.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = model.to(torch.device(f"cuda:{rank}"))
    return DDP(model, device_ids=[rank], output_device=rank)
