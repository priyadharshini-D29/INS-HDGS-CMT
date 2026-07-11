"""
================================================================
INS-HDGS-CMT — Leave-One-Subject-Out Cross Validation
Multi-GPU Edition
================================================================
Subject-isolated LOSOCV for cognitive engagement decoding.

Task: HIGH_ENGAGEMENT (1) vs LOW_ENGAGEMENT (0)

Execution modes:
  DataParallel  (default)  : all GPUs per fold, folds sequential
  Fold-parallel (opt.)     : one GPU per fold group, concurrent

Metrics per fold:
  Accuracy, F1, Balanced Accuracy, ROC-AUC, PR-AUC,
  Cohen's Kappa, MCC, ECE

Expected LOSOCV performance (realistic, no leakage):
  Accuracy   : 70 – 85%
  F1         : 0.70 – 0.85
  Kappa      : 0.60 – 0.80
  ROC-AUC    : 0.80 – 0.92
================================================================
"""

from __future__ import annotations

import re
import sys
import time
import multiprocessing
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    SUBJECT_IDS, EMBED_DIM,
    GAT_L1_HEAD_DIM, GAT_L1_HEADS,
    T_NHEAD, T_LAYERS, T_FF_DIM,
    ET_LSTM_HIDDEN, ET_LSTM_LAYERS,
    ROI_HIDDEN_DIM, FUSION_HEADS, CLS_HIDDEN,
    N_ROIS, N_WINDOWS, ET_INPUT_DIM,
    SNN_TIME_STEPS, SNN_HIDDEN_DIM,
    NS_N_RULES, NS_HIDDEN_DIM,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    TEMPERATURE, DROPOUT, FOCAL_ALPHA, FOCAL_GAMMA,
    LAMBDA_CLS, LAMBDA_CONTRAST, LAMBDA_ROI,
    LAMBDA_CONNECTIVITY, LAMBDA_MMD, LAMBDA_SECONDARY,
    DEVICE, CKPT_DIR, METRICS_DIR, BENCHMARK_DIR,
    NUM_GPUS, USE_DP, FOLD_PARALLEL, RANDOM_SEED,
    ENGAGEMENT_CLASS_NAMES, N_ENSEMBLE,
    IMBALANCE_SKIP_RATIO,
)
from data.dataset    import NeumaGraphDataset, build_dataloaders, _normalize_subject_id
from models.ins_hdgs_cmt import INS_HDGS_CMT, AblationConfig, load_pretrained_eeg
from training.losses     import MultiTaskLoss, compute_alpha_weights
from training.trainer    import Trainer
from training.metrics    import compute_metrics, print_losocv_summary
from utils.gpu import (
    get_fold_gpu_map, print_fold_gpu_map,
    benchmark_summary, log_gpu_memory,
)


# ── Resume helpers ────────────────────────────────────────────────────────────

def _parse_resume_key(path: str):
    """Parse (fold_no, ens_idx) from a checkpoint filename.

    Examples
    --------
    "ins_hdgs_cmt_fold12.pt"    → (12, None)
    "ins_hdgs_cmt_fold12_e0.pt" → (12, 0)

    Returns (None, None) if no fold number is found.
    """
    stem    = Path(path).stem
    fold_m  = re.search(r'_fold(\d+)', stem)
    ens_m   = re.search(r'_e(\d+)$', stem)
    fold_no = int(fold_m.group(1)) if fold_m else None
    ens_idx = int(ens_m.group(1)) if ens_m else None
    return fold_no, ens_idx


# ── Fold-optimal threshold ────────────────────────────────────────────────────

def _find_optimal_threshold(
    val_probs : np.ndarray,
    val_labels: np.ndarray,
    metric    : str = "balanced_acc",
) -> float:
    """
    Find the decision threshold via Youden's J statistic (TPR - FPR).

    Small-sample guard: Youden's J on fewer than MIN_YOUDEN_N samples
    has a 95% CI width of ~±0.5 on a 6-point ROC curve — effectively
    random.  When the val set is too small, return 0.5 (optimal for
    balanced binary classification with calibrated probabilities).

    When val set is sufficient, Winsorise the result to [0.30, 0.70] to
    prevent extreme thresholds that collapse F1 to zero on the test set.
    """
    from sklearn.metrics import roc_curve

    MIN_YOUDEN_N = 20   # minimum val samples for reliable Youden's J

    if len(val_labels) < 4 or len(np.unique(val_labels)) < 2:
        return 0.5

    # Too few samples for reliable threshold estimation → use balanced default
    if len(val_labels) < MIN_YOUDEN_N:
        return 0.5

    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    j_scores = tpr - fpr
    best_idx  = int(np.argmax(j_scores))
    thr       = float(thresholds[best_idx])
    # Winsorise: thresholds outside [0.30, 0.70] on small val sets are
    # artefacts of probability skew, not genuine optimal operating points.
    return round(float(np.clip(thr, 0.30, 0.70)), 4)


# ── Post-hoc Temperature Calibration ─────────────────────────────────────────

def _calibrate_temperature_posthoc(
    logits_2d : np.ndarray,
    labels    : np.ndarray,
    T_min     : float = 0.05,
    T_max     : float = 5.0,
):
    """Fit scalar T on averaged ensemble logits (N,2) by minimising NLL.

    Returns (T_post, p_cal) where p_cal is calibrated P(HIGH).
    Falls back to T=1 when val set has < 4 samples or single-class labels.
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


# ── Model Factory ─────────────────────────────────────────────────────────────

def _make_model(
    n_eeg_ch  : int,
    n_classes : int,
    ablation  : Optional[AblationConfig] = None,
    n_et_ch   : Optional[int]            = None,
    n_rois    : Optional[int]            = None,
    pretrained_eeg : Optional[str]       = None,
) -> INS_HDGS_CMT:
    model = INS_HDGS_CMT(
        n_eeg_ch        = n_eeg_ch,
        n_et_ch         = n_et_ch or ET_INPUT_DIM,
        n_rois          = n_rois  or N_ROIS,
        n_windows       = N_WINDOWS,
        n_classes       = n_classes,
        embed_dim       = EMBED_DIM,
        snn_time_steps  = SNN_TIME_STEPS,
        snn_hidden_dim  = SNN_HIDDEN_DIM,
        gat_head_dim    = GAT_L1_HEAD_DIM,
        gat_heads       = GAT_L1_HEADS,
        t_nhead         = T_NHEAD,
        t_layers        = T_LAYERS,
        t_ff_dim        = T_FF_DIM,
        et_lstm_hidden  = ET_LSTM_HIDDEN,
        et_lstm_layers  = ET_LSTM_LAYERS,
        roi_hidden      = ROI_HIDDEN_DIM,
        fusion_heads    = FUSION_HEADS,
        ns_n_rules      = NS_N_RULES,
        ns_hidden_dim   = NS_HIDDEN_DIM,
        cls_hidden      = CLS_HIDDEN,
        dropout         = DROPOUT,
        temperature     = TEMPERATURE,
        ablation        = ablation or AblationConfig.full(),
    )
    if pretrained_eeg:
        load_pretrained_eeg(model, pretrained_eeg)
    return model


# ── Main LOSOCV Entry-Point ───────────────────────────────────────────────────

def run_losocv(
    subject_ids   : Optional[List[str]]     = None,
    ablation      : Optional[AblationConfig] = None,
    epochs        : int  = EPOCHS,
    batch_size    : int  = BATCH_SIZE,
    label         : str  = "ins_hdgs_cmt",
    save_dir      : Optional[Path] = None,
    verbose       : bool = True,
    fold_parallel : bool = FOLD_PARALLEL,
    random_labels : bool = False,
    zero_roi      : bool = False,
    zero_graph    : bool = False,
    zero_et       : bool = False,
    identity_graph: bool = False,
    random_seed   : int  = RANDOM_SEED,
    resume        : Optional[str] = None,
    alpha_strategy      : str            = "balanced",
    focal_gamma_override: Optional[float] = None,
    n_ensemble_override : Optional[int]   = None,
    lambda_dann_override: Optional[float] = None,
    lambda_mmd_override : Optional[float] = None,
    mmd_mode            : str            = "marginal",
    norm_mode           : str            = "zscore",
    pretrained_eeg      : Optional[str]  = None,
) -> pd.DataFrame:
    """
    Run full LOSOCV for cognitive engagement decoding.

    Returns
    -------
    results_df : DataFrame (one row per fold)
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    save_dir = Path(save_dir or (METRICS_DIR / label))
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CKPT_DIR / label
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ablation = ablation or AblationConfig.full()

    if subject_ids is None:
        subject_ids = SUBJECT_IDS
    subject_ids = [_normalize_subject_id(s) for s in subject_ids]

    # Discover available subjects
    available: List[str] = []
    for sid in subject_ids:
        try:
            probe = NeumaGraphDataset(subject_ids=[sid], precompute_graphs=False)
            if len(probe) > 0:
                available.append(sid)
        except FileNotFoundError:
            pass

    if not available:
        raise FileNotFoundError("No subject data found. Run Phase 3 first.")

    if len(available) == 1:
        if verbose:
            print("  [LOSOCV] Single subject — 70/30 stratified split.")
        return _single_subject_run(
            available[0], ablation, epochs, batch_size,
            ckpt_dir, save_dir, label, verbose,
        )

    if verbose:
        print(f"\n  [INS-HDGS-CMT LOSOCV]  {len(available)} subjects")
        print(f"  Task: HIGH_ENGAGEMENT vs LOW_ENGAGEMENT")
        if random_labels:
            print("  [SANITY] random_labels=True — expect ≈ chance")

    assert len(set(available)) == len(available)

    _extra = dict(
        alpha_strategy       = alpha_strategy,
        focal_gamma_override = focal_gamma_override,
        n_ensemble_override  = n_ensemble_override,
        lambda_dann_override = lambda_dann_override,
        lambda_mmd_override  = lambda_mmd_override,
        mmd_mode             = mmd_mode,
        norm_mode            = norm_mode,
        pretrained_eeg       = pretrained_eeg,
    )
    if fold_parallel and NUM_GPUS > 1:
        all_rows = _run_fold_parallel(
            available, ablation, epochs, batch_size,
            label=label, ckpt_dir=ckpt_dir,
            random_labels=random_labels, zero_roi=zero_roi,
            zero_graph=zero_graph, zero_et=zero_et,
            identity_graph=identity_graph, random_seed=random_seed,
            verbose=verbose, resume=resume, **_extra,
        )
    else:
        all_rows = _run_fold_sequential(
            available, ablation, epochs, batch_size,
            label=label, ckpt_dir=ckpt_dir,
            random_labels=random_labels, zero_roi=zero_roi,
            zero_graph=zero_graph, zero_et=zero_et,
            identity_graph=identity_graph, random_seed=random_seed,
            verbose=verbose, resume=resume, **_extra,
        )

    if not all_rows:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_rows)
    out_path   = save_dir / f"losocv_{label}.csv"
    results_df.to_csv(out_path, index=False)

    if verbose:
        print_losocv_summary(results_df, label)

    _save_benchmark(all_rows, label, fold_parallel)
    return results_df


# ── Sequential Folds ──────────────────────────────────────────────────────────
def _run_fold_sequential(
    available, ablation, epochs, batch_size, *,
    label, ckpt_dir,
    random_labels, zero_roi, zero_graph, zero_et, identity_graph,
    random_seed, verbose, resume=None,
    alpha_strategy       : str            = "balanced",
    focal_gamma_override : Optional[float] = None,
    n_ensemble_override  : Optional[int]   = None,
    lambda_dann_override : Optional[float] = None,
    lambda_mmd_override  : Optional[float] = None,
    mmd_mode             : str            = "marginal",
    norm_mode            : str            = "zscore",
    pretrained_eeg       : Optional[str]  = None,
) -> list:

    rng      = np.random.default_rng(random_seed)
    all_rows = []

    for fold_idx, test_subj in enumerate(available):

        fold_no    = fold_idx + 1
        # ==========================================================
        # TRAIN / VAL / TEST SPLIT
        # ==========================================================

        candidate_train = [
            s for s in available
            if s != test_subj
        ]

        rng.shuffle(candidate_train)

        if len(candidate_train) < 2:
            print(f"[WARN] Fold {fold_no}: not enough subjects for train/val split "
                  f"(n={len(candidate_train)}) — skipping", flush=True)
            continue

        val_subj   = candidate_train[0]
        train_subs = candidate_train[1:]

        if len(train_subs) == 0:
            print(f"[WARN] Fold {fold_no}: empty train_subs after split — skipping",
                  flush=True)
            continue

        assert test_subj not in train_subs
        assert val_subj  not in train_subs

        if verbose:
            print(f"  [Fold {fold_no:02d}] TEST={test_subj} | VAL={val_subj} | "
                  f"TRAIN={len(train_subs)} subjects", flush=True)

        if verbose:

            print(f"  Validation Subject: {val_subj}")

        if verbose:

            print(f"\n{'─'*58}")

            print(
                f"  FOLD {fold_no:02d}/{len(available):02d} "
                f"| Test: {test_subj}  Train: {train_subs}"
            )

        fold_start = time.perf_counter()

        try:

            train_ds = NeumaGraphDataset(
                subject_ids=train_subs,
                precompute_graphs=True,
                augment=True,
                norm_mode=norm_mode,
            )

            val_ds = NeumaGraphDataset(
                subject_ids=[val_subj],
                precompute_graphs=True,
                norm_mode=norm_mode,
            )

            test_ds = NeumaGraphDataset(
                subject_ids=[test_subj],
                precompute_graphs=True,
                norm_mode=norm_mode,
            )

        except FileNotFoundError:

            if verbose:
                print(f"  [SKIP] Missing data fold {fold_no}")

            continue

        # ==========================================================
        # STRICT ISOLATION CHECK
        # ==========================================================

        shared = train_ds._source_dirs & test_ds._source_dirs

        if shared and (
            train_ds._has_leakage_risk
            or test_ds._has_leakage_risk
        ):

            print(f"\n  *** LEAKAGE WARNING *** fold {fold_no}")

            for d in sorted(shared):
                print(f"    shared dir: {d}")

        # ==========================================================
        # MINIMUM DATA CHECK
        # ==========================================================

        if len(train_ds) < batch_size or len(test_ds) < 2:

            if verbose:

                print(
                    f"  [SKIP] insufficient data "
                    f"(train={len(train_ds)}, test={len(test_ds)})"
                )

            continue

        # ==========================================================
        # SINGLE-CLASS TEST CHECK
        # Subjects whose global-threshold labels are all-HIGH or all-LOW
        # (S16, S31, S33, S41, S44) produce undefined AUC/MCC — skip them.
        # They still contribute to other folds' training sets.
        # ==========================================================

        te_unique = np.unique(test_ds.labels)
        if len(te_unique) < 2:
            cls_name = "ALL-HIGH" if int(te_unique[0]) == 1 else "ALL-LOW"
            if verbose:
                print(
                    f"  [SKIP] Fold {fold_no:02d} — test subject {test_subj} "
                    f"has single-class labels ({cls_name}) under global threshold. "
                    f"AUC/MCC undefined. Subject remains in other folds' training sets."
                )
            continue

        # Skip test folds with severe label imbalance (minority < IMBALANCE_SKIP_RATIO).
        # Such subjects have degenerate AUC/MCC even when both classes are present.
        te_counts    = np.bincount(test_ds.labels, minlength=2)
        te_min_ratio = te_counts.min() / te_counts.sum()
        if te_min_ratio < IMBALANCE_SKIP_RATIO:
            if verbose:
                dom_cls  = ENGAGEMENT_CLASS_NAMES[int(te_counts.argmax())]
                print(
                    f"  [SKIP] Fold {fold_no:02d} — test subject {test_subj} "
                    f"has severe imbalance ({te_min_ratio:.1%} minority, "
                    f"dominant={dom_cls}). "
                    f"Subject remains in other folds' training sets."
                )
            continue

        # ==========================================================
        # RANDOM LABEL SANITY
        # ==========================================================

        if random_labels:

            train_ds.labels = rng.permutation(train_ds.labels)

        # ==========================================================
        # PRINT DISTRIBUTION
        # ==========================================================

        if verbose:

            tr_u, tr_c = np.unique(
                train_ds.labels,
                return_counts=True
            )

            te_u, te_c = np.unique(
                test_ds.labels,
                return_counts=True
            )

            tr_dist = {
                ENGAGEMENT_CLASS_NAMES[k]: v
                for k, v in zip(
                    tr_u.tolist(),
                    tr_c.tolist()
                )
            }

            te_dist = {
                ENGAGEMENT_CLASS_NAMES[k]: v
                for k, v in zip(
                    te_u.tolist(),
                    te_c.tolist()
                )
            }

            print(f"  Train: {tr_dist}")
            print(f"  Test : {te_dist}")

        # ==========================================================
        # CLASS WEIGHTS
        # ==========================================================

        n_cls    = train_ds.n_classes
        _gamma   = focal_gamma_override if focal_gamma_override is not None else FOCAL_GAMMA
        _n_ens   = n_ensemble_override  if n_ensemble_override  is not None else N_ENSEMBLE
        _ldann   = lambda_dann_override if lambda_dann_override is not None else 0.10
        _lmmd    = lambda_mmd_override  if lambda_mmd_override  is not None else LAMBDA_MMD
        cw_np    = compute_alpha_weights(train_ds.labels,
                                         strategy=alpha_strategy, n_classes=n_cls)
        cw       = torch.tensor(cw_np, dtype=torch.float32)

        # ==========================================================
        # DATALOADERS
        # ==========================================================

        train_loader, val_loader = build_dataloaders(
            train_ds,
            val_ds,
            batch_size=batch_size
        )

        _, test_loader = build_dataloaders(
            train_ds,
            test_ds,
            batch_size=batch_size
        )

        # ==========================================================
        # MODEL
        # ==========================================================

        model = _make_model(
            train_ds.n_eeg_ch,
            train_ds.n_classes,
            ablation,
            train_ds.n_et_ch,
            pretrained_eeg=pretrained_eeg,
        )

        # ==========================================================
        # LOSS
        # ==========================================================

        loss_fn = MultiTaskLoss(

            lambda_cls          = LAMBDA_CLS,
            lambda_contrast     = LAMBDA_CONTRAST,
            lambda_roi          = LAMBDA_ROI,
            lambda_connectivity = LAMBDA_CONNECTIVITY,
            lambda_mmd          = _lmmd,
            lambda_dann         = _ldann,
            mmd_mode            = mmd_mode,

            class_weights = cw,
            focal_alpha   = FOCAL_ALPHA,
            focal_gamma   = _gamma,
        )

        # ==========================================================
        # TRAINER
        # ==========================================================

        trainer = Trainer(

            model        = model,
            device       = DEVICE,
            loss_fn      = loss_fn,
            lr           = LR,
            weight_decay = WEIGHT_DECAY,
            patience     = PATIENCE,

            ckpt_path = (
                ckpt_dir /
                f"{label}_fold{fold_no:02d}.pt"
            ),

            use_dp = USE_DP,
        )

        # ==========================================================
        # OPTIONAL ABLATIONS
        # ==========================================================

        _any = (
            zero_roi
            or zero_graph
            or zero_et
            or identity_graph
        )

        def _transform(loader):

            for batch in loader:

                if zero_roi:

                    batch["roi_vector"] = torch.zeros_like(
                        batch["roi_vector"]
                    )

                if zero_graph:

                    batch["adj_matrices"] = torch.zeros_like(
                        batch["adj_matrices"]
                    )

                if zero_et:

                    batch["et_seq"] = torch.zeros_like(
                        batch["et_seq"]
                    )

                if identity_graph:

                    B, W, N, _ = batch["adj_matrices"].shape

                    eye = torch.eye(
                        N,
                        device=batch["adj_matrices"].device
                    )

                    batch["adj_matrices"] = (
                        eye.unsqueeze(0)
                        .unsqueeze(0)
                        .expand(B, W, N, N)
                        .clone()
                    )

                yield batch

        # ==========================================================
        # ENSEMBLE TRAIN + CALIBRATE + COLLECT PROBS
        # ==========================================================
        # Train N_ENSEMBLE models with distinct seeds. Temperature-
        # calibrate each on val_ds. Average P(HIGH) before thresholding.

        ensemble_probs      : list = []
        ensemble_val_probs  : list = []   # per-member calibrated P(HIGH) on val
        ensemble_val_logits : list = []   # raw (N_val, 2) logits per member
        ensemble_tst_logits : list = []   # raw (N_test, 2) logits per member
        y_true     : list = []
        y_val_true : list = []
        total_dur_s  = 0.0

        for ens_idx in range(_n_ens):

            seed_i = RANDOM_SEED + ens_idx * 997
            torch.manual_seed(seed_i)
            rng_ens = np.random.default_rng(seed_i)

            model = _make_model(
                train_ds.n_eeg_ch,
                train_ds.n_classes,
                ablation,
                train_ds.n_et_ch,
                pretrained_eeg=pretrained_eeg,
            )

            loss_fn = MultiTaskLoss(
                lambda_cls          = LAMBDA_CLS,
                lambda_contrast     = LAMBDA_CONTRAST,
                lambda_roi          = LAMBDA_ROI,
                lambda_connectivity = LAMBDA_CONNECTIVITY,
                lambda_mmd          = _lmmd,
                lambda_dann         = _ldann,
                mmd_mode            = mmd_mode,
                class_weights = cw,
                focal_alpha   = FOCAL_ALPHA,
                focal_gamma   = _gamma,
            )

            trainer = Trainer(
                model        = model,
                device       = DEVICE,
                loss_fn      = loss_fn,
                lr           = LR,
                weight_decay = WEIGHT_DECAY,
                patience     = PATIENCE,
                ckpt_path = (
                    ckpt_dir /
                    f"{label}_fold{fold_no:02d}_e{ens_idx}.pt"
                ),
                use_dp = USE_DP,
            )

            # Match the resume checkpoint to this fold + ensemble member.
            fold_resume = None
            if resume:
                r_fold, r_ens = _parse_resume_key(resume)
                if r_fold == fold_no and (r_ens is None or r_ens == ens_idx):
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

            total_dur_s += sum(e["duration_s"] for e in trainer.perf_log)

            # Collect calibrated P(HIGH) + raw logits on val set
            model.eval()
            val_probs_i  : list = []
            val_logits_i : list = []
            with torch.no_grad():
                for batch in (
                    _transform(val_loader) if _any else val_loader
                ):
                    out = model(
                        eeg_windows   = batch["eeg_windows"].to(DEVICE),
                        adj_matrices  = batch["adj_matrices"].to(DEVICE),
                        et_seq        = batch["et_seq"].to(DEVICE),
                        roi_vector    = batch["roi_vector"].to(DEVICE),
                        weighted_adjs = batch["weighted_adjs"].to(DEVICE),
                    )
                    p = torch.softmax(
                        out["logits"] / trainer.temperature, dim=1
                    )[:, 1]
                    val_probs_i.extend(p.cpu().numpy().tolist())
                    val_logits_i.extend(out["logits"].cpu().numpy().tolist())
                    if ens_idx == 0:
                        y_val_true.extend(
                            batch["label"].cpu().numpy().tolist()
                        )
            ensemble_val_probs.append(np.array(val_probs_i))
            ensemble_val_logits.append(np.array(val_logits_i))

            # Collect calibrated P(HIGH) + raw logits on test set
            model.eval()
            probs_i   : list = []
            logits_i  : list = []

            with torch.no_grad():

                for batch in (
                    _transform(test_loader) if _any else test_loader
                ):
                    out = model(
                        eeg_windows   = batch["eeg_windows"].to(DEVICE),
                        adj_matrices  = batch["adj_matrices"].to(DEVICE),
                        et_seq        = batch["et_seq"].to(DEVICE),
                        roi_vector    = batch["roi_vector"].to(DEVICE),
                        weighted_adjs = batch["weighted_adjs"].to(DEVICE),
                    )
                    p = torch.softmax(
                        out["logits"] / trainer.temperature, dim=1
                    )[:, 1]
                    probs_i.extend(p.cpu().numpy().tolist())
                    logits_i.extend(out["logits"].cpu().numpy().tolist())
                    if ens_idx == 0:
                        y_true.extend(
                            batch["label"].cpu().numpy().tolist()
                        )

            ensemble_probs.append(np.array(probs_i))
            ensemble_tst_logits.append(np.array(logits_i))

            if verbose:
                print(f"  Fold {fold_no:02d} ens[{ens_idx}]  "
                      f"T={trainer.temperature:.3f}")

        # Average probabilities across ensemble members (original path)
        y_prob_avg = np.stack(ensemble_probs).mean(axis=0)

        # Fold-optimal threshold via Youden's J on val subject (original)
        val_prob_avg  = np.stack(ensemble_val_probs).mean(axis=0)
        opt_threshold = _find_optimal_threshold(
            val_prob_avg, np.array(y_val_true), metric="balanced_acc"
        )
        if verbose:
            print(f"  Fold {fold_no:02d} — optimal threshold: {opt_threshold:.2f} "
                  f"(val n={len(y_val_true)})")

        y_pred    = (y_prob_avg >= opt_threshold).astype(int).tolist()
        y_prob    = y_prob_avg.tolist()
        y_prob_2d = np.stack([1 - y_prob_avg, y_prob_avg], axis=1)
        final_m   = compute_metrics(
            y_true    = np.array(y_true),
            y_pred    = np.array(y_pred),
            y_prob    = y_prob_2d,
            n_classes = train_ds.n_classes,
        )

        # ── Post-hoc temperature scaling on averaged raw logits ───────────
        # Average raw logits across members, then fit a single T_post by
        # minimising NLL on the validation subject.
        avg_val_logits = np.stack(ensemble_val_logits).mean(axis=0)  # (N_val, 2)
        avg_tst_logits = np.stack(ensemble_tst_logits).mean(axis=0)  # (N_test, 2)
        T_post, val_prob_cal = _calibrate_temperature_posthoc(
            avg_val_logits, np.array(y_val_true)
        )
        from scipy.special import softmax as _sp_softmax
        tst_prob_cal  = _sp_softmax(avg_tst_logits / T_post, axis=1)[:, 1]
        opt_thr_cal   = _find_optimal_threshold(val_prob_cal, np.array(y_val_true))
        y_pred_cal    = (tst_prob_cal >= opt_thr_cal).astype(int).tolist()
        y_prob_2d_cal = np.stack([1 - tst_prob_cal, tst_prob_cal], axis=1)
        cal_m         = compute_metrics(
            y_true    = np.array(y_true),
            y_pred    = np.array(y_pred_cal),
            y_prob    = y_prob_2d_cal,
            n_classes = train_ds.n_classes,
        )
        if verbose:
            print(
                f"  Fold {fold_no:02d} — T_post={T_post:.3f}  "
                f"thr_cal={opt_thr_cal:.2f}  "
                f"BalAcc {final_m.get('balanced_acc',0):.4f}→{cal_m.get('balanced_acc',0):.4f}  "
                f"ECE {final_m.get('ece',0):.4f}→{cal_m.get('ece',0):.4f}"
            )

        # ==========================================================
        # EVALUATE  (kept for compatibility — original metrics unchanged)
        # ==========================================================

        # ==========================================================
        # METADATA
        # ==========================================================

        fold_dur = time.perf_counter() - fold_start

        row = {
            "fold"             : fold_no,
            "test_subject"     : test_subj,
            "train_n"          : len(train_ds),
            "test_n"           : len(test_ds),
            "experiment"       : label,
            "task"             : "HIGH_vs_LOW_ENGAGEMENT",
            "gpu_mode"         : f"DataParallel×{NUM_GPUS}",
            "n_ensemble"       : N_ENSEMBLE,
            "duration_s"       : round(fold_dur, 2),
            "peak_mem_gb"      : 0.0,
            "opt_threshold"    : opt_threshold,
            "T_post"           : round(T_post, 4),
            "opt_threshold_cal": round(opt_thr_cal, 4),
        }

        # ==========================================================
        # SAVE PREDICTIONS
        # ==========================================================

        row["y_true"] = y_true
        row["y_prob"] = y_prob
        row["y_pred"] = y_pred

        # ==========================================================
        # SAVE METRICS  (original + calibrated)
        # ==========================================================

        row.update(final_m)
        for _k, _v in cal_m.items():
            row[f"{_k}_cal"] = _v

        all_rows.append(row)

        if verbose:
            print(
                f"  Fold {fold_no:02d} → "
                f"Acc={final_m.get('accuracy', 0):.4f}  "
                f"BalAcc={final_m.get('balanced_acc', 0):.4f}→{cal_m.get('balanced_acc', 0):.4f}  "
                f"AUC={final_m.get('roc_auc', 0):.4f}  "
                f"ECE={final_m.get('ece', 0):.4f}→{cal_m.get('ece', 0):.4f}  "
                f"T_post={T_post:.3f}  t={fold_dur:.1f}s"
            )

        del (
            train_ds,
            val_ds,
            test_ds,
            train_loader,
            val_loader,
            test_loader,
            model,
            trainer,
            loss_fn,
        )

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return all_rows

# ── Parallel Folds ────────────────────────────────────────────────────────────

def _run_fold_parallel(
    available, ablation, epochs, batch_size, *,
    label, ckpt_dir,
    random_labels, zero_roi, zero_graph, zero_et, identity_graph,
    random_seed, verbose, resume=None,
    alpha_strategy       : str            = "balanced",
    focal_gamma_override : Optional[float] = None,
    n_ensemble_override  : Optional[int]   = None,
    lambda_dann_override : Optional[float] = None,
    lambda_mmd_override  : Optional[float] = None,
    mmd_mode             : str            = "marginal",
    norm_mode            : str            = "zscore",
    pretrained_eeg       : Optional[str]  = None,
) -> list:
    n_folds  = len(available)
    n_gpus   = min(NUM_GPUS, n_folds)
    fold_map = get_fold_gpu_map(n_folds, n_gpus)

    if verbose:
        print_fold_gpu_map(fold_map)

    rng_par = np.random.default_rng(random_seed)
    fold_cfgs = []
    for fold_idx, test_subj in enumerate(available):
        fold_no         = fold_idx + 1
        candidate_train = [s for s in available if s != test_subj]
        rng_par.shuffle(candidate_train)
        if len(candidate_train) < 2:
            print(f"[WARN] Fold {fold_no}: not enough subjects for train/val split "
                  f"(n={len(candidate_train)}) — skipping", flush=True)
            continue
        val_subj   = candidate_train[0]
        train_subs = candidate_train[1:]
        if len(train_subs) == 0:
            print(f"[WARN] Fold {fold_no}: empty train_subs after split — skipping",
                  flush=True)
            continue
        assert test_subj not in train_subs
        assert val_subj  not in train_subs
        print(f"  [Fold {fold_no:02d}] TEST={test_subj} | VAL={val_subj} | "
              f"TRAIN={len(train_subs)} subjects", flush=True)
        fold_cfgs.append(dict(
            gpu_id=fold_map[fold_no], test_subj=test_subj,
            val_subj=val_subj, train_subs=train_subs,
            fold_no=fold_no, ablation=ablation,
            epochs=epochs, batch_size=batch_size, label=label,
            ckpt_dir=str(ckpt_dir), random_labels=random_labels,
            zero_roi=zero_roi, zero_graph=zero_graph, zero_et=zero_et,
            identity_graph=identity_graph, random_seed=random_seed,
            verbose=verbose, resume=resume,
            alpha_strategy       = alpha_strategy,
            focal_gamma_override = focal_gamma_override,
            n_ensemble_override  = n_ensemble_override,
            lambda_dann_override = lambda_dann_override,
            lambda_mmd_override  = lambda_mmd_override,
            mmd_mode             = mmd_mode,
            norm_mode            = norm_mode,
            pretrained_eeg       = pretrained_eeg,
        ))

    # One process per GPU: each handles all folds assigned to that GPU.
    # CUDA_VISIBLE_DEVICES is set inside the subprocess BEFORE torch is
    # imported, so each process's CUDA context sees exactly one GPU.
    from collections import defaultdict
    from _fold_worker_entry import run_gpu_worker

    folds_by_gpu: dict = defaultdict(list)
    for cfg in fold_cfgs:
        folds_by_gpu[cfg["gpu_id"]].append(cfg)

    ctx         = multiprocessing.get_context("spawn")
    manager     = ctx.Manager()
    result_list = manager.list()
    lock        = manager.Lock()

    import os as _os
    _orig_cvd = _os.environ.get("CUDA_VISIBLE_DEVICES")

    processes = []
    for gpu_id in range(n_gpus):
        p = ctx.Process(
            target = run_gpu_worker,
            args   = (gpu_id, folds_by_gpu[gpu_id], result_list, lock),
        )
        # Set CVD in the parent immediately before start() so the spawned
        # child inherits the correct single-GPU value at exec time.
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        p.start()
        processes.append(p)

    # Restore parent's original CVD now that all children have been launched.
    if _orig_cvd is not None:
        _os.environ["CUDA_VISIBLE_DEVICES"] = _orig_cvd
    else:
        _os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    for p in processes:
        p.join()

    all_rows = list(result_list)
    return [r for r in all_rows if r is not None]


# ── Single-Subject Fallback ───────────────────────────────────────────────────

def _single_subject_run(
    subject_id, ablation, epochs, batch_size,
    ckpt_dir, save_dir, label, verbose,
):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset, DataLoader
    from data.dataset import collate_fn

    full_ds = NeumaGraphDataset(subject_ids=[subject_id], precompute_graphs=True)
    n   = len(full_ds)
    idx = np.arange(n)
    try:
        tr_idx, te_idx = train_test_split(
            idx, test_size=0.30, stratify=full_ds.labels, random_state=42
        )
    except Exception:
        tr_idx, te_idx = train_test_split(idx, test_size=0.30, random_state=42)

    bs           = min(batch_size, len(tr_idx))
    train_loader = DataLoader(Subset(full_ds, tr_idx), batch_size=bs,
                              shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(Subset(full_ds, te_idx), batch_size=bs,
                              shuffle=False, collate_fn=collate_fn)

    cw    = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(full_ds.n_classes),
                             y=full_ds.labels[tr_idx]),
        dtype=torch.float32,
    )
    model   = _make_model(full_ds.n_eeg_ch, full_ds.n_classes, ablation,
                          full_ds.n_et_ch).to(DEVICE)
    loss_fn = MultiTaskLoss(LAMBDA_CLS, LAMBDA_CONTRAST, LAMBDA_ROI,
                            LAMBDA_CONNECTIVITY, LAMBDA_MMD,
                            class_weights=cw,
                            focal_alpha=FOCAL_ALPHA, focal_gamma=FOCAL_GAMMA)
    trainer = Trainer(model, DEVICE, loss_fn, LR, WEIGHT_DECAY, PATIENCE,
                      ckpt_path=ckpt_dir / f"{label}_single.pt")
    trainer.fit(train_loader, test_loader, epochs=epochs, verbose=verbose)

    final_m = trainer.evaluate(test_loader)
    row     = {"fold": 1, "test_subject": subject_id, "experiment": label,
               "task": "HIGH_vs_LOW_ENGAGEMENT"}
    row.update(final_m)
    df = pd.DataFrame([row])
    df.to_csv(save_dir / f"losocv_{label}.csv", index=False)
    if verbose:
        print(f"  Single → Acc={final_m.get('accuracy', 0):.4f}  "
              f"F1={final_m.get('f1', 0):.4f}  "
              f"AUC={final_m.get('roc_auc', 0):.4f}")
    return df


# ── Benchmark Save ────────────────────────────────────────────────────────────

def _save_benchmark(all_rows, label, fold_parallel):
    import json
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    mode    = "FoldParallel" if fold_parallel and NUM_GPUS > 1 else "DataParallel"
    summary = benchmark_summary(all_rows, n_gpus=NUM_GPUS, mode=mode)
    if summary:
        out = BENCHMARK_DIR / f"benchmark_{label}.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Benchmark: {out}")
