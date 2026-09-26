"""
================================================================
NEUMA Phase 8C — RD-GANet Publication-Grade Training Script
================================================================
Runs full Leave-One-Subject-Out Cross-Validation and produces
all outputs required for publication:

  Training
  --------
  • Full LOSOCV with strict subject isolation
  • Multi-GPU via DataParallel (all visible GPUs)
  • AMP + gradient accumulation for OOM prevention
  • Per-fold checkpoints saved under output/checkpoints/<label>/

  Metrics
  -------
  • Per-fold: accuracy, F1, ROC-AUC, kappa, MCC, balanced-acc, ECE
  • Bootstrap 95% CI across folds
  • Per-class precision / recall / F1 / support (JSON)
  • LOSOCV summary table printed and saved

  Plots (output/plots/<label>/)
  ----
  • Normalised confusion matrix  (pooled across all folds)
  • Per-class ROC curves         (pooled across all folds)
  • Calibration (reliability) diagram
  • Per-fold metric bar chart

  Explainability (output/explainability/<label>/)
  ---------------
  • GradCAM electrode saliency map (per fold, requires --explain)
  • ROI attention heatmap           (per fold, if ETAttentionEncoder)
  • Graph connectivity importance   (per fold, if GAT active)

  Benchmarking
  ------------
  • output/benchmark/<label>_benchmark.json
  • Speedup vs single-GPU, fold runtime comparison, memory scaling

Usage
-----
  python training/train_rdganet.py [OPTIONS]

  --config        {full, phase8c, eeg_only, no_roi, no_graph,
                   no_contrastive, no_mmd, no_temporal}   [default: full]
  --epochs        training epochs per fold                [default: 150]
  --batch-size    batch size                              [default: 16]
  --fold-parallel run folds in parallel across GPUs      [flag]
  --subjects      comma-separated subject IDs            [default: all]
  --label         experiment label for file naming       [default: auto]
  --output-dir    root output directory                  [default: output]
  --explain       run GradCAM / attention after training [flag]
  --n-bootstrap   bootstrap resamples for CI             [default: 1000]
  --sanity        run with random labels (sanity check)  [flag]
  --zero-roi      zero ROI vectors (ROI ablation)        [flag]
================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE  = Path(__file__).resolve().parent
_ROOT  = _HERE.parent
sys.path.insert(0, str(_ROOT))

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
    TEMPERATURE, DROPOUT, N_CLASSES,
    LAMBDA_CLS, LAMBDA_CONTRAST, LAMBDA_ROI,
    LAMBDA_CONNECTIVITY, LAMBDA_MMD,
    DEVICE, CKPT_DIR, METRICS_DIR, PLOTS_DIR,
    EXPLAINABILITY_DIR, BENCHMARK_DIR,
    NUM_GPUS, USE_DP, FOLD_PARALLEL, RANDOM_SEED,
    ALL_OUTPUT_DIRS,
)
from data.dataset          import NeumaGraphDataset, build_dataloaders, _normalize_subject_id
from models.ins_hdgs_cmt   import INS_HDGS_CMT as RDGANet, AblationConfig
from training.losses    import MultiTaskLoss
from training.trainer   import Trainer
from evaluation.losocv  import run_losocv
from evaluation.metrics import (
    compute_metrics, per_class_report,
    bootstrap_ci, aggregate_losocv, print_losocv_summary,
    save_metrics_csv, save_per_class_json,
    plot_confusion_matrix, plot_roc_curves,
    plot_calibration, plot_losocv_summary,
)
from utils.gpu import print_gpu_summary, benchmark_summary


# ── Class labels ─────────────────────────────────────────────────────────────

CLASS_NAMES = ["LOW_ENGAGEMENT", "HIGH_ENGAGEMENT"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RD-GANet LOSOCV training — Phase 8C",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="full",
                   choices=["full", "phase8c",
                             "eeg_only", "eeg_et", "eeg_gat",
                             "no_roi", "no_graph",
                             "no_contrastive", "no_mmd", "no_temporal"],
                   help="Model configuration / ablation")
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--batch-size",  type=int,   default=BATCH_SIZE,  dest="batch_size")
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--fold-parallel", action="store_true", default=FOLD_PARALLEL,
                   dest="fold_parallel",
                   help="Run folds in parallel (one GPU per fold group)")
    p.add_argument("--subjects",    type=str,   default=None,
                   help="Comma-separated subject IDs, e.g. S01,S02,S03")
    p.add_argument("--label",       type=str,   default=None,
                   help="Experiment label for output file naming")
    p.add_argument("--output-dir",  type=str,   default=None, dest="output_dir",
                   help="Root output directory (overrides config)")
    p.add_argument("--explain",     action="store_true",
                   help="Run GradCAM / attention analysis after training")
    p.add_argument("--n-bootstrap", type=int,   default=1000, dest="n_bootstrap",
                   help="Bootstrap resamples for confidence intervals")
    p.add_argument("--sanity",      action="store_true",
                   help="Sanity check: train with shuffled labels")
    p.add_argument("--zero-roi",    action="store_true", dest="zero_roi",
                   help="Ablation: zero ROI vectors during training/eval")
    p.add_argument("--seed",        type=int,   default=RANDOM_SEED)
    p.add_argument("--ddp",         action="store_true",
                   help="(Future) Enable DistributedDataParallel via torchrun")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to a fold checkpoint to resume from, e.g. "
                        "output/checkpoints/ins_hdgs_cmt/ins_hdgs_cmt_fold12_e0.pt. "
                        "The fold and ensemble index are inferred from the filename.")
    return p.parse_args()


# ── Ablation config factory ───────────────────────────────────────────────────

_CONFIG_MAP = {
    "full"          : AblationConfig.full,
    "phase8c"       : AblationConfig.phase8c,
    "eeg_only"      : AblationConfig.eeg_only,
    "eeg_et"        : AblationConfig.eeg_et,
    "eeg_gat"       : AblationConfig.eeg_gat,
    "no_roi"        : AblationConfig.no_roi,
    "no_graph"      : AblationConfig.no_graph,
    "no_contrastive": AblationConfig.no_contrastive,
    "no_mmd"        : AblationConfig.no_mmd,
    "no_temporal"   : lambda: AblationConfig(use_temporal=False),
}


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_model(
    n_eeg_ch : int,
    n_classes: int,
    ablation : AblationConfig,
    n_et_ch  : int = ET_INPUT_DIM,
    n_rois   : int = N_ROIS,
) -> RDGANet:
    return RDGANet(
        n_eeg_ch       = n_eeg_ch,
        n_et_ch        = n_et_ch,
        n_rois         = n_rois,
        n_windows      = N_WINDOWS,
        n_classes      = n_classes,
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
        ablation       = ablation,
    )


# ── Post-LOSOCV prediction collection ────────────────────────────────────────

def _collect_fold_predictions(
    subject_ids : List[str],
    ablation    : AblationConfig,
    label       : str,
    ckpt_dir    : Path,
    verbose     : bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reload best checkpoints and collect pooled predictions across all folds.

    Returns
    -------
    y_true  : (N_total,) pooled ground-truth labels
    y_pred  : (N_total,) pooled predicted labels
    y_prob  : (N_total, C) pooled class probabilities
    """
    all_true, all_pred, all_prob = [], [], []
    n_classes = N_CLASSES

    for fold_idx, test_subj in enumerate(subject_ids):
        fold_no  = fold_idx + 1
        # Primary: bare fold checkpoint.  Fallback: first ensemble member (_e0).
        ckpt = ckpt_dir / f"{label}_fold{fold_no:02d}.pt"
        if not ckpt.exists():
            ckpt = ckpt_dir / f"{label}_fold{fold_no:02d}_e0.pt"
        if not ckpt.exists():
            if verbose:
                print(f"  [SKIP] Checkpoint not found for fold {fold_no}")
            continue

        try:
            test_ds = NeumaGraphDataset(
                subject_ids=[test_subj], precompute_graphs=True
            )
        except FileNotFoundError:
            if verbose:
                print(f"  [SKIP] Missing data for {test_subj}")
            continue

        if len(test_ds) == 0:
            continue

        _, test_loader = build_dataloaders(
            test_ds, test_ds, batch_size=min(BATCH_SIZE, len(test_ds))
        )

        model = _make_model(
            test_ds.n_eeg_ch, test_ds.n_classes, ablation,
            test_ds.n_et_ch,
        )
        try:
            ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
                model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
            else:
                model.load_state_dict(ckpt_data, strict=False)
        except Exception as e:
            print(f"[WARNING] Checkpoint skipped ({ckpt.name}): {e}")
            continue

        model = model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                batch  = {k: v.to(DEVICE) if torch.is_tensor(v) else v
                          for k, v in batch.items()}
                out    = model(
                    eeg_windows   = batch["eeg_windows"],
                    adj_matrices  = batch["adj_matrices"],
                    et_seq        = batch["et_seq"],
                    roi_vector    = batch["roi_vector"],
                    weighted_adjs = batch["weighted_adjs"],
                )
                probs  = torch.softmax(out["logits"], dim=1).cpu().numpy()
                preds  = probs.argmax(axis=1)
                labels = batch["label"].cpu().numpy()

                for i in range(len(labels)):
                    # ── Safe probability standardisation ──────────────────
                    prob = np.asarray(probs[i], dtype=np.float32)
                    prob = prob.squeeze()

                    if prob.ndim != 1:
                        print(
                            f"[WARNING] Invalid prob ndim: "
                            f"{prob.ndim} shape={prob.shape}"
                        )
                        continue

                    if len(prob) != n_classes:
                        print(
                            f"[WARNING] Invalid class count: "
                            f"{len(prob)} expected={n_classes}"
                        )
                        continue

                    if len(all_prob) < 5:
                        print(
                            f"[Predictions] "
                            f"shape={prob.shape} "
                            f"dtype={prob.dtype}"
                        )

                    all_true.append(int(labels[i]))
                    all_pred.append(int(preds[i]))
                    all_prob.append(prob)

        del model, test_ds, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_true:
        return np.array([]), np.array([]), np.array([]).reshape(0, N_CLASSES)

    y_prob = np.stack(all_prob, axis=0)

    assert y_prob.ndim == 2, f"Expected 2-D y_prob, got shape {y_prob.shape}"
    assert y_prob.shape[1] == n_classes, (
        f"Expected {n_classes} classes, got {y_prob.shape[1]}"
    )
    assert not np.isnan(y_prob).any(), "NaN values detected in y_prob"
    assert not np.isinf(y_prob).any(), "Inf values detected in y_prob"

    print(f"[Predictions] Final shape={y_prob.shape}")

    return (
        np.array(all_true),
        np.array(all_pred),
        y_prob,
    )


# ── Explainability pass ───────────────────────────────────────────────────────

def _run_explainability(
    subject_ids : List[str],
    ablation    : AblationConfig,
    label       : str,
    ckpt_dir    : Path,
    explain_dir : Path,
    verbose     : bool = True,
) -> None:
    """
    Run GradCAM and attention analysis for each fold's best checkpoint.

    Saves:
      explainability/<label>/gradcam_fold<N>.npy   — electrode saliency (C,)
      explainability/<label>/roi_attn_fold<N>.npy  — ROI attention (N_rois,)
                                                      if ETAttentionEncoder active
      explainability/<label>/graph_attn_fold<N>.npy— GAT attention (C, C)
    """
    try:
        from explainability.gradcam import EEGGradCAM
    except ImportError:
        warnings.warn("GradCAM module not found — skipping explainability pass.")
        return

    explain_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, test_subj in enumerate(subject_ids):
        fold_no = fold_idx + 1
        ckpt    = ckpt_dir / f"{label}_fold{fold_no:02d}.pt"
        if not ckpt.exists():
            ckpt = ckpt_dir / f"{label}_fold{fold_no:02d}_e0.pt"
        if not ckpt.exists():
            continue

        try:
            test_ds = NeumaGraphDataset(
                subject_ids=[test_subj], precompute_graphs=True
            )
        except FileNotFoundError:
            continue

        if len(test_ds) == 0:
            continue

        model = _make_model(test_ds.n_eeg_ch, test_ds.n_classes, ablation)
        try:
            ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
                model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
            else:
                model.load_state_dict(ckpt_data, strict=False)
        except Exception as e:
            print(f"[WARNING] Checkpoint skipped: {e}")
            continue
        model = model.to(DEVICE)

        # Build a single-sample loader used by all explainability passes
        _, test_loader = build_dataloaders(test_ds, test_ds, batch_size=1)

        # GradCAM electrode saliency
        try:
            gradcam = EEGGradCAM(model)
            sample  = next(iter(test_loader))
            result  = gradcam.generate(
                sample,
                target_class=int(sample["label"][0]),
                device=DEVICE,
            )
            cam = result.get("cam")
            if cam is not None:
                np.save(explain_dir / f"gradcam_fold{fold_no:02d}.npy", cam)
                if verbose:
                    print(f"  GradCAM fold {fold_no:02d}: "
                          f"shape={cam.shape}, "
                          f"max_electrode={int(cam.argmax())}")
            else:
                if verbose:
                    print(f"  [WARN] GradCAM fold {fold_no:02d}: cam is None (hooks not fired)")
        except Exception as e:
            if verbose:
                print(f"  [WARN] GradCAM fold {fold_no:02d} failed: {e}")

        # ROI attention (ETAttentionEncoder output)
        if ablation.use_et_attention:
            try:
                model.eval()
                with torch.no_grad():
                    sample = next(iter(test_loader))
                    sample = {k: v.to(DEVICE) if torch.is_tensor(v) else v
                              for k, v in sample.items()}
                    out = model(
                        eeg_windows   = sample["eeg_windows"],
                        adj_matrices  = sample["adj_matrices"],
                        et_seq        = sample["et_seq"],
                        roi_vector    = sample["roi_vector"],
                        weighted_adjs = sample["weighted_adjs"],
                    )
                    if out.get("et_roi_attn") is not None:
                        roi_np = out["et_roi_attn"].cpu().numpy().squeeze()
                        np.save(explain_dir / f"roi_attn_fold{fold_no:02d}.npy",
                                roi_np)
            except Exception as e:
                if verbose:
                    print(f"  [WARN] ROI attention fold {fold_no:02d} failed: {e}")

        # GAT attention
        try:
            model.eval()
            with torch.no_grad():
                sample = next(iter(test_loader))
                sample = {k: v.to(DEVICE) if torch.is_tensor(v) else v
                          for k, v in sample.items()}
                out = model(
                    eeg_windows   = sample["eeg_windows"],
                    adj_matrices  = sample["adj_matrices"],
                    et_seq        = sample["et_seq"],
                    roi_vector    = sample["roi_vector"],
                    weighted_adjs = sample["weighted_adjs"],
                )
                gat_attn = out.get("gat_attn", {})
                if isinstance(gat_attn, dict):
                    for layer_key, attn_mat in gat_attn.items():
                        if attn_mat is not None:
                            np.save(
                                explain_dir / f"graph_{layer_key}_fold{fold_no:02d}.npy",
                                attn_mat.cpu().numpy(),
                            )
        except Exception as e:
            if verbose:
                print(f"  [WARN] GAT attention fold {fold_no:02d} failed: {e}")

        del model, test_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Publication outputs ───────────────────────────────────────────────────────

def _generate_publication_outputs(
    subject_ids   : List[str],
    results_df,
    ablation      : AblationConfig,
    label         : str,
    ckpt_dir      : Path,
    plots_dir     : Path,
    metrics_dir   : Path,
    n_bootstrap   : int,
    verbose       : bool = True,
) -> None:
    """
    Generate all publication-ready figures and tables.

    1. Collect pooled predictions from all fold checkpoints
    2. Plot: confusion matrix, ROC curves, calibration, fold bar chart
    3. Compute and save: per-class report, bootstrap CIs, summary JSON
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  Generating publication outputs …")
        print("=" * 60)

    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect pooled predictions ────────────────────────────────────────────
    if verbose:
        print("  Collecting fold predictions from checkpoints …")

    y_true, y_pred, y_prob = _collect_fold_predictions(
        subject_ids, ablation, label, ckpt_dir, verbose=verbose
    )

    if len(y_true) == 0:
        if verbose:
            print("  [WARN] No predictions collected — skipping plots.")
        return

    # ── Pooled metrics ────────────────────────────────────────────────────────
    pooled_m = compute_metrics(y_true, y_pred, y_prob, n_classes=N_CLASSES)
    if verbose:
        print(f"\n  Pooled metrics ({len(y_true)} samples across all folds):")
        for k, v in pooled_m.items():
            if not np.isnan(v):
                print(f"    {k:<20} {v:.4f}")

    # ── Per-class report ──────────────────────────────────────────────────────
    cls_rows  = per_class_report(y_true, y_pred, class_names=CLASS_NAMES)
    cls_path  = metrics_dir / f"{label}_per_class.json"
    save_per_class_json(cls_rows, cls_path)
    if verbose:
        print(f"\n  Per-class report → {cls_path.name}")
        print(f"  {'Class':<12} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
        for row in cls_rows:
            print(f"  {row['class']:<12} {row['precision']:>6.3f} "
                  f"{row['recall']:>6.3f} {row['f1']:>6.3f} {row['support']:>6d}")

    # ── Bootstrap confidence intervals ───────────────────────────────────────
    if verbose:
        print(f"\n  Computing bootstrap 95% CI (n={n_bootstrap}) …")

    ci = bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=n_bootstrap, seed=42)
    ci_path = metrics_dir / f"{label}_bootstrap_ci.json"
    with ci_path.open("w") as fh:
        json.dump({k: list(v) for k, v in ci.items()}, fh, indent=2)

    if verbose:
        print(f"  CI → {ci_path.name}")
        for k, (lo, hi) in list(ci.items())[:5]:
            print(f"    {k:<20} [{lo:.4f}, {hi:.4f}]")

    # ── LOSOCV fold-level aggregation ─────────────────────────────────────────
    metric_keys = ["accuracy", "f1", "roc_auc", "kappa", "mcc", "balanced_acc"]
    fold_metric_dicts = []

    if results_df is not None and len(results_df) > 0:
        for _, row in results_df.iterrows():
            d = {k: float(row[k]) for k in metric_keys if k in row}
            fold_metric_dicts.append(d)

    if fold_metric_dicts:
        summary = aggregate_losocv(fold_metric_dicts)
        print_losocv_summary(summary)

        summary_path = metrics_dir / f"{label}_losocv_summary.json"
        with summary_path.open("w") as fh:
            json.dump(summary, fh, indent=2)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_path = plots_dir / f"{label}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true, y_pred,
        class_names = CLASS_NAMES,
        title       = f"Confusion Matrix — {label} (pooled LOSOCV)",
        save_path   = cm_path,
    )
    if verbose:
        print(f"\n  Confusion matrix → {cm_path.name}")

    # ── ROC curves ───────────────────────────────────────────────────────────
    roc_path = plots_dir / f"{label}_roc_curves.png"
    plot_roc_curves(
        y_true, y_prob,
        class_names = CLASS_NAMES,
        title       = f"ROC Curves — {label} (pooled LOSOCV)",
        save_path   = roc_path,
    )
    if verbose:
        print(f"  ROC curves        → {roc_path.name}")

    # ── Calibration ──────────────────────────────────────────────────────────
    cal_path = plots_dir / f"{label}_calibration.png"
    plot_calibration(
        y_true, y_prob,
        title     = f"Calibration — {label}  (ECE={pooled_m['ece']:.4f})",
        save_path = cal_path,
    )
    if verbose:
        print(f"  Calibration       → {cal_path.name}")

    # ── Per-fold bar chart ────────────────────────────────────────────────────
    if fold_metric_dicts:
        bar_path = plots_dir / f"{label}_fold_metrics.png"
        plot_losocv_summary(
            fold_metric_dicts,
            metric_keys = ["accuracy", "f1", "roc_auc"],
            title       = f"LOSOCV Fold Metrics — {label}",
            save_path   = bar_path,
        )
        if verbose:
            print(f"  Fold bar chart    → {bar_path.name}")

    # ── Append to master metrics CSV ─────────────────────────────────────────
    csv_path = metrics_dir / "all_experiments.csv"
    save_metrics_csv(
        pooled_m,
        csv_path = csv_path,
        fold_id  = label,
        extra    = {"n_subjects": len(subject_ids), "n_samples": len(y_true)},
    )
    if verbose:
        print(f"  Master metrics    → {csv_path.name}")


# ── DDP stub ─────────────────────────────────────────────────────────────────

def _ddp_notice() -> None:
    print("\n" + "=" * 60)
    print("  DistributedDataParallel (DDP) mode requested.")
    print("  Launch with torchrun, not python:")
    print()
    print("    torchrun --nproc_per_node=8 training/train_rdganet.py --ddp")
    print()
    print("  DDP stub is wired but not yet fully initialised.")
    print("  Falling back to DataParallel for this run.")
    print("=" * 60 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Ensure output directories exist
    for d in ALL_OUTPUT_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── GPU banner ───────────────────────────────────────────────────────────
    gpu_info = print_gpu_summary()

    if args.ddp:
        _ddp_notice()

    # ── Resolve configuration ─────────────────────────────────────────────────
    ablation = _CONFIG_MAP[args.config]()
    label    = args.label or f"rdganet_{args.config}"

    # Custom output directory override
    if args.output_dir:
        root        = Path(args.output_dir)
        plots_dir   = root / "plots"   / label
        metrics_dir = root / "metrics" / label
        ckpt_dir    = root / "checkpoints" / label
        explain_dir = root / "explainability" / label
        bench_dir   = root / "benchmark"
    else:
        plots_dir   = PLOTS_DIR   / label
        metrics_dir = METRICS_DIR / label
        ckpt_dir    = CKPT_DIR    / label
        explain_dir = EXPLAINABILITY_DIR / label
        bench_dir   = BENCHMARK_DIR

    for d in [plots_dir, metrics_dir, ckpt_dir, explain_dir, bench_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Subject list ─────────────────────────────────────────────────────────
    if args.subjects:
        subject_ids = [_normalize_subject_id(s.strip())
                       for s in args.subjects.split(",")]
    else:
        subject_ids = SUBJECT_IDS

    # ── Print run config ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  RD-GANet Training Run: {label}")
    print("=" * 60)
    print(f"  Configuration : {args.config}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Subjects      : {len(subject_ids)}")
    print(f"  Fold-parallel : {args.fold_parallel}")
    print(f"  Seed          : {args.seed}")
    print(f"  AMP           : enabled" if torch.cuda.is_available() else
          f"  AMP           : disabled (CPU)")
    if args.sanity:
        print("  [SANITY CHECK] Random labels active!")
    if args.zero_roi:
        print("  [ABLATION] ROI vectors zeroed")

    ablation_model = RDGANet(ablation=ablation)
    ablation_model.print_architecture()
    del ablation_model

    # ── LOSOCV ───────────────────────────────────────────────────────────────
    print(f"\n  Starting LOSOCV ({len(subject_ids)} folds) …\n")
    t0 = time.perf_counter()

    results_df = run_losocv(
        subject_ids   = subject_ids,
        ablation      = ablation,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        label         = label,
        save_dir      = metrics_dir,
        verbose       = True,
        fold_parallel = args.fold_parallel,
        random_labels = args.sanity,
        zero_roi      = args.zero_roi,
        random_seed   = args.seed,
        resume        = args.resume,
    )

    total_t = time.perf_counter() - t0
    print(f"\n  LOSOCV complete in {total_t/60:.1f} min")

    # ── Benchmark summary ────────────────────────────────────────────────────
    if results_df is not None and len(results_df) > 0:
        fold_rows = results_df.to_dict("records")
        bench     = benchmark_summary(
            fold_rows,
            n_gpus = NUM_GPUS,
            mode   = "fold-parallel" if args.fold_parallel else "data-parallel",
        )
        bench_path = bench_dir / f"{label}_benchmark.json"
        with bench_path.open("w") as fh:
            json.dump(bench, fh, indent=2, default=str)
        print(f"\n  Benchmark → {bench_path}")

    # ── Publication-quality outputs ───────────────────────────────────────────
    _generate_publication_outputs(
        subject_ids = subject_ids,
        results_df  = results_df,
        ablation    = ablation,
        label       = label,
        ckpt_dir    = ckpt_dir,
        plots_dir   = plots_dir,
        metrics_dir = metrics_dir,
        n_bootstrap = args.n_bootstrap,
        verbose     = True,
    )

    # ── Explainability ────────────────────────────────────────────────────────
    if args.explain:
        print("\n  Running explainability analysis …")
        _run_explainability(
            subject_ids = subject_ids,
            ablation    = ablation,
            label       = label,
            ckpt_dir    = ckpt_dir,
            explain_dir = explain_dir,
            verbose     = True,
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Run complete: {label}")
    print(f"  Total time   : {total_t/60:.1f} min")
    print(f"  Plots        : {plots_dir}")
    print(f"  Metrics      : {metrics_dir}")
    print(f"  Checkpoints  : {ckpt_dir}")
    if args.explain:
        print(f"  Explainability: {explain_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
