"""
Offline threshold analysis for completed LOSOCV runs.

This script reads the per-fold CSV produced by evaluation/losocv.py and
recomputes predictions from saved y_prob values under several decision rules.
It does not retrain models and does not modify the original result file.

Example:
    python analysis/analyze_adaptive_thresholds.py \
      --label repro_focal_g3p0_effective_num_37
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from training.metrics import compute_metrics


KEY_METRICS = [
    "accuracy",
    "f1",
    "balanced_acc",
    "roc_auc",
    "pr_auc",
    "kappa",
    "mcc",
    "ece",
    "precision",
    "recall",
]


def _parse_list(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value)
    if pd.isna(value):
        return np.asarray([])
    return np.asarray(ast.literal_eval(str(value)))


def _prob_2d(p_high: np.ndarray) -> np.ndarray:
    return np.stack([1.0 - p_high, p_high], axis=1)


def _metrics_for_fold(y_true: np.ndarray, p_high: np.ndarray, threshold: float) -> dict:
    y_pred = (p_high >= threshold).astype(int)
    out = compute_metrics(y_true=y_true, y_pred=y_pred, y_prob=_prob_2d(p_high), n_classes=2)
    out["threshold"] = float(threshold)
    out["pred_high_rate"] = float(y_pred.mean()) if len(y_pred) else float("nan")
    out["true_high_rate"] = float(y_true.mean()) if len(y_true) else float("nan")
    out["collapsed"] = int(len(np.unique(y_pred)) < 2) if len(y_pred) else 1
    return out


def _summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in per_fold.groupby("method", sort=False):
        row = {
            "method": method,
            "n_folds": int(len(g)),
            "collapsed_folds": int(g["collapsed"].sum()),
            "mean_threshold": float(g["threshold"].mean()),
        }
        for metric in KEY_METRICS:
            if metric in g:
                row[f"{metric}_mean"] = float(g[metric].mean())
                row[f"{metric}_std"] = float(g[metric].std())
                row[f"{metric}_min"] = float(g[metric].min())
                row[f"{metric}_max"] = float(g[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(
    input_csv: Path,
    out_dir: Path,
    high_rate: float,
    shrink_alpha: float,
    clip_low: float,
    clip_high: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv)
    rows = []

    for _, fold in df.iterrows():
        y_true = _parse_list(fold["y_true"]).astype(int)
        p_high = _parse_list(fold["y_prob"]).astype(float)
        if len(y_true) == 0 or len(p_high) == 0:
            continue

        fold_thr = float(fold.get("opt_threshold", 0.5))
        methods = {
            "fixed_0p50": 0.5,
            "fold_val_threshold": fold_thr,
            "fold_threshold_shrink": shrink_alpha * fold_thr + (1.0 - shrink_alpha) * 0.5,
            "fold_threshold_clipped": float(np.clip(fold_thr, clip_low, clip_high)),
            "subject_median_unsup": float(np.median(p_high)),
            "subject_prior_quantile_unsup": float(np.quantile(p_high, 1.0 - high_rate)),
        }

        for method, threshold in methods.items():
            metrics = _metrics_for_fold(y_true, p_high, threshold)
            metrics.update(
                {
                    "method": method,
                    "fold": int(fold["fold"]),
                    "test_subject": fold.get("test_subject", ""),
                    "n": int(len(y_true)),
                }
            )
            rows.append(metrics)

    per_fold = pd.DataFrame(rows)
    summary = _summarize(per_fold)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out_dir / "adaptive_thresholds_per_fold.csv", index=False)
    summary.to_csv(out_dir / "adaptive_thresholds_summary.csv", index=False)
    return per_fold, summary


def _default_csv_for_label(label: str) -> Path:
    return Path("output") / "metrics" / label / f"losocv_{label}.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=None, help="Path to losocv_<label>.csv")
    p.add_argument("--label", type=str, default=None, help="Run label under output/metrics")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    p.add_argument("--high-rate", type=float, default=0.5, help="Unsupervised subject HIGH prior for quantile rule")
    p.add_argument("--shrink-alpha", type=float, default=0.7, help="Weight on fold validation threshold")
    p.add_argument("--clip-low", type=float, default=0.35)
    p.add_argument("--clip-high", type=float, default=0.65)
    args = p.parse_args()

    if args.csv is None:
        if not args.label:
            raise SystemExit("Provide --csv or --label")
        args.csv = _default_csv_for_label(args.label)
    if args.out_dir is None:
        stem = args.label or args.csv.stem.replace("losocv_", "")
        args.out_dir = Path("output") / "threshold_analysis" / stem

    _, summary = analyze(
        input_csv=args.csv,
        out_dir=args.out_dir,
        high_rate=args.high_rate,
        shrink_alpha=args.shrink_alpha,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
    )

    show_cols = [
        "method",
        "n_folds",
        "collapsed_folds",
        "accuracy_mean",
        "f1_mean",
        "balanced_acc_mean",
        "roc_auc_mean",
        "kappa_mean",
        "mcc_mean",
        "mean_threshold",
    ]
    print(summary[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {args.out_dir}")


if __name__ == "__main__":
    main()
