"""
================================================================
INS-HDGS-CMT — Evaluation Metrics
================================================================
Comprehensive metric suite for engagement classification:

  Primary:
    Accuracy, F1 (binary), Balanced Accuracy
    ROC-AUC, PR-AUC
    Cohen's Kappa, MCC

  Calibration:
    ECE (Expected Calibration Error)
    Temperature scaling calibration

  Publication-grade reporting:
    Mean ± std across LOSOCV folds
    95% confidence intervals (bootstrap)
================================================================
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)


# ── Expected Calibration Error ────────────────────────────────────────────────

def compute_ece(
    y_true  : np.ndarray,
    y_prob  : np.ndarray,
    n_bins  : int = 10,
) -> float:
    """
    Expected Calibration Error (Naeini et al., 2015).

    Measures alignment between predicted confidence and empirical accuracy.
    Perfect calibration → ECE = 0.

    Parameters
    ----------
    y_true : (N,) ground-truth binary labels
    y_prob : (N,) predicted probability for class 1
    n_bins : number of equal-width confidence bins
    """
    bins    = np.linspace(0, 1, n_bins + 1)
    ece     = 0.0
    n       = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        conf_bin = y_prob[mask].mean()
        acc_bin  = (y_true[mask] == (y_prob[mask] >= 0.5).astype(int)).mean()
        ece     += (mask.sum() / n) * abs(conf_bin - acc_bin)

    return float(ece)


# ── Core Metric Computation ───────────────────────────────────────────────────

def compute_metrics(
    y_true    : np.ndarray,
    y_pred    : np.ndarray,
    y_prob    : Optional[np.ndarray] = None,
    n_classes : Optional[int]        = None,
    average   : str                  = "binary",
) -> dict:
    """
    Compute full metric suite for engagement decoding.

    Parameters
    ----------
    y_true    : (N,) ground-truth labels
    y_pred    : (N,) predicted labels
    y_prob    : (N, C) or (N,) class probabilities
    n_classes : number of classes (inferred if None)
    average   : sklearn averaging strategy

    Returns
    -------
    dict of metric_name → float
    """
    n_cls = n_classes or int(max(y_true.max(), y_pred.max())) + 1
    avg   = "binary" if n_cls == 2 else "macro"

    # Guard: single-class y_true makes AUC undefined; return neutral baselines.
    n_unique_true = len(np.unique(y_true))
    roc_auc_default = 0.5 if n_unique_true < 2 else float("nan")
    pr_auc_default  = float(np.mean(y_true)) if n_unique_true < 2 else float("nan")

    # Suppress "y_pred contains classes not in y_true" from precision_score,
    # recall_score, f1_score, cohen_kappa_score — all emit this UserWarning when
    # the model predicts a class absent from the (small) test fold.
    # cohen_kappa_score also raises ValueError when both arrays are constant.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kappa = float(cohen_kappa_score(y_true, y_pred))
        except ValueError:
            kappa = 0.0

        metrics = {
            "accuracy"      : float(accuracy_score(y_true, y_pred)),
            "precision"     : float(precision_score(y_true, y_pred,
                                    average=avg, zero_division=0)),
            "recall"        : float(recall_score(y_true, y_pred,
                                    average=avg, zero_division=0)),
            "f1"            : float(f1_score(y_true, y_pred,
                                    average=avg, zero_division=0)),
            "kappa"         : kappa,
            "mcc"           : float(matthews_corrcoef(y_true, y_pred)),
            "balanced_acc"  : float(balanced_accuracy_score(y_true, y_pred)),
            "roc_auc"       : roc_auc_default,
            "pr_auc"        : pr_auc_default,
            "ece"           : float("nan"),
        }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        p1 = y_prob[:, 1] if (y_prob.ndim == 2 and n_cls == 2) else y_prob
        if n_cls == 2:
            metrics["ece"] = compute_ece(y_true, p1)
        if n_unique_true >= 2:
            try:
                if n_cls == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, p1))
                    metrics["pr_auc"]  = float(average_precision_score(y_true, p1))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="macro"
                    ))
            except Exception:
                pass

    return metrics


# ── Metric Tracker ────────────────────────────────────────────────────────────

class MetricTracker:
    """Accumulates per-batch predictions and computes epoch metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._preds   = []
        self._probs   = []
        self._targets = []
        self._losses  = defaultdict(list)

    def update(
        self,
        preds   : "torch.Tensor",
        probs   : "torch.Tensor",
        targets : "torch.Tensor",
        losses  : Optional[dict] = None,
    ):
        self._preds.extend(preds.tolist())
        self._probs.extend(probs.tolist())
        self._targets.extend(targets.tolist())
        if losses:
            for k, v in losses.items():
                if torch.is_tensor(v):
                    if v.ndim > 0:
                        v = v.mean()
                    v = v.item()
                else:
                    v = float(v)
                self._losses[k].append(v)

    def compute(self) -> dict:
        y_true = np.array(self._targets)
        y_pred = np.array(self._preds)
        y_prob = np.array(self._probs)

        m = compute_metrics(y_true, y_pred, y_prob)
        for k, vals in self._losses.items():
            m[f"loss_{k}"] = float(np.mean(vals))
        return m


# ── LOSOCV Summary Statistics ─────────────────────────────────────────────────

def losocv_summary(results_df) -> dict:
    """
    Compute mean ± std across LOSOCV folds for each metric.

    Parameters
    ----------
    results_df : DataFrame with one row per fold

    Returns
    -------
    dict of metric_name → {"mean": float, "std": float}
    """
    key_metrics = [
        "accuracy", "f1", "balanced_acc", "roc_auc", "pr_auc",
        "kappa", "mcc", "ece", "precision", "recall",
    ]
    summary = {}
    for m in key_metrics:
        if m in results_df.columns:
            v = results_df[m].dropna()
            summary[m] = {
                "mean": round(float(v.mean()), 4),
                "std" : round(float(v.std()),  4),
                "min" : round(float(v.min()),  4),
                "max" : round(float(v.max()),  4),
            }
    return summary


def pooled_metrics_from_folds(results_df) -> dict:
    """
    Pool every fold's (y_true, y_prob) into one array and compute a single
    set of metrics, instead of averaging per-fold metrics.

    With ~9 epochs per held-out subject (385 epochs / 42 subjects), a
    per-subject ROC-AUC is coarse and can hit 1.00 by chance (see
    ``perfect_rank_chance_prob``). Pooling across folds before computing
    AUC gives a far more statistically stable summary and should be
    reported alongside — not instead of — the per-subject mean.

    Requires results_df to have "y_true" / "y_prob" columns holding the
    per-fold lists (as produced in-memory by run_losocv; if reloaded from
    CSV these will be strings and must be parsed with ast.literal_eval
    first).

    Returns
    -------
    dict of pooled metric_name -> float (empty dict if columns missing)
    """
    if "y_true" not in results_df.columns or "y_prob" not in results_df.columns:
        return {}

    y_true_all: list = []
    y_prob_all: list = []
    for _, row in results_df.iterrows():
        yt, yp = row.get("y_true"), row.get("y_prob")
        if yt is None or yp is None:
            continue
        y_true_all.extend(yt)
        y_prob_all.extend(yp)

    if len(y_true_all) == 0 or len(np.unique(y_true_all)) < 2:
        return {}

    y_true_all = np.array(y_true_all, dtype=int)
    y_prob_all = np.array(y_prob_all, dtype=float)
    y_pred_all = (y_prob_all >= 0.5).astype(int)
    y_prob_2d  = np.stack([1 - y_prob_all, y_prob_all], axis=1)

    m = compute_metrics(y_true_all, y_pred_all, y_prob_2d, n_classes=2)
    m["n_pooled"] = len(y_true_all)
    return m


def perfect_rank_chance_prob(n_pos: int, n_neg: int) -> float:
    """
    Probability that an *uninformative* (random) ranking of n_pos positive
    and n_neg negative samples achieves ROC-AUC = 1.0 purely by chance —
    i.e. every positive happens to rank above every negative.

    Equal to 1 / C(n_pos + n_neg, n_pos). With small per-subject LOSOCV
    folds (e.g. n=9), this is often non-negligible (e.g. 1/C(9,4) ≈ 0.008
    is small, but the more relevant comparison is against how many test
    subjects report AUC=1.00 — even a small per-fold chance rate compounds
    across 37 folds).
    """
    from math import comb
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    return 1.0 / comb(n_pos + n_neg, n_pos)


def print_losocv_summary(results_df, label: str = "INS-HDGS-CMT"):
    """Print formatted LOSOCV summary to stdout, with calibrated comparison."""
    summary = losocv_summary(results_df)
    width   = 58

    print(f"\n{'='*width}")
    print(f"  LOSOCV Summary — {label}")
    print(f"  Task: HIGH_ENGAGEMENT vs LOW_ENGAGEMENT")
    print(f"{'='*width}")
    fmt = "  {:<18}: {:6.4f} ± {:6.4f}  [min={:.4f}  max={:.4f}]"
    for m, v in summary.items():
        print(fmt.format(m, v["mean"], v["std"], v["min"], v["max"]))
    print(f"{'='*width}")

    # ── Pooled AUC (co-primary metric, more stable than per-subject mean) ───
    pooled = pooled_metrics_from_folds(results_df)
    if pooled:
        mean_auc = summary.get("roc_auc", {}).get("mean")
        print(f"\n  Pooled ROC-AUC (all {pooled['n_pooled']} test epochs "
              f"concatenated) : {pooled['roc_auc']:.4f}")
        if mean_auc is not None:
            print(f"  Mean per-subject ROC-AUC (headline above)         "
                  f": {mean_auc:.4f}")
            print(f"  -> report both; per-subject AUC is unstable at "
                  f"~{results_df['test_n'].mean():.0f} epochs/fold and can "
                  f"hit 1.00 by chance (see below).")

    # ── Small-sample AUC=1.00 diagnostic ─────────────────────────────────────
    if "roc_auc" in results_df.columns and "y_true" in results_df.columns:
        perfect = results_df[results_df["roc_auc"] >= 0.999]
        if len(perfect) > 0:
            print(f"\n  Folds with ROC-AUC ~= 1.00 ({len(perfect)} of "
                  f"{len(results_df)}) — chance-perfect-ranking probability:")
            for _, row in perfect.iterrows():
                yt = np.array(row.get("y_true") or [])
                if yt.size == 0:
                    continue
                n_pos = int((yt == 1).sum())
                n_neg = int((yt == 0).sum())
                p_chance = perfect_rank_chance_prob(n_pos, n_neg)
                subj = row.get("test_subject", "?")
                print(f"    {subj}: n={n_pos + n_neg} (pos={n_pos}, neg={n_neg})"
                      f"  P(perfect rank | random) = {p_chance:.4f}")
    print(f"{'='*width}")

    # Calibrated comparison table (only if _cal columns are present)
    cal_key_metrics = ["accuracy", "f1", "balanced_acc", "roc_auc", "pr_auc", "mcc", "ece"]
    cal_cols = [f"{m}_cal" for m in cal_key_metrics]
    if any(c in results_df.columns for c in cal_cols):
        print(f"\n{'='*width}")
        print(f"  Post-hoc Temperature Scaling — {label}")
        print(f"  Comparison: Original → Calibrated  (T_post per fold)")
        print(f"{'='*width}")
        fmt2 = "  {:<14}: {:6.4f} ± {:5.4f}  →  {:6.4f} ± {:5.4f}  Δ={:+.4f}"
        if "T_post" in results_df.columns:
            tp = results_df["T_post"].dropna()
            print(f"  {'T_post':<14}: {tp.mean():.4f} ± {tp.std():.4f}"
                  f"  [min={tp.min():.3f}  max={tp.max():.3f}]")
        for m in cal_key_metrics:
            if m in results_df.columns and f"{m}_cal" in results_df.columns:
                orig = results_df[m].dropna()
                cal  = results_df[f"{m}_cal"].dropna()
                print(fmt2.format(
                    m,
                    orig.mean(), orig.std(),
                    cal.mean(),  cal.std(),
                    cal.mean() - orig.mean(),
                ))
        print(f"{'='*width}")
