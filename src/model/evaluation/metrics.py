"""
================================================================
NEUMA Phase 8C — Extended Evaluation Metrics
================================================================
Publication-grade metric suite for the LOSOCV evaluation:

  compute_metrics()         — scalar metric dict (acc, F1, ROC-AUC, …)
  per_class_report()        — per-class precision / recall / F1 / support
  plot_confusion_matrix()   — normalised + raw confusion matrix figure
  plot_roc_curves()         — per-class OvR ROC curves with AUC
  plot_calibration()        — reliability diagram (ECE)
  bootstrap_ci()            — 95% CI via stratified bootstrap
  save_metrics_csv()        — append fold row to metrics.csv
  aggregate_losocv()        — summarise across all folds

All plotting functions write PNG files; they return the Figure for
embedding in notebooks.  Pass save_path=None to skip file output.
================================================================
"""

from __future__ import annotations

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# sklearn — mandatory
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

# matplotlib — optional; skip plotting gracefully if unavailable
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False


# ── Core Scalar Metrics ────────────────────────────────────────────────────────

def compute_metrics(
    y_true    : np.ndarray,
    y_pred    : np.ndarray,
    y_prob    : Optional[np.ndarray] = None,
    n_classes : Optional[int]        = None,
    average   : str                  = "macro",
) -> Dict[str, float]:
    """
    Compute publication-grade scalar metrics.

    Parameters
    ----------
    y_true    : (N,) integer labels
    y_pred    : (N,) predicted labels
    y_prob    : (N, C) class probabilities for ROC-AUC and ECE
    n_classes : number of classes (inferred if None)
    average   : sklearn averaging — "macro" (default) or "weighted"

    Returns
    -------
    dict[str, float]
    """
    n_classes = n_classes or (int(max(y_true.max(), y_pred.max())) + 1)
    avg       = "binary" if n_classes == 2 else average

    m: Dict[str, float] = {
        "accuracy"    : float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision"   : float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall"      : float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1"          : float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "kappa"       : float(cohen_kappa_score(y_true, y_pred)),
        "mcc"         : float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc"     : float("nan"),
        "ece"         : float("nan"),
    }

    if y_prob is not None:
        # ROC-AUC
        try:
            if n_classes == 2:
                m["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                m["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except Exception:
            pass

        # Expected Calibration Error (ECE) — 15 equal-frequency bins
        try:
            m["ece"] = float(_compute_ece(y_true, y_prob, n_bins=15))
        except Exception:
            pass

    return m


def _compute_ece(
    y_true : np.ndarray,
    y_prob : np.ndarray,
    n_bins : int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) for a multi-class model."""
    # Use max-class confidence and correctness
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct     = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0
    n         = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)

    return ece


# ── Per-Class Report ──────────────────────────────────────────────────────────

def per_class_report(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    class_names : Optional[Sequence[str]] = None,
) -> List[Dict]:
    """
    Per-class precision, recall, F1, and support.

    Returns
    -------
    list of dicts, one per class, sorted by class index.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    classes = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    rows = []
    for cls in classes:
        d = report[cls]
        rows.append({
            "class"    : cls,
            "precision": round(float(d["precision"]), 4),
            "recall"   : round(float(d["recall"]),    4),
            "f1"       : round(float(d["f1-score"]),  4),
            "support"  : int(d["support"]),
        })

    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    class_names : Optional[Sequence[str]] = None,
    title       : str                     = "Confusion Matrix",
    save_path   : Optional[str | Path]    = None,
    figsize     : Tuple[int, int]         = (10, 8),
) -> Optional["plt.Figure"]:
    """
    Plot normalised confusion matrix (with raw counts as annotations).

    Returns matplotlib Figure or None if matplotlib is unavailable.
    """
    if not _MATPLOTLIB:
        warnings.warn("matplotlib not available — skipping confusion matrix plot.")
        return None

    cm_raw  = confusion_matrix(y_true, y_pred)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
    n       = cm_raw.shape[0]
    labels  = class_names or [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalised proportion")

    ticks = np.arange(n)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(labels)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm_raw[i, j]})",
                    ha="center", va="center", color=color, fontsize=8)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curves(
    y_true    : np.ndarray,
    y_prob    : np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    title     : str                       = "ROC Curves (One-vs-Rest)",
    save_path : Optional[str | Path]      = None,
    figsize   : Tuple[int, int]           = (8, 6),
) -> Optional["plt.Figure"]:
    """
    Plot per-class OvR ROC curves with AUC in legend.

    Returns matplotlib Figure or None if matplotlib is unavailable.
    """
    if not _MATPLOTLIB:
        warnings.warn("matplotlib not available — skipping ROC plot.")
        return None

    n_classes = y_prob.shape[1]
    labels    = class_names or [f"Class {i}" for i in range(n_classes)]

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    # sklearn label_binarize returns (N, 1) for binary — expand to (N, 2)
    # so that y_bin[:, i] aligns with y_prob[:, i] for i in {0, 1}.
    if n_classes == 2 and y_bin.ndim == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, (label, color) in enumerate(zip(labels, colors)):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc         = roc_auc_score(y_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f"{label} (AUC={auc:.3f})")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_calibration(
    y_true    : np.ndarray,
    y_prob    : np.ndarray,
    n_bins    : int                    = 10,
    title     : str                    = "Calibration (Reliability Diagram)",
    save_path : Optional[str | Path]   = None,
    figsize   : Tuple[int, int]        = (7, 5),
) -> Optional["plt.Figure"]:
    """
    Plot reliability diagram with ECE annotation.

    Uses maximum-confidence prediction for calibration analysis.

    Returns matplotlib Figure or None if matplotlib is unavailable.
    """
    if not _MATPLOTLIB:
        warnings.warn("matplotlib not available — skipping calibration plot.")
        return None

    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct     = (predictions == y_true).astype(float)

    try:
        frac_pos, mean_conf = calibration_curve(
            correct, confidences, n_bins=n_bins, strategy="uniform"
        )
    except Exception:
        return None

    ece = _compute_ece(y_true, y_prob, n_bins=15)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(mean_conf, frac_pos, "s-", color="#2c7bb6", lw=2,
            label=f"Model (ECE={ece:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.fill_between(mean_conf, frac_pos, mean_conf,
                    alpha=0.1, color="#2c7bb6", label="Calibration gap")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives (accuracy)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_losocv_summary(
    fold_metrics : List[Dict[str, float]],
    metric_keys  : Sequence[str]         = ("accuracy", "f1", "roc_auc"),
    title        : str                   = "LOSOCV Fold Metrics",
    save_path    : Optional[str | Path]  = None,
    figsize      : Tuple[int, int]       = (10, 5),
) -> Optional["plt.Figure"]:
    """
    Bar chart of per-fold metrics with mean ± std annotation.

    Returns matplotlib Figure or None if matplotlib is unavailable.
    """
    if not _MATPLOTLIB:
        return None

    n_folds  = len(fold_metrics)
    n_metrics= len(metric_keys)
    x        = np.arange(n_folds)
    width    = 0.8 / n_metrics
    colors   = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    fig, ax = plt.subplots(figsize=figsize)
    for k_idx, (key, color) in enumerate(zip(metric_keys, colors)):
        vals = [m.get(key, float("nan")) for m in fold_metrics]
        offset = (k_idx - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=key, color=color, alpha=0.8)

        mean_v = np.nanmean(vals)
        std_v  = np.nanstd(vals)
        ax.axhline(mean_v, color=color, ls="--", lw=1.2, alpha=0.6)
        ax.text(n_folds - 0.5, mean_v + 0.01,
                f"{key}: {mean_v:.3f}±{std_v:.3f}",
                color=color, fontsize=7, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1.1])
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Bootstrap Confidence Intervals ───────────────────────────────────────────

def bootstrap_ci(
    y_true     : np.ndarray,
    y_pred     : np.ndarray,
    y_prob     : Optional[np.ndarray] = None,
    n_bootstrap: int                  = 1000,
    ci_level   : float                = 0.95,
    seed       : int                  = 42,
    average    : str                  = "macro",
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for all scalar metrics.

    Uses stratified resampling (maintains class distribution).

    Parameters
    ----------
    n_bootstrap : number of bootstrap resamples
    ci_level    : confidence level (default 0.95 → 95% CI)

    Returns
    -------
    dict[metric_name, (lower_bound, upper_bound)]
    """
    rng      = np.random.default_rng(seed)
    n        = len(y_true)
    alpha    = (1 - ci_level) / 2
    boot_scores: Dict[str, List[float]] = defaultdict(list)

    for _ in range(n_bootstrap):
        idx   = rng.choice(n, size=n, replace=True)
        bt    = y_true[idx]
        bp    = y_pred[idx]
        bprob = y_prob[idx] if y_prob is not None else None

        # Skip if bootstrap sample has only one class (undefined metrics)
        if len(np.unique(bt)) < 2:
            continue

        m = compute_metrics(bt, bp, bprob, average=average)
        for k, v in m.items():
            if not np.isnan(v):
                boot_scores[k].append(v)

    ci: Dict[str, Tuple[float, float]] = {}
    for k, scores in boot_scores.items():
        arr = np.array(scores)
        ci[k] = (float(np.percentile(arr, 100 * alpha)),
                 float(np.percentile(arr, 100 * (1 - alpha))))

    return ci


# ── CSV / JSON Persistence ────────────────────────────────────────────────────

def save_metrics_csv(
    metrics  : Dict[str, float],
    csv_path : str | Path,
    fold_id  : Optional[str] = None,
    extra    : Optional[Dict] = None,
) -> None:
    """
    Append a fold metrics row to a CSV file.

    Creates the file with a header row on first call; appends on subsequent calls.
    """
    csv_path = Path(csv_path)
    row = {}
    if fold_id is not None:
        row["fold_id"] = fold_id
    if extra:
        row.update(extra)
    row.update({k: round(float(v), 6) for k, v in metrics.items()})

    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_per_class_json(
    rows     : List[Dict],
    json_path: str | Path,
) -> None:
    """Save per-class report list as a JSON file."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as fh:
        json.dump(rows, fh, indent=2)


# ── LOSOCV Aggregation ────────────────────────────────────────────────────────

def aggregate_losocv(
    fold_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-fold metric dicts into mean ± std summary.

    Parameters
    ----------
    fold_metrics : list of metric dicts, one per LOSOCV fold

    Returns
    -------
    dict[metric_name, {"mean": float, "std": float, "min": float, "max": float}]
    """
    all_keys = set()
    for m in fold_metrics:
        all_keys |= set(m.keys())

    summary: Dict[str, Dict[str, float]] = {}
    for k in sorted(all_keys):
        vals = [m[k] for m in fold_metrics if k in m and not np.isnan(m[k])]
        if not vals:
            continue
        arr = np.array(vals)
        summary[k] = {
            "mean": float(arr.mean()),
            "std" : float(arr.std()),
            "min" : float(arr.min()),
            "max" : float(arr.max()),
            "n"   : len(vals),
        }

    return summary


def print_losocv_summary(summary: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print the aggregated LOSOCV summary table."""
    print("\n" + "=" * 60)
    print("  LOSOCV Summary")
    print("=" * 60)
    print(f"  {'Metric':<20}  {'Mean':>8}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    print("-" * 60)
    for k, v in summary.items():
        print(f"  {k:<20}  {v['mean']:8.4f}  {v['std']:7.4f}  "
              f"{v['min']:7.4f}  {v['max']:7.4f}")
    print("=" * 60)


# ── MetricTracker ─────────────────────────────────────────────────────────────

class MetricTracker:
    """
    Accumulates per-batch predictions and computes epoch metrics.

    Compatible with the Trainer's update call signature.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._preds   : List = []
        self._probs   : List = []
        self._targets : List = []
        self._losses  : Dict[str, List[float]] = defaultdict(list)

    def update(
        self,
        preds  : np.ndarray,
        probs  : np.ndarray,
        targets: np.ndarray,
        losses : Optional[Dict] = None,
    ) -> None:
        self._preds.extend(np.asarray(preds).tolist())
        self._probs.extend(np.asarray(probs).tolist())
        self._targets.extend(np.asarray(targets).tolist())
        if losses:
            for k, v in losses.items():
                val = v.item() if hasattr(v, "item") else float(v)
                self._losses[k].append(val)

    def compute(self) -> Dict[str, float]:
        y_true = np.array(self._targets)
        y_pred = np.array(self._preds)
        y_prob = np.array(self._probs)

        m = compute_metrics(y_true, y_pred, y_prob)
        for k, vals in self._losses.items():
            m[f"loss_{k}"] = float(np.mean(vals))

        return m

    def compute_with_ci(
        self,
        n_bootstrap: int = 500,
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        """Return (metrics, confidence_intervals) together."""
        y_true = np.array(self._targets)
        y_pred = np.array(self._preds)
        y_prob = np.array(self._probs)

        metrics = compute_metrics(y_true, y_pred, y_prob)
        ci      = bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=n_bootstrap)
        return metrics, ci
