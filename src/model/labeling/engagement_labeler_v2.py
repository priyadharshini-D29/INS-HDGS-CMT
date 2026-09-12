"""
================================================================
INS-HDGS-CMT — Engagement Labeler v2
================================================================
Improved engagement labeling with per-subject adaptive thresholding.

Root cause of single-class LOSOCV folds
-----------------------------------------
The original EngagementLabeler used a global median threshold over all
epochs.  Subjects whose scores are uniformly above or below that threshold
(e.g. S33 → all-HIGH, S31/S16 → all-LOW) produce degenerate folds where
MCC/AUC are undefined.

Fix
---
ThresholdMode.SUBJECT_MEDIAN (default): each subject's threshold is its own
score median.  This is LOSOCV-safe because only the test subject's own data
is used to compute its threshold — no information from other subjects leaks
in.  The result is approximately balanced labels for every subject.

Usage
-----
  from labeling.engagement_labeler_v2 import EngagementLabelerV2, ThresholdMode

  labeler = EngagementLabelerV2(mode=ThresholdMode.SUBJECT_MEDIAN)
  labels_dict, diag_dict = labeler.fit_transform_all(subject_scores)

  # single subject
  labels, diag = labeler.label_subject("S31", scores, global_scores=all_scores)
================================================================
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Threshold Modes ───────────────────────────────────────────────────────────

class ThresholdMode(str, Enum):
    GLOBAL_MEDIAN  = "global_median"   # original: single median over all subjects
    SUBJECT_MEDIAN = "subject_median"  # per-subject median (recommended, LOSOCV-safe)
    PERCENTILE     = "percentile"      # per-subject Nth percentile
    ZSCORE         = "zscore"          # mean + k*std of per-subject scores
    ADAPTIVE       = "adaptive"        # subject_median if n≥min_epochs else global_median


# ── Pathological Status Labels ────────────────────────────────────────────────

class LabelStatus(str, Enum):
    OK                = "OK"
    LOW_SAMPLE_COUNT  = "LOW_SAMPLE_COUNT"
    SINGLE_CLASS_HIGH = "SINGLE_CLASS_HIGH"
    SINGLE_CLASS_LOW  = "SINGLE_CLASS_LOW"
    EXTREME_IMBALANCE = "EXTREME_IMBALANCE"


# ── Per-Subject Diagnostics ───────────────────────────────────────────────────

@dataclass
class SubjectLabelDiagnostics:
    subject_id:      str
    n_epochs:        int
    n_high:          int
    n_low:           int
    class_ratio:     float    # n_high / n_epochs
    balance_score:   float    # min(n_high, n_low) / n_epochs  (0=worst, 0.5=best)
    entropy:         float    # binary entropy H(p)
    imbalance_score: float    # |class_ratio - 0.5| * 2  (0=balanced, 1=single-class)
    threshold_used:  float
    threshold_mode:  str
    score_mean:      float
    score_std:       float
    score_min:       float
    score_max:       float
    status:          str      # LabelStatus value
    recommendation:  str
    warning_flags:   List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["warning_flags"] = "; ".join(d["warning_flags"])
        return d


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


# ── Labeler ───────────────────────────────────────────────────────────────────

class EngagementLabelerV2:
    """
    Per-subject adaptive engagement labeling engine.

    Parameters
    ----------
    mode : ThresholdMode
        How to compute the decision threshold for each subject.
    percentile : float
        Percentile used when mode=PERCENTILE (default 50 → same as median).
    zscore_thresh : float
        Number of SDs above mean used when mode=ZSCORE (default 0 → mean).
    min_epochs : int
        Minimum epochs required for per-subject thresholding; subjects below
        this count fall back to the global threshold in ADAPTIVE mode.
    soft_labels : bool
        If True, also compute soft (sigmoid) engagement probabilities.
    soft_sigma : float | None
        Width parameter for soft labels.  Defaults to per-subject score std.
    label_smoothing : float
        Alpha ∈ [0, 1] for label smoothing: final = (1-α)*hard + α*0.5.
    imbalance_warn : float
        imbalance_score threshold above which EXTREME_IMBALANCE is flagged.
    """

    def __init__(
        self,
        mode:            ThresholdMode | str = ThresholdMode.SUBJECT_MEDIAN,
        percentile:      float = 50.0,
        zscore_thresh:   float = 0.0,
        min_epochs:      int   = 4,
        soft_labels:     bool  = False,
        soft_sigma:      Optional[float] = None,
        label_smoothing: float = 0.0,
        imbalance_warn:  float = 0.6,
    ):
        self.mode            = ThresholdMode(mode)
        self.percentile      = percentile
        self.zscore_thresh   = zscore_thresh
        self.min_epochs      = min_epochs
        self.soft_labels     = soft_labels
        self.soft_sigma      = soft_sigma
        self.label_smoothing = label_smoothing
        self.imbalance_warn  = imbalance_warn

    # ── Threshold computation ─────────────────────────────────────────────────

    def compute_threshold(
        self,
        scores:        np.ndarray,
        global_scores: Optional[np.ndarray] = None,
    ) -> float:
        scores = np.asarray(scores, dtype=np.float64)
        n = len(scores)

        if self.mode == ThresholdMode.GLOBAL_MEDIAN:
            ref = global_scores if global_scores is not None else scores
            return float(np.median(ref))

        elif self.mode == ThresholdMode.SUBJECT_MEDIAN:
            if n < self.min_epochs and global_scores is not None:
                return float(np.median(global_scores))
            return float(np.median(scores))

        elif self.mode == ThresholdMode.PERCENTILE:
            if n < self.min_epochs and global_scores is not None:
                return float(np.percentile(global_scores, self.percentile))
            return float(np.percentile(scores, self.percentile))

        elif self.mode == ThresholdMode.ZSCORE:
            mu  = float(np.mean(scores))
            sig = float(np.std(scores)) or 1.0
            return mu + self.zscore_thresh * sig

        elif self.mode == ThresholdMode.ADAPTIVE:
            if n >= self.min_epochs:
                return float(np.median(scores))
            if global_scores is not None:
                return float(np.median(global_scores))
            return float(np.median(scores))

        return float(np.median(scores))

    # ── Soft-label sigmoid ────────────────────────────────────────────────────

    def compute_soft_labels(
        self,
        scores:    np.ndarray,
        threshold: float,
        sigma:     Optional[float] = None,
    ) -> np.ndarray:
        """
        Sigmoid soft engagement probability centred at `threshold`.

        soft[i] = sigmoid((score[i] - threshold) / sigma)

        With label_smoothing α:
          final[i] = (1-α)*soft[i] + α*0.5
        """
        scores = np.asarray(scores, dtype=np.float32)
        if sigma is None:
            sigma = float(scores.std()) if scores.std() > 1e-8 else 1.0
        sigma = max(sigma, 1e-6)
        z    = (scores - threshold) / sigma
        soft = 1.0 / (1.0 + np.exp(-z))
        if self.label_smoothing > 0:
            alpha = float(self.label_smoothing)
            soft  = (1 - alpha) * soft + alpha * 0.5
        return soft.astype(np.float32)

    # ── Single-subject labeling ───────────────────────────────────────────────

    def label_subject(
        self,
        subject_id:    str,
        scores:        np.ndarray,
        global_scores: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, SubjectLabelDiagnostics]:
        """
        Compute engagement labels and diagnostics for one subject.

        Returns
        -------
        labels : (N,) int64 array  — 0=LOW, 1=HIGH
        diag   : SubjectLabelDiagnostics
        """
        scores = np.asarray(scores, dtype=np.float32)
        n      = len(scores)

        threshold = self.compute_threshold(scores, global_scores)
        labels    = (scores >= threshold).astype(np.int64)

        n_high = int(labels.sum())
        n_low  = n - n_high

        class_ratio    = n_high / max(n, 1)
        balance_score  = min(n_high, n_low) / max(n, 1)
        entropy        = _binary_entropy(class_ratio)
        imbalance_score = abs(class_ratio - 0.5) * 2.0

        # ── Determine status ──────────────────────────────────────────────
        warning_flags: List[str] = []

        if n < self.min_epochs:
            status = LabelStatus.LOW_SAMPLE_COUNT.value
            warning_flags.append(
                f"Only {n} epochs — fewer than min_epochs={self.min_epochs}"
            )
        elif n_high == 0:
            status = LabelStatus.SINGLE_CLASS_LOW.value
            warning_flags.append(
                "All epochs labeled LOW — global threshold is above all scores"
            )
        elif n_low == 0:
            status = LabelStatus.SINGLE_CLASS_HIGH.value
            warning_flags.append(
                "All epochs labeled HIGH — global threshold is below all scores"
            )
        elif imbalance_score > self.imbalance_warn:
            status = LabelStatus.EXTREME_IMBALANCE.value
            warning_flags.append(
                f"imbalance_score={imbalance_score:.2f} (class_ratio={class_ratio:.2f})"
            )
        else:
            status = LabelStatus.OK.value

        # ── Recommendation ────────────────────────────────────────────────
        if status == LabelStatus.SINGLE_CLASS_LOW.value:
            recommendation = (
                "Subject scores are globally low; use SUBJECT_MEDIAN to obtain "
                "balanced labels from its own distribution"
            )
        elif status == LabelStatus.SINGLE_CLASS_HIGH.value:
            recommendation = (
                "Subject scores are globally high; use SUBJECT_MEDIAN to obtain "
                "balanced labels from its own distribution"
            )
        elif status == LabelStatus.EXTREME_IMBALANCE.value:
            recommendation = (
                "Consider SUBJECT_MEDIAN or PERCENTILE-50 thresholding to improve balance"
            )
        elif status == LabelStatus.LOW_SAMPLE_COUNT.value:
            recommendation = (
                f"Collect more epochs; {n} is insufficient for reliable LOSOCV decoding"
            )
        else:
            recommendation = "Labels look healthy"

        diag = SubjectLabelDiagnostics(
            subject_id      = subject_id,
            n_epochs        = n,
            n_high          = n_high,
            n_low           = n_low,
            class_ratio     = round(float(class_ratio), 4),
            balance_score   = round(float(balance_score), 4),
            entropy         = round(float(entropy), 4),
            imbalance_score = round(float(imbalance_score), 4),
            threshold_used  = round(float(threshold), 6),
            threshold_mode  = self.mode.value,
            score_mean      = round(float(scores.mean()), 6),
            score_std       = round(float(scores.std()), 6),
            score_min       = round(float(scores.min()), 6),
            score_max       = round(float(scores.max()), 6),
            status          = status,
            recommendation  = recommendation,
            warning_flags   = warning_flags,
        )

        return labels, diag

    # ── Batch processing ──────────────────────────────────────────────────────

    def fit_transform_all(
        self,
        subject_scores: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, SubjectLabelDiagnostics]]:
        """
        Label all subjects and return (labels_dict, diagnostics_dict).

        The global score pool (concatenation of all subjects) is computed
        once and passed as the fallback for subjects with too few epochs.
        """
        all_scores = np.concatenate(list(subject_scores.values())).astype(np.float32)

        labels_out: Dict[str, np.ndarray]              = {}
        diags_out:  Dict[str, SubjectLabelDiagnostics] = {}

        for sid, scores in subject_scores.items():
            lbl, diag         = self.label_subject(sid, scores, global_scores=all_scores)
            labels_out[sid]   = lbl
            diags_out[sid]    = diag

        return labels_out, diags_out

    # ── Quality report ────────────────────────────────────────────────────────

    @staticmethod
    def save_quality_report(
        diags:      Dict[str, SubjectLabelDiagnostics],
        output_dir: Path,
        filename:   str = "label_quality_report.csv",
    ) -> Path:
        """
        Write per-subject diagnostics to a CSV quality report.

        Returns the path of the written file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename

        rows = [d.to_dict() for d in diags.values()]
        if not rows:
            return out_path

        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return out_path

    # ── Console summary ───────────────────────────────────────────────────────

    @staticmethod
    def print_summary(diags: Dict[str, SubjectLabelDiagnostics]) -> None:
        ok      = sum(1 for d in diags.values() if d.status == LabelStatus.OK.value)
        single  = sum(1 for d in diags.values()
                      if d.status in (LabelStatus.SINGLE_CLASS_HIGH.value,
                                      LabelStatus.SINGLE_CLASS_LOW.value))
        extreme = sum(1 for d in diags.values()
                      if d.status == LabelStatus.EXTREME_IMBALANCE.value)
        low_n   = sum(1 for d in diags.values()
                      if d.status == LabelStatus.LOW_SAMPLE_COUNT.value)
        total   = len(diags)

        print(f"\n  [EngagementLabelerV2] Summary ({total} subjects)")
        print(f"    OK                : {ok}")
        print(f"    Single-class      : {single}")
        print(f"    Extreme imbalance : {extreme}")
        print(f"    Low sample count  : {low_n}")
        print()

        for sid, d in sorted(diags.items()):
            flag = "  " if d.status == LabelStatus.OK.value else "⚠ "
            print(
                f"    {flag}{sid}  n={d.n_epochs:3d}  "
                f"HIGH={d.n_high}  LOW={d.n_low}  "
                f"entropy={d.entropy:.3f}  "
                f"thresh={d.threshold_used:.4f}  "
                f"[{d.status}]"
            )
