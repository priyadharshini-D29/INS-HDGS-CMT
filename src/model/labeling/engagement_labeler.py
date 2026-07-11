"""
================================================================
INS-HDGS-CMT — Engagement Labeler
================================================================
Converts raw Eye-Tracking (ET) epoch sequences into cognitive
engagement labels:

  HIGH_ENGAGEMENT (1)
  LOW_ENGAGEMENT  (0)

Optional secondary labels (when behavioral metadata available):
  BUY / NOT_BUY
  HIGH / MEDIUM / LOW visual attention
  POSITIVE / NEUTRAL / NEGATIVE preference

Labeling strategy
-----------------
Per-epoch ET metrics are extracted from the raw gaze sequence:

  1. fixation_ratio      — fraction of epoch with fixations
  2. mean_dwell_time     — mean gaze dwell per ROI cell
  3. roi_density         — peak ROI cell concentration
  4. pupil_dilation_norm — normalised pupil size
  5. gaze_entropy        — scanpath entropy (high = scattered)
  6. revisit_count       — ROI revisit events

A composite engagement score is computed as a weighted sum of
pro-engagement features minus anti-engagement features.

Thresholds are dataset-median derived (subject-independent) to
guarantee balanced label distributions for LOSOCV.

Scientific basis
----------------
Fixation duration ↑ → attention/engagement (Rayner, 1998)
Pupil dilation    ↑ → cognitive load / interest (Kahneman, 1973)
Gaze entropy      ↓ → focused viewing → engagement
ROI revisit       ↑ → recall / interest (Wedel & Pieters, 2008)
Alpha suppression ↑ → cortical engagement (Klimesch, 1999)
================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Constants ────────────────────────────────────────────────────────────────

HIGH_ENGAGEMENT = 1
LOW_ENGAGEMENT  = 0

# Composite score weights
_SCORE_WEIGHTS = {
    "fixation_ratio"    : +0.30,
    "mean_dwell_time"   : +0.25,
    "roi_density"       : +0.20,
    "pupil_dilation_norm": +0.15,
    "revisit_count"     : +0.10,
    "gaze_entropy"      : -0.30,   # negative: high entropy = low engagement
}


# ── Per-Epoch ET Feature Extraction ─────────────────────────────────────────

def _safe_float_array(et_epoch: np.ndarray) -> np.ndarray:
    """Convert ET epoch to clean float32 array; zero-fills bad values."""
    if not isinstance(et_epoch, np.ndarray):
        et_epoch = np.asarray(et_epoch)
    if et_epoch.dtype.kind not in ("f", "i", "u"):
        rows = []
        for row in et_epoch:
            safe = []
            for v in row:
                try:
                    safe.append(float(v))
                except (TypeError, ValueError):
                    safe.append(0.0)
            rows.append(safe)
        et_epoch = np.array(rows, dtype=np.float32)
    else:
        et_epoch = et_epoch.astype(np.float32)
    return np.nan_to_num(et_epoch, nan=0.0, posinf=0.0, neginf=0.0)


def _normalise_gaze(gx: np.ndarray, gy: np.ndarray,
                    image_w: float = 3000, image_h: float = 1688
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise gaze coordinates to [0, 1]."""
    gx = gx.copy()
    gy = gy.copy()
    if gx.max() > 2.0:
        gx = np.clip(gx / image_w, 0.0, 1.0)
    if gy.max() > 2.0:
        gy = np.clip(gy / image_h, 0.0, 1.0)
    return gx, gy


def extract_et_features(
    et_epoch  : np.ndarray,
    et_sr     : float = 120.0,
    grid_cols : int   = 5,
    grid_rows : int   = 2,
    image_w   : float = 3000.0,
    image_h   : float = 1688.0,
) -> Dict[str, float]:
    """
    Extract scalar cognitive engagement features from one ET epoch.

    Parameters
    ----------
    et_epoch  : (T, F) array — columns: gaze_x, gaze_y [, pupil, ...]
    et_sr     : eye-tracker sampling rate (Hz)
    grid_cols : horizontal ROI grid divisions
    grid_rows : vertical ROI grid divisions

    Returns
    -------
    dict of feature_name → float
    """
    arr = _safe_float_array(et_epoch)

    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 2:
        return {k: 0.0 for k in _SCORE_WEIGHTS}

    T = arr.shape[0]
    gx = arr[:, 0]
    gy = arr[:, 1]
    pupil = arr[:, 2] if arr.shape[1] >= 3 else np.zeros(T, dtype=np.float32)

    # ── Valid sample mask ─────────────────────────────────────────────────
    valid = (np.isfinite(gx) & np.isfinite(gy) &
             (gx > 0) & (gy > 0))
    if valid.sum() < 2:
        return {k: 0.0 for k in _SCORE_WEIGHTS}

    gx_v = gx[valid]
    gy_v = gy[valid]
    pu_v = pupil[valid]

    gx_n, gy_n = _normalise_gaze(gx_v, gy_v, image_w, image_h)

    # ── Fixation ratio ────────────────────────────────────────────────────
    # Velocity-based: consecutive distance below saccade threshold
    dx    = np.diff(gx_n)
    dy    = np.diff(gy_n)
    vel   = np.sqrt(dx**2 + dy**2) * et_sr          # deg/s proxy
    SACCADE_THRESH = 0.05                             # normalised units
    is_fix = np.concatenate([[False], vel < SACCADE_THRESH])
    fixation_ratio = float(is_fix.sum()) / max(len(is_fix), 1)

    # ── ROI cell assignment ───────────────────────────────────────────────
    col_idx  = np.clip((gx_n * grid_cols).astype(np.int32), 0, grid_cols - 1)
    row_idx  = np.clip((gy_n * grid_rows).astype(np.int32), 0, grid_rows - 1)
    cell_idx = row_idx * grid_cols + col_idx          # (T_valid,)
    n_cells  = grid_cols * grid_rows

    counts   = np.bincount(cell_idx, minlength=n_cells).astype(np.float32)
    total    = counts.sum()

    # Mean dwell time per occupied cell (normalised to [0,1])
    occupied    = counts[counts > 0]
    mean_dwell  = float(occupied.mean() / max(total, 1)) if len(occupied) > 0 else 0.0

    # ROI density: fraction of gaze in the single most-attended cell
    roi_density = float(counts.max() / max(total, 1))

    # ── Gaze entropy ──────────────────────────────────────────────────────
    p = counts / (total + 1e-10)
    p = p[p > 0]
    gaze_entropy = float(-np.sum(p * np.log2(p + 1e-10)))

    # ── Pupil dilation (normalised) ───────────────────────────────────────
    pmin, pmax = pu_v.min(), pu_v.max()
    if pmax > pmin:
        pupil_dilation_norm = float((pu_v.mean() - pmin) / (pmax - pmin))
    else:
        pupil_dilation_norm = 0.0

    # ── ROI revisit count ─────────────────────────────────────────────────
    # Count transitions back to a previously visited cell
    revisit = 0
    visited: set = set()
    for c in cell_idx:
        if c in visited:
            revisit += 1
        else:
            visited.add(c)
    revisit_count = float(revisit) / max(len(cell_idx), 1)

    return {
        "fixation_ratio"    : fixation_ratio,
        "mean_dwell_time"   : mean_dwell,
        "roi_density"       : roi_density,
        "pupil_dilation_norm": pupil_dilation_norm,
        "gaze_entropy"      : gaze_entropy,
        "revisit_count"     : revisit_count,
    }


def compute_engagement_score(features: Dict[str, float]) -> float:
    """
    Weighted composite engagement score from ET features.

    Score > 0 → HIGH engagement tendency
    Score < 0 → LOW engagement tendency
    """
    score = 0.0
    for feat, weight in _SCORE_WEIGHTS.items():
        score += weight * features.get(feat, 0.0)
    return float(score)


# ── Dataset-Level Label Assignment ──────────────────────────────────────────

class EngagementLabeler:
    """
    Compute subject-independent engagement labels for a collection of
    ET epochs using median-based thresholding.

    Usage
    -----
    labeler = EngagementLabeler()
    labels  = labeler.fit_transform(et_epochs)       # list of (T, F) arrays
    report  = labeler.report()
    """

    def __init__(
        self,
        balance_ratio : float = 0.40,
        et_sr         : float = 120.0,
        grid_cols     : int   = 5,
        grid_rows     : int   = 2,
        image_w       : float = 3000.0,
        image_h       : float = 1688.0,
        verbose       : bool  = True,
    ):
        self.balance_ratio = balance_ratio
        self.et_sr         = et_sr
        self.grid_cols     = grid_cols
        self.grid_rows     = grid_rows
        self.image_w       = image_w
        self.image_h       = image_h
        self.verbose       = verbose

        # Set after fit_transform()
        self.scores_   : Optional[np.ndarray] = None
        self.threshold_: Optional[float]       = None
        self.labels_   : Optional[np.ndarray] = None
        self.features_ : Optional[List[Dict]] = None

    def fit_transform(self, et_epochs: List[np.ndarray]) -> np.ndarray:
        """
        Compute engagement labels for a list of ET epochs.

        Threshold is set to the median score (dataset-level, subject-independent).
        If the resulting split is more imbalanced than `balance_ratio`, the
        threshold is iteratively adjusted to the nearest achievable balance point.

        Returns
        -------
        labels : (N,) int array  — 0=LOW_ENGAGEMENT, 1=HIGH_ENGAGEMENT
        """
        features = [
            extract_et_features(
                ep,
                et_sr     = self.et_sr,
                grid_cols = self.grid_cols,
                grid_rows = self.grid_rows,
                image_w   = self.image_w,
                image_h   = self.image_h,
            )
            for ep in et_epochs
        ]
        scores = np.array([compute_engagement_score(f) for f in features],
                          dtype=np.float32)

        # Median threshold (dataset-level, not per-subject)
        threshold = float(np.median(scores))

        # Check balance; adjust if too skewed
        labels = (scores >= threshold).astype(np.int64)
        n_high = int(labels.sum())
        n_low  = int((~labels.astype(bool)).sum())
        n_total = len(labels)
        minority = min(n_high, n_low) / max(n_total, 1)

        if minority < self.balance_ratio and n_total > 10:
            # Adjust threshold to the nearest percentile that satisfies balance
            target_pct = self.balance_ratio * 100
            lo_thresh  = float(np.percentile(scores, target_pct))
            hi_thresh  = float(np.percentile(scores, 100 - target_pct))
            # Choose the one that changes threshold least from median
            if abs(lo_thresh - threshold) < abs(hi_thresh - threshold):
                threshold = lo_thresh
            else:
                threshold = hi_thresh
            labels = (scores >= threshold).astype(np.int64)

        self.scores_    = scores
        self.threshold_ = threshold
        self.labels_    = labels
        self.features_  = features

        if self.verbose:
            self._print_report()

        return labels

    def _print_report(self):
        labels = self.labels_
        scores = self.scores_
        n      = len(labels)
        n_high = int(labels.sum())
        n_low  = n - n_high

        print(f"\n  [EngagementLabeler] ─────────────────────────────────")
        print(f"    Epochs              : {n}")
        print(f"    Threshold (median)  : {self.threshold_:.4f}")
        print(f"    HIGH_ENGAGEMENT     : {n_high}  ({100*n_high/n:.1f}%)")
        print(f"    LOW_ENGAGEMENT      : {n_low}   ({100*n_low/n:.1f}%)")
        print(f"    Score range         : [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"    Score mean±std      : {scores.mean():.3f} ± {scores.std():.3f}")

    def report(self) -> Dict:
        if self.labels_ is None:
            raise RuntimeError("Call fit_transform() first.")
        n      = len(self.labels_)
        n_high = int(self.labels_.sum())
        n_low  = n - n_high
        return {
            "n_epochs"      : n,
            "n_high"        : n_high,
            "n_low"         : n_low,
            "pct_high"      : round(100 * n_high / n, 2),
            "threshold"     : round(float(self.threshold_), 5),
            "score_mean"    : round(float(self.scores_.mean()), 5),
            "score_std"     : round(float(self.scores_.std()),  5),
        }

    def save_labels(self, out_dir: Path, subject_id: str = "all"):
        """Save labels and metadata to disk."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"engagement_labels_{subject_id}.npy", self.labels_)
        np.save(out_dir / f"engagement_scores_{subject_id}.npy", self.scores_)

        if self.features_:
            import csv
            feat_path = out_dir / f"engagement_features_{subject_id}.csv"
            keys = list(self.features_[0].keys()) + ["score", "label"]
            with open(feat_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for i, (feat, score, lbl) in enumerate(
                    zip(self.features_, self.scores_, self.labels_)
                ):
                    row = {k: round(v, 6) for k, v in feat.items()}
                    row["score"] = round(float(score), 6)
                    row["label"] = int(lbl)
                    w.writerow(row)

    def plot_distribution(self, out_path: Optional[Path] = None):
        """Plot engagement score distribution and threshold."""
        if self.scores_ is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Score histogram
        ax = axes[0]
        ax.hist(self.scores_, bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
        ax.axvline(self.threshold_, color="#DD4444", linewidth=2.0,
                   label=f"Threshold = {self.threshold_:.3f}")
        ax.set_xlabel("Composite Engagement Score")
        ax.set_ylabel("Count")
        ax.set_title("Engagement Score Distribution")
        ax.legend()

        # Label pie chart
        ax = axes[1]
        n_high = int(self.labels_.sum())
        n_low  = len(self.labels_) - n_high
        ax.pie(
            [n_high, n_low],
            labels=["HIGH", "LOW"],
            autopct="%1.1f%%",
            colors=["#2196F3", "#FF5722"],
            startangle=90,
        )
        ax.set_title("Engagement Label Distribution")

        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)


# ── Convenience Function ─────────────────────────────────────────────────────

def compute_engagement_label(
    et_epoch  : np.ndarray,
    threshold : float = 0.0,
    et_sr     : float = 120.0,
) -> int:
    """
    Compute engagement label for a single ET epoch given a pre-fit threshold.

    Returns HIGH_ENGAGEMENT (1) or LOW_ENGAGEMENT (0).
    """
    feats = extract_et_features(et_epoch, et_sr=et_sr)
    score = compute_engagement_score(feats)
    return HIGH_ENGAGEMENT if score >= threshold else LOW_ENGAGEMENT
