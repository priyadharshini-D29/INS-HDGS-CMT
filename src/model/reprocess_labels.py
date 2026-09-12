"""
================================================================
INS-HDGS-CMT — Engagement Label Reprocessing CLI
================================================================
Reads pre-computed engagement scores from Phase 3 output directories,
applies the v2 thresholding strategy (default: per-subject median),
and writes re-labeled outputs under:

  data_pipeline/04_segmentation/S##/output/engagement_v2/
    engagement_labels.npy          int64 (N,)  — 0=LOW, 1=HIGH
    engagement_scores_norm.npy     float32 (N,) — z-score normalised scores
    engagement_soft_labels.npy     float32 (N,) — sigmoid soft probs [optional]
    engagement_metadata.csv        per-epoch metadata

Also writes:
  output/analysis/label_quality_report.csv   (all subjects)
  output/analysis/label_quality_before.csv   (original labels for comparison)

Visualizations (per subject + global):
  output/analysis/label_plots/
    {SXX}_score_hist.png
    {SXX}_label_balance.png
    global_score_distribution.png
    global_class_balance.png
    threshold_comparison.png

Usage
-----
  # Reprocess all subjects with per-subject median (recommended)
  python reprocess_labels.py

  # Reprocess specific subjects only
  python reprocess_labels.py --subjects S31,S33,S21,S13

  # Use a different threshold mode
  python reprocess_labels.py --mode global_median
  python reprocess_labels.py --mode percentile --percentile 50

  # Also generate soft labels
  python reprocess_labels.py --soft-labels

  # Dry-run: diagnose without writing output files
  python reprocess_labels.py --dry-run
================================================================
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import PHASE3_DIR, SUBJECT_IDS
from labeling.engagement_labeler_v2 import (
    EngagementLabelerV2,
    ThresholdMode,
    SubjectLabelDiagnostics,
    LabelStatus,
)


# ── Source-score discovery ────────────────────────────────────────────────────

_SCORE_CANDIDATES = [
    # (subdir, scores_file, labels_file, tag)
    ("engagement_phase3d",
     "engagement_scores.npy", "engagement_labels.npy",
     "phase3d (multimodal)"),
    ("engagement",
     "engagement_scores.npy", "engagement_labels.npy",
     "phase3 (ET-only)"),
]


def _locate_scores(subject_dir: Path) -> Optional[tuple]:
    """Return (scores, orig_labels, source_tag) or None if not found."""
    for subdir, sf, lf, tag in _SCORE_CANDIDATES:
        sp = subject_dir / "output" / subdir / sf
        lp = subject_dir / "output" / subdir / lf
        if sp.exists() and lp.exists():
            scores       = np.load(sp, allow_pickle=True).astype(np.float32)
            orig_labels  = np.load(lp, allow_pickle=True).astype(np.int64)
            return scores, orig_labels, tag
    return None


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _save_subject_outputs(
    subject_dir: Path,
    scores:      np.ndarray,
    labels:      np.ndarray,
    diag:        SubjectLabelDiagnostics,
    soft:        Optional[np.ndarray],
) -> None:
    out_dir = subject_dir / "output" / "engagement_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # z-score normalised scores
    mu, sg = float(scores.mean()), float(scores.std()) or 1.0
    scores_norm = ((scores - mu) / sg).astype(np.float32)

    np.save(out_dir / "engagement_labels.npy",      labels)
    np.save(out_dir / "engagement_scores_norm.npy", scores_norm)

    if soft is not None:
        np.save(out_dir / "engagement_soft_labels.npy", soft)

    # per-epoch metadata CSV
    meta_path = out_dir / "engagement_metadata.csv"
    with open(meta_path, "w", newline="") as f:
        fieldnames = ["epoch_idx", "score_raw", "score_norm", "label"]
        if soft is not None:
            fieldnames.append("soft_label")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(len(labels)):
            row = {
                "epoch_idx":  i,
                "score_raw":  round(float(scores[i]), 6),
                "score_norm": round(float(scores_norm[i]), 6),
                "label":      int(labels[i]),
            }
            if soft is not None:
                row["soft_label"] = round(float(soft[i]), 6)
            w.writerow(row)


# ── Visualizations ────────────────────────────────────────────────────────────

def _plot_subject_hist(
    subject_id:  str,
    scores:      np.ndarray,
    threshold_v2: float,
    threshold_v1: float,
    labels_v2:   np.ndarray,
    labels_v1:   np.ndarray,
    out_dir:     Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{subject_id} — Engagement Score Distribution", fontsize=11)

    ax = axes[0]
    ax.hist(scores, bins=min(len(scores), 20), color="#4C72B0",
            alpha=0.75, edgecolor="white")
    ax.axvline(threshold_v2, color="#2ECC71", lw=2.0,
               label=f"v2 threshold = {threshold_v2:.3f}")
    ax.axvline(threshold_v1, color="#E74C3C", lw=2.0, ls="--",
               label=f"v1 threshold = {threshold_v1:.3f}")
    ax.set_xlabel("Engagement Score")
    ax.set_ylabel("Count")
    ax.set_title("Score histogram + thresholds")
    ax.legend(fontsize=8)

    ax = axes[1]
    n_high_v1 = int(labels_v1.sum())
    n_high_v2 = int(labels_v2.sum())
    n         = len(labels_v2)
    x         = [0, 1]
    ax.bar(x, [n_high_v1, n - n_high_v1], width=0.35, label="v1 (original)",
           color=["#E74C3C", "#95A5A6"], alpha=0.75)
    ax.bar([xi + 0.38 for xi in x],
           [n_high_v2, n - n_high_v2], width=0.35, label="v2 (new)",
           color=["#27AE60", "#BDC3C7"], alpha=0.75)
    ax.set_xticks([0.19, 1.19])
    ax.set_xticklabels(["HIGH", "LOW"])
    ax.set_ylabel("Count")
    ax.set_title("Label counts: v1 vs v2")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f"{subject_id}_score_hist.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_global_distribution(
    all_scores:     np.ndarray,
    global_thresh:  float,
    out_dir:        Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(all_scores, bins=40, color="#4C72B0", alpha=0.75, edgecolor="white")
    ax.axvline(global_thresh, color="#E74C3C", lw=2.0,
               label=f"global median = {global_thresh:.3f}")
    ax.set_xlabel("Engagement Score")
    ax.set_ylabel("Count")
    ax.set_title("Global Engagement Score Distribution (all subjects)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "global_score_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_class_balance_comparison(
    diags_v1: List[dict],
    diags_v2: List[SubjectLabelDiagnostics],
    out_dir:  Path,
) -> None:
    sids      = [d.subject_id for d in diags_v2]
    bal_v1    = [d["balance_score"] for d in diags_v1]
    bal_v2    = [d.balance_score   for d in diags_v2]

    x = np.arange(len(sids))
    fig, ax = plt.subplots(figsize=(max(10, len(sids) * 0.35), 4))
    ax.bar(x - 0.2, bal_v1, width=0.38, label="v1 (original)", color="#E74C3C", alpha=0.7)
    ax.bar(x + 0.2, bal_v2, width=0.38, label="v2 (new)",      color="#27AE60", alpha=0.7)
    ax.axhline(0.4, color="#888", lw=1.0, ls="--", label="min acceptable (0.40)")
    ax.set_xticks(x)
    ax.set_xticklabels(sids, rotation=90, fontsize=7)
    ax.set_ylabel("Balance score  (0.5 = perfect)")
    ax.set_title("Per-Subject Class Balance: v1 vs v2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "global_class_balance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_comparison(
    sids:    List[str],
    thresh_v1: float,
    thresh_v2: Dict[str, float],
    scores:    Dict[str, np.ndarray],
    out_dir:   Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(sids) * 0.35), 4))
    x = np.arange(len(sids))

    medians = [float(np.median(scores[s])) for s in sids]
    ax.scatter(x, medians, color="#27AE60", s=50, zorder=3, label="subject median (v2)")
    ax.axhline(thresh_v1, color="#E74C3C", lw=1.5, ls="--",
               label=f"global median (v1) = {thresh_v1:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(sids, rotation=90, fontsize=7)
    ax.set_ylabel("Threshold value")
    ax.set_title("Per-Subject Threshold: v1 (global) vs v2 (subject median)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Diagnostics for original labels ──────────────────────────────────────────

def _orig_diagnostics(subject_id: str, scores: np.ndarray,
                       labels: np.ndarray, global_thresh: float) -> dict:
    n      = len(labels)
    n_high = int(labels.sum())
    n_low  = n - n_high
    cr     = n_high / max(n, 1)
    bs     = min(n_high, n_low) / max(n, 1)
    return {
        "subject_id":    subject_id,
        "n_epochs":      n,
        "n_high":        n_high,
        "n_low":         n_low,
        "class_ratio":   round(cr, 4),
        "balance_score": round(bs, 4),
        "threshold":     round(float(global_thresh), 6),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reprocess engagement labels with v2 adaptive thresholding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--subjects", type=str, default=None,
        help="Comma-separated subject IDs to reprocess (default: all configured)",
    )
    p.add_argument(
        "--mode", type=str, default="subject_median",
        choices=[m.value for m in ThresholdMode],
        help="Thresholding strategy",
    )
    p.add_argument(
        "--percentile", type=float, default=50.0,
        help="Percentile for --mode percentile",
    )
    p.add_argument(
        "--zscore-thresh", type=float, default=0.0, dest="zscore_thresh",
        help="k for --mode zscore: threshold = mean + k*std",
    )
    p.add_argument(
        "--min-epochs", type=int, default=4, dest="min_epochs",
        help="Minimum epochs for per-subject thresholding (else global fallback)",
    )
    p.add_argument(
        "--soft-labels", action="store_true", dest="soft_labels",
        help="Also save sigmoid soft-label probabilities",
    )
    p.add_argument(
        "--label-smoothing", type=float, default=0.0, dest="label_smoothing",
        help="Label smoothing alpha [0, 1] applied to soft labels",
    )
    p.add_argument(
        "--output-dir", type=str, default="output/analysis", dest="output_dir",
        help="Root directory for quality report and plots",
    )
    p.add_argument(
        "--phase3-root", type=str, default=None, dest="phase3_root",
        help="Override data_pipeline/04_segmentation root directory",
    )
    p.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Diagnose without writing any output files",
    )
    p.add_argument(
        "--no-plots", action="store_true", dest="no_plots",
        help="Skip generating visualizations",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Paths
    phase3_root = (Path(args.phase3_root) if args.phase3_root
                   else _HERE.parent / "data_pipeline" / "04_segmentation")
    output_dir  = Path(args.output_dir)
    plot_dir    = output_dir / "label_plots"

    if not phase3_root.exists():
        print(f"[ERROR] data_pipeline/04_segmentation root not found: {phase3_root}")
        sys.exit(1)

    # Subject list
    if args.subjects:
        subject_ids = [s.strip() for s in args.subjects.split(",")]
    else:
        subject_ids = list(SUBJECT_IDS)

    print(f"\n  [reprocess_labels] mode={args.mode}  subjects={len(subject_ids)}")

    # ── Collect all scores ────────────────────────────────────────────────────
    subject_scores:  Dict[str, np.ndarray] = {}
    subject_dirs:    Dict[str, Path]       = {}
    source_tags:     Dict[str, str]        = {}
    orig_labels_map: Dict[str, np.ndarray] = {}

    missing = []
    for sid in subject_ids:
        sdir   = phase3_root / sid
        result = _locate_scores(sdir)
        if result is None:
            print(f"  [WARN] {sid}: no engagement scores found — skipping")
            missing.append(sid)
            continue
        scores, orig_labels, tag = result
        subject_scores[sid]  = scores
        subject_dirs[sid]    = sdir
        source_tags[sid]     = tag
        orig_labels_map[sid] = orig_labels

    processed_ids = [s for s in subject_ids if s not in missing]

    if not processed_ids:
        print("  [ERROR] No subjects with scores found.")
        sys.exit(1)

    # ── Global reference scores ───────────────────────────────────────────────
    all_scores     = np.concatenate(list(subject_scores.values()))
    global_thresh  = float(np.median(all_scores))
    print(f"  Global median = {global_thresh:.4f}  "
          f"(n_scores={len(all_scores)})")

    # ── Original label diagnostics (v1 reference) ─────────────────────────────
    diags_v1 = [
        _orig_diagnostics(sid, subject_scores[sid], orig_labels_map[sid], global_thresh)
        for sid in processed_ids
    ]

    # ── Apply v2 labeling ─────────────────────────────────────────────────────
    labeler = EngagementLabelerV2(
        mode            = ThresholdMode(args.mode),
        percentile      = args.percentile,
        zscore_thresh   = args.zscore_thresh,
        min_epochs      = args.min_epochs,
        soft_labels     = args.soft_labels,
        label_smoothing = args.label_smoothing,
    )

    labels_dict, diags_dict = labeler.fit_transform_all(subject_scores)

    # ── Print summary ─────────────────────────────────────────────────────────
    EngagementLabelerV2.print_summary(diags_dict)

    # ── Write outputs ─────────────────────────────────────────────────────────
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save per-subject labels
        for sid in processed_ids:
            scores  = subject_scores[sid]
            labels  = labels_dict[sid]
            diag    = diags_dict[sid]
            soft    = None
            if args.soft_labels:
                soft = labeler.compute_soft_labels(
                    scores, diag.threshold_used
                )
            _save_subject_outputs(subject_dirs[sid], scores, labels, diag, soft)
            src = source_tags[sid]
            print(f"  Saved {sid}  [{src}]  "
                  f"HIGH={diag.n_high}  LOW={diag.n_low}  [{diag.status}]")

        # Quality report (v2)
        report_path = EngagementLabelerV2.save_quality_report(
            diags_dict, output_dir, filename="label_quality_report.csv"
        )
        print(f"\n  Quality report  → {report_path}")

        # Quality report (v1 original for comparison)
        v1_path = output_dir / "label_quality_before.csv"
        if diags_v1:
            with open(v1_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(diags_v1[0].keys()))
                w.writeheader()
                w.writerows(diags_v1)
            print(f"  Original report → {v1_path}")

    else:
        print("  [dry-run] No files written.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots and not args.dry_run:
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Global distribution
        _plot_global_distribution(all_scores, global_thresh, plot_dir)

        # Per-subject histograms
        thresh_v2_map: Dict[str, float] = {
            sid: diags_dict[sid].threshold_used for sid in processed_ids
        }
        for sid in processed_ids:
            _plot_subject_hist(
                sid,
                subject_scores[sid],
                thresh_v2_map[sid],
                global_thresh,
                labels_dict[sid],
                orig_labels_map[sid],
                plot_dir,
            )

        # Global class balance comparison
        diags_v2_list = [diags_dict[sid] for sid in processed_ids]
        _plot_class_balance_comparison(diags_v1, diags_v2_list, plot_dir)

        # Threshold comparison
        _plot_threshold_comparison(
            processed_ids, global_thresh, thresh_v2_map,
            subject_scores, plot_dir,
        )

        print(f"  Plots saved     → {plot_dir}/")

    # ── Final change summary ──────────────────────────────────────────────────
    print("\n  Change summary (subjects where status improved):")
    v1_map = {d["subject_id"]: d for d in diags_v1}
    improved = 0
    degraded = 0
    for sid in processed_ids:
        d2 = diags_dict[sid]
        d1 = v1_map.get(sid, {})
        old_bs = d1.get("balance_score", 0.0)
        new_bs = d2.balance_score
        delta  = new_bs - old_bs
        if delta > 0.01:
            improved += 1
            print(f"    ✓ {sid}  balance_score {old_bs:.3f} → {new_bs:.3f}  "
                  f"(+{delta:.3f})  [{d2.status}]")
        elif delta < -0.01:
            degraded += 1
            print(f"    ✗ {sid}  balance_score {old_bs:.3f} → {new_bs:.3f}  "
                  f"({delta:.3f})  [{d2.status}]")

    print(f"\n  Improved: {improved}  |  Degraded: {degraded}  "
          f"|  Unchanged: {len(processed_ids) - improved - degraded}")
    print()


if __name__ == "__main__":
    main()
