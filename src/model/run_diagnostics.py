"""
================================================================
INS-HDGS-CMT — Diagnostics CLI
================================================================
Entry-point for the full post-hoc diagnostics pipeline.

Usage
-----
  # Full analysis (skips embedding extraction for speed)
  python run_diagnostics.py --label ins_hdgs_cmt --skip-embeddings

  # Complete analysis with t-SNE / UMAP latent-space plots
  python run_diagnostics.py --label ins_hdgs_cmt

  # Skip heavy data-loading (use if dataset files not mounted)
  python run_diagnostics.py --label ins_hdgs_cmt --skip-data-stats

  # Compare ablation experiments
  python run_diagnostics.py --label ins_hdgs_cmt \\
      --ablation-labels "full,eeg_only,no_mmd" \\
      --ablation-csvs   "output/metrics/losocv_ins_hdgs_cmt.csv,\\
                         output/metrics/losocv_eeg_only.csv,\\
                         output/metrics/losocv_no_mmd.csv"

  # Point to a custom results CSV or checkpoint directory
  python run_diagnostics.py \\
      --label my_exp \\
      --results-csv /path/to/losocv_my_exp.csv \\
      --ckpt-dir    /path/to/checkpoints/my_exp \\
      --output-dir  /path/to/analysis

Outputs (all saved to output/analysis/<label>/)
-------
  subject_failure_analysis.csv / .json
  failure_type_counts.json
  calibration_ece_per_fold.csv
  calibration_diagnostics.json
  subject_distribution.csv / .json
  fold_variance_summary.csv
  fold_variance_analysis.json
  embeddings.csv
  summary.txt / .json
  mcc_histogram.png
  subject_mcc_bar.png
  calibration_curves.png
  confidence_distributions.png
  easy_hard_comparison.png
  fold_variance_boxplot.png
  failure_type_pie.png
  tsne_by_engagement.png  (if --skip-embeddings not set)
  tsne_by_subject.png     (if --skip-embeddings not set)
  umap_by_engagement.png  (if umap-learn installed)
  umap_by_subject.png     (if umap-learn installed)
  ablation_summary.csv / .png  (if --ablation-* flags provided)
================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add this package's root to path before any project imports.
# The parent dir is also needed: some modules (e.g. data/dataset.py) do
# `from model.inference...` as an absolute package import.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="INS-HDGS-CMT diagnostics and failure analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--label", type=str, default="ins_hdgs_cmt",
        help="Experiment label — used to auto-locate CSV and checkpoints",
    )
    p.add_argument(
        "--results-csv", type=str, default=None, dest="results_csv",
        help="Explicit path to LOSOCV results CSV (overrides auto-detection)",
    )
    p.add_argument(
        "--output-dir", type=str, default=None, dest="output_dir",
        help="Root for analysis outputs  (default: output/analysis/<label>)",
    )
    p.add_argument(
        "--ckpt-dir", type=str, default=None, dest="ckpt_dir",
        help="Checkpoint directory  (default: output/checkpoints/<label>)",
    )
    p.add_argument(
        "--subjects", type=str, default=None,
        help="Comma-separated subject IDs for distribution analysis "
             "(default: all configured subjects)",
    )
    p.add_argument(
        "--config", type=str, default="full",
        choices=[
            "full", "phase8c", "eeg_only", "eeg_et", "eeg_gat",
            "no_roi", "no_graph", "no_contrastive", "no_mmd", "no_temporal",
        ],
        help="Model ablation config used when loading checkpoints for embedding "
             "extraction",
    )
    p.add_argument(
        "--skip-embeddings", action="store_true", dest="skip_embeddings",
        help="Skip latent-space visualisation (t-SNE / UMAP)  — faster",
    )
    p.add_argument(
        "--skip-data-stats", action="store_true", dest="skip_data_stats",
        help="Skip per-subject EEG/ET distribution profiling  — faster",
    )
    # Ablation comparison
    p.add_argument(
        "--ablation-labels", type=str, default=None, dest="ablation_labels",
        help="Comma-separated ablation labels, e.g. full,eeg_only,no_mmd",
    )
    p.add_argument(
        "--ablation-csvs", type=str, default=None, dest="ablation_csvs",
        help="Comma-separated paths to ablation LOSOCV CSVs "
             "(must match --ablation-labels order)",
    )
    return p.parse_args()


# ── Config map ─────────────────────────────────────────────────────────────────

def _make_ablation(config: str):
    from models.ins_hdgs_cmt import AblationConfig
    _map = {
        "full"          : AblationConfig.full,
        "phase8c"       : AblationConfig.full,
        "eeg_only"      : AblationConfig.eeg_only,
        "eeg_et"        : AblationConfig.eeg_et,
        "eeg_gat"       : AblationConfig.eeg_gat,
        "no_roi"        : AblationConfig.no_roi,
        "no_graph"      : AblationConfig.no_graph,
        "no_contrastive": AblationConfig.no_contrastive,
        "no_mmd"        : AblationConfig.no_mmd,
        "no_temporal"   : lambda: AblationConfig(use_temporal=False),
    }
    factory = _map.get(config, AblationConfig.full)
    return factory()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    from config.settings      import SUBJECT_IDS, CKPT_DIR, METRICS_DIR
    from data.dataset         import _normalize_subject_id
    from evaluation.diagnostics import DiagnosticsRunner

    # ── Resolve paths ──────────────────────────────────────────────────────────
    results_csv = (Path(args.results_csv) if args.results_csv
                   else METRICS_DIR / f"losocv_{args.label}.csv")
    ckpt_dir    = (Path(args.ckpt_dir) if args.ckpt_dir
                   else CKPT_DIR / args.label)
    output_dir  = (Path(args.output_dir) if args.output_dir
                   else Path("output/analysis"))

    # ── Subjects ───────────────────────────────────────────────────────────────
    if args.subjects:
        subject_ids = [_normalize_subject_id(s.strip())
                       for s in args.subjects.split(",")]
    else:
        subject_ids = SUBJECT_IDS

    # ── Ablation config ────────────────────────────────────────────────────────
    try:
        ablation = _make_ablation(args.config)
    except Exception as e:
        print(f"[WARN] Could not build ablation config: {e}")
        ablation = None

    # ── Run diagnostics ────────────────────────────────────────────────────────
    runner = DiagnosticsRunner(
        label        = args.label,
        results_csv  = results_csv,
        output_dir   = output_dir,
        ckpt_dir     = ckpt_dir,
        subject_ids  = subject_ids,
        ablation     = ablation,
    )
    runner.run_all(
        skip_embeddings = args.skip_embeddings,
        skip_data_stats = args.skip_data_stats,
    )

    # ── Optional ablation comparison ───────────────────────────────────────────
    if args.ablation_labels and args.ablation_csvs:
        labels = [lbl.strip() for lbl in args.ablation_labels.split(",")]
        csvs   = [c.strip()   for c   in args.ablation_csvs.split(",")]
        if len(labels) != len(csvs):
            print("[WARN] --ablation-labels and --ablation-csvs count mismatch "
                  "— skipping ablation comparison.")
        else:
            print("\n  Running ablation comparison …")
            DiagnosticsRunner.ablation_summary(
                ablation_csvs = dict(zip(labels, csvs)),
                output_dir    = output_dir / args.label,
            )


if __name__ == "__main__":
    main()
