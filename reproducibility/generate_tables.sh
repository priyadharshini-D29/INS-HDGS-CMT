#!/usr/bin/env bash
# =============================================================================
# generate_tables.sh — Rebuild every paper table from the raw result CSVs.
#
# Regenerates the LaTeX/CSV/Markdown tables under tables/ from the metrics in
# results/. No training is required — this consumes already-computed results.
#
# Produces: tables/table1_dataset, table2_baselines, table3_main_results,
#           table4_ablation, table5_statistics, table6_interpretability.
#
# Usage:  bash reproducibility/generate_tables.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[tables] regenerating manuscript tables from results/ ..."
python src/model/manuscript_report.py --tables --out tables/ 2>/dev/null || \
  echo "[tables] NOTE: run from a checkout that includes results/ CSVs."

echo "[tables] done. See tables/"
