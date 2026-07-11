#!/usr/bin/env bash
# =============================================================================
# generate_figures.sh — Rebuild every paper figure from the raw results.
#
# Runs the plotting scripts under scripts/figures/ to regenerate the manuscript
# figures (overview, architecture, preprocessing, fusion, explainability,
# results) as PDF/PNG into figures/. No training required.
#
# Usage:  bash reproducibility/generate_figures.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[figures] regenerating manuscript figures ..."
if [ -d scripts/figures ]; then
  for f in scripts/figures/*.py; do
    [ -e "$f" ] || continue
    echo "  -> $f"
    python "$f" || echo "     (skipped $f — check its result-file dependencies)"
  done
else
  echo "[figures] scripts/figures/ not found; see figures/*/*_source.py generators."
fi

echo "[figures] done. See figures/"
