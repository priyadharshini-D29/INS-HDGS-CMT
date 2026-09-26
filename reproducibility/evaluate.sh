#!/usr/bin/env bash
# =============================================================================
# evaluate.sh — Score trained checkpoints and emit all evaluation metrics.
#
# Loads the per-fold ensemble checkpoints produced by train.sh, applies the
# fold-wise optimal operating threshold, and writes balanced accuracy, MCC,
# ROC-AUC, PR-AUC, Cohen's kappa and F1 to results/.
#
# Prerequisite: run reproducibility/train.sh first (or place checkpoints in
#               checkpoints/ins_hdgs_cmt_headline/).
#
# Usage:  bash reproducibility/evaluate.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[evaluate] scoring held-out LOSOCV folds ..."
python src/model/main.py --evaluate-only \
  --checkpoint-dir checkpoints/ins_hdgs_cmt_headline \
  "$@"

# Aggregate per-fold probabilities and optimise thresholds.
python src/model/collect_fold_probs.py || true
python src/model/threshold_optimizer.py || true

echo "[evaluate] metrics written to results/losocv_metrics/ and results/statistics/"
