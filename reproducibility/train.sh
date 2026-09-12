#!/usr/bin/env bash
# =============================================================================
# train.sh — Train the INS-HDGS-CMT headline model under LOSOCV.
#
# Runs leave-one-subject-out cross-validation with the exact hyperparameters
# reported in the paper (7-member ensemble per fold, focal gamma=3.0,
# effective-number class weighting, DANN + MMD subject-invariance).
#
# Outputs:
#   checkpoints/ins_hdgs_cmt_headline/   per-fold model weights
#   results/losocv_metrics/              per-fold metric CSVs
#
# Usage:  bash reproducibility/train.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"
export PYTHONHASHSEED="$SEED"

echo "[train] INS-HDGS-CMT LOSOCV training  (seed=$SEED)"
python src/model/main.py \
  --focal-gamma 3.0 \
  --alpha-strategy effective_num \
  --n-ensemble 5 \
  --lambda-dann 0.1 \
  --lambda-mmd 0.1 \
  --mmd-mode marginal \
  --norm-mode zscore \
  "$@"

echo "[train] done. Checkpoints in checkpoints/  metrics in results/losocv_metrics/"
