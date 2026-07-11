#!/usr/bin/env bash
# =============================================================================
# reproduce_paper.sh — One command to reproduce every reported result.
#
# End-to-end pipeline: train (LOSOCV) -> evaluate -> ablation study ->
# tables -> figures. This is the script a reviewer runs to reproduce the
# manuscript from scratch (given the preprocessed dataset; see datasets/).
#
# Expected wall-clock: several hours to ~1 day on a single modern GPU
# (see docs/REPRODUCIBILITY_CHECKLIST.md for measured timings and expected
# metric values). Set QUICK=1 to run a reduced smoke version.
#
# Usage:  bash reproducibility/reproduce_paper.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export SEED="${SEED:-42}"

echo "=============================================================="
echo " INS-HDGS-CMT — full paper reproduction  (seed=$SEED)"
echo "=============================================================="

echo; echo ">>> [1/5] Training (LOSOCV headline model)";  bash reproducibility/train.sh
echo; echo ">>> [2/5] Evaluation";                        bash reproducibility/evaluate.sh
echo; echo ">>> [3/5] Ablation study";                    bash reproducibility/run_ablation.sh
echo; echo ">>> [4/5] Generating tables";                 bash reproducibility/generate_tables.sh
echo; echo ">>> [5/5] Generating figures";                bash reproducibility/generate_figures.sh

echo; echo "=============================================================="
echo " Done. Compare results/ and tables/ against docs/REPRODUCIBILITY_CHECKLIST.md"
echo "=============================================================="
