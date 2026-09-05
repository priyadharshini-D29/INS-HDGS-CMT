#!/usr/bin/env bash
# Component ablation (paper Table 7) — one full 37-fold LOSOCV per variant,
# every variant pinned to the production configuration
# (focal gamma=3, effective-number class weights, 5-member ensemble,
#  lambda_dann = lambda_mmd = 0.1, 3-channel eye tracking, seed 42).
#
#   bash reproducibility/run_ablation.sh                 # all variants
#   bash reproducibility/run_ablation.sh no_snn no_graph # a subset
#
# Results: results/ablation/abl_<variant>/losocv_abl_<variant>.csv
# Checkpoints: src/model/output/checkpoints/abl_<variant>/
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
VARIANTS=(
  full eeg_only eeg_only_mmd no_et no_roi
  no_graph no_snn no_fusion_transformer no_neuro_symbolic ns_rule_only
  no_contrastive no_mmd baseline_linear
)
RUN=("$@"); [ ${#RUN[@]} -eq 0 ] && RUN=("${VARIANTS[@]}")
mkdir -p logs results/ablation
for v in "${RUN[@]}"; do
  if [ -f "results/ablation/abl_${v}/losocv_abl_${v}.csv" ]; then
    echo "[ablation] $v already done — skip (delete the CSV to re-run)"; continue
  fi
  echo "[ablation] === variant: $v ==="
  ( cd src/model && python ../../scripts/analysis/run_component_ablation.py --variant "$v" ) \
    2>&1 | tee "logs/ablation_abl_${v}.log"
done
python scripts/analysis/compare_component_ablation.py || true
echo "[ablation] done. See results/ablation/component_ablation.md"
