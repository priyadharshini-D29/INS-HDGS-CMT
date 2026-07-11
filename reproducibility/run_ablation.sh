#!/usr/bin/env bash
# =============================================================================
# run_ablation.sh — Reproduce the full component-ablation study (paper Table 6).
#
# Trains and evaluates one LOSOCV run per ablated variant, disabling a single
# architectural component at a time so its marginal contribution is isolated.
# Each variant writes to results/ablation/<variant>/ and logs to logs/.
#
# Variants:
#   eeg_only               EEG branch only (no eye-tracking)
#   et_only                Eye-tracking branch only (no EEG)
#   full                   Full multimodal fusion (reference)
#   no_neuro_symbolic      Remove differentiable rule layer
#   no_graph               Remove dynamic functional graph / GAT
#   no_fusion_transformer  Remove cross-modal fusion transformer
#   no_snn                 Remove spiking (LIF) encoder
#   no_roi                 Remove ROI attention
#   no_cross_attention     Remove cross-attention (self-attention only)
#   no_dynamic_connectivity  Static (fixed) adjacency instead of dynamic graphs
#   no_contrastive         Remove contrastive objective
#   no_mmd                 Remove MMD subject-invariance term
#
# Usage:  bash reproducibility/run_ablation.sh            # all variants
#         bash reproducibility/run_ablation.sh no_snn     # a single variant
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANTS=(
  eeg_only et_only full
  no_neuro_symbolic no_graph no_fusion_transformer no_snn no_roi
  no_cross_attention no_dynamic_connectivity no_contrastive no_mmd
)

RUN=("$@"); [ ${#RUN[@]} -eq 0 ] && RUN=("${VARIANTS[@]}")

mkdir -p logs results/ablation
for v in "${RUN[@]}"; do
  echo "[ablation] === variant: $v ==="
  python src/model/main.py --ablation "$v" \
    --output results/ablation/"$v" 2>&1 | tee "logs/ablation_abl_${v}.log"
done

# Aggregate every variant into one comparison table.
python src/model/aggregate_ablations.py --out tables/table4_ablation.csv || true
echo "[ablation] done. See results/ablation/ and tables/table4_ablation.csv"
