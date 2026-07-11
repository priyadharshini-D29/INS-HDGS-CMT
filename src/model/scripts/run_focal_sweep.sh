#!/usr/bin/env bash
# Focal-loss ablation sweep — one config at a time, each as a direct shell run.
# Uses the same launch pattern as the successful v17 run (no nested Python subprocess).
# Skips configs whose results CSV already exists.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/nvidia/.venv/bin/activate
export PYTHONPATH=/home/nvidia/24PHD1314/Neuma_Model:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUTF8=1

N_ENSEMBLE=5
EPOCHS=250

GAMMAS=(0.0 1.0 1.5 2.0 3.0)
ALPHAS=(balanced effective_num sqrt_inv_freq)

run_config() {
    local gamma=$1
    local alpha=$2
    local g_str
    g_str=$(echo "$gamma" | sed 's/\./p/')
    local label="focal_abl_g${g_str}_${alpha}"
    local results_csv="output/metrics/${label}/losocv_${label}.csv"
    local log_file="ablation_${label}.log"

    if [[ -f "$results_csv" ]]; then
        echo "[SKIP] $label — results already exist"
        return 0
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Running: $label"
    echo "  gamma=$gamma  alpha=$alpha  N_ENS=$N_ENSEMBLE"
    echo "  $(date -u)"
    echo "════════════════════════════════════════════════════"

    python main.py \
        --fold-parallel \
        --label        "$label" \
        --epochs       "$EPOCHS" \
        --alpha-strategy "$alpha" \
        --n-ensemble   "$N_ENSEMBLE" \
        --focal-gamma  "$gamma" \
        2>&1 | tee "$log_file"

    if [[ -f "$results_csv" ]]; then
        echo "  ✓ $label complete"
    else
        echo "  ✗ $label FAILED — no results CSV produced"
    fi
}

echo "Focal ablation sweep: $(date -u)"
echo "Configs: CE(γ=0) + γ∈{1.0,1.5,2.0,3.0} × α∈{balanced,effective_num,sqrt_inv_freq}"
echo ""

# CE baseline (γ=0, balanced only)
run_config 0.0 balanced

# Focal configs
for gamma in 1.0 1.5 2.0 3.0; do
    for alpha in balanced effective_num sqrt_inv_freq; do
        run_config "$gamma" "$alpha"
    done
done

echo ""
echo "Sweep complete: $(date -u)"
echo "Run: python analyze_focal_ablation.py"
