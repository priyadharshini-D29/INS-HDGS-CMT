#!/usr/bin/env bash
# ================================================================
# PHASE 3 — Subject-aware normalization ablation
# ================================================================
# norm_mode ∈ {zscore (existing), robust (median/MAD), instance, layer}
# applied to EEG features before graph construction. Graph cache is keyed
# per norm_mode so schemes never reuse each other's graphs.
# All other settings at baseline (lambda_dann=0.10, lambda_mmd=0.10 marginal).
#
# Self-chaining: waits for focal_abl / si_dann / si_mmd sweeps to finish.

set -uo pipefail
cd "$(dirname "$0")/.."
source /home/nvidia/.venv/bin/activate
export PYTHONPATH=/home/nvidia/24PHD1314/Neuma_Model:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUTF8=1

N_ENSEMBLE=5
EPOCHS=250
MODES=(zscore robust instance layer)

echo "[phase3] $(date -u)  waiting for prior sweeps (focal/si_dann/si_mmd) …"
_w=0
while pgrep -f "main.py --fold-parallel --label focal_abl" >/dev/null 2>&1 \
   || pgrep -f "main.py --fold-parallel --label si_dann"  >/dev/null 2>&1 \
   || pgrep -f "main.py --fold-parallel --label si_mmd"   >/dev/null 2>&1 \
   || pgrep -f "run_focal_sweep" >/dev/null 2>&1 \
   || pgrep -f "run_phase1_dann_sweep" >/dev/null 2>&1 \
   || pgrep -f "run_phase2_mmd_sweep" >/dev/null 2>&1; do
    sleep 60; _w=$((_w+60))
    (( _w % 600 == 0 )) && echo "[phase3] still waiting … (${_w}s)"
    (( _w > 144000 )) && { echo "[phase3] WARN 40h cap, proceeding"; break; }
done
echo "[phase3] GPUs clear — starting normalization sweep at $(date -u)"

run_one() {
    local nm=$1
    local label="si_norm_${nm}"
    local csv="output/metrics/${label}/losocv_${label}.csv"
    [[ -f "$csv" ]] && { echo "[SKIP] $label"; return 0; }
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  PHASE 3 : $label  (norm_mode=$nm)"
    echo "  $(date -u)"
    echo "════════════════════════════════════════════════════"
    python main.py \
        --fold-parallel \
        --label      "$label" \
        --epochs     "$EPOCHS" \
        --n-ensemble "$N_ENSEMBLE" \
        --norm-mode  "$nm" \
        2>&1 | tee "phase3_${label}.log"
    [[ -f "$csv" ]] && echo "  ✓ $label done" || echo "  ✗ $label FAILED"
}

for nm in "${MODES[@]}"; do
    run_one "$nm"
done
echo ""
echo "[phase3] sweep complete: $(date -u)"
