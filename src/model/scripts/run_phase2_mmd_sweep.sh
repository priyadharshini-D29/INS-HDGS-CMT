#!/usr/bin/env bash
# ================================================================
# PHASE 2 — MMD distribution-alignment ablation (subject invariance)
# ================================================================
# Sweeps lambda_mmd ∈ {0.1, 0.25, 0.5, 1.0, 2.0} × {marginal, class_conditional}.
#
# NOTE: MMD was previously inert during training (compute_loss zeroes it on
# its training path). The Trainer now applies the MMD term on eeg_emb, so
# lambda_mmd is live. lambda_dann is held at its 0.10 baseline to isolate the
# MMD effect.  class_conditional aligns HIGH-HIGH and LOW-LOW across two
# balanced subject groups.
#
# Self-chaining: waits for focal_abl and si_dann (Phase 1) sweeps to finish.
# Skips configs whose results CSV already exists.

set -uo pipefail
cd "$(dirname "$0")/.."
source /home/nvidia/.venv/bin/activate
export PYTHONPATH=/home/nvidia/24PHD1314/Neuma_Model:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUTF8=1

N_ENSEMBLE=5
EPOCHS=250
# λ_mmd=0.0 → MMD alignment term fully off (plain-model reference).
LAMBDAS=(0.0 0.1 0.25 0.5 1.0 2.0)
MODES=(marginal class_conditional)

echo "[phase2] $(date -u)  waiting for focal_abl / si_dann sweeps to finish …"
_w=0
while pgrep -f "main.py --fold-parallel --label focal_abl" >/dev/null 2>&1 \
   || pgrep -f "main.py --fold-parallel --label si_dann"  >/dev/null 2>&1 \
   || pgrep -f "run_focal_sweep" >/dev/null 2>&1 \
   || pgrep -f "run_phase1_dann_sweep" >/dev/null 2>&1; do
    sleep 60; _w=$((_w+60))
    (( _w % 600 == 0 )) && echo "[phase2] still waiting … (${_w}s)"
    (( _w > 108000 )) && { echo "[phase2] WARN 30h cap, proceeding"; break; }
done
echo "[phase2] GPUs clear — starting MMD sweep at $(date -u)"

run_one() {
    local lm=$1 mode=$2
    local tag; tag=$(echo "$lm" | sed 's/\./p/')
    local mtag; if [[ "$mode" == "class_conditional" ]]; then mtag="cc"; else mtag="mg"; fi
    local label="si_mmd_${mtag}_l${tag}"
    local csv="output/metrics/${label}/losocv_${label}.csv"
    [[ -f "$csv" ]] && { echo "[SKIP] $label"; return 0; }
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  PHASE 2 : $label  (lambda_mmd=$lm  mmd_mode=$mode)"
    echo "  $(date -u)"
    echo "════════════════════════════════════════════════════"
    python main.py \
        --fold-parallel \
        --label       "$label" \
        --epochs      "$EPOCHS" \
        --n-ensemble  "$N_ENSEMBLE" \
        --lambda-mmd  "$lm" \
        --mmd-mode    "$mode" \
        2>&1 | tee "phase2_${label}.log"
    [[ -f "$csv" ]] && echo "  ✓ $label done" || echo "  ✗ $label FAILED"
}

for mode in "${MODES[@]}"; do
    for lm in "${LAMBDAS[@]}"; do
        # λ=0.0 disables MMD entirely, so class_conditional == marginal there.
        # Run the 0.0 reference once (under marginal) and skip the cc duplicate.
        if [[ "$lm" == "0.0" && "$mode" == "class_conditional" ]]; then
            echo "[SKIP] si_mmd_cc_l0p0 — identical to si_mmd_mg_l0p0 (MMD off)"
            continue
        fi
        run_one "$lm" "$mode"
    done
done
echo ""
echo "[phase2] sweep complete: $(date -u)"
