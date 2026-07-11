#!/usr/bin/env bash
# ================================================================
# PHASE 1 — GRL/DANN adversarial-weight ablation (subject invariance)
# ================================================================
# Sweeps lambda_dann ∈ {0.1, 0.25, 0.5, 1.0, 2.0}.
# The progressive GRL schedule  λ_grl = 2/(1+e^{-10p}) - 1  is ALREADY
# applied inside the Trainer (training/trainer.py); this sweep varies the
# DANN loss weight only. λ_dann=0.1 reproduces the v17 baseline under the
# controlled N_ENSEMBLE=5 protocol.
#
# Self-chaining: waits for any still-running focal_abl sweep to finish
# before starting, so the 8 GPUs are not oversubscribed.
# Skips any config whose results CSV already exists (safe to resume).

set -uo pipefail
cd "$(dirname "$0")/.."

source /home/nvidia/.venv/bin/activate
export PYTHONPATH=/home/nvidia/24PHD1314/Neuma_Model:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUTF8=1

N_ENSEMBLE=5
EPOCHS=250
# λ_dann=0.0 → adversarial term fully off (true plain-model reference);
# 0.1 reproduces the v17 baseline under the controlled N_ENSEMBLE=5 protocol.
LAMBDAS=(0.0 0.1 0.25 0.5 1.0 2.0)

# ── Wait for any running focal_abl sweep to release the GPUs ────────────────
echo "[phase1] $(date -u)  checking for running focal_abl sweep …"
_waited=0
while pgrep -f "main.py --fold-parallel --label focal_abl" >/dev/null 2>&1 \
   || pgrep -f "run_focal_sweep" >/dev/null 2>&1; do
    if (( _waited % 600 == 0 )); then
        echo "[phase1] focal_abl sweep still running — waiting … (${_waited}s)"
    fi
    sleep 60
    _waited=$((_waited + 60))
    # safety cap: 18h
    if (( _waited > 64800 )); then
        echo "[phase1] WARN: waited 18h, proceeding anyway."
        break
    fi
done
echo "[phase1] GPUs clear — starting DANN sweep at $(date -u)"

run_one() {
    local ld=$1
    local tag
    tag=$(echo "$ld" | sed 's/\./p/')
    local label="si_dann_l${tag}"
    local csv="output/metrics/${label}/losocv_${label}.csv"
    local log="phase1_${label}.log"

    if [[ -f "$csv" ]]; then
        echo "[SKIP] $label — results exist"
        return 0
    fi
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  PHASE 1 : $label   (lambda_dann=$ld, N_ENS=$N_ENSEMBLE)"
    echo "  $(date -u)"
    echo "════════════════════════════════════════════════════"

    python main.py \
        --fold-parallel \
        --label        "$label" \
        --epochs       "$EPOCHS" \
        --n-ensemble   "$N_ENSEMBLE" \
        --lambda-dann  "$ld" \
        2>&1 | tee "$log"

    if [[ -f "$csv" ]]; then echo "  ✓ $label done"; else echo "  ✗ $label FAILED"; fi
}

for ld in "${LAMBDAS[@]}"; do
    run_one "$ld"
done

echo ""
echo "[phase1] sweep complete: $(date -u)"
echo "[phase1] evaluate with:"
echo "  python subject_invariance_eval.py --label si_dann_l0p1,si_dann_l0p25,si_dann_l0p5,si_dann_l1p0,si_dann_l2p0"
