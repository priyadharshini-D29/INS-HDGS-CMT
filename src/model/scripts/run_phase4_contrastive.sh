#!/usr/bin/env bash
# ================================================================
# PHASE 4 — EEG self-supervised contrastive pretraining ablation
# ================================================================
# (1) Pretrain the EEG encoder with SimCLR/NT-Xent on augmented EEG views
#     (pretrain_contrastive.py → output/checkpoints/ssl_eeg_encoder.pt).
# (2) Run two matched LOSOCV configs to isolate the pretraining effect:
#       si_contrastive_pretrained  — fine-tune from the SSL checkpoint
#       si_contrastive_baseline    — identical protocol, random init
#     Same epochs / N_ENSEMBLE / seed; the ONLY difference is the EEG-branch
#     initialisation, so the delta is attributable to pretraining.
#
# Self-chaining: waits for focal_abl / si_dann / si_mmd / si_norm sweeps to
# release the GPUs first. Skips configs whose results CSV already exists.

set -uo pipefail
cd "$(dirname "$0")/.."
source /home/nvidia/.venv/bin/activate
export PYTHONPATH=/home/nvidia/24PHD1314/Neuma_Model:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUTF8=1

N_ENSEMBLE=5
EPOCHS=250
PRETRAIN_EPOCHS=100
CKPT="output/checkpoints/ssl_eeg_encoder.pt"

echo "[phase4] $(date -u)  waiting for prior sweeps (focal/si_dann/si_mmd/si_norm) …"
_w=0
while pgrep -f "main.py --fold-parallel --label focal_abl" >/dev/null 2>&1 \
   || pgrep -f "main.py --fold-parallel --label si_dann"  >/dev/null 2>&1 \
   || pgrep -f "main.py --fold-parallel --label si_mmd"   >/dev/null 2>&1 \
   || pgrep -f "main.py --fold-parallel --label si_norm"  >/dev/null 2>&1 \
   || pgrep -f "run_focal_sweep" >/dev/null 2>&1 \
   || pgrep -f "run_phase1_dann_sweep" >/dev/null 2>&1 \
   || pgrep -f "run_phase2_mmd_sweep" >/dev/null 2>&1 \
   || pgrep -f "run_phase3_norm_sweep" >/dev/null 2>&1; do
    sleep 60; _w=$((_w+60))
    (( _w % 600 == 0 )) && echo "[phase4] still waiting … (${_w}s)"
    (( _w > 180000 )) && { echo "[phase4] WARN 50h cap, proceeding"; break; }
done
echo "[phase4] GPUs clear — starting Phase-4 at $(date -u)"

# ── Step 1: self-supervised pretraining (single GPU; cheap vs LOSOCV) ──────────
if [[ -f "$CKPT" ]]; then
    echo "[phase4] [SKIP] pretrain — checkpoint exists: $CKPT"
else
    echo "════════════════════════════════════════════════════"
    echo "  PHASE 4 : SSL pretraining  (epochs=$PRETRAIN_EPOCHS)"
    echo "  $(date -u)"
    echo "════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=0 python pretrain_contrastive.py \
        --epochs "$PRETRAIN_EPOCHS" --batch-size 256 \
        2>&1 | tee "phase4_pretrain.log"
fi
if [[ ! -f "$CKPT" ]]; then
    echo "[phase4] ✗ pretraining failed — no checkpoint. Aborting."
    exit 1
fi

# ── Step 2: matched fine-tune vs. baseline LOSOCV ──────────────────────────────
run_one() {
    local label=$1; shift
    local csv="output/metrics/${label}/losocv_${label}.csv"
    [[ -f "$csv" ]] && { echo "[SKIP] $label"; return 0; }
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  PHASE 4 : $label"
    echo "  $(date -u)"
    echo "════════════════════════════════════════════════════"
    python main.py \
        --fold-parallel \
        --label      "$label" \
        --epochs     "$EPOCHS" \
        --n-ensemble "$N_ENSEMBLE" \
        "$@" \
        2>&1 | tee "phase4_${label}.log"
    [[ -f "$csv" ]] && echo "  ✓ $label done" || echo "  ✗ $label FAILED"
}

run_one si_contrastive_pretrained --pretrained-eeg "$CKPT"
run_one si_contrastive_baseline

echo ""
echo "[phase4] sweep complete: $(date -u)"
echo "[phase4] evaluate with:"
echo "  python subject_invariance_eval.py --label si_contrastive_pretrained,si_contrastive_baseline"
