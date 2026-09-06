#!/usr/bin/env bash
# =============================================================================
#  INS-HDGS-CMT — reviewer-revision driver (run on the GPU server / Brev)
# =============================================================================
#  Usage (from the repository root):
#     bash scripts/revision/run_revision.sh <stage> [<stage> ...]
#     bash scripts/revision/run_revision.sh all
#
#  Stages, in the order `all` runs them:
#     env        GPU / torch / data sanity check (no training)
#     data       raw .xdf -> preprocessed epochs + production labels (skips done subjects)
#     audit      R1/R2 label audit: linear recoverability of the label per modality
#     full       production model, 37-fold LOSOCV, fresh checkpoints  (needed by `ckpt`)
#     eeg_only   R1-1/3  strictly gaze-free EEG branch (MMD also off, as in the original factory)
#     eeg_only_mmd R1-1/3 gaze-free EEG branch with the MMD/DANN regularisers kept (fair "full minus gaze")
#     rule_only  R2-5    model trained with the rule gate closed (alpha = 0)
#     baselines  R2-3    18 baselines, nested 12-config search, early stopping (parallel over GPUs)
#     tau        R2-2    tau in {0.20..0.50}, EEG-only branch, all 37 folds
#     grid       R2-9    ROI grid 2x1 / 3x2 / 6x4 / 8x6, full model
#     lc         R2-6    learning curve over 10..37 subjects
#     ckpt       R2-4/5/7 rule fidelity, IG rule grounding, measured SNN cost (GPU power)
#     stats      all paired statistics / tables / figures from the CSVs above
#     verify     python scripts/revision/verify_revision.py  (what is done, what is missing)
#
#  Label tracks.  By default everything runs on the production rule-based label
#  (NEUMA_LABEL_SOURCE=phase3d, results under results/, checkpoints under
#  src/model/output/).  For the behavioural purchase-intent label run
#     bash scripts/revision/run_revision.sh purchase_labels          # once
#     NEUMA_LABEL_SOURCE=purchase bash scripts/revision/run_revision.sh full eeg_only_mmd no_et no_roi baselines purchase_stats
#  which writes to results/label_purchase/ and src/model/output_purchase/.
#
#  Every stage is idempotent: it skips work whose output CSV already exists.
#  Logs: logs/revision/<stage>.log.  Nothing here edits the manuscript.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
export PYTHONUTF8=1 PYTHONHASHSEED=42
mkdir -p logs/revision
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true     # DataLoader workers share many tensors
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
SUBJECTS="$(cd src/model && python -c 'from config.settings import SUBJECT_IDS; print(",".join(SUBJECT_IDS))')"
PROD_CSV="results/losocv_metrics/losocv_repro_focal_g3p0_effective_num_37.csv"
LS="${NEUMA_LABEL_SOURCE:-phase3d}"; export NEUMA_LABEL_SOURCE="$LS"
if [ "$LS" = phase3d ]; then RES_ROOT="results"; OUT="output"
else RES_ROOT="results/label_${LS}"; OUT="output_${LS}"; export NEUMA_OUTPUT_DIR="$OUT"; fi
mkdir -p "$RES_ROOT" "logs/revision/$LS"; LOGD="logs/revision/$LS"
# Behavioural tracks (purchase/product) hold ~2,300 pre-computed graph tensors per
# fold; multi-worker DataLoaders share every tensor with each worker and exhaust
# the fd limit ("Too many open files").  Graphs are pre-computed, so workers buy
# nothing: default to in-process loading there.  phase3d keeps the published
# setting (4 workers) so the production numbers stay reproducible.
if [ "$LS" != phase3d ]; then export NEUMA_NUM_WORKERS="${NEUMA_NUM_WORKERS:-0}"; fi
# Refuse to start on top of training processes left over from an interrupted run
# (fold workers are multiprocessing "spawn" children: their argv shows spawn_main).
# Stages are CPU-bound and leave the GPUs mostly idle, so several stages may run
# at once (one tmux window each): opt in with NEUMA_ALLOW_CONCURRENT=1, which
# skips this check and caps each worker at 4 CPU threads to limit oversubscription.
if [ -n "${NEUMA_ALLOW_CONCURRENT:-}" ]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  echo "NEUMA_ALLOW_CONCURRENT set: running alongside other stages (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
elif pgrep -f "run_component_ablation.py|run_baselines.py|sensitivity_sweep.py|learning_curve.py" >/dev/null 2>&1; then
  echo "Training processes from a previous run are still alive:"; pgrep -af "run_component_ablation.py|run_baselines.py|sensitivity_sweep.py|learning_curve.py"
  echo "If they belong to a stage that is still running, start this one with NEUMA_ALLOW_CONCURRENT=1."
  echo "If they are left over from a crash, kill them first:  pkill -f run_component_ablation.py; pkill -f run_baselines.py; pkill -f spawn_main"; exit 1
fi
if [ -z "${NEUMA_ALLOW_CONCURRENT:-}" ] && pgrep -f "multiprocessing.spawn" >/dev/null 2>&1; then
  echo "WARNING: orphaned spawn workers are running (pgrep -af multiprocessing.spawn); kill with: pkill -f spawn_main"
fi

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
run_in_model() { ( cd src/model && "$@" ); }            # every training command runs from src/model
abl() {  # abl <variant> [extra args]  -> $RES_ROOT/ablation/abl_<variant>/losocv_abl_<variant>.csv
  local v="$1"; shift
  local csv="$RES_ROOT/ablation/abl_${v}/losocv_abl_${v}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -gt 1 ]; then log "abl_${v} [$LS]: done — skip"; return; fi
  [ -f "$csv" ] && { log "abl_${v} [$LS]: previous run produced an EMPTY csv — removing and re-running"; rm -rf "$RES_ROOT/ablation/abl_${v}"; }
  log "abl_${v} [label=$LS]: start (fold-parallel over ${NGPU} GPU(s)) -> $RES_ROOT/ablation/abl_${v}"
  run_in_model python ../../scripts/analysis/run_component_ablation.py --variant "$v" --results-root "$ROOT/$RES_ROOT/ablation" "$@" 2>&1 | tee "$LOGD/abl_${v}.log"
}

stage_env() {
  log "GPUs: ${NGPU}"; nvidia-smi --query-gpu=name,memory.total --format=csv || true
  python -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
  local n; n="$(ls -d src/data_pipeline/04_segmentation/S*/output/engagement_phase3d 2>/dev/null | wc -l)"
  log "subjects with production (phase-3D) labels: ${n} / 42"
  [ "$n" -ge 42 ] || log "WARNING: run stage 'data' first (needs DataSource/*.xdf) or unpack dist/neuma_preprocessed_bundle.tgz"
  [ -f "$PROD_CSV" ] || { echo "missing $PROD_CSV"; exit 1; }
}

stage_data() {
  [ -d DataSource ] || { echo "DataSource/ with S*.xdf not found at repo root"; exit 1; }
  for x in DataSource/S*.xdf; do
    s="$(basename "$x" .xdf)"
    if [ -f "src/data_pipeline/04_segmentation/$s/output/epochs/eeg_epochs.npy" ]; then continue; fi
    log "preprocess $s"
    ( cd src/data_pipeline/03_preprocessing && NEUMA_SKIP_PLOTS=1 python main_phase2.py --subject "$s" --skip-plots ) >> logs/revision/data.log 2>&1
    ( cd src/data_pipeline/04_segmentation && NEUMA_SKIP_PLOTS=1 python main_phase3.py --subject "$s" ) >> logs/revision/data.log 2>&1
  done
  log "labels: gaze-only (engagement/) and PRODUCTION multimodal (engagement_phase3d/)"
  ( cd src/data_pipeline/04_segmentation && python engagement_labeling.py && python engagement_phase3d.py ) 2>&1 | tail -20 | tee -a logs/revision/data.log
}

stage_audit() {
  log "label recoverability audit [label=$LS]"
  python scripts/analysis/label_leakage_audit.py --label-source "$LS" --out-dir "$RES_ROOT/statistics" 2>&1 | tee "$LOGD/audit.log"
}

stage_purchase_labels() {   # behavioural label from the questionnaire (Q77) + gaze-on-product; needs DataSource/ (xlsx + Dependencies)
  [ -d DataSource/Dependencies ] || { echo "DataSource/Dependencies (bounding boxes) not found at repo root"; exit 1; }
  log "purchase-intent window labels -> S*/output/engagement_purchase/"
  ( cd src/data_pipeline/04_segmentation && python purchase_labeling.py ) 2>&1 | tail -15 | tee logs/revision/purchase_labels.log
}

stage_product_epochs() {   # NeuMa-native product-level epochs + purchase labels (needs DataSource/: xlsx, xdf, Dependencies)
  [ -d DataSource/Dependencies ] || { echo "DataSource/Dependencies not found at repo root"; exit 1; }
  log "product-level epochs -> S*/output/engagement_product/ + epochs/*_product.npy"
  ( cd src/data_pipeline/04_segmentation && python product_epoching.py ) 2>&1 | grep -E "^\s+S[0-9]+:|epochs [0-9]|epochs/subject|written" | tee logs/revision/product_epochs.log
}

stage_no_et()  { abl no_et; }
stage_no_roi() { abl no_roi; }

stage_purchase_stats() {   # paired comparisons within the current label track (all CSVs must come from this track)
  local A="$RES_ROOT/ablation"
  log "summary of every LOSOCV run under $A"
  python - "$A" <<'PY'
import sys, glob, pandas as pd
from pathlib import Path
for f in sorted(glob.glob(f"{sys.argv[1]}/abl_*/losocv_*.csv")):
    d = pd.read_csv(f); print(f"{Path(f).parent.name:22s} n={len(d):2d}  BalAcc {d.balanced_acc.mean():.3f}±{d.balanced_acc.std():.3f}  AUC {d.roc_auc.mean():.3f}±{d.roc_auc.std():.3f}  MCC {d.mcc.mean():.3f}")
PY
  if [ -f "$A/abl_full/losocv_abl_full.csv" ]; then
    python scripts/analysis/cross_modal_contribution.py --full-csv "$A/abl_full/losocv_abl_full.csv"       --no-et-csv "$A/abl_no_et/losocv_abl_no_et.csv" --no-roi-csv "$A/abl_no_roi/losocv_abl_no_roi.csv"       --no-fusion-csv "$A/abl_no_fusion_transformer/losocv_abl_no_fusion_transformer.csv"       $( [ -f "$A/abl_eeg_only_mmd/losocv_abl_eeg_only_mmd.csv" ] && echo "--eeg-only-csv $A/abl_eeg_only_mmd/losocv_abl_eeg_only_mmd.csv" )       --et-only-probs "$RES_ROOT/baselines/dl_tuned/fold_probs/probs_et_lstm.csv" --out-dir "$RES_ROOT/statistics" 2>&1 | tail -30 || true
  fi
}

stage_full()      { abl full; }
stage_eeg_only()  { abl eeg_only; }
stage_eeg_only_mmd() { abl eeg_only_mmd; }   # gaze-free but keeps the MMD/DANN regularisers
stage_rule_only() { abl ns_rule_only; }

stage_baselines() {
  local models="eegnet shallow deep cnn_lstm cnn_bilstm eeg_transformer tsception gat brain_gcn fusion_mlp et_lstm et_gru et_transformer late_fusion dual_transformer cross_attention mm_transformer dynamicgat_et"
  local i=0 pids=()
  for m in $models; do
    if [ -f "$RES_ROOT/baselines/dl_tuned/losocv_${m}.csv" ]; then log "baseline ${m} [$LS]: done — skip"; continue; fi
    local gpu=$(( NGPU > 0 ? i % NGPU : 0 ))
    log "baseline ${m} -> GPU ${gpu}"
    ( cd src/model && CUDA_VISIBLE_DEVICES="$gpu" python baselines/run_baselines.py --models "$m" --tune 12 --epochs 250 --patience 30 --early-stop --device cuda --out-dir "$ROOT/$RES_ROOT/baselines/dl_tuned" ) > "$LOGD/baseline_${m}.log" 2>&1 &
    pids+=($!); i=$((i+1))
    if [ "$NGPU" -gt 0 ] && [ $(( i % NGPU )) -eq 0 ]; then wait "${pids[@]}"; pids=(); fi
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  log "baselines finished; per-fold CSVs in $RES_ROOT/baselines/dl_tuned/"
}

stage_tau() {
  if [ -f src/model/output/metrics/sensitivity_threshold_summary.csv ]; then log "tau sweep: done — skip"; return; fi
  log "tau sweep (EEG-only branch, 37 folds, 3-member ensemble, 120 epochs)"
  run_in_model python sensitivity_sweep.py --sweep threshold --subjects "$SUBJECTS" --n-ensemble 3 --epochs 120 2>&1 | tee logs/revision/tau.log
}

stage_grid() {
  for g in 2x1 3x2 6x4 8x6; do
    if [ -f "src/model/output/metrics/grid_${g}/losocv_grid_${g}.csv" ]; then log "grid ${g}: done — skip"; continue; fi
    log "grid ${g}: full model, production configuration"
    ( cd src/model && NEUMA_GRID_COLS="${g%x*}" NEUMA_GRID_ROWS="${g#*x}" python ../../scripts/analysis/run_component_ablation.py --variant full --label "grid_${g}" --results-root output/metrics ) 2>&1 | tee "logs/revision/grid_${g}.log"
  done
}

stage_lc() {
  log "learning curve (sizes 10,16,24,30,37 x 3 draws; 3-member ensemble, 150 epochs)"
  run_in_model python ../../scripts/analysis/learning_curve.py --sizes 10,16,24,30,37 --repeats 3 --n-ensemble 3 --epochs 150 2>&1 | tee logs/revision/lc.log
}

stage_ckpt() {
  local ck="src/model/$OUT/checkpoints/abl_full"
  [ -d "$ck" ] || { echo "no checkpoints in $ck — run stage 'full' first"; exit 1; }
  local fold1; fold1="$(ls "$ck"/abl_full_fold01_e*.pt | head -1)"
  log "rule fidelity (all folds)"
  ( cd src/model && CUDA_VISIBLE_DEVICES="" python ../../scripts/analysis/rule_fidelity.py --ckpt-dir "$OUT/checkpoints/abl_full" --label abl_full \
      --fold-csv "$ROOT/$RES_ROOT/ablation/abl_full/losocv_abl_full.csv" --out-dir "$ROOT/$RES_ROOT/statistics" \
      $( [ -f "$RES_ROOT/ablation/abl_ns_rule_only/losocv_abl_ns_rule_only.csv" ] && echo "--rule-only-csv $ROOT/$RES_ROOT/ablation/abl_ns_rule_only/losocv_abl_ns_rule_only.csv" ) ) 2>&1 | tee "$LOGD/rule_fidelity.log"
  log "rule grounding (integrated gradients), single subject and population"
  ( cd src/model && CUDA_VISIBLE_DEVICES="" python ../../scripts/analysis/ground_rules_to_electrodes.py --ckpt "../../$fold1" --subjects S01 --label abl_full_S01 ) 2>&1 | tee logs/revision/grounding_S01.log
  ( cd src/model && CUDA_VISIBLE_DEVICES="" python ../../scripts/analysis/ground_rules_to_electrodes.py --ckpt "$ck"/abl_full_fold*_e0.pt --subjects all --label abl_full ) 2>&1 | tee logs/revision/grounding_all.log
  stage_energy
}
stage_ckpt_cpu() { NEUMA_SKIP_ENERGY=1 stage_ckpt; }   # rule fidelity + grounding only (CPU); run `energy` later on an idle GPU

stage_energy() {   # board power is sampled with nvidia-smi: run this only when no other job is on the GPU
  [ -n "${NEUMA_SKIP_ENERGY:-}" ] && { log "energy measurement skipped (NEUMA_SKIP_ENERGY)"; return; }
  local ck="src/model/$OUT/checkpoints/abl_full"; local fold1; fold1="$(ls "$ck"/abl_full_fold01_e*.pt | head -1)"
  local busy; busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
  [ "$busy" -gt 0 ] && log "WARNING: $busy other GPU processes are running; the power readings will include them"
  log "measured SNN cost on GPU (board power sampled with nvidia-smi)"
  ( cd src/model && CUDA_VISIBLE_DEVICES="${NEUMA_ENERGY_GPU:-0}" python ../../scripts/analysis/snn_energy_measured.py --ckpt "../../$fold1" --device cuda ) 2>&1 | tee logs/revision/snn_energy_gpu.log
}

stage_stats() {
  local eeg="results/ablation/abl_eeg_only/losocv_abl_eeg_only.csv"
  log "cross-modal contribution (R1-1/2/3, Table S14, Sec. 3.4b)"
  python scripts/analysis/cross_modal_contribution.py $( [ -f "$eeg" ] && echo "--eeg-only-csv $eeg" ) 2>&1 | tail -30 | tee logs/revision/stats_crossmodal.log
  if [ -f "$eeg" ] && ls results/baselines/dl_tuned/losocv_*.csv >/dev/null 2>&1; then
    log "Table 3 / 8 / S6: gaze-free EEG branch vs tuned EEG baselines"
    for m in balanced_acc mcc roc_auc; do
      python scripts/analysis/verify_table7_eeg_significance.py --baseline-dir results/baselines/dl_tuned --ref-csv "$eeg" --metric "$m" \
        --out "results/statistics/table7_eeg_significance_tuned_${m}.csv" 2>&1 | tail -15
    done
    log "Tables 4 / 5 / S10: full model vs tuned ET and fusion baselines"
    run_in_model python baselines/aggregate_baselines.py --dir ../../results/baselines/dl_tuned --full-csv "../../$PROD_CSV" 2>&1 | tail -5
  fi
  log "Table 7: component ablation vs full"
  python scripts/analysis/compare_component_ablation.py 2>&1 | tail -25 || true
  log "tau / grid / thresholds"
  run_in_model python ../../scripts/analysis/tau_sensitivity.py 2>&1 | tail -15
  run_in_model python ../../scripts/analysis/roi_grid_sensitivity.py 2>&1 | tail -25
  python scripts/analysis/deployment_threshold.py 2>&1 | tail -15
  [ -f results/sensitivity/learning_curve.csv ] || run_in_model python ../../scripts/analysis/learning_curve.py --summarise-only 2>&1 | tail -10 || true
  cp -f results/sensitivity/fig_tau_sensitivity.pdf paper/figures/figS4_tau_sensitivity.pdf 2>/dev/null || true
  cp -f results/sensitivity/fig_roi_grid_sensitivity.pdf paper/figures/figS6_roi_grid_sensitivity.pdf 2>/dev/null || true
  cp -f results/sensitivity/fig_learning_curve.pdf paper/figures/figS5_learning_curve.pdf 2>/dev/null || true
}

stage_verify() { python scripts/revision/verify_revision.py; }

ALL="env data audit full eeg_only eeg_only_mmd rule_only baselines tau grid lc ckpt stats verify"   # also: ckpt_cpu, energy, no_et, no_roi, purchase_labels, product_epochs, purchase_stats
# purchase track (after `purchase_labels`):  NEUMA_LABEL_SOURCE=purchase ... full eeg_only_mmd no_et no_roi baselines purchase_stats
# product track  (after `product_epochs`):   NEUMA_LABEL_SOURCE=product  ... audit full eeg_only_mmd no_et no_roi baselines purchase_stats
[ $# -gt 0 ] || { echo "usage: $0 <stage>...|all   (stages: $ALL)"; exit 1; }
for st in "$@"; do
  if [ "$st" = all ]; then for s2 in $ALL; do "stage_$s2"; done
  else "stage_$st"; fi
done
log "finished: $*"
