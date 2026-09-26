"""
================================================================
NEUMA Phase 8 — Manuscript Detail Report
================================================================
Prints/extracts the concrete numbers the review asked for that are
currently missing or unverifiable from the manuscript text alone:

  - window length / step size, epoch overlap, ICA component count
  - optimizer / LR schedule / early-stopping criterion (patience,
    min-delta, min-epochs-before-stop)
  - parameter count (full model vs EEG-only branch)
  - cross-modal attention Q/K/V dimensionality
  - per-fold runtime & peak GPU memory (from an existing benchmark
    JSON, if a LOSOCV run has already been saved)
  - genuine neuro-symbolic IF-THEN rules, decoded from an actual
    TRAINED checkpoint with real electrode names (NOT the output of
    `main.py --explain`, which builds a freshly-initialised, UNTRAINED
    model and would print meaningless/random rule text)

Usage
-----
  cd NEUMA_PHASE8

  # static facts only (no checkpoint needed)
  python manuscript_report.py

  # + genuine trained rules, pulling one real epoch from S02's cache
  python manuscript_report.py --label ins_hdgs_cmt --fold 1 --ensemble-idx 0 --subject S02
================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode unicode arrows/tau.
os.environ.setdefault("PYTHONUTF8", "1")
try:
    if isinstance(sys.stdout, __import__("io").TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # for `NEUMA_PHASE8.inference...` absolute imports
sys.path.insert(0, str(_HERE))

import torch

from config.settings import (
    EEG_SR, EEG_EPOCH_LEN, N_WINDOWS, WINDOW_SAMPLES, EPOCH_OVERLAP,
    CONN_METHOD, CONN_THRESHOLD, LABEL_MODE,
    EMBED_DIM, FUSION_HEADS, T_NHEAD, T_LAYERS, T_FF_DIM,
    GAT_L1_HEADS, GAT_L1_HEAD_DIM,
    LR, WEIGHT_DECAY, PATIENCE, EPOCHS,
    N_ENSEMBLE, RANDOM_SEED,
    CKPT_DIR, BENCHMARK_DIR,
    ET_INPUT_DIM, N_ROIS,
)
from data.channel_harmonizer import CANONICAL_CHANNELS, CANONICAL_N
from models.ins_hdgs_cmt import INS_HDGS_CMT, AblationConfig


def _print_header(title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def report_window_config() -> None:
    _print_header("WINDOW / EPOCH CONFIGURATION  (Table 2 / Sec 2.5.2)")
    step_s = WINDOW_SAMPLES / EEG_SR
    print(f"  Sampling rate            : {EEG_SR} Hz")
    print(f"  Epoch length             : {EEG_EPOCH_LEN} s "
          f"({int(EEG_SR * EEG_EPOCH_LEN)} samples)")
    print(f"  N_WINDOWS per epoch      : {N_WINDOWS}")
    print(f"  Window length (= step,   : {step_s:.3f} s  "
          f"({WINDOW_SAMPLES} samples)  [non-overlapping, contiguous]")
    print(f"    no inter-window overlap)")
    print(f"  Epoch-level overlap      : {EPOCH_OVERLAP:.0%}  "
          f"(circular-shift augmentation between consecutive raw epochs,")
    print(f"                              not between the {N_WINDOWS} PLV windows within an epoch)")
    print(f"  Connectivity method      : {CONN_METHOD!r}  "
          f"(\"plv_theta\" etc. for narrowband PLV — see sensitivity_sweep.py)")
    print(f"  Connectivity threshold (tau) : {CONN_THRESHOLD}")
    try:
        sys.path.insert(0, str(_HERE.parent / "NEUMA_PHASE2"))
        from configs.preprocessing_config import ICA_COMPONENTS
        print(f"  ICA components           : {ICA_COMPONENTS}  "
              f"(fixed hyperparameter, NOT selected via explained-variance "
              f"threshold — NEUMA_PHASE2/configs/preprocessing_config.py)")
    except Exception as e:
        print(f"  ICA components           : (could not import Phase2 config: {e})")
    print(f"  Canonical channel count   : {CANONICAL_N}  {CANONICAL_CHANNELS}")


def report_optimizer_schedule() -> None:
    _print_header("OPTIMIZER / LR SCHEDULE / EARLY STOPPING  (Sec 2.5.10 / Table 2)")
    print(f"  Optimizer                : AdamW  (lr={LR}, weight_decay={WEIGHT_DECAY})")
    print(f"  LR schedule              : CosineAnnealingWarmRestarts"
          f"(T_0=50, T_mult=1, eta_min=1e-6)")
    print(f"                              stepped once per EPOCH (not per-metric plateau)")
    print(f"                              -- source: training/trainer.py Trainer.__init__")
    print(f"  Early-stopping criterion : validation BALANCED ACCURACY (not loss)")
    print(f"  Early-stopping patience  : {PATIENCE} epochs without improvement")
    print(f"  Early-stopping min-delta : 1e-4  (val_bal_acc must improve by > this to reset patience)")
    print(f"  Minimum epochs before early stop can fire : 20")
    print(f"  Checkpoint restore       : best validation-balanced-accuracy epoch's weights"
          f" reloaded at end of fit()")
    print(f"  Ensemble size (N_ENSEMBLE): {N_ENSEMBLE} members/fold, seeds = "
          f"{RANDOM_SEED} + i*997")


def report_label_mode() -> None:
    _print_header("ENGAGEMENT LABEL MODE  (branch: subject-median-losocv-labels)")
    print(f"  Active LABEL_MODE        : {LABEL_MODE!r}  "
          f"(env override: NEUMA_LABEL_MODE)")
    print("  global          -> population/global-threshold labels; can leave some")
    print("                     held-out subjects single-class (undefined AUC/MCC).")
    print("  subject_median  -> per-subject median split; every subject has both")
    print("                     classes, at the cost of a per-subject-relative")
    print("                     (not population-absolute) definition of 'high engagement'.")
    print("  -> To report both and let reviewers see the sensitivity of headline")
    print("     numbers to this choice, use: analysis/compare_label_modes.py")
    print("     (already present on this branch) rather than re-deriving it.")


def report_baseline_tuning_parity() -> None:
    _print_header("BASELINE HYPERPARAMETER TUNING PARITY  (Fix 8)")
    print("  baselines/run_baselines.py exists on this branch with CLI flags mirroring")
    print("  the proposed model's knobs: --epochs, --lr, --wd, --batch-size, --seed,")
    print("  --label-mode, --max-folds. To state a defensible tuning-parity sentence")
    print("  in the manuscript, rerun baselines with the SAME epochs/lr/wd/batch-size/")
    print("  seed/label-mode as the headline INS-HDGS-CMT run and cite that, e.g.:")
    print(f"    python baselines/run_baselines.py --models eegnet --epochs {EPOCHS} \\")
    print(f"        --lr {LR} --batch-size 32 --seed {RANDOM_SEED} --label-mode {LABEL_MODE}")
    print("  (adjust --models to the full baseline list; check run_baselines.py --help)")


def report_validation_subject_protocol() -> None:
    _print_header("LOSOCV VALIDATION-SUBJECT SELECTION  (addresses prior Fix 6 ambiguity)")
    print("  Per fold: candidate_train = [all subjects != test_subject]")
    print("            rng.shuffle(candidate_train)   # seeded by RANDOM_SEED, per-fold state")
    print("            val_subj   = candidate_train[0]")
    print("            train_subs = candidate_train[1:]")
    print("  -> a DIFFERENT validation subject is drawn for EACH fold (not one fixed")
    print("     subject reused across all 37 folds). lambda weights are read from")
    print("     config as fixed constants (LAMBDA_CLS etc.) -- they are not re-tuned")
    print("     per fold; only the *decision threshold* and *temperature* are fit on")
    print("     each fold's (different) validation subject.")
    print("  -> to confirm which subject was used for a specific fold, grep the run's")
    print("     stdout log for a line like '[Fold 03] TEST=S07 | VAL=<subject> | ...'")
    print("     -- source: evaluation/losocv.py::_run_fold_sequential")


def report_param_counts_and_dims() -> None:
    _print_header("PARAMETER COUNT & CROSS-MODAL ATTENTION DIMENSIONS")

    for name, ablation in [("full (Table 1 headline model)", AblationConfig.full()),
                            ("eeg_only (Table 3 leakage-free branch)", AblationConfig.eeg_only())]:
        model = INS_HDGS_CMT(
            n_eeg_ch=CANONICAL_N, n_et_ch=ET_INPUT_DIM, n_rois=N_ROIS,
            n_windows=N_WINDOWS, embed_dim=EMBED_DIM,
            gat_head_dim=GAT_L1_HEAD_DIM, gat_heads=GAT_L1_HEADS,
            t_nhead=T_NHEAD, t_layers=T_LAYERS, t_ff_dim=T_FF_DIM,
            fusion_heads=FUSION_HEADS, ablation=ablation,
        )
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:45s}: {n_params:,} params ({n_trainable:,} trainable)")

    print(f"\n  Cross-modal fusion (NeuroFusionTransformer):")
    print(f"    embed_dim (D)          : {EMBED_DIM}")
    print(f"    fusion_heads           : {FUSION_HEADS}")
    print(f"    per-head Q/K/V dim     : {EMBED_DIM // FUSION_HEADS}  "
          f"(nn.MultiheadAttention splits D across heads internally)")
    print(f"    Q/K/V projection dim   : {EMBED_DIM}  (before head split)")
    print(f"    transformer layers     : {T_LAYERS}   feedforward dim: {T_FF_DIM}")
    print(f"  GAT (DynamicGAT):")
    print(f"    L1 heads x head_dim    : {GAT_L1_HEADS} x {GAT_L1_HEAD_DIM} "
          f"= {GAT_L1_HEADS * GAT_L1_HEAD_DIM} (concatenated)")


def report_runtime_and_memory(label: str) -> None:
    _print_header("PER-FOLD RUNTIME & PEAK GPU MEMORY  (Sec on graph-construction cost)")
    bench_path = BENCHMARK_DIR / f"benchmark_{label}.json"
    if not bench_path.exists():
        print(f"  No benchmark file found at {bench_path}.")
        print(f"  This is written automatically by evaluation/losocv.py after any")
        print(f"  run_losocv(..., label={label!r}) call completes -- rerun (or rerun a")
        print(f"  quick subject-subset LOSOCV) and this section will populate.")
        return
    with open(bench_path) as f:
        summary = json.load(f)
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")
    if "mean_fold_time_s" in summary and "throughput_samp_s" in summary:
        print(f"\n  -> seconds/sample (graph construction + train + inference, "
              f"amortised): {1.0/summary['throughput_samp_s']:.4f} s/sample")


def report_trained_rules(label: str, fold: int, ensemble_idx: int, subject: str) -> None:
    _print_header("GENUINE NEURO-SYMBOLIC RULES  (Fig. 4 — from a TRAINED checkpoint)")
    ckpt_dir = CKPT_DIR / label
    candidates = [
        ckpt_dir / f"{label}_fold{fold:02d}_e{ensemble_idx}.pt",
        ckpt_dir / f"{label}_fold{fold:02d}.pt",
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        print(f"  No checkpoint found. Tried:")
        for p in candidates:
            print(f"    {p}")
        print(f"  Train at least one fold first (python main.py) so a checkpoint exists,")
        print(f"  then rerun with --label/--fold/--ensemble-idx matching it.")
        print(f"\n  IMPORTANT: `python main.py --explain` does NOT load a checkpoint at all")
        print(f"  -- it builds a freshly-initialised model, so any rule text it prints")
        print(f"  is from RANDOM weights, not the trained model. Use this script instead.")
        return

    model = INS_HDGS_CMT(
        n_eeg_ch=CANONICAL_N, n_et_ch=ET_INPUT_DIM, n_rois=N_ROIS,
        n_windows=N_WINDOWS, embed_dim=EMBED_DIM,
        gat_head_dim=GAT_L1_HEAD_DIM, gat_heads=GAT_L1_HEADS,
        t_nhead=T_NHEAD, t_layers=T_LAYERS, t_ff_dim=T_FF_DIM,
        fusion_heads=FUSION_HEADS, ablation=AblationConfig.full(),
        feature_names=CANONICAL_CHANNELS,   # real electrode names, not "feature_i"
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd["model_state_dict"] if isinstance(sd, dict) and "model_state_dict" in sd else sd
    incompat = model.load_state_dict(state, strict=False)
    print(f"  Loaded: {ckpt_path}")
    if incompat.missing_keys or incompat.unexpected_keys:
        print(f"  [WARN] missing={len(incompat.missing_keys)} "
              f"unexpected={len(incompat.unexpected_keys)} keys on load")
    model.eval()

    from data.dataset import NeumaGraphDataset
    ds = NeumaGraphDataset(subject_ids=[subject], precompute_graphs=True)
    if len(ds) == 0:
        print(f"  No cached epochs found for subject {subject!r}.")
        return
    sample = ds[0]
    with torch.no_grad():
        out = model(
            eeg_windows  = sample["eeg_windows"].unsqueeze(0),
            adj_matrices = sample["adj_matrices"].unsqueeze(0),
            et_seq       = sample["et_seq"].unsqueeze(0),
            roi_vector   = sample["roi_vector"].unsqueeze(0),
        )
    explanation = model.explain(out, class_names=["LOW_ENGAGEMENT", "HIGH_ENGAGEMENT"])
    print(f"\n  Global rules (decoded from trained rule_queries, real electrode names):")
    for r in explanation["rules_text"]:
        print(f"    {r}")
    print(f"\n{explanation.get('sample_text', '')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="ins_hdgs_cmt",
                    help="Checkpoint/benchmark experiment label to inspect")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--ensemble-idx", type=int, default=0)
    ap.add_argument("--subject", default=None,
                    help="Subject ID to pull one real cached epoch from, for the rules "
                         "section. Skips that section if omitted.")
    args = ap.parse_args()

    report_window_config()
    report_label_mode()
    report_optimizer_schedule()
    report_validation_subject_protocol()
    report_param_counts_and_dims()
    report_baseline_tuning_parity()
    report_runtime_and_memory(args.label)
    if args.subject:
        report_trained_rules(args.label, args.fold, args.ensemble_idx, args.subject)
    else:
        _print_header("GENUINE NEURO-SYMBOLIC RULES")
        print("  Skipped (pass --subject S02 etc. to decode real rules from a "
              "trained checkpoint).")


if __name__ == "__main__":
    main()
