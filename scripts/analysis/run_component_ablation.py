"""
Component (leave-one-out) ablation runner — ONE variant per invocation
===========================================================================
Runs a full 37-fold LOSOCV for a single INS-HDGS-CMT component ablation,
holding EVERYTHING else identical to the blessed production focal model:

    focal γ=3.0 · α=effective_num · 5-member ensemble · zscore norm ·
    λ_dann=λ_mmd=0.10 · 3-channel ET · seed=NEUMA_SEED (default 42)

The ONLY thing that changes between variants is which architectural
component is disabled, selected via --variant and mapped to an
AblationConfig factory. The "full" variant is the baseline; it is identical
to the production model and exists so the harness is self-contained — for
paired stats you can instead reuse the production CSV (same seed → folds pair).

ET is PINNED to the blessed 3-channel config (left eye only, normalized) BEFORE
config.settings is imported, so this study is comparable to the published 0.79
model and NOT to the 9-channel default. This mirrors analysis/run_ablation_experiment.py.

This script touches NO model / loss / optimizer / scheduler / calibration code.

Run (full run uses all visible GPUs, ~2.5–3 h each on 8×A100):
    unset CUDA_VISIBLE_DEVICES                       # else it pins to 1 GPU!
    export PYTHONPATH=/home/nvidia/24PHD1314/Neuma_Model
    python analysis/run_component_ablation.py --variant no_snn

Variants:
    full no_snn no_graph no_neuro_symbolic ns_explain_only ns_rule_only
    no_et no_roi eeg_only
    no_contrastive no_mmd no_fusion_transformer baseline_linear

Revision runs (same pinned configuration, see docs/REVISION_RUNS.md):
    python analysis/run_component_ablation.py --variant eeg_only            # true EEG-only branch
    python analysis/run_component_ablation.py --variant ns_rule_only        # rule gate closed
    NEUMA_GRID_COLS=6 NEUMA_GRID_ROWS=4 python analysis/run_component_ablation.py         --variant full --label grid_6x4 --results-root output/metrics       # ROI-grid run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Parse args FIRST, then pin ET env BEFORE importing config.settings ───────
VARIANTS = [
    "full", "no_snn", "no_graph", "no_neuro_symbolic", "ns_explain_only", "ns_rule_only",
    "no_et", "no_roi", "eeg_only",
    "no_contrastive", "no_mmd", "no_fusion_transformer", "baseline_linear",
]

ap = argparse.ArgumentParser(description="Run one component-ablation LOSOCV")
ap.add_argument("--variant", required=True, choices=VARIANTS,
                help="which component to disable (full = production baseline)")
ap.add_argument("--epochs", type=int, default=None, help="override EPOCHS")
ap.add_argument("--focal-gamma", type=float, default=3.0,
                help="match production-best focal model (default 3.0)")
ap.add_argument("--alpha-strategy", type=str, default="effective_num",
                help="match production-best (default effective_num)")
ap.add_argument("--n-ensemble", type=int, default=5,
                help="ensemble size; 5 for parity with the production baseline")
ap.add_argument("--lambda-dann", type=float, default=0.10)
ap.add_argument("--lambda-mmd",  type=float, default=0.10)
ap.add_argument("--results-root", type=str, default=None)
ap.add_argument("--label", type=str, default=None,
                help="run label / output folder name (default abl_<variant>); e.g. grid_6x4 "
                     "when NEUMA_GRID_COLS/ROWS are set, so every revision run shares the "
                     "production-pinned configuration above")
fp = ap.add_mutually_exclusive_group()
fp.add_argument("--fold-parallel",    dest="fold_parallel", action="store_true")
fp.add_argument("--no-fold-parallel", dest="fold_parallel", action="store_false")
ap.set_defaults(fold_parallel=True)
args = ap.parse_args()

# ── Pin ET to the blessed 3-channel UN-NORMALIZED config (the 0.79 model) ─────
#    Must be set BEFORE config.settings is imported (it reads these at import).
#    NORMALIZE=0 is required: the blessed model used raw (un-z-scored) ET; per-
#    epoch z-score strips absolute gaze/pupil magnitude and drops AUC ~0.05.
#    Tag must be et3_b0_v0_s0_n0.
os.environ.setdefault("ET_USE_BOTH_EYES", "0")
os.environ.setdefault("ET_USE_VERGENCE",  "0")
os.environ.setdefault("ET_USE_SPEED",     "0")
os.environ.setdefault("ET_NORMALIZE",     "0")

# ── Now safe to import the project ───────────────────────────────────────────
_HERE = Path(__file__).resolve()
_PH8  = _HERE.parents[1]
_ROOT = _HERE.parents[2]
for p in (str(_PH8), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config.settings import (
    ET_INPUT_DIM, EPOCHS, N_ENSEMBLE, RANDOM_SEED,
)
from models.ins_hdgs_cmt import AblationConfig
from evaluation.losocv   import run_losocv

# variant name → AblationConfig factory
_FACTORY = {
    "full":                 AblationConfig.full,
    "no_snn":               AblationConfig.no_snn,
    "no_graph":             AblationConfig.no_graph,
    "no_neuro_symbolic":    AblationConfig.no_neuro_symbolic,
    "ns_explain_only":      AblationConfig.ns_explain_only,
    "ns_rule_only":         AblationConfig.ns_rule_only,
    "no_et":                AblationConfig.no_et,
    "eeg_only":             AblationConfig.eeg_only,      # no gaze-derived input at all
    "no_roi":               AblationConfig.no_roi,
    "no_contrastive":       AblationConfig.no_contrastive,
    "no_mmd":               AblationConfig.no_mmd,
    "no_fusion_transformer": AblationConfig.no_fusion_transformer,
    "baseline_linear":      AblationConfig.baseline_linear,
}


def main():
    label    = args.label or f"abl_{args.variant}"
    ablation = _FACTORY[args.variant]()

    print("=" * 75)
    print(f"COMPONENT ABLATION — {args.variant}   (label {label})")
    print("=" * 75)
    print(f"  ET_INPUT_DIM = {ET_INPUT_DIM}  (blessed 3-ch config; ET now hardcoded in settings)")
    assert ET_INPUT_DIM == 3, (
        f"ET_INPUT_DIM={ET_INPUT_DIM} — expected 3-ch blessed config. "
        f"Do not override ET_* env to enable extra channels for this study."
    )
    print(f"  focal_gamma={args.focal_gamma}  alpha={args.alpha_strategy}  "
          f"n_ensemble={args.n_ensemble or N_ENSEMBLE}  "
          f"epochs={args.epochs or EPOCHS}  seed={RANDOM_SEED}")
    print(f"  disabled flags: "
          f"snn={ablation.use_snn} graph={ablation.use_graph} "
          f"temporal={ablation.use_temporal} roi={ablation.use_roi} "
          f"et={ablation.use_et} fusion={ablation.use_fusion_transformer} "
          f"neuro_symbolic={ablation.use_neuro_symbolic} "
          f"contrastive={ablation.use_contrastive} mmd={ablation.use_mmd}")

    results_root = Path(args.results_root) if args.results_root else (_PH8 / "results" / "ablation")
    save_dir = results_root / label
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  results → {save_dir}/losocv_{label}.csv")
    print("=" * 75, flush=True)

    df = run_losocv(
        ablation             = ablation,
        epochs               = args.epochs or EPOCHS,
        label                = label,
        save_dir             = save_dir,
        fold_parallel        = args.fold_parallel,
        verbose              = True,
        alpha_strategy       = args.alpha_strategy,
        focal_gamma_override = args.focal_gamma,
        n_ensemble_override  = args.n_ensemble,
        lambda_dann_override = args.lambda_dann,
        lambda_mmd_override  = args.lambda_mmd,
        random_seed          = RANDOM_SEED,   # SAME seed → folds pair across variants
    )
    print(f"\n[{label}] DONE — {len(df)} folds → {save_dir}/losocv_{label}.csv")


if __name__ == "__main__":
    main()
