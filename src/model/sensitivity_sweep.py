"""
================================================================
NEUMA Phase 8 — Connectivity / Window-Length Sensitivity Sweep
================================================================
Addresses three reviewer concerns about the EEG-only branch used
for the leakage-free ranking (Table 3):

  Fix 5 : connectivity threshold tau=0.30 is fixed with no
          sensitivity analysis.
  Fix 3 : PLV is computed via Hilbert transform on broadband
          (1-45 Hz) EEG; instantaneous phase is only physiologically
          well-defined for a narrowband signal. This sweep compares
          broadband PLV against per-canonical-band PLV.
  Fix 4 : sub-second sliding windows (0.5 s default) give too few
          theta-band cycles for a stable PLV estimate; this sweep
          compares 0.5s / 1.0s / 2.5s windows.

This is a robustness *check*, not a full paper-scale re-run: it uses
a configurable subject subset and a reduced ensemble size so it
finishes in a reasonable time, per the reviewer's own suggestion
("on a subset of subjects... reviewers will accept this if it's
honest and clearly scoped as a robustness check").

Each grid point is executed in its own subprocess because
CONN_METHOD / CONN_THRESHOLD / N_WINDOWS (config/settings.py) are
imported by name (`from config.settings import CONN_METHOD`) all
over this codebase — those bindings are frozen at import time, so a
single long-lived process cannot safely switch configs between runs.
Subprocess isolation + env-var override sidesteps that entirely.

Usage
-----
  # Fix 5: threshold (tau) sensitivity, Pearson connectivity
  python sensitivity_sweep.py --sweep threshold \\
      --subjects S01,S02,S03,S05,S06,S07,S08,S09 \\
      --n-ensemble 3 --epochs 80

  # Fix 3: broadband vs narrowband PLV
  python sensitivity_sweep.py --sweep band \\
      --subjects S01,S02,S03,S05,S06,S07,S08,S09 \\
      --n-ensemble 3 --epochs 80

  # Fix 4: window length (0.5s / 1.0s / 2.5s)
  python sensitivity_sweep.py --sweep window \\
      --subjects S01,S02,S03,S05,S06,S07,S08,S09 \\
      --n-ensemble 3 --epochs 80

Each run trains the EEG-only ablation (AblationConfig.eeg_only() —
the same branch used for Table 3's leakage-free ranking) via the
existing run_losocv() pipeline, so results are directly comparable
to the paper's reported numbers, just at a smaller ensemble/subject
scale for speed.

Outputs
-------
  output/metrics/sens_<dim>_<value>/losocv_sens_<dim>_<value>.csv   (per config, from run_losocv)
  output/metrics/sensitivity_<dim>_summary.csv                      (comparison table)
================================================================
"""

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# (env_var_name, [values...]) per sweep dimension
GRIDS = {
    "threshold": ("NEUMA_CONN_THRESHOLD", [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]),
    "band":      ("NEUMA_CONN_METHOD",
                  ["pearson", "plv", "plv_delta", "plv_theta",
                   "plv_alpha", "plv_beta", "plv_gamma"]),
    "window":    ("NEUMA_N_WINDOWS", [10, 5, 2]),   # 0.5s / 1.0s / 2.5s epochs
}


def _label_for(dim: str, value, prefix: str) -> str:
    return f"{prefix}_{dim}_{str(value).replace('.', 'p')}"


def _run_one(dim: str, value, subjects: str, n_ensemble: int, epochs: int,
             prefix: str) -> None:
    env_key, _ = GRIDS[dim]
    label = _label_for(dim, value, prefix)
    env = os.environ.copy()
    env[env_key] = str(value)
    env["NEUMA_SWEEP_WORKER"]     = "1"
    env["NEUMA_SWEEP_LABEL"]      = label
    env["NEUMA_SWEEP_SUBJECTS"]   = subjects
    env["NEUMA_SWEEP_N_ENSEMBLE"] = str(n_ensemble)
    env["NEUMA_SWEEP_EPOCHS"]     = str(epochs)

    print(f"\n{'='*70}\n  RUN  {env_key}={value}   label={label}\n{'='*70}",
          flush=True)
    subprocess.run(
        [sys.executable, str(HERE / "sensitivity_sweep.py")],
        env=env, check=True, cwd=str(HERE),
    )


def _worker() -> None:
    """Runs inside the per-config subprocess: a single reduced LOSOCV run."""
    # data/dataset.py imports `model.inference...` as an absolute
    # package path, so the *parent* of this directory (src/model/, not just
    # this directory itself) must be importable too.
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(HERE))
    from models.ins_hdgs_cmt import AblationConfig
    from evaluation.losocv import run_losocv

    label      = os.environ["NEUMA_SWEEP_LABEL"]
    subjects   = os.environ["NEUMA_SWEEP_SUBJECTS"].split(",")
    n_ensemble = int(os.environ["NEUMA_SWEEP_N_ENSEMBLE"])
    epochs     = int(os.environ["NEUMA_SWEEP_EPOCHS"])

    # fold_parallel=True is safe even on a single GPU: run_losocv only takes
    # the multi-process fold-parallel path when NUM_GPUS > 1, otherwise it
    # falls back to the same sequential path used before.
    run_losocv(
        subject_ids         = subjects,
        ablation             = AblationConfig.eeg_only(),
        epochs               = epochs,
        label                = label,
        n_ensemble_override  = n_ensemble,
        # production-pinned loss / regularisation (as in run_component_ablation.py)
        alpha_strategy       = "effective_num",
        focal_gamma_override = 3.0,
        lambda_dann_override = 0.10,
        lambda_mmd_override  = 0.10,
        fold_parallel        = True,
    )


def _summarize(dim: str, prefix: str) -> None:
    sys.path.insert(0, str(HERE))
    from training.metrics import pooled_metrics_from_folds

    env_key, values = GRIDS[dim]
    rows = []
    for value in values:
        label    = _label_for(dim, value, prefix)
        csv_path = HERE / "output" / "metrics" / label / f"losocv_{label}.csv"
        if not csv_path.exists():
            print(f"  [WARN] missing {csv_path} — skipping {value}")
            continue

        df = pd.read_csv(csv_path)
        # y_true / y_prob round-trip through CSV as stringified lists.
        df["y_true"] = df["y_true"].apply(ast.literal_eval)
        df["y_prob"] = df["y_prob"].apply(ast.literal_eval)
        pooled = pooled_metrics_from_folds(df)

        rows.append({
            env_key               : value,
            "n_folds"             : len(df),
            "mean_subject_auc"    : round(df["roc_auc"].mean(), 4),
            "pooled_auc"          : round(pooled.get("roc_auc", float("nan")), 4),
            "mean_balanced_acc"   : round(df["balanced_acc"].mean(), 4),
            "mean_mcc"            : round(df["mcc"].mean(), 4),
            "n_folds_auc_perfect" : int((df["roc_auc"] >= 0.999).sum()),
        })

    if not rows:
        print("  [ERROR] No completed runs found — nothing to summarise.")
        return

    out    = pd.DataFrame(rows)
    outdir = HERE / "output" / "metrics"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"sensitivity_{dim}_summary.csv"
    out.to_csv(out_path, index=False)

    print(f"\n{'='*70}\n  SENSITIVITY SWEEP SUMMARY — {dim}\n{'='*70}")
    print(out.to_string(index=False))
    print(f"\n  Saved: {out_path}")


def main() -> None:
    if os.environ.get("NEUMA_SWEEP_WORKER") == "1":
        _worker()
        return

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", choices=list(GRIDS), required=True,
                     help="threshold (Fix 5) | band (Fix 3) | window (Fix 4)")
    ap.add_argument("--subjects", required=True,
                     help="Comma-separated subject IDs, e.g. S01,S02,S03,S05,S06,S07,S08,S09")
    ap.add_argument("--n-ensemble", type=int, default=3,
                     help="Reduced ensemble size for speed (paper uses 15)")
    ap.add_argument("--epochs", type=int, default=80,
                     help="Reduced epoch budget for speed (paper uses 250)")
    ap.add_argument("--label-prefix", default="sens")
    args = ap.parse_args()

    for value in GRIDS[args.sweep][1]:
        _run_one(args.sweep, value, args.subjects, args.n_ensemble,
                  args.epochs, args.label_prefix)

    _summarize(args.sweep, args.label_prefix)


if __name__ == "__main__":
    main()
