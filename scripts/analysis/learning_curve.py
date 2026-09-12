"""
================================================================
Learning curve vs. number of training subjects (Reviewer 2, comment 6)
================================================================
Addresses the sample-size / overfitting concern (385 epochs, 37 evaluable
subjects) with a subject-count learning curve: LOSOCV is repeated on
random subject subsets of increasing size, with the SAME model, the same
early stopping on a held-out validation subject and the same calibration.
If performance keeps rising with subject count the model is data-limited
(not saturated/overfit to idiosyncrasies of the cohort); if it plateaus
early, extra subjects would not help. Either way the slope at n = 37
lets the reader extrapolate to larger cohorts.

Each subset of size n is drawn --repeats times (different seeds), and
LOSOCV runs over the subset (n test folds, n-2 training subjects per
fold). The per-fold CSVs are written by evaluation/losocv.py as usual
under output/metrics/lc_n<n>_r<r>/.

Also reported (from the per-fold CSVs): the mean train-set size, so the
x-axis can be read either as subjects or as training epochs.

Usage (GPU server)
------------------
  cd src/model
  python ../../scripts/analysis/learning_curve.py --sizes 10,16,24,30,37 --repeats 3 \
      --n-ensemble 3 --epochs 150 [--eeg-only]
  python ../../scripts/analysis/learning_curve.py --summarise-only

Outputs
-------
  results/sensitivity/learning_curve.csv / learning_curve.md / fig_learning_curve.pdf
================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONUTF8", "1")
# Pin the eye-tracking input to the production 3-channel configuration BEFORE
# config.settings is imported (mirrors scripts/analysis/run_component_ablation.py).
os.environ.setdefault("ET_USE_BOTH_EYES", "0")
os.environ.setdefault("ET_USE_VERGENCE",  "0")
os.environ.setdefault("ET_USE_SPEED",     "0")
os.environ.setdefault("ET_NORMALIZE",     "0")
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "src" / "model"
for p in (str(MODEL), str(MODEL.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = ROOT / "results" / "sensitivity"


def summarise(sizes, repeats, metrics_dir: Path):
    rows = []
    for n in sizes:
        for r in range(repeats):
            tag = f"lc_n{n}_r{r}"
            csv = metrics_dir / tag / f"losocv_{tag}.csv"
            if not csv.exists():
                continue
            d = pd.read_csv(csv)
            rows.append(dict(n_subjects=n, repeat=r, n_folds=len(d), train_n_mean=d["train_n"].mean(),
                             balanced_acc=d["balanced_acc"].mean(), roc_auc=d["roc_auc"].mean(), mcc=d["mcc"].mean(),
                             balanced_acc_cal=d.get("balanced_acc_cal", d["balanced_acc"]).mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="10,16,24,30,37")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-ensemble", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    # production-pinned loss / regularisation (same as run_component_ablation.py)
    ap.add_argument("--focal-gamma", type=float, default=3.0)
    ap.add_argument("--alpha-strategy", type=str, default="effective_num")
    ap.add_argument("--lambda-dann", type=float, default=0.10)
    ap.add_argument("--lambda-mmd", type=float, default=0.10)
    ap.add_argument("--eeg-only", action="store_true", help="use AblationConfig.eeg_only() (leakage-free branch)")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    from config.settings import METRICS_DIR, SUBJECT_IDS
    if not args.summarise_only:
        from data.dataset import NeumaGraphDataset
        from evaluation.losocv import run_losocv
        from models.ins_hdgs_cmt import AblationConfig
        # evaluable pool = subjects with data and both classes (37 under global labels)
        pool = []
        for s in SUBJECT_IDS:
            try:
                d = NeumaGraphDataset(subject_ids=[s], precompute_graphs=False)
                if len(d) and len(np.unique(d.labels)) > 1:
                    pool.append(s)
            except FileNotFoundError:
                pass
        rng = np.random.default_rng(args.seed)
        for n in sizes:
            for r in range(args.repeats):
                sub = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
                tag = f"lc_n{n}_r{r}"
                if (METRICS_DIR / tag / f"losocv_{tag}.csv").exists():
                    print(f"[lc] {tag} exists — skip"); continue
                print(f"[lc] {tag}: {sub}")
                run_losocv(subject_ids=sub, ablation=AblationConfig.eeg_only() if args.eeg_only else AblationConfig.full(),
                           epochs=args.epochs, label=tag, n_ensemble_override=args.n_ensemble,
                           alpha_strategy=args.alpha_strategy, focal_gamma_override=args.focal_gamma,
                           lambda_dann_override=args.lambda_dann, lambda_mmd_override=args.lambda_mmd,
                           fold_parallel=True, verbose=False, random_seed=args.seed + r)
    df = summarise(sizes, args.repeats, METRICS_DIR)
    if df.empty:
        sys.exit("no learning-curve runs found")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "learning_curve.csv", index=False)
    g = df.groupby("n_subjects").agg(["mean", "std"])
    lines = ["# Learning curve vs. number of subjects (LOSOCV within random subject subsets)", "",
             "| subjects | training epochs / fold | BalAcc | ROC-AUC | MCC | repeats |", "|---|---|---|---|---|---|"]
    for n, r in g.iterrows():
        lines.append(f"| {n} | {r[('train_n_mean','mean')]:.0f} | {r[('balanced_acc','mean')]:.3f} ± {r[('balanced_acc','std')]:.3f} | "
                     f"{r[('roc_auc','mean')]:.3f} ± {r[('roc_auc','std')]:.3f} | {r[('mcc','mean')]:.3f} ± {r[('mcc','std')]:.3f} | "
                     f"{int(df[df.n_subjects==n].shape[0])} |")
    (out / "learning_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for m, lab in [("roc_auc", "ROC-AUC"), ("balanced_acc", "balanced accuracy"), ("mcc", "MCC")]:
            ax.errorbar(g.index, g[(m, "mean")], yerr=g[(m, "std")], marker="o", capsize=3, label=lab)
        ax.set_xlabel("number of subjects in LOSOCV pool"); ax.set_ylabel("held-out metric (mean over folds)")
        ax.legend(fontsize=8); ax.set_title("Learning curve (subject count)", fontweight="bold")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_learning_curve.{ext}", dpi=300, bbox_inches="tight")
    except Exception as e:  # pragma: no cover
        print(f"[lc] figure skipped: {e}")


if __name__ == "__main__":
    main()
