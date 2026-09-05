"""
================================================================
Label-free, prospective per-subject decision thresholds
(Reviewer 2, comment 8)
================================================================
Section 3.7 of the manuscript reports a per-subject "prevalence-matched"
threshold. This script makes explicit that NO test-subject labels are
used, and evaluates the strategies in the strictly *online* setting a
deployed system would face: the threshold for epoch k is computed only
from the unlabelled predicted probabilities of epochs 1..k-1 (plus the
training-set class prior), then applied to epoch k. Epochs before the
warm-up window fall back to the transferred validation-subject
threshold.

Strategies
----------
  fixed_0.5            : constant cut-off
  val_subject          : Youden-J threshold transferred from the held-out
                         validation subject (the paper's calibrated
                         operating point; never sees the test subject)
  train_prior_quantile : label-free, TRANSDUCTIVE — quantile of the test
                         subject's own P(HIGH) distribution at the training
                         class prior (uses all of the subject's unlabelled
                         probabilities)
  online_quantile[k]   : label-free, CAUSAL — same quantile rule but
                         estimated only from the first k unlabelled epochs
                         (then updated as epochs arrive)
  online_median[k]     : causal running median of P(HIGH)

For each strategy: balanced accuracy, MCC, F1, accuracy (mean ± SD across
the 37 folds), paired Wilcoxon vs. fixed_0.5 and vs. val_subject.

Inputs : the per-fold CSV written by evaluation/losocv.py (y_true,
         y_prob as stringified lists, opt_threshold_cal, train_n).
Outputs: results/threshold_analysis/deployment_threshold_{summary,per_fold}.csv
         results/threshold_analysis/DEPLOYMENT_THRESHOLD.md
================================================================
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "results" / "losocv_metrics" / "losocv_repro_focal_g3p0_effective_num_37.csv"
OUT = ROOT / "results" / "threshold_analysis"


def _parse(v):
    return np.asarray(ast.literal_eval(str(v)), dtype=float)


def _metrics(y, pred):
    return dict(balanced_acc=balanced_accuracy_score(y, pred),
                mcc=matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
                f1=f1_score(y, pred, zero_division=0),
                accuracy=accuracy_score(y, pred))


def _quantile_threshold(p_hist: np.ndarray, prior_high: float) -> float:
    """Threshold such that a fraction prior_high of the *unlabelled* history is
    predicted HIGH (prevalence matching). Uses probabilities only."""
    if len(p_hist) == 0:
        return 0.5
    q = float(np.clip(1.0 - prior_high, 0.0, 1.0))
    return float(np.quantile(p_hist, q))


def run(csv: Path, warmups=(2, 3, 5), prior_high: float | None = None):
    df = pd.read_csv(csv)
    per_fold = []
    for _, r in df.iterrows():
        y = _parse(r["y_true"]).astype(int)
        p = _parse(r["y_prob"])
        if len(np.unique(y)) < 2:
            continue
        thr_val = float(r.get("opt_threshold_cal", r.get("opt_threshold", 0.5)))
        # training-set class prior. The CSV does not store it per fold; the
        # pooled prior is 193/385 = 0.501 (Table 1). --prior-high overrides.
        pr = 0.501 if prior_high is None else prior_high

        strategies = {}
        strategies["fixed_0.5"] = (p >= 0.5).astype(int)
        strategies["val_subject"] = (p >= thr_val).astype(int)
        strategies["train_prior_quantile"] = (p >= _quantile_threshold(p, pr)).astype(int)
        for k in warmups:
            pred_q = np.zeros(len(p), int)
            pred_m = np.zeros(len(p), int)
            for i in range(len(p)):
                hist = p[:i]                     # strictly causal: epochs before i
                if len(hist) < k:
                    t_q = thr_val                # warm-up: transferred val threshold
                    t_m = thr_val
                else:
                    t_q = _quantile_threshold(hist, pr)
                    t_m = float(np.median(hist))
                pred_q[i] = int(p[i] >= t_q)
                pred_m[i] = int(p[i] >= t_m)
            strategies[f"online_quantile[k={k}]"] = pred_q
            strategies[f"online_median[k={k}]"] = pred_m

        for name, pred in strategies.items():
            m = _metrics(y, pred)
            m.update(strategy=name, test_subject=r["test_subject"], n=len(y))
            per_fold.append(m)

    pf = pd.DataFrame(per_fold)
    order = list(dict.fromkeys(pf["strategy"]))
    summ = []
    base_fixed = pf[pf.strategy == "fixed_0.5"].set_index("test_subject")
    base_val = pf[pf.strategy == "val_subject"].set_index("test_subject")
    for s in order:
        g = pf[pf.strategy == s].set_index("test_subject")
        row = dict(strategy=s, n_folds=len(g))
        for m in ["balanced_acc", "mcc", "f1", "accuracy"]:
            row[f"{m}_mean"] = g[m].mean(); row[f"{m}_sd"] = g[m].std()
        for ref_name, ref in [("fixed", base_fixed), ("val", base_val)]:
            d = g["balanced_acc"].loc[ref.index].values - ref["balanced_acc"].values
            row[f"delta_balacc_vs_{ref_name}"] = d.mean()
            try:
                row[f"p_vs_{ref_name}"] = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                row[f"p_vs_{ref_name}"] = 1.0
        summ.append(row)
    return pf, pd.DataFrame(summ)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--prior-high", type=float, default=None,
                    help="training-set P(HIGH) prior (default: pooled 193/385)")
    ap.add_argument("--warmups", default="2,3,5")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    warmups = tuple(int(x) for x in args.warmups.split(","))
    pf, summ = run(Path(args.csv), warmups, args.prior_high)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pf.to_csv(out / "deployment_threshold_per_fold.csv", index=False)
    summ.to_csv(out / "deployment_threshold_summary.csv", index=False)

    lines = ["# Label-free prospective thresholds (no test labels used at any point)", "",
             f"Source: `{Path(args.csv).name}` — {summ.n_folds.iloc[0]} evaluable folds.", "",
             "| strategy | uses test labels? | causal (online)? | BalAcc | MCC | F1 | Acc | Δ BalAcc vs fixed (p) | Δ BalAcc vs val-subject (p) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        s = r.strategy
        causal = "yes" if s.startswith("online") else ("n/a" if s in ("fixed_0.5", "val_subject") else "no (transductive)")
        lines.append(f"| {s} | no | {causal} | {r.balanced_acc_mean:.3f} ± {r.balanced_acc_sd:.3f} | "
                     f"{r.mcc_mean:.3f} ± {r.mcc_sd:.3f} | {r.f1_mean:.3f} | {r.accuracy_mean:.3f} | "
                     f"{r.delta_balacc_vs_fixed:+.3f} ({r.p_vs_fixed:.3f}) | {r.delta_balacc_vs_val:+.3f} ({r.p_vs_val:.3f}) |")
    lines += ["", "Every threshold above is computed from (i) the training pool, (ii) the held-out validation "
              "subject, or (iii) the test subject's own *unlabelled* predicted probabilities. The online "
              "variants use only epochs that precede the one being classified, so they can be applied "
              "prospectively; epochs inside the warm-up window use the transferred validation-subject threshold."]
    (out / "DEPLOYMENT_THRESHOLD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
