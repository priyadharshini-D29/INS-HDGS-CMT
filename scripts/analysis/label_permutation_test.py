#!/usr/bin/env python3
"""
Within-fold label permutation test on SAVED held-out predictions.

What this tests
---------------
Given a per-fold LOSOCV CSV whose rows carry the held-out labels (``y_true``)
and predicted P(HIGH) (``y_prob``) of one test subject each, the observed
association between predictions and labels is compared with a null
distribution obtained by shuffling the labels WITHIN each fold (subject) while
the predictions are left untouched.  Shuffling within the fold preserves each
subject's class balance and epoch count, so the null respects the LOSOCV
structure.  Three statistics are recomputed for every permutation:

  * pooled ROC-AUC over all held-out epochs concatenated across folds,
  * mean per-fold balanced accuracy at the fold's own operating point
    (``y_prob >= opt_threshold`` when the column exists, else 0.5 — the same
    convention as evaluation/losocv.py, which reproduces the stored per-fold
    ``balanced_acc`` and ``mcc`` in every fold),
  * mean per-fold MCC at the same operating point,

together with the mean per-fold ROC-AUC as a fourth, descriptive statistic.
The permutation p-value is ((count of null >= observed) + 1) / (N + 1)
(Phipson & Smyth 2010); the null mean and SD are reported as well.

What this does NOT test
-----------------------
This is a test of the association between the ALREADY-TRAINED models'
held-out predictions and the true labels.  It is NOT a test that the training
procedure cannot fit random labels: that would require retraining the model
under shuffled training labels (the ``label_shuffle`` control run by
evaluation/losocv.py), which this script never does.  Nothing is trained
here; the script only reads saved CSVs and takes seconds of CPU.

Usage
-----
  python scripts/analysis/label_permutation_test.py \
      --csv results/ablation/abl_full/losocv_abl_full.csv --label abl_full
  python scripts/analysis/label_permutation_test.py \
      --csv results/losocv_metrics/losocv_repro_focal_g3p0_effective_num_37.csv \
      --label published

Outputs results/statistics/label_permutation_<label>.md and .json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "statistics"


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_arr(s) -> np.ndarray:
    """Parse a stringified list (``"[0, 1, 1]"`` or numpy-style ``"[0 1 1]"``)."""
    s = re.sub(r"\s+", ",", str(s).strip())
    s = re.sub(r",+", ",", s).replace("[,", "[").replace(",]", "]")
    return np.asarray(ast.literal_eval(s), dtype=float)


def auc_from_ranks(ranks: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney identity; ``ranks`` are average ranks of
    the scores (ties handled), ``y`` the binary labels. NaN if one class."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bal_acc_mcc(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Balanced accuracy and MCC from counts (identical to sklearn for binary
    labels; MCC is 0 when its denominator vanishes, as in sklearn)."""
    tp = float(np.sum((y == 1) & (pred == 1)))
    tn = float(np.sum((y == 0) & (pred == 0)))
    fp = float(np.sum((y == 0) & (pred == 1)))
    fn = float(np.sum((y == 1) & (pred == 0)))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    if np.isnan(tpr) or np.isnan(tnr):        # single-class fold (never in the 37-fold files)
        bal = np.nanmean([tpr, tnr])
    else:
        bal = 0.5 * (tpr + tnr)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return float(bal), float(mcc)


def load_folds(csv: Path) -> list[dict]:
    df = pd.read_csv(csv)
    folds = []
    for _, r in df.iterrows():
        y = parse_arr(r["y_true"]).astype(int)
        p = parse_arr(r["y_prob"])
        thr = float(r["opt_threshold"]) if "opt_threshold" in df.columns and pd.notna(r.get("opt_threshold")) else 0.5
        folds.append(dict(subject=str(r["test_subject"]), y=y, p=p, thr=thr,
                          pred=(p >= thr).astype(int), ranks=rankdata(p),
                          stored_bal=float(r["balanced_acc"]) if "balanced_acc" in df.columns else float("nan"),
                          stored_mcc=float(r["mcc"]) if "mcc" in df.columns else float("nan")))
    return folds


def statistics(folds: list[dict], ys: list[np.ndarray], pooled_ranks: np.ndarray) -> dict:
    """All four statistics for one labelling ``ys`` (list of per-fold label arrays)."""
    bal, mcc, auc_f = [], [], []
    for f, y in zip(folds, ys):
        b, m = bal_acc_mcc(y, f["pred"])
        bal.append(b); mcc.append(m)
        auc_f.append(auc_from_ranks(f["ranks"], y))
    y_all = np.concatenate(ys)
    return dict(pooled_roc_auc=auc_from_ranks(pooled_ranks, y_all),
                mean_fold_balanced_acc=float(np.nanmean(bal)),
                mean_fold_mcc=float(np.nanmean(mcc)),
                mean_fold_roc_auc=float(np.nanmean(auc_f)))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="per-fold LOSOCV CSV with y_true / y_prob list columns")
    ap.add_argument("--label", required=True, help="output label (results/statistics/label_permutation_<label>.*)")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    folds = load_folds(Path(args.csv))
    pooled_ranks = rankdata(np.concatenate([f["p"] for f in folds]))
    n_epochs = int(sum(len(f["y"]) for f in folds))

    # sanity: the stored per-fold metrics are reproduced at the chosen operating point
    max_dev = max(max(abs(bal_acc_mcc(f["y"], f["pred"])[0] - f["stored_bal"]),
                      abs(bal_acc_mcc(f["y"], f["pred"])[1] - f["stored_mcc"])) for f in folds)
    observed = statistics(folds, [f["y"] for f in folds], pooled_ranks)

    rng = np.random.default_rng(args.seed)
    keys = list(observed)
    null = {k: np.empty(args.n_perm) for k in keys}
    for i in range(args.n_perm):
        ys = [rng.permutation(f["y"]) for f in folds]
        st = statistics(folds, ys, pooled_ranks)
        for k in keys:
            null[k][i] = st[k]

    res = {}
    for k in keys:
        nd = null[k]
        res[k] = dict(observed=observed[k],
                      p_value=float((np.sum(nd >= observed[k]) + 1) / (args.n_perm + 1)),
                      null_mean=float(np.nanmean(nd)), null_sd=float(np.nanstd(nd, ddof=1)),
                      null_max=float(np.nanmax(nd)))

    thr_note = ("fold-specific opt_threshold" if any(f["thr"] != 0.5 for f in folds) else "0.5")
    labels = {"pooled_roc_auc": "Pooled ROC-AUC (all held-out epochs)",
              "mean_fold_balanced_acc": f"Mean per-fold balanced accuracy ({thr_note})",
              "mean_fold_mcc": f"Mean per-fold MCC ({thr_note})",
              "mean_fold_roc_auc": "Mean per-fold ROC-AUC (descriptive)"}
    lines = [f"# Within-fold label permutation test: {args.label}", "",
             f"Source: `{Path(args.csv).as_posix()}` ({len(folds)} folds, {n_epochs} held-out epochs). "
             f"Labels shuffled within each fold, predictions fixed; N = {args.n_perm} permutations, seed = {args.seed}. "
             f"p = ((count null >= observed) + 1) / (N + 1); the smallest attainable p is {1/(args.n_perm+1):.2e}.", "",
             "This tests the association between the saved held-out predictions and the labels. "
             "It is NOT a test that the training procedure cannot fit random labels (that requires retraining "
             "under shuffled training labels, which this script does not do).", "",
             f"Stored per-fold balanced accuracy / MCC reproduced at the operating point (max |deviation| = {max_dev:.2e}).", "",
             "| statistic | observed | permutation p | null mean | null SD | null max |",
             "|---|---|---|---|---|---|"]
    for k in keys:
        r = res[k]
        lines.append(f"| {labels[k]} | {r['observed']:.4f} | {r['p_value']:.2e} | {r['null_mean']:.4f} | "
                     f"{r['null_sd']:.4f} | {r['null_max']:.4f} |")
    md = "\n".join(lines) + "\n"
    print(md)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"label_permutation_{args.label}.md").write_text(md, encoding="utf-8")
    payload = dict(label=args.label, csv=Path(args.csv).as_posix(), n_folds=len(folds), n_epochs=n_epochs,
                   n_perm=args.n_perm, seed=args.seed, operating_point=thr_note,
                   max_abs_deviation_from_stored_metrics=float(max_dev), results=res,
                   note="Association test on saved held-out predictions; not a random-label training control.")
    (out_dir / f"label_permutation_{args.label}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / f'label_permutation_{args.label}.md'} and .json")


if __name__ == "__main__":
    main()
