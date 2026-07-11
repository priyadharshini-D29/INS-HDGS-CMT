#!/usr/bin/env python3
"""
Recompute calibration (ECE) using the corrected, pooled definition.

Why this exists
---------------
The original per-fold ECE was (a) computed with a buggy formula that compared
mean P(class 1) against *thresholded accuracy* instead of the empirical positive
frequency, and (b) averaged over ~7-sample folds where ECE is pure noise.

This script reuses the now-fixed training.metrics.compute_ece and pools all
fold-level predictions per model/variant before computing a single ECE. No
retraining is needed — every prediction is already saved on disk.

Outputs results/calibration/ece_corrected.csv
"""
import sys, ast, re, glob, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.metrics import compute_ece

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def parse_arr(s):
    s = re.sub(r"\s+", ",", str(s).strip())
    s = re.sub(r",+", ",", s).replace("[,", "[").replace(",]", "]")
    return np.asarray(ast.literal_eval(s), dtype=float)


def old_ece(y, p, nb=10):
    """The previous (buggy) per-fold definition, for before/after reporting."""
    bins = np.linspace(0, 1, nb + 1); e = 0.0; n = len(y)
    for i in range(nb):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(p[m].mean() - (y[m] == (p[m] >= .5)).mean())
    return e


rows = []

def record(group, name, y, p, reported_mean=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rows.append(dict(
        group=group, model=name, n=len(y),
        ece_reported_perfold_mean=reported_mean,
        ece_old_pooled=round(old_ece(y, p), 4),
        ece_corrected_pooled=round(compute_ece(y, p), 4),
    ))

# ── Main model (predictions stored inline as y_true / y_prob arrays) ──────────
for f in sorted(glob.glob(f"{R}/losocv_metrics/losocv_*.csv")):
    df = pd.read_csv(f)
    Y = np.concatenate([parse_arr(r) for r in df["y_true"]])
    P = np.concatenate([parse_arr(r) for r in df["y_prob"]])
    rep = df["ece"].mean() if "ece" in df else None
    record("main", os.path.basename(f).replace("losocv_", "").replace(".csv", ""), Y, P, rep)

# ── Ablation variants (y_true / y_prob arrays per fold) ───────────────────────
for f in sorted(glob.glob(f"{R}/ablation/*/losocv_*.csv")):
    df = pd.read_csv(f)
    if "y_true" not in df or "y_prob" not in df:
        continue
    Y = np.concatenate([parse_arr(r) for r in df["y_true"]])
    P = np.concatenate([parse_arr(r) for r in df["y_prob"]])
    rep = df["ece"].mean() if "ece" in df else None
    record("ablation", os.path.basename(os.path.dirname(f)), Y, P, rep)

# ── DL baselines (one row per sample in fold_probs/probs_<model>.csv) ─────────
for f in sorted(glob.glob(f"{R}/baselines/dl/fold_probs/probs_*.csv")):
    df = pd.read_csv(f)
    name = os.path.basename(f).replace("probs_", "").replace(".csv", "")
    record("baseline", name, df["y_true"].values, df["p1"].values)

out = pd.DataFrame(rows)
os.makedirs(f"{R}/calibration", exist_ok=True)
dst = f"{R}/calibration/ece_corrected.csv"
out.to_csv(dst, index=False)

pd.set_option("display.width", 200)
for g in ["main", "ablation", "baseline"]:
    sub = out[out.group == g].sort_values("ece_corrected_pooled")
    if len(sub):
        print(f"\n===== {g.upper()} =====")
        print(sub.drop(columns="group").to_string(index=False))
print(f"\nWrote {dst}")
