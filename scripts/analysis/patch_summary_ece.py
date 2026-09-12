#!/usr/bin/env python3
"""
Patch the baseline summary_*.csv files in place with corrected, pooled ECE.

  ece     : reliability-diagram ECE on pooled raw probabilities (fold_probs)
  ece_cal : same, after applying each fold's saved temperature T_post

Uses the fixed training.metrics.compute_ece. No retraining.
"""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.metrics import compute_ece

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "baselines", "dl")


def temp_scale(p, T):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) / np.maximum(T, 1e-6)
    return 1.0 / (1.0 + np.exp(-z))


# Build per-model corrected raw + calibrated pooled ECE
ece_raw, ece_cal = {}, {}
for pf in glob.glob(f"{R}/fold_probs/probs_*.csv"):
    name = os.path.basename(pf).replace("probs_", "").replace(".csv", "")
    dp = pd.read_csv(pf)
    ece_raw[name] = round(compute_ece(dp["y_true"].values, dp["p1"].values), 4)
    lf = f"{R}/losocv_{name}.csv"
    if os.path.exists(lf):
        dl = pd.read_csv(lf)[["fold", "T_post"]].drop_duplicates("fold")
        m = dp.merge(dl, on="fold", how="left")
        m["T_post"] = m["T_post"].fillna(1.0)
        pcal = temp_scale(m["p1"].values, m["T_post"].values)
        ece_cal[name] = round(compute_ece(m["y_true"].values, pcal), 4)

# Apply to every summary_*.csv
for sf in glob.glob(f"{R}/summary_*.csv"):
    df = pd.read_csv(sf)
    if "model" not in df.columns:
        continue
    changed = False
    for i, mod in df["model"].items():
        if mod in ece_raw and "ece" in df.columns:
            df.at[i, "ece"] = ece_raw[mod]; changed = True
        if mod in ece_cal and "ece_cal" in df.columns:
            df.at[i, "ece_cal"] = ece_cal[mod]; changed = True
    if changed:
        df.to_csv(sf, index=False)
        print(f"patched {os.path.basename(sf):55s} "
              + ", ".join(f"{m}:{ece_raw.get(m,'-')}/{ece_cal.get(m,'-')}" for m in df['model']))

print("\n(ece / ece_cal shown per model; '-' = no probs file)")
