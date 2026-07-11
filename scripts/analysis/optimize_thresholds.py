import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)

# ============================================================
# LOAD PREDICTIONS
# ============================================================

ROOT = Path(
    "/home/nvidia/24PHD1314/Neuma_Model/NEUMA_PHASE8/output"
)

pred_file = ROOT / "predictions.csv"

df = pd.read_csv(pred_file)

print("\n========================================")
print(" THRESHOLD OPTIMIZATION")
print("========================================")

print("\nLoaded:")
print(df.head())

# ============================================================
# EXPECTED COLUMNS
# ============================================================

# y_true
# y_prob

y_true = df["y_true"].values
y_prob = df["y_prob"].values

# ============================================================
# SEARCH THRESHOLDS
# ============================================================

thresholds = np.arange(0.05, 0.96, 0.01)

rows = []

best_kappa = -999
best_row = None

for th in thresholds:

    y_pred = (y_prob >= th).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)

    row = {
        "threshold": th,
        "accuracy": acc,
        "f1": f1,
        "kappa": kap,
        "mcc": mcc,
        "balanced_acc": bal,
    }

    rows.append(row)

    if kap > best_kappa:

        best_kappa = kap
        best_row = row

# ============================================================
# RESULTS
# ============================================================

res = pd.DataFrame(rows)

print("\n========================================")
print(" TOP 10 THRESHOLDS")
print("========================================")

print(
    res.sort_values(
        "kappa",
        ascending=False
    ).head(10)
)

print("\n========================================")
print(" BEST THRESHOLD")
print("========================================")

for k, v in best_row.items():

    print(f"{k:15s}: {v}")

# ============================================================
# SAVE
# ============================================================

save_csv = ROOT / "threshold_optimization.csv"

res.to_csv(save_csv, index=False)

print("\nSaved:")
print(save_csv)
