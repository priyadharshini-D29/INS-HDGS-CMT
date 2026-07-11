import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
)

# ============================================================
# LOAD CSV
# ============================================================

CSV_PATH = Path(
    "output/metrics/losocv_ins_hdgs_cmt.csv"
)

df = pd.read_csv(CSV_PATH)

print("\nLoaded:")
print(CSV_PATH)

# ============================================================
# CHECK COLUMNS
# ============================================================

required = [
    "test_subject",
    "y_true",
    "y_prob",
]

for c in required:

    if c not in df.columns:

        raise ValueError(
            f"Missing column: {c}"
        )

# ============================================================
# ARRAY PARSER
# ============================================================

def parse_array(x):

    if isinstance(x, str):

        x = x.strip("[]")

        if len(x) == 0:
            return []

        return [
            float(v)
            for v in x.split(",")
        ]

    return x

# ============================================================
# OPTIMIZE PER SUBJECT
# ============================================================

rows = []

for _, row in df.iterrows():

    subject = row["test_subject"]

    y_true = np.array(
        parse_array(row["y_true"])
    ).astype(int)

    y_prob = np.array(
        parse_array(row["y_prob"])
    )

    best = None

    for t in np.arange(0.05, 0.96, 0.01):

        pred = (
            y_prob >= t
        ).astype(int)

        try:

            f1 = f1_score(
                y_true,
                pred,
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y_true,
                pred
            )

            kappa = cohen_kappa_score(
                y_true,
                pred
            )

            bal = balanced_accuracy_score(
                y_true,
                pred
            )

        except:

            continue

        score = (
            0.4 * mcc +
            0.3 * kappa +
            0.3 * bal
        )

        if best is None or score > best["score"]:

            best = {

                "subject": subject,
                "threshold": t,
                "f1": f1,
                "mcc": mcc,
                "kappa": kappa,
                "balanced_acc": bal,
                "score": score,
            }

    rows.append(best)

# ============================================================
# RESULTS
# ============================================================

res_df = pd.DataFrame(rows)

print("\n" + "="*70)
print(" OPTIMIZED THRESHOLDS")
print("="*70)

print(
    res_df.sort_values("mcc")
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*70)
print(" SUMMARY")
print("="*70)

print(
    f"\nMean MCC       : "
    f"{res_df['mcc'].mean():.4f}"
)

print(
    f"Mean Kappa     : "
    f"{res_df['kappa'].mean():.4f}"
)

print(
    f"Mean F1        : "
    f"{res_df['f1'].mean():.4f}"
)

print(
    f"Mean Bal Acc   : "
    f"{res_df['balanced_acc'].mean():.4f}"
)

# ============================================================
# SAVE
# ============================================================

OUT = (
    "output/metrics/"
    "optimized_thresholds.csv"
)

res_df.to_csv(
    OUT,
    index=False
)

print("\nSaved:")
print(OUT)
