import pandas as pd
import numpy as np

from pathlib import Path

# ============================================================
# LOAD RESULTS
# ============================================================

CSV_PATH = Path(
    "output/metrics/losocv_ins_hdgs_cmt.csv"
)

df = pd.read_csv(CSV_PATH)

# ============================================================
# SORT SUBJECTS
# ============================================================

df_sorted = df.sort_values(
    "mcc"
)

print("\n" + "="*70)
print(" SUBJECT FAILURE ANALYSIS")
print("="*70)

cols = [
    "test_subject",
    "accuracy",
    "f1",
    "kappa",
    "mcc",
    "balanced_acc",
    "precision",
    "recall",
]

print(
    df_sorted[cols]
    .to_string(index=False)
)

# ============================================================
# COLLAPSED SUBJECTS
# ============================================================

collapsed = df_sorted[
    df_sorted["mcc"] <= 0.0
]

print("\n" + "="*70)
print(" COLLAPSED SUBJECTS")
print("="*70)

if len(collapsed) == 0:

    print("\nNone.")

else:

    print(
        collapsed[
            [
                "test_subject",
                "accuracy",
                "f1",
                "mcc",
                "balanced_acc"
            ]
        ].to_string(index=False)
    )

# ============================================================
# STRONG SUBJECTS
# ============================================================

strong = df_sorted[
    df_sorted["mcc"] >= 0.70
]

print("\n" + "="*70)
print(" STRONG GENERALIZATION SUBJECTS")
print("="*70)

if len(strong) == 0:

    print("\nNone.")

else:

    print(
        strong[
            [
                "test_subject",
                "accuracy",
                "f1",
                "mcc",
                "balanced_acc"
            ]
        ].to_string(index=False)
    )

# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n" + "="*70)
print(" SUMMARY")
print("="*70)

print(f"\nTotal subjects : {len(df_sorted)}")

print(
    f"Collapsed      : "
    f"{len(collapsed)}"
)

print(
    f"Strong MCC>0.7 : "
    f"{len(strong)}"
)

print(
    f"\nMean MCC       : "
    f"{df_sorted['mcc'].mean():.4f}"
)

print(
    f"Std MCC        : "
    f"{df_sorted['mcc'].std():.4f}"
)

print(
    f"\nMean Kappa     : "
    f"{df_sorted['kappa'].mean():.4f}"
)

print(
    f"Mean Accuracy  : "
    f"{df_sorted['accuracy'].mean():.4f}"
)

# ============================================================
# SAVE
# ============================================================

OUT = (
    "output/metrics/"
    "subject_failure_analysis.csv"
)

df_sorted.to_csv(
    OUT,
    index=False
)

print("\nSaved:")
print(OUT)
