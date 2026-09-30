# Verify the manuscript's headline full-model numbers (Section 3.1) against the
# released per-fold CSV. The reported operating point is the fixed 0.5 threshold
# on the saved held-out probability (temperature-scaled ensemble mean, Section 2.7),
# stored in the *_cal columns; roc_auc is threshold-invariant.
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[2]
d = pd.read_csv(R / "results/ablation/abl_full/losocv_abl_full.csv").drop_duplicates(
    "test_subject", keep="last")

claims = [
    ("3.1 accuracy",          0.7666, "accuracy_cal"),
    ("3.1 balanced accuracy", 0.7192, "balanced_acc_cal"),
    ("3.1 MCC",               0.4458, "mcc_cal"),
    ("3.1 ROC-AUC",           0.8790, "roc_auc"),
]
bad = 0
for name, claimed, col in claims:
    got = d[col].mean()
    ok = abs(got - claimed) < 5e-4
    bad += (not ok)
    print(f"{name:24s} claimed={claimed:.4f}  csv={got:.6f}  {'OK' if ok else 'MISMATCH'}")
raise SystemExit(bad)
