# Verify every full-model number claimed in the manuscript against the saved per-fold CSV.
import numpy as np, pandas as pd
from pathlib import Path

R = Path(r"D:\INS-HDGS-CMT")
d = pd.read_csv(R / "results/ablation/abl_full/losocv_abl_full.csv").drop_duplicates("test_subject", keep="last")
rng = np.random.default_rng(0)

def m(c):
    return d[c].mean()

def ci(c):
    x = d[c].to_numpy(float)
    b = rng.choice(x, (10000, len(x))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)

claims = [
    ("3.1 accuracy %",            78.24, m("accuracy") * 100, 0.01),
    ("3.1 precision %",           69.91, m("precision") * 100, 0.01),
    ("3.1 recall %",              77.18, m("recall") * 100, 0.01),
    ("3.1 F1 %",                  69.06, m("f1") * 100, 0.01),
    ("3.1 AUC",                   0.88,  m("roc_auc"), 0.005),
    ("3.1 BalAcc %",              74.75, m("balanced_acc") * 100, 0.01),
    ("3.1 MCC",                   0.49,  m("mcc"), 0.005),
    ("3.1 accuracy_cal %",        76.66, m("accuracy_cal") * 100, 0.01),
    ("3.1 BalAcc_cal %",          71.92, m("balanced_acc_cal") * 100, 0.01),
    ("3.1 MCC_cal",               0.45,  m("mcc_cal"), 0.005),
    ("3.3 kappa",                 0.45,  m("kappa"), 0.005),
    ("3.3 kappa_cal",             0.41,  m("kappa_cal"), 0.005),
    ("Table 6 BalAcc",            0.75,  m("balanced_acc"), 0.005),
    ("Table 6 MCC",               0.49,  m("mcc"), 0.005),
    ("Table 6 ROC-AUC",           0.88,  m("roc_auc"), 0.005),
    ("Table 6 PR-AUC",            0.90,  m("pr_auc"), 0.005),
    ("Table 7 caption BalAcc",    0.747, m("balanced_acc"), 0.0005),
    ("Table 7 caption MCC",       0.485, m("mcc"), 0.0005),
]
bad = 0
for name, claimed, actual, tol in claims:
    ok = abs(claimed - actual) <= tol
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {name:26s} manuscript {claimed:>7} vs data {actual:.4f}")
for name, claimed, col in [("AUC CI", (0.82, 0.93), "roc_auc"),
                           ("BalAcc CI", (0.69, 0.81), "balanced_acc"),
                           ("MCC CI", (0.37, 0.60), "mcc")]:
    lo, hi = ci(col)
    ok = abs(claimed[0] - lo) <= 0.006 and abs(claimed[1] - hi) <= 0.006
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} 3.1 {name:22s} manuscript {claimed} vs data [{lo:.3f}, {hi:.3f}]")
print(f"\n{bad} discrepancies")
