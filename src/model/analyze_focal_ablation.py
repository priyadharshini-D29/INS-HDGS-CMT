"""
Focal-loss ablation analysis for INS-HDGS-CMT.

Loads every ablation run CSV, applies Global Youden thresholding,
computes metrics for the 5 hard subjects specifically, and produces:

  output/analysis/focal_loss_ablation.csv
  output/analysis/focal_loss_hard_subjects.csv

Usage:
    python analyze_focal_ablation.py
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.special import softmax as sp_softmax
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    roc_curve,
)

METRICS_DIR  = Path("output/metrics")
OUT_ANALYSIS = Path("output/analysis")
OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)

HARD_SUBJECTS = ["S03", "S13", "S21", "S35", "S36"]

GAMMAS = [0.0, 1.0, 1.5, 2.0, 3.0]
ALPHAS = ["balanced", "effective_num", "sqrt_inv_freq"]

def label_for(gamma, alpha):
    g_str = f"g{gamma:.1f}".replace(".", "p")
    return f"focal_abl_{g_str}_{alpha}"

# ── Helpers ────────────────────────────────────────────────────────────────────

def ece_score(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    val  = 0.0
    n    = len(y_true)
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if m.sum() == 0:
            continue
        conf = y_prob[m].mean()
        acc  = (y_true[m] == (y_prob[m] >= 0.5).astype(int)).mean()
        val += (m.sum() / n) * abs(conf - acc)
    return float(val)


def metrics_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    if len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in
                ["accuracy","balanced_acc","f1","mcc","roc_auc","pr_auc","ece","recall_minority"]}
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    try:
        pr = average_precision_score(y_true, y_prob)
    except Exception:
        pr = float("nan")
    # Minority recall: recall for the smaller class
    n1 = y_true.sum(); n0 = len(y_true) - n1
    minority_cls = 1 if n1 <= n0 else 0
    tp_min = ((y_pred == minority_cls) & (y_true == minority_cls)).sum()
    fn_min = ((y_pred != minority_cls) & (y_true == minority_cls)).sum()
    rec_min = tp_min / max(tp_min + fn_min, 1)
    return dict(
        accuracy       = accuracy_score(y_true, y_pred),
        balanced_acc   = balanced_accuracy_score(y_true, y_pred),
        f1             = f1_score(y_true, y_pred, zero_division=0),
        mcc            = matthews_corrcoef(y_true, y_pred),
        roc_auc        = auc,
        pr_auc         = pr,
        ece            = ece_score(y_true, y_prob),
        recall_minority= float(rec_min),
    )


def global_youden_threshold(probs_pool, labels_pool):
    """Youden's J on pooled predictions, clamped [0.05, 0.95]."""
    if len(np.unique(labels_pool)) < 2:
        return 0.5
    fpr, tpr, thrs = roc_curve(labels_pool, probs_pool)
    idx = int(np.argmax(tpr - fpr))
    return float(np.clip(thrs[idx], 0.05, 0.95))


def load_run(csv_path: Path):
    """Load a run CSV; return DataFrame or None."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "y_prob" not in df.columns or "y_true" not in df.columns:
        return None
    # Parse serialised list columns
    import ast
    for col in ["y_prob", "y_true"]:
        df[col] = df[col].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
    return df


def summarise_run(df: pd.DataFrame, label: str, gamma: float, alpha: str):
    """Apply Global Youden per fold and compute aggregate metrics."""
    fold_rows = []
    for _, row in df.iterrows():
        fn       = int(row["fold"])
        subj     = row["test_subject"]
        y_true   = np.asarray(row["y_true"], dtype=int)
        y_prob   = np.asarray(row["y_prob"], dtype=float)

        # Pool other folds for global threshold
        other_true = np.concatenate([
            np.asarray(r["y_true"], dtype=int)
            for _, r in df.iterrows() if int(r["fold"]) != fn
        ])
        other_prob = np.concatenate([
            np.asarray(r["y_prob"], dtype=float)
            for _, r in df.iterrows() if int(r["fold"]) != fn
        ])
        thr = global_youden_threshold(other_prob, other_true)
        m   = metrics_at_threshold(y_true, y_prob, thr)
        m["fold"]     = fn
        m["subject"]  = subj
        m["threshold"]= round(thr, 4)
        fold_rows.append(m)

    fold_df  = pd.DataFrame(fold_rows)
    metric_keys = ["accuracy","balanced_acc","f1","mcc","roc_auc","pr_auc","ece","recall_minority"]
    summary = {
        "label"          : label,
        "gamma"          : gamma,
        "alpha_strategy" : alpha,
        "loss_type"      : "CrossEntropy" if gamma == 0 else "FocalLoss",
        "n_folds"        : len(fold_df),
    }
    for k in metric_keys:
        v = fold_df[k].dropna()
        summary[k]           = round(float(v.mean()), 4)
        summary[f"{k}_std"]  = round(float(v.std()),  4)

    # Hard-subject detail
    hard_df = fold_df[fold_df["subject"].isin(HARD_SUBJECTS)].copy()
    summary["hard_bal_acc_mean"]      = round(float(hard_df["balanced_acc"].mean()), 4)
    summary["hard_recall_min_mean"]   = round(float(hard_df["recall_minority"].mean()), 4)
    summary["hard_any_recovered"]     = int((hard_df["balanced_acc"] > 0.50).sum())

    return summary, fold_df


# ── Main ───────────────────────────────────────────────────────────────────────

configs = []
for gamma in GAMMAS:
    for alpha in ALPHAS:
        if gamma == 0.0 and alpha != "balanced":
            continue
        configs.append((gamma, alpha))

# Also include v17 (production run) as reference
v17_csv = METRICS_DIR / "ins_hdgs_cmt_v17" / "losocv_ins_hdgs_cmt_v17.csv"

all_summaries = []
all_fold_details = []

# v17 reference
v17_df = load_run(v17_csv)
if v17_df is not None:
    s, fd = summarise_run(v17_df, "v17_reference", gamma=2.0, alpha="balanced")
    s["label"] = "0_v17_reference (γ=2, balanced, N=15)"
    all_summaries.append(s)
    fd["config"] = "v17_reference"
    all_fold_details.append(fd)
    print(f"  v17 reference loaded  ({len(v17_df)} folds)")
else:
    print("  v17 reference CSV missing or no y_prob column")

found = 0
for gamma, alpha in configs:
    lbl     = label_for(gamma, alpha)
    csv_f   = METRICS_DIR / lbl / f"losocv_{lbl}.csv"
    df      = load_run(csv_f)
    if df is None:
        print(f"  MISSING: {csv_f.name}")
        continue
    s, fd = summarise_run(df, lbl, gamma, alpha)
    all_summaries.append(s)
    fd["config"] = lbl
    all_fold_details.append(fd)
    found += 1
    print(f"  {lbl:<45}  BalAcc={s['balanced_acc']:.4f}  MCC={s['mcc']:.4f}  "
          f"Hard-rec={s['hard_recall_min_mean']:.4f}")

print(f"\n{found}/{len(configs)} ablation configs loaded.")

if not all_summaries:
    print("No data — run sweep_focal_ablation.py first.")
    sys.exit(0)

summary_df     = pd.DataFrame(all_summaries)
fold_detail_df = pd.concat(all_fold_details, ignore_index=True)

# ── Report ─────────────────────────────────────────────────────────────────────
w = 72
print(f"\n{'='*w}")
print("  FOCAL LOSS ABLATION — Global Youden Thresholding")
print(f"{'='*w}")

disp = summary_df.sort_values("balanced_acc", ascending=False).reset_index(drop=True)
hdr  = f"{'Config':<38} {'γ':>4} {'Alpha':>14}  {'Acc':>6} {'BalAcc':>7} "
hdr += f"{'F1':>6} {'MCC':>6} {'AUC':>6} {'ECE':>6} {'HR_mn':>6} {'Rec5':>5}"
print(hdr)
print("─"*w)
for _, r in disp.iterrows():
    marker = " ◄" if r["balanced_acc"] == disp["balanced_acc"].max() else ""
    print(f"{r['label']:<38} {r['gamma']:>4.1f} {r['alpha_strategy']:>14}  "
          f"{r['accuracy']:>6.4f} {r['balanced_acc']:>7.4f} "
          f"{r['f1']:>6.4f} {r['mcc']:>6.4f} {r['roc_auc']:>6.4f} "
          f"{r['ece']:>6.4f} {r['hard_bal_acc_mean']:>6.4f} "
          f"{int(r['hard_any_recovered']):>5}{marker}")

print(f"\n{'─'*w}")
print("  HARD SUBJECT DETAIL  (S03 S13 S21 S35 S36)")
print(f"{'─'*w}")
hard_pivot = (fold_detail_df[fold_detail_df["subject"].isin(HARD_SUBJECTS)]
              .pivot_table(index="subject", columns="config",
                           values="balanced_acc", aggfunc="mean")
              .round(4))
if not hard_pivot.empty:
    print(hard_pivot.to_string())

# ── Ranking ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*w}")
print("  RANKING: BalAcc → MCC → Statistical Robustness")
print(f"{'─'*w}")
ranked = summary_df.sort_values(
    ["balanced_acc","mcc","hard_any_recovered"], ascending=[False,False,False]
).reset_index(drop=True)
for i, (_, r) in enumerate(ranked.iterrows(), 1):
    if i > 8:
        break
    print(f"  #{i:>2}  γ={r['gamma']:.1f}  {r['alpha_strategy']:<16}  "
          f"BalAcc={r['balanced_acc']:.4f}  MCC={r['mcc']:.4f}  "
          f"ECE={r['ece']:.4f}  HardRec={r['hard_any_recovered']}")

best = ranked.iloc[0]
print(f"\n  ► BEST: γ={best['gamma']:.1f}  α={best['alpha_strategy']}  "
      f"BalAcc={best['balanced_acc']:.4f}  MCC={best['mcc']:.4f}")

# ── Save CSVs ─────────────────────────────────────────────────────────────────
out_cols = ["label","loss_type","gamma","alpha_strategy","n_folds",
            "accuracy","balanced_acc","f1","mcc","roc_auc","pr_auc","ece",
            "recall_minority","hard_bal_acc_mean","hard_recall_min_mean","hard_any_recovered"]
summary_df[[c for c in out_cols if c in summary_df.columns]].to_csv(
    OUT_ANALYSIS / "focal_loss_ablation.csv", index=False)

fold_detail_df.to_csv(OUT_ANALYSIS / "focal_loss_hard_subjects.csv", index=False)
print(f"\nSaved: output/analysis/focal_loss_ablation.csv")
print(f"Saved: output/analysis/focal_loss_hard_subjects.csv")
