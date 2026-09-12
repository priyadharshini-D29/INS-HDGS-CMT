"""
Threshold Strategy Comparison for INS-HDGS-CMT v17
===================================================
Benchmarks every thresholding and temperature-scaling strategy against
the saved per-fold probabilities / logits.

Input : output/fold_probs/fold*.npz
Output: output/analysis/threshold_strategy_comparison.csv
        (and a detailed console report)

No GPU required — pure numpy / sklearn.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.special  import softmax as sp_softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    roc_curve, cohen_kappa_score,
)

OUT_ANALYSIS = Path("output/analysis")
OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
PROB_DIR     = Path("output/fold_probs")

TEMPERATURES = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]
BASELINE_CSV = Path("output/metrics/ins_hdgs_cmt_v17/losocv_ins_hdgs_cmt_v17.csv")


# ── Metric helpers ─────────────────────────────────────────────────────────────

def ece(y_true, y_prob, n_bins=10):
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
    n1 = y_true.sum(); n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return dict(accuracy=float('nan'), balanced_acc=float('nan'),
                    f1=float('nan'), mcc=float('nan'),
                    roc_auc=float('nan'), pr_auc=float('nan'),
                    ece=float('nan'))
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    try:
        pr  = average_precision_score(y_true, y_prob)
    except Exception:
        pr = float('nan')
    return dict(
        accuracy     = accuracy_score(y_true, y_pred),
        balanced_acc = balanced_accuracy_score(y_true, y_pred),
        f1           = f1_score(y_true, y_pred, zero_division=0),
        mcc          = matthews_corrcoef(y_true, y_pred),
        roc_auc      = auc,
        pr_auc       = pr,
        ece          = ece(y_true, y_prob),
    )


def youden_threshold(probs, labels):
    if len(np.unique(labels)) < 2 or len(labels) < 4:
        return 0.5
    fpr, tpr, thrs = roc_curve(labels, probs)
    idx = int(np.argmax(tpr - fpr))
    return float(np.clip(thrs[idx], 0.05, 0.95))


def grid_threshold(probs, labels):
    """Maximise balanced accuracy over 91 candidates."""
    if len(np.unique(labels)) < 2:
        return 0.5
    best_thr, best_bal = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        bal = balanced_accuracy_score(labels, (probs >= thr).astype(int))
        if bal > best_bal:
            best_bal, best_thr = bal, float(thr)
    return best_thr


# ── Load fold data ─────────────────────────────────────────────────────────────

def load_folds():
    folds = {}
    for f in sorted(PROB_DIR.glob("fold*.npz")):
        d = np.load(f, allow_pickle=True)
        fn = int(d["fold_no"])
        folds[fn] = {
            "test_subj"     : str(d["test_subj"]),
            "y_true"        : d["y_true"].astype(int),
            "y_prob"        : d["y_prob"].astype(float),
            "avg_logits"    : d["avg_logits"].astype(float),
            "val_y_true"    : d["val_y_true"].astype(int),
            "val_y_prob"    : d["val_y_prob"].astype(float),
            "val_avg_logits": d["val_avg_logits"].astype(float),
            "T_per_member"  : d["T_per_member"].astype(float),
        }
    return folds


# ── Strategy definitions ───────────────────────────────────────────────────────
#
# Each strategy is a function:
#   (fold_data, all_folds) -> per_fold_metrics_dict
#
# fold_data : dict for the current test fold
# all_folds : dict of ALL folds (used for global strategies)

def _apply_T_to_logits(logits, T):
    return sp_softmax(logits / T, axis=1)[:, 1]


def strategy_baseline(fold, all_folds, fold_no):
    """Current: val-subject grid-search threshold on per-member-T probs."""
    probs = fold["y_prob"]
    thr   = grid_threshold(fold["val_y_prob"], fold["val_y_true"])
    return metrics_at_threshold(fold["y_true"], probs, thr), thr


def strategy_fixed_05(fold, all_folds, fold_no):
    probs = fold["y_prob"]
    return metrics_at_threshold(fold["y_true"], probs, 0.5), 0.5


def strategy_global_grid(fold, all_folds, fold_no):
    """Global threshold: grid search on pooled test probs of OTHER folds."""
    pool_prob  = np.concatenate([v["y_prob"]  for k, v in all_folds.items() if k != fold_no])
    pool_true  = np.concatenate([v["y_true"]  for k, v in all_folds.items() if k != fold_no])
    thr = grid_threshold(pool_prob, pool_true)
    return metrics_at_threshold(fold["y_true"], fold["y_prob"], thr), thr


def strategy_global_youden(fold, all_folds, fold_no):
    """Global Youden threshold on pooled test probs of OTHER folds."""
    pool_prob = np.concatenate([v["y_prob"]  for k, v in all_folds.items() if k != fold_no])
    pool_true = np.concatenate([v["y_true"]  for k, v in all_folds.items() if k != fold_no])
    thr = youden_threshold(pool_prob, pool_true)
    return metrics_at_threshold(fold["y_true"], fold["y_prob"], thr), thr


def _T_fixed_05(fold, all_folds, fold_no, T):
    probs = _apply_T_to_logits(fold["avg_logits"], T)
    return metrics_at_threshold(fold["y_true"], probs, 0.5), 0.5


def _T_fixed_global_grid(fold, all_folds, fold_no, T):
    probs_all  = {k: _apply_T_to_logits(v["avg_logits"], T) for k, v in all_folds.items()}
    pool_prob  = np.concatenate([probs_all[k] for k in all_folds if k != fold_no])
    pool_true  = np.concatenate([all_folds[k]["y_true"] for k in all_folds if k != fold_no])
    thr        = grid_threshold(pool_prob, pool_true)
    return metrics_at_threshold(fold["y_true"], probs_all[fold_no], thr), thr


def _T_fixed_youden(fold, all_folds, fold_no, T):
    probs_all  = {k: _apply_T_to_logits(v["avg_logits"], T) for k, v in all_folds.items()}
    pool_prob  = np.concatenate([probs_all[k] for k in all_folds if k != fold_no])
    pool_true  = np.concatenate([all_folds[k]["y_true"] for k in all_folds if k != fold_no])
    thr        = youden_threshold(pool_prob, pool_true)
    return metrics_at_threshold(fold["y_true"], probs_all[fold_no], thr), thr


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_all_strategies(folds):
    STRATEGIES = {}

    # 1. Baseline
    STRATEGIES["1_baseline_val_grid"] = lambda f, af, fn: strategy_baseline(f, af, fn)

    # 2. Fixed 0.5
    STRATEGIES["2_fixed_0.50"] = lambda f, af, fn: strategy_fixed_05(f, af, fn)

    # 3. Global grid threshold
    STRATEGIES["3_global_grid"] = lambda f, af, fn: strategy_global_grid(f, af, fn)

    # 4. Global Youden threshold
    STRATEGIES["4_global_youden"] = lambda f, af, fn: strategy_global_youden(f, af, fn)

    # 5–11. Temperature only (T × 0.5 threshold)
    for T in TEMPERATURES:
        name = f"5_T{T:.2f}_thr0.50"
        _T   = T
        STRATEGIES[name] = lambda f, af, fn, T=_T: _T_fixed_05(f, af, fn, T)

    # 12–18. Temperature + global grid threshold
    for T in TEMPERATURES:
        name = f"6_T{T:.2f}_global_grid"
        _T   = T
        STRATEGIES[name] = lambda f, af, fn, T=_T: _T_fixed_global_grid(f, af, fn, T)

    # 19–25. Temperature + global Youden threshold
    for T in TEMPERATURES:
        name = f"7_T{T:.2f}_global_youden"
        _T   = T
        STRATEGIES[name] = lambda f, af, fn, T=_T: _T_fixed_youden(f, af, fn, T)

    # Run every strategy
    results = {}     # strategy -> list of per-fold metric dicts
    thresholds = {}  # strategy -> list of per-fold thresholds

    baseline_bal = None   # computed once, used to find "recovered" folds

    for strat_name, fn in STRATEGIES.items():
        fold_metrics = {}
        fold_thrs    = {}
        for fold_no, fold_data in folds.items():
            m, thr = fn(fold_data, folds, fold_no)
            fold_metrics[fold_no] = m
            fold_thrs[fold_no]    = thr
        results[strat_name]    = fold_metrics
        thresholds[strat_name] = fold_thrs
        if strat_name == "1_baseline_val_grid":
            baseline_bal = {fn: m["balanced_acc"] for fn, m in fold_metrics.items()}

    return results, thresholds, baseline_bal


def summarise(results, baseline_bal, folds):
    rows = []
    metric_keys = ["accuracy","balanced_acc","f1","mcc","roc_auc","pr_auc","ece"]

    for strat_name, fold_metrics in results.items():
        vals = {k: [] for k in metric_keys}
        for fn, m in fold_metrics.items():
            for k in metric_keys:
                v = m.get(k, float('nan'))
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals[k].append(v)

        row = {"strategy": strat_name}
        for k in metric_keys:
            arr = np.array(vals[k])
            row[k]        = round(float(np.nanmean(arr)), 4)
            row[f"{k}_std"] = round(float(np.nanstd(arr)), 4)

        # Recovered folds: baseline BalAcc=0.5 → now >0.55
        recovered = []
        newly_broken = []
        for fn, m in fold_metrics.items():
            base = baseline_bal.get(fn, float('nan'))
            cal  = m.get("balanced_acc", float('nan'))
            subj = folds[fn]["test_subj"]
            if not np.isnan(base) and not np.isnan(cal):
                if base <= 0.50 and cal > 0.55:
                    recovered.append(subj)
                if base > 0.70 and cal < base - 0.10:
                    newly_broken.append(subj)

        row["recovered_folds"]     = len(recovered)
        row["recovered_subjects"]  = "|".join(sorted(recovered))
        row["broken_subjects"]     = "|".join(sorted(newly_broken))

        # Remaining worst folds
        worst = sorted(fold_metrics.items(), key=lambda x: x[1].get("balanced_acc", 1.0))[:3]
        row["worst_subjects"] = "|".join(folds[fn]["test_subj"] for fn, _ in worst)

        row["delta_bal_acc"] = round(row["balanced_acc"] - results["1_baseline_val_grid"].get(
            list(results["1_baseline_val_grid"].keys())[0], {}).get("balanced_acc", row["balanced_acc"]), 4)

        rows.append(row)

    # Recompute delta vs true baseline mean
    base_mean = np.nanmean([v["balanced_acc"] for v in results["1_baseline_val_grid"].values()
                             if v.get("balanced_acc") is not None])
    for row in rows:
        row["delta_bal_acc"] = round(row["balanced_acc"] - base_mean, 4)

    df = pd.DataFrame(rows)
    return df


def print_report(df, folds, results):
    w = 72
    print(f"\n{'='*w}")
    print("  THRESHOLD STRATEGY COMPARISON — INS-HDGS-CMT v17")
    print(f"  {len(folds)} folds  |  {len(df)} strategies")
    print(f"{'='*w}")

    # Full table sorted by BalAcc
    display = df.sort_values("balanced_acc", ascending=False).reset_index(drop=True)
    cols = ["strategy","accuracy","balanced_acc","f1","mcc","roc_auc","pr_auc","ece",
            "delta_bal_acc","recovered_folds"]
    hdr = f"{'Strategy':<32} {'Acc':>6} {'BalAcc':>7} {'F1':>6} {'MCC':>6} "
    hdr += f"{'AUC':>6} {'PR':>6} {'ECE':>6} {'ΔBal':>6} {'Rec':>4}"
    print(f"\n{'─'*w}")
    print(hdr)
    print(f"{'─'*w}")
    for _, r in display.iterrows():
        marker = " ◄" if r["balanced_acc"] == display["balanced_acc"].max() else ""
        print(f"{r['strategy']:<32} "
              f"{r['accuracy']:>6.4f} {r['balanced_acc']:>7.4f} {r['f1']:>6.4f} "
              f"{r['mcc']:>6.4f} {r['roc_auc']:>6.4f} {r['pr_auc']:>6.4f} "
              f"{r['ece']:>6.4f} {r['delta_bal_acc']:>+6.4f} {int(r['recovered_folds']):>4}{marker}")
    print(f"{'─'*w}")

    # ── Per-fold detail for top 3 strategies ──────────────────────────────────
    top3 = display.head(3)["strategy"].tolist()
    baseline_name = "1_baseline_val_grid"
    all_s = [baseline_name] + [s for s in top3 if s != baseline_name][:2]

    print(f"\n{'─'*w}")
    print("  PER-FOLD DETAIL  (Baseline → top strategies)")
    print(f"{'─'*w}")
    hdr2 = f"  {'Subj':>5}  {'Baseline':>8}"
    for s in all_s[1:]:
        hdr2 += f"  {s[:10]:>10}"
    hdr2 += "  Change"
    print(hdr2)
    print(f"  {'─'*65}")
    sorted_folds = sorted(folds.keys())
    for fn in sorted_folds:
        subj = folds[fn]["test_subj"]
        base = results[baseline_name][fn]["balanced_acc"]
        line = f"  {subj:>5}  {base:>8.4f}"
        best_other = base
        for s in all_s[1:]:
            v = results[s][fn]["balanced_acc"]
            best_other = max(best_other, v)
            line += f"  {v:>10.4f}"
        diff = best_other - base
        tag  = f"  {diff:+.4f}"
        if diff > 0.05:  tag += " ↑"
        elif diff < -0.05: tag += " ↓"
        print(line + tag)

    # ── Recovery analysis ────────────────────────────────────────────────────
    print(f"\n{'─'*w}")
    print("  THRESHOLD-FAILURE FOLD RECOVERY")
    print(f"{'─'*w}")
    collapsed = {fn: folds[fn]["test_subj"]
                 for fn in folds
                 if results[baseline_name][fn]["balanced_acc"] <= 0.50}
    print(f"  Baseline collapsed folds (BalAcc≤0.50): {len(collapsed)}")
    print(f"  {list(collapsed.values())}")
    print()
    for _, r in display.iterrows():
        if r["recovered_folds"] > 0:
            print(f"  {r['strategy']:<32}  recovered={int(r['recovered_folds'])}  "
                  f"subjects=[{r['recovered_subjects']}]")

    # ── Ranking ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*w}")
    print("  RANKING BY: BalAcc → MCC → Statistical Defensibility → Cost")
    print(f"{'─'*w}")
    ranked = display.copy()
    ranked["cost_score"] = ranked["strategy"].apply(lambda s:
        0 if "baseline" in s or "fixed" in s or "global" in s
        else 1)
    ranked["defensible"] = ranked["strategy"].apply(lambda s:
        3 if "global_youden" in s or "global_grid" in s
        else (2 if "fixed" in s else 1))
    ranked = ranked.sort_values(
        ["balanced_acc","mcc","defensible","cost_score"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)
    for rank, (_, r) in enumerate(ranked.iterrows(), 1):
        if rank > 8:
            break
        print(f"  #{rank:>2}  {r['strategy']:<32}  BalAcc={r['balanced_acc']:.4f}  "
              f"MCC={r['mcc']:.4f}")

    # ── Recommendation ───────────────────────────────────────────────────────
    best = ranked.iloc[0]
    print(f"\n{'='*w}")
    print("  RECOMMENDATION")
    print(f"{'='*w}")
    print(f"  Best strategy : {best['strategy']}")
    print(f"  Balanced Acc  : {best['balanced_acc']:.4f}  (baseline: {df[df.strategy=='1_baseline_val_grid']['balanced_acc'].values[0]:.4f})")
    print(f"  MCC           : {best['mcc']:.4f}")
    print(f"  ECE           : {best['ece']:.4f}")
    print(f"  Recovered folds: {int(best['recovered_folds'])}")
    print(f"{'='*w}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading fold probability files …")
    folds = load_folds()
    if not folds:
        print(f"ERROR: no .npz files found in {PROB_DIR}")
        print("Run collect_fold_probs.py first.")
        sys.exit(1)
    print(f"Loaded {len(folds)} folds: {sorted(folds.keys())}")

    print("\nRunning strategy sweep …")
    results, thresholds, baseline_bal = run_all_strategies(folds)

    print("Summarising …")
    df = summarise(results, baseline_bal, folds)

    print_report(df, folds, results)

    # Save CSV
    csv_cols = ["strategy","accuracy","balanced_acc","f1","mcc","roc_auc","pr_auc","ece",
                "delta_bal_acc","recovered_folds","recovered_subjects","broken_subjects","worst_subjects"]
    df_save = df[[c for c in csv_cols if c in df.columns]].copy()
    df_save.columns = [c.replace("balanced_acc","Balanced_Accuracy").replace("roc_auc","ROC_AUC")
                        .replace("pr_auc","PR_AUC").replace("strategy","Strategy")
                        .replace("accuracy","Accuracy").replace("delta_bal_acc","Delta_BalAcc")
                        .replace("recovered_folds","Recovered_Folds")
                        .replace("recovered_subjects","Recovered_Subjects")
                        .replace("broken_subjects","Broken_Subjects")
                        .replace("worst_subjects","Worst_Subjects")
                        for c in df_save.columns]
    out_csv = OUT_ANALYSIS / "threshold_strategy_comparison.csv"
    df_save.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Per-fold detail CSV
    per_fold_rows = []
    for strat_name, fold_metrics in results.items():
        for fn, m in fold_metrics.items():
            r = {"strategy": strat_name, "fold": fn,
                 "test_subject": folds[fn]["test_subj"],
                 "threshold": round(thresholds[strat_name][fn], 4)}
            r.update({k: round(v, 4) if v is not None else float('nan')
                      for k, v in m.items()})
            per_fold_rows.append(r)
    pd.DataFrame(per_fold_rows).to_csv(OUT_ANALYSIS / "threshold_strategies_per_fold.csv", index=False)
    print(f"Saved: {OUT_ANALYSIS}/threshold_strategies_per_fold.csv")
