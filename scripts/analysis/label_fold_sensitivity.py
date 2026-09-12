#!/usr/bin/env python3
"""
Fold-wise engagement label: does building the label with the test subject included matter?

engagement_phase3d.py min-max scales every feature over ALL epochs of ALL subjects and thresholds
the weighted score at the GLOBAL median, so each held-out subject contributes to the scaler and to
the threshold that define its own labels.  This script rebuilds the label leave-one-subject-out
(scaler and median fitted on the other subjects only, applied to the held-out subject) and counts
how many labels change.  If a production per-fold CSV is given, it also re-scores the saved
held-out probabilities against the fold-wise labels, so the effect on the headline metrics is
known without retraining.

Inputs : <phase3>/output/engagement_phase3d/multimodal_features.csv   (written by engagement_phase3d.py)
Outputs: results/statistics/label_fold_sensitivity.csv  (per subject)
         results/statistics/label_fold_sensitivity.md   (summary for Sec. 2.4 / the response)

Run (repo root, CPU, seconds):
  python scripts/analysis/label_fold_sensitivity.py \
      --prod-csv results/losocv_metrics/losocv_repro_focal_g3p0_effective_num_37.csv
"""
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = ROOT / "src/data_pipeline/04_segmentation/output/engagement_phase3d/multimodal_features.csv"

# identical to engagement_phase3d.WEIGHTS (kept literal so the script has no pipeline import)
WEIGHTS = {
    "fixation_duration": +1.0, "dwell_time": +1.0, "roi_attention": +1.0, "revisit_count": +1.0,
    "theta": +1.5, "beta": +1.2, "alpha": -1.5, "theta_beta_ratio": +1.0,
    "frontal_asymmetry": +0.5, "gaze_entropy": -1.0,
}
COLS = list(WEIGHTS)
W = np.array([WEIGHTS[c] for c in COLS])


def scores(X_fit: np.ndarray, X_apply: np.ndarray) -> np.ndarray:
    """Weighted score of X_apply under a min-max scaler fitted on X_fit (engagement_phase3d.compute_scores)."""
    sc = MinMaxScaler().fit(X_fit)
    return sc.transform(X_apply) @ W


def load_label_feature_table(path=None) -> pd.DataFrame:
    """Per-epoch label feature table written by engagement_phase3d.py (multimodal_features.csv).

    Rows are in pipeline order: subjects in the order they were processed and, within a
    subject, epochs in the order of eeg_epochs_phase3d.npy / engagement_labels.npy, so the
    k-th row of a subject is that subject's k-th epoch in the model dataset.
    """
    df = pd.read_csv(path or DEFAULT_FEATURES)
    df["subject_id"] = df["subject_id"].astype(str)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path or DEFAULT_FEATURES}: label feature columns missing: {missing}")
    return df


def foldwise_labels(feature_table: pd.DataFrame, train_subjects, return_threshold: bool = False):
    """Engagement label of EVERY row of `feature_table` under the rule fitted on `train_subjects` only.

    The imputation means, the min-max scaler and the median threshold are computed from the
    training subjects' epochs alone and then applied to every epoch (training, validation and
    held-out subjects alike), so no held-out subject contributes to the rule that labels it.
    Returns a pd.Series of ints (index = feature_table.index); with return_threshold=True the
    fold threshold is returned as well.
    """
    subj = feature_table["subject_id"].astype(str).to_numpy()
    tr = np.isin(subj, [str(s) for s in train_subjects])
    if not tr.any():
        raise ValueError("foldwise_labels: none of the training subjects has rows in the feature table")
    X_all = feature_table[COLS].astype(float)
    mu = X_all[tr].mean()                        # fold-wise imputation from training subjects only
    Xtr = X_all[tr].fillna(mu).to_numpy()
    Xa = X_all.fillna(mu).to_numpy()
    s_tr = scores(Xtr, Xtr)
    thr = float(np.median(s_tr))
    y = pd.Series((scores(Xtr, Xa) >= thr).astype(int), index=feature_table.index)
    return (y, thr) if return_threshold else y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(DEFAULT_FEATURES))
    ap.add_argument("--prod-csv", default=None,
                    help="per-fold LOSOCV CSV with y_true / y_prob lists: re-score against the fold-wise labels")
    ap.add_argument("--out-dir", default=str(ROOT / "results/statistics"))
    args = ap.parse_args()

    df = load_label_feature_table(args.features)
    X = df[COLS].astype(float)
    X = X.fillna(X.mean())                       # same imputation as the pipeline (global means)
    X = X.to_numpy()
    subj = df["subject_id"].to_numpy()
    subjects = list(dict.fromkeys(subj))         # pipeline order

    # -- reproduce the global label exactly ----------------------------------------------------
    s_glob = scores(X, X)
    thr_glob = float(np.median(s_glob))
    y_glob = (s_glob >= thr_glob).astype(int)
    if "engagement_label" in df.columns:
        y_file = (df["engagement_label"].astype(str) == "HIGH_ENGAGEMENT").astype(int).to_numpy()
        mism = int((y_file != y_glob).sum())
        print(f"[label] global rule reproduced: {len(y_glob) - mism}/{len(y_glob)} labels identical to the file"
              + ("" if mism == 0 else f"   <-- {mism} differ; check FEATURE_COLS / imputation"))
    print(f"[label] {len(df)} epochs, {len(subjects)} subjects, global median {thr_glob:.4f}, "
          f"HIGH {int(y_glob.sum())} / LOW {int((1 - y_glob).sum())}")

    # -- leave-one-subject-out label ----------------------------------------------------------
    y_fold = np.zeros_like(y_glob)               # test-subject labels under the fold-wise rule
    rows = []
    train_flips = []
    for s in subjects:
        te = subj == s
        tr = ~te
        y_all, thr = foldwise_labels(df, [x for x in subjects if x != s], return_threshold=True)
        y_all = y_all.to_numpy()
        yt = y_all[te]
        y_fold[te] = yt
        n_flip = int((yt != y_glob[te]).sum())
        y_tr_fold = y_all[tr]
        train_flips.append(int((y_tr_fold != y_glob[tr]).sum()))
        rows.append(dict(subject=s, n_epochs=int(te.sum()), threshold_fold=round(thr, 4),
                         high_global=int(y_glob[te].sum()), high_fold=int(yt.sum()),
                         flips=n_flip, flip_rate=round(n_flip / te.sum(), 4),
                         single_class_global=bool(len(np.unique(y_glob[te])) < 2),
                         single_class_fold=bool(len(np.unique(yt)) < 2),
                         train_label_flips=train_flips[-1]))
    per = pd.DataFrame(rows)
    tot = int(per["flips"].sum())
    print(f"[label] fold-wise rule: {tot}/{len(df)} test labels change ({100 * tot / len(df):.2f} %), "
          f"{int((per['flips'] > 0).sum())} subjects affected; fold thresholds "
          f"{per['threshold_fold'].min():.4f}-{per['threshold_fold'].max():.4f} (global {thr_glob:.4f}); "
          f"training-set flips per fold {min(train_flips)}-{max(train_flips)} of {len(df) - per['n_epochs'].max()}")
    sc_chg = per[per["single_class_global"] != per["single_class_fold"]]
    if len(sc_chg):
        print(f"[label] subjects whose single-class status changes: {sc_chg['subject'].tolist()}")

    # -- re-score the production probabilities against the fold-wise labels ---------------------
    rescore = None
    if args.prod_csv:
        prod = pd.read_csv(args.prod_csv)
        prod["test_subject"] = prod["test_subject"].astype(str)
        met = []
        for _, r in prod.iterrows():
            s = r["test_subject"]
            yt = np.asarray(ast.literal_eval(r["y_true"]) if isinstance(r["y_true"], str) else r["y_true"], int)
            yp = np.asarray(ast.literal_eval(r["y_prob"]) if isinstance(r["y_prob"], str) else r["y_prob"], float)
            te = subj == s
            if te.sum() != len(yt):
                print(f"  [skip] {s}: {len(yt)} saved epochs vs {int(te.sum())} feature rows (order cannot be matched)")
                continue
            if int((y_glob[te] != yt).sum()):
                print(f"  [skip] {s}: saved y_true differs from the global label ({int((y_glob[te] != yt).sum())} epochs); "
                      f"epoch order differs")
                continue
            yf = y_fold[te]
            if len(np.unique(yf)) < 2:
                print(f"  [skip] {s}: single class under the fold-wise label")
                continue
            thr_p = float(r["opt_threshold"]) if "opt_threshold" in r and pd.notna(r["opt_threshold"]) else 0.5
            pred = (yp >= thr_p).astype(int)
            met.append(dict(subject=s,
                            bal_global=balanced_accuracy_score(yt, pred), bal_fold=balanced_accuracy_score(yf, pred),
                            auc_global=roc_auc_score(yt, yp), auc_fold=roc_auc_score(yf, yp),
                            mcc_global=matthews_corrcoef(yt, pred), mcc_fold=matthews_corrcoef(yf, pred)))
        if met:
            rescore = pd.DataFrame(met)
            m = rescore.mean(numeric_only=True)
            print(f"[rescore] {len(rescore)} folds of {Path(args.prod_csv).name} against the fold-wise label: "
                  f"BalAcc {m['bal_global']:.3f} -> {m['bal_fold']:.3f}, AUC {m['auc_global']:.3f} -> {m['auc_fold']:.3f}, "
                  f"MCC {m['mcc_global']:.3f} -> {m['mcc_fold']:.3f}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "label_fold_sensitivity.csv", index=False)
    lines = ["# Fold-wise engagement label (scaler + median fitted without the test subject)", "",
             f"* {len(df)} epochs, {len(subjects)} subjects; global median {thr_glob:.4f}; fold-wise thresholds "
             f"{per['threshold_fold'].min():.4f}-{per['threshold_fold'].max():.4f}.",
             f"* Test-subject labels that change: **{tot} / {len(df)} ({100 * tot / len(df):.2f} %)**, "
             f"{int((per['flips'] > 0).sum())} subjects affected (max {int(per['flips'].max())} epochs in one subject).",
             f"* Training-set labels that change per fold: {min(train_flips)}-{max(train_flips)}."]
    if len(sc_chg):
        lines.append(f"* Single-class status changes for: {sc_chg['subject'].tolist()}.")
    if rescore is not None:
        m = rescore.mean(numeric_only=True)
        lines.append(f"* Production probabilities re-scored against the fold-wise label ({len(rescore)} folds): "
                     f"BalAcc {m['bal_global']:.3f} -> {m['bal_fold']:.3f}, ROC-AUC {m['auc_global']:.3f} -> {m['auc_fold']:.3f}, "
                     f"MCC {m['mcc_global']:.3f} -> {m['mcc_fold']:.3f}.")
        rescore.to_csv(out / "label_fold_sensitivity_rescore.csv", index=False)
    lines += ["", "| subject | epochs | flips | flip rate | HIGH global | HIGH fold | fold threshold |", "|---|---|---|---|---|---|---|"]
    for _, r in per.iterrows():
        lines.append(f"| {r.subject} | {r.n_epochs} | {r.flips} | {r.flip_rate:.3f} | {r.high_global} | {r.high_fold} | {r.threshold_fold:.4f} |")
    (out / "label_fold_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'label_fold_sensitivity.csv'} and .md")


if __name__ == "__main__":
    main()
