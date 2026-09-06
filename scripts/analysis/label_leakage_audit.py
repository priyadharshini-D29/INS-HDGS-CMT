"""
================================================================
Label-recoverability (leakage) audit of the engagement label
================================================================
The production engagement label is produced by
src/data_pipeline/04_segmentation/engagement_phase3d.py: a fixed weighted
composite of FIVE frontal-EEG band-power terms and FIVE gaze statistics,
min-max normalised over the pooled stimulus epochs and split at the pooled
median.  (Verified: the labels it writes reproduce the `y_true` of every
held-out subject in results/losocv_metrics/losocv_repro_focal_g3p0_effective_num_37.csv,
347/347.)

Because BOTH modalities enter the label, no input branch of the model is
independent of it.  This script quantifies how much of the label is linearly
recoverable from each modality's *defining* features under a subject-grouped
leave-one-subject-out protocol (the same folds as the deep models):

    EEG-5   theta, alpha, beta, theta/beta ratio, frontal asymmetry
    ET-5    |x| mean, |y| mean, std(x), revisit count, 20-bin x entropy
    ALL-10  both sets
    RULE    the exact composite score itself (upper bound: AUC 1.0 by construction)

For each set a logistic-regression probe (standardised features, C=1) is fitted
on the training subjects and scored on the held-out subject; pooled and
per-fold ROC-AUC / balanced accuracy are reported, restricted to the 37
subjects that carry both classes (identical to the model's test folds).

Usage (from anywhere; needs the preprocessed epochs under
src/data_pipeline/04_segmentation/S*/output/ and the engagement_phase3d labels):
    python scripts/analysis/label_leakage_audit.py [--ref-csv <per-fold csv>]

Outputs
    results/statistics/label_leakage_audit.csv
    results/statistics/label_leakage_audit.md
================================================================
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "src" / "data_pipeline" / "04_segmentation"
OUT = ROOT / "results" / "statistics"

EEG_TERMS = ["theta", "alpha", "beta", "theta_beta_ratio", "frontal_asymmetry"]
ET_TERMS = ["fixation_duration", "dwell_time", "roi_attention", "revisit_count", "gaze_entropy"]


def _load_phase3d():
    spec = importlib.util.spec_from_file_location("engagement_phase3d", SEG / "engagement_phase3d.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]]  # the module parses argv only inside its main(); keep it clean anyway
    spec.loader.exec_module(mod)
    return mod


def build_table(mod, subjects):
    data = []
    for s in subjects:
        d = mod.load_subject(SEG / s)
        if d is not None:
            data.append(d)
    feats = mod.build_feature_table(data)
    feats["score"] = mod.compute_scores(feats)
    feats["label"] = (feats["score"] >= np.median(feats["score"])).astype(int)
    # cross-check against the labels written to disk (if present)
    n_ok = n_tot = 0
    for s, g in feats.groupby("subject_id"):
        f = SEG / s / "output" / "engagement_phase3d" / "engagement_labels.npy"
        if f.exists():
            y = np.load(f)
            if len(y) == len(g):
                n_ok += int((y == g["label"].to_numpy()).sum()); n_tot += len(y)
    return feats, (n_ok, n_tot)


def build_meta_table(mod, subjects, sub_dir, suffix, extra_cols=()):
    """features of pre-cut epochs (<sub_dir>/engagement_metadata.csv + epochs/*_<suffix>.npy)
    + their label + the listed metadata columns."""
    recs = []
    for s in subjects:
        d = SEG / s / "output"
        meta = d / sub_dir / "engagement_metadata.csv"
        if not meta.exists():
            continue
        m = pd.read_csv(meta)
        eeg = np.load(d / "epochs" / f"eeg_epochs_{suffix}.npy", allow_pickle=True)
        et = np.load(d / "epochs" / f"et_epochs_{suffix}.npy", allow_pickle=True)
        for i, row in m.iterrows():
            fe, ft = mod.extract_eeg_features(eeg[i]), mod.extract_et_features(et[i])
            if fe is None or ft is None:
                continue
            recs.append(dict(subject_id=s, epoch_idx=i, label=int(row["label"]),
                             **{c: row[c] for c in extra_cols}, **fe, **ft))
    feats = pd.DataFrame(recs)
    feats["score"] = mod.compute_scores(feats)
    return feats


def build_product_table(mod, subjects):
    """product-level epochs (product_epoching.py) + behavioural label + dwell covariates."""
    f = build_meta_table(mod, subjects, "engagement_product", "product",
                         ("page", "total_dwell_s", "n_runs", "n_views", "anchor_run_s"))
    f["page"] = f["page"].astype(int)
    return f


def within_subject_z(feats, cols):
    """z-score every feature within subject (removes between-subject offsets; the
    per-subject mean/SD use no labels, so this is legal at deployment time)."""
    out = feats.copy()
    for c in cols:
        g = out.groupby("subject_id")[c]
        out[c] = (out[c] - g.transform("mean")) / (g.transform("std").replace(0, np.nan)).fillna(1.0)
    return out.fillna(0.0)


def within_page_ranking(feats, p, mask):
    """rank products by probe score within each (subject, page) that holds both classes:
    mean within-page AUC and the hit rate of a bought product in the top 3."""
    d = feats.loc[mask, ["subject_id", "page", "label"]].copy(); d["p"] = p
    from sklearn.metrics import roc_auc_score
    aucs, hits = [], []
    for _, g in d.groupby(["subject_id", "page"]):
        if g.label.nunique() < 2:
            continue
        aucs.append(roc_auc_score(g.label, g.p))
        top = g.sort_values("p", ascending=False).head(3)
        hits.append(float(top.label.max()))
    return (float(np.mean(aucs)) if aucs else float("nan"),
            float(np.mean(hits)) if hits else float("nan"), len(aucs))


def loso_probe(feats, cols, evaluable):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = feats[cols].to_numpy(float)
    y = feats["label"].to_numpy(int)
    sid = feats["subject_id"].to_numpy()
    rows, pooled_p, pooled_y = [], [], []
    p_rows = np.full(len(feats), np.nan)          # held-out probability in row order
    for s in evaluable:
        tr, te = sid != s, sid == s
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        p_rows[te] = p
        pooled_p.append(p); pooled_y.append(y[te])
        rows.append(dict(test_subject=s, n=int(te.sum()),
                         roc_auc=float(roc_auc_score(y[te], p)),
                         balanced_acc=float(balanced_accuracy_score(y[te], (p >= 0.5).astype(int)))))
    per = pd.DataFrame(rows)
    P, Y = np.concatenate(pooled_p), np.concatenate(pooled_y)
    extra = {}
    if "page" in feats.columns:
        mask = ~np.isnan(p_rows)
        wp_auc, top3, n_pages = within_page_ranking(feats, p_rows[mask], mask)
        extra = dict(within_page_auc=wp_auc, top3_hit=top3, n_pages=n_pages)
    return per, dict(pooled_auc=float(roc_auc_score(Y, P)), **extra,
                     pooled_balacc=float(balanced_accuracy_score(Y, (P >= 0.5).astype(int))),
                     fold_auc_mean=float(per.roc_auc.mean()), fold_auc_sd=float(per.roc_auc.std()),
                     fold_balacc_mean=float(per.balanced_acc.mean()), fold_balacc_sd=float(per.balanced_acc.std()),
                     n_folds=len(per))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-csv", default=str(ROOT / "results" / "losocv_metrics" /
                                             "losocv_repro_focal_g3p0_effective_num_37.csv"),
                    help="per-fold CSV whose test_subject column defines the evaluable folds")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--label-source", default="phase3d", choices=["phase3d", "purchase", "product", "control"],
                    help="phase3d: the rule-based index (default); purchase/product: behavioural labels; "
                         "control: browsing vs resting state (control_epoching.py)")
    args = ap.parse_args()

    mod = _load_phase3d()
    subjects = sorted(d.name for d in SEG.iterdir()
                      if d.is_dir() and d.name.startswith("S") and (d / "output" / "epochs" / "eeg_epochs.npy").exists())
    if args.label_source == "product":
        feats, n_ok, n_tot = build_product_table(mod, subjects), 0, 0
        print(f"[audit] product-level: {len(feats)} epochs from {feats.subject_id.nunique()} subjects; bought={int(feats.label.sum())} ({feats.label.mean():.3f})")
    elif args.label_source == "control":
        feats, n_ok, n_tot = build_meta_table(mod, subjects, "engagement_control", "control", ("condition",)), 0, 0
        print(f"[audit] positive control: {len(feats)} epochs from {feats.subject_id.nunique()} subjects; browsing={int(feats.label.sum())} ({feats.label.mean():.3f})")
    else:
        feats, (n_ok, n_tot) = build_table(mod, subjects)
        print(f"[audit] {len(feats)} stimulus epochs from {feats.subject_id.nunique()} subjects; "
              f"HIGH={int(feats.label.sum())}; agreement with on-disk phase3d labels {n_ok}/{n_tot}")
    suffix = {"product": "_product", "control": "_control"}.get(args.label_source, "")
    if args.label_source == "purchase":
        # replace the rule-based label by the behavioural one (dominant fixated product bought)
        parts = []
        for s in subjects:
            f = SEG / s / "output" / "engagement_purchase" / "engagement_metadata.csv"
            if f.exists():
                parts.append(pd.read_csv(f))
        if not parts:
            raise SystemExit("no engagement_purchase labels found — run 04_segmentation/purchase_labeling.py")
        pl = pd.concat(parts, ignore_index=True)
        pl = pl[pl.kept == 1][["subject_id", "epoch_idx", "label", "dom_frac", "gaze_on_bought_frac"]].rename(columns={"label": "label_purchase"})
        feats = feats.merge(pl, on=["subject_id", "epoch_idx"], how="inner")
        feats["label"] = feats["label_purchase"].astype(int)
        suffix = "_purchase"
        print(f"[audit] purchase label: {len(feats)} epochs, bought={int(feats.label.sum())} ({feats.label.mean():.3f})")

    ref = Path(args.ref_csv)
    if args.label_source in ("purchase", "product", "control"):
        ref = None                # evaluable folds are defined by the purchase label itself (both classes present)
    if ref is not None and ref.exists():
        evaluable = sorted(pd.read_csv(ref)["test_subject"].astype(str).unique())
    else:
        evaluable = [s for s, g in feats.groupby("subject_id") if g.label.nunique() > 1]
    evaluable = [s for s in evaluable if s in set(feats.subject_id)]

    sets = {"EEG-5 (frontal band power)": EEG_TERMS, "ET-5 (gaze statistics)": ET_TERMS,
            "ALL-10": EEG_TERMS + ET_TERMS, "RULE score (upper bound)": ["score"]}
    if args.label_source == "purchase":
        sets["RULE score (old engagement index)"] = sets.pop("RULE score (upper bound)")
        sets["Dominant-product dwell fraction"] = ["dom_frac"]
    if args.label_source == "control":
        sets["RULE score (old engagement index)"] = sets.pop("RULE score (upper bound)")
    if args.label_source == "product":
        sets["RULE score (old engagement index)"] = sets.pop("RULE score (upper bound)")
        sets["Total dwell on product (s)"] = ["total_dwell_s"]
        sets["Dwell + n_runs + n_views + anchor run"] = ["total_dwell_s", "n_runs", "n_views", "anchor_run_s"]
    summary, per_fold = [], []
    variants = [("", feats)]
    if args.label_source == "product":
        # second pass: features standardised within subject (subject offsets removed)
        allc = sorted({c for cols in sets.values() for c in cols})
        variants.append((" [within-subject z]", within_subject_z(feats, allc)))
    for vtag, F in variants:
        for name, cols in sets.items():
            per, agg = loso_probe(F, cols, evaluable)
            per["feature_set"] = name + vtag; per_fold.append(per)
            summary.append(dict(feature_set=name + vtag, **agg))
            wp = f"  within-page AUC {agg['within_page_auc']:.3f}  top3 {agg['top3_hit']:.3f} ({agg['n_pages']} pages)" if "within_page_auc" in agg else ""
            print(f"  {name + vtag:50s} pooled AUC {agg['pooled_auc']:.3f}  fold AUC {agg['fold_auc_mean']:.3f}±{agg['fold_auc_sd']:.3f}  "
                  f"fold BalAcc {agg['fold_balacc_mean']:.3f}{wp}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / f"label_leakage_audit{suffix}.csv", index=False)
    pd.concat(per_fold).to_csv(out / f"label_leakage_audit_per_fold{suffix}.csv", index=False)
    title = {"": "engagement_phase3d label", "_purchase": "purchase-intent label", "_product": "product-level purchase label",
             "_control": "positive control: browsing vs rest"}[suffix]
    lines = [f"# Label-recoverability audit ({title})", "",
             f"{len(feats)} stimulus epochs, {feats.subject_id.nunique()} subjects; probes evaluated on the "
             f"{len(evaluable)} subjects with both classes (same folds as the deep models). "
             f"On-disk label agreement {n_ok}/{n_tot}.", "",
             "| feature set | pooled AUC | fold AUC (mean ± SD) | pooled BalAcc | fold BalAcc (mean ± SD) |"
             + (" within-page AUC | top-3 hit |" if args.label_source == "product" else ""),
             "|---|---|---|---|---|" + ("---|---|" if args.label_source == "product" else "")]
    for r in summary:
        row = (f"| {r['feature_set']} | {r['pooled_auc']:.3f} | {r['fold_auc_mean']:.3f} ± {r['fold_auc_sd']:.3f} | "
               f"{r['pooled_balacc']:.3f} | {r['fold_balacc_mean']:.3f} ± {r['fold_balacc_sd']:.3f} |")
        if "within_page_auc" in r:
            row += f" {r['within_page_auc']:.3f} | {r['top3_hit']:.3f} |"
        lines.append(row)
    lines += ["", "Interpretation: the EEG-5 and ET-5 rows are the linear floor that a model seeing only that modality's "
              "defining statistics attains; a learned model is informative beyond the label rule only where it exceeds "
              "the corresponding row. Fill Supplementary Table S15 and Sections 2.4 / 3.2 of the manuscript from this table."]
    (out / f"label_leakage_audit{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
