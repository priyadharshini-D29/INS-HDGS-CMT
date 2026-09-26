#!/usr/bin/env python
"""Linear label-term probes under the FOLD-WISE label construction (no training of deep models).

Replicates, for the logistic probes of Table S15, exactly what the fold-wise encoder reruns of
Table S17 did to the label: per fold, the imputation means, min-max scaler and median threshold
are fitted on that fold's TRAINING subjects only (test subject and the matched validation subject
excluded), via scripts/analysis/label_fold_sensitivity.foldwise_labels — the same helper the
runner uses (src/model/evaluation/foldwise_label.py).  The probe itself follows
label_leakage_audit.py: standardised features, logistic regression C=1, fitted on the training
subjects' epochs, scored on the held-out subject.  Folds = the 37 subjects evaluable under the
pooled label (the decoder's folds); a fold is skipped for AUC if its held-out epochs are
single-class under the fold-wise label.

Writes results/statistics/label_leakage_audit_foldwise.csv and .md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import label_fold_sensitivity as lfs                      # noqa: E402

import importlib.util as _ilu                             # load val_split.py alone, without the

_spec = _ilu.spec_from_file_location("val_split", ROOT / "src/model/evaluation/val_split.py")
_vs = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_vs)  # heavy evaluation package
pick_validation_subject = _vs.pick_validation_subject

SEED = 42
EEG5 = ["theta", "alpha", "beta", "theta_beta_ratio", "frontal_asymmetry"]
GAZE5 = ["fixation_duration", "dwell_time", "roi_attention", "revisit_count", "gaze_entropy"]
SETS = {"EEG-5": EEG5, "GAZE-5": GAZE5, "ALL-10": EEG5 + GAZE5}

table = lfs.load_label_feature_table()
subj = table["subject_id"].astype(str).to_numpy()
all_subjects = list(dict.fromkeys(subj))

dec = pd.read_csv(ROOT / "results/ablation/abl_full/losocv_abl_full.csv")
folds = sorted(dec.test_subject.astype(str))                      # the 37 evaluable folds
stored = (table["engagement_label"].astype(str) == "HIGH_ENGAGEMENT").astype(int).to_numpy()
eligible = {s for s in all_subjects if len(np.unique(stored[subj == s])) == 2}
assert set(folds) <= set(all_subjects), "fold subjects missing from feature table"

rows, pooled = [], {k: ([], []) for k in SETS}
for s in folds:
    val = pick_validation_subject(s, candidates=[c for c in all_subjects if c != s],
                                  eligible=eligible, seed=SEED)
    train_subjects = [c for c in all_subjects if c not in (s, val)]
    y = lfs.foldwise_labels(table, train_subjects).to_numpy()
    tr = np.isin(subj, train_subjects)
    te = subj == s
    for name, cols in SETS.items():
        X = table[cols].astype(float)
        mu = X[tr].mean()
        Xtr, Xte = X[tr].fillna(mu).to_numpy(), X[te].fillna(mu).to_numpy()
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(sc.transform(Xtr), y[tr])
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        yte = y[te]
        auc = roc_auc_score(yte, p) if len(np.unique(yte)) == 2 else np.nan
        bal = balanced_accuracy_score(yte, (p >= 0.5).astype(int))
        rows.append(dict(test_subject=s, val_subject=val, feature_set=name, n_test=int(te.sum()),
                         single_class=int(len(np.unique(yte)) < 2), roc_auc=auc, balanced_acc=bal))
        pooled[name][0].append(yte); pooled[name][1].append(p)

df = pd.DataFrame(rows)
OUT = ROOT / "results/statistics"
df.to_csv(OUT / "label_leakage_audit_foldwise.csv", index=False)

lines = [f"# Label-recoverability probes under the FOLD-WISE label ({len(folds)} folds, seed {SEED}, matched val rule)",
         "", "Rule fitted per fold on the training subjects only (test + validation subject excluded),",
         "identical to the Table S17 encoder reruns (label_fold_sensitivity.foldwise_labels).", "",
         "| feature set | pooled AUC | fold AUC (mean ± SD) | fold BalAcc (mean ± SD) | single-class folds |",
         "|---|---|---|---|---|"]
for name in SETS:
    g = df[df.feature_set == name]
    yy, pp = np.concatenate(pooled[name][0]), np.concatenate(pooled[name][1])
    lines.append(f"| {name} | {roc_auc_score(yy, pp):.3f} | {g.roc_auc.mean():.3f} ± {g.roc_auc.std(ddof=1):.3f} "
                 f"| {g.balanced_acc.mean():.3f} ± {g.balanced_acc.std(ddof=1):.3f} | {int(g.single_class.sum())} |")
(OUT / "label_leakage_audit_foldwise.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
