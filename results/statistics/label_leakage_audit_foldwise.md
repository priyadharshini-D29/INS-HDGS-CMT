# Label-recoverability probes under the FOLD-WISE label (37 folds, seed 42, matched val rule)

Rule fitted per fold on the training subjects only (test + validation subject excluded),
identical to the Table S17 encoder reruns (label_fold_sensitivity.foldwise_labels).

| feature set | pooled AUC | fold AUC (mean ± SD) | fold BalAcc (mean ± SD) | single-class folds |
|---|---|---|---|---|
| EEG-5 | 0.629 | 0.680 ± 0.216 | 0.578 ± 0.172 | 0 |
| GAZE-5 | 0.940 | 0.927 ± 0.124 | 0.844 ± 0.169 | 0 |
| ALL-10 | 0.996 | 0.994 ± 0.033 | 0.939 ± 0.126 | 0 |
