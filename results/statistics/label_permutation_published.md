# Within-fold label permutation test: published

Source: `results/losocv_metrics/losocv_repro_focal_g3p0_effective_num_37.csv` (37 folds, 347 held-out epochs). Labels shuffled within each fold, predictions fixed; N = 5000 permutations, seed = 0. p = ((count null >= observed) + 1) / (N + 1); the smallest attainable p is 2.00e-04.

This tests the association between the saved held-out predictions and the labels. It is NOT a test that the training procedure cannot fit random labels (that requires retraining under shuffled training labels, which this script does not do).

Stored per-fold balanced accuracy / MCC reproduced at the operating point (max |deviation| = 1.11e-16).

| statistic | observed | permutation p | null mean | null SD | null max |
|---|---|---|---|---|---|
| Pooled ROC-AUC (all held-out epochs) | 0.8729 | 2.00e-04 | 0.5960 | 0.0247 | 0.6968 |
| Mean per-fold balanced accuracy (fold-specific opt_threshold) | 0.7398 | 2.00e-04 | 0.5006 | 0.0278 | 0.6083 |
| Mean per-fold MCC (fold-specific opt_threshold) | 0.4631 | 2.00e-04 | 0.0014 | 0.0540 | 0.2127 |
| Mean per-fold ROC-AUC (descriptive) | 0.9014 | 2.00e-04 | 0.5005 | 0.0400 | 0.6353 |
