# Within-fold label permutation test: abl_full

Source: `results/ablation/abl_full/losocv_abl_full.csv` (37 folds, 347 held-out epochs). Labels shuffled within each fold, predictions fixed; N = 5000 permutations, seed = 0. p = ((count null >= observed) + 1) / (N + 1); the smallest attainable p is 2.00e-04.

This tests the association between the saved held-out predictions and the labels. It is NOT a test that the training procedure cannot fit random labels (that requires retraining under shuffled training labels, which this script does not do).

Stored per-fold balanced accuracy / MCC reproduced at the operating point (max |deviation| = 1.11e-16).

| statistic | observed | permutation p | null mean | null SD | null max |
|---|---|---|---|---|---|
| Pooled ROC-AUC (all held-out epochs) | 0.8624 | 2.00e-04 | 0.5989 | 0.0243 | 0.6764 |
| Mean per-fold balanced accuracy (fold-specific opt_threshold) | 0.7475 | 2.00e-04 | 0.4992 | 0.0266 | 0.5953 |
| Mean per-fold MCC (fold-specific opt_threshold) | 0.4854 | 2.00e-04 | -0.0016 | 0.0512 | 0.1803 |
| Mean per-fold ROC-AUC (descriptive) | 0.8790 | 2.00e-04 | 0.4988 | 0.0409 | 0.6537 |
