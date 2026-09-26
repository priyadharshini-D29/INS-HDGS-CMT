# Paired effect sizes (fold-matched LOSOCV, saved per-fold CSVs)

d = A - B per test subject. r_rb = (W+ - W-)/(W+ + W-) from the Wilcoxon signed-rank statistic with zero differences dropped. P(sup) = P(d>0) - P(d<0) over all pairs incl. ties = the 'Cliff's delta of the paired differences' of Table S14. p (zeros discarded) = two-sided Wilcoxon signed-rank, scipy zero_method='wilcox'; p (zsplit) = the convention of verify_table7_eeg_significance.py / table_s10_full_vs_baselines.py. 'Cliff's delta (between)' = delta between the two per-fold distributions over all n x n pairs, the definition of Tables 9 / S6 / S10. 95% CI = subject-level bootstrap of the mean difference (10000 resamples, seed 0). Balanced accuracy and MCC are at the uncalibrated operating point stored in each CSV (threshold transferred from the validation subject; ET-LSTM at argmax 0.5, as in S14).

## (a) Gaze-free EEG branch (abl_eeg_only) vs tuned EEG baselines -- Table 9 (balanced accuracy) and Table S6 (MCC, ROC-AUC)

Existing values from results/statistics/table7_eeg_significance_tuned_<metric>.csv (between-distribution Cliff's delta; Wilcoxon zsplit p; Holm over the 8 baselines).

| Metric | A | B | n | mean A | mean B | mean d [95% CI] | median d | W/T/L | r_rb | P(sup) (paired delta) | p (zeros discarded) | p (zsplit) | Cliff's delta (between) recomputed | existing Cliff's delta on file | existing p | existing p_Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Balanced accuracy | abl_eeg_only | EEGNet | 37 | 0.552 | 0.477 | +0.075 [-0.004, +0.157] | +0.050 | 19/3/15 | +0.33 | +0.11 | 0.097 | 0.120 | +0.19 | +0.19 | 0.120 | 0.891 |
| Balanced accuracy | abl_eeg_only | GAT | 37 | 0.552 | 0.499 | +0.054 [-0.019, +0.123] | +0.083 | 22/3/12 | +0.30 | +0.27 | 0.122 | 0.111 | +0.14 | +0.14 | 0.111 | 0.891 |
| Balanced accuracy | abl_eeg_only | EEG Transformer | 37 | 0.552 | 0.519 | +0.034 [-0.054, +0.117] | +0.000 | 17/5/15 | +0.20 | +0.05 | 0.322 | 0.381 | +0.09 | +0.09 | 0.381 | 1.000 |
| Balanced accuracy | abl_eeg_only | CNN-LSTM | 37 | 0.552 | 0.516 | +0.037 [-0.037, +0.111] | +0.000 | 18/3/16 | +0.16 | +0.05 | 0.402 | 0.433 | +0.09 | +0.09 | 0.433 | 1.000 |
| Balanced accuracy | abl_eeg_only | TSception | 37 | 0.552 | 0.522 | +0.031 [-0.045, +0.105] | +0.042 | 19/3/15 | +0.14 | +0.11 | 0.462 | 0.460 | +0.04 | +0.04 | 0.460 | 1.000 |
| Balanced accuracy | abl_eeg_only | CNN-BiLSTM | 37 | 0.552 | 0.572 | -0.019 [-0.095, +0.055] | +0.000 | 13/8/16 | -0.15 | -0.08 | 0.482 | 0.501 | -0.12 | -0.12 | 0.501 | 1.000 |
| Balanced accuracy | abl_eeg_only | DeepConvNet | 37 | 0.552 | 0.585 | -0.033 [-0.130, +0.065] | -0.050 | 11/4/22 | -0.19 | -0.30 | 0.344 | 0.258 | -0.16 | -0.16 | 0.258 | 1.000 |
| Balanced accuracy | abl_eeg_only | ShallowConvNet | 37 | 0.552 | 0.590 | -0.037 [-0.123, +0.046] | -0.062 | 14/3/20 | -0.17 | -0.16 | 0.378 | 0.361 | -0.18 | -0.18 | 0.361 | 1.000 |
| MCC | abl_eeg_only | EEGNet | 37 | 0.094 | -0.036 | +0.130 [-0.020, +0.283] | +0.091 | 20/3/14 | +0.28 | +0.16 | 0.158 | 0.167 | +0.17 | +0.17 | 0.167 | 1.000 |
| MCC | abl_eeg_only | GAT | 37 | 0.094 | 0.009 | +0.085 [-0.056, +0.221] | +0.140 | 22/2/13 | +0.23 | +0.24 | 0.242 | 0.225 | +0.12 | +0.12 | 0.225 | 1.000 |
| MCC | abl_eeg_only | EEG Transformer | 37 | 0.094 | 0.017 | +0.077 [-0.097, +0.244] | +0.000 | 18/4/15 | +0.20 | +0.08 | 0.321 | 0.353 | +0.10 | +0.10 | 0.353 | 1.000 |
| MCC | abl_eeg_only | CNN-LSTM | 37 | 0.094 | 0.041 | +0.053 [-0.106, +0.216] | +0.000 | 18/3/16 | +0.09 | +0.05 | 0.638 | 0.645 | +0.06 | +0.06 | 0.645 | 1.000 |
| MCC | abl_eeg_only | TSception | 37 | 0.094 | 0.021 | +0.073 [-0.076, +0.218] | +0.091 | 19/2/16 | +0.15 | +0.08 | 0.441 | 0.451 | +0.06 | +0.06 | 0.451 | 1.000 |
| MCC | abl_eeg_only | CNN-BiLSTM | 37 | 0.094 | 0.133 | -0.039 [-0.192, +0.113] | +0.000 | 14/7/16 | -0.09 | -0.05 | 0.651 | 0.661 | -0.13 | -0.13 | 0.661 | 1.000 |
| MCC | abl_eeg_only | DeepConvNet | 37 | 0.094 | 0.154 | -0.060 [-0.251, +0.133] | -0.091 | 12/2/23 | -0.19 | -0.30 | 0.326 | 0.284 | -0.14 | -0.14 | 0.284 | 1.000 |
| MCC | abl_eeg_only | ShallowConvNet | 37 | 0.094 | 0.159 | -0.065 [-0.233, +0.100] | -0.125 | 14/3/20 | -0.14 | -0.16 | 0.462 | 0.433 | -0.17 | -0.17 | 0.433 | 1.000 |
| ROC-AUC | abl_eeg_only | EEGNet | 37 | 0.592 | 0.519 | +0.072 [-0.037, +0.177] | +0.125 | 24/1/12 | +0.26 | +0.32 | 0.167 | 0.156 | +0.17 | +0.17 | 0.156 | 1.000 |
| ROC-AUC | abl_eeg_only | GAT | 37 | 0.592 | 0.494 | +0.097 [+0.007, +0.187] | +0.000 | 18/4/15 | +0.38 | +0.08 | 0.055 | 0.087 | +0.22 | +0.22 | 0.087 | 0.694 |
| ROC-AUC | abl_eeg_only | EEG Transformer | 37 | 0.592 | 0.594 | -0.003 [-0.115, +0.106] | +0.000 | 16/5/16 | +0.01 | +0.00 | 0.955 | 0.964 | -0.01 | -0.01 | 0.964 | 1.000 |
| ROC-AUC | abl_eeg_only | CNN-LSTM | 37 | 0.592 | 0.566 | +0.026 [-0.103, +0.152] | +0.000 | 16/3/18 | +0.06 | -0.05 | 0.745 | 0.809 | +0.06 | +0.06 | 0.809 | 1.000 |
| ROC-AUC | abl_eeg_only | TSception | 37 | 0.592 | 0.562 | +0.029 [-0.090, +0.143] | +0.000 | 19/3/15 | +0.12 | +0.11 | 0.527 | 0.516 | +0.08 | +0.08 | 0.516 | 1.000 |
| ROC-AUC | abl_eeg_only | CNN-BiLSTM | 37 | 0.592 | 0.642 | -0.051 [-0.172, +0.067] | +0.000 | 16/3/18 | -0.13 | -0.05 | 0.494 | 0.516 | -0.13 | -0.13 | 0.516 | 1.000 |
| ROC-AUC | abl_eeg_only | DeepConvNet | 37 | 0.592 | 0.633 | -0.041 [-0.176, +0.084] | +0.000 | 16/3/18 | -0.04 | -0.05 | 0.844 | 0.827 | -0.09 | -0.09 | 0.827 | 1.000 |
| ROC-AUC | abl_eeg_only | ShallowConvNet | 37 | 0.592 | 0.618 | -0.026 [-0.152, +0.094] | +0.000 | 18/1/18 | -0.09 | +0.00 | 0.654 | 0.667 | -0.06 | -0.06 | 0.667 | 1.000 |

## (b) Full model (revision re-run abl_full, the headline run) vs each of the 18 tuned baselines -- Table S10 (S10 rows first, then the EEG encoders in Table 9 order and BrainGCN)

Existing values from results/statistics/tableS10_full_vs_baselines.csv (between-distribution Cliff's delta; Wilcoxon zsplit p; Holm over all 18 baselines within each metric).

| Metric | A | B | n | mean A | mean B | mean d [95% CI] | median d | W/T/L | r_rb | P(sup) (paired delta) | p (zeros discarded) | p (zsplit) | Cliff's delta (between) recomputed | existing Cliff's delta on file | existing p | existing p_Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Balanced accuracy | abl_full | ET-GRU | 37 | 0.747 | 0.754 | -0.006 [-0.063, +0.047] | +0.000 | 14/13/10 | -0.01 | +0.11 | 0.954 | 0.716 | -0.01 | -0.01 | 0.716 | 1.000 |
| Balanced accuracy | abl_full | ET-LSTM | 37 | 0.747 | 0.756 | -0.009 [-0.073, +0.050] | +0.000 | 14/15/8 | -0.07 | +0.16 | 0.770 | 0.584 | -0.04 | -0.04 | 0.584 | 1.000 |
| Balanced accuracy | abl_full | ET-Transformer | 37 | 0.747 | 0.772 | -0.025 [-0.092, +0.039] | +0.000 | 14/10/13 | -0.13 | +0.03 | 0.548 | 0.762 | -0.09 | -0.09 | 0.762 | 1.000 |
| Balanced accuracy | abl_full | Early-Fusion MLP | 37 | 0.747 | 0.729 | +0.019 [-0.054, +0.091] | +0.000 | 16/9/12 | +0.12 | +0.11 | 0.592 | 0.531 | +0.05 | +0.05 | 0.531 | 1.000 |
| Balanced accuracy | abl_full | Late Fusion | 37 | 0.747 | 0.727 | +0.021 [-0.043, +0.083] | +0.000 | 15/10/12 | +0.22 | +0.08 | 0.324 | 0.397 | +0.06 | +0.06 | 0.397 | 1.000 |
| Balanced accuracy | abl_full | Dual Transformer | 37 | 0.747 | 0.761 | -0.013 [-0.080, +0.049] | +0.000 | 14/12/11 | -0.04 | +0.08 | 0.861 | 0.862 | -0.07 | -0.07 | 0.862 | 1.000 |
| Balanced accuracy | abl_full | Cross-Attention | 37 | 0.747 | 0.774 | -0.026 [-0.092, +0.036] | +0.000 | 15/10/12 | -0.17 | +0.08 | 0.442 | 0.797 | -0.08 | -0.08 | 0.797 | 1.000 |
| Balanced accuracy | abl_full | Multimodal Transformer | 37 | 0.747 | 0.766 | -0.018 [-0.083, +0.039] | +0.042 | 19/9/9 | +0.02 | +0.27 | 0.927 | 0.459 | -0.05 | -0.05 | 0.459 | 1.000 |
| Balanced accuracy | abl_full | DynamicGAT+ET Transf. | 37 | 0.747 | 0.766 | -0.018 [-0.083, +0.041] | +0.000 | 13/13/11 | -0.07 | +0.05 | 0.753 | 0.976 | -0.08 | -0.08 | 0.976 | 1.000 |
| Balanced accuracy | abl_full | EEGNet | 37 | 0.747 | 0.477 | +0.270 [+0.193, +0.344] | +0.300 | 30/1/6 | +0.90 | +0.65 | 2.63e-06 | 2.70e-06 | +0.67 | +0.67 | 2.70e-06 | 4.59e-05 |
| Balanced accuracy | abl_full | GAT | 37 | 0.747 | 0.499 | +0.249 [+0.171, +0.323] | +0.250 | 31/2/4 | +0.87 | +0.73 | 7.74e-06 | 5.99e-06 | +0.67 | +0.67 | 5.99e-06 | 9.59e-05 |
| Balanced accuracy | abl_full | EEG Transformer | 37 | 0.747 | 0.519 | +0.229 [+0.140, +0.316] | +0.200 | 26/6/5 | +0.81 | +0.57 | 7.52e-05 | 6.33e-05 | +0.61 | +0.61 | 6.33e-05 | 8.23e-04 |
| Balanced accuracy | abl_full | CNN-LSTM | 37 | 0.747 | 0.516 | +0.232 [+0.156, +0.305] | +0.250 | 29/4/4 | +0.88 | +0.68 | 1.05e-05 | 7.60e-06 | +0.64 | +0.64 | 7.60e-06 | 1.14e-04 |
| Balanced accuracy | abl_full | TSception | 37 | 0.747 | 0.522 | +0.226 [+0.146, +0.301] | +0.292 | 28/3/6 | +0.83 | +0.59 | 2.41e-05 | 2.39e-05 | +0.59 | +0.59 | 2.39e-05 | 3.35e-04 |
| Balanced accuracy | abl_full | CNN-BiLSTM | 37 | 0.747 | 0.572 | +0.176 [+0.098, +0.250] | +0.200 | 27/6/4 | +0.70 | +0.62 | 6.21e-04 | 2.36e-04 | +0.50 | +0.50 | 2.36e-04 | 0.003 |
| Balanced accuracy | abl_full | DeepConvNet | 37 | 0.747 | 0.585 | +0.162 [+0.061, +0.259] | +0.200 | 25/3/9 | +0.55 | +0.43 | 0.005 | 0.004 | +0.42 | +0.42 | 0.004 | 0.044 |
| Balanced accuracy | abl_full | ShallowConvNet | 37 | 0.747 | 0.590 | +0.158 [+0.070, +0.243] | +0.100 | 23/6/8 | +0.62 | +0.41 | 0.002 | 0.003 | +0.42 | +0.42 | 0.003 | 0.029 |
| Balanced accuracy | abl_full | BrainGCN | 37 | 0.747 | 0.473 | +0.275 [+0.200, +0.345] | +0.333 | 32/2/3 | +0.91 | +0.78 | 2.33e-06 | 1.70e-06 | +0.77 | +0.77 | 1.70e-06 | 3.05e-05 |
| MCC | abl_full | ET-GRU | 37 | 0.485 | 0.514 | -0.028 [-0.134, +0.073] | +0.000 | 13/13/11 | -0.11 | +0.05 | 0.648 | 0.964 | -0.05 | -0.05 | 0.964 | 1.000 |
| MCC | abl_full | ET-LSTM | 37 | 0.485 | 0.517 | -0.032 [-0.157, +0.083] | +0.000 | 12/15/10 | -0.12 | +0.05 | 0.615 | 0.994 | -0.08 | -0.08 | 0.994 | 1.000 |
| MCC | abl_full | ET-Transformer | 37 | 0.485 | 0.536 | -0.050 [-0.185, +0.076] | +0.000 | 14/10/13 | -0.14 | +0.03 | 0.517 | 0.739 | -0.08 | -0.08 | 0.739 | 1.000 |
| MCC | abl_full | Early-Fusion MLP | 37 | 0.485 | 0.456 | +0.030 [-0.118, +0.176] | +0.000 | 16/9/12 | +0.07 | +0.11 | 0.750 | 0.629 | +0.03 | +0.03 | 0.629 | 1.000 |
| MCC | abl_full | Late Fusion | 37 | 0.485 | 0.454 | +0.031 [-0.096, +0.153] | +0.000 | 15/10/12 | +0.11 | +0.08 | 0.614 | 0.586 | +0.02 | +0.02 | 0.586 | 1.000 |
| MCC | abl_full | Dual Transformer | 37 | 0.485 | 0.539 | -0.053 [-0.183, +0.069] | +0.000 | 12/12/13 | -0.13 | -0.03 | 0.563 | 0.677 | -0.12 | -0.12 | 0.677 | 1.000 |
| MCC | abl_full | Cross-Attention | 37 | 0.485 | 0.542 | -0.056 [-0.185, +0.068] | +0.000 | 15/10/12 | -0.21 | +0.08 | 0.349 | 0.717 | -0.10 | -0.10 | 0.717 | 1.000 |
| MCC | abl_full | Multimodal Transformer | 37 | 0.485 | 0.528 | -0.042 [-0.169, +0.073] | +0.000 | 17/9/11 | -0.04 | +0.16 | 0.855 | 0.774 | -0.07 | -0.07 | 0.774 | 1.000 |
| MCC | abl_full | DynamicGAT+ET Transf. | 37 | 0.485 | 0.544 | -0.059 [-0.181, +0.056] | +0.000 | 12/13/12 | -0.17 | +0.00 | 0.458 | 0.693 | -0.11 | -0.11 | 0.693 | 1.000 |
| MCC | abl_full | EEGNet | 37 | 0.485 | -0.036 | +0.521 [+0.372, +0.663] | +0.600 | 30/1/6 | +0.89 | +0.65 | 3.07e-06 | 3.14e-06 | +0.68 | +0.68 | 3.14e-06 | 5.33e-05 |
| MCC | abl_full | GAT | 37 | 0.485 | 0.009 | +0.477 [+0.322, +0.625] | +0.548 | 31/2/4 | +0.85 | +0.73 | 1.22e-05 | 9.18e-06 | +0.65 | +0.65 | 9.18e-06 | 1.47e-04 |
| MCC | abl_full | EEG Transformer | 37 | 0.485 | 0.017 | +0.469 [+0.296, +0.639] | +0.400 | 26/6/5 | +0.85 | +0.57 | 3.40e-05 | 3.42e-05 | +0.61 | +0.61 | 3.42e-05 | 4.45e-04 |
| MCC | abl_full | CNN-LSTM | 37 | 0.485 | 0.041 | +0.445 [+0.294, +0.589] | +0.517 | 29/4/4 | +0.85 | +0.68 | 2.19e-05 | 1.43e-05 | +0.62 | +0.62 | 1.43e-05 | 2.02e-04 |
| MCC | abl_full | TSception | 37 | 0.485 | 0.021 | +0.464 [+0.315, +0.606] | +0.516 | 28/3/6 | +0.86 | +0.59 | 1.25e-05 | 1.34e-05 | +0.62 | +0.62 | 1.34e-05 | 2.02e-04 |
| MCC | abl_full | CNN-BiLSTM | 37 | 0.485 | 0.133 | +0.352 [+0.196, +0.500] | +0.403 | 27/6/4 | +0.68 | +0.62 | 9.23e-04 | 3.27e-04 | +0.52 | +0.52 | 3.27e-04 | 0.004 |
| MCC | abl_full | DeepConvNet | 37 | 0.485 | 0.154 | +0.332 [+0.139, +0.517] | +0.446 | 25/3/9 | +0.60 | +0.43 | 0.002 | 0.002 | +0.46 | +0.46 | 0.002 | 0.022 |
| MCC | abl_full | ShallowConvNet | 37 | 0.485 | 0.159 | +0.326 [+0.153, +0.495] | +0.270 | 24/6/7 | +0.64 | +0.46 | 0.002 | 0.001 | +0.42 | +0.42 | 0.001 | 0.016 |
| MCC | abl_full | BrainGCN | 37 | 0.485 | -0.073 | +0.558 [+0.407, +0.701] | +0.640 | 33/2/2 | +0.91 | +0.84 | 2.91e-06 | 1.79e-06 | +0.77 | +0.77 | 1.79e-06 | 3.23e-05 |
| ROC-AUC | abl_full | ET-GRU | 37 | 0.879 | 0.876 | +0.003 [-0.034, +0.037] | +0.000 | 12/18/7 | +0.18 | +0.14 | 0.494 | 0.343 | +0.05 | +0.05 | 0.343 | 1.000 |
| ROC-AUC | abl_full | ET-LSTM | 37 | 0.879 | 0.890 | -0.011 [-0.048, +0.022] | +0.000 | 10/18/9 | -0.05 | +0.03 | 0.856 | 0.945 | -0.01 | -0.01 | 0.945 | 1.000 |
| ROC-AUC | abl_full | ET-Transformer | 37 | 0.879 | 0.877 | +0.002 [-0.033, +0.032] | +0.000 | 9/23/5 | +0.30 | +0.11 | 0.331 | 0.339 | +0.01 | +0.01 | 0.339 | 1.000 |
| ROC-AUC | abl_full | Early-Fusion MLP | 37 | 0.879 | 0.847 | +0.032 [-0.036, +0.097] | +0.029 | 19/9/9 | +0.29 | +0.27 | 0.175 | 0.114 | +0.13 | +0.13 | 0.114 | 0.845 |
| ROC-AUC | abl_full | Late Fusion | 37 | 0.879 | 0.835 | +0.044 [-0.012, +0.099] | +0.000 | 15/16/6 | +0.39 | +0.24 | 0.122 | 0.076 | +0.16 | +0.16 | 0.076 | 0.683 |
| ROC-AUC | abl_full | Dual Transformer | 37 | 0.879 | 0.863 | +0.016 [-0.032, +0.066] | +0.000 | 11/20/6 | +0.24 | +0.14 | 0.394 | 0.296 | +0.08 | +0.08 | 0.296 | 1.000 |
| ROC-AUC | abl_full | Cross-Attention | 37 | 0.879 | 0.867 | +0.012 [-0.030, +0.046] | +0.000 | 13/19/5 | +0.33 | +0.22 | 0.222 | 0.111 | +0.10 | +0.10 | 0.111 | 0.845 |
| ROC-AUC | abl_full | Multimodal Transformer | 37 | 0.879 | 0.844 | +0.035 [-0.016, +0.090] | +0.000 | 13/19/5 | +0.35 | +0.22 | 0.199 | 0.106 | +0.10 | +0.10 | 0.106 | 0.845 |
| ROC-AUC | abl_full | DynamicGAT+ET Transf. | 37 | 0.879 | 0.876 | +0.003 [-0.037, +0.038] | +0.000 | 12/18/7 | +0.21 | +0.14 | 0.421 | 0.320 | +0.03 | +0.03 | 0.320 | 1.000 |
| ROC-AUC | abl_full | EEGNet | 37 | 0.879 | 0.519 | +0.360 [+0.278, +0.440] | +0.364 | 33/2/2 | +0.97 | +0.84 | 5.84e-07 | 3.98e-07 | +0.75 | +0.75 | 3.98e-07 | 6.77e-06 |
| ROC-AUC | abl_full | GAT | 37 | 0.879 | 0.494 | +0.385 [+0.296, +0.475] | +0.400 | 32/3/2 | +0.99 | +0.81 | 5.21e-07 | 3.26e-07 | +0.78 | +0.78 | 3.26e-07 | 5.88e-06 |
| ROC-AUC | abl_full | EEG Transformer | 37 | 0.879 | 0.594 | +0.285 [+0.169, +0.398] | +0.333 | 27/5/5 | +0.76 | +0.59 | 1.90e-04 | 1.22e-04 | +0.62 | +0.62 | 1.22e-04 | 0.001 |
| ROC-AUC | abl_full | CNN-LSTM | 37 | 0.879 | 0.566 | +0.313 [+0.207, +0.422] | +0.333 | 27/5/5 | +0.87 | +0.59 | 1.93e-05 | 1.88e-05 | +0.62 | +0.62 | 1.88e-05 | 2.64e-04 |
| ROC-AUC | abl_full | TSception | 37 | 0.879 | 0.562 | +0.317 [+0.228, +0.403] | +0.312 | 31/3/3 | +0.91 | +0.76 | 4.06e-06 | 2.60e-06 | +0.73 | +0.73 | 2.60e-06 | 3.90e-05 |
| ROC-AUC | abl_full | CNN-BiLSTM | 37 | 0.879 | 0.642 | +0.237 [+0.126, +0.345] | +0.250 | 28/3/6 | +0.71 | +0.59 | 3.08e-04 | 2.31e-04 | +0.51 | +0.51 | 2.31e-04 | 0.002 |
| ROC-AUC | abl_full | DeepConvNet | 37 | 0.879 | 0.633 | +0.246 [+0.144, +0.346] | +0.216 | 29/5/3 | +0.76 | +0.70 | 1.77e-04 | 6.15e-05 | +0.57 | +0.57 | 6.15e-05 | 7.38e-04 |
| ROC-AUC | abl_full | ShallowConvNet | 37 | 0.879 | 0.618 | +0.261 [+0.167, +0.353] | +0.250 | 28/5/4 | +0.84 | +0.65 | 3.28e-05 | 2.07e-05 | +0.60 | +0.60 | 2.07e-05 | 2.69e-04 |
| ROC-AUC | abl_full | BrainGCN | 37 | 0.879 | 0.474 | +0.405 [+0.323, +0.484] | +0.400 | 35/0/2 | +0.95 | +0.89 | 5.24e-07 | 5.24e-07 | +0.86 | +0.86 | 5.24e-07 | 8.39e-06 |

## (c) Full model = revision re-run abl_full (Table 8 reference) vs pathway variants -- Table S14 layout

Existing values (published reference only) from results/statistics/cross_modal_contribution.csv, whose Cliff's delta is the PAIRED delta = P(sup) here; its Wilcoxon p discards zeros; Holm over the 8 pairs per metric of that script; its bootstrap CI used seed 42, this one seed 0.

| Metric | A | B | n | mean A | mean B | mean d [95% CI] | median d | W/T/L | r_rb | P(sup) (paired delta) | p (zeros discarded) | p (zsplit) | Cliff's delta (between) recomputed | existing Cliff's delta on file | existing p | existing p_Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Balanced accuracy | abl_full | -gaze seq. (abl_no_et) | 37 | 0.747 | 0.697 | +0.051 [+0.009, +0.092] | +0.000 | 15/16/6 | +0.52 | +0.24 | 0.035 | 0.043 | +0.15 | -- | -- | -- |
| Balanced accuracy | abl_full | -ROI (abl_no_roi) | 37 | 0.747 | 0.714 | +0.034 [-0.011, +0.075] | +0.000 | 13/20/4 | +0.50 | +0.24 | 0.068 | 0.048 | +0.11 | -- | -- | -- |
| Balanced accuracy | abl_full | -fusion (abl_no_fusion_transformer) | 37 | 0.747 | 0.714 | +0.033 [-0.002, +0.070] | +0.000 | 12/21/4 | +0.46 | +0.22 | 0.103 | 0.075 | +0.11 | -- | -- | -- |
| Balanced accuracy | abl_full | EEG-only (abl_eeg_only) | 37 | 0.747 | 0.552 | +0.195 [+0.128, +0.262] | +0.200 | 27/5/5 | +0.87 | +0.59 | 1.84e-05 | 1.81e-05 | +0.56 | -- | -- | -- |
| Balanced accuracy | abl_full | EEG-only, MMD/DANN kept (abl_eeg_only_mmd) | 37 | 0.747 | 0.547 | +0.201 [+0.132, +0.268] | +0.200 | 27/5/5 | +0.87 | +0.59 | 1.69e-05 | 1.70e-05 | +0.58 | -- | -- | -- |
| Balanced accuracy | abl_full | ET-LSTM tuned (argmax operating point) | 37 | 0.747 | 0.756 | -0.009 [-0.073, +0.050] | +0.000 | 16/13/8 | -0.01 | +0.22 | 0.966 | 0.444 | -0.04 | -- | -- | -- |
| MCC | abl_full | -gaze seq. (abl_no_et) | 37 | 0.485 | 0.398 | +0.088 [+0.002, +0.171] | +0.000 | 15/16/6 | +0.47 | +0.24 | 0.058 | 0.054 | +0.12 | -- | -- | -- |
| MCC | abl_full | -ROI (abl_no_roi) | 37 | 0.485 | 0.426 | +0.059 [-0.026, +0.135] | +0.000 | 13/20/4 | +0.56 | +0.24 | 0.044 | 0.042 | +0.10 | -- | -- | -- |
| MCC | abl_full | -fusion (abl_no_fusion_transformer) | 37 | 0.485 | 0.411 | +0.074 [+0.009, +0.145] | +0.000 | 12/21/4 | +0.54 | +0.22 | 0.056 | 0.062 | +0.14 | -- | -- | -- |
| MCC | abl_full | EEG-only (abl_eeg_only) | 37 | 0.485 | 0.094 | +0.391 [+0.248, +0.536] | +0.430 | 27/5/5 | +0.81 | +0.59 | 6.28e-05 | 4.91e-05 | +0.57 | -- | -- | -- |
| MCC | abl_full | EEG-only, MMD/DANN kept (abl_eeg_only_mmd) | 37 | 0.485 | 0.083 | +0.402 [+0.256, +0.548] | +0.471 | 27/5/5 | +0.81 | +0.59 | 5.80e-05 | 4.61e-05 | +0.59 | -- | -- | -- |
| MCC | abl_full | ET-LSTM tuned (argmax operating point) | 37 | 0.485 | 0.517 | -0.032 [-0.157, +0.083] | +0.000 | 12/15/10 | -0.12 | +0.05 | 0.615 | 0.994 | -0.08 | -- | -- | -- |
| ROC-AUC | abl_full | -gaze seq. (abl_no_et) | 37 | 0.879 | 0.819 | +0.060 [-0.008, +0.138] | +0.000 | 17/16/4 | +0.56 | +0.35 | 0.024 | 0.010 | +0.22 | -- | -- | -- |
| ROC-AUC | abl_full | -ROI (abl_no_roi) | 37 | 0.879 | 0.864 | +0.015 [-0.034, +0.069] | +0.000 | 9/22/6 | +0.24 | +0.08 | 0.410 | 0.462 | +0.02 | -- | -- | -- |
| ROC-AUC | abl_full | -fusion (abl_no_fusion_transformer) | 37 | 0.879 | 0.891 | -0.012 [-0.050, +0.020] | +0.000 | 9/21/7 | +0.10 | +0.05 | 0.717 | 0.666 | -0.02 | -- | -- | -- |
| ROC-AUC | abl_full | EEG-only (abl_eeg_only) | 37 | 0.879 | 0.592 | +0.287 [+0.209, +0.374] | +0.250 | 29/6/2 | +0.96 | +0.73 | 3.38e-06 | 1.58e-06 | +0.65 | -- | -- | -- |
| ROC-AUC | abl_full | EEG-only, MMD/DANN kept (abl_eeg_only_mmd) | 37 | 0.879 | 0.592 | +0.287 [+0.209, +0.374] | +0.250 | 29/6/2 | +0.96 | +0.73 | 3.38e-06 | 1.58e-06 | +0.65 | -- | -- | -- |
| ROC-AUC | abl_full | ET-LSTM tuned (argmax operating point) | 37 | 0.879 | 0.890 | -0.011 [-0.048, +0.022] | +0.000 | 10/18/9 | -0.05 | +0.03 | 0.856 | 0.945 | -0.01 | -- | -- | -- |

## (c) Full model = published run (Table S14 reference) vs pathway variants -- Table S14 layout

Existing values (published reference only) from results/statistics/cross_modal_contribution.csv, whose Cliff's delta is the PAIRED delta = P(sup) here; its Wilcoxon p discards zeros; Holm over the 8 pairs per metric of that script; its bootstrap CI used seed 42, this one seed 0.

| Metric | A | B | n | mean A | mean B | mean d [95% CI] | median d | W/T/L | r_rb | P(sup) (paired delta) | p (zeros discarded) | p (zsplit) | Cliff's delta (between) recomputed | existing Cliff's delta on file | existing p | existing p_Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Balanced accuracy | published full | -gaze seq. (abl_no_et) | 37 | 0.740 | 0.697 | +0.043 [-0.008, +0.095] | +0.000 | 14/14/9 | +0.36 | +0.14 | 0.132 | 0.199 | +0.13 | +0.24 | 0.035 | 0.210 |
| Balanced accuracy | published full | -ROI (abl_no_roi) | 37 | 0.740 | 0.714 | +0.026 [-0.012, +0.065] | +0.000 | 9/22/6 | +0.38 | +0.08 | 0.201 | 0.390 | +0.08 | +0.24 | 0.068 | 0.320 |
| Balanced accuracy | published full | -fusion (abl_no_fusion_transformer) | 37 | 0.740 | 0.714 | +0.025 [-0.018, +0.071] | +0.000 | 12/16/9 | +0.21 | +0.08 | 0.394 | 0.460 | +0.08 | +0.22 | 0.103 | 0.320 |
| Balanced accuracy | published full | EEG-only (abl_eeg_only) | 37 | 0.740 | 0.552 | +0.187 [+0.113, +0.261] | +0.182 | 26/6/5 | +0.79 | +0.57 | 1.22e-04 | 9.23e-05 | +0.54 | +0.59 | 1.84e-05 | 1.47e-04 |
| Balanced accuracy | published full | EEG-only, MMD/DANN kept (abl_eeg_only_mmd) | 37 | 0.740 | 0.547 | +0.193 [+0.117, +0.267] | +0.182 | 26/6/5 | +0.79 | +0.57 | 1.13e-04 | 8.67e-05 | +0.56 | -- | -- | -- |
| Balanced accuracy | published full | ET-LSTM tuned (argmax operating point) | 37 | 0.740 | 0.756 | -0.017 [-0.087, +0.051] | +0.000 | 15/8/14 | -0.08 | +0.03 | 0.697 | 0.833 | -0.06 | +0.22 | 0.966 | 1.000 |
| MCC | published full | -gaze seq. (abl_no_et) | 37 | 0.463 | 0.398 | +0.065 [-0.039, +0.168] | +0.000 | 14/14/9 | +0.30 | +0.14 | 0.201 | 0.242 | +0.11 | +0.24 | 0.058 | 0.223 |
| MCC | published full | -ROI (abl_no_roi) | 37 | 0.463 | 0.426 | +0.037 [-0.036, +0.109] | +0.000 | 9/22/6 | +0.33 | +0.08 | 0.256 | 0.412 | +0.07 | +0.24 | 0.044 | 0.221 |
| MCC | published full | -fusion (abl_no_fusion_transformer) | 37 | 0.463 | 0.411 | +0.052 [-0.038, +0.145] | +0.000 | 12/16/9 | +0.24 | +0.08 | 0.339 | 0.433 | +0.09 | +0.22 | 0.056 | 0.223 |
| MCC | published full | EEG-only (abl_eeg_only) | 37 | 0.463 | 0.094 | +0.369 [+0.217, +0.520] | +0.391 | 26/5/6 | +0.79 | +0.54 | 1.04e-04 | 1.02e-04 | +0.54 | +0.59 | 6.28e-05 | 5.02e-04 |
| MCC | published full | EEG-only, MMD/DANN kept (abl_eeg_only_mmd) | 37 | 0.463 | 0.083 | +0.380 [+0.226, +0.531] | +0.391 | 26/5/6 | +0.79 | +0.54 | 9.65e-05 | 9.57e-05 | +0.56 | -- | -- | -- |
| MCC | published full | ET-LSTM tuned (argmax operating point) | 37 | 0.463 | 0.517 | -0.054 [-0.186, +0.075] | +0.000 | 12/7/18 | -0.14 | -0.16 | 0.491 | 0.411 | -0.10 | +0.05 | 0.615 | 1.000 |
| ROC-AUC | published full | -gaze seq. (abl_no_et) | 37 | 0.901 | 0.819 | +0.083 [+0.021, +0.147] | +0.062 | 19/14/4 | +0.79 | +0.41 | 9.14e-04 | 0.001 | +0.29 | +0.35 | 0.024 | 0.119 |
| ROC-AUC | published full | -ROI (abl_no_roi) | 37 | 0.901 | 0.864 | +0.038 [-0.003, +0.094] | +0.000 | 8/23/6 | +0.33 | +0.05 | 0.272 | 0.529 | +0.05 | +0.08 | 0.410 | 1.000 |
| ROC-AUC | published full | -fusion (abl_no_fusion_transformer) | 37 | 0.901 | 0.891 | +0.010 [-0.007, +0.032] | +0.000 | 7/26/4 | +0.27 | +0.08 | 0.424 | 0.449 | +0.01 | +0.05 | 0.717 | 1.000 |
| ROC-AUC | published full | EEG-only (abl_eeg_only) | 37 | 0.901 | 0.592 | +0.310 [+0.235, +0.390] | +0.300 | 29/6/2 | +0.98 | +0.73 | 1.91e-06 | 1.00e-06 | +0.71 | +0.73 | 3.38e-06 | 2.71e-05 |
| ROC-AUC | published full | EEG-only, MMD/DANN kept (abl_eeg_only_mmd) | 37 | 0.901 | 0.592 | +0.310 [+0.235, +0.390] | +0.300 | 29/6/2 | +0.98 | +0.73 | 1.91e-06 | 1.00e-06 | +0.71 | -- | -- | -- |
| ROC-AUC | published full | ET-LSTM tuned (argmax operating point) | 37 | 0.901 | 0.890 | +0.011 [-0.019, +0.043] | +0.000 | 12/16/9 | +0.20 | +0.08 | 0.414 | 0.469 | +0.02 | +0.03 | 0.856 | 1.000 |

## (c, direct pair) abl_eeg_only_mmd vs abl_eeg_only

No existing table value.

| Metric | A | B | n | mean A | mean B | mean d [95% CI] | median d | W/T/L | r_rb | P(sup) (paired delta) | p (zeros discarded) | p (zsplit) | Cliff's delta (between) recomputed | existing Cliff's delta on file | existing p | existing p_Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Balanced accuracy | abl_eeg_only_mmd | abl_eeg_only | 37 | 0.547 | 0.552 | -0.005 [-0.016, +0.000] | +0.000 | 0/36/1 | -1.00 | -0.03 | 0.317 | 0.752 | -0.03 | -- | -- | -- |
| MCC | abl_eeg_only_mmd | abl_eeg_only | 37 | 0.083 | 0.094 | -0.011 [-0.032, +0.000] | +0.000 | 0/36/1 | -1.00 | -0.03 | 0.317 | 0.752 | -0.02 | -- | -- | -- |
| ROC-AUC | abl_eeg_only_mmd | abl_eeg_only | 37 | 0.592 | 0.592 | +0.000 [+0.000, +0.000] | +0.000 | 0/37/0 | -- | +0.00 | 1.000 | 1.000 | +0.00 | -- | -- | -- |

## Consistency with the values on file

- Balanced accuracy: published full vs -gaze seq. (abl_no_et) -- recomputed +0.135 vs on file +0.243
- Balanced accuracy: published full vs -ROI (abl_no_roi) -- recomputed +0.081 vs on file +0.243
- Balanced accuracy: published full vs -fusion (abl_no_fusion_transformer) -- recomputed +0.081 vs on file +0.216
- Balanced accuracy: published full vs EEG-only (abl_eeg_only) -- recomputed +0.568 vs on file +0.595
- Balanced accuracy: published full vs ET-LSTM tuned (argmax operating point) -- recomputed +0.027 vs on file +0.216
- MCC: published full vs -gaze seq. (abl_no_et) -- recomputed +0.135 vs on file +0.243
- MCC: published full vs -ROI (abl_no_roi) -- recomputed +0.081 vs on file +0.243
- MCC: published full vs -fusion (abl_no_fusion_transformer) -- recomputed +0.081 vs on file +0.216
- MCC: published full vs EEG-only (abl_eeg_only) -- recomputed +0.541 vs on file +0.595
- MCC: published full vs ET-LSTM tuned (argmax operating point) -- recomputed -0.162 vs on file +0.054
- ROC-AUC: published full vs -gaze seq. (abl_no_et) -- recomputed +0.405 vs on file +0.351
- ROC-AUC: published full vs -ROI (abl_no_roi) -- recomputed +0.054 vs on file +0.081
- ROC-AUC: published full vs -fusion (abl_no_fusion_transformer) -- recomputed +0.081 vs on file +0.054
- ROC-AUC: published full vs ET-LSTM tuned (argmax operating point) -- recomputed +0.081 vs on file +0.027
