# Cross-modal contribution (paired, fold-matched LOSOCV, n=37 subjects)

Variant definitions: full = EEG+ET+ROI+fusion; no_et = ET sequence branch and fusion transformer removed (ROI dwell vector still supplied — this is the configuration tabulated as the 'EEG branch' in Table 3); no_roi = ROI gating/modulation removed (ET sequence kept); no_fusion = cross-modal transformer replaced by simple cross-attention; eeg_only = no gaze-derived input at all (AblationConfig.eeg_only()); et_only = ET-LSTM baseline (argmax operating point).

## Per-variant means ± SD

| variant | balanced_acc | roc_auc | mcc |
|---|---|---|---|
| full | 0.740 ± 0.189 | 0.901 ± 0.138 | 0.463 ± 0.369 |
| no_et | 0.697 ± 0.165 | 0.819 ± 0.196 | 0.398 ± 0.316 |
| no_roi | 0.714 ± 0.169 | 0.864 ± 0.217 | 0.426 ± 0.328 |
| no_fusion | 0.714 ± 0.182 | 0.891 ± 0.156 | 0.411 ± 0.357 |
| et_only(ET-LSTM) | 0.739 ± 0.188 | 0.847 ± 0.178 | 0.493 ± 0.366 |

## Paired comparisons (Wilcoxon signed-rank, Holm within metric family)

| metric | A | B | mean Δ (A−B) | median Δ | 95% CI | p | p(Holm) | Cliff's δ | W/T/L |
|---|---|---|---|---|---|---|---|---|---|
| balanced_acc | full | no_et | +0.043 | +0.000 | [-0.008, +0.097] | 0.1317 | 0.7900 | +0.14 | 14/14/9 |
| balanced_acc | full | no_roi | +0.026 | +0.000 | [-0.012, +0.066] | 0.2012 | 1.0000 | +0.08 | 9/22/6 |
| balanced_acc | full | no_fusion | +0.025 | +0.000 | [-0.018, +0.072] | 0.3940 | 1.0000 | +0.08 | 12/16/9 |
| balanced_acc | no_et | no_roi | -0.017 | +0.000 | [-0.064, +0.027] | 0.5592 | 1.0000 | -0.03 | 9/18/10 |
| balanced_acc | full | et_only(ET-LSTM) | +0.001 | +0.000 | [-0.065, +0.065] | 0.7467 | 1.0000 | +0.03 | 13/12/12 |
| balanced_acc | no_et | et_only(ET-LSTM) | -0.042 | +0.000 | [-0.107, +0.024] | 0.2652 | 1.0000 | -0.11 | 14/5/18 |
| roc_auc | full | no_et | +0.083 | +0.062 | [+0.020, +0.146] | 0.0009 | 0.0055 * | +0.41 | 19/14/4 |
| roc_auc | full | no_roi | +0.038 | +0.000 | [-0.003, +0.095] | 0.2718 | 0.8155 | +0.05 | 8/23/6 |
| roc_auc | full | no_fusion | +0.010 | +0.000 | [-0.006, +0.032] | 0.4236 | 0.8472 | +0.08 | 7/26/4 |
| roc_auc | no_et | no_roi | -0.045 | +0.000 | [-0.104, +0.024] | 0.0170 | 0.0679 | -0.38 | 4/15/18 |
| roc_auc | full | et_only(ET-LSTM) | +0.055 | +0.000 | [-0.004, +0.107] | 0.0093 | 0.0464 * | +0.35 | 18/14/5 |
| roc_auc | no_et | et_only(ET-LSTM) | -0.028 | +0.000 | [-0.095, +0.031] | 0.6578 | 0.8472 | +0.05 | 13/13/11 |
| mcc | full | no_et | +0.065 | +0.000 | [-0.038, +0.172] | 0.2015 | 1.0000 | +0.14 | 14/14/9 |
| mcc | full | no_roi | +0.037 | +0.000 | [-0.037, +0.111] | 0.2560 | 1.0000 | +0.08 | 9/22/6 |
| mcc | full | no_fusion | +0.052 | +0.000 | [-0.040, +0.147] | 0.3392 | 1.0000 | +0.08 | 12/16/9 |
| mcc | no_et | no_roi | -0.028 | +0.000 | [-0.121, +0.059] | 0.5883 | 1.0000 | +0.00 | 10/17/10 |
| mcc | full | et_only(ET-LSTM) | -0.030 | +0.000 | [-0.153, +0.088] | 0.8936 | 1.0000 | -0.05 | 14/7/16 |
| mcc | no_et | et_only(ET-LSTM) | -0.096 | +0.000 | [-0.221, +0.028] | 0.1499 | 0.8995 | -0.11 | 14/5/18 |
