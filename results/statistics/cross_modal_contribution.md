# Cross-modal contribution (paired, fold-matched LOSOCV, n=37 subjects)

Variant definitions: full = EEG+ET+ROI+fusion; no_et = ET sequence branch and fusion transformer removed (ROI dwell vector still supplied — this is the configuration tabulated as the 'EEG branch' in Table 3); no_roi = ROI gating/modulation removed (ET sequence kept); no_fusion = cross-modal transformer replaced by simple cross-attention; eeg_only = no gaze-derived input at all (AblationConfig.eeg_only()); et_only = ET-LSTM baseline (argmax operating point).

## Per-variant means ± SD

| variant | balanced_acc | roc_auc | mcc |
|---|---|---|---|
| full | 0.747 ± 0.185 | 0.879 ± 0.167 | 0.485 ± 0.359 |
| no_et | 0.697 ± 0.165 | 0.819 ± 0.196 | 0.398 ± 0.316 |
| no_roi | 0.714 ± 0.169 | 0.864 ± 0.217 | 0.426 ± 0.328 |
| no_fusion | 0.714 ± 0.182 | 0.891 ± 0.156 | 0.411 ± 0.357 |
| eeg_only | 0.552 ± 0.169 | 0.592 ± 0.263 | 0.094 ± 0.334 |
| et_only(ET-LSTM) | 0.756 ± 0.188 | 0.890 ± 0.145 | 0.517 ± 0.367 |

## Paired comparisons (Wilcoxon signed-rank, Holm within metric family)

| metric | A | B | mean Δ (A−B) | median Δ | 95% CI | p | p(Holm) | Cliff's δ | W/T/L |
|---|---|---|---|---|---|---|---|---|---|
| balanced_acc | full | no_et | +0.051 | +0.000 | [+0.009, +0.092] | 0.0351 | 0.2105 | +0.24 | 15/16/6 |
| balanced_acc | full | no_roi | +0.034 | +0.000 | [-0.012, +0.076] | 0.0683 | 0.3196 | +0.24 | 13/20/4 |
| balanced_acc | full | no_fusion | +0.033 | +0.000 | [-0.002, +0.069] | 0.1029 | 0.3196 | +0.22 | 12/21/4 |
| balanced_acc | no_et | no_roi | -0.017 | +0.000 | [-0.064, +0.027] | 0.5592 | 1.0000 | -0.03 | 9/18/10 |
| balanced_acc | full | eeg_only | +0.195 | +0.200 | [+0.129, +0.261] | 0.0000 | 0.0001 * | +0.59 | 27/5/5 |
| balanced_acc | no_et | eeg_only | +0.145 | +0.125 | [+0.081, +0.210] | 0.0004 | 0.0030 * | +0.54 | 25/7/5 |
| balanced_acc | full | et_only(ET-LSTM) | -0.009 | +0.000 | [-0.073, +0.052] | 0.9658 | 1.0000 | +0.22 | 16/13/8 |
| balanced_acc | no_et | et_only(ET-LSTM) | -0.059 | +0.000 | [-0.114, -0.004] | 0.0639 | 0.3196 | -0.16 | 12/7/18 |
| roc_auc | full | no_et | +0.060 | +0.000 | [-0.009, +0.138] | 0.0238 | 0.1192 | +0.35 | 17/16/4 |
| roc_auc | full | no_roi | +0.015 | +0.000 | [-0.034, +0.070] | 0.4101 | 1.0000 | +0.08 | 9/22/6 |
| roc_auc | full | no_fusion | -0.012 | +0.000 | [-0.050, +0.020] | 0.7173 | 1.0000 | +0.05 | 9/21/7 |
| roc_auc | no_et | no_roi | -0.045 | +0.000 | [-0.104, +0.024] | 0.0170 | 0.1019 | -0.38 | 4/15/18 |
| roc_auc | full | eeg_only | +0.287 | +0.250 | [+0.207, +0.374] | 0.0000 | 0.0000 * | +0.73 | 29/6/2 |
| roc_auc | no_et | eeg_only | +0.227 | +0.200 | [+0.148, +0.309] | 0.0000 | 0.0002 * | +0.57 | 27/4/6 |
| roc_auc | full | et_only(ET-LSTM) | -0.011 | +0.000 | [-0.049, +0.023] | 0.8563 | 1.0000 | +0.03 | 10/18/9 |
| roc_auc | no_et | et_only(ET-LSTM) | -0.071 | +0.000 | [-0.145, -0.008] | 0.0299 | 0.1195 | -0.27 | 7/13/17 |
| mcc | full | no_et | +0.088 | +0.000 | [-0.000, +0.172] | 0.0582 | 0.2229 | +0.24 | 15/16/6 |
| mcc | full | no_roi | +0.059 | +0.000 | [-0.027, +0.135] | 0.0442 | 0.2212 | +0.24 | 13/20/4 |
| mcc | full | no_fusion | +0.074 | +0.000 | [+0.009, +0.143] | 0.0557 | 0.2229 | +0.22 | 12/21/4 |
| mcc | no_et | no_roi | -0.028 | +0.000 | [-0.121, +0.059] | 0.5883 | 1.0000 | +0.00 | 10/17/10 |
| mcc | full | eeg_only | +0.391 | +0.430 | [+0.250, +0.535] | 0.0001 | 0.0005 * | +0.59 | 27/5/5 |
| mcc | no_et | eeg_only | +0.304 | +0.298 | [+0.167, +0.441] | 0.0005 | 0.0033 * | +0.54 | 25/7/5 |
| mcc | full | et_only(ET-LSTM) | -0.032 | +0.000 | [-0.155, +0.088] | 0.6148 | 1.0000 | +0.05 | 12/15/10 |
| mcc | no_et | et_only(ET-LSTM) | -0.119 | -0.085 | [-0.226, -0.014] | 0.0324 | 0.1945 | -0.22 | 11/7/19 |
