# Cross-modal contribution (paired, fold-matched LOSOCV, n=42 subjects)

Variant definitions: full = EEG+ET+ROI+fusion; no_et = ET sequence branch and fusion transformer removed (ROI dwell vector still supplied — this is the configuration tabulated as the 'EEG branch' in Table 3); no_roi = ROI gating/modulation removed (ET sequence kept); no_fusion = cross-modal transformer replaced by simple cross-attention; eeg_only = no gaze-derived input at all (AblationConfig.eeg_only()); et_only = ET-LSTM baseline (argmax operating point).

## Per-variant means ± SD

| variant | balanced_acc | roc_auc | mcc |
|---|---|---|---|
| full | 0.887 ± 0.135 | 0.989 ± 0.016 | 0.795 ± 0.240 |
| eeg_only | 0.721 ± 0.121 | 0.882 ± 0.106 | 0.484 ± 0.225 |

## Paired comparisons (Wilcoxon signed-rank, Holm within metric family)

| metric | A | B | mean Δ (A−B) | median Δ | 95% CI | p | p(Holm) | Cliff's δ | W/T/L |
|---|---|---|---|---|---|---|---|---|---|
| balanced_acc | full | eeg_only | +0.167 | +0.208 | [+0.109, +0.220] | 0.0000 | 0.0000 * | +0.79 | 37/1/4 |
| roc_auc | full | eeg_only | +0.106 | +0.075 | [+0.077, +0.138] | 0.0000 | 0.0000 * | +0.90 | 40/0/2 |
| mcc | full | eeg_only | +0.311 | +0.381 | [+0.206, +0.408] | 0.0000 | 0.0000 * | +0.79 | 37/1/4 |
