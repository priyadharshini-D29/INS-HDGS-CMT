# Cross-modal contribution (paired, fold-matched LOSOCV, n=42 subjects)

Variant definitions: full = EEG+ET+ROI+fusion; no_et = ET sequence branch and fusion transformer removed (ROI dwell vector still supplied — this is the configuration tabulated as the 'EEG branch' in Table 3); no_roi = ROI gating/modulation removed (ET sequence kept); no_fusion = cross-modal transformer replaced by simple cross-attention; eeg_only = no gaze-derived input at all (AblationConfig.eeg_only()); et_only = ET-LSTM baseline (argmax operating point).

## Per-variant means ± SD

| variant | balanced_acc | roc_auc | mcc |
|---|---|---|---|
| full | 0.566 ± 0.091 | 0.606 ± 0.118 | 0.112 ± 0.151 |

## Paired comparisons (Wilcoxon signed-rank, Holm within metric family)

| metric | A | B | mean Δ (A−B) | median Δ | 95% CI | p | p(Holm) | Cliff's δ | W/T/L |
|---|---|---|---|---|---|---|---|---|---|
