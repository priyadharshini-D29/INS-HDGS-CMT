### Table 2. Eye-tracking encoder comparison (LOSOCV).

> ⚠️ Labels are derived from eye-tracking gaze features; ET / fusion models partially observe the label-generating signal, so these scores are optimistically biased and are reported for modality-information analysis only, not as a leakage-free comparison.

| Model | Acc | BalAcc | Macro-F1 | MCC | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|---|---|
| ET-GRU | 0.802 ± 0.137 | 0.772 ± 0.176 | 0.685 ± 0.288 | 0.465 ± 0.358 | 0.842 ± 0.179 | 0.869 ± 0.235 | 0.506 ± 0.215 |
| ET-LSTM | 0.790 ± 0.161 | 0.756 ± 0.183 | 0.665 ± 0.302 | 0.435 ± 0.374 | 0.805 ± 0.199 | 0.845 ± 0.242 | 0.498 ± 0.214 |
| ET-Transformer | 0.756 ± 0.172 | 0.735 ± 0.190 | 0.648 ± 0.296 | 0.390 ± 0.373 | 0.804 ± 0.179 | 0.837 ± 0.250 | 0.514 ± 0.260 |
