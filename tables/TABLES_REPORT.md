# INS-HDGS-CMT — Publication tables (LOSOCV)

**Framing:** the EEG-encoder comparison (Table 1) is the leakage-free headline. ET/fusion tables (2-3) are modality-information analyses; Labels are derived from eye-tracking gaze features; ET / fusion models partially observe the label-generating signal, so these scores are optimistically biased and are reported for modality-information analysis only, not as a leakage-free comparison.

### Table 1. EEG encoder comparison (LOSOCV; leakage-free headline). Proposed = INS-HDGS-CMT EEG branch (DynamicGAT + LIF).

| Model | Acc | BalAcc | Macro-F1 | MCC | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|---|---|
| **INS-HDGS-CMT (EEG-only)** | 0.704 ± 0.197 | 0.697 ± 0.163 | 0.657 ± 0.257 | 0.398 ± 0.312 | 0.819 ± 0.193 | 0.845 ± 0.184 | 0.371 ± 0.134 |
| ShallowConvNet | 0.604 ± 0.203 | 0.625 ± 0.208 | 0.527 ± 0.307 | 0.197 ± 0.391 | 0.659 ± 0.222 | 0.715 ± 0.285 | 0.492 ± 0.220 |
| EEGNet | 0.576 ± 0.189 | 0.597 ± 0.204 | 0.488 ± 0.251 | 0.184 ± 0.341 | 0.614 ± 0.248 | 0.688 ± 0.284 | 0.483 ± 0.185 |
| CNN-BiLSTM | 0.555 ± 0.217 | 0.593 ± 0.172 | 0.479 ± 0.308 | 0.128 ± 0.292 | 0.606 ± 0.256 | 0.675 ± 0.288 | 0.494 ± 0.246 |
| DeepConvNet | 0.561 ± 0.193 | 0.556 ± 0.208 | 0.461 ± 0.259 | 0.111 ± 0.315 | 0.575 ± 0.242 | 0.648 ± 0.290 | 0.477 ± 0.237 |
| TSception | 0.537 ± 0.165 | 0.543 ± 0.172 | 0.463 ± 0.259 | 0.096 ± 0.308 | 0.588 ± 0.246 | 0.644 ± 0.296 | 0.494 ± 0.253 |
| CNN-LSTM | 0.531 ± 0.192 | 0.531 ± 0.184 | 0.448 ± 0.281 | 0.064 ± 0.334 | 0.557 ± 0.271 | 0.638 ± 0.308 | 0.512 ± 0.241 |
| GAT | 0.531 ± 0.237 | 0.521 ± 0.187 | 0.403 ± 0.299 | -0.031 ± 0.236 | 0.513 ± 0.219 | 0.628 ± 0.268 | 0.318 ± 0.120 |
| EEG Transformer | 0.497 ± 0.172 | 0.517 ± 0.180 | 0.430 ± 0.264 | 0.042 ± 0.298 | 0.562 ± 0.213 | 0.622 ± 0.267 | 0.505 ± 0.255 |

**Leakage-free significance (INS-HDGS-CMT (EEG-only) vs each, Holm-corrected Wilcoxon):**

- *BalAcc*: Friedman p=1.55e-05; CD=1.98; sig. better than: EEG Transformer, TSception, GAT, CNN-LSTM, CNN-BiLSTM, EEGNet, DeepConvNet
- *MCC*: Friedman p=6.09e-06; CD=1.98; sig. better than: EEG Transformer, TSception, GAT, CNN-LSTM, CNN-BiLSTM, EEGNet, ShallowConvNet, DeepConvNet
- *ROC-AUC*: Friedman p=5.93e-07; CD=1.98; sig. better than: EEG Transformer, TSception, GAT, CNN-LSTM, CNN-BiLSTM, EEGNet, ShallowConvNet, DeepConvNet

### Table 2. Eye-tracking encoder comparison (LOSOCV).

> ⚠️ Labels are derived from eye-tracking gaze features; ET / fusion models partially observe the label-generating signal, so these scores are optimistically biased and are reported for modality-information analysis only, not as a leakage-free comparison.

| Model | Acc | BalAcc | Macro-F1 | MCC | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|---|---|
| ET-GRU | 0.802 ± 0.137 | 0.772 ± 0.176 | 0.685 ± 0.288 | 0.465 ± 0.358 | 0.842 ± 0.179 | 0.869 ± 0.235 | 0.506 ± 0.215 |
| ET-LSTM | 0.790 ± 0.161 | 0.756 ± 0.183 | 0.665 ± 0.302 | 0.435 ± 0.374 | 0.805 ± 0.199 | 0.845 ± 0.242 | 0.498 ± 0.214 |
| ET-Transformer | 0.756 ± 0.172 | 0.735 ± 0.190 | 0.648 ± 0.296 | 0.390 ± 0.373 | 0.804 ± 0.179 | 0.837 ± 0.250 | 0.514 ± 0.260 |

### Table 3. Multimodal fusion comparison (LOSOCV).

> ⚠️ Labels are derived from eye-tracking gaze features; ET / fusion models partially observe the label-generating signal, so these scores are optimistically biased and are reported for modality-information analysis only, not as a leakage-free comparison.

| Model | Acc | BalAcc | Macro-F1 | MCC | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|---|---|
| Cross-Attention | 0.791 ± 0.165 | 0.775 ± 0.184 | 0.700 ± 0.304 | 0.457 ± 0.383 | 0.805 ± 0.211 | 0.838 ± 0.250 | 0.516 ± 0.257 |
| Multimodal Transformer | 0.777 ± 0.161 | 0.758 ± 0.175 | 0.674 ± 0.299 | 0.424 ± 0.360 | 0.780 ± 0.210 | 0.803 ± 0.275 | 0.514 ± 0.268 |
| INS-HDGS-CMT w/o Neuro-Symbolic | 0.754 ± 0.180 | 0.742 ± 0.171 | 0.666 ± 0.297 | 0.483 ± 0.339 | 0.875 ± 0.177 | 0.902 ± 0.166 | 0.420 ± 0.132 |
| Dual Transformer | 0.753 ± 0.171 | 0.742 ± 0.188 | 0.689 ± 0.270 | 0.424 ± 0.379 | 0.765 ± 0.238 | 0.814 ± 0.268 | 0.516 ± 0.266 |
| **INS-HDGS-CMT (full)** | 0.753 ± 0.182 | 0.740 ± 0.186 | 0.688 ± 0.275 | 0.463 ± 0.364 | 0.901 ± 0.136 | 0.909 ± 0.160 | 0.418 ± 0.143 |
| DynamicGAT+ET Transformer | 0.747 ± 0.169 | 0.726 ± 0.174 | 0.634 ± 0.290 | 0.405 ± 0.341 | 0.764 ± 0.183 | 0.805 ± 0.249 | 0.519 ± 0.255 |
| Late Fusion (CNN-LSTM+ET-LSTM) | 0.677 ± 0.184 | 0.665 ± 0.188 | 0.574 ± 0.310 | 0.271 ± 0.375 | 0.668 ± 0.256 | 0.731 ± 0.280 | 0.497 ± 0.234 |

