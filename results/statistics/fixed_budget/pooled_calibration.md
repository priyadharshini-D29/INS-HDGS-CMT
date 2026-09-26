# Pooled out-of-fold calibration (saved held-out predictions, uncalibrated P(HIGH))

Proposed-model runs from their per-fold y_true / y_prob lists; tuned baselines from `results/baselines/dl/fold_probs/probs_<model>.csv` (p1). ECE = binary reliability ECE (mean predicted probability vs empirical positive frequency per equal-width bin), computed once over all held-out epochs pooled (10 and 15 bins); Brier = mean (p - y)^2. "Mean per-fold ECE (corrected)" averages the same 10-bin reliability ECE over the ~9-epoch folds and is descriptive only. "Stored" is the mean of the per-fold ECE column saved by the run (earlier formula: bin confidence vs thresholded accuracy), i.e. the number printed in the manuscript Tables 4-6.

| Model | n epochs | Pooled ECE (10 bins) | Pooled ECE (15 bins) | Brier | Mean per-fold ECE (corrected) | Stored mean per-fold ECE (old formula) |
|---|---|---|---|---|---|---|
| INS-HDGS-CMT (full), published run | 347 | 0.161 | 0.171 | 0.173 | 0.279 | 0.418 |
| INS-HDGS-CMT (full), revision re-run (abl_full) | 347 | 0.139 | 0.155 | 0.178 | 0.273 | 0.412 |
| INS-HDGS-CMT (EEG-only, no gaze input) | 347 | 0.052 | 0.032 | 0.245 | 0.231 | 0.243 |
| INS-HDGS-CMT (EEG-only, MMD/DANN kept) | 347 | 0.046 | 0.032 | 0.246 | 0.231 | 0.243 |
| ShallowConvNet | 385 | 0.300 | 0.300 | 0.332 | 0.380 | 0.492 |
| DeepConvNet | 385 | 0.371 | 0.372 | 0.410 | 0.406 | 0.477 |
| CNN-BiLSTM | 385 | 0.374 | 0.375 | 0.393 | 0.424 | 0.494 |
| CNN-LSTM | 385 | 0.426 | 0.424 | 0.435 | 0.450 | 0.512 |
| EEG Transformer | 385 | 0.464 | 0.464 | 0.468 | 0.502 | 0.505 |
| TSception | 385 | 0.396 | 0.400 | 0.421 | 0.433 | 0.494 |
| GAT | 385 | 0.078 | 0.080 | 0.259 | 0.313 | 0.318 |
| EEGNet | 385 | 0.299 | 0.299 | 0.346 | 0.405 | 0.483 |
| BrainGCN | 385 | 0.085 | 0.081 | 0.270 | 0.305 | 0.313 |
| ET-Transformer | 385 | 0.232 | 0.231 | 0.235 | 0.244 | 0.514 |
| ET-LSTM | 385 | 0.090 | 0.093 | 0.163 | 0.235 | 0.498 |
| ET-GRU | 385 | 0.056 | 0.079 | 0.145 | 0.233 | 0.506 |
| Cross-Attention | 385 | 0.199 | 0.202 | 0.206 | 0.211 | 0.516 |
| Multimodal Transformer | 385 | 0.213 | 0.215 | 0.215 | 0.216 | 0.514 |
| DynamicGAT + ET Transformer | 385 | 0.224 | 0.224 | 0.225 | 0.244 | 0.519 |
| Dual Transformer | 385 | 0.226 | 0.228 | 0.226 | 0.242 | 0.516 |
| Early-Fusion MLP | 385 | 0.175 | 0.177 | 0.201 | 0.240 | 0.508 |
| Late Fusion (CNN-LSTM + ET-LSTM) | 385 | 0.242 | 0.247 | 0.279 | 0.323 | 0.497 |

Pooled prevalence of HIGH over the held-out epochs: 0.493. A reference predictor that outputs the prevalence for every epoch has Brier = 0.250.
