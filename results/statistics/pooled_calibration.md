# Pooled out-of-fold calibration (saved held-out predictions, uncalibrated P(HIGH))

Proposed-model runs from their per-fold y_true / y_prob lists; tuned baselines from `results/baselines/dl_tuned/fold_probs/probs_<model>.csv` (p1). ECE = binary reliability ECE (mean predicted probability vs empirical positive frequency per equal-width bin), computed once over all held-out epochs pooled (10 and 15 bins); Brier = mean (p - y)^2. "Mean per-fold ECE (corrected)" averages the same 10-bin reliability ECE over the ~9-epoch folds and is descriptive only. "Stored" is the mean of the per-fold ECE column saved by the run (earlier formula: bin confidence vs thresholded accuracy), i.e. the number printed in the manuscript Tables 4-6.

| Model | n epochs | Pooled ECE (10 bins) | Pooled ECE (15 bins) | Brier | Mean per-fold ECE (corrected) | Stored mean per-fold ECE (old formula) |
|---|---|---|---|---|---|---|
| INS-HDGS-CMT (full), published run | 347 | 0.161 | 0.171 | 0.173 | 0.279 | 0.418 |
| INS-HDGS-CMT (full), revision re-run (abl_full) | 347 | 0.139 | 0.155 | 0.178 | 0.273 | 0.412 |
| INS-HDGS-CMT (EEG-only, no gaze input) | 347 | 0.052 | 0.032 | 0.245 | 0.231 | 0.243 |
| INS-HDGS-CMT (EEG-only, MMD/DANN kept) | 347 | 0.046 | 0.032 | 0.246 | 0.231 | 0.243 |
| ShallowConvNet | 347 | 0.221 | 0.226 | 0.295 | 0.372 | 0.461 |
| DeepConvNet | 347 | 0.289 | 0.291 | 0.359 | 0.397 | 0.471 |
| CNN-BiLSTM | 347 | 0.202 | 0.208 | 0.317 | 0.374 | 0.425 |
| CNN-LSTM | 347 | 0.256 | 0.269 | 0.351 | 0.401 | 0.432 |
| EEG Transformer | 347 | 0.238 | 0.241 | 0.317 | 0.347 | 0.399 |
| TSception | 347 | 0.304 | 0.308 | 0.338 | 0.398 | 0.431 |
| GAT | 347 | 0.073 | 0.117 | 0.263 | 0.287 | 0.288 |
| EEGNet | 347 | 0.192 | 0.168 | 0.298 | 0.373 | 0.393 |
| BrainGCN | 347 | 0.074 | 0.065 | 0.257 | 0.259 | 0.268 |
| ET-Transformer | 347 | 0.071 | 0.082 | 0.156 | 0.235 | 0.512 |
| ET-LSTM | 347 | 0.098 | 0.073 | 0.164 | 0.271 | 0.455 |
| ET-GRU | 347 | 0.084 | 0.098 | 0.156 | 0.263 | 0.461 |
| Cross-Attention | 347 | 0.077 | 0.082 | 0.153 | 0.238 | 0.515 |
| Multimodal Transformer | 347 | 0.150 | 0.152 | 0.184 | 0.250 | 0.517 |
| DynamicGAT + ET Transformer | 347 | 0.072 | 0.099 | 0.155 | 0.249 | 0.506 |
| Dual Transformer | 347 | 0.080 | 0.096 | 0.160 | 0.226 | 0.515 |
| Early-Fusion MLP | 347 | 0.064 | 0.076 | 0.168 | 0.263 | 0.486 |
| Late Fusion (CNN-LSTM + ET-LSTM) | 347 | 0.121 | 0.118 | 0.186 | 0.263 | 0.437 |

Pooled prevalence of HIGH over the held-out epochs: 0.493. A reference predictor that outputs the prevalence for every epoch has Brier = 0.250.
