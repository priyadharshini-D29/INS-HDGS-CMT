# DL baselines vs INS-HDGS-CMT (LOSOCV, fold-matched n=37)

All models scored on the **identical** subject set the full model was evaluated on, so rows are directly comparable.

| model | folds | bal_acc | f1 | roc_auc | mcc | acc | ece |
|---|---|---|---|---|---|---|---|
| INS-HDGS-CMT (full) | 37 | 0.733±0.200 | 0.679±0.306 | 0.887±0.145 | 0.461±0.381 | 0.764±0.183 | 0.487±0.164 |
| fusion_mlp | 37 | 0.715±0.188 | 0.627±0.321 | 0.863±0.199 | 0.425±0.377 | 0.729±0.197 | 0.518±0.201 |
| eegnet | 37 | 0.619±0.191 | 0.530±0.257 | 0.616±0.277 | 0.211±0.357 | 0.605±0.171 | 0.488±0.181 |
| shallow | 37 | 0.593±0.188 | 0.501±0.255 | 0.680±0.204 | 0.172±0.366 | 0.579±0.195 | 0.499±0.187 |
| deep | 37 | 0.532±0.216 | 0.436±0.257 | 0.585±0.276 | 0.061±0.401 | 0.531±0.194 | 0.504±0.209 |
| cnn_bilstm | 37 | 0.522±0.166 | 0.408±0.312 | 0.590±0.238 | 0.027±0.345 | 0.530±0.194 | 0.519±0.202 |

## Paired significance — full vs each baseline (Wilcoxon signed-rank)

Δ = full − baseline (per subject); positive ⇒ full model better. p from two-sided Wilcoxon on paired folds.

| baseline | metric | n | full | baseline | median Δ | p |
|---|---|---|---|---|---|---|
| cnn_bilstm | balanced_acc | 37 | 0.733 | 0.522 | +0.312 | 0.0001 * |
| cnn_bilstm | mcc | 37 | 0.461 | 0.027 | +0.639 | 0.0001 * |
| cnn_bilstm | f1 | 37 | 0.679 | 0.408 | +0.242 | 0.0001 * |
| cnn_bilstm | roc_auc | 37 | 0.887 | 0.590 | +0.300 | 0.0000 * |
| deep | balanced_acc | 37 | 0.733 | 0.532 | +0.200 | 0.0006 * |
| deep | mcc | 37 | 0.461 | 0.061 | +0.436 | 0.0002 * |
| deep | f1 | 37 | 0.679 | 0.436 | +0.298 | 0.0001 * |
| deep | roc_auc | 37 | 0.887 | 0.585 | +0.343 | 0.0000 * |
| eegnet | balanced_acc | 37 | 0.733 | 0.619 | +0.100 | 0.0097 * |
| eegnet | mcc | 37 | 0.461 | 0.211 | +0.270 | 0.0033 * |
| eegnet | f1 | 37 | 0.679 | 0.530 | +0.218 | 0.0046 * |
| eegnet | roc_auc | 37 | 0.887 | 0.616 | +0.250 | 0.0000 * |
| fusion_mlp | balanced_acc | 37 | 0.733 | 0.715 | +0.000 | 0.6236 |
| fusion_mlp | mcc | 37 | 0.461 | 0.425 | +0.000 | 0.5360 |
| fusion_mlp | f1 | 37 | 0.679 | 0.627 | +0.000 | 0.1833 |
| fusion_mlp | roc_auc | 37 | 0.887 | 0.863 | +0.000 | 0.6642 |
| shallow | balanced_acc | 37 | 0.733 | 0.593 | +0.125 | 0.0021 * |
| shallow | mcc | 37 | 0.461 | 0.172 | +0.278 | 0.0024 * |
| shallow | f1 | 37 | 0.679 | 0.501 | +0.204 | 0.0001 * |
| shallow | roc_auc | 37 | 0.887 | 0.680 | +0.188 | 0.0000 * |
