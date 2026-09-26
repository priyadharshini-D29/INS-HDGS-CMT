# Label-free prospective thresholds (no test labels used at any point)

Source: `losocv_abl_full.csv` — 37 evaluable folds.

| strategy | uses test labels? | causal (online)? | BalAcc | MCC | F1 | Acc | Δ BalAcc vs fixed (p) | Δ BalAcc vs val-subject (p) |
|---|---|---|---|---|---|---|---|---|
| fixed_0.5 | no | n/a | 0.719 ± 0.194 | 0.446 ± 0.378 | 0.657 | 0.767 | +0.000 (1.000) | +0.000 (1.000) |
| val_subject | no | n/a | 0.719 ± 0.194 | 0.446 ± 0.378 | 0.657 | 0.767 | +0.000 (1.000) | +0.000 (1.000) |
| train_prior_quantile | no | no (transductive) | 0.740 ± 0.189 | 0.433 ± 0.339 | 0.663 | 0.692 | +0.020 (0.629) | +0.020 (0.629) |
| online_quantile[k=2] | no | yes | 0.733 ± 0.179 | 0.429 ± 0.328 | 0.680 | 0.709 | +0.013 (0.959) | +0.013 (0.959) |
| online_median[k=2] | no | yes | 0.733 ± 0.179 | 0.429 ± 0.328 | 0.680 | 0.709 | +0.013 (0.959) | +0.013 (0.959) |
| online_quantile[k=3] | no | yes | 0.748 ± 0.183 | 0.464 ± 0.339 | 0.701 | 0.735 | +0.029 (0.710) | +0.029 (0.710) |
| online_median[k=3] | no | yes | 0.748 ± 0.183 | 0.464 ± 0.339 | 0.701 | 0.735 | +0.029 (0.710) | +0.029 (0.710) |
| online_quantile[k=5] | no | yes | 0.746 ± 0.190 | 0.474 ± 0.355 | 0.693 | 0.754 | +0.027 (0.355) | +0.027 (0.355) |
| online_median[k=5] | no | yes | 0.746 ± 0.190 | 0.474 ± 0.355 | 0.693 | 0.754 | +0.027 (0.355) | +0.027 (0.355) |

Every threshold above is computed from (i) the training pool, (ii) the held-out validation subject, or (iii) the test subject's own *unlabelled* predicted probabilities. The online variants use only epochs that precede the one being classified, so they can be applied prospectively; epochs inside the warm-up window use the transferred validation-subject threshold.
