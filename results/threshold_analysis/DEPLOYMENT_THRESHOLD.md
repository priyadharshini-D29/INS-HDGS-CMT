# Label-free prospective thresholds (no test labels used at any point)

Source: `losocv_repro_focal_g3p0_effective_num_37.csv` — 37 evaluable folds.

| strategy | uses test labels? | causal (online)? | BalAcc | MCC | F1 | Acc | Δ BalAcc vs fixed (p) | Δ BalAcc vs val-subject (p) |
|---|---|---|---|---|---|---|---|---|
| fixed_0.5 | no | n/a | 0.753 ± 0.180 | 0.507 ± 0.354 | 0.674 | 0.797 | +0.000 (1.000) | +0.000 (1.000) |
| val_subject | no | n/a | 0.753 ± 0.180 | 0.507 ± 0.354 | 0.674 | 0.797 | +0.000 (1.000) | +0.000 (1.000) |
| train_prior_quantile | no | no (transductive) | 0.767 ± 0.158 | 0.477 ± 0.291 | 0.682 | 0.711 | +0.014 (0.918) | +0.014 (0.918) |
| online_quantile[k=2] | no | yes | 0.750 ± 0.157 | 0.467 ± 0.285 | 0.696 | 0.719 | -0.004 (0.469) | -0.004 (0.469) |
| online_median[k=2] | no | yes | 0.750 ± 0.157 | 0.467 ± 0.285 | 0.696 | 0.719 | -0.004 (0.469) | -0.004 (0.469) |
| online_quantile[k=3] | no | yes | 0.768 ± 0.153 | 0.502 ± 0.295 | 0.718 | 0.751 | +0.015 (0.897) | +0.015 (0.897) |
| online_median[k=3] | no | yes | 0.768 ± 0.153 | 0.502 ± 0.295 | 0.718 | 0.751 | +0.015 (0.897) | +0.015 (0.897) |
| online_quantile[k=5] | no | yes | 0.772 ± 0.165 | 0.516 ± 0.316 | 0.712 | 0.773 | +0.019 (0.536) | +0.019 (0.536) |
| online_median[k=5] | no | yes | 0.772 ± 0.165 | 0.516 ± 0.316 | 0.712 | 0.773 | +0.019 (0.536) | +0.019 (0.536) |

Every threshold above is computed from (i) the training pool, (ii) the held-out validation subject, or (iii) the test subject's own *unlabelled* predicted probabilities. The online variants use only epochs that precede the one being classified, so they can be applied prospectively; epochs inside the warm-up window use the transferred validation-subject threshold.
