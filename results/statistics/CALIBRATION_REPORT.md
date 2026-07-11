# Subject-adaptive threshold experiment (post-hoc, leakage-free)

Full model `repro_focal_g3p0_effective_num_37` — 37 two-class folds. All thresholds set without test labels.

| Strategy | Bal-Acc | MCC | Macro-F1 | Acc | Δ vs val_opt | p (Wilcoxon) |
|---|---|---|---|---|---|---|
| prevalence_match | 0.778 ± 0.151 | 0.502 | 0.705 | 0.723 | +0.000 | 0.3338 |
| subject_median | 0.767 ± 0.156 | 0.477 | 0.693 | 0.711 | +0.000 | 0.5713 |
| fixed_0.5 | 0.753 ± 0.178 | 0.507 | 0.715 | 0.797 | +0.000 | 0.3053 |
| val_opt | 0.740 ± 0.186 | 0.463 | 0.669 | 0.753 | — | — |
