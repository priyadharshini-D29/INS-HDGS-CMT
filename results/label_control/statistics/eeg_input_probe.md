# Stage-wise linear probe of the model's EEG input path (control label)

1846 epochs, 42 subjects, LOSO logistic regression (C=1).

| stage | features | pooled AUC | fold AUC (mean ± SD) | pooled BalAcc |
|---|---|---|---|---|
| R0 raw epoch: whole-head log band power (5) + rel. alpha | 6 | 0.730 | 0.780 ± 0.155 | 0.681 |
| R1 after loader per-subject z-score: same features | 6 | 0.751 | 0.794 ± 0.141 | 0.697 |
| S1 graph-builder node features, epoch mean, 19 electrodes x 5 bands (log) | 95 | 0.838 | 0.888 ± 0.101 | 0.764 |
| S1 graph-builder node features, whole-head mean (5 bands, log) | 5 | 0.772 | 0.825 ± 0.113 | 0.714 |
| S2 after model per-window LayerNorm: epoch mean, 19 x 5 | 95 | 0.874 | 0.909 ± 0.075 | 0.790 |
| S3 SNN proxy (band-weighted sum of S2): electrode means (19) | 19 | 0.691 | 0.749 ± 0.157 | 0.639 |
| S3 SNN proxy, all 10 windows x 19 electrodes | 190 | 0.660 | 0.682 ± 0.138 | 0.610 |

Reading: the first stage whose AUC falls well below R0/R1 is the transformation that removes the information the deep model would need; S3 is the only input the spiking encoder receives.
