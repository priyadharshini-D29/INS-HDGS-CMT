# Label-recoverability audit (purchase-intent label)

383 stimulus epochs, 42 subjects; probes evaluated on the 41 subjects with both classes (same folds as the deep models). On-disk label agreement 385/385.

| feature set | pooled AUC | fold AUC (mean ± SD) | pooled BalAcc | fold BalAcc (mean ± SD) |
|---|---|---|---|---|
| EEG-5 (frontal band power) | 0.409 | 0.438 ± 0.262 | 0.507 | 0.502 ± 0.010 |
| ET-5 (gaze statistics) | 0.620 | 0.674 ± 0.250 | 0.505 | 0.515 ± 0.106 |
| ALL-10 | 0.596 | 0.649 ± 0.252 | 0.489 | 0.501 ± 0.090 |
| RULE score (old engagement index) | 0.225 | 0.318 ± 0.234 | 0.500 | 0.500 ± 0.000 |
| Dominant-product dwell fraction | 0.659 | 0.714 ± 0.245 | 0.546 | 0.556 ± 0.158 |

Interpretation: the EEG-5 and ET-5 rows are the linear floor that a model seeing only that modality's defining statistics attains; a learned model is informative beyond the label rule only where it exceeds the corresponding row. Fill Supplementary Table S15 and Sections 2.4 / 3.2 of the manuscript from this table.
