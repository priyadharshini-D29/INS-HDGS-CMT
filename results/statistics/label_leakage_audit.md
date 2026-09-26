# Label-recoverability audit (engagement_phase3d label)

385 stimulus epochs, 42 subjects; probes evaluated on the 37 subjects with both classes (same folds as the deep models). On-disk label agreement 385/385.

| feature set | pooled AUC | fold AUC (mean ± SD) | pooled BalAcc | fold BalAcc (mean ± SD) |
|---|---|---|---|---|
| EEG-5 (frontal band power) | 0.605 | 0.671 ± 0.216 | 0.571 | 0.582 ± 0.172 |
| ET-5 (gaze statistics) | 0.939 | 0.917 ± 0.137 | 0.862 | 0.831 ± 0.187 |
| ALL-10 | 0.996 | 0.999 ± 0.004 | 0.965 | 0.957 ± 0.103 |
| RULE score (upper bound) | 1.000 | 1.000 ± 0.000 | 0.989 | 0.989 ± 0.044 |

Interpretation: the EEG-5 and ET-5 rows are the linear floor that a model seeing only that modality's defining statistics attains; a learned model is informative beyond the label rule only where it exceeds the corresponding row. Fill Supplementary Table S15 and Sections 2.4 / 3.2 of the manuscript from this table.
