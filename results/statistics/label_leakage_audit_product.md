# Label-recoverability audit (purchase-intent label)

2338 stimulus epochs, 42 subjects; probes evaluated on the 42 subjects with both classes (same folds as the deep models). On-disk label agreement 0/0.

| feature set | pooled AUC | fold AUC (mean ± SD) | pooled BalAcc | fold BalAcc (mean ± SD) |
|---|---|---|---|---|
| EEG-5 (frontal band power) | 0.505 | 0.527 ± 0.129 | 0.500 | 0.500 ± 0.000 |
| ET-5 (gaze statistics) | 0.571 | 0.570 ± 0.126 | 0.500 | 0.500 ± 0.000 |
| ALL-10 | 0.577 | 0.581 ± 0.096 | 0.499 | 0.498 ± 0.012 |
| RULE score (old engagement index) | 0.489 | 0.530 ± 0.119 | 0.500 | 0.500 ± 0.000 |
| Total dwell on product (s) | 0.853 | 0.880 ± 0.100 | 0.642 | 0.605 ± 0.099 |
| Dwell + n_runs + n_views + anchor run | 0.899 | 0.911 ± 0.067 | 0.734 | 0.720 ± 0.117 |

Interpretation: the EEG-5 and ET-5 rows are the linear floor that a model seeing only that modality's defining statistics attains; a learned model is informative beyond the label rule only where it exceeds the corresponding row. Fill Supplementary Table S15 and Sections 2.4 / 3.2 of the manuscript from this table.
