# Label-recoverability audit (positive control: browsing vs rest)

1774 stimulus epochs, 42 subjects; probes evaluated on the 42 subjects with both classes (same folds as the deep models). On-disk label agreement 0/0.

| feature set | pooled AUC | fold AUC (mean ± SD) | pooled BalAcc | fold BalAcc (mean ± SD) |
|---|---|---|---|---|
| EEG-5 (frontal band power) | 0.584 | 0.622 ± 0.140 | 0.575 | 0.572 ± 0.103 |
| ET-5 (gaze statistics) | 0.887 | 0.878 ± 0.118 | 0.850 | 0.846 ± 0.101 |
| ALL-10 | 0.888 | 0.891 ± 0.112 | 0.856 | 0.854 ± 0.100 |
| EEG whole-head log band power (5 bands + rel. alpha) | 0.729 | 0.787 ± 0.154 | 0.690 | 0.683 ± 0.122 |
| EEG whole-head + ET-5 | 0.890 | 0.907 ± 0.102 | 0.835 | 0.836 ± 0.106 |
| RULE score (old engagement index) | 0.615 | 0.632 ± 0.165 | 0.591 | 0.595 ± 0.144 |
| EEG-5 (frontal band power) [within-subject z] | 0.575 | 0.555 ± 0.209 | 0.564 | 0.557 ± 0.151 |
| ET-5 (gaze statistics) [within-subject z] | 0.899 | 0.885 ± 0.111 | 0.862 | 0.860 ± 0.111 |
| ALL-10 [within-subject z] | 0.901 | 0.892 ± 0.106 | 0.865 | 0.868 ± 0.092 |
| EEG whole-head log band power (5 bands + rel. alpha) [within-subject z] | 0.782 | 0.765 ± 0.184 | 0.724 | 0.721 ± 0.163 |
| EEG whole-head + ET-5 [within-subject z] | 0.914 | 0.900 ± 0.107 | 0.850 | 0.848 ± 0.118 |
| RULE score (old engagement index) [within-subject z] | 0.630 | 0.632 ± 0.165 | 0.620 | 0.619 ± 0.140 |

Interpretation: the EEG-5 and ET-5 rows are the linear floor that a model seeing only that modality's defining statistics attains; a learned model is informative beyond the label rule only where it exceeds the corresponding row. Fill Supplementary Table S15 and Sections 2.4 / 3.2 of the manuscript from this table.
