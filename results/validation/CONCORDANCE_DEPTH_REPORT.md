# Single-marker concordance — engagement is multivariate (manuscript Fig. fig_concordance)

Within-subject label-permutation tests (20,000 perms, 347 epochs, 37 subjects).
Source: results/validation/eeg_concordance.json, connectivity_concordance.json.
Figure: figures/validation/fig_concordance_depth.{png,pdf}; rebuild with analysis/eeg_concordance.py + connectivity_concordance.py then /tmp script.

| Marker (HIGH-LOW) | Cohen's d | perm p | significant? |
|---|---|---|---|
| Frontal-theta band power | +0.09 | 0.18 | no |
| Posterior-alpha band power | -0.03 | 0.40 | no |
| Fronto-posterior PLV, theta | +0.04 | 0.72 | no |
| Fronto-posterior PLV, alpha | -0.04 | 0.66 | no |

All |d| < 0.2 (negligible) and inside the permutation null -> single-feature ROC-AUC ~ 0.5.
Same epochs decoded at ROC-AUC 0.82 (EEG branch) / 0.90 (full model) -> signal is multivariate,
recovered only by graph-structured integration. More rigorous than the descriptive
connectivity summaries in Kalaganis et al. (2025), which report no label-concordance test.
