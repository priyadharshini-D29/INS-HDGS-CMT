# Statistical environment and settings for the manuscript comparisons

- Paired comparisons: two-sided Wilcoxon signed-rank test, zero differences discarded
  (`scipy.stats.wilcoxon(x, zero_method="wilcox")`, default two-sided alternative and
  automatic exact/normal method selection), Holmâ€“Bonferroni correction within each stated
  family; matched-pairs rank-biserial correlation as effect size; 95% bootstrap CIs of the
  mean paired difference (10,000 resamples). Implementation: `scripts/revision/round2_tests.py`
  and the released analysis scripts.
- SciPy on this machine when the 2026-10-04 refresh was run: 1.17.1
  (numpy 2.4.2, pandas 3.0.1). The original Brev runs used the
  instance's own SciPy; `VERIFICATION_2026-09-29.md` documents the version-dependent
  p-value differences observed between SciPy builds (exact vs normal-approximation paths).
- `losocv_eegnet_et.csv`, `losocv_shallow_et.csv` (engagement index) and the
  `dl_tuned_v2` fold-wise `losocv_{eegnet,shallow,et_lstm}.csv` are reconstructed from the
  sliced Brev training logs (`scripts/revision/reconstruct_losocv_from_logs.py`): per-fold
  metrics are exact to the 3 decimals the logs print, so paired statistics recomputed from
  them can differ from the published full-precision values in tie counts and in the third
  decimal of p-values. Their epoch-level probabilities (ECE/Brier, pooled curves) were
  computed on the Brev instance and are not reconstructable from the logs.
- The final production decoder's per-fold operating-point metrics are
  `results/ablation/abl_full/losocv_abl_full_final.csv` (mean balanced accuracy 0.71923,
  ROC-AUC 0.87899, MCC 0.44584), derived from the saved held-out probabilities; the
  historical `losocv_abl_full.csv` (0.7475/0.8790/0.4854) is the earlier 24-channel run.
- Known remaining gap: the per-fold held-out probability files of the rule-only variant
  (`abl_ns_rule_only`) and of the two engagement-index late fusions were produced on the
  Brev instance and are not in this tree, so Supplementary Table S13's rule-only row and
  the fusions' ECE/Brier cannot be recomputed here; their fold-level metrics are bracketed
  or reproduced by the files named above.
- Reconciliation with the manuscript (fourth external audit, 2026-10-05): on the
  reconstructed fold-wise inputs the ET-LSTM pooled-to-fold-wise ROC-AUC test gives
  raw p = 0.021 and p_Holm = 0.062 (16 of the 37 paired differences tie after the
  3-decimal rounding; `round2_tests.md`, section B), against the original
  full-precision computation's p = 0.012 / p_Holm = 0.036 run on the Brev instance.
  The manuscript reports the -0.031 decrease as descriptive and discloses both
  computations (main Section 3.2 and the Table S17 caption).
- Operating point of the product-label reports: the decoder's stored
  balanced_acc/mcc columns in `label_product/ablation/abl_full/losocv_abl_full.csv`
  were written at the per-fold opt_threshold; `tests_product_label.md` and
  `label_product/statistics/cross_modal_contribution.md` re-score the decoder at
  the common fixed 0.5 operating point from the saved held-out probabilities,
  matching manuscript Table 7 (0.559 / 0.606 / 0.107).
