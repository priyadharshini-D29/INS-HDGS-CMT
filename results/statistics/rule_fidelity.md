# Soft-rule fidelity — abl_full (37 folds)

Learned bypass gate alpha: mean 0.570 (range 0.567–0.573 across folds/members; init sigmoid(0.30)=0.574).

Decision fidelity: argmax(R) agrees with the final decision on 39.4% of held-out epochs as a mean over folds (min 0%); pooled over the 347 held-out epochs the share is 36.6% (127/347). Pearson r between the rule margin (R_HIGH−R_LOW) and the final logit margin: mean within-fold -0.691.

| decision rule at inference | BalAcc | ROC-AUC | MCC |
|---|---|---|---|
| gated (as reported, learned α) | 0.719 ± 0.194 | 0.879 ± 0.167 | 0.446 ± 0.378 |
| rule evidence only (α forced 0) | 0.392 ± 0.185 | 0.221 ± 0.237 | -0.220 ± 0.379 |
| bypass only (α forced 1) | 0.725 ± 0.193 | 0.877 ± 0.167 | 0.457 ± 0.377 |
- gated − rule-only (balanced_acc): Δ = +0.327, Wilcoxon p = 0.000
- gated − rule-only (roc_auc): Δ = +0.658, Wilcoxon p = 0.000
- gated − rule-only (mcc): Δ = +0.665, Wilcoxon p = 0.000

## Per-rule activation vs. final P(HIGH) (pooled held-out epochs)

| rule | mean a_r | a_r HIGH | a_r LOW | Spearman ρ(a_r, P(HIGH)) | dominant in |
|---|---|---|---|---|---|
| 1 | 0.125 | 0.125 | 0.125 | +0.28 | 1% |
| 2 | 0.125 | 0.125 | 0.125 | +0.07 | 13% |
| 3 | 0.125 | 0.125 | 0.125 | +0.15 | 3% |
| 4 | 0.125 | 0.125 | 0.125 | +0.22 | 0% |
| 5 | 0.125 | 0.125 | 0.125 | -0.18 | 0% |
| 6 | 0.125 | 0.125 | 0.125 | -0.21 | 1% |
| 7 | 0.125 | 0.125 | 0.125 | -0.07 | 18% |
| 8 | 0.125 | 0.125 | 0.126 | -0.12 | 64% |

## Model TRAINED with α fixed at 0 (`losocv_abl_ns_rule_only.csv`, 37 paired folds)

> **SUPERSEDED (2026-10-04).** The comparison that stood here (full 0.747 vs rule-only 0.715,
> p = 0.047/0.014/0.093) paired the earlier 24-channel full run with the rule-only variant's
> training-time export. The manuscript's final comparison (Supplementary Table S13) pairs the
> saved held-out probabilities of the final production run (`losocv_abl_full_final.csv`;
> balanced accuracy 0.719, ROC-AUC 0.879, MCC 0.446) with those of the rule-only variant
> (0.734, 0.839, 0.497) at the common 0.5 operating point: Δ = +0.014 / −0.040 / +0.051,
> Wilcoxon p = 0.42 / 0.018 / 0.23. The rule-only variant's per-fold probability files were
> produced on the Brev instance and are not in this tree; its training-time export brackets
> the published row (balanced_acc 0.7147 uncalibrated, 0.7335 calibrated).

