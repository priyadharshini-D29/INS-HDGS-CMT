# Cohen's kappa — exact, reproducible values (manuscript Table tab:kappa)

Recomputed from the released per-fold LOSOCV predictions (`y_true`, `y_prob`) with
`sklearn.metrics.cohen_kappa_score`; the stored `kappa`/`kappa_cal` columns are **not**
trusted. Reproduce with `python analysis/verify_cohen_kappa.py`.

| Model | raw (mean per-subj) | calibrated | pooled @0.5 |
|---|---|---|---|
| Full multimodal model | 0.4264 → **0.43** | 0.4724 → **0.47** | 0.5964 → 0.60 |
| EEG branch (leakage-free) | 0.3578 → **0.36** | 0.4314 → **0.43** | 0.4992 → 0.50 |

- "raw" = validation-tuned decision threshold (`opt_threshold`), leakage-free.
- "calibrated" = standard temperature scaling (`T_post`) + calibrated threshold (`opt_threshold_cal`).
- "mean per-subject" matches the aggregation used by Kalaganis et al. (2025) (κ=0.35), so it is the fair comparator.
- Per-fold raw κ reproduces the run's stored `kappa` column to machine precision (max abs diff 0.0e+00).
- Prior best for context: Kalaganis et al. GFT-hybrid κ = 0.35 (Buy/NoBuy, within-subject LOOCV).
