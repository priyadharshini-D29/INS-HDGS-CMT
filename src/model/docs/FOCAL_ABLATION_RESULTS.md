# Focal-Loss Ablation Grid — LOSOCV Results

Run: 2026-05-30 → 2026-05-31 · 37-fold LOSOCV · 8 GPUs · ~10.3 h/cell
Grid: γ ∈ {0, 1.0, 1.5, 2.0, 3.0} × weighting ∈ {balanced, effective_num, sqrt_inv_freq}
Metrics: **raw → post-hoc adjusted** (per-fold optimal threshold + 5-member ensemble, N_ens=5)

## Ranked by adjusted accuracy

| Rank | Config | Acc (adj) | Bal-Acc (adj) | MCC (adj) | Raw Acc | ROC-AUC |
|------|--------|-----------|---------------|-----------|---------|---------|
| 1 | **g3p0_effective_num** | **0.7976** | 0.7594 | **0.5304** | 0.7569 | 0.8700 |
| 2 | g0p0_balanced (plain CE) | 0.7974 | **0.7602** | 0.5244 | 0.7481 | 0.8522 |
| 3 | g2p0_balanced | 0.7888 | 0.7484 | 0.5055 | 0.7431 | 0.8686 |
| 4 | g1p5_balanced | 0.7798 | 0.7399 | 0.4851 | 0.7543 | 0.8633 |
| 5 | g3p0_balanced | 0.7781 | 0.7353 | 0.4838 | 0.7523 | 0.8876 |
| 6 | g1p0_balanced | 0.7778 | 0.7461 | 0.5066 | 0.7347 | 0.8722 |
| 7 | g1p5_sqrt_inv_freq | 0.7774 | 0.7364 | 0.4755 | 0.7387 | 0.8529 |
| 8 | g1p5_effective_num | 0.7739 | 0.7364 | 0.4821 | 0.7546 | 0.8691 |
| 9 | g2p0_effective_num | 0.7724 | 0.7274 | 0.4644 | 0.7510 | 0.8718 |
| 10 | g1p0_effective_num | 0.7692 | 0.7405 | 0.4860 | 0.7458 | 0.8857 |
| 11 | g1p0_sqrt_inv_freq | 0.7641 | 0.7422 | 0.4942 | 0.7467 | 0.8485 |
| 12 | g2p0_sqrt_inv_freq | 0.7621 | 0.7229 | 0.4418 | 0.7382 | 0.8730 |
| — | g3p0_sqrt_inv_freq | *(running)* | | | | |

## Findings
- **Ceiling ~0.797** — no cell crossed 0.80 on accuracy; balanced-acc tops out ~0.76.
- Post-hoc threshold + ensemble adds **~+0.04 acc** consistently → AUC (~0.87) is strong but the global decision threshold leaves accuracy on the table.
- `sqrt_inv_freq` weighting is consistently the weakest; `balanced` and `effective_num` lead.
- γ (focal) has small effect; γ=0 (plain CE) ties for best — focal is not the lever.
- **Real bottleneck = per-fold variance** (acc std ≈ 0.19). Worst folds: F02=0.286, F17=0.412, F26=0.429. Per-fold ROC-AUC min drops to 0.0–0.2 → the model is *inverted* on a few subjects (subject-level distribution shift).
- Per-fold optimal thresholds are fit on tiny val sets (n=6–16) and overfit (e.g. Fold34 thr=0.05).
- Single-class subjects skipped as test folds: S16 (all-LOW), S33 / S41 (all-HIGH).

## Path to 0.80 (see analysis)
1. Ensemble probabilities **across γ configs** (different γ make different errors).
2. **Temperature-scaling calibration** per fold (ECE was 0.4166 in v13).
3. **Shrink per-fold threshold toward the global prior** (mitigate tiny-val overfit).
4. Target the few **inverted-AUC hard subjects** with the DANN/MMD invariance chain.
5. **Test-time augmentation** (window averaging) at inference.
