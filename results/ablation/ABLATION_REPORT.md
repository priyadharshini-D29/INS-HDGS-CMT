# INS-HDGS-CMT — Ablation Study Report

**Task:** HIGH-vs-LOW engagement (NeuMa).
**Protocol:** Leave-One-Subject-Out CV, 37 evaluable held-out subjects.
**Pairing:** Every variant uses the *same seed → identical folds* as the full
model, so all comparisons are paired. Metrics are **calibrated** (post-hoc
temperature + decision threshold tuned on a held-out validation subject —
never the test subject). Significance is a paired Wilcoxon signed-rank test on
per-subject balanced accuracy vs. the full model.

---

## 1. Leave-one-component-out results (calibrated)

Full model: **balanced accuracy 0.753**, MCC 0.507. A *negative* Δ means the
component helps (removing it hurts).

| Configuration | BalAcc | Δ BalAcc | Δ MCC | Wilcoxon p |
|---|---|---|---|---|
| **Full model** | 0.753 | — | — | — |
| − Dynamic graph | 0.696 | **−0.057** | **−0.110** | **0.009** \* |
| − Contrastive objective | 0.729 | −0.024 | −0.028 | 0.221 |
| − MMD alignment | 0.731 | −0.022 | −0.037 | 0.200 |
| − Eye-tracking branch | 0.735 | −0.018 | −0.037 | 0.468 |
| − Spiking encoder | 0.752 | −0.001 | +0.005 | 0.405 |
| − ROI guidance | 0.754 | +0.001 | +0.005 | 0.571 |
| − Fusion transformer | 0.756 | +0.003 | +0.017 | 0.950 |
| − Neuro-symbolic module | 0.773 | +0.019 | +0.050 | 0.714 |

\* Only the dynamic EEG graph survives as statistically significant (p < 0.05).
Rows ordered by importance (most degrading first).

*Note: the `ns_explain_only` confirmatory variant has no result CSV yet — that
re-run did not flush output and must be relaunched before it can be reported.*

---

## 2. Findings

1. **The dynamic EEG graph is the single load-bearing component.** Removing it
   costs 5.7 balanced-accuracy points and 0.110 MCC (p = 0.009) — the only
   ablation that is significant. This is consistent with the standalone
   EEG-encoder comparison, where the graph-based EEG branch ranked first.

2. **Domain-alignment and the ET branch contribute small, non-significant
   gains.** Removing the contrastive objective (Δ −0.024), MMD alignment
   (Δ −0.022) or the eye-tracking branch (Δ −0.018) each produces a minor,
   non-significant drop — they help at the margin but are not decisive.

3. **The spiking encoder, ROI guidance, fusion transformer and neuro-symbolic
   module are accuracy-neutral** (|Δ| ≤ 0.02, all p > 0.4). Their justification
   is *not* raw accuracy:
   - **Spiking (LIF) encoder** → energy efficiency (Section 3).
   - **ROI guidance + neuro-symbolic module** → interpretability / auditability
     (rule traces, attribution), used in an explanation-only role.

4. **Removing the neuro-symbolic decision path slightly *raises* accuracy**
   (Δ +0.019, n.s.). This motivates the explanation-only configuration: keep
   the rule traces for interpretability while letting the standard classifier
   make the decision, avoiding the small symbolic-refinement cost.

---

## 3. Spiking-encoder efficiency (why we keep an accuracy-neutral module)

Measured on the trained LIF encoder over 300 EEG epochs (10 simulation steps):

- **Mean firing rate 10.6 %** (89.4 % temporal sparsity — neurons silent most steps).
- **Spiking layers:** 31,241 pJ vs. 1,507,328 pJ for an equivalent dense
  network → **2.1 % of the energy** (48× lower).
- **Whole encoder** (dense input projection counted as MAC in both): 10.5 % of
  the dense-equivalent energy.
- Energy model: E_MAC = 4.6 pJ, E_AC = 0.9 pJ (45 nm CMOS, Horowitz, ISSCC 2014).

---

## 4. Conclusion

The architecture's accuracy is driven by **dynamic graph modelling of EEG**; the
spiking, ROI and neuro-symbolic components are retained for **efficiency and
interpretability**, not classification gain, and the paper states this honestly
rather than claiming accuracy benefits they do not provide.

*Sources:* `results/ablation/abl_*/losocv_abl_*.csv`,
`results/losocv_metrics/losocv_repro_focal_g3p0_effective_num_37.csv`,
`results/statistics/snn_energy*.md`. Calibrated columns (`*_cal`).
