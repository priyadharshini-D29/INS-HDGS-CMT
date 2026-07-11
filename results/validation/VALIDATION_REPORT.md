# INS-HDGS-CMT — Four-Pillar Validation Report

_Generated 2026-06-16. Production model: `repro_focal_g3p0_effective_num_37`
(focal γ=3.0 · effective-num weighting · 5-member ensemble · 3-channel ET ·
37 evaluable LOSOCV folds). Headline metrics are **AUC, balanced-accuracy and
MCC** (not raw accuracy). Per-fold predictions, scripts and JSON are under
`results/validation/`._

---

## Headline metrics (subject-independent LOSOCV)

Reported as **mean ± SD across folds** (the LOSOCV-standard unit = subject) and,
secondarily, **pooled** over all 347 held-out epochs.

| Metric | Mean ± SD / fold | Pooled | Subject-level bootstrap 95% CI (mean/fold) |
|---|---|---|---|
| **ROC-AUC** | **0.901 ± 0.138** | 0.873 | **[0.854, 0.942]** |
| **Balanced accuracy** | **0.740 ± 0.189** | 0.754 | **[0.679, 0.799]** |
| **MCC** | **0.463 ± 0.369** | 0.517 | **[0.343, 0.580]** |
| F1 | 0.688 ± 0.279 | 0.772 | — |
| Cohen's κ | 0.426 ± 0.366 | 0.506 | — |
| Accuracy | 0.753 ± 0.184 | 0.752 | — |

All three headline-metric bootstrap CIs exclude chance (AUC 0.5, bal-acc 0.5,
MCC 0.0) by a wide margin → the central result is statistically robust to which
subjects happen to be in the cohort.

---

## 1. Internal validity — leakage-free LOSOCV  ✅ PASS

| Guarantee | Evidence |
|---|---|
| **Subject isolation** | `evaluation/leakage_audit.py --quick` → **Check 1 (data isolation) PASS**: every subject maps to its own `engagement_phase3d` data; no subject is served from a shared pooled directory. |
| **No test-subject leakage** | `evaluation/losocv.py` asserts `test_subj not in train_subs` **and** `val_subj not in train_subs` for every fold (`_run_fold_sequential` / `_run_fold_parallel`). Test subject is fully held out — never in train or val. |
| **Train/val/test split** | Per fold: 1 test subject (held out), 1 val subject (drawn from the *remaining training* subjects), the rest train. Val is itself a held-out training subject, so it is leakage-free w.r.t. test. |
| **Per-fold decision threshold refit on training data only** | The Youden-J operating point (`_find_optimal_threshold`) is fit on the **val subject's** probabilities only (a training-pool subject), then applied to test. No test labels touch threshold selection. Threshold winsorised to [0.30, 0.70] and defaults to 0.5 when the val set is too small (`MIN_YOUDEN_N=20`). |
| **Single-class / degenerate folds excluded a-priori** | Test subjects whose labels are all-HIGH/all-LOW or have minority ratio < `IMBALANCE_SKIP_RATIO` are skipped (AUC/MCC undefined); they still contribute to *other* folds' training pools. 37 of 42 subjects are evaluable. |

### ⚠️ One wording mismatch to fix before submission
The manuscript (Methods, l.123) states z-score normalisation uses *"statistics
estimated on the training folds only."* The implementation
(`data/dataset.py:502–525`) actually applies **per-subject** z-scoring: each
subject (including the test subject) is standardised by **its own** per-channel
mean/SD, computed at inference time **without labels**.

* This is **leakage-free** — it is unsupervised, uses no other subject's data,
  and is exactly what a deployed system would do for a new user (transductive /
  test-time normalisation, standard in subject-independent BCI).
* But it is **not** "train-only" normalisation. **Action:** change the Methods
  sentence to describe per-subject (transductive) unsupervised normalisation, OR
  switch the code to freeze training-pool statistics and re-run LOSOCV. The
  former is recommended (the current scheme is defensible and deployment-faithful).

---

## 2. Statistical validity — significance vs chance  ✅ PASS (strong)

Computed on saved per-fold predictions (`evaluation/validate_statistical.py`).

### Permutation / label-shuffle test (5,000 within-fold permutations)
Labels are shuffled **within each fold** (preserves per-fold class balance);
model probabilities/predictions are fixed. Under H₀ the score carries no
information about the label.

| Statistic | Observed | Null (mean ± SD) | p-value |
|---|---|---|---|
| AUC (mean/fold) | **0.901** | 0.501 ± 0.041 | **2×10⁻⁴** |
| Balanced-acc (mean/fold) | **0.740** | 0.501 ± 0.028 | **2×10⁻⁴** |
| MCC (mean/fold) | **0.463** | 0.001 ± 0.054 | **2×10⁻⁴** |
| AUC (pooled) | 0.873 | 0.596 ± 0.025 | **2×10⁻⁴** |
| Balanced-acc (pooled) | 0.754 | 0.568 ± 0.020 | **2×10⁻⁴** |
| MCC (pooled) | 0.517 | 0.138 ± 0.041 | **2×10⁻⁴** |

Every headline metric **collapses to chance** under label shuffle (mean-per-fold
AUC→0.50, bal-acc→0.50, MCC→0.00) and the observed value is the most extreme of
all 5,001 outcomes → p = 1/(5000+1). The model's discrimination is not a
small-sample artefact. (The *pooled* null sits slightly above 0.5 because
between-fold prevalence differences still help pooled ranking even under
within-fold shuffling; the **mean-per-fold** null = 0.50 is the clean reference.)

### Bootstrap CIs (subject = resampling unit, 10,000 reps)
AUC [0.854, 0.942] · balanced-acc [0.679, 0.799] · MCC [0.343, 0.580] — all
exclude chance (see headline table).

### Per-fold significance (binomial, one-sided)
Against each subject's **majority-class** baseline, **4/37** folds individually
clear p<0.05. This is expected and **not** a weakness of the model: with only
~6–17 epochs per subject and majority prevalence up to 0.86, a single subject's
*accuracy* cannot beat its own majority rate at p<0.05 even at AUC≈0.9. This is
why inference is done on **balanced metrics** (AUC/bal-acc/MCC) via the
permutation and bootstrap above, not on per-subject raw accuracy. The handful of
genuinely hard subjects (S21, S27, S02, S19) match the ranking-failure subjects
already documented in `RESULTS_threshold_and_baseline.md`.

---

## 3. Construct validity — EEG ↔ engagement concordance  ⚠️ WEAK (honest)

**Why this matters.** The HIGH/LOW labels are derived **entirely from
eye-tracking** (fixation ratio, dwell, ROI density, pupil, gaze entropy →
composite → per-subject median; `labeling/engagement_labeler.py`), and ET is
*also* a model input. A reviewer can call the task circular. The test:
does the **independent EEG modality** (which plays no role in labelling) carry
the engagement signal? (`analysis/eeg_concordance.py`)

A-priori signatures tested: posterior **alpha (8–13 Hz) suppression** (Pz, Oz,
O1, O2, P3, P4, P7, P8) and **frontal-midline theta (4–8 Hz) increase** (Fz, F3,
F4, FC1, FC2) in HIGH vs LOW, relative band power (scale-invariant), 37 subjects.

| Test | Frontal theta (HIGH>LOW) | Posterior alpha (HIGH<LOW) |
|---|---|---|
| Direction | correct (Δ>0) | correct (Δ<0) |
| Within-subject permutation (primary) | d=+0.091, p=0.18 | d=−0.028, p=0.40 |
| Per-subject-mean Wilcoxon | p=0.27 (20/37 subj) | p=0.30 (19/37 subj) |

**Verdict: directionally consistent but not significant.** Two honest reasons,
not a labelling failure:
1. The 5-s eyes-open ad-viewing EEG is **delta-dominated with no posterior alpha
   peak** (band power: δ≈16 vs α≈1.0, spectral peak ≈3 Hz) — there is little
   resting alpha to suppress in this paradigm.
2. ~6–17 epochs/subject → low power for a subtle single-trial spectral contrast.

### Connectivity concordance (the model's actual quantity) — also null
Because the EEG **graph** is the dominant model component (§5), the better
construct probe is fronto-parietal **PLV** (the quantity the model consumes),
not band power. `analysis/connectivity_concordance.py` (theta & alpha PLV
between {Fz,F3,F4,FC1,FC2} × {Pz,Oz,O1,O2,P3,P4,P7,P8}, within-subject
permutation, 37 subj / 347 epochs):

| | within-subject perm | per-subject Wilcoxon |
|---|---|---|
| Theta fronto-parietal PLV | d=+0.04, p=0.72 | p=0.71 (22/37 Δ>0) |
| Alpha fronto-parietal PLV | d=−0.04, p=0.66 | p=0.57 |

So **no single hand-picked EEG summary (regional power or fronto-parietal PLV)
separates HIGH from LOW at the epoch level.**

### What this means, and the defensible anti-circularity argument
Direct neurophysiological concordance via canonical summaries is **not
established** in this paradigm — do **not** claim alpha-suppression / theta-
increase / PLV significance in the text. However, the circularity worry
(ET-label predicted from ET-input) is **refuted by the ablation**, which is the
stronger evidence:

* Removing the **ET branch** barely changes performance (Δ bal-acc −0.018,
  Wilcoxon **p=0.25, n.s.**). A purely circular ET→ET-label model would collapse
  without ET — ours does not.
* Removing the **EEG connectivity graph** is the single most damaging ablation
  (Δ bal-acc −0.057, **p=0.007**).

⇒ The model's discrimination is **driven by EEG connectivity, not by re-reading
the ET labels**. The EEG therefore carries decodable engagement-related
information — it is simply **high-dimensional** (full 24-node dynamic PLV graph ×
10 windows learned by the GAT), not reducible to one regional contrast, which is
why the scalar summaries above are flat. Frame construct validity on (a) this
ablation dissociation and (b) the existing integrated-gradients / graph-attention
importance maps (manuscript §Explainability), **not** on the epoch-level
power/PLV contrasts.

---

## 4. External validity — vs Kalaganis et al. 2025 (same NeuMa data)  ✅ FAVOURABLE (with caveats)

Kalaganis et al. 2025 (Brain Informatics; doi:10.1186/s40708-025-00272-z) is the
most directly comparable prior work — same NeuMa corpus, same EEG+gaze fusion,
42 subjects, EEG @300 Hz.

| Study | Task | Protocol | Cohen's κ | F1 |
|---|---|---|---|---|
| Kalaganis 2025 — EEG-Indices | Buy/NoBuy | within-subject LOOCV | 0.05 | 0.31 |
| Kalaganis 2025 — ET-Indices | Buy/NoBuy | within-subject LOOCV | 0.27 | 0.48 |
| Kalaganis 2025 — GFT-EEG | Buy/NoBuy | within-subject LOOCV | 0.18 | 0.42 |
| **Kalaganis 2025 — GFT-Hybrid (their best)** | Buy/NoBuy | **within-subject LOOCV** | **0.35** | **0.54** |
| **INS-HDGS-CMT (ours)** | Engagement HIGH/LOW | **subject-independent LOSOCV** | **0.43 / 0.51** | **0.69 / 0.77** |

(ours shown mean-per-fold / pooled.)

**Interpretation.** We exceed their best NeuMa hybrid on **both** shared metrics
(κ and F1) **while evaluating under the stricter, subject-independent LOSOCV** —
they use within-subject LOOCV, which is systematically more optimistic because it
trains and tests on the same participant. Beating an easier protocol from the
harder one is a strong external-validity signal.

**Caveats to state in the paper (honesty):** the **tasks differ** (their
Buy/NoBuy decision vs our ET-derived engagement), so this is *indicative*, not a
like-for-like benchmark; they report only κ/F1 (no AUC/accuracy). Keep the
"absolute scores not directly commensurable" sentence already in the manuscript.

---

## 5. Robustness — ablations + calibration

### Component ablations (leave-one-component-out, paired across 37 folds)
Δ = variant − full; negative ⇒ the component **helps**. (`results/ablation/`)

| Component removed | Δ balanced-acc | Wilcoxon p | Δ MCC | Verdict |
|---|---|---|---|---|
| **EEG dynamic graph** | **−0.057** | **0.007** | **−0.110** | **helps (significant)** |
| Eye-tracking branch | −0.018 | 0.255 | −0.037 | helps (n.s.) |
| Spiking front-end | −0.001 | 0.535 | +0.005 | neutral |
| Neuro-symbolic module | +0.019 | 0.794 | +0.051 | neutral / slight ↑ when removed |

**Status:** 4 of the planned component ablations are complete. The remaining
variants (`no_contrastive`, `no_fusion_transformer`, `no_mmd`, `no_roi`) were
queued (`run_component_ablation.sh`) but are **not currently running** — relaunch
to complete the robustness table. The dominant, statistically significant result
is already in: the EEG functional-connectivity graph is the key component.

### Calibration (ECE)
* **Pooled ECE = 0.16** (15-bin, uncalibrated, 347 epochs) — moderate; the model
  is somewhat overconfident.
* The **per-fold mean ECE of 0.42 is not interpretable** — 15-bin ECE on ~7–17
  test epochs is dominated by binning noise. Report the **pooled** ECE.
* Post-hoc per-subject **temperature scaling did not help** (per-fold ECE 0.42→0.51):
  fitting T on a single ~7–17-epoch val subject is unreliable. Recommend
  **cross-fold** calibration (pool val sets) rather than per-subject T, or report
  the uncalibrated pooled ECE and note calibration as future work.

---

## Bottom line

| Pillar | Verdict |
|---|---|
| Internal validity | ✅ Leakage-free (fix one Methods sentence on normalisation) |
| Statistical validity | ✅ Strong — permutation collapses to chance (p=2×10⁻⁴); bootstrap CIs exclude chance |
| Construct validity | ⚠️ No canonical EEG summary (power or PLV) separates HIGH/LOW; **anti-circularity rests on ablation** (ET removable n.s.; EEG-graph dominant p=0.007) |
| External validity | ✅ Beats Kalaganis 2025 (κ, F1) under a stricter protocol; flag task difference |
| Robustness | ✅ EEG-graph ablation significant; finish remaining ablations; report pooled ECE 0.16 |

**Defensible headline for the manuscript:** under strict subject-independent
LOSOCV, INS-HDGS-CMT attains **AUC 0.90 ± 0.14, balanced-accuracy 0.74 ± 0.19,
MCC 0.46 ± 0.37** (bootstrap 95% CIs [0.85,0.94] / [0.68,0.80] / [0.34,0.58]);
performance is highly significant (label-permutation p = 2×10⁻⁴), exceeds the
most comparable NeuMa prior work, and is driven primarily by EEG functional
connectivity (ablation p = 0.007).

### Artefacts
- `results/validation/statistical_validity_*.json`, `per_fold_metrics_*.csv`, `per_fold_significance_*.csv`
- `results/validation/eeg_concordance.json`, `eeg_concordance_per_subject.csv`, `eeg_concordance_per_epoch.csv`
- Scripts: `evaluation/validate_statistical.py`, `analysis/eeg_concordance.py`
