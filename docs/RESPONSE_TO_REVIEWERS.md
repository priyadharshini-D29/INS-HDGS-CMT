# Response to Reviewers — INS-HDGS-CMT (Brain Informatics revision)

We thank both reviewers for their careful reading. Below, every comment is
quoted, followed by our response and the exact changes made. Text in
**[bracketed red]** in the revised manuscript (`\todoval{...}`) marks values
that come from the re-runs listed in `docs/REVISION_RUNS.md`; the analysis code,
the runs that were possible without GPU retraining, and the manuscript text
around them are complete, and the remaining numbers are inserted once the
server jobs finish. All new analysis scripts live in `scripts/analysis/` and
their outputs in `results/`.

---

## Reviewer 1

### R1-1. The framework is presented as cross-modal, but evaluation focuses on EEG; the contribution of the cross-modal interaction is unclear.

**Response.** We agree that the gain attributable to the eye-tracking pathway
had to be isolated explicitly. We added a dedicated section (new **Section
3.4b, "Contribution of the Eye-Tracking Pathway"**, with **Supplementary
Table S14**) that compares, on the same 37 held-out subjects and with paired,
Holm-corrected Wilcoxon tests, the full model with (i) the gaze-sequence
branch and cross-modal transformer removed, (ii) the ROI vector removed, (iii)
the cross-modal transformer replaced by a single cross-attention, and (iv)
every gaze-derived input removed. Using the existing fold-matched runs:

| Comparison (37 folds) | Δ ROC-AUC [95% CI] | p (Holm) | Δ BalAcc | Δ MCC |
|---|---|---|---|---|
| full − (no gaze sequence, ROI kept) | +0.083 [+0.020, +0.146] | **0.006** | +0.043 (n.s.) | +0.065 (n.s.) |
| full − (no ROI vector) | +0.038 | 0.82 | +0.026 (n.s.) | +0.037 (n.s.) |
| full − (no cross-modal transformer) | +0.010 | 0.85 | +0.025 (n.s.) | +0.052 (n.s.) |
| full − ET-LSTM (best ET-only encoder) | +0.055 | **0.046** | +0.001 (n.s.) | −0.030 (n.s.) |

The cross-modal interaction therefore contributes a real but bounded gain of
about 0.08 in ROC-AUC, confined to ranking quality; the operating-point metrics
are carried by the EEG pathway. The full model also exceeds the strongest
eye-tracking-only encoder in ROC-AUC while matching it on balanced accuracy and
MCC, so the neural pathway adds ranking information that gaze alone does not
supply. The abstract, Section 3.2 and the Conclusions were rewritten to state
this bounded contribution rather than a general superiority claim.

While preparing this analysis we found that the configuration tabulated as the
"EEG branch" in Table 3 (`AblationConfig.no_et()`) removed the gaze *sequence*
but still received the gaze-derived ROI dwell histogram through the ROI-attention
gate. We have corrected this: the revised Table 3 reports a strictly gaze-free
branch (`AblationConfig.eeg_only()`, no ROI input, no ROI modulation, no
contrastive alignment), the earlier configuration is reported transparently as
the "ROI-retained" variant in Section 3.4b / Table S14, and the text no longer
describes the earlier branch as never accessing gaze. The gaze-free numbers are
being produced by the re-run listed as run 5 in `docs/REVISION_RUNS.md`.

### R1-2. Statistical significance tests for the improvements over the baselines.

**Response.** Paired significance testing was already applied to the EEG-encoder
comparison (Table 8, Supplementary Table S6: Friedman omnibus, Holm-corrected
Wilcoxon signed-rank tests over the 37 folds, Cliff's δ). We have now (a)
extended it to every eye-tracking and multimodal baseline (**new Supplementary
Table S10**: Holm-corrected Wilcoxon and Cliff's δ on balanced accuracy, MCC and
ROC-AUC), (b) reported it for the pathway ablations of R1-1 (Table S14), and
(c) rewritten Section 3.2 to state the outcome plainly: the full model's ROC-AUC
advantage is significant against Late Fusion, the Multimodal Transformer and the
ET-Transformer, but not against the Dual Transformer, Cross-Attention fusion,
DynamicGAT+ET or ET-LSTM, and no difference on balanced accuracy or MCC between
the full model and any ET/fusion baseline survives correction. The multimodal
result is now described as "parity on operating-point metrics with a modest,
partly significant advantage in ranking quality and calibration". All tests will
be recomputed against the individually tuned baselines (R2-3).

### R1-3. Report EEG decoding performance when the ET embedding / ET-related information is removed.

**Response.** Three levels of removal are now reported (Section 3.4b, Table 7,
Table S14): removing the gaze sequence (ROI vector kept), removing the ROI
vector (gaze sequence kept), and removing all gaze-derived information. Under
the common training budget the first gives balanced accuracy 0.70, ROC-AUC 0.82,
MCC 0.40 (previously reported as the EEG branch); the strictly gaze-free run is
run 5 of `docs/REVISION_RUNS.md` and its values populate the abstract, Table 3,
Table 7 and Table S14.

### R1-4. Why is an EEG-based model trained with eye-tracking information necessary, given that eye tracking measures attention directly?

**Response.** We added a paragraph to the Introduction (before the contribution
list) giving the three reasons. (1) Gaze indexes *overt* attention; engagement in
consumer neuroscience also comprises covert processing, arousal and affective
evaluation not expressed in the scanpath. The NeuMa label is a fixed rule over
frontal EEG band power *and* gaze statistics (Section 2.4, Supplementary Table
S5), so neither modality is independent of it; we therefore quantify how much of
the label a linear probe recovers from each modality's own defining terms
(Supplementary Table S15) and read every learned model against those floors.
The gaze-free EEG branch answers the reviewer's question directly: it is the
model that decodes the index without any gaze input.
(2) The modalities fail in different situations: eye tracking needs a calibrated
tracker and stimulus-locked coordinates and is uninformative for auditory,
dynamic-video or ambient formats, whereas EEG is portable and stimulus-agnostic;
evaluating the EEG pathway on its own and quantifying what gaze adds covers both
deployment settings. (3) Within the multimodal model, gaze is used as a
training-time supervisory signal (ROI guidance, cross-modal objectives) that
shapes the shared representation; Section 3.4b quantifies what this buys.

---

## Reviewer 2

### R2-1. Expand the literature review (BCINetV1; hybrid adaptive EEG tokenization; LaBraM; EEGPT).

**Response.** All four works are now cited and discussed in the Introduction
(new refs 55–58) and taken up in Future Directions, where we propose using
LaBraM/EEGPT-style pretrained encoders or BCINetV1-style convolutional attention
as drop-in EEG branches under the same leakage-aware protocol, and note that
these models are single-modality, pretrained on corpora far larger than any
neuromarketing dataset, and evaluated on motor-imagery/abnormality/emotion
targets.

### R2-2. τ = 0.30 in Eq. (3) is arbitrary; provide a sensitivity analysis on graph density and downstream metrics.

**Response.** Two analyses were added (`scripts/analysis/tau_sensitivity.py`;
Section 2.5.2, Section 3.9, **Supplementary Table S7 and Fig. S4**). *Graph
structure* (computed over all 3,850 windows of the 385 epochs): edge density
falls smoothly from 0.85 (τ=0.10) through **0.57 at τ=0.30** (mean degree 10.3)
to 0.15 (τ=0.70); fragmentation (electrodes with fewer than three edges) is
negligible up to τ=0.30 (0.5 per window) and rises steeply beyond 0.40 (2.0) and
0.50 (4.9), so τ=0.30 is the last value before the graphs begin to fragment,
which is the structural rationale for the a-priori choice. *Downstream*: the
EEG-only branch is retrained at full 37-subject scale for
τ ∈ {0.20, 0.25, 0.30, 0.35, 0.40, 0.50} (`sensitivity_sweep.py --sweep
threshold`; run 7), and the metrics per τ fill the right-hand columns of Table
S7 and the sentence in Section 3.9.

### R2-3. Baselines trained under a fixed budget without tuning — unfair comparison.

**Response.** We agree. The baseline runner (`src/model/baselines/run_baselines.py`,
now included in the repository) gained a `--tune N` mode implementing the same
nested protocol used for the proposed model: for every fold and architecture,
twelve configurations (learning rate, weight decay, dropout, batch size, from an
architecture-specific search space that always contains the earlier common
budget) are trained with early stopping on the validation subject's balanced
accuracy (patience 30, ≤250 epochs), and the configuration with the best
validation balanced accuracy is applied to the test subject; the selected
hyper-parameters are saved per fold. Section 2.6 describes the protocol, Tables
3–5 and 8 and Supplementary Tables S6/S10 will report the tuned values (run 6),
and the earlier common-budget values are retained in **Supplementary Table S8**
for transparency. The Limitations paragraph was rewritten accordingly.

### R2-4. Removing the spiking front-end costs nothing; tone down its necessity and measure energy physically.

**Response.** The claims were toned down throughout (contribution list, pipeline
table, Section 3.4, Sections 4.2 and 4.5): the encoder is no longer described as
"efficient" or "low-power" on the hardware used, only as an event-driven
representation whose energy saving is a *projection* for neuromorphic hardware.
We also measured what can be measured (`scripts/analysis/snn_energy_measured.py`,
**Supplementary Table S9**): on the commodity CPU of Section 2.5.10 the trained
LIF encoder takes 2.70 ms per single-epoch inference versus 0.45 ms for a dense
twin of identical widths (3.45 vs 0.69 ms at batch 32), i.e. it is ~6× *slower*
on conventional hardware; GPU latency and board-power-derived energy are
measured with the same script (run 12). We state explicitly that we have no
access to neuromorphic hardware, so the 2.1 % figure remains an operation-count
projection and is labelled as such.

### R2-5. The learnable bypass gate α in Eq. (18) lets the classifier ignore the rules; report α = 0 performance or the correlation between rule outputs and predictions.

**Response.** Both were implemented. The rule layer gained an `alpha_mode`
(`learned` / `rule_only` / `bypass_only`; `AblationConfig.ns_rule_only()`), and
the model now exposes the rule evidence R, the bypass logit and α at inference.
`scripts/analysis/rule_fidelity.py` reports, on the trained fold checkpoints:
the learned α per fold; the agreement between arg max R and the gated decision;
the correlation between the rule margin and the final logit margin; per-rule
Spearman correlation of a_r with P(HIGH); LOSOCV metrics when α is forced to 0
(and to 1) post hoc; and, for a model *trained* with α ≡ 0 (run 10), a paired
Wilcoxon test against the full model. Section 2.5.8 announces the analysis,
Section 3.10.3 reports it, Table 7 gains a "rule gate closed" row, and
**Supplementary Table S13** collects the numbers (runs 9–10). The text now
draws the conclusion either way: high agreement and small loss at α = 0 licenses
reading the traces as explanations; otherwise the module is retained only as an
inspection probe.

### R2-6. 385 epochs / 37 subjects is small for a complex transformer; address overfitting and generalisation to larger cohorts.

**Response.** Section 2.6 now lists the safeguards (≈740 training epochs per fold
after deterministic augmentation plus per-sample stochastic augmentation,
dropout, MMD/adversarial subject-invariance terms, seed-averaged ensemble, early
stopping on a held-out *subject*, no selection on test subjects) and the
empirical checks (LOSOCV, label-permutation test). We added a subject-count
learning curve (`scripts/analysis/learning_curve.py`; LOSOCV repeated on random
subsets of 10/16/24/30/37 participants, three draws each; **Supplementary Fig.
S5**, Section 3.9, run 13) whose slope at the full cohort is the quantity to
extrapolate to larger cohorts, and the Limitations section now states that
generalisation to larger cohorts is extrapolated from this curve rather than
demonstrated.

### R2-7. Fig. 8D shows abstract latent indices (z46, z22); project the rules onto electrodes / frequency bands.

**Response.** Done. `scripts/analysis/ground_rules_to_electrodes.py` attributes
each rule activation a_r to the physical inputs by integrated gradients (zero
baseline, 16 steps) over held-out epochs, yielding a 19 × 5 electrode-by-band map
per rule (plus the three eye-tracking streams and the ROI cells); the three
largest-magnitude terms with their sign form the grounded premise
(e.g. "Fz-theta↑ ∧ Pz-alpha↓ ∧ pupil↑ → HIGH"), and the rule head gives the
conclusion. Section 2.5.8 and 3.10.3 describe the method, the Fig. 8D caption
was rewritten, and the figure panel is regenerated from the production
checkpoint (run 11; the script also writes a full 8-rule heatmap figure for the
Supplement).

### R2-8. Per-subject threshold tuning needs test labels and violates LOSOCV; explain prospective deployment.

**Response.** The threshold in Section 3.7 never used test labels — it is a
quantile of the subject's own *unlabelled* predicted probabilities at the
training-set class prior — but we agree the text did not make this explicit and
that the rule was transductive. We added `scripts/analysis/deployment_threshold.py`
and **Supplementary Table S12**, which evaluate on the saved probabilities:
fixed/validation-subject threshold (balanced accuracy 0.753, MCC 0.51),
the transductive label-free quantile (0.767 / 0.48), and a strictly *causal*
variant in which the threshold for epoch k uses only the preceding unlabelled
epochs with the validation-subject threshold during a warm-up: 0.768 / 0.50 with
a 3-epoch (15 s) warm-up and 0.772 / 0.52 with 5 epochs. Section 3.7 and
Section 4.5 now describe this online procedure. (We also corrected the earlier
0.778 / 0.50 to the reproducible 0.767 / 0.48.)

### R2-9. Fig. 1 uses a fixed 24-region grid without evaluating alternatives; analyse grid resolution.

**Response.** The 24 boxes in Fig. 1 are the brochure's product layout and are
not a model input; the model's ROI saliency vector is a 5 × 2 gaze-occupancy
histogram, and the engagement label contains no spatial grid at all (it is built
from frontal band power and one-dimensional gaze statistics, Supplementary Table
S5). This is now stated in the Fig. 1 caption and Sections 2.4 and 2.5.3.
`scripts/analysis/roi_grid_sensitivity.py` (Section 3.9, **Supplementary Table
S11, Fig. S6**) evaluates the two levels that remain: (a) rebuilding the saliency
vector on 2×1 … 10×8 grids (including 6×4) preserves the per-epoch
attentional-concentration ordering (Spearman ρ 0.71–0.82 vs 5×2 for every grid of
≥6 cells; 0.56 for 2×1); (b) the full model is retrained with the ROI vector
rebuilt on 2×1, 3×2, 6×4 and 8×6 grids via `NEUMA_GRID_COLS/ROWS` (stage `grid`)
and compared with paired Wilcoxon tests.

---

## Other corrections made during the revision

* **Label construction corrected (Section 2.4, Supplementary Table S5).** The
  submitted manuscript described the engagement label as a gaze-only composite.
  Re-deriving the labels from the raw recordings showed that the labels of the
  reported run are produced by the dataset pipeline's *multimodal* rule
  (`engagement_phase3d.py`: five frontal-EEG band-power terms and five gaze
  statistics, fixed weights, pooled-median threshold); this rule reproduces the
  held-out labels of all 37 folds exactly (347/347), whereas the gaze-only rule
  agrees at chance level (48.7 %). Because both modalities enter the label, the
  earlier wording "leakage-independent EEG branch" was withdrawn throughout and
  replaced by "gaze-free"; a new linear-probe audit (Supplementary Table S15,
  `scripts/analysis/label_leakage_audit.py`) reports how much of the label each
  modality's own terms recover, and every learned result is read against it.
* The repository was missing `src/model/data/` (dataset loader); it is restored,
  together with the baseline implementations, so all results can be reproduced.
* The main text now states which decision thresholds are transferred,
  transductive or causal, and reports the reproducible values.
