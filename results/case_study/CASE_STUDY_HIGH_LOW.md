# Case Study — HIGH vs LOW Engagement: Complete Workflow

Two representative held-out epochs traced through the **same** canonical
INS-HDGS-CMT ensemble (`repro_focal_g3p0_effective_num_37`), one per class:

| | HIGH case | LOW case |
|---|---|---|
| Subject / fold | **S24** / fold-22 | **S30** / fold-28 |
| Representative epoch | 11 | 7 |
| Ground truth | HIGH | LOW |
| Model decision | HIGH ✅ | LOW ✅ |
| Confidence | p(HIGH) = **0.698** | p(HIGH) = **0.270** (p(LOW) = 0.730) |

Both are the **most-confident correctly-classified** epoch of their class for
that subject — clean exemplars at opposite poles of the decision.

Figures: [fig_regions](../../figures/case_study_high_low/fig_regions.png) ·
[fig_gaze_pred](../../figures/case_study_high_low/fig_gaze_pred.png) ·
full S24 layer-traversal panels in [figures/case_study_S24/](../../figures/case_study_S24).
Table: [high_low_comparison.md](high_low_comparison.md) / `.tex` / `.csv`.

---

## 1. The workflow (what each module does for the two epochs)

Both epochs pass through the identical pipeline; the table shows what each stage
produces for HIGH vs LOW.

| Stage | Module | HIGH (S24) | LOW (S30) |
|---|---|---|---|
| ① Input | EEG band-power (10×24×5), gaze (600×3), ROI (10) | — | — |
| ② Spiking encoding | LIF SNN | sparse spikes from band-power | sparse spikes |
| ③ **Dynamic graph** | DynamicGAT | attends fronto-posterior edges | attends edges |
| ④ ROI modulation | ROIGraphMod + ROIAttention | **sustained** ROI saliency (windows 5–8) | ROI saliency **collapses** after window 3 → 0 |
| ⑤ ET encoding | ET-Transformer | broad gaze exploration | gaze confined to narrow band |
| ⑥ Fusion | NeuroFusion Transformer | integrates EEG + gaze + ROI | dominated by ROI signal |
| ⑦ Decision | Classifier | **p(HIGH)=0.70** | **p(LOW)=0.73** |
| ⑧ Explanation | IG + neuro-symbolic | distributed attribution | ROI-dominated attribution |

---

## 2. How the regions differ

**Within-subject HIGH−LOW contrast** (removes cross-subject z-score confound —
compares each subject's own HIGH vs LOW epochs):

| Marker | HIGH case (S24) | LOW case (S30) | Expected | Verdict |
|---|---|---|---|---|
| Δ Frontal **θ** (HIGH−LOW) | **+0.016** | **+0.012** | >0 (theta ↑ when engaged) | ✅ consistent in both |
| Δ Posterior **α** (HIGH−LOW) | +0.005 | −0.004 | <0 (alpha suppressed when engaged) | ⚠️ weak / mixed |

- **Frontal theta** behaves as theory predicts in *both* subjects: higher in
  HIGH-engagement epochs → **frontal attentional-control signature**.
- **Posterior alpha** is weak and inconsistent at the single-subject level —
  exactly matching the population concordance result (alpha effect d≈−0.03,
  p≈0.40; theta is the stronger marker). The model does **not** rely on a single
  band; it integrates distributed structure via the dynamic graph.
- **Functional connectivity** (off-diagonal, [fig_regions](../../figures/case_study_high_low/fig_regions.png))
  is similar in bulk magnitude (frontal ≈0.23 both) — the discriminative
  information is in the *attention-weighted graph pattern*, not raw edge strength
  (consistent with IG-connectivity ≈ 0).

---

## 3. How the result is achieved (the decision mechanism)

The clearest difference is **temporal ROI dynamics + attribution**, in
[fig_gaze_pred](../../figures/case_study_high_low/fig_gaze_pred.png):

**HIGH (S24):**
- **ROI saliency is sustained** across the epoch — meaningful values in windows
  5–8 (peak ≈0.32) → the viewer keeps attending salient content over time.
- **Gaze explores** a wide 2-D area (scanning the stimulus).
- **Attribution is distributed:** IG ≈ EEG 0.36 · ROI 0.33 · Gaze 0.31 → the
  model fuses *covert* (EEG) and *overt* (gaze/ROI) attention to reach HIGH.

**LOW (S30):**
- **ROI saliency collapses early** — concentrated in windows 0–3 (peak ≈0.62)
  then **drops to zero** for windows 4–9 → attention disengages after an initial
  glance.
- **Gaze is confined** to a narrow horizontal band (little content exploration).
- **Attribution is ROI-dominated:** IG ≈ ROI **0.89** · EEG 0.08 · Gaze 0.03 →
  the LOW decision is driven by the *absence of sustained salient fixation*.

**Interpretation:** HIGH engagement = sustained, multimodally-coherent attention
(brain + eye agree over time); LOW engagement = early attentional drop-off the
model detects primarily from the decaying ROI-saliency trajectory.

---

## 4. Per-class detection context (why these are representative)
Pooled over all 37 folds, the model detects **LOW** with precision 0.82 /
recall 0.65 and **HIGH** with precision 0.71 / recall 0.85; balanced accuracy
0.74 weights both equally. S24/S30 are confident, correctly-classified members
of each class.

## 5. Honest caveats
- Single-epoch band-power is per-subject z-scored, so **absolute** cross-subject
  magnitudes are not directly comparable; the within-subject Δ and the temporal
  ROI/gaze dynamics are the meaningful contrasts.
- Engagement labels are gaze-derived, so the ROI/gaze dominance in the LOW case
  partly reflects the label definition; the EEG (frontal-theta) contrast is the
  modality-independent signal.

*Sources:* `analysis/case_study_high_low.py`,
`results/case_study/high_low_comparison.{csv,md,tex}`,
`results/case_study/high_low_provenance.json`, canonical ensemble fold-22 / fold-28.
