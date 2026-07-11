# Label-validity note & reframed claims (LOSOCV study)

## Finding (confirmed by code audit)

Engagement labels are **not** behavioural ground truth. They are produced by
`labeling/engagement_labeler.py`:

```
score = 0.30·fixation_ratio + 0.25·mean_dwell_time + 0.20·roi_density
      + 0.15·pupil_dilation_norm + 0.10·revisit_count − 0.30·gaze_entropy
label = 1[score ≥ median(score)]
```

All six terms are **eye-tracking gaze features**; no EEG enters the label at any
stage (the `engagement_phase3d` "multimodal (EEG+ET)" tag is a directory name,
and `reprocess_labels.py` only re-thresholds this same ET-derived score). No
independent behavioural labels (purchase intent / ratings) exist in the repo.

## Consequence (with measured magnitude)

Models that consume the raw ET stream (ET-LSTM/GRU/Transformer and all fusion
baselines) **partially observe the label-generating signal**, so their scores are
optimistically biased. The magnitude is **moderate, not near-deterministic**: a
subject-grouped logistic regression from the six label-defining gaze features
recovers the labels at **ROC-AUC ≈ 0.77** (leakage-free CV; `leakage_audit.py`
Check 3b), well above chance (0.50) but below the ET-LSTM's own 0.905. Two
readings are consistent with this: the raw ET *sequence* carries engagement
signal beyond the six summary features, and/or the re-derived composite does not
exactly reproduce the original pre-computed labeler. Either way the labels are
**not independent of the ET modality**, so the ET/fusion comparison cannot serve
as a leakage-free demonstration that those architectures decode engagement
better. EEG encoders have no access to the gaze signal, so the EEG comparison is
the leakage-free ranking — and this conclusion holds regardless of the exact
recoverability value.

## Reframed, defensible claims

1. **Headline (Table 1, leakage-free):** the proposed EEG branch
   (DynamicGAT + LIF) significantly outperforms eight established EEG encoders
   (EEGNet, ShallowConvNet, DeepConvNet, CNN-LSTM, CNN-BiLSTM, EEG-Transformer,
   TSception, GAT) on balanced accuracy, MCC and ROC-AUC — Holm-corrected
   Wilcoxon p < 0.05 for every comparison; Friedman p ≈ 1e-5–1e-7.
   Proposed EEG-only: BalAcc 0.697, ROC-AUC 0.819, MCC 0.398
   vs best baseline (EEGNet) 0.619 / 0.602 / 0.186.
2. **Ablation (leakage-free component that matters):** removing DynamicGAT is the
   only change that significantly degrades performance (Δ bal-acc −0.057,
   p = 0.007). LIF spiking, ROI guidance, the NeuroFusion transformer and the
   neuro-symbolic layer show no significant accuracy contribution — report these
   honestly (e.g. interpretability/efficiency roles, not accuracy gains).
3. **ET / fusion (Tables 2–3):** present as a *modality-information* analysis
   with the confound caveat; do **not** claim the full model beats them, since
   that comparison is not leakage-free.

## Suggested manuscript limitation paragraph (LaTeX, ready to paste)

```latex
\paragraph{Label-derivation limitation.}
Engagement labels in this study are not behavioural ground truth; they are
obtained by median-thresholding a weighted composite of six eye-tracking gaze
features (fixation ratio, dwell time, ROI density, pupil dilation, revisit count
and gaze entropy). Consequently, encoders that ingest the raw eye-tracking
stream partially observe the label-generating signal, and their reported
performance is optimistically biased. We therefore designate the EEG-encoder
comparison---whose models have no access to the gaze signal---as the
leakage-free headline evaluation, and present the eye-tracking and multimodal
fusion results as a modality-information analysis rather than a leakage-free
ranking. Establishing behaviourally validated engagement labels (e.g. purchase
intent or self-reported ratings) is left to future work.
```

## Recommended audit hardening

`evaluation/leakage_audit.py` checks data isolation, duplicates, shuffled-label
sanity and graph leakage, but **not label-source circularity**. Adding a probe
that flags when labels are a deterministic function of any input modality would
catch this class of issue before publication.
```
