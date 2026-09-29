# Integrated-Gradients feature importance (LOSOCV)

Attribution to HIGH_ENGAGEMENT logit, 37 folds, 347 epochs, 16-step IG, mean (per-feature) baseline. Importance = normalised mean |IG| per input element.

| Modality | Feature | Importance | Interpretation |
|---|---|---|---|
| EEG | Posterior alpha power | 0.269 | Visual attention allocation (alpha suppression) |
| ET | Gaze position | 0.262 | Overt fixation location |
| ROI | ROI saliency | 0.229 | Attended stimulus region |
| EEG | Frontal theta power | 0.218 | Attentional control / engagement |
| ET | Pupil dynamics | 0.023 | Arousal / cognitive load |
| EEG | Frontal functional connectivity | 0.000 | Fronto-cortical network integration |
