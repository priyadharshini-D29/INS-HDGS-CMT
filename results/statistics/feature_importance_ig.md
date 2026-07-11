# Integrated-Gradients feature importance (LOSOCV)

Attribution to HIGH_ENGAGEMENT logit, 37 folds, 347 epochs, 24-step IG, mean (per-feature) baseline. Importance = normalised mean |IG| per input element.

| Modality | Feature | Importance | Interpretation |
|---|---|---|---|
| ROI | ROI saliency | 0.523 | Attended stimulus region |
| EEG | Posterior alpha power | 0.176 | Visual attention allocation (alpha suppression) |
| EEG | Frontal theta power | 0.160 | Attentional control / engagement |
| ET | Gaze position | 0.131 | Overt fixation location |
| ET | Pupil dynamics | 0.011 | Arousal / cognitive load |
| EEG | Frontal functional connectivity | 0.000 | Fronto-cortical network integration |
