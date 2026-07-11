# `results/` — Experimental outputs

Numerical results and plots produced by the evaluation pipeline. Subfolders:

| Folder | Contents |
|---|---|
| `losocv_metrics/` | Per-fold LOSOCV metric CSVs (headline numbers) |
| `accuracy/` | Accuracy summaries |
| `roc/` · `pr/` | ROC and precision–recall curves/data |
| `confusion_matrices/` | Per-fold / aggregate confusion matrices |
| `calibration/` | Reliability diagrams, ECE |
| `subject_wise/` | Per-subject performance breakdown |
| `attention_maps/` · `gradcam/` · `saliency/` | Model attribution maps |
| `feature_importance/` | Integrated-gradients feature importance |
| `statistics/` | Significance tests, ranks |
| `ablation/` | Per-variant ablation outputs |
| `baselines/` · `validation/` · `threshold_analysis/` · `case_study/` | Comparisons and analyses |

Regenerate with `../reproducibility/evaluate.sh`. See the reproducibility note
in the top-level README about run-to-run variance.
