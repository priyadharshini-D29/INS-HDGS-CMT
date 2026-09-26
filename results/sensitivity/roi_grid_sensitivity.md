# ROI-grid spatial-resolution sensitivity

(A) ROI saliency vector (model input) rebuilt on alternative grids — 385 epochs, 42 subjects; reference = 5x2.

| grid | cells | max-cell share | norm. entropy | empty cells | ρ(concentration) vs ref | ρ(entropy) vs ref |
|---|---|---|---|---|---|---|
| 2x1 | 2 | 0.737 | 0.705 | 0.06 | 0.555 | 0.349 |
| 3x2 | 6 | 0.514 | 0.796 | 0.31 | 0.824 | 0.710 |
| 5x2 **(model)** | 10 | 0.412 | 0.827 | 0.41 | 1.000 | 1.000 |
| 4x3 | 12 | 0.403 | 0.817 | 0.45 | 0.705 | 0.493 |
| 6x4 *(Fig. 1 layout)* | 24 | 0.319 | 0.832 | 0.58 | 0.713 | 0.552 |
| 8x6 | 48 | 0.288 | 0.819 | 0.72 | 0.765 | 0.607 |
| 10x8 | 80 | 0.225 | 0.833 | 0.76 | 0.718 | 0.542 |

(B) The production label (engagement_phase3d.py: frontal band power + gaze statistics, global median) contains no spatial grid, so grid resolution cannot affect it.

(C) Full model retrained with r rebuilt on each grid (paired vs. the 5x2 run):

| grid | folds | BalAcc | ROC-AUC | MCC | Δ BalAcc (p) | Δ AUC (p) | Δ MCC (p) |
|---|---|---|---|---|---|---|---|
| 2x1 | 37 | 0.760 ± 0.173 | 0.899 ± 0.128 | 0.521 ± 0.335 | +0.013 (0.948) | +0.020 (0.587) | +0.036 (0.711) |
| 3x2 | 37 | 0.736 ± 0.192 | 0.886 ± 0.144 | 0.477 ± 0.368 | -0.012 (0.532) | +0.007 (0.717) | -0.008 (0.609) |
| 5x2 | 37 | 0.747 ± 0.185 | 0.879 ± 0.167 | 0.485 ± 0.359 | +0.000 (nan) | +0.000 (nan) | +0.000 (nan) |
| 6x4 | 37 | 0.720 ± 0.213 | 0.882 ± 0.160 | 0.428 ± 0.407 | -0.027 (0.379) | +0.003 (0.535) | -0.058 (0.233) |
| 8x6 | 37 | 0.716 ± 0.202 | 0.893 ± 0.158 | 0.440 ± 0.398 | -0.031 (0.208) | +0.014 (0.834) | -0.045 (0.263) |
