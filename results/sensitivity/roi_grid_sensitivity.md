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

(B) Engagement-label stability when the labelling pipeline's own spatial resolutions are varied (385 stimulus epochs; published labels use 8 entropy bins, a 4x4 revisit grid and a 60% central region).

| parameter | value | ρ(score) vs published | κ(label) vs published | labels flipped | HIGH fraction |
|---|---|---|---|---|---|
| entropy_bins | 4.0 | 0.966 | 0.823 | 8.8% | 0.499 |
| entropy_bins | 6.0 | 0.970 | 0.865 | 6.8% | 0.499 |
| entropy_bins | 8.0 **(published)** | 1.000 | 1.000 | 0.0% | 0.499 |
| entropy_bins | 12.0 | 0.968 | 0.834 | 8.3% | 0.499 |
| entropy_bins | 16.0 | 0.963 | 0.834 | 8.3% | 0.499 |
| revisit_grid | 2.0 | 0.722 | 0.532 | 23.4% | 0.499 |
| revisit_grid | 3.0 | 0.834 | 0.616 | 19.2% | 0.499 |
| revisit_grid | 4.0 **(published)** | 1.000 | 1.000 | 0.0% | 0.499 |
| revisit_grid | 6.0 | 0.897 | 0.761 | 11.9% | 0.499 |
| revisit_grid | 8.0 | 0.893 | 0.761 | 11.9% | 0.499 |
| central_fraction | 0.4 | 0.851 | 0.657 | 17.1% | 0.499 |
| central_fraction | 0.5 | 0.928 | 0.771 | 11.4% | 0.499 |
| central_fraction | 0.6 **(published)** | 1.000 | 1.000 | 0.0% | 0.499 |
| central_fraction | 0.7 | 0.945 | 0.792 | 10.4% | 0.499 |
| central_fraction | 0.8 | 0.856 | 0.647 | 17.7% | 0.499 |

(C) Model side: no `grid_<c>x<r>` LOSOCV runs found yet. Launch on the GPU server:

```
NEUMA_GRID_COLS=2 NEUMA_GRID_ROWS=1 python main.py --fold-parallel --label grid_2x1
NEUMA_GRID_COLS=3 NEUMA_GRID_ROWS=2 python main.py --fold-parallel --label grid_3x2
NEUMA_GRID_COLS=4 NEUMA_GRID_ROWS=3 python main.py --fold-parallel --label grid_4x3
NEUMA_GRID_COLS=6 NEUMA_GRID_ROWS=4 python main.py --fold-parallel --label grid_6x4
NEUMA_GRID_COLS=8 NEUMA_GRID_ROWS=6 python main.py --fold-parallel --label grid_8x6
NEUMA_GRID_COLS=10 NEUMA_GRID_ROWS=8 python main.py --fold-parallel --label grid_10x8
```
