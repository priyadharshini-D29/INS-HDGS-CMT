# Eye-tracking behavioural phenotype of engagement (manuscript Fig. fig_et_phenotype)

Per-epoch raw gaze/pupil descriptors over all 385 epochs (193 HIGH / 192 LOW, 42 subjects).
Source arrays: et_seq (600x3: gaze x, gaze y, pupil @120 Hz) from NeumaGraphDataset.
Data: results/validation/et_phenotype.csv (per-epoch), et_phenotype_arrays.npz (pooled gaze/pupil).

| Descriptor | HIGH | LOW | Cohen's d |
|---|---|---|---|
| Gaze dispersion (spatial spread) | 0.262 | 0.199 | +1.02 |
| Scanpath length | 4.227 | 3.536 | +0.44 |
| Gaze speed | 0.847 | 0.708 | +0.44 |
| Pupil mean | 2.874 | 2.730 | +0.30 |
| Pupil range | 0.852 | 0.932 | -0.08 |

HIGH engagement = broader, more exploratory gaze + larger pupil; LOW = spatially confined.
Consistent with the manuscript case study (HIGH=wide spatial extent, LOW=narrow band).
CAVEAT: the engagement label is ET-derived, so these are a descriptive behavioural
signature of the label, NOT independent prediction -> EEG comparison is the leakage-free one.
NOTE: a 2D spatial "gaze entropy" was computed but excluded from the figure to avoid a
name-clash with the labeler's differently-defined gaze_entropy term.
