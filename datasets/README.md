# Datasets

This repository does **not** redistribute the NeuMa recordings. Download them from
the original deposit and place them here (everything in `datasets/` except this
file is ignored by git).

## NeuMa (Georgiadis, Kalaganis, Riskos et al., 2023)

| Item | Reference |
|---|---|
| Data descriptor | Georgiadis K, Kalaganis FP, Riskos K, et al. NeuMa — the absolute neuromarketing dataset en route to an holistic understanding of consumer behaviour. *Scientific Data* 10:508 (2023). https://doi.org/10.1038/s41597-023-02392-9 |
| Raw release (figshare, CC BY 4.0) | https://doi.org/10.6084/m9.figshare.22117001 |
| Preprocessed release (figshare, CC BY 4.0) | https://doi.org/10.6084/m9.figshare.22117124 |

What the study uses (see the manuscript, Section 2.1):

- 44 participants released; the 42 with per-subject engagement labels are analysed,
  of which 37 are evaluable test folds under the global label (5 single-class
  participants are kept for training only).
- EEG: Wearable Sensing DSI-24 at 300 Hz. Only the 19 scalp electrodes of the
  10–20 system are used (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4,
  T5, T6, Fz, Cz, Pz); the auxiliary/EOG, mastoid-reference and trigger channels
  are excluded from graph construction.
- Eye tracking synchronised with the EEG; 5-s epochs; the engagement label is the
  dataset's fixed rule over EEG band power and gaze statistics.

## Expected layout

```
datasets/
├── README.md          this file (tracked)
└── raw/               downloaded NeuMa files (ignored by git)
```

Run the data pipeline stages in `src/data_pipeline/` in order
(`01_validation` → `02_signal_qc` → `03_preprocessing` → `04_segmentation` →
`05_feature_extraction` → `06_dataset_aggregation`); each stage's README states
its inputs and outputs. The model then reads the aggregated epochs (see
`src/model/config/settings.py` for the paths and the label mode).

## Licence

NeuMa is released under CC BY 4.0 by its authors; cite the descriptor and the
deposit you use. Nothing in this folder is covered by this repository's MIT
licence.
