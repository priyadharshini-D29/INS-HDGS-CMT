# `datasets/` — Dataset access & preprocessing

> ⚠️ **No raw data is redistributed in this repository.** The NeuMa dataset is
> third-party and governed by its own license/terms. This folder only documents
> how to obtain and prepare it.

## Supported dataset: NeuMa

**NeuMa** — a public neuromarketing dataset with synchronized **EEG** and
**eye-tracking** recordings from 44 participants viewing product stimuli.

> Georgiadis, K., Kalaganis, F.P., Riskos, K. *et al.* NeuMa — the absolute
> neuromarketing dataset en route to a holistic understanding of consumer
> behaviour. *Sci Data* **10**, 508 (2023).
> https://doi.org/10.1038/s41597-023-02392-9

## 1. Download

1. Obtain NeuMa from the source in the paper above (follow its access terms).
2. Place the raw per-subject files under `datasets/raw/` (git-ignored):

```
datasets/raw/
├── S01.xdf          # EEG + eye-tracking stream (LabStreamingLayer .xdf)
├── S01.xlsx         # per-subject metadata / questionnaire responses
├── S02.xdf
├── S02.xlsx
└── ...              # S01–S44 (note: S04, S11 lack per-subject engagement labels)
```

## 2. Expected folder structure after preprocessing

The preprocessing pipeline (`src/data_pipeline/`) writes derived artifacts to
its own `output/` folders (git-ignored). Stages run in order:

```
src/data_pipeline/
├── 01_validation/            integrity checks on raw .xdf/.xlsx
├── 02_signal_qc/             EEG/ET signal quality control
├── 03_preprocessing/         filtering, artifact handling, resampling
├── 04_segmentation/          epoching (5 s epochs, 75% overlap) -> output/
├── 05_feature_extraction/    band powers, connectivity, ET/ROI features
└── 06_dataset_aggregation/   per-subject + pooled global datasets -> output/global/
```

Run it with:

```bash
python -m data_pipeline.orchestration   # or see src/data_pipeline/orchestration/
```

The model reads the segmentation and aggregation outputs (see
`PHASE3_DIR` / `PHASE6_DIR` in `src/model/config/settings.py`).

## 3. Required preprocessing summary

| Parameter | Value |
|---|---|
| EEG sampling rate | 300 Hz |
| EEG epoch length | 5.0 s (75% overlap) |
| EEG channels | 19 (10–20 montage, real channels) |
| Frequency bands | delta/theta/alpha/beta/gamma |
| ET sampling rate | 120 Hz |
| ET features | gaze_x, gaze_y, pupil (+ ROI dwell/entropy) |
| Engagement label | per-subject median split → HIGH / LOW |

## 4. Sample metadata

A tiny, non-identifying sample manifest is provided so you can validate your
folder layout without the real data: [`sample_metadata.csv`](sample_metadata.csv).

## Notes

- `S04` and `S11` have no per-subject engagement labels (excluded).
- `S16, S31, S33, S41, S44` are single-class under the global threshold: kept as
  training subjects but skipped as LOSOCV **test** folds → 37 valid test folds.
