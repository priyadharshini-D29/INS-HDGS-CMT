# NEUMA PHASE 3 — Event Segmentation

Multimodal EEG + Eye Tracking epoch pipeline for the NeuMa neuromarketing dataset.

## Prerequisites

Phase 2 must have run and saved:
```
NEUMA_PHASE2/output/clean_data/
    eeg_clean.npy
    et_synced.npy
    eeg_timestamps.npy   ← add to Phase 2 main if missing
    et_timestamps.npy    ← add to Phase 2 main if missing
```

Event markers must be exported to:
```
output/events/markers.csv   (columns: timestamp, label)
```

## Run

```bash
cd NEUMA_PHASE3
python main_phase3.py
```

## Outputs

```
output/epochs/
    eeg_epochs.npy   — object array of variable-length EEG epochs (T, C)
    et_epochs.npy    — object array of variable-length ET epochs  (T, C)
    labels.npy       — string label per epoch
output/metadata/
    metadata.csv     — timestamp, label, sample counts, fixations, ROI dwell
output/plots/
    sample_eeg_epoch.png
    sample_et_epoch.png
    epoch_summary.png
    fixation_density.png
```

## Scientific design decisions

| Decision | Reason |
|---|---|
| No global ET resampling | Preserves native 120 Hz timing; avoids artificial interpolation |
| Blink NaN gaps preserved | Blinks are a real physiological event; filling them fakes data |
| Timestamp-mask windowing | Exact temporal boundaries without sample-count assumptions |
| Variable-length epochs saved as object arrays | Epochs near recording edges may be shorter |
| LOSOCV-ready | Subject ID can be used as fold key in metadata.csv |
