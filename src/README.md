# `src/` — Source code

Two importable packages:

- **`model/`** — the INS-HDGS-CMT model and everything around it: graph
  construction (`graphs/`), architecture (`models/` — GAT, spiking encoder,
  cross-modal transformer, neuro-symbolic layer), `training/`, `evaluation/`,
  `explainability/`, `inference/`, `labeling/`, `data/`, `utils/`, `config/`,
  plus the CLI entry point `main.py` and analysis utilities.
- **`data_pipeline/`** — the staged EEG/ET preprocessing pipeline
  (`01_validation` → `02_signal_qc` → `03_preprocessing` → `04_segmentation`
  → `05_feature_extraction` → `06_dataset_aggregation`), orchestrated from
  `orchestration/`.

Entry points:
```
python src/model/main.py            # train/evaluate one LOSOCV configuration
python src/model/main.py --ablation <variant>
python src/model/main.py --explain  # neuro-symbolic rules + integrated gradients
```
Configuration defaults live in `model/config/settings.py`; YAML mirrors are in
`../configs/`.
