# Release Notes — v1.0.0

**INS-HDGS-CMT — initial public release**
_An Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking Cross-Modal
Transformer for Consumer Engagement Prediction Using EEG and Eye Tracking._

## Highlights
- Full model source: dynamic functional-graph + GAT encoder, spiking (LIF)
  temporal encoder, cross-modal fusion transformer, and a differentiable
  neuro-symbolic rule layer for interpretability.
- Subject-independent **LOSOCV** evaluation pipeline (37 valid test folds).
- Complete **ablation study** (12 variants) with one-command reproduction.
- End-to-end reproducibility scripts (`reproducibility/`) and modular YAML
  configs (`configs/`).
- All manuscript **tables** and **figures** with editable sources.
- Explainability: integrated gradients + neuro-symbolic rule extraction.

## What's included
- `src/` model package and data pipeline
- `configs/` modular YAML configuration
- `reproducibility/` + `ablation/` scripts
- `results/`, `tables/`, `figures/`, `paper/`, `supplementary/`
- `docs/` architecture + reproducibility/publication/release checklists
- CI, issue/PR templates, `CITATION.cff`, MIT `LICENSE`

## What's not included
- The raw **NeuMa** dataset (third-party; see `datasets/README.md`).
- Pre-trained checkpoints if they exceed hosting limits — regenerate with
  `reproducibility/train.sh`.

## Reproducibility note
Committed `results/` CSVs come from the original LOSOCV run. Re-running the
current code reproduces the paper's figures within ordinary run-to-run variance
(calibrated accuracy within ~0.25 pts, MCC within ~0.012, AUC within ~0.015),
not bit-for-bit. See `docs/REPRODUCIBILITY_CHECKLIST.md`.

## Known limitations
- Small per-subject folds make individual-subject AUC noisy; single-class
  subjects are excluded as test folds.
- AMP + multi-GPU scheduling introduce minor non-determinism.

## Citation
See `CITATION.cff`. DOI to be minted via Zenodo on release.
