# Reproducibility Checklist

This checklist follows the spirit of the ML Reproducibility / NeurIPS checklists
and the FAIR principles. It records everything needed to reproduce the reported
results.

## Environment

| Item | Value |
|---|---|
| Language | Python ≥ 3.11 |
| Deep-learning framework | PyTorch 2.x |
| CUDA (reference run) | cu124 wheels; driver/runtime reported CUDA 13.0 on the training host |
| OS (reference run) | Linux (multi-GPU cloud instance) + Windows 11 (dev) |
| Package versions | pinned in [`requirements.txt`](../requirements.txt) / [`environment.yml`](../environment.yml) |
| Random seed | **42** (`RANDOM_SEED` in `src/model/config/settings.py`; `SEED` env var in scripts) |

## Hardware & timing

| Item | Value |
|---|---|
| GPU (reference) | 8 × NVIDIA A100-SXM4-80GB (compute capability 8.0) |
| Multi-GPU | used (fold-parallel LOSOCV auto-enabled when >1 GPU) |
| Approx. full LOSOCV training | several hours to ~1 day (37 test folds × 5–7 ensemble members) |
| Ablation study (12 variants) | multiply a single LOSOCV run accordingly |
| Memory | fits comfortably on a single 80 GB GPU; per-fold train set ≈ 300 samples |

> The reference run used an 8×A100 node with fold-parallel scheduling. A single
> GPU reproduces the same results; only wall-clock time increases. Record your
> own exact wall-clock time here when you re-run.

## Determinism

- Global seed set for `random`, `numpy`, and `torch` (+ `PYTHONHASHSEED`).
- Ensemble members intentionally use different seeds; probabilities are averaged.
- Mixed precision (AMP) and multi-GPU scheduling introduce small,
  non-deterministic floating-point variation. Expect results **close to but not
  bit-identical** to the reported numbers.

## Expected outputs

| Artifact | Location |
|---|---|
| Per-fold LOSOCV metrics | `results/losocv_metrics/*.csv` |
| ROC / PR / confusion / calibration plots | `results/roc,pr,confusion_matrices,calibration/` |
| Subject-wise breakdown | `results/subject_wise/` |
| Statistical tests | `results/statistics/` |
| Ablation comparison | `results/ablation/`, `tables/table4_ablation.csv` |
| Manuscript tables | `tables/` |
| Manuscript figures | `figures/` |

## Expected performance (LOSOCV, subject-independent)

Reference operating point from the paper's headline configuration (focal
γ=3.0, effective-number weighting, ensemble, DANN+MMD). Re-running the current
code reproduces these within ordinary run-to-run variance:

| Metric | Tolerance vs reported |
|---|---|
| Calibrated accuracy | within ~0.25 pts |
| Balanced accuracy | within ~0.21 pts |
| MCC | within ~0.012 |
| ROC-AUC | within ~0.015 |

> The exact headline values are in `results/losocv_metrics/` and `tables/`.
> The committed `results/` CSVs were produced from the original source repo at
> commit `89aaaff` ("Add INS-HDGS-CMT model code, experiment results, and
> manuscript figures").
> `reported_run_commit: 89aaaff`

## Expected directory structure after a full run

```
checkpoints/ins_hdgs_cmt_headline/fold_*/member_*.pt
results/losocv_metrics/losocv_*.csv
results/ablation/<variant>/...
tables/table*.csv
figures/Figure*/...
logs/*.log
```

## Data availability

The **NeuMa** dataset is third-party and public but **not redistributed** here.
See [`../datasets/README.md`](../datasets/README.md) for download + preprocessing.
