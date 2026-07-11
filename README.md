<div align="center">

# INS-HDGS-CMT

### An Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking Cross-Modal Transformer for Consumer Engagement Prediction Using EEG and Eye Tracking

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-blue.svg)](https://doi.org/)
[![Paper](https://img.shields.io/badge/Paper-Brain%20Informatics-b31b1b.svg)](paper/)
[![Journal](https://img.shields.io/badge/Journal-Springer%20Nature-0057B8.svg)](https://braininformatics.springeropen.com/)
[![Reproducible](https://img.shields.io/badge/Reproducible-LOSOCV-brightgreen.svg)](reproducibility/)

</div>

---

Official code, results, figures and manuscript for the paper **"An Interpretable
Neuro-Symbolic Hybrid Dynamic Graph Spiking Cross-Modal Transformer for Consumer
Engagement Prediction Using EEG and Eye Tracking"** (submitted to *Brain
Informatics*).

## Overview

INS-HDGS-CMT predicts consumer engagement (**HIGH** vs **LOW**) from synchronized
EEG and eye-tracking recordings, evaluated under **leave-one-subject-out
cross-validation (LOSOCV)** — the strict, subject-independent protocol. The
architecture:

1. represents EEG as a sequence of **dynamic functional graphs** (Pearson/PLV
   connectivity), encoded with a **graph attention network (GAT)** and a
   **spiking (LIF) encoder** for energy-efficient temporal coding;
2. encodes eye-tracking gaze/pupil dynamics with a recurrent + **ROI-attention**
   branch;
3. **fuses** the modalities with a **cross-modal transformer** (EEG ← Graph,
   EEG ← Eye-Tracking); and
4. refines the decision with a **differentiable neuro-symbolic rule layer** that
   makes the prediction *interpretable* — each decision is attributable to a
   small set of human-readable rule activations.

## Motivation

Neuromarketing needs models that generalise **across people** and **explain
themselves**. Most EEG engagement models are evaluated with subject-mixed splits
(which leak identity and inflate accuracy) and act as black boxes. INS-HDGS-CMT
targets both gaps: a subject-independent LOSOCV protocol and built-in
neuro-symbolic interpretability, while remaining efficient through spiking
temporal encoding.

## Architecture

<div align="center">
<img src="figures/Figure2_Architecture/fig2_architecture_preview.png" alt="INS-HDGS-CMT architecture" width="720"/>
</div>

Full-resolution vector figure: [`figures/Figure2_Architecture/fig2_architecture.pdf`](figures/Figure2_Architecture/fig2_architecture.pdf).
Editable source: [`figures/Figure2_Architecture/fig2_architecture_source.py`](figures/Figure2_Architecture/fig2_architecture_source.py).

## Dataset

We use the public third-party **NeuMa** neuromarketing dataset (EEG + eye
tracking). It is **not redistributed** here. See
[`datasets/README.md`](datasets/README.md) for how to download it, the expected
folder layout, and the required preprocessing.

> Georgiadis, K., Kalaganis, F.P., Riskos, K. *et al.* NeuMa — the absolute
> neuromarketing dataset en route to a holistic understanding of consumer
> behaviour. *Sci Data* **10**, 508 (2023).
> https://doi.org/10.1038/s41597-023-02392-9

## Requirements

- Python ≥ 3.11
- PyTorch 2.x (install separately, matching your CUDA version)
- See [`requirements.txt`](requirements.txt) / [`environment.yml`](environment.yml)

## Installation

```bash
git clone https://github.com/priyadharshini-D29/INS-HDGS-CMT.git
cd INS-HDGS-CMT

# 1) install PyTorch matching your machine (GPU shown; use plain `pip install torch` for CPU)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2) install the rest
pip install -r requirements.txt          # or:  conda env create -f environment.yml
pip install -e .                         # installs the `ins_hdgs_cmt` package (optional)
```

Large binary artifacts (figures, PDFs, logs, checkpoints) are tracked with
**Git LFS** — run `git lfs install` once before cloning to fetch them.

## Quick Start

```bash
# Smoke test on a single held-out subject
python src/model/main.py --subject S24 --epochs 5 --n-ensemble 1
```

## Training

```bash
bash reproducibility/train.sh
# equivalent to the paper headline configuration:
python src/model/main.py \
  --focal-gamma 3.0 --alpha-strategy effective_num \
  --n-ensemble 5 --lambda-dann 0.1 --lambda-mmd 0.1 \
  --mmd-mode marginal --norm-mode zscore
```

## Testing

```bash
pytest -q                      # unit / smoke tests (see tests/)
```

## Evaluation

```bash
bash reproducibility/evaluate.sh
```
Writes balanced accuracy, MCC, ROC-AUC, PR-AUC, Cohen's κ and F1 to `results/`.

## Ablation Studies

```bash
bash reproducibility/run_ablation.sh              # all variants (Table 6)
bash reproducibility/run_ablation.sh no_snn       # a single variant
```
See [`ablation/README.md`](ablation/README.md) for the full variant list.

## Reproducing Paper Results

```bash
bash reproducibility/run_all.sh        # env check + train + eval + ablation + tables + figures
```
Measured runtimes and expected metric values:
[`docs/REPRODUCIBILITY_CHECKLIST.md`](docs/REPRODUCIBILITY_CHECKLIST.md).

## Repository Layout

```
INS-HDGS-CMT/
├── src/                 Source code (model package + data pipeline)
│   ├── model/           GAT · SNN · cross-modal transformer · neuro-symbolic layer, training, eval, explainability
│   └── data_pipeline/   EEG/ET validation → QC → preprocessing → segmentation → features → aggregation
├── configs/             Modular YAML configs (model / training / evaluation / hyperparameters)
├── reproducibility/     One-command scripts to reproduce every experiment
├── ablation/            Ablation manifest + docs (driver in reproducibility/)
├── scripts/             Figure + analysis/statistics generators
├── results/             Metrics, ROC/PR, confusion, calibration, subject-wise, ablation
├── tables/              Every paper table (CSV / Markdown / LaTeX)
├── figures/             Every paper figure (PDF / PNG + editable source)
├── paper/               Manuscript LaTeX sources, bibliography, compiled PDF
├── supplementary/       Additional tables/figures, extended analyses
├── docs/                Architecture, checklists (reproducibility / publication / release)
├── datasets/            How to obtain & preprocess NeuMa (no raw data redistributed)
├── checkpoints/         Trained weights (Git LFS; populated by training)
├── logs/                Training / ablation logs
├── notebooks/           Exploratory notebooks
├── examples/            Small example outputs
├── tests/               Unit / smoke tests (run in CI)
└── assets/              Static images used by docs
```
Each folder contains its own `README.md` explaining what belongs there.

## Expected Outputs

- Per-fold LOSOCV metric CSVs in `results/losocv_metrics/`
- ROC / PR / confusion / calibration plots in `results/`
- Ablation comparison in `tables/table4_ablation.csv`
- Regenerated manuscript tables/figures in `tables/` and `figures/`

Reference numbers for a correct run are listed in
[`docs/REPRODUCIBILITY_CHECKLIST.md`](docs/REPRODUCIBILITY_CHECKLIST.md).

## Citation

If you use this code or results, please cite the paper (see
[`CITATION.cff`](CITATION.cff)):

```bibtex
@article{inshdgscmt2026,
  title   = {An Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking Cross-Modal
             Transformer for Consumer Engagement Prediction Using EEG and Eye Tracking},
  author  = {Priyadharshini, D. and Shridevi, S.},
  journal = {Brain Informatics},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

We thank the authors of the **NeuMa** dataset for making it publicly available.
This work builds on the open-source PyTorch, MNE-Python and scientific-Python
ecosystems.

## Contact

- **Authors:** Priyadharshini D. and Shridevi S. (corresponding author),
  Vellore Institute of Technology (VIT), Chennai, India.
- For code questions, open a
  [GitHub issue](https://github.com/priyadharshini-D29/INS-HDGS-CMT/issues); for
  research questions, use
  [Discussions](https://github.com/priyadharshini-D29/INS-HDGS-CMT/discussions).
