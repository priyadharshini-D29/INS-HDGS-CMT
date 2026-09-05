<div align="center">

# INS-HDGS-CMT

### Dynamic Functional Graph Learning for Subject-Independent Consumer Engagement Decoding from EEG and Eye Tracking: A Leakage-Aware NeuMa Study

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Brain%20Informatics-b31b1b.svg)](paper/)
[![Status](https://img.shields.io/badge/Status-Under%20review-orange.svg)](#citation)
[![Evaluation](https://img.shields.io/badge/Evaluation-LOSOCV-brightgreen.svg)](reproducibility/)

**Priyadharshini D** · **Shridevi S**
School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India

</div>

---

Official code, results, figures and manuscript sources for **"Dynamic Functional
Graph Learning for Subject-Independent Consumer Engagement Decoding from EEG and
Eye Tracking: A Leakage-Aware NeuMa Study"**, submitted to *Brain Informatics*
(Springer Nature).

## Overview

Consumer engagement is central to advertising effectiveness, yet self-report
measures are retrospective and biased. EEG and eye tracking provide
complementary objective measures, but existing frameworks often rely on simple
fusion, subject-mixed evaluation that overstates generalisation, and static
within-trial connectivity.

**INS-HDGS-CMT** — the Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking
Cross-Modal Transformer — is a subject-independent framework that:

1. represents each EEG epoch as a **sequence of dynamic functional-connectivity
   graphs**, encoded with hierarchical **graph-attention** and **spiking (LIF)**
   modules;
2. encodes gaze/pupil dynamics with a recurrent + **ROI-attention** branch;
3. **fuses** the modalities with a **cross-modal transformer**; and
4. exposes each decision through a **soft-rule inspection module**.

Evaluation uses **leave-one-subject-out cross-validation (LOSOCV)** on the public
**NeuMa** dataset — the strict, subject-independent protocol.

<div align="center">
<img src="assets/architecture.png" alt="INS-HDGS-CMT architecture" width="760"/>
</div>

Full-resolution vector figure: [`paper/figures/fig2_Architecture.pdf`](paper/figures/fig2_Architecture.pdf).

## Leakage-aware evaluation — please read first

Engagement labels in NeuMa are **derived from gaze features**. Any model that
receives eye tracking as input is therefore partly reconstructing its own label
source. This repository follows the manuscript in keeping the two claims
strictly separate:

| Claim | Branch | ROC-AUC | Status |
|---|---|---|---|
| **Leakage-independent (headline)** | **EEG-only** — never accesses gaze | **0.82** | Primary result |
| Label-coupled (secondary) | Full multimodal (EEG + ET) | 0.90 | Secondary analysis, **not** independent prediction |

The EEG-only branch significantly outperformed **eight** EEG baseline
architectures on ROC-AUC and MCC after Holm correction. Ablation identified
**dynamic functional-graph modelling** as the component whose removal most
degraded performance, and the only ablation remaining significant after
correction.

## Figures — paper → repository

Manuscript figures live in [`paper/figures/`](paper/figures/). Filenames are kept
exactly as the manuscript sources reference them so the paper still compiles;
this table maps them to the paper's numbering.

| Paper | Caption (abridged) | File |
|---|---|---|
| Fig. 1 | Representative synchronised multimodal (EEG, gaze, pupil) epoch, subject S24 | [`fig1_real_overlay.pdf`](paper/figures/fig1_real_overlay.pdf) |
| Fig. 2 | Overview of the INS-HDGS-CMT architecture | [`fig2_Architecture.pdf`](paper/figures/fig2_Architecture.pdf) |
| Fig. 3 | Dynamic EEG functional-graph construction | [`fig_graph.pdf`](paper/figures/fig_graph.pdf) |
| Fig. 4 | Cross-modal fusion and neuro-symbolic reasoning | [`fig4_cross-modal.pdf`](paper/figures/fig4_cross-modal.pdf) |
| Fig. 5 | Classification performance under LOSOCV | [`fig3_losocv_results.pdf`](paper/figures/fig3_losocv_results.pdf) |
| Fig. 6 | EEG leakage-controlled comparison (Nemenyi critical difference) | [`fig6_combined.pdf`](paper/figures/fig6_combined.pdf) |
| Fig. 7 | Engagement is encoded multivariately, not by any single marker | [`fig_concordance_depth.pdf`](paper/figures/fig_concordance_depth.pdf) |
| Fig. 8 | Explainability for one held-out subject (S01) | [`fig5_explainability.pdf`](paper/figures/fig5_explainability.pdf) |
| Fig. 9 | Eye-tracking phenotype of engagement (385 epochs) | [`fig_et_phenotype.pdf`](paper/figures/fig_et_phenotype.pdf) |
| Fig. 10 | Gaze, ROI-saliency, decision and attribution (S24 / S30) | [`fig_gaze_pred.pdf`](paper/figures/fig_gaze_pred.pdf) |

> **Note:** figure *filenames* do not follow the paper's figure *numbers* — e.g.
> `fig3_losocv_results.pdf` is **Figure 5** and `fig5_explainability.pdf` is
> **Figure 8**. Use this table or [`paper/FIGURES.md`](paper/FIGURES.md).

## Tables — paper → repository

Typeset fragments in [`tables/latex/`](tables/latex/); machine-readable values in
[`tables/`](tables/).

| Paper | Subject | Typeset | Data |
|---|---|---|---|
| Table 1 | NeuMa dataset characteristics | [`table1_dataset.tex`](tables/latex/table1_dataset.tex) | — |
| Table 2 | Implementation details / hyperparameters | [`table2_implementation.tex`](tables/latex/table2_implementation.tex) | — |
| Table 3 | EEG-encoder comparison (leakage-free headline) | [`table3_eeg_encoders.tex`](tables/latex/table3_eeg_encoders.tex) | [`table1_eeg_encoders.csv`](tables/table1_eeg_encoders.csv) |
| Table 4 | Eye-tracking encoders | [`table4_et_encoders.tex`](tables/latex/table4_et_encoders.tex) | [`table2_et_encoders.csv`](tables/table2_et_encoders.csv) |
| Table 5 | Multimodal fusion (label-coupled) | [`table5_fusion.tex`](tables/latex/table5_fusion.tex) | [`table3_fusion.csv`](tables/table3_fusion.csv) |
| Table 6 | Contextual Cohen's κ vs prior NeuMa work | *manuscript only* | — |
| Table 7 | Component ablation | [`table6_ablation.tex`](tables/latex/table6_ablation.tex) | — |
| Table 8 | Proposed EEG branch vs each baseline (Wilcoxon, Holm) | [`table7_significance.tex`](tables/latex/table7_significance.tex) | [`ranks_eeg_mcc.csv`](tables/ranks_eeg_mcc.csv) |
| Pipeline table | Eight-stage processing pipeline | *manuscript only* | — |

> **Note:** the last two typeset filenames are offset from the paper's numbering —
> `table6_ablation.tex` is **Table 7**, `table7_significance.tex` is **Table 8**.
> `table8_ig_features.tex` and `table9_case_study.tex` support the supplementary
> material.


## Revision analyses (reviewer response)

Server runbook: [`docs/BREV_RUNBOOK.md`](docs/BREV_RUNBOOK.md) — `bash scripts/revision/run_revision.sh all` runs every revision stage and `scripts/revision/verify_revision.py` re-checks the outputs.

The revision for *Brain Informatics* added the following, all under
[`scripts/analysis/`](scripts/analysis/) with outputs in [`results/`](results/);
the point-by-point letter is [`docs/RESPONSE_TO_REVIEWERS.md`](docs/RESPONSE_TO_REVIEWERS.md)
and the list of GPU-server runs still to execute is
[`docs/REVISION_RUNS.md`](docs/REVISION_RUNS.md).

| Script | Reviewer point | What it does |
|---|---|---|
| `cross_modal_contribution.py` | R1-1/2/3 | paired Wilcoxon/Holm/Cliff's δ: full vs. no-gaze-sequence, no-ROI, no-fusion, EEG-only, ET-only |
| `deployment_threshold.py` | R2-8 | label-free per-subject thresholds, transductive and strictly causal (online) |
| `tau_sensitivity.py` | R2-2 | graph density / fragmentation vs. τ over all windows; merges the downstream τ sweep |
| `roi_grid_sensitivity.py` | R2-9 | ROI-saliency, label and model sensitivity to the spatial grid |
| `rule_fidelity.py` | R2-5 | learned bypass gate α, rule/decision agreement, post-hoc α = 0 / 1, rule-only-trained model |
| `ground_rules_to_electrodes.py` | R2-7 | integrated-gradient projection of each soft rule onto electrodes × bands (replaces latent z_k indices) |
| `snn_energy_measured.py` | R2-4 | measured latency / GPU power of the LIF encoder vs. a dense twin, next to the neuromorphic projection |
| `learning_curve.py` | R2-6 | LOSOCV on random subject subsets of increasing size |
| `src/model/baselines/run_baselines.py --tune N` | R2-3 | nested per-architecture hyper-parameter search with early stopping on the validation subject |

Supporting code changes: `NeuroSymbolicRuleLayer(alpha_mode=...)`,
`AblationConfig.ns_rule_only()/ns_explain_only()`, env overrides
`NEUMA_NS_ALPHA_MODE`, `NEUMA_GRID_COLS`, `NEUMA_GRID_ROWS`; the dataset loader
`src/model/data/` (previously missing from the public tree) is restored.

## Dataset

This study analyses the **publicly available NeuMa dataset** (EEG + eye
tracking). **No new data were generated in this work**, and raw recordings are
**not redistributed** here. See [`datasets/README.md`](datasets/README.md) for
how to download it, the expected folder layout and the required preprocessing.

> Georgiadis, K., Kalaganis, F.P., Riskos, K. *et al.* NeuMa — the absolute
> neuromarketing dataset en route to a holistic understanding of consumer
> behaviour. *Sci Data* **10**, 508 (2023).
> https://doi.org/10.1038/s41597-023-02392-9

## Requirements

- Python ≥ 3.11 (the paper used Python 3.12)
- PyTorch 2.x — install separately, matching your CUDA version (paper: PyTorch
  2.7.1 / CUDA 12.6)
- See [`requirements.txt`](requirements.txt) / [`environment.yml`](environment.yml)

## Installation

Large binary artifacts (figures, PDFs, logs, checkpoints) are tracked with **Git
LFS** — run `git lfs install` once *before* cloning to fetch them.

```bash
git lfs install
git clone https://github.com/priyadharshini-D29/INS-HDGS-CMT.git
cd INS-HDGS-CMT

# 1) install PyTorch matching your machine (GPU shown; use plain `pip install torch` for CPU)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2) install the rest
pip install -r requirements.txt          # or:  conda env create -f environment.yml
pip install -e .                         # installs the `ins_hdgs_cmt` package (optional)
```

## Quick start

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

## Evaluation

```bash
bash reproducibility/evaluate.sh
```

Writes balanced accuracy, MCC, ROC-AUC, PR-AUC, Cohen's κ and F1 to `results/`.

## Ablation studies

```bash
bash reproducibility/run_ablation.sh              # all variants (paper Table 7)
bash reproducibility/run_ablation.sh no_snn       # a single variant
```

See [`ablation/README.md`](ablation/README.md) for the full variant list.

## Reproducing the paper

```bash
bash reproducibility/reproduce_paper.sh   # full pipeline
bash reproducibility/run_all.sh           # env check + train + eval + ablation + tables + figures
```

Or stage by stage:

```bash
bash reproducibility/train.sh              # LOSOCV training
bash reproducibility/evaluate.sh           # metrics
bash reproducibility/run_ablation.sh       # ablation
bash reproducibility/generate_tables.sh    # tables/
bash reproducibility/generate_figures.sh   # paper/figures/
```

All experiments use a fixed base random seed (**42**) applied identically to the
Python, NumPy and PyTorch generators across folds. Hyperparameters, optimiser
configuration and training schedule are listed in full in **paper Table 2**.
Measured runtimes and expected metric values:
[`docs/REPRODUCIBILITY_CHECKLIST.md`](docs/REPRODUCIBILITY_CHECKLIST.md).

> LOSOCV over all subjects is GPU-intensive. CI runs only the smoke tests in
> [`tests/`](tests/); it does not train.

## Testing

```bash
pytest -q                      # smoke tests (see tests/)
```

## Repository layout

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
├── paper/               Manuscript LaTeX sources, bibliography, compiled PDF
│   └── figures/         Every manuscript figure, as referenced by the .tex
├── supplementary/       Additional tables/figures, extended analyses
├── docs/                Architecture, checklists (reproducibility / publication / release)
├── datasets/            How to obtain & preprocess NeuMa (no raw data redistributed)
├── checkpoints/         Trained weights (Git LFS; populated by training)
├── logs/                Training / ablation logs
├── notebooks/           Exploratory notebooks
├── examples/            Small example outputs
├── tests/               Smoke tests (run in CI)
└── assets/              Static images used by docs
```

Each folder contains its own `README.md` explaining what belongs there.

## Expected outputs

- Per-fold LOSOCV metric CSVs in `results/losocv_metrics/`
- ROC / PR / confusion / calibration plots in `results/`
- Regenerated manuscript tables in `tables/` and figures in `paper/figures/`

Reference numbers for a correct run are listed in
[`docs/REPRODUCIBILITY_CHECKLIST.md`](docs/REPRODUCIBILITY_CHECKLIST.md).

## Citation

If you use this code or results, please cite the paper (machine-readable
metadata in [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{inshdgscmt2026,
  title   = {Dynamic Functional Graph Learning for Subject-Independent Consumer
             Engagement Decoding from EEG and Eye Tracking: A Leakage-Aware NeuMa Study},
  author  = {D, Priyadharshini and S, Shridevi},
  journal = {Brain Informatics},
  year    = {2026},
  note    = {Under review}
}
```

Please also cite the **NeuMa dataset** (Georgiadis et al., 2023).

## License

Released under the [MIT License](LICENSE). The NeuMa dataset is subject to its
own licence and terms.

## Acknowledgements

We thank the authors of the **NeuMa** dataset for making it publicly available,
and **NVIDIA Corporation** for GPU computing resources provided through the
NVIDIA Brev platform, which accelerated training, hyperparameter optimisation and
subject-independent evaluation. This work builds on the open-source PyTorch,
MNE-Python and scientific-Python ecosystems.

## Contact

- **Priyadharshini D** — priyadharshini.2024b@vitstudent.ac.in (corresponding author)
- **Shridevi S** — shridevi.s@vit.ac.in

School of Computer Science and Engineering, Vellore Institute of Technology,
Chennai, India.

For code questions, open a
[GitHub issue](https://github.com/priyadharshini-D29/INS-HDGS-CMT/issues); for
research questions, use
[Discussions](https://github.com/priyadharshini-D29/INS-HDGS-CMT/discussions).
