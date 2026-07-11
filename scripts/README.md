# `scripts/` — Figure & analysis generators

Standalone scripts that turn raw results into the manuscript's figures,
tables and statistics. They consume files in `../results/` and write to
`../figures/` and `../tables/`.

- `figures/` — one script per manuscript figure (overview, architecture,
  preprocessing, fusion, explainability, results) + combiners.
- `analysis/` — statistics and interpretability behind specific numbers:
  integrated gradients, neuro-symbolic rule extraction, SNN energy estimate,
  Cohen's κ verification, significance tests, threshold optimisation,
  classical baselines, case studies.
- `manuscript_report.py` — regenerates the consolidated tables report.

Driven by `../reproducibility/generate_figures.sh` and `generate_tables.sh`.
