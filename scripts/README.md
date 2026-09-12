# `scripts/` — Figure & analysis generators

Standalone scripts that turn raw results into figures, tables and statistics.
They consume files in `../results/` and write to `../paper/figures/` and
`../tables/` (both created on demand — nothing needs to exist beforehand).

- `figures/` — one script per figure (architecture, preprocessing, fusion,
  explainability, results) + combiners.
- `analysis/` — statistics and interpretability behind specific numbers:
  integrated gradients, neuro-symbolic rule extraction, SNN energy estimate,
  Cohen's κ verification, significance tests, threshold optimisation,
  classical baselines, case studies.

Driven by `../reproducibility/generate_figures.sh` and `generate_tables.sh`.
