# `reproducibility/` — One-command experiment reproduction

Shell entry points that reproduce every experiment in the paper. All scripts
are idempotent, resolve paths relative to the repository root, and honour the
`SEED` environment variable (default `42`).

| Script | What it does |
|---|---|
| `run_all.sh` | Prints the environment (Python/CUDA/GPU/seed), then runs the full pipeline. |
| `reproduce_paper.sh` | End-to-end: train → evaluate → ablation → tables → figures. |
| `train.sh` | Trains the headline LOSOCV model (paper hyperparameters). |
| `evaluate.sh` | Scores trained checkpoints; writes all metrics + optimal thresholds. |
| `run_ablation.sh` | Runs the full component-ablation study (Table 6). Accepts a single variant name. |
| `generate_tables.sh` | Rebuilds every manuscript table from `results/` CSVs. |
| `generate_figures.sh` | Rebuilds every manuscript figure via `scripts/figures/`. |

### Typical usage

```bash
# Everything, from scratch (needs the preprocessed dataset; see ../datasets/)
bash reproducibility/run_all.sh

# Or step by step
bash reproducibility/train.sh
bash reproducibility/evaluate.sh
bash reproducibility/run_ablation.sh
bash reproducibility/generate_tables.sh
bash reproducibility/generate_figures.sh
```

Expected runtimes, hardware, and expected metric values are documented in
[`../docs/REPRODUCIBILITY_CHECKLIST.md`](../docs/REPRODUCIBILITY_CHECKLIST.md).

> **Note on exact numbers.** The committed `results/` CSVs come from the
> original LOSOCV run. Re-running the current code reproduces the paper's
> figures within ordinary run-to-run variance (calibrated accuracy within
> ~0.25 pts, MCC within ~0.012, AUC within ~0.015), not bit-for-bit. See the
> reproducibility note in the top-level `README.md`.
