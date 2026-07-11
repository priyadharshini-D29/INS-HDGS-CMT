# `ablation/` — Component ablation study

Everything needed to reproduce the ablation study (paper Table 6), which
isolates the marginal contribution of each architectural component by removing
one at a time and re-running the full LOSOCV protocol.

The driver script lives in `reproducibility/run_ablation.sh`; the per-variant
logic and result aggregation live in `src/model/` (`main.py --ablation`,
`aggregate_ablations.py`) and `scripts/analysis/` (`run_component_ablation.py`,
`compare_component_ablation.py`). This folder holds the ablation **manifest**
and any variant-specific overrides.

## Variants

| Variant | Component removed / changed |
|---|---|
| `full` | none — full multimodal reference model |
| `eeg_only` | eye-tracking branch removed |
| `et_only` | EEG branch removed |
| `no_neuro_symbolic` | differentiable neuro-symbolic rule layer |
| `no_graph` | dynamic functional graph + GAT encoder |
| `no_fusion_transformer` | cross-modal fusion transformer |
| `no_snn` | spiking (LIF) encoder |
| `no_roi` | ROI attention head |
| `no_cross_attention` | cross-attention (self-attention only) |
| `no_dynamic_connectivity` | dynamic graphs → static/fixed adjacency |
| `no_contrastive` | contrastive objective |
| `no_mmd` | MMD subject-invariance term |

## Reproduce

```bash
# All variants
bash reproducibility/run_ablation.sh

# A single variant
bash reproducibility/run_ablation.sh no_snn
```

Outputs are written to `results/ablation/<variant>/`, logs to `logs/`, and the
combined comparison table to `tables/table4_ablation.csv`. Raw logs from the
original ablation runs are preserved in `logs/ablation_abl_*.log` for
provenance.
