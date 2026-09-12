# `configs/` — Configuration templates

Modular YAML configuration for every part of the pipeline. `default.yaml` is
the master entry point and composes the sub-configs. All values mirror the
canonical Python defaults in [`src/model/config/settings.py`](../src/model/config/settings.py)
— the headline configuration reported in the paper — so the YAML and the code
never drift.

```
configs/
├── default.yaml              Master config: composes everything below
├── hyperparameters.yaml      Flat, single-view summary of all tunables
├── model/
│   ├── graph.yaml            Dynamic connectivity + GAT encoder
│   ├── snn.yaml              Spiking (LIF) encoder
│   ├── transformer.yaml      Temporal Transformer + ET/ROI encoders
│   ├── fusion.yaml           Cross-modal fusion transformer
│   └── neuro_symbolic.yaml   Differentiable rule layer + classifier heads
├── training/
│   ├── train.yaml            Loop, loss weights, ensemble, domain adaptation
│   ├── optimizer.yaml        Optimizer (Adam)
│   └── scheduler.yaml        LR scheduler
└── evaluation/
    ├── losocv.yaml           Leave-One-Subject-Out CV (primary protocol)
    ├── cross_validation.yaml Stratified k-fold (comparison)
    ├── random_split.yaml     Random subject-mixed split (upper-bound baseline)
    └── test.yaml             Inference / evaluation-only from checkpoints
```

**Usage.** Any field can be overridden on the command line — see
`python src/model/main.py --help`. The paper's headline run is:

```bash
python src/model/main.py \
  --focal-gamma 3.0 --alpha-strategy effective_num \
  --n-ensemble 5 --lambda-dann 0.1 --lambda-mmd 0.1 \
  --mmd-mode marginal --norm-mode zscore
```
