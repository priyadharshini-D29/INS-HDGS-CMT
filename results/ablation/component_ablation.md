# INS-HDGS-CMT — Component (leave-one-out) Ablation  [calibrated]

**Baseline (full):** `repro_focal_g3p0_effective_num_37` · 37 folds. Each variant disables ONE component; everything else identical (focal γ=3.0 · effective_num · n_ens=5 · 3-ch ET · same seed → paired).

Δ = variant − full. Negative Δ on bal-acc/MCC ⇒ the component **helps** (removing it hurts). Ranked by Δ balanced accuracy (most important first).

| Component removed | full bal-acc | variant bal-acc | Δ bal-acc | Wilcoxon p | Δ MCC | n |
|---|---|---|---|---|---|---|
| no_graph | 0.7531 | 0.6963 | -0.0568 | 0.0073 | -0.1102 | 37 |
| no_contrastive | 0.7531 | 0.7290 | -0.0241 | 0.1587 | -0.0277 | 37 |
| no_mmd | 0.7531 | 0.7314 | -0.0217 | 0.3004 | -0.0371 | 37 |
| no_et | 0.7531 | 0.7349 | -0.0182 | 0.2547 | -0.0371 | 37 |
| no_snn | 0.7531 | 0.7522 | -0.0009 | 0.5348 | +0.0050 | 37 |
| no_roi | 0.7531 | 0.7537 | +0.0006 | 0.6871 | +0.0055 | 37 |
| no_fusion_transformer | 0.7531 | 0.7559 | +0.0027 | 0.8017 | +0.0168 | 37 |
| no_neuro_symbolic | 0.7531 | 0.7726 | +0.0195 | 0.7938 | +0.0505 | 37 |
