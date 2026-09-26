# INS-HDGS-CMT — Component (leave-one-out) Ablation  [calibrated]

**Baseline (full):** `abl_full` · 37 folds. Each variant disables ONE component; everything else identical (focal γ=3.0 · effective_num · n_ens=5 · 3-ch ET · same seed → paired).

Δ = variant − full. Negative Δ on bal-acc/MCC ⇒ the component **helps** (removing it hurts). Ranked by Δ balanced accuracy (most important first).

| Component removed | full bal-acc | variant bal-acc | Δ bal-acc | Wilcoxon p | Δ MCC | n |
|---|---|---|---|---|---|---|
| eeg_only | 0.7192 | 0.5628 | -0.1564 | 0.0002 | -0.3300 | 37 |
| no_graph | 0.7192 | 0.6963 | -0.0229 | 0.2838 | -0.0490 | 37 |
| no_contrastive | 0.7192 | 0.7290 | +0.0098 | 0.8442 | +0.0334 | 37 |
| no_mmd | 0.7192 | 0.7314 | +0.0122 | 0.6006 | +0.0241 | 37 |
| ns_rule_only | 0.7192 | 0.7335 | +0.0143 | 0.4792 | +0.0510 | 37 |
| no_et | 0.7192 | 0.7349 | +0.0157 | 0.2938 | +0.0241 | 37 |
| no_snn | 0.7192 | 0.7522 | +0.0330 | 0.4770 | +0.0662 | 37 |
| no_roi | 0.7192 | 0.7537 | +0.0345 | 0.2118 | +0.0667 | 37 |
| no_fusion_transformer | 0.7192 | 0.7559 | +0.0366 | 0.0476 | +0.0780 | 37 |
| no_neuro_symbolic | 0.7192 | 0.7726 | +0.0534 | 0.0393 | +0.1117 | 37 |
