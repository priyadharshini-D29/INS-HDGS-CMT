# INS-HDGS-CMT — Component (leave-one-out) Ablation  [raw]

**Baseline (full):** `abl_full` · 37 folds. Each variant disables ONE component; everything else identical (focal γ=3.0 · effective_num · n_ens=5 · 3-ch ET · same seed → paired).

Δ = variant − full. Negative Δ on bal-acc/MCC ⇒ the component **helps** (removing it hurts). Ranked by Δ balanced accuracy (most important first).

| Component removed | full bal-acc | variant bal-acc | Δ bal-acc | Wilcoxon p | Δ MCC | n |
|---|---|---|---|---|---|---|
| eeg_only | 0.7475 | 0.5524 | -0.1951 | 0.0000 | -0.3914 | 37 |
| no_graph | 0.7475 | 0.6556 | -0.0919 | 0.0051 | -0.1694 | 37 |
| no_et | 0.7475 | 0.6969 | -0.0506 | 0.0351 | -0.0877 | 37 |
| no_snn | 0.7475 | 0.7018 | -0.0457 | 0.1109 | -0.0810 | 37 |
| no_roi | 0.7475 | 0.7136 | -0.0339 | 0.0683 | -0.0594 | 37 |
| no_fusion_transformer | 0.7475 | 0.7143 | -0.0332 | 0.1029 | -0.0741 | 37 |
| ns_rule_only | 0.7475 | 0.7147 | -0.0328 | 0.0467 | -0.0477 | 37 |
| no_mmd | 0.7475 | 0.7164 | -0.0311 | 0.1972 | -0.0469 | 37 |
| full_graph_static | 0.7475 | 0.7407 | -0.0068 | 0.7362 | +0.0115 | 37 |
| no_neuro_symbolic | 0.7475 | 0.7417 | -0.0058 | 0.5861 | -0.0024 | 37 |
| full_graph_random | 0.7475 | 0.7490 | +0.0015 | 0.9587 | +0.0240 | 37 |
| no_contrastive | 0.7475 | 0.7529 | +0.0054 | 0.8867 | +0.0109 | 37 |
