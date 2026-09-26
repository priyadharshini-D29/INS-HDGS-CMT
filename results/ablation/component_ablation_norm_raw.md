# INS-HDGS-CMT — Component (leave-one-out) Ablation  [raw]

**Baseline (full):** `abl_full` · 37 folds. Each variant disables ONE component; everything else identical (focal γ=3.0 · effective_num · n_ens=5 · 3-ch ET · same seed → paired).

Δ = variant − full. Negative Δ on bal-acc/MCC ⇒ the component **helps** (removing it hurts). Ranked by Δ balanced accuracy (most important first).

| Component removed | full bal-acc | variant bal-acc | Δ bal-acc | Wilcoxon p | Δ MCC | n |
|---|---|---|---|---|---|---|
| full_norm_instance | 0.7475 | 0.7634 | +0.0159 | 0.4817 | +0.0291 | 37 |
