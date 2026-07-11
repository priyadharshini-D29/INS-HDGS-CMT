# INS-HDGS-CMT — Project Architecture & Technical Reference

**Project:** NeuMa Phase 8 — Cross-subject *cognitive engagement* decoding for neuromarketing
**Model:** INS-HDGS-CMT — *Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking Cognitive Multimodal Transformer*
**Primary task:** Binary classification — `HIGH_ENGAGEMENT (1)` vs `LOW_ENGAGEMENT (0)`
**Evaluation:** Leave-One-Subject-Out Cross-Validation (LOSOCV), 37 evaluable folds
**Last updated:** 2026-06-14

> This document describes the full pipeline, model internals, losses, training, and evaluation in detail. File/line citations point to the implementation. Where a number is the *honest publishable* figure vs a post-hoc-optimized figure, both are stated explicitly.

---

## 1. Task Definition & Scientific Framing

The model decodes a viewer's **cognitive engagement state** from synchronized EEG + eye-tracking while they browse an advertising brochure.

- Earlier the project classified brochure *page identity* (`ImagePage_1…6`). This was abandoned: EEG/ET encode cognitive *state*, not image identity, so page labels had no neuroscientific basis and failed cross-subject.
- The current target — `HIGH_ENGAGEMENT` vs `LOW_ENGAGEMENT` — is derived from eye-tracking metrics (fixation ratio, dwell time, ROI density, pupil dilation, gaze entropy, revisit count) via a **median threshold**.
- Labeling mode is controlled by `NEUMA_LABEL_MODE` ([config/settings.py:77](config/settings.py#L77)); the production setting is **`global`** (population-level threshold). Per-subject-median labeling was tested and **rejected** — it destroys cross-subject label coherence (pooled MCC collapsed to ~0.006).

> ⚠️ **Note on "Buy/Not-Buy":** the engagement task is *not* the same as the NeuMa dataset's native purchase ("Bought") label. The dataset provides Buy/Not-Buy per product; engagement labels are derived by this project and are not part of the original release.

---

## 2. Dataset & Input Representation

### 2.1 Source dataset (NeuMa)
- Multimodal neuromarketing dataset; 42 released subjects (45 recorded, 3 excluded for artifacts).
- Modalities used here: **EEG** (Wearable Sensing DSI-24, 300 Hz, 21 dry electrodes, 10-20 system, A1/A2 mastoid reference) and **Eye Tracking** (Tobii Pro Fusion, 120 Hz).
- 6 brochure pages × 24 supermarket products/page.

### 2.2 Subject pool ([config/settings.py:26-35](config/settings.py#L26-L35))
- `SUBJECT_IDS` = `S01…S44` **minus S04, S11** (those two lack per-subject engagement labels and would fall back to a shared pooled directory → cross-subject duplicate epochs = leakage).
- Pool math: **44 total → 42 valid (−S04, −S11) → 37 valid test folds.**
- 5 single-class subjects under the global threshold — **S16 (all-LOW), S31, S33, S41, S44** — are **retained in training** but **skipped as test folds** (degenerate AUC/MCC).

### 2.3 Per-epoch tensors (model inputs)
Built in [data/dataset.py](data/dataset.py); each sample (`__getitem__`) returns:

| Key | Shape | Meaning |
|-----|-------|---------|
| `eeg_windows` | `(W=10, C, 5)` | band-power node features per temporal window |
| `adj_matrices` | `(W, C, C)` | binary per-window adjacency (functional connectivity) |
| `weighted_adjs` | `(W, C, C)` | weighted adjacency (loss/visualization) |
| `et_seq` | `(T_et=600, C_et=3)` | raw ET sequence `[gaze_x, gaze_y, pupil]` |
| `roi_vector` | `(N_rois=10,)` | normalized ROI dwell-time |
| `label` | scalar int | 0 = LOW, 1 = HIGH |

Key signal parameters ([config/settings.py:37-71](config/settings.py#L37-L71)):
- EEG: `EEG_SR=300`, epoch length `5.0 s` → `EEG_SAMPLES=1500`; `N_WINDOWS=10` → `WINDOW_SAMPLES=150` (~500 ms/window).
- **Epoch overlap = 0.75** (75% overlap between successive epochs → ~4× more epochs; data augmentation, applied only when `augment=True`).
- 5 EEG bands → 5 node features/electrode: delta (1–4), theta (4–8), alpha (8–13), beta (13–30), gamma (30–45). Theta/alpha are the engagement-relevant bands.
- ET: `ET_SR=120`, 5 s → `ET_SAMPLES=600`, `ET_INPUT_DIM=3`.
- ROI: `N_ROIS=10`, grid `5×2` over a `3000×1688` image.
- Graph construction: `CONN_METHOD="pearson"`, `CONN_THRESHOLD=0.30`, `MIN_EDGES=3`.

### 2.4 Normalization (two stages — both ON in production)
1. **Subject-aware z-score** per channel, applied in the dataset before graph construction (`norm_mode="zscore"`, `USE_SUBJECT_NORM=True`) — removes inter-subject amplitude shift. ([data/dataset.py:502-520](data/dataset.py#L502-L520))
2. **Intra-epoch LayerNorm** over the `(C, 5)` dims per window inside the model forward pass — removes window-to-window drift. ([models/ins_hdgs_cmt.py:710-712](models/ins_hdgs_cmt.py#L710-L712))

> **ET is 3-channel, UN-normalized in production.** The 9-channel (both eyes + vergence + speed) and per-epoch z-scored ET variants were both tested and **hurt** performance. Reproduce with `ET_USE_BOTH_EYES=0 ET_USE_VERGENCE=0 ET_USE_SPEED=0 ET_NORMALIZE=0` (tag `et3_b0_v0_s0_n0`).

---

## 3. Model Architecture (INS-HDGS-CMT)

Universal embedding dimension **`EMBED_DIM = 128`** throughout. Defined in [models/ins_hdgs_cmt.py](models/ins_hdgs_cmt.py). Six functional components + a training-only DANN head; all components are ablatable via `AblationConfig`. Production = `AblationConfig.full()` (every component ON).

### Forward dataflow
```
eeg_windows (B,W,C,5) ─► [LayerNorm] ─► DynamicGAT ───────► graph_emb (B,128)  + window_seq (B,W,128)
                                         │
                                         ├─ band-weighted proxy ─► SpikingEEGEncoder ─► snn_emb (B,128)
                                         │
graph_emb + snn_emb ─► eeg_merge (256→128) ──────────────► eeg_emb (B,128)
et_seq (B,600,3) ─► ETAttentionEncoder ─► et_emb (B,128), et_roi_attn (B,10), et_window_seq (B,W,128)
roi_vector / et_roi_attn ─► ROIGraphModulation (modulates adjacency BEFORE GAT)
eeg_emb + roi_vector ─► ROIAttention (gating) ─► eeg_emb'
[eeg_emb', graph_emb, et_emb] ─► NeuroFusionTransformer ─► fused (B,128)
fused ─► NeuroSymbolicRuleLayer ─► logits (B,2)
```

### 3.1 SpikingEEGEncoder (LIF SNN) — [models/spiking_encoder.py](models/spiking_encoder.py)
- Multi-layer **Leaky Integrate-and-Fire** network; `time_steps=10`, `n_layers=2`, `hidden_dim=128`, `decay=0.90`, `threshold=1.0`, surrogate `beta=5.0`, dropout 0.30.
- LIF dynamics per step: `V[t] = decay·V[t−1] + W·x[t]` → spike `S[t] = H(V[t]−threshold)` → hard reset `V[t] = V[t]·(1−S[t])`.
- Non-differentiable Heaviside replaced by **fast-sigmoid surrogate gradient** (Zenke & Ganguli 2021): `dH/dV ≈ grad / (β·|V−thr| + 1)²`.
- Input `(B,C,S)` → `adaptive_avg_pool1d` to `time_steps` → per-step Linear→GELU→LN → LIF stack → temporal **mean + max** pool concat `(B,2D)` → Linear→GELU→LN → `(B,128)`.
- In production the SNN consumes a **band-weighted proxy** of the GAT input (no separate raw EEG passed): `softmax(snn_band_weights) · eeg_windows` summed over bands. `snn_band_weights` is a learnable `nn.Parameter` initialized `[0.05, 0.40, 0.35, 0.10, 0.10]` — strongly favoring **theta (idx 1)** and **alpha (idx 2)**. ([ins_hdgs_cmt.py:474-477](models/ins_hdgs_cmt.py#L474-L477), [:727-731](models/ins_hdgs_cmt.py#L727-L731))

### 3.2 DynamicGAT (dynamic temporal graph) — [models/dynamic_gat.py](models/dynamic_gat.py) + [models/gat_encoder.py](models/gat_encoder.py)
- **Weight-tied GATEncoder** shared across all W=10 windows (Temporal Graph Network paradigm).
- **GATEncoder** (Veličković et al. 2018 attention): Layer 1 = **4 heads × 32 dim, concat → 128**; Layer 2 = **1 head → 128 with residual** (`res_proj`), LayerNorm + ELU; dropout 0.20, LeakyReLU slope 0.20. Returns per-layer attention (`l1_attn`, `l2_attn`) for explainability.
- Non-edges masked to `−∞` before softmax; isolated-node NaNs zeroed.
- Node→window pooling: **learned attention pooling** over the C electrodes (`node_pool` Linear→softmax), not plain mean.
- Temporal aggregation over windows: **TransformerEncoder** (`t_nhead=4`, `t_layers=3`, `t_ff_dim=512`, pre-norm) with **learnable positional embeddings** for the 10 window positions; mean-pool over windows → `graph_emb (B,128)`. (BiLSTM is available as an ablation.)
- Returns `graph_emb`, `gat_attn`, and the per-window sequence `temp_out (B,W,128)` (used as cross-attention keys/values in fusion).

### 3.3 ETAttentionEncoder — [models/et_encoder.py](models/et_encoder.py)
- Per-timestep MLP (`3→64→64`, GELU+LN) → prepend **learnable CLS token** → **sinusoidal positional encoding** → **TransformerEncoder** (`n_heads=4`, `n_layers=2`, ff = 4×hidden, pre-norm).
- CLS token → `emb_proj` → **`et_emb (B,128)`**.
- `et_emb` → `roi_proj` (Linear→Softmax) → **`et_roi_attn (B,10)`** (ROI attention distribution, feeds ROI modulation).
- Per-window ET sequence: the 600 ET tokens are chunked into W=10 windows, mean-pooled, projected → **`et_window_seq (B,W,128)`** (fusion keys/values).
- Blink-gap NaNs replaced with 0. (A plain BiLSTM `ETEncoder` exists as the non-attention ablation.)

### 3.4 ROI modules
- **ROIGraphModulation** ([models/roi_modulation.py](models/roi_modulation.py)) — modulates the adjacency **before** GAT: `roi_elec = σ(W·roi_vec) ∈ R^C`; mask = outer product `roi_elec ⊗ roi_elec`; `A_roi = A · (1 + α·mask)` with learnable `α` (init 0.5); then **symmetric normalization** `Â = D^{-1/2} A D^{-1/2}`. Uses `et_roi_attn` when available, else `roi_vector`.
- **ROIAttention** ([models/roi_attention.py](models/roi_attention.py)) — embedding-level gating: `h̃ = LayerNorm(h ⊙ σ(MLP(roi_vec)))`; also emits `roi_logits (B,10)` for the auxiliary ROI loss.

### 3.5 NeuroFusionTransformer (4-stage cross-modal) — [models/fusion_transformer.py](models/fusion_transformer.py)
- **Stage 1 — Modality tokenization:** add learnable type embeddings to `[eeg, graph, et]` → tokens `(B,3,128)`.
- **Stage 2 — Self-attention** over the 3 modal tokens (TransformerEncoder, `num_heads=4`, `num_layers=t_layers=3`, ff=512, pre-norm).
- **Stage 3 — Directed cross-modal attention:** EEG query attends over the **per-window** graph sequence and ET sequence (real W tokens, not collapsed vectors): `EEG←Graph` and `EEG←ET` via two `MultiheadAttention` blocks.
- **Stage 4 — Gated residual fusion:** `concat=[eeg, cross_g, cross_e]` → `gate=σ(W_g·concat)`, `fused=LayerNorm(σ-gate ⊙ W_f·concat)` → `fused (B,128)`. The gate prevents fusion collapse when a modality is uninformative.
- (A simpler `CrossModalFusion` is the non-transformer ablation.)
- `eeg_merge` ([ins_hdgs_cmt.py:557-564](models/ins_hdgs_cmt.py#L557-L564)) fuses SNN+graph: `Linear(256→128)+GELU+LayerNorm`.

### 3.6 NeuroSymbolicRuleLayer (interpretable head) — [models/neuro_symbolic.py](models/neuro_symbolic.py)
- `n_rules=8`, `hidden_dim=64`, `temperature=1.0`.
- `keys = MLP(fused)`; rule activation `acts = softmax((rule_queries · keys)/√D / T)` `(B,8)`.
- Each rule has a small linear head → `rule_logits (B,8,2)`; aggregated `agg = Σ acts·rule_logits`.
- **Bypass mix:** `logits = σ(bypass_alpha)·bypass(fused) + (1−σ(bypass_alpha))·agg`, with learnable `bypass_alpha` (init 0.30) — stabilizes the rule ensemble.
- Produces human-readable IF-THEN rules (`explain_rules`, `explain_sample`).
- When disabled, replaced by a standard MLP classifier `Linear(128→64)+GELU+Dropout+Linear(64→2)`.

### 3.7 DANN subject classifier (training-only) — [models/ins_hdgs_cmt.py:599-609](models/ins_hdgs_cmt.py#L599-L609)
- Gradient Reversal Layer + subject classifier on `fused`; pushes subject-invariant representations. **Never called at inference** (zero pipeline change). GRL `alpha` scheduled `2/(1+e^{−10p})−1` over training.
- ⚠️ DANN was **refuted** as a benefit in controlled dose-response screens — kept in code but the production lever is `λ_dann=0.10` with no demonstrated gain; report as an ablation, not a contribution.

---

## 4. Loss Functions

Total multi-task loss ([losses.py](training/losses.py) `MultiTaskLoss` → [ins_hdgs_cmt.py:878](models/ins_hdgs_cmt.py#L878) `compute_loss`):

```
L_total = λ_cls·L_focal + λ_contrast·L_contrast + λ_roi·L_roi
        + λ_conn·L_conn + λ_mmd·L_mmd + λ_rules·L_rules   (+ λ_dann·L_dann during training)
```

### 4.1 Classification — Focal loss (the winning lever)
- `ce = cross_entropy(logits, labels, weight=class_weights, label_smoothing=0.05, reduction='none')`
- `p_t = softmax(logits)[label]`, `focal_w = α·(1−p_t)^γ`, `L_focal = mean(focal_w · ce)`.
- **Production: `γ = 3.0` (`FOCAL_GAMMA` overridden), `α = 1.0`** (`FOCAL_ALPHA`; no global downscale — the old 0.25 compressed cls loss and let contrastive dominate).
- `label_smoothing=0.05` prevents p→0/1 collapse and lowers ECE.

### 4.2 Class weighting — `effective_num` (Cui et al. CVPR 2019)
[losses.py:26-70](training/losses.py#L26-L70) `compute_alpha_weights`. Production strategy = **`effective_num`**: `β=(N−1)/N`, `w = (1−β)/(1−β^{n_c})`, mean-normalized. (`balanced` ties it; `sqrt_inv_freq` consistently worst.) Computed per-fold from the training labels.

### 4.3 Auxiliary losses
- **L_contrast** — EEG↔ET alignment. Default **InfoNCE with hard-negative mining** (`temperature=0.07`, `hard_neg_weight=0.50`); NT-Xent is the fallback. ([models/contrastive.py](models/contrastive.py))
- **L_roi** — KL divergence between predicted ROI distribution and `et_roi_attn` (or `roi_vector`).
- **L_conn** — graph sparsity: MSE of off-diagonal weighted adjacency toward 0.
- **L_mmd** — multi-scale RBF MMD (`gammas=(0.1,0.5,1,2,5)`) for cross-subject domain adaptation. ([models/mmd.py](models/mmd.py)) `mmd_mode="marginal"`.
- **L_rules** — rule diversity (pairwise cosine) + activation sparsity (entropy).

### 4.4 Loss weights ([config/settings.py:167-172](config/settings.py#L167-L172))
`LAMBDA_CLS=5.0`, `LAMBDA_CONTRAST=0.05`, `LAMBDA_ROI=0.05`, `LAMBDA_CONNECTIVITY=0.02`, `LAMBDA_MMD=0.10`, `λ_rules≈0.02`, `λ_dann=0.10`.
Classification deliberately dominates early to prevent majority-class collapse.

### 4.5 Auxiliary warm-up ramp
All auxiliary λ's ramp **0→1 over the first 40% of epochs** (`aux_ramp`); classification is always full weight. ([trainer.py:590-597](training/trainer.py#L590-L597))

---

## 5. Training Protocol ([training/trainer.py](training/trainer.py))

| Aspect | Setting |
|--------|---------|
| Optimizer | **AdamW**, `lr=5e-4`, `weight_decay=0` |
| LR schedule | **CosineAnnealingWarmRestarts** (`T_0=50`, `T_mult=1`, `eta_min=1e-6`), steps per epoch |
| Epochs | **250** (`EPOCHS`) |
| Early stopping | patience **100** on **val balanced-accuracy** (not F1 — F1 rewards class collapse); cannot fire before epoch 20 |
| Checkpoint criterion | best `val_balanced_acc`; full state saved (model+opt+sched+scaler+counters), atomic, resumable |
| Batch size | **32** (capped to train-set size in fold-parallel → ~9 batches/epoch on ~300-sample folds) |
| Grad accumulation | 1 |
| Mixed precision | **AMP on** (`torch.amp`, `GradScaler`) |
| Grad clipping | `clip_grad_norm_(…, 1.0)` ([trainer.py:433](training/trainer.py#L433)) |
| Global dropout | 0.25–0.30 |
| Random seed | 42 (`RANDOM_SEED`); ensemble member i uses `42 + i·997` |
| Multi-GPU | DataParallel; **fold-parallel** when `NUM_GPUS>1` (one fold per GPU concurrently) |

### 5.1 Per-fold ensemble
- **`N_ENSEMBLE = 15`** in production (`config/settings.py`); the focal *sweep* used 5 for speed and the winner is meant to be re-run at 15 for the paper number.
- Each member trained with a distinct seed; probabilities averaged before thresholding.

### 5.2 Post-hoc temperature calibration ([trainer.py:443-495](training/trainer.py#L443-L495))
- Platt-style scalar **T** fit by LBFGS (minimize NLL) on the **validation subject's** logits.
- Guards: skip (T=1.0) if `<20` val samples; clamp `T ∈ [0.30, 3.0]` (prevents the T≈0.05 over-sharpening that blew up ECE).

---

## 6. Evaluation — LOSOCV ([evaluation/losocv.py](evaluation/losocv.py))

For each of the 37 evaluable held-out test subjects ([losocv.py:371-400](evaluation/losocv.py#L371-L400)):
1. **Split:** test = held-out subject; remaining 41 subjects shuffled (seed 42) → **1 validation subject** + 40 training subjects. *Calibration and threshold are fit on the validation subject, never the test subject → leakage-free.*
2. Skip the fold if the test subject's minority class < `IMBALANCE_SKIP_RATIO` (production 0.0 → all evaluated; single-class subjects already excluded).
3. Compute per-fold `effective_num` class weights from train labels.
4. Train **N_ENSEMBLE** models (distinct seeds); temperature-calibrate each on val.
5. **Aggregate:** average calibrated `P(HIGH)` across members.
6. **Threshold:** Youden-J / balanced-accuracy-optimal threshold fit on the **validation** subject, winsorized to `[0.30, 0.70]` ([losocv.py:125](evaluation/losocv.py#L125)) to curb tiny-val overfit.
7. **Reported calibrated path:** average raw logits across members → fit single `T_post` on val → calibrated test probs → val-derived threshold → metrics.
8. Each fold row stores `y_true`, `y_prob`, `y_pred`, thresholds, `T_post`, durations + raw and `_cal` metrics.

> ⚠️ **Persistence:** in fold-parallel mode the results CSV (`losocv_<label>.csv`) is written **once, after all folds finish** ([losocv.py:340-342](evaluation/losocv.py#L340-L342)). Nothing is saved mid-run. Always launch detached (`setsid nohup … > run.log 2>&1 &`) so output is captured and the process survives session close.

---

## 7. Metrics ([training/metrics.py](training/metrics.py))

`compute_metrics` returns (per fold and pooled, each in raw + `_cal`):
**accuracy, balanced_acc, F1, MCC (Matthews), Cohen's κ, ROC-AUC, PR-AUC, precision, recall, ECE**.
- ECE = expected calibration error (top-label, binned).
- Threshold-independent metrics (AUC, balanced-acc, MCC) are the headline framing — immune to "you tuned the threshold."

---

## 8. Best Result & Production Config

**Configuration `focal_abl_g3p0_effective_num`** — focal γ=3.0 × `effective_num` weighting, 3-ch un-normalized ET, full model.

| Metric | Value |
|--------|-------|
| Accuracy | **0.7976** (post-hoc threshold) / **~0.79 leakage-free** |
| Balanced accuracy | 0.7594 |
| MCC | **0.5304** |
| ROC-AUC | 0.870 per-fold / 0.861 pooled |
| PR-AUC | ~0.898 |
| Raw accuracy (stored threshold) | 0.7569 |
| ECE | 0.166 → **0.070** after cross-fold isotonic recalibration (no retrain) |
| Folds | 37 of 42 subjects (5 single-class excluded) |

**Reporting guidance / guardrails:**
- The publishable headline is **~0.79 leakage-free** (beats the ~0.77 literature benchmark). Lead with **AUC + balanced-acc + MCC**, report accuracy alongside, state validation-derived calibration explicitly.
- **Do NOT** report the test-tuned `0.804` global threshold, nor cherry-pick seeds/folds.
- ECE is reported from the **pooled** cross-fold isotonic recalibration (`analysis/recalibrate_crossfold.py`); AUC/PR-AUC from raw probs (discrimination is calibration-independent).

**Ceiling & bottleneck:** ~0.797 ceiling; main limiter is **high per-fold variance** (acc std ~0.19) from ~5–6 *inverted-AUC* subjects (subject distribution shift, e.g. S21), not the loss function. Open lever: reliability-aware fusion / test-time adaptation for those subjects.

**Approaches tested and REJECTED** (report as ablations, do not retry): DANN invariance, contrastive SSL pretraining, ET feature expansion (9-ch), ET normalization, per-subject-median labeling, reliability/recalibration static fusion, percentile-tail labeling.

---

## 9. Ablation Configurations ([models/ins_hdgs_cmt.py:125-368](models/ins_hdgs_cmt.py#L125-L368))

`AblationConfig` toggles every component (canonical flags + semantic aliases + legacy RD-GANet aliases, kept in sync by `__post_init__`). Factory presets:
`full()` (production), `eeg_only()`, `no_snn()`, `no_graph()`, `no_neuro_symbolic()`, `no_et()`, `no_fusion_transformer()`, `no_roi()`, `no_contrastive()`, `no_mmd()`, `baseline_linear()`, plus legacy `eeg_gat()`, `et_only()`, `eeg_et()`.
When graph+SNN are both off, a `_LinearEEGEncoder` (Flatten→Linear→ReLU→LN) is the fallback.

---

## 10. Run Launch Checklist (silent footguns)

All of these fail **silently** — verify each before any blessed-config run:
1. **`unset CUDA_VISIBLE_DEVICES`** — a stray `CVD=7` from the shell profile pins all 37 folds to one GPU (sequential, ~6–8× slower).
2. **ET flags:** `ET_USE_BOTH_EYES=0 ET_USE_VERGENCE=0 ET_USE_SPEED=0 ET_NORMALIZE=0` (tag `et3_b0_v0_s0_n0`) — defaults are 9-ch + normalized, which is *not* the blessed model.
3. **Detach:** launch with `setsid nohup python main.py … > run.log 2>&1 &` (+ `disown`); plain `&` dies on session close (SIGHUP).
4. **Disk:** `output/checkpoints` is ~177 GB and the disk has ~30 GB free — never `cp -r` checkpoints; back up only small metrics/analysis dirs.

Example (production reproduce):
```bash
unset CUDA_VISIBLE_DEVICES
ET_USE_BOTH_EYES=0 ET_USE_VERGENCE=0 ET_USE_SPEED=0 ET_NORMALIZE=0 \
setsid nohup python main.py --fold-parallel \
  --label focal_g3p0_effnum_ens15 \
  --epochs 250 --alpha-strategy effective_num \
  --focal-gamma 3.0 --n-ensemble 15 \
  > focal_g3p0_ens15.log 2>&1 & disown
pgrep -af main.py   # confirm it's alive
```

---

## 11. Configuration Quick Reference ([config/settings.py](config/settings.py))

| Param | Value | Param | Value |
|-------|-------|-------|-------|
| `N_CLASSES` | 2 | `EMBED_DIM` | 128 |
| `EEG_SR` | 300 | `ET_SR` | 120 |
| `EEG_SAMPLES` | 1500 | `ET_SAMPLES` | 600 |
| `N_WINDOWS` | 10 | `ET_INPUT_DIM` | 3 |
| `EPOCH_OVERLAP` | 0.75 | `N_ROIS` | 10 |
| `GAT_L1_HEADS×DIM` | 4×32 | `GAT_L2` | 1×128 |
| `T_NHEAD / T_LAYERS / T_FF` | 4 / 3 / 512 | `FUSION_HEADS` | 4 |
| `SNN_TIME_STEPS` | 10 | `SNN_DECAY` | 0.90 |
| `SNN_THRESHOLD` | 1.0 | `SNN_HIDDEN_DIM` | 128 |
| `NS_N_RULES` | 8 | `NS_HIDDEN_DIM` | 64 |
| `BATCH_SIZE` | 32 | `EPOCHS` | 250 |
| `LR` | 5e-4 | `WEIGHT_DECAY` | 0 |
| `PATIENCE` | 100 | `DROPOUT` | 0.25 |
| `FOCAL_ALPHA` | 1.0 | `FOCAL_GAMMA` | 2.0 (prod override 3.0) |
| `TEMPERATURE` | 0.07 | `N_ENSEMBLE` | 15 |
| `LAMBDA_CLS` | 5.0 | `LAMBDA_CONTRAST` | 0.05 |
| `LAMBDA_ROI` | 0.05 | `LAMBDA_CONNECTIVITY` | 0.02 |
| `LAMBDA_MMD` | 0.10 | `LAMBDA_SECONDARY` | 0.15 |
| `RANDOM_SEED` | 42 | `LABEL_MODE` | global |
| `CONN_METHOD` | pearson | `CONN_THRESHOLD` | 0.30 |
| `USE_AMP` | True | `USE_SUBJECT_NORM` | True |

---

## 12. Key Source Files

| File | Role |
|------|------|
| [config/settings.py](config/settings.py) | All hyperparameters & paths |
| [models/ins_hdgs_cmt.py](models/ins_hdgs_cmt.py) | Top-level model, ablation config, forward, loss |
| [models/spiking_encoder.py](models/spiking_encoder.py) | LIF SNN encoder |
| [models/dynamic_gat.py](models/dynamic_gat.py) / [gat_encoder.py](models/gat_encoder.py) | Dynamic temporal graph attention |
| [models/et_encoder.py](models/et_encoder.py) | ET attention encoder |
| [models/fusion_transformer.py](models/fusion_transformer.py) | 4-stage cross-modal fusion |
| [models/neuro_symbolic.py](models/neuro_symbolic.py) | Interpretable rule head |
| [models/roi_modulation.py](models/roi_modulation.py) / [roi_attention.py](models/roi_attention.py) | ROI graph modulation + gating |
| [models/contrastive.py](models/contrastive.py) / [mmd.py](models/mmd.py) | InfoNCE / NT-Xent, MMD |
| [training/losses.py](training/losses.py) | MultiTaskLoss, focal, class weights |
| [training/trainer.py](training/trainer.py) | Train loop, AMP, calibration, early stop |
| [training/metrics.py](training/metrics.py) | Metric computation, ECE |
| [evaluation/losocv.py](evaluation/losocv.py) | LOSOCV protocol, ensemble, thresholds |
| [main.py](main.py) | Pipeline entry point / CLI |
| [evaluation/stats_table5.py](evaluation/stats_table5.py) | Friedman/Nemenyi/Wilcoxon/Holm/Cliff's δ/bootstrap CIs (Table 5) |
| [evaluation/make_tables.py](evaluation/make_tables.py) | Publication Tables 1–3 (EEG-headline framing) |
| [evaluation/calibration_experiment.py](evaluation/calibration_experiment.py) | Subject-adaptive (prevalence-matched) thresholding |
| [analysis/integrated_gradients.py](analysis/integrated_gradients.py) | IG feature attribution (Table `tab:features`) |
| [analysis/snn_energy.py](analysis/snn_energy.py) | LIF spike-sparsity & energy proxy |
| [baselines/baseline_models.py](baselines/baseline_models.py) | Named EEG/ET/fusion baselines (Pipelines 1–3) |
| [evaluation/leakage_audit.py](evaluation/leakage_audit.py) | Leakage audit incl. label-source circularity probe |

---

## 13. Session additions (2026-06-17) — efficiency, attribution, calibration, explanation-only

These extend §3–§7 with measurements and one new architectural mode.

### 13.1 Neuro-symbolic **explanation-only** mode (new)
`AblationConfig.neuro_symbolic_explain_only` / factory `AblationConfig.ns_explain_only()`.
When set, the rule layer still computes activations and human-readable traces, but the
**decision** is taken by the standard classifier (`logits = classifier(fused)`), so symbolic
refinement cannot change predictions. Implemented in
[models/ins_hdgs_cmt.py](models/ins_hdgs_cmt.py) (`self.ns_explain_only`, Step-5 branch) and
exposed as ablation variant `ns_explain_only`. Rationale: the symbolic mixing was
accuracy-neutral-to-negative (see §13.4); this keeps interpretability at no accuracy cost.

### 13.2 SNN spike-sparsity & energy (efficiency justification for the LIF encoder)
Measured on the trained encoder over 347 LOSOCV epochs ([analysis/snn_energy.py](analysis/snn_energy.py)):
- **Mean firing rate ≈ 10.6%** (89% temporal sparsity).
- LIF layers ≈ **2.1%** of an equivalent dense-ANN's energy (≈48× lower); whole encoder ≈10.5%
  (45 nm CMOS proxy; AC = 0.9 pJ vs MAC = 4.6 pJ, Horowitz 2014).
- The SNN is accuracy-neutral (ablation §13.4) → retained on **efficiency** grounds.

### 13.3 Integrated-Gradients feature attribution (fills `tab:features`)
[analysis/integrated_gradients.py](analysis/integrated_gradients.py): IG to the HIGH_ENGAGEMENT
logit, **mean baseline**, 24 steps, 37 folds (each fold's checkpoint attributes its own held-out
subject). Importance = normalised mean |IG| per input element:

| Modality | Feature | Importance |
|---|---|---|
| ROI | ROI saliency | 0.523 |
| EEG | Posterior alpha power | 0.176 |
| EEG | Frontal theta power | 0.160 |
| ET | Gaze position | 0.131 |
| ET | Pupil dynamics | 0.011 |
| EEG | Frontal functional connectivity | ~0 (marginal IG; structural role shown by the graph ablation instead) |

Note: attribution is to the model's **actual inputs** (band power, gaze, pupil, ROI), not to the
engineered dwell/fixation features (those define the labels and are not model inputs).

### 13.4 Component ablation — honest reading (LOSOCV, paired Wilcoxon, calibrated)
| Removed | Δ bal-acc | p | Reading |
|---|---|---|---|
| Dynamic graph | **−0.057** | **0.007** | only significant component (accuracy driver) |
| Contrastive / MMD / ET | −0.024 … −0.018 | n.s. | small, non-significant |
| SNN / ROI / Fusion-TF / Neuro-symbolic | ≈0 (NS +0.019) | n.s. | accuracy-neutral; kept for efficiency / interpretability |

### 13.5 Subject-adaptive calibration ([evaluation/calibration_experiment.py](evaluation/calibration_experiment.py))
The model's ROC-AUC (0.90) far exceeds its fixed-threshold balanced accuracy (0.74): a
per-subject probability-scale shift. A **leakage-free prevalence-matched** threshold (uses only
the test subject's predicted-prob distribution + training class prior) raises mean balanced
accuracy to **0.778** and MCC to 0.50 (heterogeneous; Wilcoxon p=0.33) → the gap is a
threshold-transfer effect, not a representational one.

### 13.6 Evaluation framing (label-source confound)
Engagement labels are an eye-tracking gaze-feature composite (§1). The leakage-free
**EEG-encoder comparison** is the headline (proposed EEG branch significantly beats 8 EEG
encoders, Holm-Wilcoxon p<0.05); ET/fusion comparisons are a modality-information analysis.
`leakage_audit.py` Check 3b quantifies label recoverability from ET (subject-grouped CV AUC ≈ 0.77).
