# Manuscript Blueprint — *Brain Informatics* (Springer Nature)

**Working title:** *An Interpretable Neuro-Symbolic Dynamic Graph Cross-Modal Transformer for Consumer Engagement Prediction Using EEG and Eye Tracking*

**Framework:** INS-HDGS-CMT · **Dataset:** NeuMa · **Signals:** EEG + Eye-tracking · **Task:** binary consumer-engagement prediction (HIGH vs LOW) · **Protocol:** 37-fold Leave-One-Subject-Out CV.

This document is a *writing blueprint*: a section-by-section plan with (i) the narrative role of each section, (ii) the paragraph-level content to write, (iii) the **verified, real numbers** from our experiments to cite, (iv) faithful equations and figure/table captions, and (v) transitions. British English throughout; past tense for Methods/Results.

---

## ⚠️ FIDELITY & INTEGRITY NOTES — read before drafting

These reconcile the requested template with what the trained model *actually does*. Writing the template verbatim would misrepresent the method and breach the ethical-writing rules. Resolve each before submission.

1. **Connectivity = Pearson, not PLV.** The model that produced every result uses `CONN_METHOD = "pearson"` with threshold `τ = 0.30` (`config/settings.py`; confirmed by the case-study panels labelled "Pearson functional connectivity (DynamicGAT input)"). PLV is implemented in `build_connectivity.py` but was **not** used. → Either (a) rewrite Eq. 2–3 as Pearson correlation (recommended; provided below), **or** (b) re-run LOSOCV with `CONN_METHOD="plv"` and report *those* numbers. Do not present PLV equations beside Pearson-derived results.
2. **Spiking-encoder input is a learnable band-weighted proxy, not raw EEG.** The LIF encoder receives a softmax-weighted band-power proxy (weights initialised to favour θ/α), not the raw waveform. Keep Eq. 6–8 (LIF dynamics are correct) but state the input explicitly.
3. **ET model input is the raw 600×3 sequence (gaze-x, gaze-y, pupil), not hand-crafted features.** Fixation duration / dwell time (Eq. 4–5) are computed for *explainability/visualisation* (Fig. 7B), not as model inputs. Frame them as derived interpretive measures, not encoder inputs.
4. **Neuro-symbolic fusion uses a *learned* gate, not a fixed β.** Decision = differentiable 8-rule layer + bypass classifier combined by a learned gate. Rewrite Eq. 15 with a learned/sample-wise gate (provided below) rather than a constant β.
5. **Cross-modal attention is *directed* and *gated*.** Two directed streams (EEG←graph, EEG←ET) over three modality tokens, merged by a gated residual — not a symmetric concatenation. Adjust Eq. 12–13 (provided).
6. **Labels = global multimodal threshold, 2 classes.** Engagement HIGH/LOW is defined by a single global threshold on a multimodal engagement score (per-subject-median labelling was tested and *rejected* — it collapsed cross-subject performance). State the global-threshold definition; mention the rejected alternative only if discussing label design.
7. **Headline metric discipline.** Report the **fixed-0.50 threshold, 37-fold LOSOCV** numbers: **acc 0.797, ROC-AUC 0.901, MCC 0.507, balanced-acc 0.753**. Do **not** headline the test-tuned 0.804 or the leaky per-subject oracle 0.92 (use the oracle only as an explicit upper bound in Discussion). Lead with AUC + balanced-acc + MCC because the cohort is class-imbalanced per subject.
8. **External benchmark numbers (Mallick, Usman, EEGEyeNet) are UNVERIFIED.** Mark every external cell `[VERIFY: source, year]`. Note the **task-mismatch caveat**: Usman et al. predict *Buy/Not-Buy*, which is **not** the engagement task here; any comparison must say so explicitly. Do not imply head-to-head superiority on identical tasks.
9. **Integrated Gradients (Eq. 20)** — confirm it is actually run. Current explainability uses graph-attention weights, cross-attention maps, ROI attention, and rule activations (all real). If IG is not computed, either implement it or replace Eq. 20 with the attention-/attribution-based explanation actually used.
10. **Component ablation (Table 5)** is **still running** at time of writing (only `no_snn` started). Until it completes, populate Table 5 from the **completed modality ablation** (fusion vs EEG-only vs ET-only) and mark architectural rows `[pending component-ablation run]`. Do not invent ablation deltas.

---

## VERIFIED RESULT LEDGER (single source of truth for all numbers)

Cite from here; keep decimals consistent (3 dp for metrics).

| Quantity | Value | Source |
|---|---|---|
| Subjects / folds (LOSOCV) | 37 | `losocv_repro_focal_g3p0_effective_num_37.csv` |
| Total test epochs | 347 (≈9.4 / subject) | same |
| EEG | 24 channels, 300 Hz, 1500 samples (5 s) | dataset / provenance |
| EEG graph features | 10 windows × 24 ch × 5 band-powers; adjacency 10×24×24 | model trace |
| Eye-tracking | 600 samples (120 Hz), 3 features: gaze-x, gaze-y, pupil | dataset |
| ROIs | 10 (learned attention regions) | model |
| **Production model (focal γ=3 + effective-num, fixed-0.50)** | **acc 0.797 · ROC-AUC 0.901 · MCC 0.507 · balanced-acc 0.753 · PR-AUC ≈0.904 · F1 ≈0.69** | `RESULTS_threshold_and_baseline.md` §2 |
| Production mean ± SD across subjects | acc 0.797 ± 0.18; AUC 0.901 ± 0.14; MCC 0.46–0.51 ± 0.37 | focal CSV |
| 37-subject mean 95% CI | ≈ ±0.06 (acc) | §3 of results md |
| γ=0 baseline (cross-fold threshold) | acc 0.770 · AUC 0.887 · MCC 0.490 · bal-acc 0.745 | §1 |
| Classical baselines (pooled) | LogReg AUC 0.509 · SVM-RBF AUC 0.405 · RF AUC 0.479 (≈ chance) | `classical_baselines.csv` |
| Best subject | **S24**: acc 0.923 · bal-acc 0.938 · MCC 0.854 · AUC 1.000 | focal CSV (fold-22) |
| Other strong subjects | S23 (0.917/0.845), S01 (0.933/0.829), S38 (1.000) | focal CSV |
| Ranking-failure subject | **S21**: AUC 0.40, MCC −0.45 (genuine representation failure) | focal CSV / S21 analysis |
| Collapsed-F1 folds | 5 (S21, S39, S34, S03, S13): model predicts ≈all-LOW | §1 |
| Oracle per-subject ceiling | acc 0.920 (+0.123 over fixed-0.5) — **upper bound only** | §3 |
| Modality ablation (LOSOCV MCC) | fusion 0.452 > ET-only 0.394 ≫ EEG-only 0.071 | modality study |
| Case-study sample (Fig. 7) | S24, epoch 11, ImagePage_6, GT HIGH, ensemble p(HIGH)=0.698, correct | `case_study_S24/provenance.json` |

**Key interpretive line (use in Results + Discussion):** AUC ≈ 0.90 means the representation *ranks* engagement well; the accuracy shortfall vs the 0.92 oracle is a *per-subject decision-threshold* problem, not a representation problem — and DANN, MMD, SSL and reliability-fusion were each tested and did **not** help, which is itself a reportable, honest finding.

---

# Abstract (structured, ~250 words)

Write last; one paragraph or labelled sentences. Draft:

- **Background.** Consumer engagement underpins marketing decisions, yet self-reports are retrospective, subjective and weakly predictive of behaviour. Electroencephalography (EEG) and eye-tracking (ET) offer complementary, objective windows onto neural engagement and overt visual attention, but existing multimodal models rely on early fusion, are rarely validated subject-independently, and seldom offer mechanistic interpretability.
- **Objective.** We propose INS-HDGS-CMT, an interpretable neuro-symbolic dynamic-graph cross-modal transformer that integrates EEG functional-connectivity dynamics with eye-tracking attention for subject-independent engagement prediction.
- **Methods.** Using the NeuMa dataset (37 participants; EEG 24 ch @ 300 Hz; ET gaze + pupil @ 120 Hz), dynamic EEG functional graphs (Pearson connectivity) are encoded by a graph-attention network with a leaky integrate-and-fire spiking temporal pathway; eye-tracking is encoded by a transformer with learned region-of-interest attention; modalities are merged by a ROI-guided directed cross-attention transformer and refined by a differentiable 8-rule neuro-symbolic layer. Models were evaluated by 37-fold leave-one-subject-out cross-validation.
- **Results.** INS-HDGS-CMT achieved ROC-AUC 0.901, accuracy 0.797 ± 0.18 and MCC 0.507, exceeding a focal-loss baseline (AUC 0.887) and near-chance classical baselines (AUC ≤ 0.54). Fusion outperformed unimodal variants (MCC 0.452 vs 0.394 ET / 0.071 EEG). Graph-attention and cross-modal maps localised engagement to frontal-parietal connectivity and salient visual regions.
- **Conclusions.** Principled multimodal integration with neuro-symbolic reasoning yields accurate, interpretable, subject-independent engagement prediction, and shows the residual error is a per-subject calibration problem rather than a representational one.

**Keywords (6–8):** consumer neuroscience; neuromarketing; electroencephalography; eye-tracking; graph neural networks; spiking neural networks; neuro-symbolic learning; explainable AI.

---

# 1 Introduction

**Narrative role:** Problem → why existing approaches fall short → gap → proposal → contributions. *No survey-style listing.* End on bullet contributions.

**1.1 Background and Motivation.** Open broadly: engagement as a driver of attention, memory encoding and purchase intent; the economic stakes of measuring it. Argue that questionnaires/self-reports are *retrospective, introspective and socially biased*, capture conscious appraisal rather than moment-to-moment processing, and correlate weakly with behaviour. Motivate objective neurophysiological measurement. End by signposting EEG and ET.

**1.2 EEG and Eye Tracking in Consumer Neuroscience.** Introduce the two modalities *separately, then their complementarity*. EEG: millisecond temporal resolution; frontal asymmetry/approach-motivation, theta/alpha dynamics linked to attention and cognitive engagement. ET: overt visual attention, fixations, dwell, pupil-linked arousal. Make the complementarity explicit: EEG indexes *internal* neural engagement; ET indexes *overt* attentional allocation — neither alone is sufficient.

**1.3 Challenges in Multimodal Neuromarketing.** Critically (not chronologically) analyse weaknesses of prior multimodal work, organised by theme: (i) **early/naïve fusion** that ignores modality-specific temporal structure and cross-modal dependency; (ii) **subject-dependent validation** (within-subject or random splits) that inflates accuracy via identity leakage and does not test deployment to new consumers; (iii) **limited explainability** — black-box predictions unsuitable for scientific or regulatory use; (iv) **absence of symbolic/structured reasoning**, so models cannot encode or expose decision rules.

**1.4 Research Gap.** State crisply: *no existing EEG–ET framework simultaneously (a) models dynamic EEG connectivity, (b) fuses modalities through directed cross-attention, (c) validates strictly subject-independently, and (d) provides neuro-symbolic, neurophysiologically interpretable explanations.*

**1.5 Contributions** (bullet list, end of section):
- A subject-independent multimodal framework (INS-HDGS-CMT) integrating dynamic EEG functional-connectivity graphs with eye-tracking attention.
- A dynamic graph–spiking EEG encoder coupling Pearson-connectivity graph attention with LIF spiking temporal dynamics.
- A ROI-guided directed cross-modal transformer that lets neural representations selectively attend to visual-attention evidence.
- A differentiable neuro-symbolic decision layer that refines and exposes predictions as interpretable rule activations.
- A rigorous 37-fold LOSOCV evaluation on NeuMa with statistical testing, modality/component ablation, and neurophysiological interpretation, including an honest analysis showing the residual error is a per-subject calibration problem.

*Transition:* "Before detailing the framework, we describe the dataset, acquisition and evaluation protocol."

*(Optional Related Work as §1.x or standalone §2 — if used, follow the thematic synthesis rules: group EEG / ET / EEG–ET fusion / explainability, compare-and-contrast, end with unresolved challenges. No "Author A did X" enumeration.)*

---

# 2 Materials and Methods

**Style:** factual, reproducible, past tense; *no results, no superiority claims.*

**2.1 NeuMa Dataset.** Describe NeuMa: participants, brochure-page stimuli (ImagePage_1–6), consumer-engagement paradigm. State 37 participants entered the analysis. [VERIFY any demographic detail against the NeuMa source before stating.]

**2.2 Experimental Paradigm and Signal Acquisition.** Simultaneous EEG + ET during stimulus viewing. EEG: 24-channel 10–20 montage (list channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz, Oz, FC1, FC2, FC5, FC6), 300 Hz. ET: binocular gaze + pupil, 120 Hz, reduced to 3 model features (gaze-x, gaze-y, pupil).

**2.3 EEG Preprocessing.** Report exactly what was done (filtering, artefact handling, per-subject z-scoring/normalisation, montage harmonisation to 24 canonical channels, zero-filling of absent channels). Justify per-subject z-scoring as removing inter-subject amplitude shift (a subject-independence safeguard).

**2.4 Eye-Tracking Preprocessing.** Gaze normalisation to screen coordinates; both-eyes averaging; handling of missing samples; epoching to 600 samples (5 s). Note that fixation/dwell are derived later for interpretation.

**2.5 EEG–Eye-Tracking Synchronisation.** Temporal alignment of the two streams to a common 5-s epoch grid; resampling/window correspondence (10 EEG band-power windows ↔ ET sequence).

**2.6 Epoch Generation and Label Definition.** 5-s epochs. EEG → 10 windows × 24 ch × 5 band-powers + dynamic adjacency (10×24×24); ET → 600×3. **Labels:** binary HIGH/LOW engagement via a *single global threshold* on a multimodal engagement score (state the score and threshold rule). Note class balance (cite Table 1). Mention briefly that per-subject-median labelling was evaluated and rejected (collapsed cross-subject generalisation) to justify the global choice.

**2.7 Evaluation Protocol.** 37-fold LOSOCV: each fold holds out one subject entirely for testing and trains on the remaining 36 — no subject appears in both train and test (subject-independent). State the 5-member ensemble per fold, fixed-0.50 decision threshold for the headline, and that metrics are computed per fold then averaged (report mean ± SD). Name metrics: accuracy, balanced accuracy, precision, recall, F1, MCC, ROC-AUC, PR-AUC; justify AUC/balanced-acc/MCC as primary under per-subject imbalance.

**Figure 1 — NeuMa experimental paradigm and processing pipeline.**
*Caption:* "Overview of the NeuMa acquisition and processing pipeline. Participants viewed brochure stimuli while EEG (24 channels, 300 Hz) and eye-tracking (gaze and pupil, 120 Hz) were recorded simultaneously. Signals were preprocessed, temporally synchronised, segmented into 5-s epochs, and labelled as HIGH or LOW consumer engagement via a global multimodal threshold. *What to observe:* the strictly parallel, subject-independent flow from acquisition to labelled epochs. *Why it matters:* it establishes the reproducible data substrate on which all subsequent modelling and leave-one-subject-out evaluation are built." (Use existing `figures/fig1_dataset.*`.)

**Table 1 — NeuMa dataset characteristics.**
Columns: Subjects | EEG channels | EEG sampling rate | ET features | ET sampling rate | Epoch length | Total epochs | Class distribution.
Row: 37 | 24 | 300 Hz | 3 (gaze-x, gaze-y, pupil) | 120 Hz | 5 s | 347 (test) | [report HIGH/LOW counts — compute pooled; per-subject example S24 was 5 HIGH / 8 LOW]. *Self-contained caption; define abbreviations in a footnote.*

*Equations:* minimal — only define the epoching/normalisation if needed.

*Transition:* "Given this data substrate, we now present the INS-HDGS-CMT framework."

---

# 3 Proposed Method

**Style:** the scientific core. For each subsection: **motivation → formulation → computation → expected contribution.** Define every symbol immediately. Notation: italic scalars (*x*), bold-lower vectors (**h**), bold-upper matrices (**A**, **W**). embed dim *d* = 128 throughout.

**3.1 Overview.** Walk input→prediction. Two-stream encoder (EEG dynamic graph-spiking; ET attention) → ROI-guided modulation → directed cross-modal transformer → neuro-symbolic decision → HIGH/LOW. Reference **Fig. 2** (use the new clean schematic `figures/architecture_v2/OuterModel_INS_HDGS_CMT.*` for the overview and `InnerModel_INS_HDGS_CMT.*` for the detailed view — these are vector schematics, *not* code screenshots, satisfying the figure rules).

**3.2 Dynamic EEG Functional Graph Construction.** *Motivation:* engagement is a network phenomenon; static connectivity discards within-epoch dynamics, so we build a sequence of graphs.

- **Eq. 1 — Dynamic graph.** $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t, \mathbf{A}_t)$ — at window *t*, nodes $\mathcal{V}$ are the 24 EEG channels, edges $\mathcal{E}_t$ and weighted adjacency $\mathbf{A}_t$ vary across the 10 windows. Node features are the 5 band-powers per channel.
- **Eq. 2 — Pearson functional connectivity (FAITHFUL replacement for PLV).**
  $$\mathbf{A}_t(i,j) = \frac{\sum_{s}\big(x_i^{t}(s)-\bar{x}_i^{t}\big)\big(x_j^{t}(s)-\bar{x}_j^{t}\big)}{\sqrt{\sum_s (x_i^{t}(s)-\bar{x}_i^{t})^2}\sqrt{\sum_s (x_j^{t}(s)-\bar{x}_j^{t})^2}}$$
  where $x_i^t(s)$ is the EEG sample of channel *i* at time *s* within window *t*; the numerator/denominator give the Pearson correlation between channels *i* and *j*. *(If you instead choose PLV, present the PLV form from your template and re-run — do not mix.)*
- **Eq. 3 — Thresholded adjacency.** $\mathbf{A}_t(i,j) = \mathbf{A}_t(i,j)$ if $|\mathbf{A}_t(i,j)| \ge \tau$, else $0$, with $\tau = 0.30$. *Why:* sparsifies to retain salient functional links and suppress spurious low correlations.
- *Computation:* per-window correlation → threshold → 10 adjacency matrices fed to graph attention. *Contribution:* preserves temporal evolution of connectivity.

**3.3 Eye-Tracking Attention Feature Encoding.** *Motivation:* overt attention complements neural engagement. *State clearly:* the **encoder input is the raw 600×3 sequence**; fixation/dwell are derived for interpretation.
- **Eq. 4 — Fixation duration (derived/interpretive):** $FD = t_{\text{offset}} - t_{\text{onset}}$.
- **Eq. 5 — Dwell time (derived/interpretive):** $DT = \sum FD$ over a region.
- Describe the ET transformer encoder (2 layers, 4 heads, hidden 64) → **e**$_{ET}\in\mathbb{R}^{128}$ and a learned ROI-attention head over 10 regions. *Contribution:* data-driven visual-attention representation aligned to stimulus regions.

**3.4 Spiking Temporal Representation Learning.** *Motivation:* energy-efficient, biologically grounded temporal coding of EEG. *State input:* a learnable softmax band-weighted proxy (initialised to favour θ/α) over the band-power tensor, fed to a 2-layer LIF encoder with *T* = 10 steps.
- **Eq. 6 — LIF dynamics:** $\tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + R\,I(t)$.
- **Eq. 7 — Spike:** $S(t)=1$ if $V \ge V_{th}$ else $0$.
- **Eq. 8 — Discrete membrane update:** $V_t = \lambda V_{t-1} + I_t - S_t V_{th}$ (surrogate-gradient training). Define $\tau_m, V_{\text{rest}}, R, I, V_{th}, \lambda$. *Contribution:* sparse temporal code (**s**$_{SNN}\in\mathbb{R}^{128}$) complementary to the graph embedding.

**3.5 Hierarchical Dynamic Graph Learning.** *Motivation:* learn channel interactions per window then integrate across time.
- **Eq. 9 — Attention coefficient:** $e_{ij} = \text{LeakyReLU}\big(\mathbf{a}^\top[\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j]\big)$.
- **Eq. 10 — Node update:** $\mathbf{h}'_i = \sigma\big(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\mathbf{W}\mathbf{h}_j\big)$, with $\alpha_{ij}=\text{softmax}_j(e_{ij})$. Define **W**, **a**, $\mathcal{N}_i$. Note: 4 attention heads × 32-d, then a 3-layer temporal transformer (ff 512) integrates the 10 window embeddings → **g**$_{graph}\in\mathbb{R}^{128}$. The graph and spiking embeddings are merged (concat → Linear 256→128 → GELU → LayerNorm) into **e**$_{EEG}$. *Contribution:* hierarchical spatial-then-temporal EEG representation; ROI-guided modulation gates the adjacency before attention.

**3.6 Cross-Modal Transformer Fusion.** *Motivation:* let neural representation selectively read visual-attention evidence (directed, gated).
- **Eq. 11 — Scaled dot-product attention:** $\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\big)\mathbf{V}$.
- **Eq. 12 — Directed cross-attention (two streams):**
  $\mathbf{Z}_{EEG\leftarrow graph} = \text{Attention}(\mathbf{Q}_{EEG}, \mathbf{K}_{graph}, \mathbf{V}_{graph})$, and
  $\mathbf{Z}_{EEG\leftarrow ET} = \text{Attention}(\mathbf{Q}_{EEG}, \mathbf{K}_{ET}, \mathbf{V}_{ET})$.
- **Eq. 13 — Gated residual fusion (FAITHFUL):** $\mathbf{z}_{\text{fusion}} = \mathbf{e}_{EEG} + \mathbf{u}\odot\mathbf{Z}_{EEG\leftarrow graph} + (\mathbf{1}-\mathbf{u})\odot\mathbf{Z}_{EEG\leftarrow ET}$, where **u** is a learned gate (replace your concatenation form, or keep concatenation only if you switch to `CrossModalFusion`). 3 modality tokens (graph, EEG, ET), 4 heads, 3 layers. *Contribution:* asymmetric, content-dependent integration → **z**$_{\text{fusion}}\in\mathbb{R}^{128}$.

**3.7 Neuro-Symbolic Decision Reasoning.** *Motivation:* interpretable, rule-based refinement.
- **Eq. 14 — Rule aggregation:** $R = \sum_{k=1}^{8} w_k\,\rho_k(\mathbf{z}_{\text{fusion}})$, where $\rho_k\in[0,1]$ is the (soft, differentiable) activation of rule *k* and $w_k$ its learned weight. (Your $I(C_k)$ indicator is the hard-rule idealisation; state the differentiable relaxation actually used.)
- **Eq. 15 — Gated decision (FAITHFUL):** $\hat{\mathbf{y}} = (\mathbf{1}-\boldsymbol{\beta})\odot\hat{\mathbf{y}}_{DL} + \boldsymbol{\beta}\odot R$, where $\boldsymbol{\beta}$ is a **learned, sample-wise gate** (not a constant), $\hat{\mathbf{y}}_{DL}$ the bypass-classifier logits. *Contribution:* auditable decisions; bypass path preserves accuracy when no rule fires.

**3.8 Classification and Optimisation.**
- **Eq. 16 — Classification loss (focal form actually used):** state focal cross-entropy with γ=3 and effective-number class weighting; the plain CE $L_{cls} = -\sum_c y_c \log \hat{y}_c$ is the special case γ=0. *(Be accurate: the production model uses focal-γ3, which the ablation in §4 shows beats γ0.)*
- **Eq. 17 — Total loss:** $L_{\text{total}} = L_{cls} + \lambda_1 L_{\text{rule}} + \lambda_2 L_{\text{contrast}}\,(+\,\lambda_3 L_{\text{MMD}})$, where $L_{\text{rule}}$ is rule diversity+sparsity regularisation and $L_{\text{contrast}}$ a supervised EEG↔ET contrastive term. (Map your $L_{graph}/L_{symbolic}$ onto these real terms; note DANN/MMD were tested and did not improve LOSOCV — see §6.)

**Figure 2 — Overall INS-HDGS-CMT architecture.** Use the clean vector schematics. *Caption follows: what it shows (two-stream encoder → ROI-guided fusion → neuro-symbolic decision, with tensor shapes) → what to observe (where each modality enters and how they merge) → why it matters (architecture realises the dynamic-graph + cross-modal + symbolic design).*

**Figure 3 — Dynamic EEG graph construction.** *Caption:* EEG epoch → per-window correlation → thresholded adjacency → dynamic graph sequence → graph attention. (If you keep "Phase Extraction/PLV" in the figure, it must match the equations and the run — otherwise relabel to "windowing → Pearson correlation".)

**Figure 4 — Cross-modal fusion and neuro-symbolic reasoning.** *Caption:* EEG and ET representations → directed cross-attention → gated joint representation → differentiable rules → refined prediction. Match notation to Eq. 11–15.

**Table 2 — INS-HDGS-CMT implementation details.** Module | Configuration. Rows (real defaults): embed dim 128; EEG band-power windows 10; DynamicGAT 4 heads × 32-d + 3-layer temporal transformer (ff 512); LIF encoder 2 layers, T=10; ET encoder 2 layers, 4 heads, hidden 64; ROIs 10; fusion 3 modalities, 4 heads, 3 layers; rules 8; dropout 0.30; optimiser [AdamW — VERIFY], learning rate [VERIFY], batch size [VERIFY], epochs [VERIFY], loss focal-γ3 + effective-num weighting + rule/contrastive aux; ensemble 5 members/fold; threshold 0.50. *Fill [VERIFY] from the training config before submission.*

*Transition:* "Having defined the framework, we evaluate it under strict subject-independent validation."

---

# 4 Experiments and Results

**Style:** each subsection = observation → quantitative evidence → interpretation. Mean ± SD; absolute *and* relative gains; don't restate every table cell.

**4.1 Experimental Setup.** Recap LOSOCV (37 folds), ensemble, hardware [VERIFY GPU], software, metrics, decision rule. State that the *same* preprocessing and threshold were applied to all models to ensure fair comparison.

**4.2 Performance of the Proposed Model.** Headline: INS-HDGS-CMT reached **ROC-AUC 0.901, accuracy 0.797 ± 0.18, balanced-acc 0.753, MCC 0.507** over 37 unseen subjects. Interpret: AUC ≈ 0.90 indicates strong, subject-independent ranking of engagement; the ±0.18 spread foreshadows subject heterogeneity (§4.6).

**4.3 Comparison with Existing EEG–ET Studies (Table 3).** Position relative to prior work; **explicitly flag the task-mismatch** (Usman = Buy/Not-Buy ≠ engagement). Do not claim identical-task superiority. State our contribution is strict subject-independent validation with interpretability.

**Table 3 — Benchmark comparison with existing EEG–ET studies.** Study | Year | Dataset | EEG | ET | Method | Protocol | Accuracy | F1. Rows: Mallick et al. `[VERIFY]`; EEGEyeNet-based `[VERIFY]`; "Predicting fixations from EEG" `[VERIFY]`; Usman et al. `[VERIFY + task-mismatch note]`; **INS-HDGS-CMT | 2026 | NeuMa | 24 ch | gaze+pupil | dynamic-graph + spiking + cross-modal + neuro-symbolic | LOSOCV | 0.797 | 0.69** (bold = ours). Footnote: external values to be verified from primary sources; protocols differ.

**4.4 Comparison with Baseline Models (Table 4).** Internal, same data/protocol.
**Table 4 — Internal baseline comparison.** Model | Accuracy | Precision | Recall | F1 | AUC. Rows from real data: EEG-only (MCC 0.071 → weak; report acc/AUC), ET-only (MCC 0.394), CNN-LSTM `[run or VERIFY]`, Transformer `[run or VERIFY]`, TSception `[run or VERIFY]`, **INS-HDGS-CMT (0.797 / … / 0.901)** bold. Classical chance-level baselines (LogReg/SVM/RF, AUC ≤ 0.54) can be a footnote or extra rows to show the task is non-trivial. *Observation:* fusion ≫ unimodal; deep multimodal ≫ classical.

**4.5 Ablation Study (Table 5).** Until the component ablation finishes, report the **completed modality ablation** as the core evidence and mark architectural rows pending.
**Table 5 — Ablation study.** Configuration | Accuracy | F1 | ΔAccuracy. Real rows: Full model (0.797, —, ref); − EEG branch → ET-only (MCC 0.394; report acc/Δ); − ET branch → EEG-only (MCC 0.071; large drop); − Dynamic Graph / − Spiking Encoder / − Transformer / − Neuro-Symbolic → `[pending component-ablation run]`. *Observation:* removing ET collapses performance most (EEG-graph alone is weak), establishing ET as the dominant carrier and fusion as additive.

**4.6 Subject-Wise LOSOCV Analysis (Fig. 5).** Report the spread: best **S24 (acc 0.923, AUC 1.00)**, strong S23/S01/S38; failure **S21 (AUC 0.40, MCC −0.45)** and 5 collapsed-F1 folds. Interpret heterogeneity: most subjects are well-modelled; a minority show ranking failure (S21) traced to representation, not thresholding.
**Figure 5 — Subject-wise LOSOCV performance.** Use `figures/fig3_losocv_results.*`/`fig6_losocv.*`. *Caption: per-subject bars (AUC/MCC), highlighting S24 and S21; what to observe = the long tail; why it matters = motivates per-subject calibration.*

**4.7 Statistical Significance Analysis (Table 6).** Compare INS-HDGS-CMT vs γ0 baseline and vs best unimodal using paired tests over the 37 fold-scores (Wilcoxon signed-rank) and McNemar on pooled predictions; report p and effect size.
- **Eq. 18 — McNemar:** $\chi^2 = \frac{(|b-c|-1)^2}{b+c}$ (b, c = discordant counts).
- **Eq. 19 — Effect size:** $r = Z/\sqrt{N}$.
**Table 6 — Statistical significance.** Comparison | Test | p-value | Effect size. `[Compute from fold CSVs; report honestly — with n=37 and ±0.06 CI some contrasts may not reach significance; say so.]`

**Figure 6 — ROC curve comparison.** Proposed vs baselines (pooled or mean ROC with CI band). *Caption: curves + AUC legend; observe separation from chance diagonal and from baselines; matters because it is threshold-independent.*

*Transition:* "Beyond aggregate accuracy, a key contribution is interpretability, which we examine next."

---

# 5 Explainability and Neurophysiological Interpretation

**Style:** insight, not decoration. For each: feature → neurophysiological relevance → consumer interpretation → implication. *No causal claims from attribution.* Use the **real S24 case-study panels** (`figures/case_study_S24/`).

**5.1 EEG Graph-Attention Analysis.** Which channels/edges received highest attention; relate to frontal–parietal engagement networks, θ/α involvement. Consumer reading: heightened fronto-parietal coupling during HIGH engagement.

**5.2 Cross-Modal Attention Interpretation.** Read the directed cross-attention maps (EEG←graph, EEG←ET): when neural representation weighted visual evidence more strongly. Interpret as moments where overt attention informed neural engagement decoding.

**5.3 Eye-Tracking Visual-Attention Analysis.** ROI-attention heatmaps + fixation/dwell (Eq. 4–5) on the stimulus; link salient regions to engagement. Consumer reading: which brochure regions drove engagement.

**5.4 Case-Level Decision Explanation.** Trace one real sample end-to-end: **S24, ImagePage_6, GT HIGH, ensemble p(HIGH)=0.698, correctly classified** — show EEG graph, spike raster, gaze/fixations, ROI attention, cross-attention, the 8 rule activations, and final probability. This is the interpretability showcase.

**5.5 Neurophysiological Relevance of Learned Features.** Synthesise: do learned features align with known engagement correlates (frontal asymmetry, θ/α, parietal attention, pupil-linked arousal)? State alignment honestly; flag where the model relies on features without a clean physiological story.

- **Eq. 20 — Integrated Gradients** (only if actually computed): $IG_i(x) = (x_i - x'_i)\int_0^1 \frac{\partial F(x' + \alpha(x-x'))}{\partial x_i}\,d\alpha$. Otherwise replace with the attention-/rule-activation attribution actually used and say so.

**Figure 7 — Explainability panel (A EEG attention/topography; B ET fixation heatmaps; C cross-modal attention; D neuro-symbolic reasoning trace).** Built from real S24 intermediates. *Caption per sub-panel: what it shows → what is observed → neurophysiological meaning.*

**Table 7 — Most influential EEG and ET features.** Modality | Feature | Importance | Neurophysiological interpretation. Populate from attention/attribution rankings; give a physiological reading per row; avoid causal language.

*Transition:* "We now interpret why the framework behaves as observed and what it means for consumer neuroscience."

---

# 6 Discussion

**Style:** explain *why*; do not repeat Results; no new results; no overstatement.

**6.1 Principal Findings.** Subject-independent AUC ≈ 0.90 with interpretability; fusion > unimodal; ET dominant, EEG-graph complementary; residual error is a *decision-threshold* problem (oracle 0.92) not representation.

**6.2 Comparison with Prior Neuromarketing Studies.** Agreements/disagreements; emphasise our stricter LOSOCV and interpretability; restate the Usman task-mismatch caveat; avoid claiming SOTA on identical tasks unless verified.

**6.3 Methodological Strengths.** Dynamic connectivity + spiking temporal code; directed gated cross-attention; differentiable rules; rigorous validation; transparent negative results (DANN/MMD/SSL/reliability-fusion did not help — a contribution to the literature's reproducibility).

**6.4 Practical Implications.** Subject-independent deployment to new consumers; interpretable evidence for marketing/regulatory use; few-shot per-subject calibration (2–4 labelled trials, or a 2-component GMM on per-subject probabilities) as a realistic route to the +0.12 accuracy headroom.

**6.5 Limitations (constructive, honest).** Single dataset (NeuMa); modest cohort (37) with wide subject variance (±0.18); per-subject imbalance and small per-subject n (≈9 epochs); residual signal noise; a minority of subjects (S21) show genuine ranking failure; computational cost of the multi-branch model; binary engagement abstraction.

**6.6 Future Work.** Few-shot/unsupervised per-subject decision calibration; multi-dataset generalisation; graded/continuous engagement; PLV/coherence connectivity comparison; lighter-weight deployment; linking rules to formal cognitive-engagement theory.

---

# 7 Conclusions

One tight paragraph: problem (objective, subject-independent engagement measurement) → solution (INS-HDGS-CMT: dynamic-graph + spiking EEG, attention ET, directed cross-modal fusion, neuro-symbolic decision) → key findings (AUC 0.901, acc 0.797, fusion>unimodal, interpretable, residual error is calibration not representation) → contribution to neuromarketing and Brain Informatics (a rigorous, interpretable multimodal template) → one strong closing sentence on principled, transparent consumer-neuroscience modelling. *No new results, no abstract repetition.*

---

# Supplementary & Back-matter

- **S1** Subject-wise LOSOCV metrics — directly from `losocv_repro_focal_g3p0_effective_num_37.csv` (all 37 rows).
- **S2** Hyperparameter search space — from training configs `[VERIFY]`.
- **Fig. S1** Additional ROC curves (per-fold / per-baseline).
- **Fig. S2** Extended explainability (more subjects/epochs from `case_study_S24` + `case_study_S01`).
- **Author Contributions / Funding / Data & Code Availability / Declarations** (ethics approval & consent — NeuMa governance `[VERIFY]`; competing interests: none `[confirm]`).
- **References** — verify every entry is traceable; replace all `[VERIFY]` placeholders with primary sources.

---

## Asset map (what exists vs to-build)

| Manuscript item | Status / file |
|---|---|
| Fig. 1 dataset pipeline | exists `figures/fig1_dataset.*` |
| Fig. 2 architecture | use `figures/architecture_v2/OuterModel_*` + `InnerModel_*` (new, clean) |
| Fig. 3 dynamic graph | **to build** (vector schematic; relabel Pearson) |
| Fig. 4 fusion + neuro-symbolic | **to build** or crop from InnerModel |
| Fig. 5 subject-wise LOSOCV | exists `fig3_losocv_results.*` / `fig6_losocv.*` |
| Fig. 6 ROC comparison | **to build** from fold probabilities |
| Fig. 7 explainability | exists `figures/case_study_S24/` (real S24 panels) |
| Table 1 dataset | compute pooled class counts |
| Table 2 implementation | fill [VERIFY] from configs |
| Table 3 external benchmark | [VERIFY external sources] |
| Table 4 internal baselines | partial (modality + classical real); run CNN-LSTM/Transformer/TSception |
| Table 5 ablation | modality real; component ablation **running** |
| Table 6 significance | compute from fold CSVs |
| Table 7 influential features | from attention/attribution ranking |
| LaTeX skeleton | `paper/sn-article.tex` (Springer Nature template already present) |
