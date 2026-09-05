# Revision runs — what to execute on the GPU server and where the numbers go

> **Use `docs/BREV_RUNBOOK.md` and `bash scripts/revision/run_revision.sh <stage>`** —
> the driver wraps every command below with the production-pinned configuration,
> skips finished work, and `scripts/revision/verify_revision.py` re-checks all outputs.
> **Read consistency issue 2 first: the label is multimodal (EEG + gaze), not gaze-only.**

Every `\todoval{...}` marker in `paper/INS_HDGS_CMT_manuscript.tex` and
`paper/INS_HDGS_CMT_supplementary.tex` is a number that must come from one of
the runs below. All commands are run from `src/model/` on the machine that
holds the preprocessed NeuMa data (`src/data_pipeline/04_segmentation/S*/output/`)
and the production checkpoints. Set `PYTHONUTF8=1` on Windows.

Analyses that need **no** training (0–5 already computed locally on the
regenerated 385-epoch cohort and on the published per-fold CSVs) are marked ✅;
their outputs are already under `results/`.

| # | Reviewer point | Run | Output → manuscript location | Status |
|---|---|---|---|---|
| 0 | R1-1/2/3 ET contribution (existing runs) | `python ../../scripts/analysis/cross_modal_contribution.py` | `results/statistics/cross_modal_contribution.md` → Sec. 3.4b, Table S14 | ✅ |
| 1 | R2-8 label-free thresholds | `python ../../scripts/analysis/deployment_threshold.py` | `results/threshold_analysis/DEPLOYMENT_THRESHOLD.md` → Sec. 3.7, Table S12 | ✅ |
| 2 | R2-2 τ graph structure | `python ../../scripts/analysis/tau_sensitivity.py` | `results/sensitivity/tau_sensitivity.md`, Fig. S4 → Sec. 2.5.2/3.9, Table S7 (left) | ✅ |
| 3 | R2-9 ROI grid (data side) | `python ../../scripts/analysis/roi_grid_sensitivity.py` | `results/sensitivity/roi_grid_sensitivity.md`, Fig. S6 → Sec. 3.9, Table S11 (top/middle) | ✅ |
| 4 | R2-4 SNN measured cost (CPU) | `python ../../scripts/analysis/snn_energy_measured.py --ckpt <fold ckpt> --device cpu` | Table S9 (CPU rows) | ✅ (CPU) |
| 5 | **R1-1/3 true EEG-only branch** (no gaze input at all) | `python ../../scripts/analysis/run_component_ablation.py --variant eeg_only` (writes `results/ablation/abl_eeg_only/losocv_abl_eeg_only.csv` with the production-pinned γ=3 / effective-number / 5-member configuration) then `python ../../scripts/analysis/cross_modal_contribution.py --eeg-only-csv ../../results/ablation/abl_eeg_only/losocv_abl_eeg_only.csv` | Abstract, Sec. 3.2 (Table 3 first row), Sec. 3.4b, Table 7 new row, Table S14, Conclusions | ⏳ |
| 6 | R2-3 tuned baselines | `python baselines/run_baselines.py --models all --tune 12 --epochs 250 --patience 30 --device cuda` (writes `results/baselines/dl_tuned/`); then re-run `scripts/analysis/verify_table7_eeg_significance.py` and the Table 5 statistics script with the tuned CSVs | Tables 3, 4, 5, 8; Supplementary Tables S6, S10 | ⏳ |
| 7 | R2-2 τ downstream | `python sensitivity_sweep.py --sweep threshold --subjects $(python -c "from config.settings import SUBJECT_IDS;print(','.join(SUBJECT_IDS))") --n-ensemble 3 --epochs 120` then re-run `tau_sensitivity.py` | Table S7 (right columns), Sec. 3.9 | ⏳ |
| 8 | R2-9 ROI grid (model side) | `NEUMA_GRID_COLS=c NEUMA_GRID_ROWS=r python ../../scripts/analysis/run_component_ablation.py --variant full --label grid_cxr --results-root output/metrics` for (c,r) ∈ {(2,1),(3,2),(6,4),(8,6)} (do **not** use `main.py`: its defaults are 7 members / γ=2 / balanced weights, not the production configuration); then re-run `roi_grid_sensitivity.py` | Table S11 (bottom), Sec. 3.9 | ⏳ |
| 9 | R2-5 rule fidelity on existing checkpoints | `CUDA_VISIBLE_DEVICES="" python ../../scripts/analysis/rule_fidelity.py --ckpt-dir output/checkpoints/<label> --label <label> --fold-csv output/metrics/<label>/losocv_<label>.csv` | Sec. 3.10.3, Table S13 (first four rows) | ⏳ |
| 10 | R2-5 rule-only trained model | `python ../../scripts/analysis/run_component_ablation.py --variant ns_rule_only` (writes `results/ablation/abl_ns_rule_only/losocv_abl_ns_rule_only.csv`), then `rule_fidelity.py ... --rule-only-csv ../../results/ablation/abl_ns_rule_only/losocv_abl_ns_rule_only.csv` | Table 7 new row, Table S13 last row | ⏳ |
| 11 | R2-7 grounded rules (Fig. 8D) | `CUDA_VISIBLE_DEVICES="" python ../../scripts/analysis/ground_rules_to_electrodes.py --ckpt output/checkpoints/<label>/<label>_fold01_e*.pt --subjects S01 --label <label>`; population map with `--subjects all` | `results/explainability/rule_grounding_<label>.md`, `fig_rule_grounding_<label>.pdf` → replace panel D of `paper/figures/fig5_explainability.pdf`, Sec. 3.10.3 | ⏳ |
| 12 | R2-4 SNN measured cost (GPU power) | `python ../../scripts/analysis/snn_energy_measured.py --ckpt <fold ckpt> --device cuda` | Table S9 (GPU rows) | ⏳ |
| 13 | R2-6 learning curve | `python ../../scripts/analysis/learning_curve.py --sizes 10,16,24,30,37 --repeats 3 --n-ensemble 3 --epochs 150` (loss/regularisation pinned to the production configuration by default) | Fig. S5, Sec. 3.9 | ⏳ |

Every changed Python file was checked with `python -m py_compile`; nothing was trained locally.

`<label>` is the production run whose checkpoints exist on the server
(`ins_hdgs_cmt_ch19fix` per the project notes, or `repro_focal_g3p0_effective_num_37`
for the numbers currently in the manuscript — use the same run for every analysis).

## Notes on the code changes that support these runs

* `src/model/data/` (dataset loader + channel harmoniser) was missing from the
  public repository and is restored; nothing in `src/model` imports without it.
* `NeuroSymbolicRuleLayer(alpha_mode=...)` / `AblationConfig.ns_rule_only()` /
  `AblationConfig.ns_explain_only()` / env `NEUMA_NS_ALPHA_MODE` fix the bypass gate
  at 0 or 1. The model now also returns `rule_evidence`, `bypass_logits` and
  `bypass_alpha` in evaluation mode.
* `NEUMA_GRID_COLS`, `NEUMA_GRID_ROWS` (and `NEUMA_N_ROIS`) rebuild the ROI
  saliency vector on a different grid without code changes.
* `config/settings.py` resolves `PHASE3_DIR` relative to the source tree, so the
  analysis scripts run from any working directory.
* `src/model/baselines/run_baselines.py --tune N` performs the nested
  per-architecture random search with early stopping on the validation subject.

## Consistency issues found while preparing the revision (decide before resubmission)

1. **Table 3 "EEG branch" = `abl_no_et`**, whose `AblationConfig.no_et()` keeps
   `use_roi=True` and `use_roi_modulation=True`, i.e. the branch received the
   gaze-derived ROI dwell histogram. The revised text says so and reports the
   strictly gaze-free `AblationConfig.eeg_only()` run as the headline (run 5).
2. **The label actually used is the multimodal `engagement_phase3d.py` rule.**
   Verified on 2026-09-05: run on the regenerated epochs, `engagement_phase3d.py`
   reproduces the `y_true` of all 37 held-out folds of the reported run exactly
   (347/347); the gaze-only `engagement_labeling.py` agrees at chance (48.7 %), so
   the "five unit-weight gaze features" description written earlier in this
   revision was itself wrong and has been replaced. The label = frontal theta
   (+1.5), beta (+1.2), alpha (−1.5), theta/beta (+1.0), frontal asymmetry (+0.5),
   mean |x| (+1.0), mean |y| (+1.0), std x (+1.0), stability count (+1.0), 20-bin
   x-entropy (−1.0), min–max over pooled stimulus epochs, global median. Because
   EEG band power enters the label, no branch is "leakage-independent"; the text
   now says "gaze-free", discloses the rule (Sec. 2.4, Table S5) and adds a
   linear-recoverability audit (Table S15, run `audit`). The label has no spatial
   grid, so the label-stability block of Table S11 / Fig. S6 was removed.
   The dataset loader prefers `engagement_phase3d/`, so generating it (stage
   `data`) makes any machine's dataset identical to production.
3. **Ensemble size** — resolved in the text: the per-fold CSVs of the reported run
   and of every ablation record `n_ensemble = 5` (and `run_component_ablation.py`
   documents the production model as a 5-member ensemble), so Table 2, Sec. 2.5.10
   and the Code-availability statement now say 5 (previously 15; `settings.py`
   still defaults to 7 for `main.py`, which is why the revision runs go through
   `run_component_ablation.py`).
4. **Sec. 3.7 prevalence-matched threshold**: the reproducible value from the
   saved probabilities is balanced accuracy 0.767 / MCC 0.48 (transductive) and
   0.772 / 0.52 (causal, 5-epoch warm-up), not 0.778 / 0.50; the text now cites
   the reproducible values.
5. **Focal loss** — resolved in the text: the reported run and all ablations use
   focal γ = 3 with effective-number class weights (`alpha_strategy = effective_num`
   in every per-fold CSV); Table 2 now states this explicitly (the `FOCAL_GAMMA = 2`
   in `settings.py` is only the `main.py` default).
6. Reference 58 (hybrid adaptive EEG tokenization) is now complete: Huang B, Aziz MZ,
   Guo X, He X, Zheng J, Yu X (2026) Biomed Signal Process Control 126:110765,
   doi 10.1016/j.bspc.2026.110765.
