# Reviewer-revision runbook for the GPU server (NVIDIA Brev)

Everything below is copy-paste. Run the stages in order; each stage skips work
that is already done, so you can stop and resume. Nothing here edits the
manuscript: at the end, `verify` prints every number that replaces a red
`\todoval{}` marker in `paper/*.tex`.

Time estimates assume one 8×A100 node; on a single GPU multiply by ~6.

## 0. Instance and repository

Pick an instance with **8 GPUs (A100/H100)** if possible: the LOSOCV harness runs
one held-out subject per GPU. 80 GB per GPU is far more than needed; memory is
not the constraint, wall-clock is.

```bash
sudo apt-get update && sudo apt-get install -y git git-lfs
git lfs install
git clone https://github.com/priyadharshini-D29/INS-HDGS-CMT.git
cd INS-HDGS-CMT
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # driver >= 570; use cu124 on older drivers
pip install -r requirements.txt
nvidia-smi -L
```

## 1. Data (choose one)

**Option A — upload the preprocessed bundle (fast, 369 MB).** It was built on
the laptop from the raw NeuMa recordings and contains the epochs and the
production labels (`engagement_phase3d/`), which reproduce the held-out labels
of the published run exactly (347/347).

```bash
# on the laptop
scp D:/INS-HDGS-CMT/dist/neuma_preprocessed_bundle.tgz <user>@<brev-host>:~/INS-HDGS-CMT/
# on the server, from the repo root
tar xzf neuma_preprocessed_bundle.tgz && rm neuma_preprocessed_bundle.tgz
ls -d src/data_pipeline/04_segmentation/S*/output/engagement_phase3d | wc -l   # must print 42
```

**Option B — regenerate from the raw recordings (663 MB, ~15 min).** Upload the
42 `S*.xdf` files into `DataSource/` at the repository root and run:

```bash
bash scripts/revision/run_revision.sh data
```

Either way, confirm before training:

```bash
bash scripts/revision/run_revision.sh env
```

## 2. Stages, one reviewer point at a time

| Stage | Reviewer point | Command | Output that feeds the paper | ~Time (8 GPU) |
|---|---|---|---|---|
| `audit` | label construction (R1-1/3, R2 leakage remarks) | `bash scripts/revision/run_revision.sh audit` | `results/statistics/label_leakage_audit.md` → Abstract, Sec. 3.2, Table S15 | 1 min |
| `full` | fresh production run + checkpoints | `... run_revision.sh full` | `results/ablation/abl_full/`, `src/model/output/checkpoints/abl_full/` | 2–3 h |
| `eeg_only` | **R1-1, R1-3** gaze-free EEG branch | `... run_revision.sh eeg_only` | `results/ablation/abl_eeg_only/` → Abstract, Table 3 row 1, Table 7, Sec. 3.4b, Table S14 | 1–2 h |
| `rule_only` | **R2-5** gate closed, α ≡ 0 | `... run_revision.sh rule_only` | `results/ablation/abl_ns_rule_only/` → Table 7, Table S13 | 2–3 h |
| `baselines` | **R2-3** tuned baselines | `... run_revision.sh baselines` | `results/baselines/dl_tuned/` → Tables 3, 4, 5, 8, S6, S10 | 4–8 h |
| `tau` | **R2-2** τ downstream | `... run_revision.sh tau` | `src/model/output/metrics/sensitivity_threshold_summary.csv` → Table S7, Sec. 3.9 | 2–3 h |
| `grid` | **R2-9** ROI grid | `... run_revision.sh grid` | `src/model/output/metrics/grid_*/` → Table S11, Sec. 3.9 | 8–12 h |
| `lc` | **R2-6** learning curve | `... run_revision.sh lc` | `results/sensitivity/learning_curve.*` → Fig. S5, Sec. 3.9 | 3–4 h |
| `ckpt` | **R2-4, R2-5, R2-7** | `... run_revision.sh ckpt` | `results/statistics/rule_fidelity*`, `results/explainability/rule_grounding_*`, `results/statistics/snn_energy_measured.*` → Sec. 3.10.3, Table S9, Table S13, Fig. 8D | 30 min |
| `stats` | all paired statistics | `... run_revision.sh stats` | `results/statistics/*`, `results/sensitivity/*`, figures S4–S6 | 5 min |
| `verify` | re-verification | `... run_revision.sh verify` | printed table of every number for the `\todoval{}` markers | seconds |

Or simply:

```bash
nohup bash scripts/revision/run_revision.sh all > logs/revision/all.log 2>&1 &
tail -f logs/revision/all.log
```

Order matters only in two places: `ckpt` needs the checkpoints written by
`full`, and `stats` should run after every training stage you intend to report.
Run `stats` and `verify` again whenever a new stage finishes.

## 3. What "verified" means for each point

* **R1-1/2/3 (cross-modal contribution, significance, EEG without ET).**
  `cross_modal_contribution.md` gives full − EEG-only, full − (−gaze sequence),
  full − (−ROI), full − (−fusion) and full − ET-LSTM with paired Wilcoxon, Holm,
  Cliff's δ and bootstrap CIs over the same 37 folds. `verify` checks that the
  EEG-only CSV has 37 folds, every test fold has both classes, and prints ΔAUC.
* **R1-4 (why train EEG with ET).** Answered by the same table: if the EEG-only
  branch is within noise of the full model on balanced accuracy/MCC and the
  full model only improves ranking (AUC), the text in Sec. 3.4b stands as
  written; if EEG-only is far below, the sentence in the Abstract must be
  weakened — `verify` prints both numbers side by side.
* **Label audit (new, forced by the reviewers' leakage remarks).** The production
  label is the *multimodal* rule in `engagement_phase3d.py` (frontal EEG band
  power + gaze statistics), not a gaze-only composite. `audit` reports how well a
  linear probe recovers the label from the five EEG terms, the five gaze terms
  and all ten (Table S15). Any model result must be read against those floors.
* **R2-2 (τ).** Structural part already computed (Table S7 left, Fig. S4); `tau`
  adds 37-fold LOSOCV at six τ values for the EEG-only branch; `stats` merges both.
* **R2-3 (baseline tuning).** Each baseline: 12 configurations per fold, chosen on
  the validation subject's balanced accuracy, patience 30, ≤250 epochs;
  `hparams_<model>.csv` records the selected configuration of every fold.
* **R2-4 (SNN).** `ckpt` measures latency and board power (nvidia-smi sampling)
  of the trained spiking encoder against its dense twin on the GPU (Table S9).
* **R2-5 (α gate).** `ckpt` reports the learned α, rule/gated agreement, margin
  correlation and post-hoc α = 0 metrics on every fold; `rule_only` adds the model
  trained with the gate closed (paired test vs. full).
* **R2-6 (overfitting).** `lc` gives the subject-count learning curve (Fig. S5).
* **R2-7 (Fig. 8D).** `ckpt` produces electrode × band attribution maps per rule
  and grounded IF–THEN text (`results/explainability/`); replace panel D of
  `paper/figures/fig5_explainability.pdf` with `fig_rule_grounding_abl_full.pdf`.
* **R2-8 (thresholds).** Already computed from saved probabilities (Table S12);
  `stats` recomputes it.
* **R2-9 (grid).** The label contains no spatial grid (Table S5), so only the
  model side remains: `grid` retrains the full model on four grids (Table S11).

## 4. After the runs

1. `bash scripts/revision/run_revision.sh verify` — every line must be `OK`.
2. Copy the printed numbers into the `\todoval{}` markers of
   `paper/INS_HDGS_CMT_manuscript.tex` and `paper/INS_HDGS_CMT_supplementary.tex`
   (grep for `todoval` to find them), and replace Fig. 8D.
3. Rebuild: `cd paper && pdflatex INS_HDGS_CMT_manuscript.tex` (twice) and the
   supplementary; both must compile with zero errors.
4. Commit `results/`, `paper/*.tex`, `paper/figures/*.pdf`, and
   `docs/RESPONSE_TO_REVIEWERS.md` after filling its bracketed values.

## 5. If something fails

* `CUDA out of memory` in `baselines`: rerun the stage; it resumes per model.
  Reduce parallelism by exporting `NGPU=4` before the command.
* A LOSOCV stage stops midway: rerun the same stage; `run_losocv` resumes from
  the per-fold CSV it has written so far (`resume=` in `evaluation/losocv.py`).
* `AssertionError: ET_INPUT_DIM=…`: an `ET_*` environment variable is set in the
  shell; `unset ET_USE_BOTH_EYES ET_USE_VERGENCE ET_USE_SPEED ET_NORMALIZE`.
* `No Phase-3 epoch data`: the bundle was unpacked somewhere other than the
  repository root; `src/data_pipeline/04_segmentation/S01/output/epochs/` must exist.
