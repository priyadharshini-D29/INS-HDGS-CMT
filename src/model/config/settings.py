"""
================================================================
INS-HDGS-CMT — Phase 8 Configuration
Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking
Cognitive Multimodal Transformer
================================================================
Cognitive State Decoding for Neuromarketing Intelligence.

PRIMARY TASK:
  HIGH_ENGAGEMENT (1) vs LOW_ENGAGEMENT (0)

SECONDARY TASKS (optional heads):
  Purchase Intent · Visual Attention · Consumer Preference
  Cognitive Saliency

Replaces the previous ImagePage_1→6 classification objective,
which was weakly separable in cross-subject EEG.
================================================================
"""

import os
from pathlib import Path

import torch

# ── Data Sources ────────────────────────────────────────────────────────────
# src/model/ and src/data_pipeline/ are siblings under src/, matching the
# old NEUMA_PHASE8/NEUMA_PHASE3/NEUMA_PHASE6 sibling-under-repo-root layout,
# so these remain simple "../<sibling>" references -- just renamed.
# Resolved relative to this file (not the working directory) so that analysis
# scripts under scripts/ can be launched from the repository root as well.
_SRC_DIR     = Path(__file__).resolve().parents[2]          # <repo>/src
PHASE3_DIR   = _SRC_DIR / "data_pipeline" / "04_segmentation" / "output"
PHASE6_DIR   = _SRC_DIR / "data_pipeline" / "06_dataset_aggregation" / "output" / "global"
# S04 and S11 have no per-subject engagement labels — they fall back to the
# shared pooled directory, producing cross-subject duplicate epochs (leakage).
# S16, S31, S33, S41, S44 have single-class labels under the global threshold;
# LOSOCV skips them as test folds but retains them as training subjects.
# Full pool: 44 total → 42 valid (minus S04, S11) → 37 valid test folds.
SUBJECT_IDS  = [f"S{i:02d}" for i in range(1, 45) if i not in (4, 11)]
SUBJECT_ID   = os.environ.get("NEUMA_SUBJECT", "S01")

# ── EEG Signal Parameters ───────────────────────────────────────────────────
EEG_SR           = 300
EEG_EPOCH_LEN    = 5.0
EEG_SAMPLES      = int(EEG_SR * EEG_EPOCH_LEN)   # 1500

# Overridable via env var for window-length sensitivity checks (Table 2 /
# Sec 2.5.2 reviewer request): NEUMA_N_WINDOWS=5 -> 1.0s windows,
# NEUMA_N_WINDOWS=2 -> 2.5s windows (default 10 -> 0.5s windows).
N_WINDOWS        = int(os.environ.get("NEUMA_N_WINDOWS", "10"))
WINDOW_SAMPLES   = EEG_SAMPLES // N_WINDOWS        # 150 samples ≈ 500 ms at default

EPOCH_OVERLAP    = 0.75   # 75% overlap between successive epochs → ~4× more epochs

EEG_BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),   # frontal theta → engagement marker
    "alpha": (8.0, 13.0),   # alpha suppression → attention
    "beta":  (13.0, 30.0),  # beta → active engagement
    "gamma": (30.0, 45.0),
}
N_BAND_FEATURES  = len(EEG_BANDS)    # 5 node features per electrode

# ── Eye-Tracking Parameters ─────────────────────────────────────────────────
ET_SR            = 120
ET_EPOCH_LEN     = 5.0
ET_SAMPLES       = int(ET_SR * ET_EPOCH_LEN)      # 600
ET_INPUT_DIM     = 3                               # gaze_x, gaze_y, pupil

# ── ROI Parameters ──────────────────────────────────────────────────────────
# The ROI saliency vector is a GRID_ROWS x GRID_COLS occupancy histogram of
# gaze over the stimulus. Overridable via env vars for the grid-resolution
# sensitivity analysis (Reviewer 2, comment 9): e.g. NEUMA_GRID_COLS=6
# NEUMA_GRID_ROWS=4 reproduces the 24-cell product layout of Fig. 1.
IMAGE_W          = 3000
IMAGE_H          = 1688
GRID_COLS        = int(os.environ.get("NEUMA_GRID_COLS", "5"))
GRID_ROWS        = int(os.environ.get("NEUMA_GRID_ROWS", "2"))
N_ROIS           = int(os.environ.get("NEUMA_N_ROIS", str(GRID_COLS * GRID_ROWS)))

# ── Graph Construction ──────────────────────────────────────────────────────
# Overridable via env var for connectivity-threshold / PLV-band sensitivity
# sweeps (see sensitivity_sweep.py) without touching code.
CONN_THRESHOLD   = float(os.environ.get("NEUMA_CONN_THRESHOLD", "0.30"))
CONN_METHOD      = os.environ.get("NEUMA_CONN_METHOD", "pearson")
# "pearson" | "coherence" | "plv" (broadband) | "plv_<band>" for
# band in {delta, theta, alpha, beta, gamma} (narrowband PLV, computed on
# the signal after band-pass filtering rather than raw 1-45 Hz broadband).
MIN_EDGES        = 3

# ── Engagement Labeling Thresholds ──────────────────────────────────────────
# All thresholds are computed per-dataset (median-based) unless overridden.
# These defaults activate only if auto-thresholding fails.
ENGAGEMENT_FIXATION_THRESH   = 0.30   # fraction of epoch with fixations
ENGAGEMENT_DWELL_THRESH      = 0.40   # normalised dwell-time in high-ROI regions
ENGAGEMENT_PUPIL_THRESH      = 0.50   # normalised pupil dilation (0-1)
ENGAGEMENT_GAZE_ENTROPY_MAX  = 2.50   # high entropy → scattered → low engagement
ENGAGEMENT_REVISIT_THRESH    = 1      # ROI revisit count threshold
ENGAGEMENT_BALANCE_RATIO     = 0.40   # min fraction of minority class (0.40 = 40/60 max)
# Minimum minority-class fraction required for a test fold to be evaluated.
# Subjects below this (e.g. 80/20 split) have undefined or degenerate AUC.
IMBALANCE_SKIP_RATIO         = 0.00   # 0.0 = evaluate all folds (v3 behaviour)

# ── Model Architecture ──────────────────────────────────────────────────────
EMBED_DIM        = 128                # universal embedding dimension

# GAT — Layer 1 (multi-head concat)
GAT_L1_HEAD_DIM  = 32
GAT_L1_HEADS     = 4                 # 4 × 32 = 128
# GAT — Layer 2 (single-head output)
GAT_L2_DIM       = EMBED_DIM
GAT_L2_HEADS     = 1
GAT_DROPOUT      = 0.20
GAT_ALPHA        = 0.20

# Temporal Transformer
T_NHEAD          = 4
T_LAYERS         = 3
T_FF_DIM         = 512
T_DROPOUT        = 0.10
T_MAX_LEN        = 200

# Spiking Neural Network (LIF Neurons)
SNN_TIME_STEPS   = 10                 # simulation time steps
SNN_THRESHOLD    = 1.0                # LIF firing threshold
SNN_DECAY        = 0.90              # membrane potential decay
SNN_HIDDEN_DIM   = 128

# ET Encoder
ET_LSTM_HIDDEN   = 64
ET_LSTM_LAYERS   = 2
ET_DROPOUT       = 0.20

# ROI Attention
ROI_HIDDEN_DIM   = 64

# Cross-Modal Fusion
FUSION_HEADS     = 4

# Neuro-Symbolic Rule Layer
NS_N_RULES       = 8                  # number of differentiable rules
NS_HIDDEN_DIM    = 64                 # rule premise embedding size
NS_TEMPERATURE   = 1.0               # rule softmax temperature
# Bypass-gate mode of Eq. (18): "learned" (paper default) | "rule_only"
# (alpha fixed at 0, prediction = soft-rule evidence only) | "bypass_only".
# Env-overridable so the rule-only LOSOCV run (Reviewer 2, comment 5) needs
# no code change:  NEUMA_NS_ALPHA_MODE=rule_only python main.py --label ns_rule_only
NS_ALPHA_MODE    = os.environ.get("NEUMA_NS_ALPHA_MODE", "learned")

# Classification heads
CLS_HIDDEN       = 64
N_CLASSES        = 2                  # HIGH_ENGAGEMENT (1) vs LOW_ENGAGEMENT (0)

# Secondary task heads (optional, off by default)
N_PURCHASE_CLASSES    = 2             # BUY / NOT_BUY
N_ATTENTION_LEVELS    = 3             # LOW / MEDIUM / HIGH visual attention
N_PREFERENCE_CLASSES  = 3             # NEGATIVE / NEUTRAL / POSITIVE

# ── Training ────────────────────────────────────────────────────────────────
# Fold-parallel: batch_size is capped to len(train_ds) inside run_fold_on_device,
# so using 32 gives ~9 batches/epoch for the typical ~300-sample training fold.
BATCH_SIZE       = 32
EPOCHS           = 250
LR               = 5e-4
WEIGHT_DECAY     = 0
PATIENCE         = 50    # v3 best-known-good (commit 6cdf306, accuracy=0.8063) — was 100
DROPOUT          = 0.25

# Focal loss parameters (engagement class imbalance handling)
# focal_alpha=1.0 means no global downscaling (the original 0.25 compressed
# cls loss to 1/4 of its natural magnitude, letting contrastive dominate).
FOCAL_ALPHA      = 1.0
FOCAL_GAMMA      = 2.0               # focusing exponent (hard-example weight)

# Contrastive learning
TEMPERATURE      = 0.07

# Multi-task loss weights
# At random init, contrastive ≈ log(batch_size) ≈ 3.5 and cls ≈ 0.17.
# LAMBDA_CLS=5 / LAMBDA_CONTRAST=0.05 → cls≈0.85 vs contrast≈0.18.
# Classification must dominate early to avoid majority-class collapse.
LAMBDA_CLS          = 5.00
LAMBDA_CONTRAST     = 0.05
LAMBDA_ROI          = 0.05
LAMBDA_CONNECTIVITY = 0.02
LAMBDA_MMD          = 0.10
LAMBDA_SECONDARY    = 0.15           # weight for secondary task heads

# ── Hardware ────────────────────────────────────────────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()

USE_DP           = True
FOLD_PARALLEL    = NUM_GPUS > 1   # True when multiple GPUs — runs folds concurrently
USE_DDP          = False

# Mixed precision
USE_AMP          = True
GRAD_ACCUM_STEPS = 1   # no accumulation — train_ds is already small (~300 samples)

# ── Ensemble ─────────────────────────────────────────────────────────────────
# Models trained per fold (different seeds); probabilities averaged before
# thresholding.  Set to 1 to disable (identical speed to a single model).
N_ENSEMBLE       = 7    # v3 best-known-good (commit 6cdf306, accuracy=0.8063) — was 15

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED      = 42

# ── Output Directories ──────────────────────────────────────────────────────
OUTPUT_DIR         = Path("output")
METRICS_DIR        = OUTPUT_DIR / "metrics"
PLOTS_DIR          = OUTPUT_DIR / "plots"
CKPT_DIR           = OUTPUT_DIR / "checkpoints"
EXPLAINABILITY_DIR = OUTPUT_DIR / "explainability"
ABLATION_DIR       = OUTPUT_DIR / "ablation"
BENCHMARK_DIR      = OUTPUT_DIR / "benchmark"
LABELS_DIR         = OUTPUT_DIR / "engagement_labels"

ALL_OUTPUT_DIRS = [
    METRICS_DIR, PLOTS_DIR, CKPT_DIR,
    EXPLAINABILITY_DIR, ABLATION_DIR, BENCHMARK_DIR, LABELS_DIR,
]

# ── Engagement Class Names ──────────────────────────────────────────────────
ENGAGEMENT_CLASS_NAMES = {0: "LOW_ENGAGEMENT", 1: "HIGH_ENGAGEMENT"}

# ── Performance Boost Flags (v9) ────────────────────────────────────────────
# Subject normalization: already applied as z-score in NeumaGraphDataset.__init__
USE_SUBJECT_NORM         = True

# Temporal consistency loss: penalise large prediction jumps within a mini-batch
USE_TEMPORAL_LOSS        = False
TEMPORAL_LOSS_WEIGHT     = 0.10

# Curriculum learning: train easy subjects before hard ones (sequential mode)
CURRICULUM_LEARNING      = False
CURRICULUM_HARD_SUBJECTS = ["S13", "S22", "S31", "S33"]

# Stubs for future model-level features (not yet wired into architecture)
USE_DYNAMIC_ADJACENCY    = False   # learnable A — requires DynamicGAT changes
USE_GRAPH_MASKING        = False   # per-channel mask  — requires model changes
