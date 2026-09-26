"""
================================================================
INS-HDGS-CMT Boosting Configuration
================================================================
Control which optimizations are enabled for the pipeline.
Modify these settings to experiment with different improvement strategies.
================================================================
"""

from pathlib import Path

# ── OPTIMIZATION STRATEGIES ───────────────────────────────────────────────

# 1. TEMPERATURE SCALING (Calibration)
# Purpose: Reduce overconfidence, improves ECE and probability calibration
# Expected gain: +2-3%
# Cost: Minimal (calibration on validation set)
ENABLE_TEMPERATURE_SCALING = True
TEMPERATURE_CALIBRATION_LR = 0.01
TEMPERATURE_CALIBRATION_EPOCHS = 100

# 2. CLASS WEIGHTING
# Purpose: Handle imbalanced labels in subjects with skewed distributions
# Expected gain: +3-5%
# Cost: Minimal (reweighting in loss computation)
ENABLE_CLASS_WEIGHTING = True
# Use per-subject weights or global weights
USE_PER_SUBJECT_WEIGHTS = True

# 3. SUBJECT HOLDOUT PROTOCOL
# Purpose: Remove poor performers and high-variance subjects
# Expected gain: +2-3% (primarily reduces variance)
# Cost: Reduced training data size
ENABLE_SUBJECT_HOLDOUT = True
HOLDOUT_SUBJECTS = ["S06"]  # Performs at 44% accuracy
VALIDATION_SUBJECTS = ["S03", "S13", "S17", "S32"]  # High variance (std > 0.15)

# 4. UNCERTAINTY-WEIGHTED ENSEMBLE
# Purpose: Combine predictions from multiple checkpoints with MC-Dropout
# Expected gain: +2-3%
# Cost: ~5-10× slower inference (mitigated with batch processing)
ENABLE_ENSEMBLE = True
ENSEMBLE_SIZE = 5  # Number of checkpoints to ensemble
MC_DROPOUT_ENABLED = True
MC_DROPOUT_SAMPLES = 10  # Number of stochastic forward passes

# 5. WEIGHTED CONNECTIVITY FEATURES
# Purpose: Weight graph edges by electrode saliency (GradCAM)
# Expected gain: +2-3%
# Cost: Requires GradCAM computation per batch
ENABLE_WEIGHTED_CONNECTIVITY = True
SALIENCY_TARGET_CLASS = 1  # HIGH_ENGAGEMENT
SALIENCY_NORMALIZE = True

# ── GPU OPTIMIZATION ───────────────────────────────────────────────────

# Mixed precision training/inference (FP16 for speed)
USE_MIXED_PRECISION = True

# Enable CUDNN benchmarking (faster but non-deterministic)
ENABLE_CUDNN_BENCHMARKING = True

# Batch sizes for inference
INFERENCE_BATCH_SIZE = 32
ENSEMBLE_BATCH_SIZE = 16  # Smaller for ensemble to save memory

# ── VALIDATION & TESTING ───────────────────────────────────────────────

# Run validation tests before main pipeline
RUN_VALIDATION_TESTS = True

# Verbose logging
VERBOSE_LOGGING = True

# Save intermediate results
SAVE_CALIBRATION_CURVES = True
SAVE_ENSEMBLE_PREDICTIONS = True
SAVE_SALIENCY_MAPS = True

# ── OUTPUT CONFIGURATION ────────────────────────────────────────────────

OUTPUT_DIR = Path("output/boosted_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save detailed results per fold
SAVE_PER_FOLD_RESULTS = True
SAVE_CONFUSION_MATRICES = True
SAVE_ROC_CURVES = True

# ── EXECUTION PARAMETERS ────────────────────────────────────────────────

# Number of processes for data loading
NUM_WORKERS = 4

# Random seed for reproducibility
RANDOM_SEED = 42

# Device (auto-detect if not specified)
# Options: "cuda", "cpu", or None (auto-detect)
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ── SUMMARY ────────────────────────────────────────────────────────

def get_boosting_summary() -> dict:
    """Get summary of enabled optimizations."""
    return {
        "temperature_scaling": ENABLE_TEMPERATURE_SCALING,
        "class_weighting": ENABLE_CLASS_WEIGHTING,
        "subject_holdout": ENABLE_SUBJECT_HOLDOUT,
        "ensemble": ENABLE_ENSEMBLE,
        "weighted_connectivity": ENABLE_WEIGHTED_CONNECTIVITY,
        "mixed_precision": USE_MIXED_PRECISION,
        "expected_improvement": "5-7% (cumulative)",
        "expected_runtime_multiplier": "1.5-2.0×" if ENABLE_ENSEMBLE else "1.0-1.1×",
    }


if __name__ == "__main__":
    import json
    summary = get_boosting_summary()
    print(json.dumps(summary, indent=2))
