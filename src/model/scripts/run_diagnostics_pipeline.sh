#!/bin/bash
#
# ================================================================
# HARD-SUBJECT DIAGNOSTIC FRAMEWORK — END-TO-END RUNNER
# ================================================================
#
# This script runs the complete hard-subject diagnostic pipeline
# for INS-HDGS-CMT LOSOCV.
#
# Prerequisites:
#   1. LOSOCV fold probabilities generated (output/fold_probs/)
#   2. Model checkpoints available (output/checkpoints/ins_hdgs_cmt_v17/)
#   3. Dataset accessible (data/ directory)
#
# Usage:
#   bash run_diagnostics_pipeline.sh [OPTIONS]
#
# Options:
#   --skip-embeddings        Skip expensive embedding extraction
#   --quick                  Skip embeddings and large visualizations
#   --debug                  Verbose output
#
# ================================================================

set -e  # Exit on error

# This script lives in scripts/; the model code (main.py, output/, etc.)
# is one directory up.
cd "$(dirname "$0")/.."

# Initialize conda (ensure environment is set up)
if [ -f "$HOME/miniconda_311/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda_311/etc/profile.d/conda.sh"
    conda activate base
fi

# Parse arguments
SKIP_EMBEDDINGS=false
QUICK_MODE=false
DEBUG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-embeddings)
            SKIP_EMBEDDINGS=true
            shift
            ;;
        --quick)
            SKIP_EMBEDDINGS=true
            QUICK_MODE=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   HARD-SUBJECT DIAGNOSTIC FRAMEWORK                            ║"
echo "║   INS-HDGS-CMT LOSOCV Pipeline                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Hard Subjects: S21, S03, S13, S35, S36"
echo "  Skip Embeddings: $SKIP_EMBEDDINGS"
echo "  Quick Mode: $QUICK_MODE"
echo "  Debug: $DEBUG"
echo ""

# ─────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────

echo -e "${BLUE}[0/2] PRE-FLIGHT CHECKS${NC}"

CHECKS_PASSED=true

# Check fold probs
if [ ! -d "output/fold_probs" ]; then
    echo -e "${RED}✗ Missing: output/fold_probs/${NC}"
    CHECKS_PASSED=false
else
    FOLD_COUNT=$(ls output/fold_probs/fold*.npz 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found fold probabilities: $FOLD_COUNT files${NC}"
fi

# Check checkpoints
if [ ! -d "output/checkpoints/ins_hdgs_cmt_v17" ]; then
    echo -e "${YELLOW}⚠ Checkpoints may be needed for embedding extraction${NC}"
else
    CKPT_COUNT=$(ls output/checkpoints/ins_hdgs_cmt_v17/*.pt 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found model checkpoints: $CKPT_COUNT files${NC}"
fi

# Check dataset
if [ ! -d "data" ]; then
    echo -e "${YELLOW}⚠ data/ directory not found (needed for embeddings)${NC}"
else
    echo -e "${GREEN}✓ Dataset directory accessible${NC}"
fi

if [ "$CHECKS_PASSED" = false ]; then
    echo -e "${RED}✗ Pre-flight checks FAILED${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All checks passed${NC}"
echo ""

# ─────────────────────────────────────────────────────────────
# PHASE 1: INDIVIDUAL SUBJECT DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

echo -e "${BLUE}[1/2] INDIVIDUAL SUBJECT DIAGNOSTICS${NC}"

OUTPUT_DIR="output/diagnostics"
mkdir -p "$OUTPUT_DIR"

CMD="python hard_subject_diagnostics.py \
    --label ins_hdgs_cmt_v17 \
    --hard-subjects S21,S03,S13,S35,S36 \
    --output-dir $OUTPUT_DIR"

if [ "$SKIP_EMBEDDINGS" = true ]; then
    CMD="$CMD --skip-embeddings"
fi

if [ "$DEBUG" = true ]; then
    echo -e "${YELLOW}Running:${NC} $CMD"
    echo ""
fi

echo "Processing hard subjects..."
echo "  • S21 (Subject 19 / Fold 19)"
echo "  • S03 (Subject 3 / Fold 3)"
echo "  • S13 (Subject 11 / Fold 11)"
echo "  • S35 (Subject 33 / Fold 33)"
echo "  • S36 (Subject 34 / Fold 34)"
echo ""

START_TIME=$(date +%s)

if eval $CMD; then
    ELAPSED=$(($(date +%s) - START_TIME))
    echo -e "${GREEN}✓ Phase 1 complete (${ELAPSED}s)${NC}"
    echo -e "   Output: $OUTPUT_DIR/{S21,S03,S13,S35,S36}/"
else
    echo -e "${RED}✗ Phase 1 FAILED${NC}"
    exit 1
fi

echo ""

# ─────────────────────────────────────────────────────────────
# PHASE 2: SUMMARY & AGGREGATION
# ─────────────────────────────────────────────────────────────

echo -e "${BLUE}[2/2] SUMMARY & AGGREGATION${NC}"

SUMMARY_FILE="output/hard_subjects_summary.txt"

CMD="python hard_subject_summary.py \
    --diagnostics-dir $OUTPUT_DIR \
    --output $SUMMARY_FILE"

if [ "$DEBUG" = true ]; then
    echo -e "${YELLOW}Running:${NC} $CMD"
    echo ""
fi

echo "Generating aggregate reports..."

START_TIME=$(date +%s)

if eval $CMD; then
    ELAPSED=$(($(date +%s) - START_TIME))
    echo -e "${GREEN}✓ Phase 2 complete (${ELAPSED}s)${NC}"
    echo -e "   Summary: $SUMMARY_FILE"
    echo -e "   Plots: output/comparison_*.png"
else
    echo -e "${RED}✗ Phase 2 FAILED${NC}"
    exit 1
fi

echo ""

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ DIAGNOSTIC FRAMEWORK COMPLETE${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo "Generated Outputs:"
echo ""

echo -e "${YELLOW}Per-Subject Diagnostics:${NC}"
for subj in S21 S03 S13 S35 S36; do
    if [ -d "$OUTPUT_DIR/$subj" ]; then
        echo "  $subj:"
        echo "    ✓ probability_histogram.png"
        echo "    ✓ roc_curve.png"
        echo "    ✓ pr_curve.png"
        echo "    ✓ threshold_sweep.png + best_threshold.json"
        echo "    ✓ confusion_matrix.png"
        echo "    ✓ calibration.png"
        if [ "$SKIP_EMBEDDINGS" = false ]; then
            echo "    ✓ tsne_label.png, umap_label.png"
        fi
        echo "    ✓ subject_shift_report.json"
        echo "    ✓ ranking_analysis.csv"
        echo "    ✓ diagnosis.txt & diagnosis.json"
        echo ""
    fi
done

echo -e "${YELLOW}Aggregate Reports:${NC}"
echo "  ✓ hard_subjects_summary.txt"
echo "    - Executive summary"
echo "    - Per-subject metrics table"
echo "    - Failure mode distribution"
echo "    - Prioritized recommendations"
echo "    - Detailed subject profiles"
echo ""

echo "  ✓ comparison_auc.png"
echo "    - Hard subject AUC comparison"
echo ""

echo "  ✓ comparison_metrics.png"
echo "    - Multi-metric comparison (2×2 grid)"
echo ""

# ─────────────────────────────────────────────────────────────
# NEXT STEPS
# ─────────────────────────────────────────────────────────────

echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Review Summary Report:"
echo "   cat $SUMMARY_FILE | less"
echo ""
echo "2. View Individual Diagnoses:"
echo "   cat output/diagnostics/S21/diagnosis.txt"
echo ""
echo "3. Check Failure Mode Distribution:"
echo "   grep -E 'A\)|B\)|C\)|D\)|E\)' $SUMMARY_FILE"
echo ""
echo "4. Based on Diagnoses, Implement Fixes:"
echo ""
echo "   Mode A (Threshold Issue):"
echo "     → Apply best_threshold.json to inference pipeline"
echo ""
echo "   Mode B (Calibration Issue):"
echo "     → Apply temperature scaling (see calibration.png)"
echo ""
echo "   Mode C (Subject Distribution Shift):"
echo "     → Run subject-specific fine-tuning"
echo ""
echo "   Mode D (Label Noise):"
echo "     → Manually review ground truth labels"
echo ""
echo "   Mode E (Representation Failure):"
echo "     → Debug EEG/ET preprocessing pipeline"
echo ""

echo -e "${YELLOW}Visualization:${NC}"
echo "  • Open output/diagnostics/<subject>/probability_histogram.png"
echo "  • Open output/diagnostics/<subject>/roc_curve.png"
echo "  • Open output/comparison_metrics.png"
echo ""

echo -e "${YELLOW}Questions?${NC}"
echo "  See HARD_SUBJECT_DIAGNOSTICS_README.md for detailed guide"
echo ""

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
