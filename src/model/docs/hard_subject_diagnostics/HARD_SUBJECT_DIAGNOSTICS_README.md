"""
================================================================
HARD-SUBJECT DIAGNOSTIC FRAMEWORK — QUICK START GUIDE
================================================================

This framework provides comprehensive diagnostics for hard subjects
in your INS-HDGS-CMT LOSOCV pipeline.

Hard Subjects Tracked:
  S21, S03, S13, S35, S36

Each subject receives 10 diagnostic analyses to identify failure modes.

================================================================
QUICK START
================================================================

1. Generate Individual Subject Diagnostics
   ──────────────────────────────────────────────────────────

   python hard_subject_diagnostics.py \\
       --label ins_hdgs_cmt_v17 \\
       --hard-subjects S21,S03,S13,S35,S36 \\
       --output-dir output/diagnostics

   Optional:
   - --skip-embeddings       (faster, skip t-SNE/UMAP)

   Output Structure:
   ├── output/diagnostics/
   │   ├── S21/
   │   │   ├── probability_histogram.png
   │   │   ├── roc_curve.png
   │   │   ├── pr_curve.png
   │   │   ├── threshold_sweep.png
   │   │   ├── best_threshold.json
   │   │   ├── confusion_matrix.png
   │   │   ├── calibration.png
   │   │   ├── tsne_label.png
   │   │   ├── umap_label.png
   │   │   ├── subject_shift_report.json
   │   │   ├── ranking_analysis.csv
   │   │   ├── diagnosis.txt
   │   │   └── diagnosis.json
   │   ├── S03/
   │   ├── S13/
   │   ├── S35/
   │   └── S36/

   Time: ~10-20 minutes per subject (with embeddings)
         ~2-5 minutes per subject (without embeddings)


2. Generate Aggregated Summary Report
   ──────────────────────────────────────────────────────────

   python hard_subject_summary.py \\
       --diagnostics-dir output/diagnostics \\
       --output output/hard_subjects_summary.txt

   Output Files:
   - hard_subjects_summary.txt          (detailed text report)
   - output/comparison_auc.png          (AUC comparison chart)
   - output/comparison_metrics.png      (multi-metric comparison)

   Time: ~1 minute


================================================================
WHAT EACH DIAGNOSTIC TELLS YOU
================================================================

[1/10] PROBABILITY HISTOGRAM
───────────────────────────
Purpose: Determine if classes are separable but threshold is wrong

Read it:
  ✓ If HIGH and LOW distributions don't overlap → classes are separable
  ✗ If distributions overlap significantly → embedding/representation issue
  ? If distributions are identical → model not learning

Action:
  - Non-overlapping → adjust threshold (see threshold_sweep.png)
  - Overlapping → review subject_shift_report.json for distribution anomalies


[2/10] ROC CURVE
────────────────
Purpose: Evaluate ranking quality regardless of threshold

Read it:
  ✓ AUC > 0.80 → good ranking quality
  ✓ AUC 0.70-0.80 → acceptable ranking
  ✗ AUC < 0.60 → poor ranking (not a threshold issue)

Action:
  - High AUC + poor accuracy → threshold adjustment needed
  - Low AUC → representation/feature extraction problem


[3/10] PRECISION-RECALL CURVE
──────────────────────────────
Purpose: Evaluate trade-off between precision and recall

Read it:
  ✓ AP (Area Under Curve) > 0.70 → good performance
  ✗ AP < 0.60 → poor precision-recall trade-off

Action:
  - High AP, low F1 → optimal threshold not at 0.5
  - Low AP → fundamental classification difficulty


[4/10] THRESHOLD SWEEP
──────────────────────
Purpose: Find optimal decision threshold (0.05 → 0.95)

Read it:
  - Blue line (Accuracy): overall correctness
  - Orange line (Balanced Accuracy): per-class correctness
  - Green line (F1 Score): harmonic mean precision/recall
  - Red line (MCC): correlation coefficient

Action:
  - If current T=0.5 is NOT at peak → adjust to best_threshold.json
  - If best threshold = 0.5 → threshold is correct, problem elsewhere


[5/10] CONFUSION MATRIX
──────────────────────
Purpose: Identify systematic classification patterns

Read it:
  - Top-left (True Neg): correctly predicted LOW
  - Top-right (False Pos): wrongly predicted HIGH
  - Bottom-left (False Neg): wrongly predicted LOW
  - Bottom-right (True Pos): correctly predicted HIGH

Action:
  - If one cell dominates → severe class imbalance or threshold issue
  - If all cells filled → threshold or calibration issue


[6/10] CALIBRATION ANALYSIS
────────────────────────────
Purpose: Check if predicted probabilities match actual frequencies

Read it:
  - Dots near diagonal → well-calibrated
  - Dots above diagonal → overconfident (predicts too high)
  - Dots below diagonal → underconfident (predicts too low)
  - ECE (Expected Calibration Error):
    • < 0.05  → excellent calibration
    • 0.05-0.15 → acceptable calibration
    • > 0.15  → poor calibration

Action:
  - High ECE + reasonable ROC → apply temperature scaling
  - High ECE + poor ROC → representation issue


[7/10] EMBEDDING VISUALIZATION
───────────────────────────────
Purpose: Check if embeddings separate by engagement or by subject

Read it:
  - tsne_label.png: colored by TRUE engagement (should separate)
  - umap_label.png: UMAP version of above

  ✓ RED and BLUE clusters separate → model learned engagement
  ✗ RED and BLUE clusters mixed → subject distribution shift
  ✗ Clusters by subject ID instead → subject-specific bias

Action:
  - Good separation → threshold/calibration issue (easy fix)
  - Poor separation → subject-specific fine-tuning needed


[8/10] SUBJECT SHIFT REPORT
────────────────────────────
Purpose: Detect if hard subject is distribution outlier

Read it:
  - Mahalanobis Distance: measure of dissimilarity from training subjects
    • Mean > 2σ → subject is outlier
    • Mean < σ → subject is similar to training set

  - Cosine Distance: angle between feature vectors
    • High distance → different feature patterns

Action:
  - is_outlier=true → consider subject-specific training
  - is_outlier=false → problem not distribution shift


[9/10] RANKING FAILURE ANALYSIS
────────────────────────────────
Purpose: Check if model ranking is correct

Read it:
  - CSV sorted by predicted_prob (descending)
  - For each row: did model assign correct probability?

  ✓ HIGH samples have high prob, LOW samples have low prob → ranking OK
  ✗ Probabilities scattered randomly → ranking broken

Action:
  - Good ranking but wrong threshold → adjust threshold
  - Bad ranking → model learning problem


[10/10] AUTOMATIC DIAGNOSIS
────────────────────────────
Purpose: Automated failure mode classification

Failure Modes:

  A) THRESHOLD ISSUE
     Signature: AUC > 0.70 AND |best_T - 0.5| > 0.10
     Action: Adjust threshold from 0.5 to best_T
     Priority: EASY FIX ✓

  B) CALIBRATION ISSUE
     Signature: AUC > 0.65 AND ECE > 0.15
     Action: Apply temperature scaling or Platt scaling
     Priority: EASY FIX ✓

  C) SUBJECT DISTRIBUTION SHIFT
     Signature: AUC < 0.60 AND F1 < 0.40
     Action: Subject-specific fine-tuning
     Priority: MEDIUM ⚠

  D) LABEL NOISE
     Signature: 0.50 < AUC < 0.60 AND F1 > 0.30
     Action: Manually review ground truth labels
     Priority: INVESTIGATION REQUIRED

  E) REPRESENTATION FAILURE
     Signature: AUC < 0.50 AND Balanced_Acc < 0.50
     Action: Investigate preprocessing, artifacts, labels
     Priority: HARD ✗

Read diagnosis.txt for specific recommendations.

================================================================
PRIORITY RANKING
================================================================

Recommended action order:

Priority 1: THRESHOLD ISSUE (Mode A)
  → Easiest fix
  → Adjust threshold in inference pipeline

Priority 2: CALIBRATION ISSUE (Mode B)
  → Easy fix
  → Apply post-hoc calibration

Priority 3: SUBJECT-SPECIFIC TRAINING (Mode C)
  → Medium effort
  → Fine-tune on subject-specific data

Priority 4: LABEL REVIEW (Mode D)
  → Investigation required
  → Manual review and correction

Priority 5: DEEP INVESTIGATION (Mode E)
  → Hardest fix
  → Debug preprocessing pipeline

================================================================
INTERPRETING THE SUMMARY REPORT
================================================================

hard_subjects_summary.txt contains:

1. EXECUTIVE SUMMARY
   - Overall mean/std for all metrics
   - Performance ranges across hard subjects

2. PER-SUBJECT TABLE
   - All metrics side-by-side
   - Easy identification of outliers

3. FAILURE MODE DISTRIBUTION
   - How many subjects per failure mode
   - Identifies systematic issues

4. RECOMMENDATIONS BY PRIORITY
   - Ranked by confidence score
   - Specific action for each subject

5. DETAILED PROFILES
   - Full diagnostic information per subject
   - Complete diagnosis text

================================================================
COMMON PATTERNS & INTERPRETATION
================================================================

Pattern 1: Multiple subjects with Mode A (Threshold Issue)
  → Suggests global threshold optimization could help all subjects
  → Consider learnable/adaptive thresholds

Pattern 2: Multiple subjects with Mode C (Distribution Shift)
  → Suggests dataset has high subject variability
  → Consider: domain adaptation, subject embeddings

Pattern 3: All subjects have good AUC but poor F1
  → Global threshold optimization problem
  → Likely fixable without retraining

Pattern 4: All subjects have poor AUC
  → Fundamental model issue
  → Review: architecture, preprocessing, features

Pattern 5: Mix of Mode A and B
  → Some subjects need threshold, others need calibration
  → Apply both: threshold optimization + calibration

================================================================
NEXT STEPS AFTER DIAGNOSTICS
================================================================

Based on findings, recommended actions:

If Mode A (Threshold Issue) dominant:
  1. Apply best_threshold.json values to inference
  2. Re-evaluate on test set
  3. Validate across all hard subjects

If Mode B (Calibration Issue) dominant:
  1. Compute temperature scaling factor
  2. Apply to model outputs
  3. Verify ECE improvement

If Mode C (Distribution Shift) dominant:
  1. Collect subject-specific data
  2. Fine-tune model on subject data
  3. Compare to baseline

If Mode D (Label Noise) suspected:
  1. Manually review samples marked as errors
  2. Correct labels or remove samples
  3. Retrain model

If Mode E (Representation Failure) suspected:
  1. Review EEG/ET preprocessing steps
  2. Check for artifact contamination
  3. Verify label generation logic

================================================================
FILE GLOSSARY
================================================================

probability_histogram.png
  - Class-wise probability distributions
  - Use: identify if classes are separable

roc_curve.png
  - ROC curve with operating point marked
  - Use: evaluate ranking quality

pr_curve.png
  - Precision-recall curve with AP score
  - Use: evaluate precision/recall trade-off

threshold_sweep.png
  - Metrics vs threshold (0.05 → 0.95)
  - Use: identify optimal threshold

best_threshold.json
  - Recommended threshold value
  - Use: configure inference pipeline

confusion_matrix.png
  - Raw counts and percentages
  - Use: identify systematic errors

calibration.png
  - Reliability diagram with ECE
  - Use: assess probability calibration

tsne_label.png
  - t-SNE colored by engagement label
  - Use: check label separability in latent space

umap_label.png
  - UMAP colored by engagement label
  - Use: alternative embedding visualization

subject_shift_report.json
  - Mahalanobis/Cosine distance to training set
  - Use: detect distribution outliers

ranking_analysis.csv
  - Samples sorted by predicted probability
  - Use: inspect individual predictions

diagnosis.txt / diagnosis.json
  - Automatic failure mode classification
  - Use: identify improvement strategy

================================================================
QUESTIONS & TROUBLESHOOTING
================================================================

Q: Embeddings plot is empty / failed?
A: Try with --skip-embeddings first to get other diagnostics.
   Embedding extraction requires model inference, which may timeout.

Q: No best_threshold.json found?
A: It's always generated. Check permissions and disk space.

Q: Threshold sweep shows flat line?
A: May indicate only one class prediction (probability ~0 or ~1).
   Check probability_histogram.png and confusion_matrix.png.

Q: Why are diagnosis recommendations conflicting?
A: Subjects may have mixed failure modes (Mode A+B, Mode C+D, etc).
   Follow confidence_score ranking for prioritization.

Q: All metrics look good but model still fails in production?
A: Check for data distribution shift in production vs. training.
   Verify label consistency and preprocessing pipeline.

================================================================
"""

import sys

if __name__ == "__main__":
    print(__doc__)
