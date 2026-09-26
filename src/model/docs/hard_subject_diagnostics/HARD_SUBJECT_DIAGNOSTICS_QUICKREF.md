================================================================================
HARD-SUBJECT DIAGNOSTIC FRAMEWORK — QUICK REFERENCE & CHECKLIST
================================================================================

This document provides a quick reference for running and interpreting the
hard-subject diagnostic framework.

================================================================================
✓ IMPLEMENTATION CHECKLIST
================================================================================

[✓] hard_subject_diagnostics.py (Main diagnostic module)
    ├─ 10 diagnostic functions
    ├─ Automatic failure mode detection
    ├─ JSON + CSV output
    └─ CLI entry point

[✓] hard_subject_summary.py (Aggregation module)
    ├─ Load all subject diagnostics
    ├─ Generate summary report (TXT)
    ├─ Create comparison visualizations
    └─ CLI entry point

[✓] scripts/run_diagnostics_pipeline.sh (Automation script)
    ├─ Pre-flight checks
    ├─ Phase 1: Individual diagnostics
    ├─ Phase 2: Summary aggregation
    ├─ Progress tracking
    ├─ Next-steps guidance
    └─ Executable (chmod +x)

[✓] HARD_SUBJECT_DIAGNOSTICS_README.md (User guide)
    ├─ Quick start
    ├─ 10 diagnostic interpretations
    ├─ Common patterns
    ├─ Troubleshooting
    └─ File glossary

[✓] HARD_SUBJECT_DIAGNOSTICS_IMPLEMENTATION.txt (Detailed reference)
    ├─ Complete documentation
    ├─ Step-by-step workflow
    ├─ Pattern recognition
    ├─ Metric reference
    └─ Troubleshooting

================================================================================
⚡ QUICK START (2 Commands)
================================================================================

Command 1: Run All Diagnostics
──────────────────────────────

  $ cd /home/nvidia/24PHD1314/Neuma_Model/NEUMA_PHASE8/
  $ bash scripts/run_diagnostics_pipeline.sh

Expected output:
  ✓ Diagnostics for 5 hard subjects (S21, S03, S13, S35, S36)
  ✓ Summary report (hard_subjects_summary.txt)
  ✓ Comparison plots (comparison_auc.png, comparison_metrics.png)

Time: ~30-45 minutes (with embeddings) or ~15-20 minutes (--skip-embeddings)


Command 2: View Results
───────────────────────

  $ cat output/hard_subjects_summary.txt
  $ ls -la output/diagnostics/S21/
  $ cat output/diagnostics/S21/diagnosis.txt

================================================================================
🎯 DIAGNOSTIC OUTPUTS (Per Subject)
================================================================================

Location: output/diagnostics/<SUBJECT>/

┌─ Probability Distribution Analysis ──────────────────────────────────┐
│ probability_histogram.png                                            │
│ → Visualizes predicted probabilities by true class                   │
│ → Interpretation: Are classes linearly separable?                    │
│ → Action: If separated → threshold adjustment issue                  │
│          If overlapped → representation issue                        │
└──────────────────────────────────────────────────────────────────────┘

┌─ Ranking Quality Analysis ───────────────────────────────────────────┐
│ roc_curve.png              [Ranking quality regardless of threshold]  │
│ pr_curve.png               [Precision-recall trade-off]              │
│ → Metrics: AUC, AP scores                                             │
│ → Interpretation: Is ranking fundamentally good?                     │
│ → Action: High AUC → threshold/calibration issue                     │
│          Low AUC → representation issue                              │
└──────────────────────────────────────────────────────────────────────┘

┌─ Threshold Optimization ─────────────────────────────────────────────┐
│ threshold_sweep.png        [Metrics vs threshold, 0.05-0.95]         │
│ best_threshold.json        [Recommended threshold value]             │
│ → Shows: Accuracy, Balanced Accuracy, F1, MCC at each threshold      │
│ → Interpretation: Where is the optimal decision boundary?            │
│ → Action: If best_T ≠ 0.5 → adjust inference threshold               │
│          If best_T ≈ 0.5 → threshold not the issue                   │
└──────────────────────────────────────────────────────────────────────┘

┌─ Classification Pattern Analysis ────────────────────────────────────┐
│ confusion_matrix.png       [Raw counts + normalized percentages]     │
│ → Shows: True Neg, False Pos, False Neg, True Pos                    │
│ → Interpretation: Which errors dominate?                             │
│ → Action: Identify systematic bias patterns                          │
└──────────────────────────────────────────────────────────────────────┘

┌─ Probability Calibration Analysis ───────────────────────────────────┐
│ calibration.png            [Reliability diagram + ECE]               │
│ → Shows: Predicted vs actual probability                             │
│ → Metrics: ECE (Expected Calibration Error)                          │
│ → Interpretation: Are probabilities trustworthy?                     │
│ → Action: High ECE → apply temperature scaling                       │
│          Low ECE → calibration OK                                    │
└──────────────────────────────────────────────────────────────────────┘

┌─ Latent Space Representation Analysis ───────────────────────────────┐
│ tsne_label.png             [t-SNE projection colored by engagement]  │
│ umap_label.png             [UMAP projection colored by engagement]   │
│ → Shows: 2D embedding visualization                                  │
│ → Interpretation: Do engagement classes cluster?                     │
│ → Action: Good separation → threshold/calibration issue              │
│          Poor separation → subject-specific fine-tuning needed       │
└──────────────────────────────────────────────────────────────────────┘

┌─ Subject Anomaly Detection ──────────────────────────────────────────┐
│ subject_shift_report.json  [Mahalanobis & Cosine distances]          │
│ → Metrics: Distance to training subject distribution                 │
│ → Interpretation: Is this subject an outlier?                        │
│ → Action: is_outlier=true → subject-specific issue                   │
│          is_outlier=false → problem not distribution shift           │
└──────────────────────────────────────────────────────────────────────┘

┌─ Individual Prediction Analysis ─────────────────────────────────────┐
│ ranking_analysis.csv       [Samples sorted by predicted probability] │
│ → Shows: Sample ID, True Label, Predicted Probability (sorted)       │
│ → Interpretation: Are individual predictions reasonable?             │
│ → Action: Review top/bottom predictions for anomalies                │
└──────────────────────────────────────────────────────────────────────┘

┌─ Automatic Failure Mode Diagnosis ───────────────────────────────────┐
│ diagnosis.json             [Structured failure mode data]            │
│ diagnosis.txt              [Human-readable diagnosis]                │
│ → Metrics: AUC, F1, Balanced Acc, ECE, Confidence Score              │
│ → Classification: Mode A/B/C/D/E                                     │
│ → Recommendation: Specific action for this subject                   │
│ → Confidence: How certain is this diagnosis?                         │
│ → Interpretation: What should I fix for this subject?                │
│ → Action: Follow diagnosis recommendation                            │
└──────────────────────────────────────────────────────────────────────┘

================================================================================
🔍 FAILURE MODE QUICK REFERENCE
================================================================================

MODE A: THRESHOLD ISSUE
═══════════════════════
Signature:       AUC > 0.70 AND |best_T - 0.5| > 0.10
Meaning:         Classes are separable, but decision threshold is wrong
Probability:     P(fail) = 1 - (AUC - 0.70) if large threshold diff
Expected Fix:    Adjust threshold from 0.5 → best_threshold
Effort:          ✓ EASY (inference-time only, no retraining)
Time to Fix:     < 1 hour
Expected Result: +10-30% improvement in F1/Accuracy

Checklist:
  [ ] Read best_threshold.json
  [ ] Update inference pipeline threshold
  [ ] Re-evaluate on validation set
  [ ] Deploy with new threshold


MODE B: CALIBRATION ISSUE
═══════════════════════════
Signature:       AUC > 0.65 AND ECE > 0.15
Meaning:         Ranking is good, but probabilities are miscalibrated
Probability:     P(fix helps) = 0.8 if ECE > 0.2
Expected Fix:    Apply temperature scaling or Platt scaling
Effort:          ✓ EASY (post-hoc, no model retraining)
Time to Fix:     < 2 hours
Expected Result: Reduce ECE by 50%, better calibration

Checklist:
  [ ] View calibration.png
  [ ] Compute temperature factor from validation set
  [ ] Apply T to model logits
  [ ] Verify ECE improvement
  [ ] Deploy with temperature scaling


MODE C: SUBJECT DISTRIBUTION SHIFT
════════════════════════════════════
Signature:       AUC < 0.60 AND F1 < 0.40
Meaning:         Subject has different EEG/ET characteristics
Probability:     P(subject-specific works) = 0.8
Expected Fix:    Subject-specific fine-tuning or domain adaptation
Effort:          ⚠ MEDIUM (requires training data & compute)
Time to Fix:     4-8 hours
Expected Result: +20-40% improvement in per-subject performance

Checklist:
  [ ] View subject_shift_report.json
  [ ] Verify is_outlier=true
  [ ] Collect subject-specific training data
  [ ] Fine-tune model on subject data
  [ ] Re-evaluate on test set
  [ ] Repeat for other Mode C subjects


MODE D: LABEL NOISE
═════════════════════
Signature:       0.50 < AUC < 0.60 AND F1 > 0.30
Meaning:         Ground truth labels may be incorrect or inconsistent
Probability:     P(labels are wrong) = 0.6-0.8
Expected Fix:    Manually review and correct ground truth labels
Effort:          ⚠ INVESTIGATION (manual review required)
Time to Fix:     2-4 hours (depends on dataset size)
Expected Result: Fix systematic label errors

Checklist:
  [ ] View ranking_analysis.csv
  [ ] Review bottom 10-20 HIGH-pred but LOW-true samples
  [ ] Review bottom 10-20 LOW-pred but HIGH-true samples
  [ ] Manually verify 5-10% of labels
  [ ] Correct obvious errors
  [ ] Retrain model with corrected labels


MODE E: REPRESENTATION FAILURE
════════════════════════════════
Signature:       AUC < 0.50 AND Balanced_Acc < 0.50
Meaning:         Model cannot learn meaningful representation
Probability:     P(fundamental issue) = 0.9
Expected Fix:    Debug preprocessing, artifacts, or architecture
Effort:          ✗ HARD (may require significant investigation)
Time to Fix:     8-16 hours (deep debugging required)
Expected Result: Fix fundamental issue

Checklist:
  [ ] Review probability_histogram.png (are distributions identical?)
  [ ] View embeddings (tsne_label.png, umap_label.png)
  [ ] Check EEG preprocessing: filtering, artifact removal
  [ ] Check ET preprocessing: calibration, outlier handling
  [ ] Verify engagement label generation logic
  [ ] Validate ground truth labels (Mode D check)
  [ ] Review model architecture vs other subjects
  [ ] Consider: different model, architecture change, feature engineering

================================================================================
📊 SUMMARY REPORT INTERPRETATION
================================================================================

File: output/hard_subjects_summary.txt

Sections:

1. EXECUTIVE SUMMARY
   ─────────────────
   Shows: Mean/std metrics across all hard subjects
   
   Look for:
   - Mean AUC < 0.60 → Fundamental ranking issue
   - Mean F1 < 0.50 → Poor classification overall
   - Mean ECE > 0.15 → Calibration issues
   - Wide std dev → High subject variability
   
   Questions to ask:
   Q: Why are hard subjects harder than easy subjects?
   Q: Is there a systematic pattern?


2. PER-SUBJECT TABLE
   ─────────────────
   Shows: All metrics for all subjects side-by-side
   
   Look for:
   - AUC: Identify subjects with AUC < 0.60
   - F1: Identify subjects with F1 < 0.50
   - ECE: Identify subjects with ECE > 0.15
   - Confidence: Prioritize by confidence score
   
   Questions to ask:
   Q: Which subject is the worst?
   Q: Which subjects have same failure mode?


3. FAILURE MODE DISTRIBUTION
   ──────────────────────────
   Shows: Count & percentage of each failure mode (A-E)
   
   Look for:
   - If one mode dominates → Global fix possible
   - If modes are mixed → Different fixes needed
   
   Examples:
   ✓ All Mode A (Threshold) → Apply threshold optimization
   ✓ All Mode B (Calibration) → Apply temperature scaling
   ✗ Mixed A+B+C → Different fixes per subject


4. RECOMMENDATIONS BY PRIORITY
   ───────────────────────────
   Shows: Subjects ranked by confidence score
   
   Look for:
   - Top priority: Highest confidence scores
   - Action: Specific recommendation per subject
   - Effort: Estimated fix difficulty
   
   Workflow:
   1. Fix highest confidence subjects first
   2. Build momentum
   3. Leave low-confidence subjects for deep investigation


5. DETAILED SUBJECT PROFILES
   ─────────────────────────
   Shows: Complete diagnostic info per subject
   
   Look for:
   - Metrics: AUC, F1, ECE for this subject
   - Diagnosis text: What's wrong & why
   - Recommendation: What to do about it
   - Threshold: Current vs recommended


================================================================================
🚀 FAST EXECUTION PATHS
================================================================================

Path 1: Skip Embeddings (Fastest)
─────────────────────────────────
$ bash scripts/run_diagnostics_pipeline.sh --skip-embeddings

Time: ~15-20 minutes
Skips: t-SNE, UMAP
Includes: All other diagnostics (9/10)


Path 2: Quick Mode (Super Fast)
────────────────────────────────
$ bash scripts/run_diagnostics_pipeline.sh --quick

Time: ~15 minutes
Skips: t-SNE, UMAP, large visualizations
Includes: Core diagnostics


Path 3: Single Subject Diagnostics
──────────────────────────────────
$ python hard_subject_diagnostics.py --hard-subjects S21

Time: ~3-5 minutes per subject
Use: To focus on one subject


Path 4: Summary Only
────────────────────
$ python hard_subject_summary.py \\
    --diagnostics-dir output/diagnostics

Time: ~1 minute
Use: If diagnostics already generated

================================================================================
📝 COMMON WORKFLOWS
================================================================================

Workflow 1: Initial Diagnostics
───────────────────────────────
1. $ bash scripts/run_diagnostics_pipeline.sh --skip-embeddings  [15 min]
2. $ cat output/hard_subjects_summary.txt               [2 min]
3. $ for s in S21 S03 S13 S35 S36; do \
     echo "=== $s ===" && \
     jq '.failure_modes, .confidence_score' \
       output/diagnostics/$s/diagnosis.json
   done                                                  [1 min]

Total: ~18 minutes to identify all issues


Workflow 2: Deep Dive on One Subject
─────────────────────────────────────
1. $ python hard_subject_diagnostics.py --hard-subjects S21  [3 min]
2. Open: output/diagnostics/S21/probability_histogram.png   [30 sec]
3. Open: output/diagnostics/S21/roc_curve.png               [30 sec]
4. Open: output/diagnostics/S21/threshold_sweep.png         [30 sec]
5. $ cat output/diagnostics/S21/diagnosis.txt               [1 min]

Total: ~6 minutes to understand S21 issue


Workflow 3: Implement Threshold Fix (Mode A)
──────────────────────────────────────────────
1. $ jq '.recommended_threshold' output/diagnostics/S21/diagnosis.json  [10 sec]
2. Update: inference pipeline threshold config                         [5 min]
3. $ python validate_threshold_fix.py                                  [2 min]
4. $ bash scripts/run_diagnostics_pipeline.sh --skip-embeddings  [15 min]
5. $ cat output/hard_subjects_summary.txt | grep -E "S21|AUC"  [1 min]

Total: ~23 minutes (test new threshold)


Workflow 4: Verify Calibration Fix (Mode B)
──────────────────────────────────────────────
1. $ python apply_temperature_scaling.py --subject S21       [5 min]
2. $ cat calibration_factors.json                           [30 sec]
3. Update: inference pipeline temperature scaling          [5 min]
4. $ python validate_calibration_fix.py                    [2 min]
5. $ bash scripts/run_diagnostics_pipeline.sh --skip-embeddings    [15 min]

Total: ~27 minutes (test new calibration)

================================================================================
⏱️  TIME ESTIMATES
================================================================================

Task                                    Estimate
────────────────────────────────────────────────────────────────────────────
Diagnostics (all 5 subjects)            30-45 min (embeddings)
                                        15-20 min (--skip-embeddings)
                                        ~10 min (--quick)

Per-subject diagnostics                 5-10 min (embeddings)
                                        2-3 min (--skip-embeddings)

Summary report generation               ~1 min

View & interpret results                5-15 min

Implement threshold fix                 15-30 min (+ 15 min validation)

Implement calibration fix               20-30 min (+ 15 min validation)

Subject-specific fine-tuning            2-4 hours (+ data collection)

Label review & correction               2-4 hours (manual review)

Deep investigation (Mode E)             8-16 hours (full debugging)


Total (initial diagnostics + quick fix) 1-2 hours
Total (diagnostics + all fixes)         1-2 days

================================================================================
🆘 TROUBLESHOOTING QUICK REFERENCE
================================================================================

Issue: "fold probs not found"
Cause: collect_fold_probs.py not run
Fix: $ python collect_fold_probs.py
    Then: $ bash scripts/run_diagnostics_pipeline.sh

Issue: Embeddings extraction times out
Cause: Model inference is slow
Fix: $ bash scripts/run_diagnostics_pipeline.sh --skip-embeddings
    Or: $ python hard_subject_diagnostics.py --skip-embeddings

Issue: Threshold sweep shows flat line
Cause: Model predicts mostly one class
Fix: Check probability_histogram.png
    Check confusion_matrix.png
    Review confidence/calibration issues

Issue: All subjects Mode C (distribution shift)
Cause: High subject variability or subject-specific model needed
Fix: Consider: per-subject models or domain adaptation
    Or: Verify preprocessing is consistent across subjects

Issue: diagnosis.txt/json missing
Cause: Diagnostics run failed
Fix: Check terminal output for errors
    Run: $ python hard_subject_diagnostics.py --debug
    Check disk space and permissions

Issue: Comparison plots not generated
Cause: hard_subject_summary.py not run
Fix: $ python hard_subject_summary.py \\
      --diagnostics-dir output/diagnostics

================================================================================
📞 WHERE TO GET HELP
================================================================================

For:                            See:
──────────────────────────────────────────────────────────────────────────────
How to run                      → This document (QUICK START)
How to interpret                → HARD_SUBJECT_DIAGNOSTICS_README.md
Detailed workflow               → HARD_SUBJECT_DIAGNOSTICS_IMPLEMENTATION.txt
Troubleshooting                 → This document (TROUBLESHOOTING)
API reference                   → Function docstrings in Python files
Output structure                → This document (DIAGNOSTIC OUTPUTS)

================================================================================
✅ VERIFICATION CHECKLIST
================================================================================

Before running:
  [ ] All prerequisite files exist
      • output/fold_probs/fold*.npz
      • output/checkpoints/ins_hdgs_cmt_v17/ (for embeddings)
      • data/ directory (for feature extraction)
  [ ] Python environment configured
  [ ] Disk space available (~1-2 GB)
  [ ] GPU available (for embeddings)

After running:
  [ ] output/diagnostics/ directory created
  [ ] All 5 subject directories present (S21, S03, S13, S35, S36)
  [ ] Each subject has 11-13 output files
  [ ] hard_subjects_summary.txt generated
  [ ] comparison_*.png files generated

Before implementing fixes:
  [ ] All diagnostics reviewed
  [ ] Failure modes identified (A-E)
  [ ] Priority ranking established
  [ ] Specific actions defined per subject
  [ ] Confidence scores reviewed

After implementing fixes:
  [ ] Changes integrated into pipeline
  [ ] Re-ran diagnostics
  [ ] Metrics improved as expected
  [ ] Changes deployed or tested

================================================================================
🎓 LEARNING PATH
================================================================================

Beginner (Just Started):
1. Read: QUICK START
2. Run: bash scripts/run_diagnostics_pipeline.sh --skip-embeddings
3. View: cat output/hard_subjects_summary.txt

Intermediate (Want to Implement Fixes):
1. Read: FAILURE MODE QUICK REFERENCE
2. Read: COMMON WORKFLOWS
3. Study: diagnosis.txt files per subject
4. Implement: Mode A/B fixes

Advanced (Deep Investigation):
1. Read: HARD_SUBJECT_DIAGNOSTICS_IMPLEMENTATION.txt (full document)
2. Study: Individual PNG visualizations
3. Analyze: Subject-specific features
4. Implement: Mode C/D/E fixes

================================================================================
END OF QUICK REFERENCE
================================================================================
