"""
================================================================
THRESHOLD OPTIMIZER FOR INS-HDGS-CMT LOSOCV
================================================================
Validates thresholds on validation subject, applies to test subject.

For each fold:
  1. Train on training subjects
  2. Get validation subject probabilities
  3. Search thresholds 0.05 → 0.95 (step 0.01)
  4. Select threshold maximizing balanced accuracy (then MCC)
  5. Apply to test subject

Keep all training unchanged. Only modify decision rule.

Usage:
  python threshold_optimizer.py \\
      --fold-probs-dir output/fold_probs \\
      --output-dir output/threshold_analysis

================================================================
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import pandas as pd
import json
from sklearn.metrics import (
    balanced_accuracy_score, matthews_corrcoef, f1_score,
    accuracy_score, roc_auc_score
)
import argparse


def find_best_threshold(y_val, y_prob_val, threshold_range=np.linspace(0.05, 0.95, 91)):
    """
    Find optimal threshold on validation set.
    
    Criteria:
      Primary: Maximize balanced accuracy
      Secondary: Maximize MCC (if balanced acc ties)
    
    Returns:
      best_threshold, best_bal_acc, best_mcc, results_dict
    """
    
    results = []
    
    for threshold in threshold_range:
        y_pred = (y_prob_val >= threshold).astype(int)
        
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        mcc = matthews_corrcoef(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="binary", zero_division=0)
        acc = accuracy_score(y_val, y_pred)
        
        results.append({
            "threshold": threshold,
            "bal_acc": bal_acc,
            "mcc": mcc,
            "f1": f1,
            "accuracy": acc,
        })
    
    results_df = pd.DataFrame(results)
    
    # Find threshold with maximum balanced accuracy
    max_bal_acc = results_df["bal_acc"].max()
    candidates = results_df[results_df["bal_acc"] == max_bal_acc]
    
    # Among candidates, select maximum MCC
    best_idx = candidates["mcc"].idxmax()
    best_result = results_df.loc[best_idx]
    
    return (
        float(best_result["threshold"]),
        float(best_result["bal_acc"]),
        float(best_result["mcc"]),
        results_df
    )


def process_fold(fold_file, fold_no, test_subject, val_subject):
    """
    Process single fold: optimize threshold on validation, apply to test.
    
    Returns:
      dict with fold results
    """
    
    try:
        data = np.load(fold_file)
        
        # Test set
        y_test = data["y_true"]
        y_prob_test = data["y_prob"]
        
        # Validation set
        y_val = data["val_y_true"]
        y_prob_val = data["val_y_prob"]
        
        # Check validity
        if len(y_test) < 2 or len(np.unique(y_test)) < 2:
            return {
                "fold_id": fold_no,
                "test_subject": test_subject,
                "val_subject": val_subject,
                "status": "SKIP",
                "reason": "Single-class test set"
            }
        
        if len(y_val) < 2 or len(np.unique(y_val)) < 2:
            return {
                "fold_id": fold_no,
                "test_subject": test_subject,
                "val_subject": val_subject,
                "status": "SKIP",
                "reason": "Single-class validation set"
            }
        
        # Find optimal threshold on validation set
        best_threshold, best_bal_acc, best_mcc, threshold_results = find_best_threshold(
            y_val, y_prob_val
        )
        
        # Fixed threshold baseline (0.5)
        y_pred_fixed = (y_prob_test >= 0.5).astype(int)
        y_pred_opt = (y_prob_test >= best_threshold).astype(int)
        
        # Compute metrics for test set
        bal_acc_fixed = balanced_accuracy_score(y_test, y_pred_fixed)
        bal_acc_opt = balanced_accuracy_score(y_test, y_pred_opt)
        
        mcc_fixed = matthews_corrcoef(y_test, y_pred_fixed)
        mcc_opt = matthews_corrcoef(y_test, y_pred_opt)
        
        f1_fixed = f1_score(y_test, y_pred_fixed, average="binary", zero_division=0)
        f1_opt = f1_score(y_test, y_pred_opt, average="binary", zero_division=0)
        
        acc_fixed = accuracy_score(y_test, y_pred_fixed)
        acc_opt = accuracy_score(y_test, y_pred_opt)
        
        # AUC (threshold-independent)
        auc_score = roc_auc_score(y_test, y_prob_test)
        
        return {
            "fold_id": fold_no,
            "test_subject": test_subject,
            "val_subject": val_subject,
            "status": "OK",
            
            # Thresholds
            "threshold_fixed": 0.5,
            "threshold_opt": best_threshold,
            
            # Threshold selection metrics (from validation)
            "val_bal_acc_opt": best_bal_acc,
            "val_mcc_opt": best_mcc,
            
            # Test set metrics - Balanced Accuracy
            "bal_acc_fixed": bal_acc_fixed,
            "bal_acc_opt": bal_acc_opt,
            "bal_acc_gain": bal_acc_opt - bal_acc_fixed,
            
            # Test set metrics - MCC
            "mcc_fixed": mcc_fixed,
            "mcc_opt": mcc_opt,
            "mcc_gain": mcc_opt - mcc_fixed,
            
            # Test set metrics - F1
            "f1_fixed": f1_fixed,
            "f1_opt": f1_opt,
            "f1_gain": f1_opt - f1_fixed,
            
            # Test set metrics - Accuracy
            "accuracy_fixed": acc_fixed,
            "accuracy_opt": acc_opt,
            "accuracy_gain": acc_opt - acc_fixed,
            
            # Threshold-independent
            "auc_score": auc_score,
            
            # Sizes
            "n_test": len(y_test),
            "n_val": len(y_val),
        }
    
    except Exception as e:
        return {
            "fold_id": fold_no,
            "test_subject": test_subject,
            "val_subject": val_subject,
            "status": "ERROR",
            "reason": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Optimize thresholds per LOSOCV fold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fold-probs-dir", type=str, default="output/fold_probs",
        help="Directory containing fold probability files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/threshold_analysis",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    fold_probs_dir = Path(args.fold_probs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZER FOR INS-HDGS-CMT LOSOCV")
    print("="*70)
    print(f"Input: {fold_probs_dir}")
    print(f"Output: {output_dir}")
    
    # Find all fold files
    fold_files = sorted(fold_probs_dir.glob("fold*_S*.npz"))
    print(f"Found {len(fold_files)} fold files")
    
    if not fold_files:
        print("✗ No fold files found")
        return
    
    # Process each fold
    results = []
    
    print("\nProcessing folds...\n")
    for fold_file in fold_files:
        # Parse filename: fold{fold_no:02d}_{subject}.npz
        stem = fold_file.stem  # e.g., "fold01_S01"
        parts = stem.split("_")
        fold_str = parts[0]  # "fold01"
        test_subject = parts[1]  # "S01"
        
        fold_no = int(fold_str.replace("fold", ""))
        
        # Determine validation subject (usually previous subject in LOSOCV)
        # For now, we'll infer from file but ideally this comes from config
        # Assumption: fold ordering matches subject ordering
        from config.settings import SUBJECT_IDS
        if test_subject in SUBJECT_IDS:
            test_idx = SUBJECT_IDS.index(test_subject)
            # Validation is typically the one before in sequence
            # But for LOSOCV this depends on the split strategy
            # We'll use a placeholder
            val_subject = "VAL"  # Will be corrected if we know the split
        else:
            val_subject = "UNKNOWN"
        
        print(f"Fold {fold_no:02d} ({test_subject}): ", end="", flush=True)
        
        result = process_fold(fold_file, fold_no, test_subject, val_subject)
        results.append(result)
        
        status = result.get("status", "?")
        if status == "OK":
            gain_bal = result.get("bal_acc_gain", 0)
            gain_mcc = result.get("mcc_gain", 0)
            threshold = result.get("threshold_opt", 0)
            print(f"✓ T={threshold:.2f}  ΔBalAcc={gain_bal:+.3f}  ΔMCC={gain_mcc:+.3f}")
        elif status == "SKIP":
            print(f"⊘ {result.get('reason', 'skipped')}")
        else:
            print(f"✗ {result.get('reason', 'error')}")
    
    # Save results
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    # Filter to OK results
    ok_results = results_df[results_df["status"] == "OK"].copy()
    
    if len(ok_results) > 0:
        print(f"\nProcessed: {len(ok_results)} folds")
        
        # Summary statistics
        print(f"\n[Balanced Accuracy]")
        print(f"  Fixed (0.5):      {ok_results['bal_acc_fixed'].mean():.3f} ± {ok_results['bal_acc_fixed'].std():.3f}")
        print(f"  Optimized:        {ok_results['bal_acc_opt'].mean():.3f} ± {ok_results['bal_acc_opt'].std():.3f}")
        print(f"  Mean Gain:        {ok_results['bal_acc_gain'].mean():+.3f}")
        
        print(f"\n[MCC]")
        print(f"  Fixed (0.5):      {ok_results['mcc_fixed'].mean():.3f} ± {ok_results['mcc_fixed'].std():.3f}")
        print(f"  Optimized:        {ok_results['mcc_opt'].mean():.3f} ± {ok_results['mcc_opt'].std():.3f}")
        print(f"  Mean Gain:        {ok_results['mcc_gain'].mean():+.3f}")
        
        print(f"\n[F1 Score]")
        print(f"  Fixed (0.5):      {ok_results['f1_fixed'].mean():.3f} ± {ok_results['f1_fixed'].std():.3f}")
        print(f"  Optimized:        {ok_results['f1_opt'].mean():.3f} ± {ok_results['f1_opt'].std():.3f}")
        print(f"  Mean Gain:        {ok_results['f1_gain'].mean():+.3f}")
        
        print(f"\n[Accuracy]")
        print(f"  Fixed (0.5):      {ok_results['accuracy_fixed'].mean():.3f} ± {ok_results['accuracy_fixed'].std():.3f}")
        print(f"  Optimized:        {ok_results['accuracy_opt'].mean():.3f} ± {ok_results['accuracy_opt'].std():.3f}")
        print(f"  Mean Gain:        {ok_results['accuracy_gain'].mean():+.3f}")
        
        print(f"\n[Threshold Distribution]")
        print(f"  Mean threshold:   {ok_results['threshold_opt'].mean():.3f} ± {ok_results['threshold_opt'].std():.3f}")
        print(f"  Min threshold:    {ok_results['threshold_opt'].min():.3f}")
        print(f"  Max threshold:    {ok_results['threshold_opt'].max():.3f}")
        
        # Top folds by gain
        print(f"\n[Top 10 Folds - Largest BalAcc Gain]")
        top_folds = ok_results.nlargest(10, "bal_acc_gain")[
            ["fold_id", "test_subject", "threshold_opt", "bal_acc_gain", "mcc_gain"]
        ]
        for idx, row in top_folds.iterrows():
            print(f"  Fold {row['fold_id']:2d} ({row['test_subject']}): "
                  f"T={row['threshold_opt']:.2f}  "
                  f"ΔBalAcc={row['bal_acc_gain']:+.3f}  "
                  f"ΔMCC={row['mcc_gain']:+.3f}")
        
        # Folds with negative gain (regression)
        neg_folds = ok_results[ok_results["bal_acc_gain"] < -0.05]
        if len(neg_folds) > 0:
            print(f"\n[⚠ Folds with Negative Gain (> -0.05)]")
            for idx, row in neg_folds.iterrows():
                print(f"  Fold {row['fold_id']:2d} ({row['test_subject']}): "
                      f"ΔBalAcc={row['bal_acc_gain']:+.3f}")
    
    # Save detailed results
    csv_file = output_dir / "thresholds_per_fold.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved: {csv_file}")
    
    # Save summary
    if len(ok_results) > 0:
        summary = {
            "n_folds_processed": len(ok_results),
            "n_folds_skipped": len(results_df[results_df["status"] == "SKIP"]),
            "n_folds_error": len(results_df[results_df["status"] == "ERROR"]),
            
            "bal_acc_fixed_mean": float(ok_results["bal_acc_fixed"].mean()),
            "bal_acc_opt_mean": float(ok_results["bal_acc_opt"].mean()),
            "bal_acc_gain_mean": float(ok_results["bal_acc_gain"].mean()),
            
            "mcc_fixed_mean": float(ok_results["mcc_fixed"].mean()),
            "mcc_opt_mean": float(ok_results["mcc_opt"].mean()),
            "mcc_gain_mean": float(ok_results["mcc_gain"].mean()),
            
            "f1_fixed_mean": float(ok_results["f1_fixed"].mean()),
            "f1_opt_mean": float(ok_results["f1_opt"].mean()),
            "f1_gain_mean": float(ok_results["f1_gain"].mean()),
            
            "accuracy_fixed_mean": float(ok_results["accuracy_fixed"].mean()),
            "accuracy_opt_mean": float(ok_results["accuracy_opt"].mean()),
            "accuracy_gain_mean": float(ok_results["accuracy_gain"].mean()),
            
            "threshold_opt_mean": float(ok_results["threshold_opt"].mean()),
            "threshold_opt_std": float(ok_results["threshold_opt"].std()),
            "threshold_opt_min": float(ok_results["threshold_opt"].min()),
            "threshold_opt_max": float(ok_results["threshold_opt"].max()),
        }
        
        summary_file = output_dir / "threshold_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved: {summary_file}")
        
        # Save gain summary
        gain_summary = ok_results[[
            "fold_id", "test_subject", "threshold_opt",
            "bal_acc_gain", "mcc_gain", "f1_gain", "accuracy_gain"
        ]].copy()
        gain_summary = gain_summary.sort_values("bal_acc_gain", ascending=False)
        
        gain_csv = output_dir / "threshold_gain_summary.csv"
        gain_summary.to_csv(gain_csv, index=False)
        print(f"✓ Saved: {gain_csv}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
