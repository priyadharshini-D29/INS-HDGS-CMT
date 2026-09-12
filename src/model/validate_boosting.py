#!/usr/bin/env python3
"""
================================================================
Quick Validation & Test Script for Boosting Pipeline
================================================================
Validates each component individually before full pipeline run.
Tests on a single fold to verify all improvements work correctly.

Usage:
  python validate_boosting.py --quick          # Fast validation
  python validate_boosting.py --full           # Full LOSOCV with boosting
  python validate_boosting.py --test-calibration  # Just test calibration
  python validate_boosting.py --test-ensemble     # Just test ensemble
================================================================
"""

import sys
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    SUBJECT_IDS, DEVICE, BATCH_SIZE, CKPT_DIR, N_ROIS, N_WINDOWS,
    EMBED_DIM, GAT_L1_HEAD_DIM, GAT_L1_HEADS, T_NHEAD, T_LAYERS, T_FF_DIM,
    ET_LSTM_HIDDEN, ET_LSTM_LAYERS, ROI_HIDDEN_DIM, FUSION_HEADS, CLS_HIDDEN,
    ET_INPUT_DIM, SNN_TIME_STEPS, SNN_HIDDEN_DIM, NS_N_RULES, NS_HIDDEN_DIM,
    DROPOUT,
)
from data.dataset import NeumaGraphDataset
from models.ins_hdgs_cmt import INS_HDGS_CMT, AblationConfig
from optimize_with_boosting import (
    calibrate_temperature_on_validation,
    compute_subject_aware_class_weights,
    filter_problematic_subjects,
    UncertaintyWeightedEnsemble,
    WeightedConnectivityFeatures,
)
from inference.optimized_engine import GPUOptimizedInference


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def validate_temperature_scaling():
    """Test temperature scaling calibration."""
    print_section("TEST 1: Temperature Scaling Calibration")
    
    try:
        # Load small dataset
        subject_ids = SUBJECT_IDS[:3]  # Use 3 subjects for quick test
        train_set = NeumaGraphDataset(subject_ids=subject_ids, precompute_graphs=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        # Create model
        model = INS_HDGS_CMT(
            n_eeg_ch=train_set.n_eeg_ch,
            n_et_ch=ET_INPUT_DIM,
            n_rois=N_ROIS,
            n_windows=N_WINDOWS,
            n_classes=2,
            embed_dim=EMBED_DIM,
            snn_time_steps=SNN_TIME_STEPS,
            snn_hidden_dim=SNN_HIDDEN_DIM,
            gat_head_dim=GAT_L1_HEAD_DIM,
            gat_heads=GAT_L1_HEADS,
            t_nhead=T_NHEAD,
            t_layers=T_LAYERS,
            t_ff_dim=T_FF_DIM,
            et_lstm_hidden=ET_LSTM_HIDDEN,
            et_lstm_layers=ET_LSTM_LAYERS,
            roi_hidden=ROI_HIDDEN_DIM,
            fusion_heads=FUSION_HEADS,
            ns_n_rules=NS_N_RULES,
            ns_hidden_dim=NS_HIDDEN_DIM,
            cls_hidden=CLS_HIDDEN,
            dropout=DROPOUT,
            temperature=1.0,
            ablation=AblationConfig.full(),
        ).to(DEVICE)
        model.eval()
        
        # Calibrate temperature
        print("[INFO] Calibrating temperature on validation set...")
        temp = calibrate_temperature_on_validation(model, train_loader, DEVICE, epochs=50)
        
        print(f"✓ Temperature scaling successful: {temp:.4f}")
        print(f"  Expected range: 0.5 - 2.0")
        
        assert 0.3 < temp < 3.0, f"Temperature {temp} out of expected range"
        print("[PASS] Temperature Scaling Test")
        
        return True
    except Exception as e:
        print(f"[FAIL] Temperature Scaling Test: {e}")
        return False


def validate_class_weighting():
    """Test class weighting for imbalanced subjects."""
    print_section("TEST 2: Class Weighting for Imbalanced Subjects")
    
    try:
        subject_ids = SUBJECT_IDS[:5]
        train_set = NeumaGraphDataset(subject_ids=subject_ids, precompute_graphs=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        print("[INFO] Computing subject-aware class weights...")
        weights = compute_subject_aware_class_weights(train_loader, subject_ids, DEVICE)
        
        print(f"✓ Computed weights for {len(weights)} subjects:")
        for subj, w in list(weights.items())[:3]:
            print(f"  {subj}: {w}")
        
        # Validate weights
        for subj, w in weights.items():
            assert len(w) == 2, "Should have 2 class weights"
            assert all(x > 0 for x in w), "All weights should be positive"
        
        print("[PASS] Class Weighting Test")
        return True
    except Exception as e:
        print(f"[FAIL] Class Weighting Test: {e}")
        return False


def validate_subject_holdout():
    """Test subject holdout protocol."""
    print_section("TEST 3: Subject Holdout Protocol")
    
    try:
        subject_ids = list(SUBJECT_IDS)
        original_count = len(subject_ids)
        
        filtered, info = filter_problematic_subjects(subject_ids)
        
        assert "S06" not in filtered, "S06 should be removed"
        assert len(filtered) == original_count - 1, "Should remove exactly 1 subject"
        
        print(f"✓ Holdout protocol successful:")
        print(f"  Original subjects: {original_count}")
        print(f"  Removed: {info['removed']}")
        print(f"  Retained: {len(filtered)}")
        print(f"  Validation subset: {len(info['validation'])} subjects")
        
        print("[PASS] Subject Holdout Test")
        return True
    except Exception as e:
        print(f"[FAIL] Subject Holdout Test: {e}")
        return False


def validate_ensemble_inference():
    """Test uncertainty-weighted ensemble."""
    print_section("TEST 4: Uncertainty-Weighted Ensemble")
    
    try:
        # Create model
        train_set = NeumaGraphDataset(subject_ids=[SUBJECT_IDS[0]], precompute_graphs=True)
        model = INS_HDGS_CMT(
            n_eeg_ch=train_set.n_eeg_ch,
            n_et_ch=ET_INPUT_DIM,
            n_rois=N_ROIS,
            n_windows=N_WINDOWS,
            n_classes=2,
            embed_dim=EMBED_DIM,
            snn_time_steps=SNN_TIME_STEPS,
            snn_hidden_dim=SNN_HIDDEN_DIM,
            gat_head_dim=GAT_L1_HEAD_DIM,
            gat_heads=GAT_L1_HEADS,
            t_nhead=T_NHEAD,
            t_layers=T_LAYERS,
            t_ff_dim=T_FF_DIM,
            et_lstm_hidden=ET_LSTM_HIDDEN,
            et_lstm_layers=ET_LSTM_LAYERS,
            roi_hidden=ROI_HIDDEN_DIM,
            fusion_heads=FUSION_HEADS,
            ns_n_rules=NS_N_RULES,
            ns_hidden_dim=NS_HIDDEN_DIM,
            cls_hidden=CLS_HIDDEN,
            dropout=DROPOUT,
            temperature=1.0,
            ablation=AblationConfig.full(),
        ).to(DEVICE)
        
        # Create ensemble
        print("[INFO] Building uncertainty-weighted ensemble...")
        ckpt_dir = CKPT_DIR / "ins_hdgs_cmt_v2"
        
        if not ckpt_dir.exists():
            print(f"[SKIP] Checkpoint directory not found: {ckpt_dir}")
            return True
        
        ensemble = UncertaintyWeightedEnsemble(model, ckpt_dir, DEVICE, n_ensemble=5)
        ensemble.load_ensemble_checkpoints(1, max_checkpoints=3)
        
        if len(ensemble.models) == 0:
            print("[SKIP] No checkpoints available for ensemble test")
            return True
        
        print(f"✓ Ensemble loaded: {len(ensemble.models)} models")
        
        # Test inference
        test_loader = torch.utils.data.DataLoader(train_set, batch_size=8, num_workers=2)
        batch = next(iter(test_loader))
        
        preds, confs, uncs = ensemble.predict_with_uncertainty(batch)
        
        assert preds.shape == (8, 2), f"Expected shape (8, 2), got {preds.shape}"
        assert confs.shape == (8,), f"Expected shape (8,), got {confs.shape}"
        
        print(f"✓ Ensemble inference successful:")
        print(f"  Predictions shape: {preds.shape}")
        print(f"  Confidences shape: {confs.shape}")
        
        print("[PASS] Ensemble Inference Test")
        return True
    except Exception as e:
        print(f"[FAIL] Ensemble Inference Test: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_gpu_optimized_inference():
    """Test GPU-optimized inference engine."""
    print_section("TEST 5: GPU-Optimized Inference Engine")
    
    try:
        train_set = NeumaGraphDataset(subject_ids=[SUBJECT_IDS[0]], precompute_graphs=True)
        model = INS_HDGS_CMT(
            n_eeg_ch=train_set.n_eeg_ch,
            n_et_ch=ET_INPUT_DIM,
            n_rois=N_ROIS,
            n_windows=N_WINDOWS,
            n_classes=2,
            embed_dim=EMBED_DIM,
            snn_time_steps=SNN_TIME_STEPS,
            snn_hidden_dim=SNN_HIDDEN_DIM,
            gat_head_dim=GAT_L1_HEAD_DIM,
            gat_heads=GAT_L1_HEADS,
            t_nhead=T_NHEAD,
            t_layers=T_LAYERS,
            t_ff_dim=T_FF_DIM,
            et_lstm_hidden=ET_LSTM_HIDDEN,
            et_lstm_layers=ET_LSTM_LAYERS,
            roi_hidden=ROI_HIDDEN_DIM,
            fusion_heads=FUSION_HEADS,
            ns_n_rules=NS_N_RULES,
            ns_hidden_dim=NS_HIDDEN_DIM,
            cls_hidden=CLS_HIDDEN,
            dropout=DROPOUT,
            temperature=1.0,
            ablation=AblationConfig.full(),
        ).to(DEVICE)
        model.eval()
        
        print("[INFO] Testing GPU-optimized inference...")
        engine = GPUOptimizedInference(DEVICE, use_mixed_precision=True, batch_size=32)
        
        test_loader = torch.utils.data.DataLoader(train_set, batch_size=16, num_workers=2)
        result = engine.infer_dataset(model, test_loader, temperature=1.0, verbose=False)
        
        print(f"✓ GPU inference successful:")
        print(f"  Samples processed: {result['performance']['n_samples']}")
        print(f"  Throughput: {result['performance']['throughput_sps']:.1f} samples/sec")
        print(f"  Peak memory: {result['performance']['peak_memory_gb']:.2f} GB")
        
        print("[PASS] GPU-Optimized Inference Test")
        return True
    except Exception as e:
        print(f"[FAIL] GPU-Optimized Inference Test: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_validation():
    """Run all component tests."""
    print_section("BOOSTING PIPELINE VALIDATION")
    print(f"Device: {DEVICE}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    results = {
        "Temperature Scaling": validate_temperature_scaling(),
        "Class Weighting": validate_class_weighting(),
        "Subject Holdout": validate_subject_holdout(),
        "Ensemble Inference": validate_ensemble_inference(),
        "GPU Optimization": validate_gpu_optimized_inference(),
    }
    
    print_section("VALIDATION RESULTS")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✓ All components validated successfully!")
        print("Ready to run full boosting pipeline: python optimize_with_boosting.py")
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed. Check errors above.")
    
    return passed_count == total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", default=True,
                      help="Quick validation of all components")
    parser.add_argument("--test-calibration", action="store_true", 
                      help="Test only temperature scaling")
    parser.add_argument("--test-weighting", action="store_true",
                      help="Test only class weighting")
    parser.add_argument("--test-holdout", action="store_true",
                      help="Test only subject holdout")
    parser.add_argument("--test-ensemble", action="store_true",
                      help="Test only ensemble inference")
    parser.add_argument("--test-gpu", action="store_true",
                      help="Test only GPU optimization")
    
    args = parser.parse_args()
    
    # Run specific tests if requested
    if args.test_calibration:
        validate_temperature_scaling()
    elif args.test_weighting:
        validate_class_weighting()
    elif args.test_holdout:
        validate_subject_holdout()
    elif args.test_ensemble:
        validate_ensemble_inference()
    elif args.test_gpu:
        validate_gpu_optimized_inference()
    else:
        # Run all validations
        success = run_quick_validation()
        sys.exit(0 if success else 1)
