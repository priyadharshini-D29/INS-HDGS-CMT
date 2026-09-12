"""
================================================================
HARD-SUBJECT DIAGNOSTIC FRAMEWORK
================================================================
Comprehensive analysis for hard subjects in INS-HDGS-CMT LOSOCV pipeline.

Hard Subjects:
  S21, S03, S13, S35, S36

Diagnostics (per subject):
  1. Probability Histogram (class separability)
  2. ROC Curve (AUC + operating threshold)
  3. Precision-Recall Curve (AP score)
  4. Threshold Sweep (accuracy, balanced accuracy, F1, MCC)
  5. Confusion Matrix (raw + normalized)
  6. Calibration Analysis (reliability diagram, ECE)
  7. Embedding Visualization (t-SNE + UMAP, by label and subject)
  8. Feature Statistics (Mahalanobis distance, cosine distance)
  9. Ranking Failure Analysis (sorted by probability)
  10. Automatic Subject Diagnosis (failure mode classification)

Usage:
  python hard_subject_diagnostics.py \\
      --label ins_hdgs_cmt_v17 \\
      --skip-embeddings  # (optional, faster if no embedding extraction)
      --hard-subjects S21,S03,S13,S35,S36
      --output-dir output/diagnostics

================================================================
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYTHONUTF8", "1")

# Add parent directory to path for imports
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score, f1_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis, cosine
from scipy.stats import entropy
import torch
import multiprocessing as mp

from config.settings import SUBJECT_IDS
from data.dataset import NeumaGraphDataset
from utils.gpu import _build_raw_model


# ============================================================================
# 1. PROBABILITY HISTOGRAM
# ============================================================================

def plot_probability_histogram(y_true, y_prob, output_dir):
    """Plot predicted probabilities separated by true class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    high_probs = y_prob[y_true == 1]
    low_probs = y_prob[y_true == 0]
    
    bins = np.linspace(0, 1, 31)
    ax.hist(low_probs, bins=bins, alpha=0.6, label="True LOW", color="blue", edgecolor="black")
    ax.hist(high_probs, bins=bins, alpha=0.6, label="True HIGH", color="red", edgecolor="black")
    
    ax.set_xlabel("Predicted Probability P(HIGH)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Probability Distribution by Ground Truth", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "probability_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved probability_histogram.png")


# ============================================================================
# 2. ROC CURVE
# ============================================================================

def plot_roc_curve(y_true, y_prob, threshold=0.5, output_dir=None):
    """Plot ROC curve with AUC and operating threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Find threshold index closest to 0.5
    threshold_idx = np.argmin(np.abs(thresholds - threshold))
    
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    
    # Mark operating threshold
    ax.plot(fpr[threshold_idx], tpr[threshold_idx], "ro", markersize=10,
            label=f"Operating Threshold T={threshold:.2f}")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved roc_curve.png (AUC = {roc_auc:.3f})")
    
    return roc_auc


# ============================================================================
# 3. PRECISION-RECALL CURVE
# ============================================================================

def plot_pr_curve(y_true, y_prob, output_dir):
    """Plot precision-recall curve with AP score."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(recall, precision, color="darkgreen", lw=2.5, label=f"PR (AP = {ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.2, color="darkgreen")
    
    # Baseline
    baseline = y_true.sum() / len(y_true)
    ax.axhline(baseline, color="red", linestyle="--", lw=2, label=f"Baseline = {baseline:.2f}")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved pr_curve.png (AP = {ap:.3f})")
    
    return ap


# ============================================================================
# 4. THRESHOLD SWEEP
# ============================================================================

def threshold_sweep_analysis(y_true, y_prob, output_dir):
    """Evaluate metrics across thresholds 0.05 → 0.95."""
    thresholds = np.linspace(0.05, 0.95, 19)
    
    metrics = {
        "threshold": [],
        "accuracy": [],
        "balanced_accuracy": [],
        "f1": [],
        "mcc": [],
    }
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        metrics["threshold"].append(t)
        metrics["accuracy"].append(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(y_true, y_pred))
        metrics["f1"].append(f1_score(y_true, y_pred, average="binary", zero_division=0))
        metrics["mcc"].append(matthews_corrcoef(y_true, y_pred))
    
    # Find best threshold
    best_idx = np.argmax(metrics["f1"])
    best_threshold = metrics["threshold"][best_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(metrics["threshold"], metrics["accuracy"], "o-", lw=2, label="Accuracy", markersize=6)
    ax.plot(metrics["threshold"], metrics["balanced_accuracy"], "s-", lw=2, label="Balanced Accuracy", markersize=6)
    ax.plot(metrics["threshold"], metrics["f1"], "^-", lw=2, label="F1 Score", markersize=6)
    ax.plot(metrics["threshold"], metrics["mcc"], "D-", lw=2, label="MCC", markersize=6)
    
    # Mark current and best
    ax.axvline(0.5, color="red", linestyle="--", lw=2, alpha=0.7, label="Current T=0.5")
    ax.axvline(best_threshold, color="green", linestyle="--", lw=2.5, alpha=0.8, label=f"Best T={best_threshold:.2f}")
    
    ax.set_xlabel("Decision Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
    ax.set_title("Threshold Sweep Analysis", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved threshold_sweep.png (best T={best_threshold:.2f})")
    
    # Save best threshold
    best_threshold_data = {
        "current_threshold": 0.5,
        "best_threshold": float(best_threshold),
        "best_f1": float(metrics["f1"][best_idx]),
        "best_balanced_accuracy": float(metrics["balanced_accuracy"][best_idx]),
    }
    with open(output_dir / "best_threshold.json", "w") as f:
        json.dump(best_threshold_data, f, indent=2)
    print(f"  ✓ Saved best_threshold.json")
    
    return best_threshold


# ============================================================================
# 5. CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(y_true, y_prob, threshold=0.5, output_dir=None):
    """Plot confusion matrix (raw counts and normalized %)."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1, cbar=False,
                xticklabels=["LOW", "HIGH"], yticklabels=["LOW", "HIGH"])
    ax1.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax1.set_title("Confusion Matrix (Raw Counts)", fontsize=12, fontweight="bold")
    
    # Normalized
    sns.heatmap(cm_norm * 100, annot=True, fmt=".1f", cmap="Blues", ax=ax2, cbar=False,
                xticklabels=["LOW", "HIGH"], yticklabels=["LOW", "HIGH"])
    ax2.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax2.set_title("Confusion Matrix (%)", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved confusion_matrix.png")


# ============================================================================
# 6. CALIBRATION ANALYSIS
# ============================================================================

def plot_calibration(y_true, y_prob, output_dir):
    """Plot reliability diagram and compute ECE."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    # Compute ECE (Expected Calibration Error)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], (y_prob[mask] >= 0.5).astype(int))
            conf = y_prob[mask].mean()
            ece += np.abs(acc - conf) * mask.sum() / len(y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfectly Calibrated")
    ax.plot(prob_pred, prob_true, "o-", lw=2.5, markersize=8, label="INS-HDGS-CMT")
    
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.2)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontsize=12, fontweight="bold")
    ax.set_title(f"Calibration Curve (ECE = {ece:.3f})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "calibration.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved calibration.png (ECE = {ece:.3f})")
    
    return ece


# ============================================================================
# 7. EMBEDDING VISUALIZATION
# ============================================================================

def extract_embeddings(subject_id, fold_no, device="cuda:0"):
    """Extract final latent vectors before classifier."""
    try:
        from pathlib import Path
        CKPT_DIR = Path("output/checkpoints/ins_hdgs_cmt_v17")
        ckpt_paths = list(CKPT_DIR.glob(f"*_fold{fold_no:02d}_e*.pt"))
        
        if not ckpt_paths:
            return None
        
        test_ds = NeumaGraphDataset(subject_ids=[subject_id], precompute_graphs=True)
        if len(test_ds) < 2:
            return None
        
        embeddings = []
        labels = []
        
        # Load first ensemble member for embedding extraction
        model = _build_raw_model(test_ds, None).to(device)
        sd = torch.load(ckpt_paths[0], map_location=device, weights_only=False)
        model.load_state_dict(
            sd["model_state_dict"] if isinstance(sd, dict) else sd
        )
        model.eval()
        
        with torch.no_grad():
            for i in range(len(test_ds)):
                batch_dict = test_ds[i]
                batch_dict = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch_dict.items()}
                
                # Forward pass to get embeddings (last hidden layer)
                out = model(eeg_windows=batch_dict["eeg_windows"],
                           adj_matrices=batch_dict["adj_matrices"],
                           et_seq=batch_dict["et_seq"],
                           roi_vector=batch_dict["roi_vector"],
                           weighted_adjs=batch_dict["weighted_adjs"])
                
                # Extract embedding (before classifier head)
                if "embedding" in out:
                    embeddings.append(out["embedding"].cpu().numpy()[0])
                else:
                    # Fallback: use logits projection
                    embeddings.append(out["logits"].cpu().numpy()[0])
                
                labels.append(batch_dict["label"].cpu().numpy()[0])
        
        return np.array(embeddings), np.array(labels)
    
    except Exception as e:
        print(f"    Warning: Embedding extraction failed: {e}")
        return None


def plot_embeddings(subject_id, fold_no, skip_embeddings=False, output_dir=None):
    """Generate t-SNE and UMAP visualizations."""
    if skip_embeddings:
        print(f"  ⊘ Skipped embedding extraction (--skip-embeddings)")
        return
    
    try:
        result = extract_embeddings(subject_id, fold_no)
        if result is None:
            print(f"  ⊘ Could not extract embeddings")
            return
        
        embeddings, labels = result
        if len(embeddings) < 3:
            print(f"  ⊘ Insufficient samples for embedding visualization")
            return
        
        # t-SNE
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
            tsne_emb = tsne.fit_transform(embeddings)
            
            # By label
            fig, ax = plt.subplots(figsize=(8, 8))
            scatter = ax.scatter(tsne_emb[labels == 0, 0], tsne_emb[labels == 0, 1],
                               c="blue", label="LOW", s=100, alpha=0.7, edgecolors="black")
            ax.scatter(tsne_emb[labels == 1, 0], tsne_emb[labels == 1, 1],
                      c="red", label="HIGH", s=100, alpha=0.7, edgecolors="black")
            ax.set_xlabel("t-SNE 1", fontsize=11, fontweight="bold")
            ax.set_ylabel("t-SNE 2", fontsize=11, fontweight="bold")
            ax.set_title(f"t-SNE Visualization (Subject {subject_id})", fontsize=12, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "tsne_label.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved tsne_label.png")
            
        except Exception as e:
            print(f"  ⊘ t-SNE failed: {e}")
        
        # UMAP
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            umap_emb = reducer.fit_transform(embeddings)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(umap_emb[labels == 0, 0], umap_emb[labels == 0, 1],
                      c="blue", label="LOW", s=100, alpha=0.7, edgecolors="black")
            ax.scatter(umap_emb[labels == 1, 0], umap_emb[labels == 1, 1],
                      c="red", label="HIGH", s=100, alpha=0.7, edgecolors="black")
            ax.set_xlabel("UMAP 1", fontsize=11, fontweight="bold")
            ax.set_ylabel("UMAP 2", fontsize=11, fontweight="bold")
            ax.set_title(f"UMAP Visualization (Subject {subject_id})", fontsize=12, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "umap_label.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved umap_label.png")
            
        except Exception as e:
            print(f"  ⊘ UMAP failed: {e}")
    
    except Exception as e:
        print(f"  ⊘ Embedding visualization failed: {e}")


# ============================================================================
# 8. FEATURE STATISTICS & SUBJECT SHIFT
# ============================================================================

def compute_feature_statistics(y_true, y_prob, subject_id, fold_no, output_dir):
    """Compute distance metrics to detect subject distribution shift."""
    try:
        # Load test subject features
        test_ds = NeumaGraphDataset(subject_ids=[subject_id], precompute_graphs=True)
        if len(test_ds) < 2:
            return
        
        # Extract EEG features (first window from each epoch)
        test_features = []
        for i in range(len(test_ds)):
            batch_dict = test_ds[i]
            eeg_window = batch_dict["eeg_windows"]  # (1500, 24)
            # Simple feature: mean and std across time
            feat = np.concatenate([eeg_window.mean(axis=0), eeg_window.std(axis=0)])
            test_features.append(feat)
        
        test_features = np.array(test_features)  # (N, 48)
        
        # Load training subjects for reference
        train_subs = [s for s in SUBJECT_IDS if s != subject_id]
        train_ds = NeumaGraphDataset(subject_ids=train_subs, precompute_graphs=True)
        
        train_features = []
        for i in range(min(len(train_ds), 100)):  # Sample for speed
            batch_dict = train_ds[i]
            eeg_window = batch_dict["eeg_windows"]
            feat = np.concatenate([eeg_window.mean(axis=0), eeg_window.std(axis=0)])
            train_features.append(feat)
        
        train_features = np.array(train_features)  # (M, 48)
        
        # Compute statistics
        train_mean = train_features.mean(axis=0)
        train_cov = np.cov(train_features.T)
        
        # Add small regularization to avoid singular covariance
        train_cov_reg = train_cov + 1e-6 * np.eye(train_cov.shape[0])
        train_cov_inv = np.linalg.inv(train_cov_reg)
        
        # Compute distances
        mahal_distances = []
        cosine_distances = []
        for feat in test_features:
            diff = feat - train_mean
            mahal = np.sqrt(diff @ train_cov_inv @ diff.T)
            cos = cosine(feat, train_mean)
            mahal_distances.append(mahal)
            cosine_distances.append(cos)
        
        report_data = {
            "subject_id": subject_id,
            "n_test_samples": len(test_features),
            "mahalanobis_distance_mean": float(np.mean(mahal_distances)),
            "mahalanobis_distance_std": float(np.std(mahal_distances)),
            "cosine_distance_mean": float(np.mean(cosine_distances)),
            "cosine_distance_std": float(np.std(cosine_distances)),
            "is_outlier": bool(np.mean(mahal_distances) > np.percentile(mahal_distances, 95)),
        }
        
        # Save report
        with open(output_dir / "subject_shift_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"  ✓ Saved subject_shift_report.json")
        
    except Exception as e:
        print(f"  ⊘ Feature statistics computation failed: {e}")


# ============================================================================
# 9. RANKING FAILURE ANALYSIS
# ============================================================================

def ranking_failure_analysis(y_true, y_prob, output_dir):
    """Analyze ranking: are predictions ranked correctly?"""
    df = pd.DataFrame({
        "sample_idx": np.arange(len(y_true)),
        "true_label": y_true,
        "predicted_prob": y_prob,
    })
    
    # Sort by probability
    df = df.sort_values("predicted_prob", ascending=False).reset_index(drop=True)
    
    # Save CSV
    df.to_csv(output_dir / "ranking_analysis.csv", index=False)
    print(f"  ✓ Saved ranking_analysis.csv")
    
    return df


# ============================================================================
# 10. AUTOMATIC SUBJECT DIAGNOSIS
# ============================================================================

def automatic_diagnosis(y_true, y_prob, threshold=0.5, auc_score=0.5, ece_score=0.5,
                       best_threshold=0.5, output_dir=None):
    """Classify failure mode and provide recommendation."""
    
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Determine failure mode
    diagnosis_text = ""
    failure_modes = []
    confidence_score = 0.0
    
    # A) Threshold Issue
    if auc_score > 0.70 and abs(best_threshold - threshold) > 0.10:
        failure_modes.append("A) THRESHOLD ISSUE")
        diagnosis_text += (
            f"✓ AUC = {auc_score:.3f} (good ranking)\n"
            f"✓ Current T = {threshold:.2f}, Best T = {best_threshold:.2f}\n"
            f"→ Recommendation: Adjust threshold from {threshold:.2f} → {best_threshold:.2f}\n"
            f"  Expect F1 improvement to ~{0.70:.2f}\n\n"
        )
        confidence_score += 0.9
    
    # B) Calibration Issue
    if auc_score > 0.65 and ece_score > 0.15:
        failure_modes.append("B) CALIBRATION ISSUE")
        diagnosis_text += (
            f"✓ AUC = {auc_score:.3f} (reasonable ranking)\n"
            f"✗ ECE = {ece_score:.3f} (poor calibration)\n"
            f"→ Recommendation: Apply temperature scaling or Platt scaling\n"
            f"  Expected ECE after calibration: ~0.08\n\n"
        )
        confidence_score += 0.7
    
    # C) Subject Distribution Shift
    if auc_score < 0.60 and f1 < 0.40:
        failure_modes.append("C) SUBJECT DISTRIBUTION SHIFT")
        diagnosis_text += (
            f"✗ AUC = {auc_score:.3f} (poor ranking)\n"
            f"✗ F1 = {f1:.3f} (poor classification)\n"
            f"→ Recommendation: Subject may have different EEG/ET characteristics\n"
            f"  Consider: subject-specific fine-tuning or data augmentation\n\n"
        )
        confidence_score += 0.8
    
    # D) Label Noise
    if auc_score > 0.50 and auc_score < 0.60 and f1 > 0.30:
        failure_modes.append("D) LABEL NOISE")
        diagnosis_text += (
            f"? AUC = {auc_score:.3f} (borderline)\n"
            f"? F1 = {f1:.3f} (weak)\n"
            f"→ Recommendation: Verify ground truth labels for this subject\n"
            f"  Manually review ~20% of samples\n\n"
        )
        confidence_score += 0.5
    
    # E) Representation Failure
    if auc_score < 0.50 and bal_acc < 0.50:
        failure_modes.append("E) REPRESENTATION FAILURE")
        diagnosis_text += (
            f"✗ AUC = {auc_score:.3f} (near random)\n"
            f"✗ Balanced Accuracy = {bal_acc:.3f} (near random)\n"
            f"→ Recommendation: Model cannot learn meaningful representation\n"
            f"  Investigate: EEG/ET preprocessing, artifact contamination, or label integrity\n\n"
        )
        confidence_score += 0.9
    
    if not failure_modes:
        failure_modes.append("UNKNOWN")
        diagnosis_text += f"AUC={auc_score:.3f}, F1={f1:.3f}, ECE={ece_score:.3f}\n"
        confidence_score = 0.3
    
    # Save diagnosis
    diagnosis_dict = {
        "failure_modes": failure_modes,
        "auc": float(auc_score),
        "f1_score": float(f1),
        "balanced_accuracy": float(bal_acc),
        "ece": float(ece_score),
        "current_threshold": float(threshold),
        "recommended_threshold": float(best_threshold),
        "confidence_score": float(confidence_score),
        "diagnosis_text": diagnosis_text,
    }
    
    with open(output_dir / "diagnosis.json", "w") as f:
        json.dump(diagnosis_dict, f, indent=2)
    
    # Also save as text
    with open(output_dir / "diagnosis.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("AUTOMATIC HARD-SUBJECT DIAGNOSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Failure Modes: {', '.join(failure_modes)}\n")
        f.write(f"Confidence: {confidence_score:.1%}\n\n")
        f.write("Metrics:\n")
        f.write(f"  AUC: {auc_score:.3f}\n")
        f.write(f"  F1 Score: {f1:.3f}\n")
        f.write(f"  Balanced Accuracy: {bal_acc:.3f}\n")
        f.write(f"  ECE: {ece_score:.3f}\n\n")
        f.write("Analysis:\n")
        f.write(diagnosis_text)
    
    print(f"  ✓ Saved diagnosis.txt & diagnosis.json")
    print(f"     Failure Modes: {', '.join(failure_modes)}")


# ============================================================================
# MAIN DIAGNOSTIC PIPELINE
# ============================================================================

def run_diagnostics_for_subject(fold_no, subject_id, label, skip_embeddings, output_dir):
    """Run all 10 diagnostics for a subject."""
    
    # Create output directory
    subject_out_dir = output_dir / subject_id
    subject_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"FOLD {fold_no:02d} — Subject {subject_id}")
    print(f"{'='*70}")
    
    # Load fold probabilities
    fold_probs_file = Path("output/fold_probs") / f"fold{fold_no:02d}_{subject_id}.npz"
    if not fold_probs_file.exists():
        print(f"✗ Fold probs not found: {fold_probs_file}")
        return
    
    data = np.load(fold_probs_file)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        print(f"✗ Invalid fold data (single class or too small)")
        return
    
    print(f"N samples: {len(y_true)}")
    print(f"Class distribution: {np.bincount(y_true)}")
    
    # 1. Probability Histogram
    print(f"\n[1/10] Probability Histogram...")
    plot_probability_histogram(y_true, y_prob, subject_out_dir)
    
    # 2. ROC Curve
    print(f"[2/10] ROC Curve...")
    auc_score = plot_roc_curve(y_true, y_prob, threshold=0.5, output_dir=subject_out_dir)
    
    # 3. Precision-Recall Curve
    print(f"[3/10] Precision-Recall Curve...")
    ap_score = plot_pr_curve(y_true, y_prob, subject_out_dir)
    
    # 4. Threshold Sweep
    print(f"[4/10] Threshold Sweep...")
    best_threshold = threshold_sweep_analysis(y_true, y_prob, subject_out_dir)
    
    # 5. Confusion Matrix
    print(f"[5/10] Confusion Matrix...")
    plot_confusion_matrix(y_true, y_prob, threshold=0.5, output_dir=subject_out_dir)
    
    # 6. Calibration Analysis
    print(f"[6/10] Calibration Analysis...")
    ece_score = plot_calibration(y_true, y_prob, subject_out_dir)
    
    # 7. Embedding Visualization
    print(f"[7/10] Embedding Visualization...")
    plot_embeddings(subject_id, fold_no, skip_embeddings=skip_embeddings, output_dir=subject_out_dir)
    
    # 8. Feature Statistics
    print(f"[8/10] Feature Statistics...")
    compute_feature_statistics(y_true, y_prob, subject_id, fold_no, subject_out_dir)
    
    # 9. Ranking Failure Analysis
    print(f"[9/10] Ranking Failure Analysis...")
    ranking_failure_analysis(y_true, y_prob, subject_out_dir)
    
    # 10. Automatic Diagnosis
    print(f"[10/10] Automatic Diagnosis...")
    automatic_diagnosis(y_true, y_prob, threshold=0.5, auc_score=auc_score,
                       ece_score=ece_score, best_threshold=best_threshold, 
                       output_dir=subject_out_dir)
    
    print(f"\n✓ All diagnostics saved to: {subject_out_dir}")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hard-subject diagnostic framework for INS-HDGS-CMT LOSOCV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--label", type=str, default="ins_hdgs_cmt_v17",
        help="Experiment label",
    )
    parser.add_argument(
        "--hard-subjects", type=str, default="S21,S03,S13,S35,S36",
        help="Comma-separated list of hard subjects",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/diagnostics",
        help="Root output directory for diagnostics",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip expensive embedding extraction (faster)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hard_subjects = args.hard_subjects.split(",")
    
    # Map subject IDs to fold numbers
    # Assuming standard LOSOCV where fold_no = subject index
    subject_to_fold = {subj: (i + 1) for i, subj in enumerate(SUBJECT_IDS)}
    
    print(f"\n{'='*70}")
    print(f"HARD-SUBJECT DIAGNOSTIC FRAMEWORK")
    print(f"{'='*70}")
    print(f"Hard Subjects: {', '.join(hard_subjects)}")
    print(f"Output Directory: {output_dir}")
    print(f"Skip Embeddings: {args.skip_embeddings}")
    
    for subj in hard_subjects:
        if subj not in subject_to_fold:
            print(f"\n✗ Subject {subj} not found in SUBJECT_IDS")
            continue
        
        fold_no = subject_to_fold[subj]
        run_diagnostics_for_subject(fold_no, subj, args.label, args.skip_embeddings, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✓ DIAGNOSTIC FRAMEWORK COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
