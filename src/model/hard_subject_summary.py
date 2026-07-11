"""
================================================================
HARD-SUBJECT SUMMARY & AGGREGATION REPORT
================================================================
Aggregate diagnostic results across all hard subjects.

Generates:
  - Summary statistics table
  - Failure mode distribution
  - Recommendation prioritization
  - Cross-subject comparison plots

Usage:
  python hard_subject_summary.py \\
      --diagnostics-dir output/diagnostics \\
      --output output/diagnostics_summary.txt

================================================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_subject_diagnostics(subject_id, diag_dir):
    """Load all diagnostic files for a subject."""
    subject_dir = diag_dir / subject_id
    
    if not subject_dir.exists():
        return None
    
    try:
        diagnosis = json.load(open(subject_dir / "diagnosis.json"))
        best_threshold = json.load(open(subject_dir / "best_threshold.json"))
        
        subject_shift = {}
        if (subject_dir / "subject_shift_report.json").exists():
            subject_shift = json.load(open(subject_dir / "subject_shift_report.json"))
        
        ranking = pd.read_csv(subject_dir / "ranking_analysis.csv", nrows=5)
        
        return {
            "subject_id": subject_id,
            "diagnosis": diagnosis,
            "best_threshold": best_threshold,
            "subject_shift": subject_shift,
            "ranking_top5": ranking,
        }
    except Exception as e:
        print(f"Warning: Could not load diagnostics for {subject_id}: {e}")
        return None


def generate_summary_report(diagnostics_dir, output_file):
    """Generate aggregated summary report."""
    
    diag_dir = Path(diagnostics_dir)
    hard_subjects = ["S21", "S03", "S13", "S35", "S36"]
    
    print(f"\nLoading diagnostics from: {diag_dir}")
    
    all_data = []
    for subj in hard_subjects:
        data = load_subject_diagnostics(subj, diag_dir)
        if data:
            all_data.append(data)
    
    if not all_data:
        print("✗ No diagnostic data found")
        return
    
    # Generate report
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HARD-SUBJECT DIAGNOSTIC SUMMARY\n")
        f.write("INS-HDGS-CMT LOSOCV Pipeline\n")
        f.write("=" * 80 + "\n\n")
        
        # Table of contents
        f.write("TABLE OF CONTENTS\n")
        f.write("─" * 80 + "\n")
        f.write("1. Executive Summary\n")
        f.write("2. Per-Subject Breakdown\n")
        f.write("3. Failure Mode Distribution\n")
        f.write("4. Recommendations by Priority\n")
        f.write("5. Detailed Subject Profiles\n")
        f.write("\n" * 2)
        
        # ── SECTION 1: Executive Summary ────────────────────────────────────
        f.write("=" * 80 + "\n")
        f.write("1. EXECUTIVE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        auc_scores = [d["diagnosis"]["auc"] for d in all_data]
        f1_scores = [d["diagnosis"]["f1_score"] for d in all_data]
        ece_scores = [d["diagnosis"]["ece"] for d in all_data]
        
        f.write(f"Total Hard Subjects: {len(all_data)}\n")
        f.write(f"\nPerformance Summary:\n")
        f.write(f"  Mean AUC:                 {np.mean(auc_scores):.3f} (±{np.std(auc_scores):.3f})\n")
        f.write(f"  Mean F1 Score:            {np.mean(f1_scores):.3f} (±{np.std(f1_scores):.3f})\n")
        f.write(f"  Mean ECE:                 {np.mean(ece_scores):.3f} (±{np.std(ece_scores):.3f})\n")
        f.write(f"\n  AUC Range:                [{min(auc_scores):.3f}, {max(auc_scores):.3f}]\n")
        f.write(f"  F1 Range:                 [{min(f1_scores):.3f}, {max(f1_scores):.3f}]\n")
        f.write(f"  ECE Range:                [{min(ece_scores):.3f}, {max(ece_scores):.3f}]\n")
        f.write("\n" * 2)
        
        # ── SECTION 2: Per-Subject Breakdown ────────────────────────────────
        f.write("=" * 80 + "\n")
        f.write("2. PER-SUBJECT BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")
        
        summary_data = []
        for d in all_data:
            diag = d["diagnosis"]
            summary_data.append({
                "Subject": d["subject_id"],
                "AUC": f"{diag['auc']:.3f}",
                "F1": f"{diag['f1_score']:.3f}",
                "Bal.Acc": f"{diag['balanced_accuracy']:.3f}",
                "ECE": f"{diag['ece']:.3f}",
                "Curr.T": f"{diag['current_threshold']:.2f}",
                "Best.T": f"{diag['recommended_threshold']:.2f}",
                "Confidence": f"{diag['confidence_score']:.1%}",
                "Failure Modes": " | ".join(diag["failure_modes"]),
            })
        
        summary_df = pd.DataFrame(summary_data)
        f.write(summary_df.to_string(index=False))
        f.write("\n\n" * 2)
        
        # ── SECTION 3: Failure Mode Distribution ────────────────────────────
        f.write("=" * 80 + "\n")
        f.write("3. FAILURE MODE DISTRIBUTION\n")
        f.write("=" * 80 + "\n\n")
        
        failure_mode_counts = {}
        for d in all_data:
            for mode in d["diagnosis"]["failure_modes"]:
                failure_mode_counts[mode] = failure_mode_counts.get(mode, 0) + 1
        
        sorted_modes = sorted(failure_mode_counts.items(), key=lambda x: x[1], reverse=True)
        for mode, count in sorted_modes:
            pct = 100 * count / len(all_data)
            f.write(f"  {mode:<30} {count:>2}  ({pct:>5.1f}%)\n")
        
        f.write("\n" * 2)
        
        # ── SECTION 4: Recommendations by Priority ──────────────────────────
        f.write("=" * 80 + "\n")
        f.write("4. RECOMMENDATIONS BY PRIORITY\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by confidence score descending
        sorted_data = sorted(all_data, key=lambda x: x["diagnosis"]["confidence_score"], reverse=True)
        
        for rank, d in enumerate(sorted_data, 1):
            diag = d["diagnosis"]
            f.write(f"Priority {rank}: {d['subject_id']}\n")
            f.write(f"  Confidence: {diag['confidence_score']:.1%}\n")
            f.write(f"  Primary Issues: {', '.join(diag['failure_modes'])}\n")
            f.write(f"  Action:\n")
            
            # Generate specific action
            if "A) THRESHOLD ISSUE" in diag["failure_modes"]:
                f.write(f"    → Adjust threshold from {diag['current_threshold']:.2f} to "
                       f"{diag['recommended_threshold']:.2f}\n")
            if "B) CALIBRATION ISSUE" in diag["failure_modes"]:
                f.write(f"    → Apply temperature scaling (current ECE: {diag['ece']:.3f})\n")
            if "C) SUBJECT DISTRIBUTION SHIFT" in diag["failure_modes"]:
                f.write(f"    → Subject-specific fine-tuning (AUC: {diag['auc']:.3f})\n")
            if "D) LABEL NOISE" in diag["failure_modes"]:
                f.write(f"    → Manually review ground truth labels\n")
            if "E) REPRESENTATION FAILURE" in diag["failure_modes"]:
                f.write(f"    → Investigate EEG/ET preprocessing and artifact contamination\n")
            
            f.write("\n")
        
        f.write("\n" * 2)
        
        # ── SECTION 5: Detailed Subject Profiles ────────────────────────────
        f.write("=" * 80 + "\n")
        f.write("5. DETAILED SUBJECT PROFILES\n")
        f.write("=" * 80 + "\n\n")
        
        for d in all_data:
            diag = d["diagnosis"]
            f.write(f"┌─ {d['subject_id']} ─────────────────────────────────────────\n")
            f.write(f"│ Metrics:\n")
            f.write(f"│   AUC: {diag['auc']:.3f}\n")
            f.write(f"│   F1 Score: {diag['f1_score']:.3f}\n")
            f.write(f"│   Balanced Accuracy: {diag['balanced_accuracy']:.3f}\n")
            f.write(f"│   ECE: {diag['ece']:.3f}\n")
            f.write(f"│\n")
            f.write(f"│ Threshold Analysis:\n")
            f.write(f"│   Current: {diag['current_threshold']:.2f}\n")
            f.write(f"│   Recommended: {diag['recommended_threshold']:.2f}\n")
            f.write(f"│\n")
            f.write(f"│ Diagnosis:\n")
            for line in diag["diagnosis_text"].split("\n"):
                if line.strip():
                    f.write(f"│   {line}\n")
            f.write(f"└─────────────────────────────────────────────────────\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Summary report saved to: {output_file}")


def generate_comparison_plots(diagnostics_dir, output_dir):
    """Generate cross-subject comparison visualizations."""
    
    diag_dir = Path(diagnostics_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    hard_subjects = ["S21", "S03", "S13", "S35", "S36"]
    
    all_data = []
    for subj in hard_subjects:
        data = load_subject_diagnostics(subj, diag_dir)
        if data:
            all_data.append(data)
    
    if not all_data:
        return
    
    # Extract metrics
    subjects = [d["subject_id"] for d in all_data]
    aucs = [d["diagnosis"]["auc"] for d in all_data]
    f1s = [d["diagnosis"]["f1_score"] for d in all_data]
    ece_scores = [d["diagnosis"]["ece"] for d in all_data]
    bal_accs = [d["diagnosis"]["balanced_accuracy"] for d in all_data]
    
    # Plot 1: AUC Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(subjects, aucs, color=["green" if x > 0.70 else "orange" if x > 0.60 else "red" for x in aucs],
                  edgecolor="black", linewidth=1.5, alpha=0.7)
    ax.axhline(0.70, color="green", linestyle="--", lw=2, alpha=0.5, label="Good (0.70)")
    ax.axhline(0.60, color="orange", linestyle="--", lw=2, alpha=0.5, label="Acceptable (0.60)")
    ax.axhline(0.50, color="red", linestyle="--", lw=2, alpha=0.5, label="Random (0.50)")
    
    ax.set_ylabel("AUC", fontsize=12, fontweight="bold")
    ax.set_title("Hard Subject AUC Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim([0.4, 1.0])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    for i, (subj, auc) in enumerate(zip(subjects, aucs)):
        ax.text(i, auc + 0.02, f"{auc:.3f}", ha="center", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_auc.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comparison_auc.png")
    
    # Plot 2: Multi-metric comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].bar(subjects, aucs, color="steelblue", edgecolor="black", linewidth=1.5)
    axes[0, 0].set_ylabel("AUC", fontweight="bold")
    axes[0, 0].set_title("AUC Scores", fontweight="bold")
    axes[0, 0].set_ylim([0.4, 1.0])
    axes[0, 0].grid(axis="y", alpha=0.3)
    
    axes[0, 1].bar(subjects, f1s, color="darkorange", edgecolor="black", linewidth=1.5)
    axes[0, 1].set_ylabel("F1 Score", fontweight="bold")
    axes[0, 1].set_title("F1 Scores", fontweight="bold")
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].grid(axis="y", alpha=0.3)
    
    axes[1, 0].bar(subjects, ece_scores, color="darkred", edgecolor="black", linewidth=1.5)
    axes[1, 0].set_ylabel("ECE", fontweight="bold")
    axes[1, 0].set_title("Calibration Error (ECE)", fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=0.3)
    
    axes[1, 1].bar(subjects, bal_accs, color="darkgreen", edgecolor="black", linewidth=1.5)
    axes[1, 1].set_ylabel("Balanced Accuracy", fontweight="bold")
    axes[1, 1].set_title("Balanced Accuracy", fontweight="bold")
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].grid(axis="y", alpha=0.3)
    
    plt.suptitle("Hard Subject Performance Metrics", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comparison_metrics.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hard-subject diagnostic summary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--diagnostics-dir", type=str, default="output/diagnostics",
        help="Directory containing individual subject diagnostics",
    )
    parser.add_argument(
        "--output", type=str, default="output/hard_subjects_summary.txt",
        help="Output file for summary report",
    )
    parser.add_argument(
        "--comparison-dir", type=str, default="output",
        help="Directory for comparison plots",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("HARD-SUBJECT SUMMARY & AGGREGATION")
    print("=" * 80)
    
    generate_summary_report(args.diagnostics_dir, args.output)
    generate_comparison_plots(args.diagnostics_dir, args.comparison_dir)
    
    print("\n" + "=" * 80)
    print("✓ SUMMARY GENERATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
