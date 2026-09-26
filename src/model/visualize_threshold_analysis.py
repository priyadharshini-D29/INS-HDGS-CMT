"""
================================================================
THRESHOLD ANALYSIS VISUALIZATION
================================================================
Generate visualizations for threshold optimization results.

Plots:
  1. threshold_distribution.png   - Histogram of optimal thresholds
  2. balacc_improvement.png       - Balanced accuracy gains
  3. mcc_improvement.png          - MCC gains
  4. metrics_comparison.png       - All metrics side-by-side
  5. threshold_vs_gain.png        - Threshold vs improvement

Usage:
  python visualize_threshold_analysis.py \\
      --results-csv output/threshold_analysis/thresholds_per_fold.csv \\
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
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"


def plot_threshold_distribution(df_ok, output_dir):
    """Plot histogram of optimal thresholds."""
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    thresholds = df_ok["threshold_opt"].values
    
    ax.hist(thresholds, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(thresholds.mean(), color="red", linestyle="--", lw=2.5,
               label=f"Mean = {thresholds.mean():.3f}")
    ax.axvline(0.5, color="orange", linestyle="--", lw=2.5,
               label="Fixed (0.50)")
    
    ax.set_xlabel("Optimal Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Distribution of Validation-Optimized Thresholds", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved threshold_distribution.png")


def plot_balacc_improvement(df_ok, output_dir):
    """Plot balanced accuracy gains."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fold_ids = np.arange(len(df_ok))
    
    # Absolute values
    ax1.bar(fold_ids, df_ok["bal_acc_fixed"], alpha=0.6, label="Fixed (0.5)",
            color="orange", edgecolor="black", linewidth=0.8)
    ax1.bar(fold_ids, df_ok["bal_acc_opt"], alpha=0.6, label="Optimized",
            color="green", edgecolor="black", linewidth=0.8)
    ax1.set_xlabel("Fold", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Balanced Accuracy", fontsize=11, fontweight="bold")
    ax1.set_title("Balanced Accuracy per Fold", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # Gains
    gains = df_ok["bal_acc_gain"].values
    colors = ["green" if g > 0 else "red" for g in gains]
    ax2.bar(fold_ids, gains, color=colors, edgecolor="black", linewidth=0.8, alpha=0.7)
    ax2.axhline(0, color="black", linestyle="-", lw=1)
    ax2.axhline(gains.mean(), color="darkgreen", linestyle="--", lw=2,
                label=f"Mean = {gains.mean():+.3f}")
    ax2.set_xlabel("Fold", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Improvement (ΔBalAcc)", fontsize=11, fontweight="bold")
    ax2.set_title("Balanced Accuracy Improvement", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "balacc_improvement.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved balacc_improvement.png")


def plot_mcc_improvement(df_ok, output_dir):
    """Plot MCC gains."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fold_ids = np.arange(len(df_ok))
    
    # Absolute values
    ax1.bar(fold_ids, df_ok["mcc_fixed"], alpha=0.6, label="Fixed (0.5)",
            color="orange", edgecolor="black", linewidth=0.8)
    ax1.bar(fold_ids, df_ok["mcc_opt"], alpha=0.6, label="Optimized",
            color="purple", edgecolor="black", linewidth=0.8)
    ax1.set_xlabel("Fold", fontsize=11, fontweight="bold")
    ax1.set_ylabel("MCC", fontsize=11, fontweight="bold")
    ax1.set_title("Matthews Correlation Coefficient per Fold", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim([-1, 1])
    
    # Gains
    gains = df_ok["mcc_gain"].values
    colors = ["green" if g > 0 else "red" for g in gains]
    ax2.bar(fold_ids, gains, color=colors, edgecolor="black", linewidth=0.8, alpha=0.7)
    ax2.axhline(0, color="black", linestyle="-", lw=1)
    ax2.axhline(gains.mean(), color="darkgreen", linestyle="--", lw=2,
                label=f"Mean = {gains.mean():+.3f}")
    ax2.set_xlabel("Fold", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Improvement (ΔMCC)", fontsize=11, fontweight="bold")
    ax2.set_title("MCC Improvement", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mcc_improvement.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved mcc_improvement.png")


def plot_metrics_comparison(df_ok, output_dir):
    """Plot all metrics comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    fold_ids = np.arange(len(df_ok))
    
    # Balanced Accuracy
    ax = axes[0, 0]
    ax.plot(fold_ids, df_ok["bal_acc_fixed"], "o-", lw=2, markersize=5,
            label="Fixed (0.5)", color="orange")
    ax.plot(fold_ids, df_ok["bal_acc_opt"], "s-", lw=2, markersize=5,
            label="Optimized", color="green")
    ax.set_ylabel("Balanced Accuracy", fontweight="bold")
    ax.set_title("Balanced Accuracy per Fold", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    # MCC
    ax = axes[0, 1]
    ax.plot(fold_ids, df_ok["mcc_fixed"], "o-", lw=2, markersize=5,
            label="Fixed (0.5)", color="orange")
    ax.plot(fold_ids, df_ok["mcc_opt"], "s-", lw=2, markersize=5,
            label="Optimized", color="purple")
    ax.set_ylabel("MCC", fontweight="bold")
    ax.set_title("MCC per Fold", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([-1, 1])
    
    # F1
    ax = axes[1, 0]
    ax.plot(fold_ids, df_ok["f1_fixed"], "o-", lw=2, markersize=5,
            label="Fixed (0.5)", color="orange")
    ax.plot(fold_ids, df_ok["f1_opt"], "s-", lw=2, markersize=5,
            label="Optimized", color="darkgreen")
    ax.set_xlabel("Fold", fontweight="bold")
    ax.set_ylabel("F1 Score", fontweight="bold")
    ax.set_title("F1 Score per Fold", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Accuracy
    ax = axes[1, 1]
    ax.plot(fold_ids, df_ok["accuracy_fixed"], "o-", lw=2, markersize=5,
            label="Fixed (0.5)", color="orange")
    ax.plot(fold_ids, df_ok["accuracy_opt"], "s-", lw=2, markersize=5,
            label="Optimized", color="blue")
    ax.set_xlabel("Fold", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title("Accuracy per Fold", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.suptitle("Threshold Optimization: All Metrics Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved metrics_comparison.png")


def plot_threshold_vs_gain(df_ok, output_dir):
    """Scatter plot: threshold vs gain."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Threshold vs BalAcc Gain
    scatter1 = ax1.scatter(df_ok["threshold_opt"], df_ok["bal_acc_gain"],
                           c=df_ok["bal_acc_gain"], cmap="RdYlGn", s=100,
                           edgecolors="black", linewidth=0.8)
    ax1.axhline(0, color="red", linestyle="--", lw=1.5, alpha=0.5)
    ax1.set_xlabel("Optimal Threshold", fontsize=11, fontweight="bold")
    ax1.set_ylabel("BalAcc Improvement", fontsize=11, fontweight="bold")
    ax1.set_title("Threshold vs Balanced Accuracy Gain", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label("Gain", fontweight="bold")
    
    # Threshold vs MCC Gain
    scatter2 = ax2.scatter(df_ok["threshold_opt"], df_ok["mcc_gain"],
                           c=df_ok["mcc_gain"], cmap="RdYlGn", s=100,
                           edgecolors="black", linewidth=0.8)
    ax2.axhline(0, color="red", linestyle="--", lw=1.5, alpha=0.5)
    ax2.set_xlabel("Optimal Threshold", fontsize=11, fontweight="bold")
    ax2.set_ylabel("MCC Improvement", fontsize=11, fontweight="bold")
    ax2.set_title("Threshold vs MCC Gain", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label("Gain", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_vs_gain.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved threshold_vs_gain.png")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize threshold optimization results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-csv", type=str, default="output/threshold_analysis/thresholds_per_fold.csv",
        help="Path to results CSV",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/threshold_analysis",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    results_csv = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("THRESHOLD ANALYSIS VISUALIZATION")
    print("="*70)
    
    if not results_csv.exists():
        print(f"✗ Results file not found: {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    df_ok = df[df["status"] == "OK"].copy()
    
    if len(df_ok) == 0:
        print("✗ No valid results to visualize")
        return
    
    print(f"Visualizing {len(df_ok)} folds\n")
    
    # Generate plots
    plot_threshold_distribution(df_ok, output_dir)
    plot_balacc_improvement(df_ok, output_dir)
    plot_mcc_improvement(df_ok, output_dir)
    plot_metrics_comparison(df_ok, output_dir)
    plot_threshold_vs_gain(df_ok, output_dir)
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
