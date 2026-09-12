"""
================================================================
NEUMA Phase 8 — Comprehensive Results Visualizer
================================================================
Generates all publication-ready plots from saved run artifacts:

  1. GradCAM topomap          — electrode saliency (averaged across folds)
  2. EEG Connectivity heatmap — mean weighted adjacency across all epochs
  3. Band power comparison     — HIGH vs LOW engagement per frequency band
  4. ROI attention bar chart   — eye-tracking spatial attention
  5. GAT attention heatmap     — graph edge importance (C × C)
  6. Per-fold performance      — accuracy / AUC bar chart across subjects
  7. Subject difficulty scatter— accuracy vs test set size
  8. Confusion matrix          — already in plots/, re-rendered cleanly
  9. LOSOCV summary bar        — mean ± std across metrics

Usage
-----
  python visualize_results.py
  python visualize_results.py --label ins_hdgs_cmt --out output/plots/ins_hdgs_cmt
================================================================
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Parent dir also needed: this module does `from model.inference...`
# as an absolute package import (see compute_epoch_graphs/compute_node_features below).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, path: Path, dpi: int = 150):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    try:
        display = path.relative_to(_HERE)
    except ValueError:
        display = path
    print(f"  Saved → {display}")


# ── 1. GradCAM Topomap ────────────────────────────────────────────────────────

def plot_gradcam_topomap(explain_dir: Path, out_dir: Path, label: str):
    files = sorted(explain_dir.glob("gradcam_fold*.npy"))
    if not files:
        print("  [SKIP] No GradCAM files found — run with --explain first")
        return

    maps = [np.load(f) for f in files]
    mean_sal = np.mean(maps, axis=0)           # (C,)
    mean_sal = (mean_sal - mean_sal.min()) / (mean_sal.max() - mean_sal.min() + 1e-10)
    C = len(mean_sal)

    from explainability.topomap import save_topomap
    from data.channel_harmonizer import CANONICAL_CHANNELS
    ch_names = list(CANONICAL_CHANNELS)[:C]
    save_topomap(mean_sal, ch_names,
                 out_dir / f"{label}_gradcam_topomap.png",
                 title=f"GradCAM Electrode Saliency\n(mean over {len(files)} folds)")
    print(f"  GradCAM: {len(files)} folds, peak electrode = ch{int(mean_sal.argmax())}")


# ── 2. EEG Connectivity Heatmap ───────────────────────────────────────────────

def plot_connectivity_heatmap(out_dir: Path, label: str):
    from config.settings import SUBJECT_IDS, PHASE3_DIR
    from data.dataset import _load_precomputed_engagement, _normalize_subject_id
    from model.inference.graphs.graph_builder import compute_epoch_graphs
    from config.settings import N_WINDOWS, EEG_SR, CONN_METHOD, CONN_THRESHOLD

    # Per-subject dirs live at PHASE3_DIR.parent / sid / "output"
    subj_base = PHASE3_DIR.parent
    hi_adjs, lo_adjs = [], []

    for sid in SUBJECT_IDS:
        sid = _normalize_subject_id(sid)
        subj_dir = subj_base / sid / "output" / "epochs"
        result = _load_precomputed_engagement(sid, subj_dir)
        if result is None:
            continue
        eeg_list, _, labels = result
        for eeg, lbl in zip(eeg_list, labels):
            try:
                _, _, wadj = compute_epoch_graphs(
                    eeg, n_windows=N_WINDOWS, fs=EEG_SR,
                    conn_method=CONN_METHOD, threshold=CONN_THRESHOLD
                )
                mean_adj = wadj.mean(axis=0)     # (C, C)
                if lbl == 1:
                    hi_adjs.append(mean_adj)
                else:
                    lo_adjs.append(mean_adj)
            except Exception:
                pass

    if not hi_adjs or not lo_adjs:
        print("  [SKIP] No connectivity data loaded")
        return

    hi_mean = np.mean(hi_adjs, axis=0)
    lo_mean = np.mean(lo_adjs, axis=0)
    diff    = hi_mean - lo_mean

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mat, title, cmap in zip(
        axes,
        [lo_mean, hi_mean, diff],
        ["LOW Engagement\n(mean adjacency)", "HIGH Engagement\n(mean adjacency)",
         "HIGH − LOW\n(connectivity difference)"],
        ["Blues", "Reds", "RdBu_r"]
    ):
        im = ax.imshow(mat, cmap=cmap, vmin=mat.min(), vmax=mat.max(), aspect="auto")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Electrode")
        ax.set_ylabel("Electrode")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("EEG Functional Connectivity", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / f"{label}_connectivity_heatmap.png")
    print(f"  Connectivity: HIGH n={len(hi_adjs)}, LOW n={len(lo_adjs)} epochs")


# ── 3. Band Power Comparison ──────────────────────────────────────────────────

def plot_band_power_comparison(out_dir: Path, label: str):
    from config.settings import SUBJECT_IDS, PHASE3_DIR, EEG_SR
    from data.dataset import _load_precomputed_engagement, _normalize_subject_id
    from model.inference.graphs.graph_builder import compute_node_features

    subj_base = PHASE3_DIR.parent
    bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    hi_powers, lo_powers = [], []

    for sid in SUBJECT_IDS:
        sid = _normalize_subject_id(sid)
        subj_dir = subj_base / sid / "output" / "epochs"
        result = _load_precomputed_engagement(sid, subj_dir)
        if result is None:
            continue
        eeg_list, _, labels = result
        for eeg, lbl in zip(eeg_list, labels):
            try:
                feat = compute_node_features(eeg, fs=EEG_SR)  # (C, 5)
                mean_power = feat.mean(axis=0)                  # (5,) avg over channels
                if lbl == 1:
                    hi_powers.append(mean_power)
                else:
                    lo_powers.append(mean_power)
            except Exception:
                pass

    if not hi_powers or not lo_powers:
        print("  [SKIP] No band power data loaded")
        return

    hi = np.array(hi_powers)  # (N_hi, 5)
    lo = np.array(lo_powers)  # (N_lo, 5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean ± sem bar chart
    ax = axes[0]
    x = np.arange(5)
    w = 0.35
    hi_m, hi_e = hi.mean(0), hi.std(0) / np.sqrt(len(hi))
    lo_m, lo_e = lo.mean(0), lo.std(0) / np.sqrt(len(lo))

    ax.bar(x - w/2, lo_m, w, yerr=lo_e, label="LOW", color="#4878CF", alpha=0.85,
           capsize=4, error_kw={"elinewidth": 1.5})
    ax.bar(x + w/2, hi_m, w, yerr=hi_e, label="HIGH", color="#D65F5F", alpha=0.85,
           capsize=4, error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=11)
    ax.set_ylabel("Relative Band Power")
    ax.set_title("Band Power: HIGH vs LOW Engagement", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: Cohen's d effect size per band
    ax2 = axes[1]
    pooled_std = np.sqrt((hi.var(0) * (len(hi)-1) + lo.var(0) * (len(lo)-1))
                         / (len(hi) + len(lo) - 2))
    cohen_d = (hi_m - lo_m) / (pooled_std + 1e-10)
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in cohen_d]
    ax2.bar(x, cohen_d, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=1)
    ax2.axhline( 0.2, color="gray", lw=1, ls="--", label="small (d=0.2)")
    ax2.axhline(-0.2, color="gray", lw=1, ls="--")
    ax2.axhline( 0.5, color="gray", lw=1, ls="-.", label="medium (d=0.5)")
    ax2.axhline(-0.5, color="gray", lw=1, ls="-.")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands, fontsize=11)
    ax2.set_ylabel("Cohen's d (HIGH − LOW)")
    ax2.set_title("Effect Size per Band", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("EEG Band Power Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / f"{label}_band_power.png")


# ── 4. ROI Attention ─────────────────────────────────────────────────────────

def plot_roi_attention(explain_dir: Path, out_dir: Path, label: str):
    files = sorted(explain_dir.glob("roi_attn_fold*.npy"))
    if not files:
        print("  [SKIP] No ROI attention files")
        return

    maps = np.stack([np.load(f).squeeze() for f in files])   # (F, N_rois)
    mean_attn = maps.mean(axis=0)
    mean_attn = mean_attn / (mean_attn.sum() + 1e-10)

    from config.settings import N_ROIS, GRID_COLS, GRID_ROWS
    n_rois = len(mean_attn)
    grid = mean_attn.reshape(GRID_ROWS, GRID_COLS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(grid, cmap="hot", aspect="auto")
    axes[0].set_title(f"ET ROI Attention Heatmap\n(mean {len(files)} folds)",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Grid Column")
    axes[0].set_ylabel("Grid Row")
    plt.colorbar(im, ax=axes[0], shrink=0.8, label="Attention weight")

    axes[1].bar(range(n_rois), mean_attn, color="steelblue", alpha=0.85)
    axes[1].set_xlabel("ROI Index")
    axes[1].set_ylabel("Mean Attention Weight")
    axes[1].set_title("Per-ROI Eye-Tracking Attention", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    top3 = np.argsort(mean_attn)[-3:][::-1]
    for r in top3:
        axes[1].text(r, mean_attn[r] + 0.002, f"ROI{r}", ha="center",
                     fontsize=8, color="darkred")

    plt.tight_layout()
    _save(fig, out_dir / f"{label}_roi_attention.png")


# ── 5. GAT Attention Heatmap ─────────────────────────────────────────────────

def plot_gat_attention(explain_dir: Path, out_dir: Path, label: str):
    files_l1 = sorted(explain_dir.glob("graph_layer1_attn_fold*.npy"))
    files_l2 = sorted(explain_dir.glob("graph_layer2_attn_fold*.npy"))
    all_files = files_l1 or files_l2 or sorted(explain_dir.glob("graph_*_fold*.npy"))
    if not all_files:
        print("  [SKIP] No GAT attention files")
        return

    mats = []
    for f in all_files:
        m = np.load(f)
        if m.ndim == 3:
            m = m.mean(axis=0)
        if m.ndim == 2:
            mats.append(m)

    if not mats:
        return

    mean_gat = np.mean(mats, axis=0)
    np.fill_diagonal(mean_gat, 0)   # remove self-loops for clarity
    mean_gat = mean_gat / (mean_gat.max() + 1e-10)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mean_gat, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"GAT Edge Importance\n(mean over {len(mats)} folds)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Target Electrode")
    ax.set_ylabel("Source Electrode")
    plt.colorbar(im, ax=ax, label="Normalised attention weight")
    plt.tight_layout()
    _save(fig, out_dir / f"{label}_gat_attention.png")


# ── 6. Per-Fold Performance Bar Chart ────────────────────────────────────────

def plot_per_fold_performance(metrics_dir: Path, out_dir: Path, label: str):
    # CSV lives inside metrics_dir itself, not one level up
    csv_path = metrics_dir / f"losocv_{label}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df = df.sort_values("fold").reset_index(drop=True)

    metrics_to_plot = [c for c in ["accuracy", "roc_auc", "mcc", "balanced_acc"]
                       if c in df.columns]
    colors = {"accuracy": "#3498db", "roc_auc": "#2ecc71",
              "mcc": "#e67e22", "balanced_acc": "#9b59b6"}
    labels = {"accuracy": "Accuracy", "roc_auc": "ROC-AUC",
              "mcc": "MCC", "balanced_acc": "Balanced Acc"}

    fig, ax = plt.subplots(figsize=(max(14, len(df)*0.6), 6))
    x = np.arange(len(df))
    w = 0.8 / len(metrics_to_plot)

    for i, m in enumerate(metrics_to_plot):
        offset = (i - len(metrics_to_plot)/2 + 0.5) * w
        bars = ax.bar(x + offset, df[m], w, label=labels.get(m, m),
                      color=colors.get(m, "gray"), alpha=0.82)
        for bar, val in zip(bars, df[m]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    subjects = df["test_subject"].tolist() if "test_subject" in df.columns else x
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Per-Fold Performance — {label}", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="gray", lw=1, ls="--", label="Chance")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / f"{label}_per_fold_bar.png")


# ── 7. LOSOCV Summary Radar / Bar ─────────────────────────────────────────────

def plot_losocv_summary_enhanced(metrics_dir: Path, out_dir: Path, label: str):
    json_path = metrics_dir / f"{label}_losocv_summary.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        summary = json.load(f)

    metric_names = list(summary.keys())
    means = [summary[m]["mean"] for m in metric_names]
    stds  = [summary[m]["std"]  for m in metric_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metric_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=["#3498db","#2ecc71","#e67e22","#e74c3c","#9b59b6","#1abc9c"][:len(x)],
                  alpha=0.85, error_kw={"elinewidth": 2, "capthick": 2})
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metric_names], fontsize=10)
    ax.set_ylabel("Score (mean ± std)", fontsize=11)
    ax.set_title(f"LOSOCV Metrics Summary — {label}", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="gray", lw=1, ls="--", label="Chance")
    ax.set_ylim(0, 1.1)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / f"{label}_summary_bar.png")


# ── 8. Subject Difficulty Scatter ─────────────────────────────────────────────

def plot_subject_difficulty(metrics_dir: Path, out_dir: Path, label: str):
    csv_path = metrics_dir / f"losocv_{label}.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if "accuracy" not in df.columns or "test_n" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy vs test set size
    ax = axes[0]
    sc = ax.scatter(df["test_n"], df["accuracy"],
                    c=df.get("roc_auc", df["accuracy"]),
                    cmap="RdYlGn", s=80, alpha=0.9, vmin=0.3, vmax=1.0)
    for _, row in df.iterrows():
        ax.annotate(row.get("test_subject", ""), (row["test_n"], row["accuracy"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Chance")
    ax.set_xlabel("Test Set Size (n epochs)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Subject Accuracy vs. Test Size", fontsize=12, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="ROC-AUC")
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy distribution histogram
    ax2 = axes[1]
    ax2.hist(df["accuracy"], bins=12, color="#3498db", alpha=0.8, edgecolor="white")
    ax2.axvline(df["accuracy"].mean(), color="red", lw=2, ls="--",
                label=f"Mean = {df['accuracy'].mean():.3f}")
    ax2.axvline(0.5, color="gray", lw=1, ls=":", label="Chance")
    ax2.set_xlabel("Accuracy")
    ax2.set_ylabel("Number of Folds")
    ax2.set_title("Fold Accuracy Distribution", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir / f"{label}_subject_difficulty.png")


# ── 9. Engagement Score Distribution ─────────────────────────────────────────

def plot_engagement_distribution(out_dir: Path, label: str):
    from config.settings import SUBJECT_IDS, PHASE3_DIR
    from data.dataset import _normalize_subject_id

    # Per-subject data lives at PHASE3_DIR.parent / sid / "output" / ...
    subj_base = PHASE3_DIR.parent

    subj_scores = {}
    for sid in SUBJECT_IDS:
        sid = _normalize_subject_id(sid)
        lbl_path = subj_base / sid / "output" / "engagement_phase3d" / "engagement_labels.npy"
        if lbl_path.exists():
            labels = np.load(lbl_path)
            subj_scores[sid] = labels

    if not subj_scores:
        print("  [SKIP] No engagement labels found")
        return

    sids  = sorted(subj_scores.keys())
    hi_pct = [subj_scores[s].mean() * 100 for s in sids]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#e74c3c" if p < 30 or p > 70 else "#2ecc71" for p in hi_pct]
    axes[0].bar(sids, hi_pct, color=colors, alpha=0.85)
    axes[0].axhline(50, color="black", lw=1.5, ls="--", label="50% (balanced)")
    axes[0].axhline(30, color="orange", lw=1, ls=":", label="30/70 imbalance")
    axes[0].axhline(70, color="orange", lw=1, ls=":")
    axes[0].set_ylabel("% HIGH engagement epochs")
    axes[0].set_xticklabels(sids, rotation=60, ha="right", fontsize=8)
    axes[0].set_title("Label Balance per Subject\n(global threshold)",
                      fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    n_epochs = [len(subj_scores[s]) for s in sids]
    axes[1].bar(sids, n_epochs, color="steelblue", alpha=0.85)
    axes[1].axhline(np.mean(n_epochs), color="red", lw=1.5, ls="--",
                    label=f"Mean = {np.mean(n_epochs):.1f}")
    axes[1].set_ylabel("Number of Epochs")
    axes[1].set_xticklabels(sids, rotation=60, ha="right", fontsize=8)
    axes[1].set_title("Epochs per Subject", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir / f"{label}_engagement_distribution.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="NEUMA Phase 8 Visualization")
    p.add_argument("--label",   default="ins_hdgs_cmt")
    p.add_argument("--out",     default=None, help="Output plot directory")
    p.add_argument("--skip-connectivity", action="store_true",
                   help="Skip connectivity heatmap (slow — reads all epochs)")
    p.add_argument("--skip-band-power", action="store_true",
                   help="Skip band power analysis (slow)")
    args = p.parse_args()

    label       = args.label
    out_dir     = Path(args.out) if args.out else Path(f"output/plots/{label}")
    metrics_dir = Path(f"output/metrics/{label}")
    explain_dir = Path(f"output/explainability/{label}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Visualize — label={label}")
    print(f"  Output  → {out_dir}/")
    print()

    print("  [1/9] GradCAM topomap …")
    plot_gradcam_topomap(explain_dir, out_dir, label)

    if not args.skip_connectivity:
        print("  [2/9] EEG connectivity heatmap …")
        plot_connectivity_heatmap(out_dir, label)
    else:
        print("  [2/9] Connectivity — skipped")

    if not args.skip_band_power:
        print("  [3/9] Band power comparison …")
        plot_band_power_comparison(out_dir, label)
    else:
        print("  [3/9] Band power — skipped")

    print("  [4/9] ROI attention …")
    plot_roi_attention(explain_dir, out_dir, label)

    print("  [5/9] GAT attention heatmap …")
    plot_gat_attention(explain_dir, out_dir, label)

    print("  [6/9] Per-fold performance bar chart …")
    plot_per_fold_performance(metrics_dir, out_dir, label)

    print("  [7/9] LOSOCV summary bar …")
    plot_losocv_summary_enhanced(metrics_dir, out_dir, label)

    print("  [8/9] Subject difficulty scatter …")
    plot_subject_difficulty(metrics_dir, out_dir, label)

    print("  [9/9] Engagement distribution …")
    plot_engagement_distribution(out_dir, label)

    print(f"\n  All plots saved to {out_dir}/\n")


if __name__ == "__main__":
    main()
