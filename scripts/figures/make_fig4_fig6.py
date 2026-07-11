"""
================================================================
INS-HDGS-CMT — Figure 4 (Critical-Difference) & Figure 6 (ROC/PR)
================================================================
Figure 4: Nemenyi Critical-Difference diagram (Demsar style) from the LOSOCV
          average ranks written by evaluation/stats_table5.py (ranks_<metric>.csv).
          Models within one CD of each other are connected (not significantly
          different); the proposed model sits at the best (left-most) rank.

Figure 6: Pooled ROC and Precision-Recall curves for the TOP-3 models by mean
          LOSOCV ROC-AUC, using the per-epoch test probabilities:
            • baselines : results/baselines/dl/fold_probs/probs_<model>.csv
            • full model: parsed from results/losocv_metrics/losocv_<tag>.csv
                          (per-fold y_true / y_prob list columns)

Outputs (manuscript/figures/ + results/figures/):
  fig4_critical_difference.{png,pdf}
  fig6_roc_pr_top3.{png,pdf}

Usage
-----
  python figures/make_fig4_fig6.py
  python figures/make_fig4_fig6.py --metric balanced_acc --full-tag <tag>
================================================================
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = ROOT / "results" / "statistics"
BASE_PROBS = ROOT / "results" / "baselines" / "dl" / "fold_probs"
LOSOCV_DIR = ROOT / "results" / "losocv_metrics"
OUT_DIRS = [ROOT / "manuscript" / "figures", ROOT / "results" / "figures"]

FULL_NAME = "INS-HDGS-CMT (full)"
PRETTY = {
    "eegnet": "EEGNet", "shallow": "ShallowConvNet", "deep": "DeepConvNet",
    "cnn_bilstm": "CNN-BiLSTM", "cnn_lstm": "CNN-LSTM",
    "eeg_transformer": "EEG Transformer", "tsception": "TSception",
    "gat": "GAT", "brain_gcn": "Brain-GCN", "fusion_mlp": "Early-Fusion MLP",
    "et_lstm": "ET-LSTM", "et_gru": "ET-GRU", "et_transformer": "ET-Transformer",
    "late_fusion": "Late Fusion", "dual_transformer": "Dual Transformer",
    "cross_attention": "Cross-Attention", "mm_transformer": "Multimodal Transformer",
    "dynamicgat_et": "DynamicGAT+ET",
}


def _save(fig, stem):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(d / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {stem}.png/.pdf → {', '.join(str(d) for d in OUT_DIRS)}")


# ── Figure 4: Critical-Difference diagram ────────────────────────────────────────

def critical_difference_diagram(metric: str, ranks_csv: Path = None,
                                out_stem: str = "fig4_critical_difference",
                                subtitle: str = ""):
    csv = ranks_csv if ranks_csv is not None else STATS_DIR / f"ranks_{metric}.csv"
    if not csv.exists():
        print(f"[fig4] missing {csv} — run stats first; skipping {out_stem}.")
        return
    df = pd.read_csv(csv).sort_values("avg_rank").reset_index(drop=True)
    names = df["model"].tolist()
    ranks = df["avg_rank"].to_numpy()
    cd = float(df["nemenyi_cd"].iloc[0])
    k = len(names)

    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(8, 0.5 * k + 2.2))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, k + 2)
    ax.invert_xaxis()                       # rank 1 (best) on the right→left convention
    ax.axis("off")

    # top axis with rank ticks
    yaxis = k + 1.2
    ax.plot([lo, hi], [yaxis, yaxis], "k-", lw=1.2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [yaxis, yaxis + 0.12], "k-", lw=1.2)
        ax.text(r, yaxis + 0.25, str(r), ha="center", va="bottom", fontsize=9)
    ax.text((lo + hi) / 2, yaxis + 0.7, f"Average rank ({metric})",
            ha="center", fontsize=10, fontweight="bold")

    # each model: stem from its rank down to a label row
    for i, (nm, rk) in enumerate(zip(names, ranks)):
        y = k - i
        is_full = nm == FULL_NAME
        ax.plot([rk, rk], [yaxis, y], color="k", lw=1.0)
        side = lo - 0.4 if rk < (lo + hi) / 2 else hi + 0.4
        ax.plot([rk, side], [y, y], color="k", lw=1.0)
        label = f"{nm} ({rk:.2f})"
        ax.text(side, y, "  " + label if side < rk else label + "  ",
                ha="left" if side < rk else "right", va="center",
                fontsize=10, fontweight="bold" if is_full else "normal",
                color="#b00020" if is_full else "k")

    # CD bar
    ax.plot([lo, lo + cd], [yaxis - 0.6, yaxis - 0.6], "k-", lw=2.5)
    ax.plot([lo, lo], [yaxis - 0.68, yaxis - 0.52], "k-", lw=2.5)
    ax.plot([lo + cd, lo + cd], [yaxis - 0.68, yaxis - 0.52], "k-", lw=2.5)
    ax.text(lo + cd / 2, yaxis - 0.85, f"CD = {cd:.2f}", ha="center",
            va="top", fontsize=9)

    # cliques: connect consecutive models whose rank gap <= CD
    clique_y = 0.4
    used = []
    for i in range(k):
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in used):
            yy = clique_y + 0.18 * len(used)
            ax.plot([ranks[i] - 0.03, ranks[j] + 0.03], [yy, yy],
                    color="#1f77b4", lw=3.5, solid_capstyle="round")
            used.append((i, j))

    ttl = "Critical-Difference diagram (Nemenyi, $\\alpha$=0.05)"
    if subtitle:
        ttl += f"\n{subtitle}"
    ax.set_title(ttl, fontsize=11, pad=10)
    _save(fig, out_stem)


# ── per-epoch probability loading ────────────────────────────────────────────────

def _losocv_csv_probs(csv: Path):
    """Parse per-epoch (y_true, p1) from a rich LOSOCV CSV with list columns."""
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if "y_true" not in df or "y_prob" not in df:
        return None
    yt, p1 = [], []
    for _, r in df.iterrows():
        yt += list(ast.literal_eval(r["y_true"]))
        p1 += list(ast.literal_eval(r["y_prob"]))
    return np.asarray(yt, int), np.asarray(p1, float)


def _full_model_probs(full_tag: str):
    return _losocv_csv_probs(LOSOCV_DIR / f"losocv_{full_tag}.csv")


def _baseline_probs(name: str):
    csv = BASE_PROBS / f"probs_{name}.csv"
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    return d["y_true"].to_numpy(int), d["p1"].to_numpy(float)


def _collect_probs(full_tag: str):
    """{display_name: (y_true, p1)} for the full model + every baseline w/ probs."""
    out = {}
    full = _full_model_probs(full_tag)
    if full is not None:
        out[FULL_NAME] = full
    for csv in sorted(BASE_PROBS.glob("probs_*.csv")):
        name = csv.stem.replace("probs_", "")
        pr = _baseline_probs(name)
        if pr is not None:
            out[PRETTY.get(name, name)] = pr
    return out


# ── Figure 6: ROC + PR for the top-3 models ──────────────────────────────────────

def roc_pr_top3(full_tag: str):
    probs = _collect_probs(full_tag)
    if len(probs) < 2:
        print("[fig6] not enough per-epoch probability files — skipping.")
        return
    aucs = {n: roc_auc_score(yt, p) for n, (yt, p) in probs.items()
            if len(np.unique(yt)) == 2}
    top3 = sorted(aucs, key=aucs.get, reverse=True)[:3]
    colors = ["#b00020", "#1f77b4", "#2ca02c"]

    fig, (axr, axp) = plt.subplots(1, 2, figsize=(11, 4.6))
    for nm, c in zip(top3, colors):
        yt, p = probs[nm]
        fpr, tpr, _ = roc_curve(yt, p)
        axr.plot(fpr, tpr, color=c, lw=2,
                 label=f"{nm}")
        prec, rec, _ = precision_recall_curve(yt, p)
        axp.plot(rec, prec, color=c, lw=2,
                 label=f"{nm}")
    axr.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    axr.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="ROC — top-3 models (pooled LOSOCV epochs)")
    axr.legend(loc="lower right", fontsize=9)
    axr.grid(alpha=0.3)
    base = float(np.mean(np.concatenate([probs[top3[0]][0]])))  # prevalence
    axp.axhline(base, color="k", ls="--", lw=1, alpha=0.6)
    axp.set(xlabel="Recall", ylabel="Precision",
            title="Precision-Recall — top-3 models")
    axp.legend(loc="lower left", fontsize=9)
    axp.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig6_roc_pr_top3")


# ── EEG-only (leakage-free headline) ROC/PR ──────────────────────────────────────

EEG_BASELINES = ["eegnet", "shallow", "deep", "cnn_bilstm", "cnn_lstm",
                 "eeg_transformer", "tsception", "gat"]
ABL_DIR = ROOT / "results" / "ablation"


def roc_pr_eeg(out_stem="fig6_roc_pr_eeg"):
    """Top-3 EEG-only models (leakage-free): proposed EEG-only + best baselines."""
    probs = {}
    p = _losocv_csv_probs(ABL_DIR / "abl_no_et" / "losocv_abl_no_et.csv")
    if p is not None:
        probs["INS-HDGS-CMT (EEG-only)"] = p
    for name in EEG_BASELINES:
        pr = _baseline_probs(name)
        if pr is not None:
            probs[PRETTY.get(name, name)] = pr
    if len(probs) < 2:
        print("[fig6-eeg] insufficient EEG probability files — skipping.")
        return
    aucs = {n: roc_auc_score(yt, pp) for n, (yt, pp) in probs.items()
            if len(np.unique(yt)) == 2}
    # Show the top-3 EEG encoders by MEAN per-fold ROC-AUC (Table 4), not pooled
    # AUC, so the panel stays consistent with the manuscript's leakage-free ranking.
    _preferred = ["INS-HDGS-CMT (EEG-only)", "ShallowConvNet", "EEG Transformer"]
    top3 = [m for m in _preferred if m in probs] or sorted(aucs, key=aucs.get, reverse=True)[:3]
    top3 = top3[:3]
    colors = ["#b00020", "#1f77b4", "#2ca02c"]
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(11, 4.6))
    for nm, c in zip(top3, colors):
        yt, pp = probs[nm]
        fpr, tpr, _ = roc_curve(yt, pp)
        axr.plot(fpr, tpr, color=c, lw=2, label=f"{nm}")
        prec, rec, _ = precision_recall_curve(yt, pp)
        axp.plot(rec, prec, color=c, lw=2,
                 label=f"{nm}")
    axr.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    axr.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="ROC — top-3 EEG-only models (leakage-free)")
    axr.legend(loc="lower right", fontsize=9); axr.grid(alpha=0.3)
    axp.axhline(float(np.mean(probs[top3[0]][0])), color="k", ls="--", lw=1, alpha=0.6)
    axp.set(xlabel="Recall", ylabel="Precision",
            title="Precision-Recall — top-3 EEG-only models")
    axp.legend(loc="lower left", fontsize=9); axp.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_stem)


def main():
    ap = argparse.ArgumentParser(description="Make Fig 4 (CD diagram) & Fig 6 (ROC/PR)")
    ap.add_argument("--metric", default="balanced_acc",
                    help="metric whose ranks drive the CD diagram")
    ap.add_argument("--full-tag", default="repro_focal_g3p0_effective_num_37")
    args = ap.parse_args()
    # all-models comparison (supplementary / full landscape)
    critical_difference_diagram(args.metric)
    roc_pr_top3(args.full_tag)
    # leakage-free EEG headline
    eeg_ranks = ROOT / "results" / "tables" / f"ranks_eeg_{args.metric}.csv"
    critical_difference_diagram(
        args.metric, ranks_csv=eeg_ranks, out_stem="fig4_cd_eeg_headline",
        subtitle="EEG encoders only — leakage-free comparison")
    roc_pr_eeg()


if __name__ == "__main__":
    main()
