#!/usr/bin/env python3
"""Fig. 5 of the manuscript (file fig3_losocv_results): LOSOCV performance of the production model,
drawn from the revision runs so that every number matches Tables 3-7.

  (A) Pooled ROC over the 37 held-out subjects: full model, the tuned ET-LSTM (gaze only) and the
      gaze-free EEG branch, from the per-epoch probabilities.
  (B) Per-fold distribution of the full model's calibrated balanced accuracy, ROC-AUC, calibrated MCC
      and calibrated accuracy (the Table 3 conventions).
  (C) Mean per-fold ROC-AUC (+/- SD) of the label's own gaze terms (linear probe), the full model, the
      tuned ET-LSTM, the gaze-free branch, the best tuned EEG encoder and the label's own EEG terms:
      where the accuracy sits relative to the ceiling the label's gaze terms set.

Inputs : results/ablation/abl_full/losocv_abl_full.csv
         results/ablation/abl_eeg_only/losocv_abl_eeg_only.csv
         results/baselines/dl_tuned/losocv_*.csv, results/baselines/dl_tuned/fold_probs/probs_et_lstm.csv
         results/statistics/label_leakage_audit.csv (Table S15 probes; optional)
Output : paper/figures/fig3_losocv_results.{pdf,png} (+ results/figures/)
Run    : python scripts/figures/fig3_losocv_results.py   (repository root; CPU; server)
"""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FULL_CSV = ROOT / "results/ablation/abl_full/losocv_abl_full.csv"
EEG_CSV = ROOT / "results/ablation/abl_eeg_only/losocv_abl_eeg_only.csv"
BASE = ROOT / "results/baselines/dl_tuned"
AUDIT = ROOT / "results/statistics/label_leakage_audit.csv"
EEG_ENCODERS = ["eegnet", "shallow", "deep", "cnn_lstm", "cnn_bilstm", "eeg_transformer", "tsception", "gat", "brain_gcn"]
PRETTY = {"eegnet": "EEGNet", "shallow": "ShallowConvNet", "deep": "DeepConvNet", "cnn_lstm": "CNN-LSTM",
          "cnn_bilstm": "CNN-BiLSTM", "eeg_transformer": "EEG-Transformer", "tsception": "TSception",
          "gat": "GAT", "brain_gcn": "BrainGCN"}
C_FULL, C_ET, C_EEG, C_PROBE, C_BASE = "#2EA37A", "#55a868", "#B2592E", "#999999", "#5B8DEF"
C_PROBE_EEG = "#8172B2"   # the label's EEG-terms probe, distinct from the gaze-terms probe (grey)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def per_fold(csv):
    return pd.read_csv(csv).drop_duplicates("test_subject", keep="last")


def pooled_from_losocv(csv):
    d = per_fold(csv); yt, yp = [], []
    for _, r in d.iterrows():
        yt += list(ast.literal_eval(r["y_true"])); yp += list(ast.literal_eval(r["y_prob"]))
    return np.array(yt, int), np.array(yp, float)


def pooled_from_probs(name):
    p = BASE / "fold_probs" / f"probs_{name}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d["y_true"].to_numpy(int), d["p1"].to_numpy(float)


def probe(prefixes):
    if not AUDIT.exists():
        return None
    d = pd.read_csv(AUDIT)
    for pre in prefixes:
        m = d[d["feature_set"].str.startswith(pre)]
        if len(m):
            return float(m.iloc[0]["fold_auc_mean"]), float(m.iloc[0]["fold_auc_sd"])
    return None


def main():
    d = per_fold(FULL_CSV)
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.35], wspace=0.34)

    # (A) pooled ROC ---------------------------------------------------------------------
    axA = fig.add_subplot(gs[0])
    curves = [("Full model (EEG + gaze)", pooled_from_losocv(FULL_CSV), C_FULL, 2.4)]
    et = pooled_from_probs("et_lstm")
    if et is not None:
        curves.append(("ET-LSTM, gaze only (tuned)", et, C_ET, 1.6))
    if EEG_CSV.exists():
        curves.append(("Gaze-free EEG branch", pooled_from_losocv(EEG_CSV), C_EEG, 1.6))
    for name, (yt, yp), c, lw in curves:
        fpr, tpr, _ = roc_curve(yt, yp)
        axA.plot(fpr, tpr, color=c, lw=lw, label=f"{name} (AUC {roc_auc_score(yt, yp):.2f})")
    axA.plot([0, 1], [0, 1], "--", color="#999", lw=1.0)
    axA.set_xlabel("False positive rate"); axA.set_ylabel("True positive rate")
    axA.set_title("(A) Pooled LOSOCV ROC (37 subjects)", fontweight="bold", fontsize=10)
    axA.legend(loc="lower right", fontsize=7.4, frameon=False); axA.set_aspect("equal"); axA.grid(alpha=.25)

    # (B) per-fold distribution ----------------------------------------------------------
    axB = fig.add_subplot(gs[1])
    mets = [("balanced_acc_cal", "Balanced\nacc (cal.)"), ("roc_auc", "ROC-AUC"),
            ("mcc_cal", "MCC (cal.)"), ("accuracy_cal", "Accuracy\n(cal.)")]
    data = [d[m].astype(float).values for m, _ in mets]
    bp = axB.boxplot(data, patch_artist=True, widths=.6, showmeans=True,
                     meanprops=dict(marker="D", mfc="white", mec="k", ms=5))
    for patch in bp["boxes"]:
        patch.set(facecolor=C_FULL, alpha=.35)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        axB.scatter(rng.normal(i + 1, .05, len(vals)), vals, s=10, color=C_FULL, alpha=.6, zorder=3)
    axB.set_xticks(range(1, len(mets) + 1)); axB.set_xticklabels([l for _, l in mets])
    axB.set_ylim(-.4, 1.05); axB.axhline(0, color="#bbb", lw=.8); axB.axhline(0.5, color="#ddd", lw=.8, ls=":")
    axB.set_title(f"(B) Full model, per fold (n={len(d)})", fontweight="bold", fontsize=10)
    axB.grid(axis="y", alpha=.25)

    # (C) where the accuracy sits --------------------------------------------------------
    axC = fig.add_subplot(gs[2])
    bars = []
    g = probe(["Gaze-5", "ET-5"])
    if g: bars.append(("Label's gaze terms\n(linear probe)", g[0], g[1], C_PROBE))
    bars.append(("Full model\n(EEG + gaze)", d["roc_auc"].mean(), d["roc_auc"].std(), C_FULL))
    et_csv = BASE / "losocv_et_lstm.csv"
    if et_csv.exists():
        e = per_fold(et_csv); bars.append(("ET-LSTM\n(gaze only, tuned)", e["roc_auc"].mean(), e["roc_auc"].std(), C_ET))
    if EEG_CSV.exists():
        e = per_fold(EEG_CSV); bars.append(("Gaze-free\nEEG branch", e["roc_auc"].mean(), e["roc_auc"].std(), C_EEG))
    best = None
    for name in EEG_ENCODERS:
        p = BASE / f"losocv_{name}.csv"
        if p.exists():
            e = per_fold(p); m = e["roc_auc"].mean()
            if best is None or m > best[1]:
                best = (PRETTY[name], m, e["roc_auc"].std())
    if best: bars.append((f"Best tuned EEG\nencoder ({best[0]})", best[1], best[2], C_BASE))
    ee = probe(["EEG-5"])
    if ee: bars.append(("Label's EEG terms\n(linear probe)", ee[0], ee[1], C_PROBE_EEG))
    x = np.arange(len(bars))
    axC.bar(x, [b[1] for b in bars], yerr=[b[2] for b in bars], color=[b[3] for b in bars], capsize=3, alpha=.9, zorder=3)
    for xi, b in zip(x, bars):
        axC.text(xi, b[1] + 0.03, f"{b[1]:.2f}", ha="center", fontsize=8)
    axC.axhline(0.5, color="k", ls="--", lw=0.8)
    axC.set_xticks(x); axC.set_xticklabels([b[0] for b in bars], fontsize=6.6, rotation=30, ha="right", rotation_mode="anchor")
    axC.set_ylim(0.3, 1.05); axC.set_ylabel("ROC-AUC, mean per fold ($\\pm$ SD)")
    axC.set_title("(C) Mean per-fold ROC-AUC against the label's own terms", fontweight="bold", fontsize=10)
    axC.grid(axis="y", alpha=.25)

    fig.suptitle("LOSOCV performance on the engagement index (37 evaluable subjects)", fontsize=12, fontweight="bold", y=1.02)
    for out in (ROOT / "paper/figures", ROOT / "results/figures"):
        out.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig3_losocv_results.{ext}", dpi=300, bbox_inches="tight")
    print("panel C:", [(b[0].replace(chr(10), ' '), round(b[1], 3), round(b[2], 3)) for b in bars])
    print("wrote paper/figures/fig3_losocv_results.{pdf,png}")


if __name__ == "__main__":
    main()
