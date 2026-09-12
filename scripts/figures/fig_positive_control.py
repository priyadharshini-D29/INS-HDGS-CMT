#!/usr/bin/env python3
"""Positive control figure (Section 3.2 / Supplementary Table S16): the same EEG encoders scored
on three labels recorded from the same participants.

  (A) Positive control  : resting state vs brochure browsing (NEUMA_LABEL_SOURCE=control, 42 folds)
  (B) Engagement index  : the production label (phase3d, 37 evaluable folds)
  The behavioural label (product bought vs not bought, 42 folds) is not drawn: only the full
  model was run on that track (ROC-AUC 0.61, Supplementary Table S16), so a panel would carry a
  single bar; the value is reported in the text and in Table S16 instead.

Bars are the mean per-fold ROC-AUC (uncalibrated; ROC-AUC is threshold-free) with the SD over folds.
Rungs missing on a track are left blank and listed on stdout, never fabricated.

Output : paper/figures/fig_positive_control.{pdf,png}, results/figures/positive_control.csv
Run    : python scripts/figures/fig_positive_control.py     (repository root; CPU; server)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
TRACKS = [("A", "Positive control:\nresting state vs browsing", R / "label_control"),
          ("B", "Engagement index\n(production label)",           R)]

# display name -> candidate relative paths (first that exists is used)
RUNGS = [
    ("EEGNet, tuned",                       ["baselines/dl_tuned/losocv_eegnet.csv"]),
    ("ShallowConvNet, tuned",               ["baselines/dl_tuned/losocv_shallow.csv"]),
    ("DeepConvNet, tuned",                  ["baselines/dl_tuned/losocv_deep.csv"]),
    ("Linear encoder, node features",       ["ablation/abl_eeg_lin/losocv_abl_eeg_lin.csv"]),
    ("Graph pathway only",                  ["ablation/abl_eeg_graph_plain/losocv_abl_eeg_graph_plain.csv"]),
    ("Spiking encoder only (raw EEG)",      ["ablation/abl_eeg_snn_raw/losocv_abl_eeg_snn_raw.csv"]),
    ("Spiking + graph (plain head)",        ["ablation/abl_eeg_snn_plain/losocv_abl_eeg_snn_plain.csv"]),
    ("Proposed gaze-free branch",           ["ablation/abl_eeg_only/losocv_abl_eeg_only.csv",
                                             "ablation/abl_eeg_only_mmd/losocv_abl_eeg_only_mmd.csv"]),
    ("Full model (EEG + gaze)",             ["ablation/abl_full/losocv_abl_full.csv",
                                             "losocv_metrics/losocv_repro_focal_g3p0_effective_num_37.csv"]),
]
GAZE_ROWS = {"Full model (EEG + gaze)"}
EEG_COLOR, GAZE_COLOR = "#4c72b0", "#c44e52"


def fold_auc(csv):
    d = pd.read_csv(csv).drop_duplicates("test_subject", keep="last")
    a = d["roc_auc"].astype(float).to_numpy()
    return float(np.nanmean(a)), float(np.nanstd(a)), int(np.isfinite(a).sum())


def main():
    recs = []
    fig, axes = plt.subplots(1, len(TRACKS), figsize=(4.7 * len(TRACKS) + 0.6, 4.8), sharey=True)
    for ax, (tag, title, base) in zip(axes, TRACKS):
        names, means, sds, cols = [], [], [], []
        for name, cands in RUNGS:
            path = next((base / c for c in cands if (base / c).exists()), None)
            if path is None:
                print(f"[{tag}] {name}: missing on this track")
                names.append(name); means.append(np.nan); sds.append(0.0)
                cols.append(GAZE_COLOR if name in GAZE_ROWS else EEG_COLOR)
                continue
            m, s, n = fold_auc(path)
            recs.append(dict(track=tag, model=name, auc_mean=m, auc_sd=s, folds=n, csv=str(path.relative_to(ROOT))))
            names.append(name); means.append(m); sds.append(s)
            cols.append(GAZE_COLOR if name in GAZE_ROWS else EEG_COLOR)
        x = np.arange(len(names))
        ax.bar(x, np.nan_to_num(means), yerr=sds, color=cols, capsize=3, zorder=3, alpha=0.9)
        for xi, m in zip(x, means):
            if np.isnan(m):
                ax.text(xi, 0.52, "not run", ha="center", va="bottom", fontsize=7, color="0.5", rotation=90)
            else:
                ax.text(xi, m + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"({tag}) {title}", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3); ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel("ROC-AUC, mean per held-out subject ($\\pm$ SD)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=EEG_COLOR, label="EEG only (gaze-free)"),
               plt.Rectangle((0, 0), 1, 1, color=GAZE_COLOR, label="EEG + gaze")]
    axes[-1].legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    df = pd.DataFrame(recs)
    for out in (ROOT / "paper/figures", ROOT / "results/figures"):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "fig_positive_control.pdf", bbox_inches="tight")
        fig.savefig(out / "fig_positive_control.png", dpi=250, bbox_inches="tight")
    df.to_csv(ROOT / "results/figures/positive_control.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("wrote paper/figures/fig_positive_control.{pdf,png}, results/figures/positive_control.csv")


if __name__ == "__main__":
    main()
