#!/usr/bin/env python
"""Regenerate Supplementary Table S8 (fixed-budget baselines) from the stored per-fold files.

Inputs (all committed):
    results/baselines/dl/losocv_<model>.csv           per-fold metrics of the common-budget run
    results/baselines/dl/fold_probs/probs_<model>.csv held-out probabilities per epoch
    results/ablation/abl_full/losocv_abl_full.csv     defines the 37 evaluable folds of Table 4

Outputs:
    results/statistics/table_s8_fixed_budget.csv, and the rows on stdout.

Conventions (stated in the S8 caption): mean +/- SD over the same 37 folds as Table 4;
ECE = 10-bin reliability (identical to scripts/analysis/pooled_calibration.py) and Brier,
both pooled over the 347 held-out epochs of those folds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DL = ROOT / "results/baselines/dl"

NAMES = {  # stem -> row label, grouped as in the table
    "shallow": "ShallowConvNet", "eegnet": "EEGNet", "cnn_bilstm": "CNN-BiLSTM",
    "deep": "DeepConvNet", "tsception": "TSception", "cnn_lstm": "CNN-LSTM",
    "eeg_transformer": "EEG Transformer", "gat": "GAT", "brain_gcn": "BrainGCN",
    "et_gru": "ET-GRU", "et_lstm": "ET-LSTM", "et_transformer": "ET-Transformer",
    "cross_attention": "Cross-Attention", "dual_transformer": "Dual Transformer",
    "mm_transformer": "Multimodal Transformer", "dynamicgat_et": "DynamicGAT+ET Transf.",
    "fusion_mlp": "Early-Fusion MLP", "late_fusion": "Late Fusion",
}


def ece10(y: np.ndarray, p: np.ndarray) -> float:
    """10-bin reliability ECE (identical to scripts/analysis/pooled_calibration.ece_reliability)."""
    bins = np.linspace(0, 1, 11)
    e, n = 0.0, len(y)
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def main() -> None:
    subjects = sorted(pd.read_csv(ROOT / "results/ablation/abl_full/losocv_abl_full.csv").test_subject)
    rows = []
    for stem, name in NAMES.items():
        lo = pd.read_csv(DL / f"losocv_{stem}.csv")
        lo = lo[lo.test_subject.isin(subjects)]
        pr = pd.read_csv(DL / "fold_probs" / f"probs_{stem}.csv")
        pr = pr[pr.test_subject.isin(subjects)]
        assert len(lo) == len(subjects), (stem, len(lo))
        y, p = pr.y_true.to_numpy(float), pr.p1.to_numpy(float)
        rows.append(dict(
            model=name, n_folds=len(lo), n_epochs=len(y),
            balacc_mean=lo.balanced_acc.mean(), balacc_sd=lo.balanced_acc.std(ddof=1),
            mcc_mean=lo.mcc.mean(), mcc_sd=lo.mcc.std(ddof=1),
            auc_mean=lo.roc_auc.mean(), auc_sd=lo.roc_auc.std(ddof=1),
            ece_pooled=ece10(y, p), brier_pooled=float(np.mean((p - y) ** 2))))
    df = pd.DataFrame(rows)
    out = ROOT / "results/statistics/table_s8_fixed_budget.csv"
    df.to_csv(out, index=False)
    for r in rows:
        print(f"{r['model']:23s} {r['balacc_mean']:.2f}+-{r['balacc_sd']:.2f}  "
              f"{r['mcc_mean']:+.2f}+-{r['mcc_sd']:.2f}  {r['auc_mean']:.2f}+-{r['auc_sd']:.2f}  "
              f"ECE {r['ece_pooled']:.2f}  Brier {r['brier_pooled']:.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
