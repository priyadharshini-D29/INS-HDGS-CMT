#!/usr/bin/env python3
"""
Pooled out-of-fold calibration of every model from its SAVED held-out
predictions: reliability ECE (10 and 15 equal-width bins) and Brier score over
all held-out epochs, plus the mean per-fold ECE with the corrected formula.

Sources
-------
  proposed model   per-fold LOSOCV CSVs with stringified y_true / y_prob lists
                   (published run, abl_full, abl_eeg_only, abl_eeg_only_mmd)
  tuned baselines  results/baselines/dl_tuned/fold_probs/probs_<model>.csv
                   (one row per held-out epoch: fold, test_subject, y_true,
                   y_pred, p1)

ECE here is the binary reliability form: within each probability bin the mean
predicted P(HIGH) is compared with the empirical positive frequency (the same
formula as training.metrics.compute_ece after its correction and as
scripts/analysis/recompute_ece_pooled.py).  The "stored" per-fold ECE column
reproduces the manuscript Tables 4-6 ECE column, which was computed with the
earlier formula (bin confidence versus thresholded accuracy) and averaged over
~9-epoch folds; it is listed only so the two can be compared.

Nothing is trained; the script reads CSVs and runs in seconds.

Usage
-----
  python scripts/analysis/pooled_calibration.py
Outputs results/statistics/pooled_calibration.md and .csv
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "statistics"

# proposed-model runs: label -> (display name, per-fold CSV)
PROPOSED = {
    "published":    ("INS-HDGS-CMT (full), published run",
                     RES / "losocv_metrics" / "losocv_repro_focal_g3p0_effective_num_37.csv"),
    "abl_full":     ("INS-HDGS-CMT (full), revision re-run (abl_full)",
                     RES / "ablation" / "abl_full" / "losocv_abl_full.csv"),
    "eeg_only":     ("INS-HDGS-CMT (EEG-only, no gaze input)",
                     RES / "ablation" / "abl_eeg_only" / "losocv_abl_eeg_only.csv"),
    "eeg_only_mmd": ("INS-HDGS-CMT (EEG-only, MMD/DANN kept)",
                     RES / "ablation" / "abl_eeg_only_mmd" / "losocv_abl_eeg_only_mmd.csv"),
}

# tuned baselines: file stem -> manuscript display name (Tables 4, 5, 6); order = table order
BASELINES = [
    # Table 4 (EEG encoders)
    ("shallow",          "ShallowConvNet"),
    ("deep",             "DeepConvNet"),
    ("cnn_bilstm",       "CNN-BiLSTM"),
    ("cnn_lstm",         "CNN-LSTM"),
    ("eeg_transformer",  "EEG Transformer"),
    ("tsception",        "TSception"),
    ("gat",              "GAT"),
    ("eegnet",           "EEGNet"),
    ("brain_gcn",        "BrainGCN"),                     # in the S10 Holm family, not tabulated in Table 4
    # Table 5 (eye-tracking encoders)
    ("et_transformer",   "ET-Transformer"),
    ("et_lstm",          "ET-LSTM"),
    ("et_gru",           "ET-GRU"),
    # Table 6 (multimodal fusion)
    ("cross_attention",  "Cross-Attention"),
    ("mm_transformer",   "Multimodal Transformer"),
    ("dynamicgat_et",    "DynamicGAT + ET Transformer"),
    ("dual_transformer", "Dual Transformer"),
    ("fusion_mlp",       "Early-Fusion MLP"),
    ("late_fusion",      "Late Fusion (CNN-LSTM + ET-LSTM)"),
]


# ── metrics ───────────────────────────────────────────────────────────────────

def ece_reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Binary reliability ECE (identical to training.metrics.compute_ece)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0; n = len(y)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def ece_old(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """The earlier per-fold definition (bin confidence vs thresholded accuracy)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0; n = len(y)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(p[m].mean() - (y[m] == (p[m] >= 0.5)).mean())
    return float(e)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def parse_arr(s) -> np.ndarray:
    s = re.sub(r"\s+", ",", str(s).strip())
    s = re.sub(r",+", ",", s).replace("[,", "[").replace(",]", "]")
    return np.asarray(ast.literal_eval(s), dtype=float)


def summarise(name: str, group: str, folds: list[tuple[np.ndarray, np.ndarray]], stored_ece=None) -> dict:
    y = np.concatenate([f[0] for f in folds]); p = np.concatenate([f[1] for f in folds])
    return dict(model=name, group=group, n_folds=len(folds), n_epochs=len(y),
                pooled_ece10=ece_reliability(y, p, 10), pooled_ece15=ece_reliability(y, p, 15),
                brier=brier(y, p),
                mean_fold_ece10_corrected=float(np.mean([ece_reliability(a, b, 10) for a, b in folds])),
                mean_fold_ece10_old=float(np.mean([ece_old(a, b, 10) for a, b in folds])),
                stored_mean_fold_ece=(float(stored_ece) if stored_ece is not None else float("nan")),
                pooled_prevalence=float(y.mean()), mean_prob=float(p.mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-dir", default=str(RES / "baselines" / "dl_tuned"))
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    base_dir = Path(args.baseline_dir)

    rows = []
    for key, (name, csv) in PROPOSED.items():
        if not csv.exists():
            print(f"[info] {key}: {csv} not found, skipped"); continue
        df = pd.read_csv(csv)
        folds = [(parse_arr(r["y_true"]), parse_arr(r["y_prob"])) for _, r in df.iterrows()]
        rows.append(summarise(name, "proposed", folds, df["ece"].mean() if "ece" in df else None))

    for stem, name in BASELINES:
        pf = base_dir / "fold_probs" / f"probs_{stem}.csv"
        if not pf.exists():
            print(f"[info] {stem}: {pf} not found, skipped"); continue
        d = pd.read_csv(pf)
        folds = [(g["y_true"].to_numpy(float), g["p1"].to_numpy(float)) for _, g in d.groupby("fold", sort=True)]
        lf = base_dir / f"losocv_{stem}.csv"
        stored = pd.read_csv(lf)["ece"].mean() if lf.exists() else None
        rows.append(summarise(name, "baseline", folds, stored))

    out = pd.DataFrame(rows)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "pooled_calibration.csv", index=False)

    lines = ["# Pooled out-of-fold calibration (saved held-out predictions, uncalibrated P(HIGH))", "",
             f"Proposed-model runs from their per-fold y_true / y_prob lists; tuned baselines from "
             f"`{base_dir.relative_to(ROOT).as_posix()}/fold_probs/probs_<model>.csv` (p1). "
             "ECE = binary reliability ECE (mean predicted probability vs empirical positive frequency per equal-width bin), "
             "computed once over all held-out epochs pooled (10 and 15 bins); Brier = mean (p - y)^2. "
             "\"Mean per-fold ECE (corrected)\" averages the same 10-bin reliability ECE over the ~9-epoch folds and is "
             "descriptive only. \"Stored\" is the mean of the per-fold ECE column saved by the run "
             "(earlier formula: bin confidence vs thresholded accuracy), i.e. the number printed in the manuscript Tables 4-6.", "",
             "| Model | n epochs | Pooled ECE (10 bins) | Pooled ECE (15 bins) | Brier | Mean per-fold ECE (corrected) | Stored mean per-fold ECE (old formula) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {r['model']} | {int(r['n_epochs'])} | {r['pooled_ece10']:.3f} | {r['pooled_ece15']:.3f} | "
                     f"{r['brier']:.3f} | {r['mean_fold_ece10_corrected']:.3f} | {r['stored_mean_fold_ece']:.3f} |")
    lines += ["", f"Pooled prevalence of HIGH over the held-out epochs: {out['pooled_prevalence'].iloc[0]:.3f}. "
              "A reference predictor that outputs the prevalence for every epoch has Brier = "
              f"{out['pooled_prevalence'].iloc[0] * (1 - out['pooled_prevalence'].iloc[0]):.3f}."]
    md = "\n".join(lines) + "\n"
    (out_dir / "pooled_calibration.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out_dir / 'pooled_calibration.md'} and .csv")


if __name__ == "__main__":
    main()
