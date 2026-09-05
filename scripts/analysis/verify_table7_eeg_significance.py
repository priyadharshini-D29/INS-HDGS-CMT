#!/usr/bin/env python3
"""Regenerate & verify manuscript Table 7 (EEG-branch significance).

Reference model = INS-HDGS-CMT (EEG-only) = abl_no_et (raw, uncalibrated
balanced_acc), compared against each EEG baseline with a paired Wilcoxon
signed-rank test (zsplit), Holm-Bonferroni correction and Cliff's delta -
exactly as evaluation/make_tables.py::_eeg_significance computes it.

Writes results/statistics/table7_eeg_significance.csv and prints a comparison
against the values currently printed in the manuscript.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "baselines" / "dl"
ABL = ROOT / "results" / "ablation"
OUT = ROOT / "results" / "statistics" / "table7_eeg_significance.csv"

REF = "INS-HDGS-CMT (EEG-only)"
SRC = {
    REF:               ABL / "abl_no_et" / "losocv_abl_no_et.csv",
    "EEG Transformer": BASE / "losocv_eeg_transformer.csv",
    "TSception":       BASE / "losocv_tsception.csv",
    "GAT":             BASE / "losocv_gat.csv",
    "CNN-LSTM":        BASE / "losocv_cnn_lstm.csv",
    "CNN-BiLSTM":      BASE / "losocv_cnn_bilstm.csv",
    "EEGNet":          BASE / "losocv_eegnet.csv",
    "ShallowConvNet":  BASE / "losocv_shallow.csv",
    "DeepConvNet":     BASE / "losocv_deep.csv",
}
METRIC = "balanced_acc"  # raw / uncalibrated, matching make_tables.py default

# Manuscript Table 7 values (balanced accuracy) for comparison.
MS = {  # baseline: (median_delta, p, p_holm, cliffs_delta)
    "GAT":             (0.250, "<0.001", "<0.001", 0.68),
    "CNN-BiLSTM":      (0.200, "<0.001", "<0.001", 0.56),
    "TSception":       (0.167, "<0.001", "<0.001", 0.50),
    "DeepConvNet":     (0.133, "0.001",  "0.005",  0.42),
    "EEG Transformer": (0.125, "0.004",  "0.017",  0.41),
    "CNN-LSTM":        (0.125, "0.012",  "0.035",  0.39),
    "ShallowConvNet":  (0.129, "0.021",  "0.043",  0.32),
    "EEGNet":          (0.083, "0.042",  "0.043",  0.26),
}


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def load(name):
    df = pd.read_csv(SRC[name]).set_index("test_subject")
    return df[METRIC].dropna()


def main():
    global BASE, OUT, METRIC
    ap = argparse.ArgumentParser(description="Table 7 / S6 significance: reference EEG branch vs EEG baselines")
    ap.add_argument("--baseline-dir", default=str(BASE), help="results/baselines/dl (common budget) or dl_tuned")
    ap.add_argument("--ref-csv", default=str(SRC[REF]), help="per-fold CSV of the reference EEG branch "
                    "(default abl_no_et = ROI-retained; use results/ablation/abl_eeg_only/losocv_abl_eeg_only.csv "
                    "for the strictly gaze-free branch)")
    ap.add_argument("--metric", default=METRIC, help="balanced_acc | mcc | roc_auc (raw, uncalibrated)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    BASE, OUT, METRIC = Path(args.baseline_dir), Path(args.out), args.metric
    SRC[REF] = Path(args.ref_csv)
    for k in list(SRC):
        if k != REF:
            SRC[k] = BASE / SRC[k].name
    missing = [k for k, v in SRC.items() if not Path(v).exists()]
    if missing:
        raise SystemExit(f"missing per-fold CSVs: {missing}")
    series = {n: load(n) for n in SRC}
    common = sorted(set.intersection(*[set(s.index) for s in series.values()]))
    ref = series[REF].loc[common].to_numpy(float)
    rows, raw_p, names = [], [], []
    for name in SRC:
        if name == REF:
            continue
        m = series[name].loc[common].to_numpy(float)
        diff = ref - m
        p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff, zero_method="zsplit").pvalue)
        rows.append(dict(baseline=name, ref_mean=ref.mean(), model_mean=m.mean(),
                         median_delta=float(np.median(diff)),
                         wilcoxon_p=p, cliffs_delta=cliffs_delta(ref, m)))
        raw_p.append(p)
        names.append(name)
    for r, a in zip(rows, holm(raw_p)):
        r["p_holm"] = float(a)

    df = pd.DataFrame(rows).sort_values("cliffs_delta", ascending=False).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"n_subjects = {len(common)}  (reference = {REF}, metric = {METRIC} raw)\n")
    print(f"{'baseline':16s} {'medianΔ':>8s} {'p':>9s} {'p_holm':>9s} {'cliffδ':>7s}   "
          f"|  manuscript: {'medΔ':>5s} {'p':>7s} {'holm':>7s} {'δ':>5s}   match?")
    for _, r in df.iterrows():
        b = r["baseline"]
        ms = MS.get(b)
        ms_str = f"{ms[0]:+.3f} {ms[1]:>7s} {ms[2]:>7s} {ms[3]:+.2f}" if ms else "-"
        # match check (median to 3dp, cliff to 2dp; p compared loosely)
        ok = ""
        if ms:
            okd = abs(round(r['median_delta'], 3) - ms[0]) <= 0.001
            okc = abs(round(r['cliffs_delta'], 2) - ms[3]) <= 0.01
            ok = "OK" if (okd and okc) else "DIFF"
        print(f"{b:16s} {r['median_delta']:+8.3f} {r['wilcoxon_p']:9.4f} "
              f"{r['p_holm']:9.4f} {r['cliffs_delta']:+7.2f}   |  {ms_str:>28s}   {ok}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
