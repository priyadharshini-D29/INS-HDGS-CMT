#!/usr/bin/env python3
"""Supplementary Table S10 (and the significance columns of Tables 4 / 5):
full INS-HDGS-CMT versus every tuned baseline, paired over the LOSOCV folds.

Same conventions as verify_table7_eeg_significance.py: raw (uncalibrated)
per-fold metric, paired Wilcoxon signed-rank (zsplit), Holm-Bonferroni across
ALL baselines found in --baseline-dir within each metric, Cliff's delta with a
positive value favouring the full model.  Prints a LaTeX block with the
eye-tracking / multimodal rows in the manuscript's order and writes one CSV
with every baseline (EEG ones included, so the Holm family is explicit).

    python scripts/analysis/table_s10_full_vs_baselines.py
    python scripts/analysis/table_s10_full_vs_baselines.py --baseline-dir results/baselines/dl_tuned
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "results" / "ablation" / "abl_full" / "losocv_abl_full.csv"
BASE = ROOT / "results" / "baselines" / "dl_tuned"
OUT = ROOT / "results" / "statistics" / "tableS10_full_vs_baselines.csv"
METRICS = [("balanced_acc", "Balanced accuracy"), ("mcc", "MCC"), ("roc_auc", "ROC-AUC")]

# file stem -> manuscript name; order = Table S10 rows (ET / multimodal block)
NAMES = {
    "eegnet": "EEGNet", "shallow": "ShallowConvNet", "deep": "DeepConvNet",
    "cnn_lstm": "CNN-LSTM", "cnn_bilstm": "CNN-BiLSTM", "eeg_transformer": "EEG Transformer",
    "tsception": "TSception", "gat": "GAT", "brain_gcn": "BrainGCN",
    "et_gru": "ET-GRU", "et_lstm": "ET-LSTM", "et_transformer": "ET-Transformer",
    "fusion_mlp": "Early-Fusion MLP", "late_fusion": "Late Fusion",
    "dual_transformer": "Dual Transformer", "cross_attention": "Cross-Attention",
    "mm_transformer": "Multimodal Transformer", "dynamicgat_et": "DynamicGAT+ET Transf.",
}
S10_ROWS = ["et_gru", "et_lstm", "et_transformer", "fusion_mlp", "late_fusion",
            "dual_transformer", "cross_attention", "mm_transformer", "dynamicgat_et"]


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
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def fmt_p(p):
    return "$<0.001^{*}$" if p < 0.001 else (f"{p:.3f}$^{{*}}$" if p < 0.05 else f"{p:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-csv", default=str(FULL))
    ap.add_argument("--baseline-dir", default=str(BASE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rows", action="store_true",
                    help="also print mean ± SD rows (BalAcc, F1, MCC, ROC-AUC, PR-AUC, ECE) for Tables 3/4/5")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra per-fold CSVs to print as rows with --rows (e.g. abl_eeg_only, abl_full, production)")
    a = ap.parse_args()
    full = pd.read_csv(a.full_csv).set_index("test_subject")
    base_dir = Path(a.baseline_dir)
    files = {p.stem.replace("losocv_", ""): p for p in sorted(base_dir.glob("losocv_*.csv"))}
    files = {k: v for k, v in files.items() if k in NAMES}
    if not files:
        raise SystemExit(f"no baseline CSVs under {base_dir}")
    print(f"full model: {len(full)} folds   baselines: {len(files)} ({', '.join(files)})")

    rows = []
    for col, disp in METRICS:
        per = []
        for stem, path in files.items():
            b = pd.read_csv(path).set_index("test_subject")
            common = sorted(set(full.index) & set(b.index))
            f = full.loc[common, col].to_numpy(float)
            m = b.loc[common, col].to_numpy(float)
            ok = ~(np.isnan(f) | np.isnan(m))
            f, m = f[ok], m[ok]
            diff = f - m
            p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff, zero_method="zsplit").pvalue)
            if len(common) != len(full):
                print(f"  [warn] {stem}: {len(common)}/{len(full)} folds matched")
            per.append(dict(baseline=NAMES[stem], stem=stem, metric=col, n=int(ok.sum()),
                            full_mean=f.mean(), base_mean=m.mean(),
                            median_delta=float(np.median(diff)), wilcoxon_p=p,
                            cliffs_delta=cliffs_delta(f, m)))
        for r, ph in zip(per, holm([r["wilcoxon_p"] for r in per])):
            r["p_holm"] = float(ph)
        rows.extend(per)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")

    # ---- LaTeX rows for Table S10 (p_Holm, Cliff's delta per metric) ----
    n_fam = len(files)
    print(f"\n% Table S10 rows: full vs baseline, Holm over {n_fam} baselines within each metric, "
          f"Cliff's delta (>0 favours the full model)")
    for stem in S10_ROWS:
        if stem not in files:
            print(f"% {NAMES[stem]}: no CSV")
            continue
        cells = []
        for col, _ in METRICS:
            r = df[(df.stem == stem) & (df.metric == col)].iloc[0]
            cells.append(f"{fmt_p(r.p_holm)} & ${r.cliffs_delta:+.2f}$")
        print(f"{NAMES[stem]} & " + " & ".join(cells) + r" \\")

    if a.rows:
        cols = [("balanced_acc", "BalAcc"), ("f1", "Macro-F1"), ("mcc", "MCC"),
                ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("ece", "ECE")]

        def row(name, d):
            cells = []
            for c, _ in cols:
                if c not in d:
                    cells.append("--"); continue
                v = d[c].astype(float)
                cells.append(f"{v.mean():.2f}$\\pm${v.std():.2f}" if c != "ece" else f"{v.mean():.2f}")
            return f"{name} & " + " & ".join(cells) + f" \\\\   % n={len(d)}"

        groups = [("Table 3 (EEG encoders)", ["eegnet", "shallow", "deep", "cnn_lstm", "cnn_bilstm",
                                              "eeg_transformer", "tsception", "gat", "brain_gcn"]),
                  ("Table 4 (eye-tracking encoders)", ["et_lstm", "et_gru", "et_transformer"]),
                  ("Table 5 (multimodal fusion)", ["fusion_mlp", "late_fusion", "dual_transformer",
                                                   "cross_attention", "mm_transformer", "dynamicgat_et"])]
        print("\n% ---- mean ± SD over folds, uncalibrated operating point; columns "
              + " | ".join(n for _, n in cols))
        for title, stems in groups:
            print(f"% {title}")
            for stem in stems:
                if stem in files:
                    print(row(NAMES[stem], pd.read_csv(files[stem])))
        if a.extra:
            print("% extra per-fold CSVs")
            for e in a.extra:
                pe = Path(e)
                if pe.exists():
                    print(row(pe.stem.replace("losocv_", ""), pd.read_csv(pe)))
                else:
                    print(f"% {e}: not found")

    # ---- summary for Tables 4 / 5 (means + p) ----
    print("\n% means (full / baseline) and raw Wilcoxon p, balanced accuracy | MCC | ROC-AUC")
    for stem in files:
        parts = []
        for col, _ in METRICS:
            r = df[(df.stem == stem) & (df.metric == col)].iloc[0]
            parts.append(f"{r.base_mean:.3f} (p={r.wilcoxon_p:.3f}, Holm {r.p_holm:.3f})")
        print(f"{NAMES[stem]:24s} " + " | ".join(parts))


if __name__ == "__main__":
    main()
