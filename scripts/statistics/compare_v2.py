#!/usr/bin/env python3
"""
Protocol-v2 robustness check: does the fold-wise label (and the matched validation
participant) change any conclusion?

For every run that exists in both protocols it prints, over the held-out subjects the
two runs share, the mean of each metric under the published protocol (pooled label,
legacy validation subject) and under protocol v2 (NEUMA_LABEL_FOLDWISE=1,
NEUMA_VAL_RULE=matched), the paired difference with a bootstrap 95 % CI, and the paired
Wilcoxon p (zero differences discarded, the convention of every paired test in the paper).

Decision rule (docs/REVISION_RUNS.md, "V2 PLAN"): a v2 run replaces its v1 counterpart as
the reported run when the difference is smaller than the run-to-run spread of the same
configuration, which is measured by the replicates in
results/statistics/full_model_replicates.csv (balanced accuracy 0.718-0.748, ROC-AUC
0.863-0.879, i.e. a spread of 0.030 and 0.016).  The verdict per metric is printed.

Nothing is trained: this reads the saved per-fold CSVs only.

Run:  python scripts/revision/compare_v2.py            (repository root, CPU, seconds)
      python scripts/revision/compare_v2.py --out-dir results/statistics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
ABL = ROOT / "results" / "ablation"
BASE = ROOT / "results" / "baselines"
REPLICATES = ROOT / "results" / "statistics" / "full_model_replicates.csv"

METRICS = [("balanced_acc", "BalAcc"), ("roc_auc", "ROC-AUC"), ("mcc", "MCC")]

# (label, v1 csv, v2 csv)
PAIRS = [
    ("Full model", ABL / "abl_full/losocv_abl_full.csv", ABL / "abl_full_v2/losocv_abl_full_v2.csv"),
    ("Gaze-free EEG branch", ABL / "abl_eeg_only/losocv_abl_eeg_only.csv", ABL / "abl_eeg_only_v2/losocv_abl_eeg_only_v2.csv"),
    ("Gaze-free branch, MMD kept", ABL / "abl_eeg_only_mmd/losocv_abl_eeg_only_mmd.csv", ABL / "abl_eeg_only_mmd_v2/losocv_abl_eeg_only_mmd_v2.csv"),
    ("EEG-only, gaze teacher", ABL / "abl_eeg_only_et_teacher/losocv_abl_eeg_only_et_teacher.csv", ABL / "abl_eeg_only_et_teacher_v2/losocv_abl_eeg_only_et_teacher_v2.csv"),
    ("Full model, single member", ABL / "abl_full_single/losocv_abl_full_single.csv", ABL / "abl_full_single_v2/losocv_abl_full_single_v2.csv"),
]
BASELINES = [("ET-LSTM", "et_lstm"), ("CNN-BiLSTM", "cnn_bilstm"),
             ("DynamicGAT+ET Transf.", "dynamicgat_et"), ("Cross-Attention", "cross_attention"),
             ("ET-GRU", "et_gru"), ("ET-Transformer", "et_transformer"),
             ("Early-Fusion MLP", "fusion_mlp"), ("Late Fusion", "late_fusion"),
             ("Dual Transformer", "dual_transformer"), ("Multimodal Transformer", "mm_transformer"),
             ("EEGNet", "eegnet"), ("ShallowConvNet", "shallow"), ("DeepConvNet", "deep"),
             ("CNN-LSTM", "cnn_lstm"), ("EEG Transformer", "eeg_transformer"),
             ("TSception", "tsception"), ("GAT", "gat"), ("BrainGCN", "brain_gcn")]
for _name, _stem in BASELINES:
    PAIRS.append((_name, BASE / f"dl_tuned/losocv_{_stem}.csv", BASE / f"dl_tuned_v2/losocv_{_stem}.csv"))


def load(csv: Path) -> pd.DataFrame | None:
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    return d.drop_duplicates("test_subject", keep="last").set_index("test_subject")


def boot_ci(d: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def replicate_spread() -> dict[str, float]:
    """Run-to-run range of the same configuration, from the saved replicate runs."""
    if not REPLICATES.exists():
        return {}
    r = pd.read_csv(REPLICATES)
    r = r[r["run"].astype(str).str.startswith("abl_full")]     # same montage, same seed, same code
    return {m: float(r[m].max() - r[m].min()) for m, _ in METRICS if m in r.columns}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "statistics"))
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    spread = replicate_spread()
    recs, missing = [], []
    for label, v1_csv, v2_csv in PAIRS:
        a, b = load(v1_csv), load(v2_csv)
        if a is None or b is None:
            missing.append((label, v1_csv if a is None else v2_csv))
            continue
        common = sorted(set(a.index) & set(b.index))
        for m, mname in METRICS:
            if m not in a.columns or m not in b.columns:
                continue
            x = a.loc[common, m].astype(float).to_numpy()
            y = b.loc[common, m].astype(float).to_numpy()
            d = y - x                                          # v2 minus published protocol
            lo, hi = boot_ci(d, args.n_boot)
            try:
                p = float(wilcoxon(y, x, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                p = float("nan")
            sp = spread.get(m)
            verdict = ("--" if sp is None
                       else "within run-to-run spread" if abs(d.mean()) <= sp
                       else "LARGER than run-to-run spread")
            recs.append(dict(run=label, metric=mname, n=len(common), v1=x.mean(), v2=y.mean(),
                             delta=d.mean(), ci_lo=lo, ci_hi=hi, wilcoxon_p=p,
                             replicate_spread=sp, verdict=verdict,
                             wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum())))
    if not recs:
        print("no run exists under both protocols yet")
        for label, csv in missing:
            print(f"  [missing] {label}: {csv.relative_to(ROOT)}")
        return

    df = pd.DataFrame(recs)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "protocol_v2_comparison.csv", index=False)

    lines = ["# Protocol-v2 robustness check (fold-wise label + matched validation participant)", "",
             "Paired over the held-out subjects the two runs share; delta = v2 minus published protocol; "
             "95 % CI from 10,000 bootstrap resamples of the mean; paired Wilcoxon with zero differences discarded.",
             ""]
    if spread:
        lines += ["Run-to-run spread of the same configuration (replicates of the full model): "
                  + ", ".join(f"{mn} {spread[m]:.3f}" for m, mn in METRICS if m in spread), ""]
    lines += ["| Run | Metric | n | published | v2 | delta [95% CI] | p | W/T/L | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.run} | {r.metric} | {r.n} | {r.v1:.3f} | {r.v2:.3f} | "
                     f"{r.delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.wilcoxon_p:.3f} | "
                     f"{r.wins}/{r.ties}/{r.losses} | {r.verdict} |")
    if missing:
        lines += ["", "Not yet available under both protocols:"] + [f"- {lab} (`{c.relative_to(ROOT)}`)" for lab, c in missing]
    (out / "protocol_v2_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pd.set_option("display.width", 200)
    print(df[["run", "metric", "n", "v1", "v2", "delta", "ci_lo", "ci_hi", "wilcoxon_p", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    if missing:
        print("\nnot yet available under both protocols:")
        for lab, c in missing:
            print(f"  {lab}: {c.relative_to(ROOT)}")
    print(f"\nwrote {out / 'protocol_v2_comparison.md'} and .csv")


if __name__ == "__main__":
    main()
