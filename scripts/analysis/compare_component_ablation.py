"""
Component (leave-one-out) ablation — vs-full comparison & importance ranking
===========================================================================
Loads the blessed "full" production per-fold CSV and each ablation variant's
per-fold CSV, then for every variant computes the paired delta vs full
(Δ = variant − full) on each metric, plus a Wilcoxon signed-rank p-value over
the shared test subjects. Pure analysis — touches no training code.

Component importance is ranked by the drop in balanced accuracy when the
component is removed: a larger drop ⇒ the component contributes more.

Inputs:
    full baseline (default): output/metrics/focal_abl_g3p0_effective_num/
                             losocv_focal_abl_g3p0_effective_num.csv
    variants:                results/ablation/abl_<variant>/losocv_abl_<variant>.csv

Outputs (under results/ablation/):
    importance_ranking.csv          variants ranked by Δ balanced_accuracy
    delta_vs_full_<metric>.csv      per-variant Δ table, one file per metric
    component_ablation.md           human-readable report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# reuse the proven loaders / stats from the ET-ablation comparator
from compare_ablation import (
    METRIC_COLS, LOWER_IS_BETTER, _load, _metric_series, _agg, _paired_tests,
)

_ROOT = Path(__file__).resolve().parents[2]          # <repo>
DEFAULT_FULL = (_ROOT / "results" / "losocv_metrics"
                / "losocv_repro_focal_g3p0_effective_num_37.csv")   # published production run
ABL_ROOT = _ROOT / "results" / "ablation"

ALL_VARIANTS = [
    "no_snn", "no_graph", "no_neuro_symbolic", "no_et", "no_roi",
    "no_fusion_transformer", "no_contrastive", "no_mmd", "baseline_linear",
    "eeg_only", "ns_rule_only",                       # revision variants (present once run)
]
DISP = [d for d, _ in METRIC_COLS] + ["brier"]


def _variant_csv(variant: str) -> Path:
    return ABL_ROOT / f"abl_{variant}" / f"losocv_abl_{variant}.csv"


def compare_one(df_full, df_var, calibrated: bool) -> dict:
    """Return {metric: (full_mean, var_mean, delta, wilcoxon_p, n)}."""
    out = {}
    for disp, col in METRIC_COLS + [("brier", "brier")]:
        sF = _metric_series(df_full, disp, col, calibrated)
        sV = _metric_series(df_var,  disp, col, calibrated)
        paired = pd.concat([sF.rename("F"), sV.rename("V")], axis=1, join="inner")
        n, w_p, _ = _paired_tests(paired["F"].values, paired["V"].values)
        fmean, vmean = _agg(sF)["mean"], _agg(sV)["mean"]
        out[disp] = (fmean, vmean, vmean - fmean, w_p, n)
    return out


def main():
    ap = argparse.ArgumentParser(description="Compare component ablations vs full")
    ap.add_argument("--full-csv", type=str, default=str(DEFAULT_FULL),
                    help="blessed full-model per-fold CSV (the baseline)")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="subset of variants; default = all present on disk")
    ap.add_argument("--calibrated", action="store_true", default=True,
                    help="use *_cal deployment columns (default True)")
    ap.add_argument("--raw", dest="calibrated", action="store_false",
                    help="use uncalibrated columns instead")
    args = ap.parse_args()

    full_csv = Path(args.full_csv)
    df_full = _load(full_csv)
    print(f"Loaded FULL baseline: {len(df_full)} folds  ({full_csv.name})")

    variants = args.variants or [v for v in ALL_VARIANTS if _variant_csv(v).exists()]
    if not variants:
        print(f"No variant CSVs found under {ABL_ROOT}. Run run_component_ablation.sh first.")
        return

    rows = []
    per_metric = {d: [] for d in DISP}
    for v in variants:
        csv = _variant_csv(v)
        if not csv.exists():
            print(f"  [skip] {v}: missing {csv}")
            continue
        df_var = _load(csv)
        res = compare_one(df_full, df_var, args.calibrated)
        bacc = res["balanced_accuracy"]
        mcc  = res["mcc"]
        rows.append({
            "variant": v, "n_paired": bacc[4],
            "full_bal_acc": bacc[0], "var_bal_acc": bacc[1],
            "delta_bal_acc": bacc[2], "wilcoxon_p_bal_acc": bacc[3],
            "delta_mcc": mcc[2], "wilcoxon_p_mcc": mcc[3],
        })
        for d in DISP:
            fmean, vmean, delta, w_p, n = res[d]
            per_metric[d].append({
                "variant": v, "full_mean": fmean, "var_mean": vmean,
                "delta": delta, "wilcoxon_p": w_p, "n_paired": n,
            })

    # importance: removing a component that HELPS lowers bal-acc → delta negative.
    # rank by most-negative delta_bal_acc = most important component.
    rank = pd.DataFrame(rows).sort_values("delta_bal_acc").reset_index(drop=True)
    ABL_ROOT.mkdir(parents=True, exist_ok=True)
    rank.to_csv(ABL_ROOT / "importance_ranking.csv", index=False)
    for d in DISP:
        pd.DataFrame(per_metric[d]).to_csv(ABL_ROOT / f"delta_vs_full_{d}.csv", index=False)

    # ── console + markdown report ────────────────────────────────────────────
    tag = "calibrated" if args.calibrated else "raw"
    lines = []
    lines.append(f"# INS-HDGS-CMT — Component (leave-one-out) Ablation  [{tag}]\n")
    lines.append(f"**Baseline (full):** `{full_csv.parent.name}` · {len(df_full)} folds. "
                 f"Each variant disables ONE component; everything else identical "
                 f"(focal γ=3.0 · effective_num · n_ens=5 · 3-ch ET · same seed → paired).\n")
    lines.append("Δ = variant − full. Negative Δ on bal-acc/MCC ⇒ the component **helps** "
                 "(removing it hurts). Ranked by Δ balanced accuracy (most important first).\n")
    lines.append("| Component removed | full bal-acc | variant bal-acc | Δ bal-acc | Wilcoxon p | Δ MCC | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in rank.iterrows():
        lines.append(
            f"| {r['variant']} | {r['full_bal_acc']:.4f} | {r['var_bal_acc']:.4f} | "
            f"{r['delta_bal_acc']:+.4f} | {r['wilcoxon_p_bal_acc']:.4f} | "
            f"{r['delta_mcc']:+.4f} | {int(r['n_paired'])} |"
        )
    report = "\n".join(lines) + "\n"
    (ABL_ROOT / "component_ablation.md").write_text(report)

    print(f"\n=== Component importance (Δ balanced accuracy vs full, {tag}) ===")
    print(rank[["variant", "full_bal_acc", "var_bal_acc",
                "delta_bal_acc", "wilcoxon_p_bal_acc", "delta_mcc"]]
          .to_string(index=False))
    print(f"\nWrote: {ABL_ROOT}/component_ablation.md  (+ importance_ranking.csv, "
          f"delta_vs_full_*.csv)")


if __name__ == "__main__":
    main()
