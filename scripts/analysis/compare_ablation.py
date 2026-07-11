"""
A vs C ET-ablation — statistical comparison
===========================================================================
Loads the per-fold LOSOCV CSVs produced by run_ablation_experiment.py and
produces a rigorous, paired comparison. Pure analysis — touches no training,
model, loss, optimizer, or calibration code.

Inputs (default):
    results/baseline_A/losocv_baseline_A.csv
    results/experiment_C/losocv_experiment_C.csv

Outputs (under results/):
    baseline_A/aggregate_baseline_A.csv        mean/std/min/max per metric
    experiment_C/aggregate_experiment_C.csv
    baseline_A/subjectwise_baseline_A.csv       per-subject (=per-fold) metrics
    experiment_C/subjectwise_experiment_C.csv
    comparison_A_vs_C_calibrated.csv            A | C | Delta | wilcoxon_p | ttest_p
    comparison_A_vs_C_raw.csv
    foldwise_paired.csv                         per-subject A & C side-by-side
    comparison_A_vs_C.md                        human-readable report

Metrics: accuracy, balanced_accuracy, f1, precision, recall, mcc, ece, brier.
  - calibrated table uses the *_cal columns (deployment metrics);
  - raw table uses the uncalibrated columns;
  - brier is computed from the stored (uncalibrated, ensemble-averaged) y_prob
    in BOTH tables (no calibrated probability is persisted by the pipeline).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# metric display-name → raw CSV column (calibrated adds the _cal suffix)
METRIC_COLS = [
    ("accuracy",          "accuracy"),
    ("balanced_accuracy", "balanced_acc"),
    ("f1",                "f1"),
    ("precision",         "precision"),
    ("recall",            "recall"),
    ("mcc",               "mcc"),
    ("ece",               "ece"),
]
LOWER_IS_BETTER = {"ece", "brier"}


def _parse_floats(s) -> np.ndarray:
    """Parse a stored python-list string like '[0.1, 0.9, nan]' → float array."""
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=float)
    if not isinstance(s, str):
        return np.array([], dtype=float)
    out = []
    for tok in s.strip().lstrip("[").rstrip("]").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def _brier_per_fold(df: pd.DataFrame) -> np.ndarray:
    """Brier score per fold from stored y_true / y_prob (P(HIGH))."""
    briers = []
    for _, row in df.iterrows():
        y = _parse_floats(row.get("y_true"))
        p = _parse_floats(row.get("y_prob"))
        if len(y) == 0 or len(y) != len(p):
            briers.append(np.nan)
        else:
            briers.append(float(np.mean((p - y) ** 2)))
    return np.asarray(briers, dtype=float)


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing per-fold CSV: {path}\n"
            f"  Run it first:  python analysis/run_ablation_experiment.py --exp "
            f"{'A' if 'baseline' in path.name else 'C'}"
        )
    df = df0 = pd.read_csv(path)
    df = df.copy()
    df["brier"] = _brier_per_fold(df0)
    return df


def _metric_series(df: pd.DataFrame, disp: str, col: str, calibrated: bool) -> pd.Series:
    """Return the per-fold series for a metric, indexed by test_subject."""
    if disp == "brier":
        c = "brier"                       # same uncalibrated brier for both tables
    else:
        c = f"{col}_cal" if calibrated else col
    if c not in df.columns:
        return pd.Series(dtype=float)
    return pd.Series(df[c].values, index=df["test_subject"].values, name=disp)


def _agg(series: pd.Series) -> dict:
    v = series.dropna().values
    if len(v) == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, n=0)
    return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1) if len(v) > 1 else 0.0),
                min=float(np.min(v)), max=float(np.max(v)), n=int(len(v)))


def _paired_tests(a: np.ndarray, b: np.ndarray):
    """Wilcoxon signed-rank + paired t-test on matched, non-nan pairs."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    w_p = t_p = np.nan
    if n >= 2:
        try:
            if np.any(a - b):                       # wilcoxon needs nonzero diffs
                w_p = float(stats.wilcoxon(a, b).pvalue)
            else:
                w_p = 1.0
        except ValueError:
            w_p = np.nan
        try:
            t_p = float(stats.ttest_rel(a, b).pvalue)
        except ValueError:
            t_p = np.nan
    return n, w_p, t_p


def build_comparison(dfA, dfC, calibrated: bool) -> pd.DataFrame:
    rows = []
    disp_cols = METRIC_COLS + [("brier", "brier")]
    for disp, col in disp_cols:
        sA = _metric_series(dfA, disp, col, calibrated)
        sC = _metric_series(dfC, disp, col, calibrated)
        aggA, aggC = _agg(sA), _agg(sC)
        # pair on shared test_subject
        paired = pd.concat([sA.rename("A"), sC.rename("C")], axis=1, join="inner")
        n, w_p, t_p = _paired_tests(paired["A"].values, paired["C"].values)
        delta = aggC["mean"] - aggA["mean"]
        better = ("C" if (delta < 0) else "A") if disp in LOWER_IS_BETTER \
                 else ("C" if (delta > 0) else "A")
        rows.append({
            "metric": disp,
            "A_mean": aggA["mean"], "A_std": aggA["std"],
            "C_mean": aggC["mean"], "C_std": aggC["std"],
            "delta_C_minus_A": delta,
            "better": better if np.isfinite(delta) and delta != 0 else "tie",
            "n_paired": n, "wilcoxon_p": w_p, "ttest_rel_p": t_p,
        })
    return pd.DataFrame(rows)


def aggregate_table(df: pd.DataFrame, calibrated: bool) -> pd.DataFrame:
    rows = []
    for disp, col in METRIC_COLS + [("brier", "brier")]:
        s = _metric_series(df, disp, col, calibrated)
        a = _agg(s)
        rows.append({"metric": disp, **a})
    return pd.DataFrame(rows)


def subjectwise_table(df: pd.DataFrame, calibrated: bool) -> pd.DataFrame:
    cols = ["test_subject"]
    out = df[["test_subject"]].copy()
    for disp, col in METRIC_COLS + [("brier", "brier")]:
        s = _metric_series(df, disp, col, calibrated)
        out[disp] = s.values
    return out


def _fmt(x, p=4):
    return "nan" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.results_root) if args.results_root else (Path(__file__).resolve().parents[1] / "results")
    pA = root / "baseline_A"   / "losocv_baseline_A.csv"
    pC = root / "experiment_C" / "losocv_experiment_C.csv"
    dfA, dfC = _load(pA), _load(pC)
    print(f"Loaded A: {len(dfA)} folds | C: {len(dfC)} folds")

    # ── aggregates + subject-wise (per experiment) ──────────────────────────
    for tag, df, sub in [("baseline_A", dfA, pA.parent), ("experiment_C", dfC, pC.parent)]:
        aggregate_table(df, calibrated=True).to_csv(sub / f"aggregate_{tag}.csv", index=False)
        subjectwise_table(df, calibrated=True).to_csv(sub / f"subjectwise_{tag}.csv", index=False)

    # ── comparison tables (calibrated = primary, raw = secondary) ───────────
    cmp_cal = build_comparison(dfA, dfC, calibrated=True)
    cmp_raw = build_comparison(dfA, dfC, calibrated=False)
    cmp_cal.to_csv(root / "comparison_A_vs_C_calibrated.csv", index=False)
    cmp_raw.to_csv(root / "comparison_A_vs_C_raw.csv", index=False)

    # ── paired fold-level table (subjects shared by both) ───────────────────
    paired_frames = []
    for disp, col in METRIC_COLS + [("brier", "brier")]:
        sA = _metric_series(dfA, disp, col, True).rename(f"{disp}_A")
        sC = _metric_series(dfC, disp, col, True).rename(f"{disp}_C")
        paired_frames.append(pd.concat([sA, sC], axis=1, join="inner"))
    paired = pd.concat(paired_frames, axis=1)
    paired.index.name = "test_subject"
    paired.to_csv(root / "foldwise_paired.csv")

    # ── markdown report ─────────────────────────────────────────────────────
    lines = ["# ET Ablation — Baseline A vs Experiment C (LOSOCV)", ""]
    lines += [f"- Folds: A={len(dfA)}, C={len(dfC)}",
              "- A = left eye only (ET dim 3); C = both eyes + vergence + speed (ET dim 9)",
              "- Identical model / losses / optimizer / scheduler / calibration; same seed → paired folds",
              "- Primary table = **calibrated** metrics; Δ = C − A; *for ece/brier lower is better*", ""]
    lines += ["## Calibrated metrics", "",
              "| Metric | Baseline A (mean±std) | Experiment C (mean±std) | Δ (C−A) | Better | Wilcoxon p | t-test p |",
              "|---|---|---|---|---|---|---|"]
    for _, r in cmp_cal.iterrows():
        lines.append(
            f"| {r['metric']} | {_fmt(r['A_mean'])} ± {_fmt(r['A_std'])} "
            f"| {_fmt(r['C_mean'])} ± {_fmt(r['C_std'])} | {_fmt(r['delta_C_minus_A'],4)} "
            f"| {r['better']} | {_fmt(r['wilcoxon_p'],4)} | {_fmt(r['ttest_rel_p'],4)} |")
    lines += ["", "## Raw (uncalibrated) metrics", "",
              "| Metric | Baseline A | Experiment C | Δ (C−A) | Better | Wilcoxon p | t-test p |",
              "|---|---|---|---|---|---|---|"]
    for _, r in cmp_raw.iterrows():
        lines.append(
            f"| {r['metric']} | {_fmt(r['A_mean'])} | {_fmt(r['C_mean'])} "
            f"| {_fmt(r['delta_C_minus_A'],4)} | {r['better']} "
            f"| {_fmt(r['wilcoxon_p'],4)} | {_fmt(r['ttest_rel_p'],4)} |")
    lines += ["", "_Brier uses the stored uncalibrated ensemble-averaged probabilities in both tables._",
              "_Significance at n≈30+ paired folds; treat p-values as guidance given small per-fold test n._"]
    (root / "comparison_A_vs_C.md").write_text("\n".join(lines))

    # ── console summary ──────────────────────────────────────────────────────
    print("\n=== Calibrated comparison (Δ = C − A) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(cmp_cal[["metric", "A_mean", "C_mean", "delta_C_minus_A",
                       "better", "wilcoxon_p", "ttest_rel_p"]].to_string(index=False))
    print(f"\nWrote outputs under {root}/")
    print("  comparison_A_vs_C.md  (+ *_calibrated.csv, *_raw.csv, foldwise_paired.csv,")
    print("   aggregate_*.csv, subjectwise_*.csv)")


if __name__ == "__main__":
    main()
