"""
================================================================
Cross-modal contribution analysis (Reviewer 1, comments 1-3)
================================================================
Quantifies how much the eye-tracking (ET) pathway and the cross-modal
fusion contribute to the reported gains, using the SAME 37 held-out
subjects (paired, fold-matched) and the per-fold CSVs already produced
by evaluation/losocv.py.

Variants compared (all LOSOCV, identical folds):
  full        full INS-HDGS-CMT (EEG + ET + ROI + fusion)
  no_et       AblationConfig.no_et()   — ET sequence branch and fusion
              transformer removed; ROI dwell vector still supplied
  no_roi      AblationConfig.no_roi()  — ROI gating/modulation removed;
              ET sequence branch kept
  eeg_only    AblationConfig.eeg_only() — NO gaze-derived input at all
              (no ET sequence, no ROI vector, no fusion). Optional:
              supplied via --eeg-only-csv once that run exists.
  et_only     strongest ET-only encoder (ET-LSTM) from the baseline set,
              via its fold_probs file (for the "ET is a direct measure"
              question, comment 4).

For every pair the script reports the paired median difference, a
two-sided Wilcoxon signed-rank test, Holm correction within each metric
family, Cliff's delta, and a subject-level bootstrap CI of the mean
difference. It also reports the per-fold win/tie/loss counts.

Outputs
-------
  results/statistics/cross_modal_contribution.csv
  results/statistics/cross_modal_contribution.md

Usage
-----
  python scripts/analysis/cross_modal_contribution.py
  python scripts/analysis/cross_modal_contribution.py \
      --eeg-only-csv src/model/output/metrics/eeg_only/losocv_eeg_only.csv
================================================================
"""
from __future__ import annotations

import argparse
import ast
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "statistics"

METRICS = ["balanced_acc", "roc_auc", "mcc"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)[:, None]
    b = np.asarray(b)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size / 1.0) * 1.0) if a.size and b.size else float("nan")


def _paired_cliffs(d: np.ndarray) -> float:
    """Cliff's delta of paired differences against zero (P(d>0) - P(d<0))."""
    d = np.asarray(d, float)
    return float(((d > 0).mean() - (d < 0).mean()))


def _boot_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_losocv(csv: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df[["test_subject"] + [m for m in METRICS if m in df]].copy()
    df["variant"] = name
    return df


def _load_fold_probs(csv: Path, name: str) -> pd.DataFrame:
    """Baseline fold_probs file -> per-subject metrics at the argmax operating point."""
    d = pd.read_csv(csv)
    rows = []
    for subj, g in d.groupby("test_subject"):
        y, p = g["y_true"].values, g["p1"].values
        if len(np.unique(y)) < 2:
            continue
        pred = (p >= 0.5).astype(int)
        rows.append(dict(test_subject=subj,
                         balanced_acc=balanced_accuracy_score(y, pred),
                         roc_auc=roc_auc_score(y, p),
                         mcc=matthews_corrcoef(y, pred)))
    df = pd.DataFrame(rows)
    df["variant"] = name
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-csv", default=str(RES / "losocv_metrics" /
                    "losocv_repro_focal_g3p0_effective_num_37.csv"))
    ap.add_argument("--no-et-csv", default=str(RES / "ablation" / "abl_no_et" / "losocv_abl_no_et.csv"))
    ap.add_argument("--no-roi-csv", default=str(RES / "ablation" / "abl_no_roi" / "losocv_abl_no_roi.csv"))
    ap.add_argument("--no-fusion-csv", default=str(RES / "ablation" / "abl_no_fusion_transformer" /
                    "losocv_abl_no_fusion_transformer.csv"))
    ap.add_argument("--eeg-only-csv", default=None,
                    help="per-fold CSV of AblationConfig.eeg_only() (no gaze input at all)")
    ap.add_argument("--et-only-probs", default=str(RES / "baselines" / "dl" / "fold_probs" / "probs_et_lstm.csv"))
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    variants = {
        "full":   _load_losocv(Path(args.full_csv), "full"),
        "no_et":  _load_losocv(Path(args.no_et_csv), "no_et"),
        "no_roi": _load_losocv(Path(args.no_roi_csv), "no_roi"),
        "no_fusion": _load_losocv(Path(args.no_fusion_csv), "no_fusion"),
    }
    if args.eeg_only_csv and Path(args.eeg_only_csv).exists():
        variants["eeg_only"] = _load_losocv(Path(args.eeg_only_csv), "eeg_only")
    else:
        print("[info] eeg_only CSV not supplied/found — the no-gaze-input variant will "
              "be added once `NEUMA_LABEL=eeg_only` LOSOCV has been run.")
    if Path(args.et_only_probs).exists():
        variants["et_only(ET-LSTM)"] = _load_fold_probs(Path(args.et_only_probs), "et_only(ET-LSTM)")

    # align on common subjects
    common = set.intersection(*[set(v["test_subject"]) for v in variants.values()])
    common = sorted(common)
    print(f"[info] {len(common)} fold-matched subjects across {len(variants)} variants")
    wide = {m: pd.DataFrame({k: v.set_index("test_subject").loc[common, m].values
                             for k, v in variants.items()}, index=common) for m in METRICS}

    pairs = [("full", "no_et"), ("full", "no_roi"), ("full", "no_fusion"),
             ("no_et", "no_roi")]
    if "eeg_only" in variants:
        pairs += [("full", "eeg_only"), ("no_et", "eeg_only")]
    if "et_only(ET-LSTM)" in variants:
        pairs += [("full", "et_only(ET-LSTM)"), ("no_et", "et_only(ET-LSTM)")]

    rows = []
    for m in METRICS:
        W = wide[m]
        fam = []
        for a, b in pairs:
            d = W[a].values - W[b].values
            d = d[np.isfinite(d)]
            try:
                p = wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = 1.0
            lo, hi = _boot_ci(d)
            fam.append(dict(metric=m, a=a, b=b, n=len(d),
                            mean_a=float(W[a].mean()), mean_b=float(W[b].mean()),
                            mean_delta=float(d.mean()), median_delta=float(np.median(d)),
                            ci95_lo=lo, ci95_hi=hi, wilcoxon_p=float(p),
                            cliffs_delta=_paired_cliffs(d),
                            wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum())))
        ps = _holm([r["wilcoxon_p"] for r in fam])
        for r, ph in zip(fam, ps):
            r["p_holm"] = float(ph)
        rows.extend(fam)

    res = pd.DataFrame(rows)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "cross_modal_contribution.csv", index=False)

    # summary table of variant means
    means = pd.DataFrame({m: {k: float(wide[m][k].mean()) for k in variants} for m in METRICS})
    sds = pd.DataFrame({m: {k: float(wide[m][k].std()) for k in variants} for m in METRICS})

    lines = ["# Cross-modal contribution (paired, fold-matched LOSOCV, n=%d subjects)" % len(common), "",
             "Variant definitions: full = EEG+ET+ROI+fusion; no_et = ET sequence branch and fusion "
             "transformer removed (ROI dwell vector still supplied — this is the configuration "
             "tabulated as the 'EEG branch' in Table 3); no_roi = ROI gating/modulation removed "
             "(ET sequence kept); no_fusion = cross-modal transformer replaced by simple cross-attention; "
             "eeg_only = no gaze-derived input at all (AblationConfig.eeg_only()); "
             "et_only = ET-LSTM baseline (argmax operating point).", "",
             "## Per-variant means ± SD", "", "| variant | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for k in variants:
        lines.append(f"| {k} | " + " | ".join(f"{means.loc[k, m]:.3f} ± {sds.loc[k, m]:.3f}" for m in METRICS) + " |")
    lines += ["", "## Paired comparisons (Wilcoxon signed-rank, Holm within metric family)", "",
              "| metric | A | B | mean Δ (A−B) | median Δ | 95% CI | p | p(Holm) | Cliff's δ | W/T/L |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        star = " *" if r["p_holm"] < 0.05 else ""
        lines.append(f"| {r['metric']} | {r['a']} | {r['b']} | {r['mean_delta']:+.3f} | {r['median_delta']:+.3f} | "
                     f"[{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] | {r['wilcoxon_p']:.4f} | {r['p_holm']:.4f}{star} | "
                     f"{r['cliffs_delta']:+.2f} | {r['wins']}/{r['ties']}/{r['losses']} |")
    (out_dir / "cross_modal_contribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[saved] {out_dir / 'cross_modal_contribution.csv'}")


if __name__ == "__main__":
    main()
