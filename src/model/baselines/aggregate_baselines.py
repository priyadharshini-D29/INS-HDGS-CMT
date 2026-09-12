"""
Aggregate DL-baseline LOSOCV results into one comparison table and rank
them against the full INS-HDGS-CMT model.

IMPORTANT — fold-matched comparison:
  The DL baselines are run over every available subject (42 folds), but the
  blessed full model is evaluated on 37 subjects (5 subjects are excluded
  upstream). Comparing 42-fold baseline means against the 37-fold full model
  is apples-to-oranges. This script therefore restricts every baseline to the
  *exact* subject set the full model used, so all rows are on identical folds.
  Because folds are then matched 1:1 by subject, we also run a paired Wilcoxon
  signed-rank test (full vs each baseline) per metric.

Outputs (results/baselines/dl/):
  comparison_vs_full.csv     mean±std per model (fold-matched) + full model
  comparison_vs_full.md      human-readable, sorted by balanced acc, with the
                             paired-significance table appended
  paired_significance.csv    per-baseline Wilcoxon p-values & median deltas
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PHASE8 = Path(__file__).resolve().parents[1]
DL = PHASE8 / "results" / "baselines" / "dl"
FULL_CSV = PHASE8 / "results" / "losocv_metrics" / "losocv_repro_g0p0_balanced_37.csv"

METRICS = ["balanced_acc", "f1", "roc_auc", "mcc", "accuracy", "ece"]
# metrics used for the paired full-vs-baseline significance test
PAIRED_METRICS = ["balanced_acc", "mcc", "f1", "roc_auc"]
SUBJ_COL = "test_subject"


def summarize(df, name):
    row = {"model": name, "n_folds": len(df)}
    for k in METRICS:
        if k in df:
            v = pd.to_numeric(df[k], errors="coerce").dropna()
            row[f"{k}_mean"] = float(v.mean()) if len(v) else float("nan")
            row[f"{k}_std"] = float(v.std(ddof=0)) if len(v) else float("nan")
    return row


def paired_tests(full_df, base_df, base_name, subjects):
    """Wilcoxon signed-rank, full vs baseline, paired by subject."""
    f = full_df.set_index(SUBJ_COL)
    b = base_df.set_index(SUBJ_COL)
    out = []
    for k in PAIRED_METRICS:
        if k not in f or k not in b:
            continue
        fv = pd.to_numeric(f.loc[subjects, k], errors="coerce")
        bv = pd.to_numeric(b.loc[subjects, k], errors="coerce")
        pair = pd.concat([fv.rename("full"), bv.rename("base")], axis=1).dropna()
        if len(pair) < 2:
            continue
        diff = pair["full"].to_numpy() - pair["base"].to_numpy()
        median_delta = float(np.median(diff))   # full − baseline; >0 ⇒ full better
        if np.allclose(diff, 0):
            p = 1.0
        else:
            # zsplit keeps zero-diffs from collapsing the test
            p = float(wilcoxon(diff, zero_method="zsplit").pvalue)
        out.append({"baseline": base_name, "metric": k, "n_pairs": len(pair),
                    "full_mean": float(pair["full"].mean()),
                    "base_mean": float(pair["base"].mean()),
                    "median_delta_full_minus_base": median_delta,
                    "wilcoxon_p": p})
    return out


def main():
    global DL, FULL_CSV
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate DL-baseline LOSOCV CSVs and test each against the full model")
    ap.add_argument("--dir", default=str(DL), help="results/baselines/dl or results/baselines/dl_tuned")
    ap.add_argument("--full-csv", default=str(PHASE8 / "results" / "losocv_metrics" /
                                              "losocv_repro_focal_g3p0_effective_num_37.csv"))
    args = ap.parse_args()
    DL, FULL_CSV = Path(args.dir), Path(args.full_csv)
    if not FULL_CSV.exists():
        print(f"  [fatal] full-model CSV not found: {FULL_CSV}"); return
    full = pd.read_csv(FULL_CSV)
    if SUBJ_COL not in full:
        print(f"  [fatal] full-model CSV missing '{SUBJ_COL}' column"); return
    subjects = sorted(full[SUBJ_COL].astype(str).unique())
    print(f"[fold-match] full model defines {len(subjects)} subjects; "
          f"restricting all baselines to these folds.")

    rows, sig_rows = [], []
    for csv in sorted(DL.glob("losocv_*.csv")):
        name = csv.stem.replace("losocv_", "")
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"  skip {csv.name}: {e}"); continue
        if SUBJ_COL not in df:
            print(f"  skip {name}: no '{SUBJ_COL}' column"); continue
        df[SUBJ_COL] = df[SUBJ_COL].astype(str)
        n_before = len(df)
        df = df[df[SUBJ_COL].isin(subjects)].drop_duplicates(SUBJ_COL)
        if len(df) != len(subjects):
            print(f"  [warn] {name}: {len(df)}/{len(subjects)} matched "
                  f"(had {n_before}); missing "
                  f"{sorted(set(subjects) - set(df[SUBJ_COL]))}")
        rows.append(summarize(df, name))
        sig_rows.extend(paired_tests(full, df, name, subjects))

    rows.append(summarize(full, "INS-HDGS-CMT (full)"))

    out = pd.DataFrame(rows).sort_values("balanced_acc_mean", ascending=False)
    out.to_csv(DL / "comparison_vs_full.csv", index=False)

    sig = pd.DataFrame(sig_rows)
    if not sig.empty:
        sig.to_csv(DL / "paired_significance.csv", index=False)

    # ---- markdown ----
    lines = [f"# DL baselines vs INS-HDGS-CMT (LOSOCV, fold-matched n={len(subjects)})\n",
             "All models scored on the **identical** subject set the full model "
             "was evaluated on, so rows are directly comparable.\n",
             "| model | folds | bal_acc | f1 | roc_auc | mcc | acc | ece |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        def cell(k):
            m, s = r.get(f"{k}_mean", float("nan")), r.get(f"{k}_std", float("nan"))
            return f"{m:.3f}±{s:.3f}" if m == m else "—"
        lines.append(f"| {r['model']} | {int(r['n_folds'])} | "
                     f"{cell('balanced_acc')} | {cell('f1')} | {cell('roc_auc')} | "
                     f"{cell('mcc')} | {cell('accuracy')} | {cell('ece')} |")

    if not sig.empty:
        lines += ["",
                  "## Paired significance — full vs each baseline (Wilcoxon signed-rank)\n",
                  "Δ = full − baseline (per subject); positive ⇒ full model better. "
                  "p from two-sided Wilcoxon on paired folds.\n",
                  "| baseline | metric | n | full | baseline | median Δ | p |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in sig.iterrows():
            star = " *" if r["wilcoxon_p"] < 0.05 else ""
            lines.append(f"| {r['baseline']} | {r['metric']} | {int(r['n_pairs'])} | "
                         f"{r['full_mean']:.3f} | {r['base_mean']:.3f} | "
                         f"{r['median_delta_full_minus_base']:+.3f} | "
                         f"{r['wilcoxon_p']:.4f}{star} |")

    (DL / "comparison_vs_full.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[aggregate] → {DL/'comparison_vs_full.csv'}")
    if not sig.empty:
        print(f"[aggregate] → {DL/'paired_significance.csv'}")


if __name__ == "__main__":
    main()
