#!/usr/bin/env python3
"""
Paired effect sizes for every fold-matched comparison in the manuscript
tables, from the SAVED per-fold LOSOCV CSVs (nothing is trained).

For paired per-fold differences d_i = A_i - B_i (matched on test_subject):

  r_rb     matched-pairs rank-biserial correlation from the Wilcoxon
           signed-rank statistic, r_rb = (W+ - W-) / (W+ + W-), zero
           differences dropped before ranking (Kerby 2014);
  P(sup)   paired probability of superiority P(d > 0) - P(d < 0) over ALL
           pairs, ties included -- identical to the "Cliff's delta of the
           paired differences" reported in Supplementary Table S14
           (cross_modal_contribution.py::_paired_cliffs);
  p        two-sided Wilcoxon signed-rank test, zero differences discarded
           (scipy zero_method="wilcox"); the zsplit p used by
           verify_table7_eeg_significance.py / table_s10_full_vs_baselines.py
           is listed beside it for reference;
  mean d, median d, subject-level bootstrap 95% CI of the mean difference
           (10 000 resamples, seed 0), win/tie/loss counts.

The between-distribution Cliff's delta of Tables 9 / S6 / S10
(cliffs_delta(A, B) over all 37 x 37 fold pairs) is recomputed and the value
currently on file in results/statistics is placed beside it, so the tables
can be updated with r_rb without re-deriving anything.

Comparisons
-----------
  (a) gaze-free branch abl_eeg_only vs each of the 8 tuned EEG baselines
      (Table 9 balanced accuracy; Table S6 MCC / ROC-AUC), in Table 9 order;
  (b) published full model vs each of the 18 tuned baselines (Table S10 and
      the significance columns of Tables 5 / 6), S10 rows first, then the
      EEG encoders in Table 9 order and BrainGCN;
  (c) full model vs the pathway variants (Table S14 layout) with BOTH the
      revision re-run abl_full (the Table 8 reference) and the published run
      (the Table S14 reference) as A, plus the tuned ET-LSTM at its argmax
      operating point (as in S14) and the direct pair
      abl_eeg_only_mmd vs abl_eeg_only.

Usage
-----
  python scripts/analysis/paired_effect_sizes.py
Outputs results/statistics/paired_effect_sizes.md and .csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
STATS = RES / "statistics"
ABL = RES / "ablation"
TUNED = RES / "baselines" / "dl_tuned"
PUBLISHED = RES / "losocv_metrics" / "losocv_repro_focal_g3p0_effective_num_37.csv"

METRICS = [("balanced_acc", "Balanced accuracy"), ("mcc", "MCC"), ("roc_auc", "ROC-AUC")]

# Table 9 order (manuscript tab8) -- file stems under results/baselines/dl_tuned
EEG_ORDER = [("eegnet", "EEGNet"), ("gat", "GAT"), ("eeg_transformer", "EEG Transformer"),
             ("cnn_lstm", "CNN-LSTM"), ("tsception", "TSception"), ("cnn_bilstm", "CNN-BiLSTM"),
             ("deep", "DeepConvNet"), ("shallow", "ShallowConvNet")]
# Table S10 row order, then the EEG family and BrainGCN (all 18 form the S10 Holm family)
S10_ORDER = [("et_gru", "ET-GRU"), ("et_lstm", "ET-LSTM"), ("et_transformer", "ET-Transformer"),
             ("fusion_mlp", "Early-Fusion MLP"), ("late_fusion", "Late Fusion"),
             ("dual_transformer", "Dual Transformer"), ("cross_attention", "Cross-Attention"),
             ("mm_transformer", "Multimodal Transformer"), ("dynamicgat_et", "DynamicGAT+ET Transf.")]
ALL18_ORDER = S10_ORDER + EEG_ORDER + [("brain_gcn", "BrainGCN")]
# Table S14 rows: (variant key, S14 label, per-fold CSV)
VARIANTS = [("no_et", "-gaze seq. (abl_no_et)", ABL / "abl_no_et" / "losocv_abl_no_et.csv"),
            ("no_roi", "-ROI (abl_no_roi)", ABL / "abl_no_roi" / "losocv_abl_no_roi.csv"),
            ("no_fusion", "-fusion (abl_no_fusion_transformer)",
             ABL / "abl_no_fusion_transformer" / "losocv_abl_no_fusion_transformer.csv"),
            ("eeg_only", "EEG-only (abl_eeg_only)", ABL / "abl_eeg_only" / "losocv_abl_eeg_only.csv"),
            ("eeg_only_mmd", "EEG-only, MMD/DANN kept (abl_eeg_only_mmd)",
             ABL / "abl_eeg_only_mmd" / "losocv_abl_eeg_only_mmd.csv")]


# ── effect sizes ──────────────────────────────────────────────────────────────

def rank_biserial(d: np.ndarray) -> tuple[float, float, float]:
    """Matched-pairs rank-biserial r_rb = (W+ - W-)/(W+ + W-), zeros dropped."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if d.size == 0:
        return float("nan"), 0.0, 0.0
    r = rankdata(np.abs(d))
    w_pos = float(r[d > 0].sum()); w_neg = float(r[d < 0].sum())
    return (w_pos - w_neg) / (w_pos + w_neg), w_pos, w_neg


def paired_superiority(d: np.ndarray) -> float:
    """P(d > 0) - P(d < 0) over all pairs incl. ties (= Table S14's paired Cliff's delta)."""
    d = np.asarray(d, float)
    return float((d > 0).mean() - (d < 0).mean())


def cliffs_delta_between(a: np.ndarray, b: np.ndarray) -> float:
    """Between-distribution Cliff's delta over all len(a) x len(b) pairs (Tables 9 / S6 / S10)."""
    a = np.asarray(a, float)[:, None]; b = np.asarray(b, float)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


def wilcoxon_p(d: np.ndarray, zero_method: str) -> float:
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, zero_method=zero_method, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def boot_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(a: pd.Series, b: pd.Series, n_boot: int, seed: int) -> dict:
    common = sorted(set(a.index) & set(b.index))
    x = a.loc[common].to_numpy(float); y = b.loc[common].to_numpy(float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    d = x - y
    r_rb, w_pos, w_neg = rank_biserial(d)
    lo, hi = boot_ci(d, n_boot, seed)
    return dict(n=int(len(d)), mean_a=float(x.mean()), mean_b=float(y.mean()),
                mean_delta=float(d.mean()), median_delta=float(np.median(d)), ci95_lo=lo, ci95_hi=hi,
                p_wilcox=wilcoxon_p(d, "wilcox"), p_zsplit=wilcoxon_p(d, "zsplit"),
                r_rb=r_rb, w_plus=w_pos, w_minus=w_neg, paired_superiority=paired_superiority(d),
                cliffs_between=cliffs_delta_between(x, y),
                wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()))


# ── loaders ───────────────────────────────────────────────────────────────────

def load_metric(csv: Path, metric: str) -> pd.Series:
    df = pd.read_csv(csv)
    return df.set_index("test_subject")[metric].astype(float)


def load_fold_probs(csv: Path) -> pd.DataFrame:
    """Baseline fold_probs -> per-subject metrics at the argmax (0.5) operating point,
    as cross_modal_contribution.py does for the ET-LSTM row of Table S14."""
    d = pd.read_csv(csv)
    rows = []
    for subj, g in d.groupby("test_subject"):
        y, p = g["y_true"].to_numpy(int), g["p1"].to_numpy(float)
        if len(np.unique(y)) < 2:
            continue
        pred = (p >= 0.5).astype(int)
        rows.append(dict(test_subject=subj, balanced_acc=balanced_accuracy_score(y, pred),
                         roc_auc=roc_auc_score(y, p), mcc=matthews_corrcoef(y, pred)))
    return pd.DataFrame(rows).set_index("test_subject")


def existing_table7(metric: str) -> pd.DataFrame | None:
    f = STATS / f"table7_eeg_significance_tuned_{metric}.csv"
    return pd.read_csv(f).set_index("baseline") if f.exists() else None


def existing_s10() -> pd.DataFrame | None:
    f = STATS / "tableS10_full_vs_baselines.csv"
    return pd.read_csv(f).set_index(["stem", "metric"]) if f.exists() else None


def existing_s14() -> pd.DataFrame | None:
    f = STATS / "cross_modal_contribution.csv"
    return pd.read_csv(f).set_index(["metric", "a", "b"]) if f.exists() else None


# ── formatting ────────────────────────────────────────────────────────────────

def fmt(v, nd=3, sign=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def fmt_p(p):
    return "--" if p is None or np.isnan(p) else (f"{p:.2e}" if p < 0.001 else f"{p:.3f}")


HEAD = ("| Metric | A | B | n | mean A | mean B | mean d [95% CI] | median d | W/T/L | r_rb | P(sup) (paired delta) | "
        "p (zeros discarded) | p (zsplit) | Cliff's delta (between) recomputed | existing Cliff's delta on file | existing p | existing p_Holm |")
SEP = "|" + "---|" * 17


def row_md(r: dict) -> str:
    return (f"| {r['metric_name']} | {r['A']} | {r['B']} | {r['n']} | {fmt(r['mean_a'], sign=False)} | {fmt(r['mean_b'], sign=False)} | "
            f"{fmt(r['mean_delta'])} [{fmt(r['ci95_lo'])}, {fmt(r['ci95_hi'])}] | {fmt(r['median_delta'])} | "
            f"{r['wins']}/{r['ties']}/{r['losses']} | {fmt(r['r_rb'], 2)} | {fmt(r['paired_superiority'], 2)} | "
            f"{fmt_p(r['p_wilcox'])} | {fmt_p(r['p_zsplit'])} | {fmt(r['cliffs_between'], 2)} | "
            f"{fmt(r.get('existing_cliffs'), 2)} | {fmt_p(r.get('existing_p', float('nan')))} | "
            f"{fmt_p(r.get('existing_p_holm', float('nan')))} |")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(STATS))
    args = ap.parse_args()

    rows: list[dict] = []
    sections: list[tuple[str, str, list[dict]]] = []

    # (a) gaze-free branch vs 8 tuned EEG baselines (Table 9 / S6)
    eeg_only = ABL / "abl_eeg_only" / "losocv_abl_eeg_only.csv"
    sec = []
    for col, mname in METRICS:
        ref = load_metric(eeg_only, col)
        ex = existing_table7(col)
        for stem, name in EEG_ORDER:
            r = compare(ref, load_metric(TUNED / f"losocv_{stem}.csv", col), args.n_boot, args.seed)
            r.update(section="a", metric=col, metric_name=mname, A="abl_eeg_only", B=name)
            if ex is not None and name in ex.index:
                r.update(existing_cliffs=float(ex.loc[name, "cliffs_delta"]), existing_p=float(ex.loc[name, "wilcoxon_p"]),
                         existing_p_holm=float(ex.loc[name, "p_holm"]))
            sec.append(r)
    sections.append(("(a) Gaze-free EEG branch (abl_eeg_only) vs tuned EEG baselines -- Table 9 (balanced accuracy) and Table S6 (MCC, ROC-AUC)",
                     "Existing values from results/statistics/table7_eeg_significance_tuned_<metric>.csv "
                     "(between-distribution Cliff's delta; Wilcoxon zsplit p; Holm over the 8 baselines).", sec))
    rows += sec

    # (b) full model (abl_full) vs 18 tuned baselines (Table S10)
    sec = []
    ex = existing_s10()
    for col, mname in METRICS:
        ref = load_metric(ABL / "abl_full" / "losocv_abl_full.csv", col)
        for stem, name in ALL18_ORDER:
            r = compare(ref, load_metric(TUNED / f"losocv_{stem}.csv", col), args.n_boot, args.seed)
            r.update(section="b", metric=col, metric_name=mname, A="abl_full", B=name)
            if ex is not None and (stem, col) in ex.index:
                e = ex.loc[(stem, col)]
                r.update(existing_cliffs=float(e["cliffs_delta"]), existing_p=float(e["wilcoxon_p"]),
                         existing_p_holm=float(e["p_holm"]))
            sec.append(r)
    sections.append(("(b) Full model (revision re-run abl_full, the headline run) vs each of the 18 tuned baselines -- Table S10 (S10 rows first, then the EEG encoders in Table 9 order and BrainGCN)",
                     "Existing values from results/statistics/tableS10_full_vs_baselines.csv "
                     "(between-distribution Cliff's delta; Wilcoxon zsplit p; Holm over all 18 baselines within each metric).", sec))
    rows += sec

    # (c) full model vs pathway variants (Table S14 layout), both references, plus ET-LSTM and the direct pair
    ex14 = existing_s14()
    et_lstm = load_fold_probs(TUNED / "fold_probs" / "probs_et_lstm.csv")
    refs = [("abl_full", ABL / "abl_full" / "losocv_abl_full.csv", "revision re-run abl_full (Table 8 reference)"),
            ("published full", PUBLISHED, "published run (Table S14 reference)")]
    for ref_key, ref_csv, ref_desc in refs:
        sec = []
        for col, mname in METRICS:
            ref = load_metric(ref_csv, col)
            for vkey, vname, vcsv in VARIANTS:
                if not vcsv.exists():
                    continue
                r = compare(ref, load_metric(vcsv, col), args.n_boot, args.seed)
                r.update(section=f"c_{ref_key.split()[0]}", metric=col, metric_name=mname, A=ref_key, B=vname)
                if ref_key == "published full" and ex14 is not None and (col, "full", vkey) in ex14.index:
                    e = ex14.loc[(col, "full", vkey)]
                    r.update(existing_cliffs=float(e["cliffs_delta"]), existing_p=float(e["wilcoxon_p"]),
                             existing_p_holm=float(e["p_holm"]))
                sec.append(r)
            r = compare(ref, et_lstm[col], args.n_boot, args.seed)
            r.update(section=f"c_{ref_key.split()[0]}", metric=col, metric_name=mname, A=ref_key,
                     B="ET-LSTM tuned (argmax operating point)")
            if ref_key == "published full" and ex14 is not None and (col, "full", "et_only(ET-LSTM)") in ex14.index:
                e = ex14.loc[(col, "full", "et_only(ET-LSTM)")]
                r.update(existing_cliffs=float(e["cliffs_delta"]), existing_p=float(e["wilcoxon_p"]),
                         existing_p_holm=float(e["p_holm"]))
            sec.append(r)
        sections.append((f"(c) Full model = {ref_desc} vs pathway variants -- Table S14 layout",
                         "Existing values (published reference only) from results/statistics/cross_modal_contribution.csv, "
                         "whose Cliff's delta is the PAIRED delta = P(sup) here; its Wilcoxon p discards zeros; Holm over the 8 pairs "
                         "per metric of that script; its bootstrap CI used seed 42, this one seed 0.", sec))
        rows += sec

    sec = []
    for col, mname in METRICS:
        r = compare(load_metric(ABL / "abl_eeg_only_mmd" / "losocv_abl_eeg_only_mmd.csv", col),
                    load_metric(eeg_only, col), args.n_boot, args.seed)
        r.update(section="c_direct", metric=col, metric_name=mname, A="abl_eeg_only_mmd", B="abl_eeg_only")
        sec.append(r)
    sections.append(("(c, direct pair) abl_eeg_only_mmd vs abl_eeg_only", "No existing table value.", sec))
    rows += sec

    # ---- write ----
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "paired_effect_sizes.csv", index=False)

    lines = ["# Paired effect sizes (fold-matched LOSOCV, saved per-fold CSVs)", "",
             "d = A - B per test subject. r_rb = (W+ - W-)/(W+ + W-) from the Wilcoxon signed-rank statistic with zero "
             "differences dropped. P(sup) = P(d>0) - P(d<0) over all pairs incl. ties = the 'Cliff's delta of the paired "
             "differences' of Table S14. p (zeros discarded) = two-sided Wilcoxon signed-rank, scipy zero_method='wilcox'; "
             "p (zsplit) = the convention of verify_table7_eeg_significance.py / table_s10_full_vs_baselines.py. "
             "'Cliff's delta (between)' = delta between the two per-fold distributions over all n x n pairs, the definition of "
             "Tables 9 / S6 / S10. 95% CI = subject-level bootstrap of the mean difference "
             f"({args.n_boot} resamples, seed {args.seed}). Balanced accuracy and MCC are at the uncalibrated operating point "
             "stored in each CSV (threshold transferred from the validation subject; ET-LSTM at argmax 0.5, as in S14).", ""]
    flags = []
    for title, note, sec in sections:
        lines += [f"## {title}", "", note, "", HEAD, SEP]
        for r in sec:
            lines.append(row_md(r))
            ec = r.get("existing_cliffs")
            if ec is not None and not np.isnan(ec):
                mine = r["paired_superiority"] if r["section"].startswith("c_") else r["cliffs_between"]
                if abs(round(mine, 2) - round(ec, 2)) > 0.011:
                    flags.append(f"{r['metric_name']}: {r['A']} vs {r['B']} -- recomputed {mine:+.3f} vs on file {ec:+.3f}")
        lines.append("")
    lines += ["## Consistency with the values on file", ""]
    lines += ([f"- {f}" for f in flags] if flags else ["- Every recomputed Cliff's delta agrees with the value on file to 0.01."])
    md = "\n".join(lines) + "\n"
    (out_dir / "paired_effect_sizes.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out_dir / 'paired_effect_sizes.md'} and .csv")


if __name__ == "__main__":
    main()
