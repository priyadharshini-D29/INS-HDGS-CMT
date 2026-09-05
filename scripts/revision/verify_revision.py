"""
Re-verification of the reviewer-revision outputs.

Checks, for every reviewer point, that the expected result files exist, that the
LOSOCV CSVs have the right number of folds with both classes and sane metric
ranges, and prints the numbers that replace the red \\todoval{} markers in
paper/INS_HDGS_CMT_manuscript.tex and paper/INS_HDGS_CMT_supplementary.tex.

    python scripts/revision/verify_revision.py            # report
    python scripts/revision/verify_revision.py --strict   # exit 1 if anything is missing

No training is performed here.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
M = ROOT / "src" / "model" / "output" / "metrics"
PROD = R / "losocv_metrics" / "losocv_repro_focal_g3p0_effective_num_37.csv"

OK, MISSING, BAD = "OK", "MISSING", "CHECK"
rows = []


def add(point, item, status, detail=""):
    rows.append((point, item, status, detail))


def fold_csv(path: Path, expect_n=37):
    """Sanity-check a per-fold LOSOCV CSV; return (status, detail, df)."""
    if not path.exists():
        return MISSING, str(path.relative_to(ROOT)), None
    df = pd.read_csv(path)
    d = []
    if len(df) != expect_n:
        d.append(f"{len(df)} folds (expected {expect_n})")
    if "y_true" in df:
        one_class = [r.test_subject for r in df.itertuples() if len(set(ast.literal_eval(r.y_true))) < 2]
        if one_class:
            d.append(f"single-class test folds: {one_class}")
    for k in ("balanced_acc", "roc_auc", "mcc"):
        if k in df and not (0 <= df[k].min() and df[k].max() <= 1 or k == "mcc" and -1 <= df[k].min()):
            d.append(f"{k} out of range")
    summ = " | ".join(f"{k} {df[k].mean():.3f}±{df[k].std():.3f}" for k in ("balanced_acc", "roc_auc", "mcc") if k in df)
    return (BAD if d else OK), (("; ".join(d) + " | ") if d else "") + summ, df


def paired(df_a, df_b, metric="roc_auc"):
    from scipy.stats import wilcoxon
    a = df_a.set_index("test_subject")[metric]; b = df_b.set_index("test_subject")[metric]
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).to_numpy(float)
    p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff, zero_method="zsplit").pvalue)
    return len(common), float(diff.mean()), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    st, det, prod = fold_csv(PROD)
    add("reference", "production run (Table 1, Sec. 3.1)", st, det)

    # ---- label (Sec. 2.4, Table S5, S15) ----
    n_lab = len(list((ROOT / "src/data_pipeline/04_segmentation").glob("S*/output/engagement_phase3d/engagement_labels.npy")))
    add("label", "production labels (engagement_phase3d) present", OK if n_lab >= 42 else MISSING, f"{n_lab}/42 subjects")
    if prod is not None and n_lab:
        agree = tot = 0
        for r in prod.itertuples():
            f = ROOT / "src/data_pipeline/04_segmentation" / r.test_subject / "output/engagement_phase3d/engagement_labels.npy"
            if f.exists():
                y = np.array(ast.literal_eval(r.y_true)); l = np.load(f)
                if len(l) == len(y):
                    agree += int((l == y).sum()); tot += len(y)
        add("label", "phase-3D labels reproduce the run's y_true", OK if tot and agree == tot else BAD, f"{agree}/{tot}")
    aud = R / "statistics" / "label_leakage_audit.csv"
    if aud.exists():
        a = pd.read_csv(aud)
        add("label", "Table S15 / Sec. 3.2 recoverability", OK,
            "; ".join(f"{r.feature_set}: pooled AUC {r.pooled_auc:.3f}, fold AUC {r.fold_auc_mean:.3f}±{r.fold_auc_sd:.3f}, BalAcc {r.fold_balacc_mean:.3f}" for r in a.itertuples()))
    else:
        add("label", "Table S15 / Sec. 3.2 recoverability", MISSING, "stage audit")

    # ---- R1-1/3: EEG-only ----
    st, det, eeg = fold_csv(R / "ablation/abl_eeg_only/losocv_abl_eeg_only.csv")
    add("R1-1/3", "gaze-free EEG-only branch (Abstract, Table 3 row 1, Table 7, Sec. 3.4b)", st, det)
    if eeg is not None and prod is not None:
        n, d, p = paired(prod, eeg)
        add("R1-1/3", "full − EEG-only ΔAUC (Sec. 3.4b)", OK, f"n={n} Δ={d:+.3f} p={p:.4f} (Holm in cross_modal_contribution.md)")
    st, det, eegm = fold_csv(R / "ablation/abl_eeg_only_mmd/losocv_abl_eeg_only_mmd.csv")
    add("R1-1/3", "gaze-free EEG branch with MMD kept (fair full-minus-gaze)", st, det)
    if eegm is not None and prod is not None:
        n, d, p = paired(prod, eegm)
        add("R1-1/3", "full - EEG-only(MMD kept) dAUC", OK, f"n={n} d={d:+.3f} p={p:.4f}")
    st, det, full = fold_csv(R / "ablation/abl_full/losocv_abl_full.csv")
    add("reference", "fresh full re-run (checkpoints for stage ckpt)", st, det)
    if full is not None and prod is not None:
        n, d, p = paired(prod, full)
        add("reference", "published vs fresh full run (reproducibility)", OK if abs(d) < 0.03 else BAD, f"ΔAUC={d:+.3f} p={p:.3f} (expect |Δ|<0.03)")
    cm = R / "statistics" / "cross_modal_contribution.md"
    add("R1-1/2", "cross-modal contribution table (Table S14)", OK if cm.exists() else MISSING,
        "contains EEG-only rows" if cm.exists() and "EEG-only" in cm.read_text(encoding="utf-8") else "re-run stats after eeg_only")

    # ---- R2-3 tuned baselines ----
    tuned = sorted((R / "baselines/dl_tuned").glob("losocv_*.csv"))
    add("R2-3", "tuned baselines (Tables 3-5, 8, S6, S10)", OK if len(tuned) >= 18 else (BAD if tuned else MISSING), f"{len(tuned)}/18 per-fold CSVs")
    for m in ("balanced_acc", "mcc", "roc_auc"):
        f = R / "statistics" / f"table7_eeg_significance_tuned_{m}.csv"
        if f.exists():
            t = pd.read_csv(f)
            add("R2-3", f"EEG-only vs tuned EEG baselines, {m}", OK, "; ".join(f"{r.baseline}: Δ{r.median_delta:+.3f} pH={r.p_holm:.3f}" for r in t.itertuples()))
        else:
            add("R2-3", f"EEG-only vs tuned EEG baselines, {m}", MISSING, "stage stats")

    # ---- R2-2 tau ----
    ts = M / "sensitivity_threshold_summary.csv"
    if ts.exists():
        t = pd.read_csv(ts); cols = [c for c in t.columns if c in ("value", "balanced_acc", "roc_auc", "mcc", "balanced_acc_mean", "roc_auc_mean")]
        add("R2-2", "tau downstream (Table S7 right, Sec. 3.9)", OK, t[cols].round(3).to_string(index=False).replace("\n", " || "))
    else:
        add("R2-2", "tau downstream (Table S7 right, Sec. 3.9)", MISSING, "stage tau")
    add("R2-2", "tau graph structure (Table S7 left, Fig. S4)", OK if (R / "sensitivity/tau_sensitivity.md").exists() else MISSING)

    # ---- R2-9 grid ----
    for g in ("2x1", "3x2", "6x4", "8x6"):
        st, det, _ = fold_csv(M / f"grid_{g}" / f"losocv_grid_{g}.csv")
        add("R2-9", f"grid {g} (Table S11 bottom)", st, det)
    add("R2-9", "ROI saliency vs grid (Table S11 top, Fig. S6)", OK if (R / "sensitivity/roi_grid_sensitivity.md").exists() else MISSING)

    # ---- R2-6 learning curve ----
    lc = R / "sensitivity" / "learning_curve.csv"
    if lc.exists():
        t = pd.read_csv(lc).groupby("n_subjects")[["roc_auc", "balanced_acc"]].mean().round(3)
        add("R2-6", "learning curve (Fig. S5, Sec. 3.9)", OK, t.to_string().replace("\n", " || "))
    else:
        add("R2-6", "learning curve (Fig. S5, Sec. 3.9)", MISSING, "stage lc")

    # ---- R2-5 rule gate ----
    rf = sorted((R / "statistics").glob("rule_fidelity*.md"))
    add("R2-5", "rule fidelity (Table S13, Sec. 3.10.3)", OK if rf else MISSING, ", ".join(p.name for p in rf) or "stage ckpt")
    st, det, _ = fold_csv(R / "ablation/abl_ns_rule_only/losocv_abl_ns_rule_only.csv")
    add("R2-5", "rule-only trained model (Table 7 row, Table S13 last row)", st, det)

    # ---- R2-7 grounding ----
    gr = sorted((R / "explainability").glob("rule_grounding_*.md")) if (R / "explainability").exists() else []
    add("R2-7", "grounded rules (Fig. 8D, Sec. 3.10.3)", OK if gr else MISSING, ", ".join(p.name for p in gr) or "stage ckpt")

    # ---- R2-4 energy ----
    en = R / "statistics" / "snn_energy_measured.json"
    if en.exists():
        import json
        j = json.loads(en.read_text(encoding="utf-8"))
        add("R2-4", "measured SNN cost (Table S9)", OK, f"device={j.get('device')} keys={list(j)[:8]}")
    else:
        add("R2-4", "measured SNN cost (Table S9)", MISSING, "stage ckpt")

    # ---- R2-8 thresholds ----
    add("R2-8", "label-free thresholds (Table S12, Sec. 3.7)", OK if (R / "threshold_analysis/DEPLOYMENT_THRESHOLD.md").exists() else MISSING)

    # ---- report ----
    w = max(len(r[1]) for r in rows)
    print(f"\n{'point':10s} {'item':{w}s} {'status':8s} detail")
    print("-" * (w + 40))
    for p, item, st, det in rows:
        print(f"{p:10s} {item:{w}s} {st:8s} {det}")
    n_missing = sum(r[2] == MISSING for r in rows); n_bad = sum(r[2] == BAD for r in rows)
    tex = (ROOT / "paper/INS_HDGS_CMT_manuscript.tex").read_text(encoding="utf-8").count("\\todoval{") + \
          (ROOT / "paper/INS_HDGS_CMT_supplementary.tex").read_text(encoding="utf-8").count("\\todoval{") - 2  # minus the macro definitions
    print(f"\nmissing: {n_missing}   needs attention: {n_bad}   red placeholders still in the LaTeX: {tex}")
    if args.strict and (n_missing or n_bad):
        sys.exit(1)


if __name__ == "__main__":
    main()
