"""
================================================================
ROI-grid spatial-resolution sensitivity — Reviewer 2, comment 9
================================================================
Three different spatial partitions of the stimulus appear in the study:

  (i)   Fig. 1 draws the 24 product boxes of the brochure page (6 x 4).
        This is the stimulus layout only; it is not a model input.
  (ii)  The model's ROI saliency vector r (input to ROI attention and to
        ROI graph modulation; Section 2.5.9) is a gaze-occupancy histogram
        over a GRID_ROWS x GRID_COLS grid (5 x 2 = 10 cells by default).
  (iii) The engagement label (engagement_labeling.py) uses grid-free /
        fixed-resolution gaze statistics: an 8 x 8 histogram for gaze
        entropy, a 4 x 4 grid on the gaze bounding box for the revisit
        count, and the central 60 % x 60 % region for ROI density.

This script quantifies the sensitivity to each partition.

(A) ROI saliency vs. grid (model input, data side; seconds)
    For grids from 2x1 to 10x8: mean max-cell share, normalised entropy,
    and the Spearman correlation between the per-epoch saliency
    concentration under grid G and under the default 5x2 grid.

(B) Label stability vs. label-feature resolution (data side; seconds)
    Re-runs the ACTUAL labelling pipeline with the entropy histogram
    (4..16 bins), the revisit grid (2..8) and the central-region fraction
    (0.4..0.8) varied one at a time; reports Cohen's kappa and the
    fraction of flipped labels against the published labels.

(C) Model side (training; GPU server)
    Retrain the full model with r rebuilt on grid G:
      NEUMA_GRID_COLS=6 NEUMA_GRID_ROWS=4 python main.py --label grid_6x4 --fold-parallel
    (N_ROIS follows cols*rows automatically.) Re-run this script afterwards;
    it summarises src/model/output/metrics/grid_<c>x<r>/losocv_grid_<c>x<r>.csv
    with a paired Wilcoxon test against the 5x2 run.

Outputs
-------
  results/sensitivity/roi_grid_saliency.csv
  results/sensitivity/roi_grid_label_stability.csv
  results/sensitivity/roi_grid_model_side.csv       (when runs exist)
  results/sensitivity/roi_grid_sensitivity.md
  results/sensitivity/fig_roi_grid_sensitivity.pdf/.png
================================================================
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import cohen_kappa_score

os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "src" / "model"
SEG = ROOT / "src" / "data_pipeline" / "04_segmentation"
for p in (str(MODEL), str(MODEL.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config.settings import GRID_COLS, GRID_ROWS, IMAGE_H, IMAGE_W, SUBJECT_IDS   # noqa: E402
from data.dataset import NeumaGraphDataset                                       # noqa: E402

OUT = ROOT / "results" / "sensitivity"
GRIDS = [(2, 1), (3, 2), (5, 2), (4, 3), (6, 4), (8, 6), (10, 8)]   # (cols, rows)


def _load_labeling_module():
    spec = importlib.util.spec_from_file_location("engagement_labeling", SEG / "engagement_labeling.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clean_et(et):
    et = np.asarray(et)
    if et.dtype.kind not in "fiu":
        et = np.array([[float(v) if v is not None else np.nan for v in row] for row in et], dtype=np.float32)
    return np.nan_to_num(et.astype(np.float32), nan=0.0)


def roi_vector(et, cols, rows):
    gx, gy = et[:, 0], et[:, 1]
    valid = np.isfinite(gx) & np.isfinite(gy) & (gx > 0) & (gy > 0)
    gx, gy = gx[valid], gy[valid]
    if len(gx) == 0:
        return np.zeros(cols * rows, np.float32)
    if gx.max() > 2.0: gx = np.clip(gx / IMAGE_W, 0, 1)
    if gy.max() > 2.0: gy = np.clip(gy / IMAGE_H, 0, 1)
    ci = np.clip((gx * cols).astype(int), 0, cols - 1)
    ri = np.clip((gy * rows).astype(int), 0, rows - 1)
    counts = np.bincount(ri * cols + ci, minlength=cols * rows).astype(np.float32)
    return counts / (counts.sum() + 1e-10)


def norm_entropy(p):
    p = p[p > 0]
    return 0.0 if len(p) <= 1 else float(-(p * np.log(p)).sum() / np.log(len(p)))


def saliency_sweep(ds, grids):
    et_list = [_clean_et(e) for e in ds.raw_et]
    ref = np.stack([roi_vector(e, GRID_COLS, GRID_ROWS) for e in et_list])
    ref_conc = ref.max(1); ref_ent = np.array([norm_entropy(r) for r in ref])
    rows = []
    for cols, rws in grids:
        R = np.stack([roi_vector(e, cols, rws) for e in et_list])
        conc = R.max(1); ent = np.array([norm_entropy(r) for r in R])
        rows.append(dict(grid=f"{cols}x{rws}", n_cells=cols * rws, n_epochs=len(R),
                         max_cell_share=float(conc.mean()), norm_entropy=float(ent.mean()),
                         empty_cells_frac=float((R == 0).mean()),
                         rho_concentration_vs_ref=float(spearmanr(conc, ref_conc).correlation),
                         rho_entropy_vs_ref=float(spearmanr(ent, ref_ent).correlation)))
    return pd.DataFrame(rows)


def label_stability(EL):
    """Re-run the real labelling pipeline with its spatial resolutions varied."""
    subject_dirs = sorted(d for d in SEG.iterdir() if d.is_dir() and d.name.startswith("S")
                          and (d / "output" / "epochs" / "eeg_epochs.npy").exists())
    data = [EL.load_subject_data(d, EL.STIMULUS_PATTERN) for d in subject_dirs]
    data = [d for d in data if d is not None]
    epochs = [(sd["subject_id"], ep, row) for sd in data for ep, row in zip(sd["et_list"], sd["meta_df"].to_dict("records"))]

    def features(n_bins=8, grid_size=4, c_frac=0.60):
        recs = []
        for sid, ep, row in epochs:
            fix = float(row.get("mean_fix_dur", 0.0)) / EL.ET_SR
            dwell = float(row.get("total_gaze_sec", 0.0)) or EL.compute_dwell_time(ep)
            recs.append(dict(subject_id=sid, fixation_duration=fix, dwell_time=dwell,
                             roi_density=EL.compute_roi_density(ep, cx_frac=c_frac, cy_frac=c_frac),
                             revisit_count=EL.compute_revisit_count(ep, grid_size=grid_size),
                             gaze_entropy=EL.compute_gaze_entropy(ep, n_bins=n_bins)))
        df = pd.DataFrame(recs)
        s = EL.compute_engagement_scores(df)
        return df, s, (s > np.median(s)).astype(int)

    _, s_ref, y_ref = features()
    variants = ([("entropy_bins", b, dict(n_bins=b)) for b in (4, 6, 8, 12, 16)] +
                [("revisit_grid", g, dict(grid_size=g)) for g in (2, 3, 4, 6, 8)] +
                [("central_fraction", c, dict(c_frac=c)) for c in (0.4, 0.5, 0.6, 0.7, 0.8)])
    rows = []
    for name, val, kw in variants:
        _, s, y = features(**kw)
        rows.append(dict(parameter=name, value=val, n_epochs=len(y),
                         rho_score_vs_ref=float(spearmanr(s, s_ref).correlation),
                         kappa_vs_ref=float(cohen_kappa_score(y, y_ref)),
                         label_flip_frac=float((y != y_ref).mean()), frac_high=float(y.mean()),
                         is_default=(name, val) in (("entropy_bins", 8), ("revisit_grid", 4), ("central_fraction", 0.6))))
    return pd.DataFrame(rows), len(y_ref)


def model_side(grids, ref_csv: Path | None):
    rows = []
    ref = pd.read_csv(ref_csv).set_index("test_subject") if ref_csv and ref_csv.exists() else None
    for cols, rws in grids:
        tag = f"grid_{cols}x{rws}"
        csv = MODEL / "output" / "metrics" / tag / f"losocv_{tag}.csv"
        if not csv.exists():
            continue
        d = pd.read_csv(csv).set_index("test_subject")
        row = dict(grid=f"{cols}x{rws}", n_folds=len(d))
        for m in ["balanced_acc", "roc_auc", "mcc"]:
            row[f"{m}_mean"] = d[m].mean(); row[f"{m}_sd"] = d[m].std()
            if ref is not None:
                common = d.index.intersection(ref.index)
                delta = d.loc[common, m].values - ref.loc[common, m].values
                row[f"{m}_delta_vs_ref"] = delta.mean()
                try:
                    row[f"{m}_p_vs_ref"] = wilcoxon(delta).pvalue if np.any(delta != 0) else 1.0
                except ValueError:
                    row[f"{m}_p_vs_ref"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--ref-csv", default=str(ROOT / "results" / "losocv_metrics" /
                    "losocv_repro_focal_g3p0_effective_num_37.csv"))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--legacy-label-stability", action="store_true",
                    help="also run block (B) for the superseded gaze-only labeller (engagement_labeling.py). "
                         "The production label (engagement_phase3d.py) uses NO spatial grid, so block (B) is "
                         "not part of the reported analysis.")
    args = ap.parse_args()
    subjects = args.subjects.split(",") if args.subjects else SUBJECT_IDS
    ds = NeumaGraphDataset(subject_ids=subjects, precompute_graphs=False, augment=False)
    print(f"[grid] {len(ds)} epochs from {len(ds.unique_subjects)} subjects; model grid {GRID_COLS}x{GRID_ROWS}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    sal = saliency_sweep(ds, GRIDS); sal.to_csv(out / "roi_grid_saliency.csv", index=False)
    lab, n_lab = pd.DataFrame(), 0
    if args.legacy_label_stability:
        EL = _load_labeling_module()
        lab, n_lab = label_stability(EL); lab.to_csv(out / "roi_grid_label_stability.csv", index=False)
    ms = model_side(GRIDS, Path(args.ref_csv))
    if len(ms):
        ms.to_csv(out / "roi_grid_model_side.csv", index=False)

    lines = ["# ROI-grid spatial-resolution sensitivity", "",
             f"(A) ROI saliency vector (model input) rebuilt on alternative grids — {len(ds)} epochs, "
             f"{len(ds.unique_subjects)} subjects; reference = {GRID_COLS}x{GRID_ROWS}.", "",
             "| grid | cells | max-cell share | norm. entropy | empty cells | ρ(concentration) vs ref | ρ(entropy) vs ref |",
             "|---|---|---|---|---|---|---|"]
    for _, r in sal.iterrows():
        mark = " **(model)**" if r.grid == f"{GRID_COLS}x{GRID_ROWS}" else (" *(Fig. 1 layout)*" if r.grid == "6x4" else "")
        lines.append(f"| {r.grid}{mark} | {int(r.n_cells)} | {r.max_cell_share:.3f} | {r.norm_entropy:.3f} | "
                     f"{r.empty_cells_frac:.2f} | {r.rho_concentration_vs_ref:.3f} | {r.rho_entropy_vs_ref:.3f} |")
    lines += ["", "(B) The production label (engagement_phase3d.py: frontal band power + gaze statistics, "
              "global median) contains no spatial grid, so grid resolution cannot affect it."]
    if len(lab):
        lines += ["", f"(B-legacy) Stability of the superseded gaze-only labeller ({n_lab} stimulus epochs):", "",
                  "| parameter | value | ρ(score) vs published | κ(label) vs published | labels flipped | HIGH fraction |",
                  "|---|---|---|---|---|---|"]
    for _, r in lab.iterrows():
        mark = " **(published)**" if r.is_default else ""
        lines.append(f"| {r.parameter} | {r.value}{mark} | {r.rho_score_vs_ref:.3f} | {r.kappa_vs_ref:.3f} | "
                     f"{r.label_flip_frac*100:.1f}% | {r.frac_high:.3f} |")
    if len(ms):
        lines += ["", "(C) Full model retrained with r rebuilt on each grid (paired vs. the 5x2 run):", "",
                  "| grid | folds | BalAcc | ROC-AUC | MCC | Δ BalAcc (p) | Δ AUC (p) | Δ MCC (p) |", "|---|---|---|---|---|---|---|---|"]
        for _, r in ms.iterrows():
            def _d(m):
                return (f"{r[f'{m}_delta_vs_ref']:+.3f} ({r[f'{m}_p_vs_ref']:.3f})" if f"{m}_delta_vs_ref" in r else "–")
            lines.append(f"| {r.grid} | {int(r.n_folds)} | {r.balanced_acc_mean:.3f} ± {r.balanced_acc_sd:.3f} | "
                         f"{r.roc_auc_mean:.3f} ± {r.roc_auc_sd:.3f} | {r.mcc_mean:.3f} ± {r.mcc_sd:.3f} | "
                         f"{_d('balanced_acc')} | {_d('roc_auc')} | {_d('mcc')} |")
    else:
        lines += ["", "(C) Model side: no `grid_<c>x<r>` LOSOCV runs found yet. Launch on the GPU server:", "", "```",
                  *[f"NEUMA_GRID_COLS={c} NEUMA_GRID_ROWS={r} python ../../scripts/analysis/run_component_ablation.py "
                    f"--variant full --label grid_{c}x{r} --results-root output/metrics"
                    for c, r in GRIDS if (c, r) != (GRID_COLS, GRID_ROWS)], "```"]
    (out / "roi_grid_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n = 3 if len(ms) else 2
        fig, ax = plt.subplots(1, n, figsize=(5.2 * n, 3.8))
        x = sal.n_cells.values
        ax[0].plot(x, sal.rho_concentration_vs_ref, marker="o", label="ρ(max-cell share) vs 5×2")
        ax[0].plot(x, sal.rho_entropy_vs_ref, marker="s", label="ρ(saliency entropy) vs 5×2")
        ax[0].plot(x, sal.norm_entropy, marker="^", label="norm. saliency entropy")
        ax[0].set_xscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(sal.grid, rotation=45)
        ax[0].axvline(GRID_COLS * GRID_ROWS, ls="--", color="k", lw=1); ax[0].axvline(24, ls=":", color="gray", lw=1)
        ax[0].set_xlabel("ROI grid (cols×rows)"); ax[0].set_title("(A) ROI saliency vs. grid", fontweight="bold"); ax[0].legend(fontsize=7)
        if len(lab):
            for name, mk in [("entropy_bins", "o"), ("revisit_grid", "s"), ("central_fraction", "^")]:
                g = lab[lab.parameter == name]
                ax[1].plot(range(len(g)), g.kappa_vs_ref, marker=mk, label=f"{name}: {list(g.value)}")
            ax[1].set_ylim(0.5, 1.02); ax[1].set_xlabel("parameter setting (index)"); ax[1].set_ylabel("κ vs. published labels")
            ax[1].set_title("(B-legacy) Label stability, superseded labeller", fontweight="bold"); ax[1].legend(fontsize=6.5)
        else:
            ax[1].axis("off")
            ax[1].text(0.5, 0.5, "The production label (engagement_phase3d)\ncontains no spatial grid:\nno label-side sensitivity exists.",
                       ha="center", va="center", fontsize=9)
            ax[1].set_title("(B) Label side", fontweight="bold")
        if len(ms):
            xm = ms.grid.map(lambda g: int(g.split("x")[0]) * int(g.split("x")[1])).values
            for m, lab_ in [("balanced_acc", "balanced accuracy"), ("roc_auc", "ROC-AUC"), ("mcc", "MCC")]:
                ax[2].errorbar(xm, ms[f"{m}_mean"], yerr=ms[f"{m}_sd"] / np.sqrt(ms.n_folds), marker="o", capsize=3, label=lab_)
            ax[2].set_xscale("log"); ax[2].set_xticks(xm); ax[2].set_xticklabels(ms.grid, rotation=45)
            ax[2].set_title("(C) LOSOCV metrics vs. grid", fontweight="bold"); ax[2].legend(fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_roi_grid_sensitivity.{ext}", dpi=300, bbox_inches="tight")
    except Exception as e:  # pragma: no cover
        print(f"[grid] figure skipped: {e}")


if __name__ == "__main__":
    main()
