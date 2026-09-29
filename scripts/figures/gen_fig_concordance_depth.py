#!/usr/bin/env python3
"""
Regenerate fig_concordance_depth from committed real data.
Sources: results/validation/eeg_concordance.json          (frontal-theta, posterior-alpha band power)
         results/validation/connectivity_concordance.json  (fronto-posterior PLV, theta / alpha)
         results/ablation/abl_eeg_only/losocv_abl_eeg_only.csv   (gaze-free EEG branch, per fold)
         results/ablation/abl_full/losocv_abl_full.csv  (full model, per fold)
Output : paper/figures/fig_concordance_depth.{png,pdf}  (+ results/figures/)
Panel A: within-subject effect size and permutation p of each univariate marker
         (after the 2026-09-29 channel-map correction, frontal-midline theta shows a
         small HIGH>LOW effect; the other markers do not separate the classes).
Panel B: ROC-AUC of the best single marker (pooled and mean per subject), the gaze-free EEG
         branch and the full model, all on the same 37 evaluable subjects.
Run    : python scripts/figures/gen_fig_concordance_depth.py   (from the repository root)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import roc_auc_score
P8 = Path(__file__).resolve().parents[2]          # <repo>
eeg = json.load(open(P8 / "results/validation/eeg_concordance.json"))
con = json.load(open(P8 / "results/validation/connectivity_concordance.json"))

ft = eeg["within_subject_permutation"]["frontal_theta_HIGH_gt_LOW"]
pa = eeg["within_subject_permutation"]["posterior_alpha_HIGH_lt_LOW"]
pt = con["theta"]["within_subject_perm"]
pl = con["alpha"]["within_subject_perm"]
def pval(o): return o.get("p_value_two_sided", o.get("p_value"))   # two-sided throughout

markers = [
    ("Frontal-theta band power",        ft["cohens_d"], pval(ft)),
    ("Posterior-alpha band power",      pa["cohens_d"], pval(pa)),
    ("Fronto-posterior PLV, theta",     pt["cohens_d"], pval(pt)),
    ("Fronto-posterior PLV, alpha",     pl["cohens_d"], pval(pl)),
]
names = [m[0] for m in markers]; ds = [m[1] for m in markers]; ps = [m[2] for m in markers]
y = np.arange(len(markers))

fig, (ax, axb) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.55, 1]})
ax.axvspan(-0.2, 0.2, color="#e9e9e9", zorder=0, label="negligible |d|<0.2")
ax.barh(y, ds, color=["#C44E52" if v>=0 else "#4C72B0" for v in ds], zorder=3)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlabel("Cohen's d  (HIGH − LOW, within-subject)")
ax.set_xlim(-0.35, 0.35)
for yi, v, p in zip(y, ds, ps):
    tag = f"d={v:+.2f}, p={p:.3f}" + ("" if p < 0.05 else " (ns)")
    ax.text(v + (0.01 if v>=0 else -0.01), yi, tag,
            va="center", ha="left" if v>=0 else "right", fontsize=8.5)
ax.set_title("Single-marker concordance — frontal-midline theta shows a small HIGH>LOW effect;\n"
             "no other marker separates the classes (20,000-perm two-sided within-subject tests, 347 epochs, 37 subjects)",
             fontsize=10.5)
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("A", loc="left", fontweight="bold")

# ---- panel B: single markers vs the learned models, same epochs ----------------
def marker_auc(csv, col):
    d = pd.read_csv(csv)
    pooled = roc_auc_score(d["label"], d[col])
    per = [roc_auc_score(g["label"], g[col]) for _, g in d.groupby("subject") if g["label"].nunique() == 2]
    return pooled, float(np.mean(per))

cand = {}
for col in ("frontal_theta", "post_alpha"):
    cand[col] = marker_auc(P8 / "results/validation/eeg_concordance_per_epoch.csv", col)
for col in ("fp_plv_theta", "fp_plv_alpha"):
    cand[col] = marker_auc(P8 / "results/validation/connectivity_concordance_per_epoch.csv", col)
best = max(cand, key=lambda k: max(cand[k][0], 1 - cand[k][0]))
bp, bm = cand[best]
bp, bm = max(bp, 1 - bp), max(bm, 1 - bm)        # direction-free
def fold_auc(csv):
    d = pd.read_csv(csv); return float(d["roc_auc"].mean()), float(d["roc_auc"].std())
eeg_csv = P8 / "results/ablation/abl_eeg_only/losocv_abl_eeg_only.csv"
full_csv = P8 / "results/ablation/abl_full/losocv_abl_full.csv"
bars = [(f"best single marker\n({best.replace('_', ' ')}, per-subject mean)", bm, None),
        ("gaze-free EEG branch\n(mean per fold)", *fold_auc(eeg_csv)),
        ("full model, EEG + gaze\n(mean per fold)", *fold_auc(full_csv))]
xb = np.arange(len(bars))
axb.bar(xb, [b[1] for b in bars], yerr=[b[2] if b[2] else 0 for b in bars], capsize=4,
        color=["#999999", "#b00020", "#1f77b4"], zorder=3)
axb.axhline(0.5, color="k", ls="--", lw=0.8)
axb.set_xticks(xb); axb.set_xticklabels([b[0] for b in bars], fontsize=8.5)
axb.set_ylim(0.3, 1.0); axb.set_ylabel("ROC-AUC (37 evaluable subjects)")
for xi, b in zip(xb, bars):
    axb.text(xi, b[1] + 0.02, f"{b[1]:.2f}", ha="center", fontsize=9)
axb.set_title("B", loc="left", fontweight="bold")
axb.set_title("Decoding of the index: markers vs learned models", fontsize=10)
axb.grid(axis="y", alpha=0.3)
fig.tight_layout()
for out in (P8 / "paper/figures", P8 / "results/figures"):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_concordance_depth.{ext}", dpi=200)
print("saved fig_concordance_depth to paper/figures and results/figures",
      "| markers:", [(n, round(d, 3), round(p, 2)) for n, d, p in markers],
      f"| best marker {best}: pooled {bp:.3f}, per-subject {bm:.3f}",
      f"| eeg-only {fold_auc(eeg_csv)[0]:.3f} | full {fold_auc(full_csv)[0]:.3f}")
