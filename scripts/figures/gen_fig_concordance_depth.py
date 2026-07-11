#!/usr/bin/env python3
"""
Regenerate fig_concordance_depth from committed real data.
Sources: results/validation/eeg_concordance.json          (frontal-theta, posterior-alpha band power)
         results/validation/connectivity_concordance.json  (fronto-posterior PLV, theta / alpha)
Output : figures/regen/fig_concordance_depth.{png,pdf}
Point  : every single univariate marker has |Cohen's d| < 0.2 and permutation p >> 0.05
         -> engagement is NOT captured by any one marker (multivariate; see ROC-AUC 0.82/0.90).
Run    : python figures/gen_fig_concordance_depth.py   (from NEUMA_PHASE8/)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P8 = Path(__file__).resolve().parents[1]
eeg = json.load(open(P8 / "results/validation/eeg_concordance.json"))
con = json.load(open(P8 / "results/validation/connectivity_concordance.json"))

ft = eeg["within_subject_permutation"]["frontal_theta_HIGH_gt_LOW"]
pa = eeg["within_subject_permutation"]["posterior_alpha_HIGH_lt_LOW"]
pt = con["theta"]["within_subject_perm"]
pl = con["alpha"]["within_subject_perm"]
def pval(o): return o.get("p_value", o.get("p_value_two_sided"))

markers = [
    ("Frontal-theta band power",        ft["cohens_d"], pval(ft)),
    ("Posterior-alpha band power",      pa["cohens_d"], pval(pa)),
    ("Fronto-posterior PLV, theta",     pt["cohens_d"], pval(pt)),
    ("Fronto-posterior PLV, alpha",     pl["cohens_d"], pval(pl)),
]
names = [m[0] for m in markers]; ds = [m[1] for m in markers]; ps = [m[2] for m in markers]
y = np.arange(len(markers))

fig, ax = plt.subplots(figsize=(8.8, 4.2))
ax.axvspan(-0.2, 0.2, color="#e9e9e9", zorder=0, label="negligible |d|<0.2")
ax.barh(y, ds, color=["#C44E52" if v>=0 else "#4C72B0" for v in ds], zorder=3)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlabel("Cohen's d  (HIGH − LOW, within-subject)")
ax.set_xlim(-0.35, 0.35)
for yi, v, p in zip(y, ds, ps):
    ax.text(v + (0.01 if v>=0 else -0.01), yi, f"d={v:+.2f}, p={p:.2f} (ns)",
            va="center", ha="left" if v>=0 else "right", fontsize=8.5)
ax.set_title("Single-marker concordance — no univariate EEG marker separates engagement\n"
             "(20,000-perm within-subject tests, 347 epochs, 37 subjects)", fontsize=11)
ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
out = P8 / "figures/regen"; out.mkdir(exist_ok=True)
for ext in ("png","pdf"): fig.savefig(out / f"fig_concordance_depth.{ext}", dpi=200)
print("saved", out / "fig_concordance_depth.png",
      "| markers:", [(n, round(d,3), round(p,2)) for n,d,p in markers])
