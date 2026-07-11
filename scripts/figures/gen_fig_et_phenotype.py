#!/usr/bin/env python3
"""
Regenerate fig_et_phenotype from committed real data.
Sources: results/validation/et_phenotype.csv        (per-epoch gaze/pupil descriptors, 385 epochs)
         results/validation/et_phenotype_arrays.npz  (pooled gaze points + pupil traces)
Output : figures/regen/fig_et_phenotype.{png,pdf}
Run    : python figures/gen_fig_et_phenotype.py   (from NEUMA_PHASE8/)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P8 = Path(__file__).resolve().parents[1]
df = pd.read_csv(P8 / "results/validation/et_phenotype.csv")
arr = np.load(P8 / "results/validation/et_phenotype_arrays.npz", allow_pickle=True)
gH, gL, pH, pL = arr["gazeH"], arr["gazeL"], arr["pupH"], arr["pupL"]

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / sp if sp else 0.0

descr = [("gaze_disp","Gaze dispersion"),("path_len","Scanpath length"),
         ("gaze_speed","Gaze speed"),("pupil_mean","Pupil mean"),("pupil_range","Pupil range")]
hi, lo = df[df.label==1], df[df.label==0]

fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
# (A) pooled gaze density HIGH vs LOW
ax[0].scatter(gL[:,0], gL[:,1], s=2, alpha=0.15, color="#4C72B0", label="LOW")
ax[0].scatter(gH[:,0], gH[:,1], s=2, alpha=0.15, color="#C44E52", label="HIGH")
ax[0].set_title("A · Gaze spatial extent"); ax[0].set_xlabel("gaze x"); ax[0].set_ylabel("gaze y")
ax[0].legend(markerscale=4, frameon=False)
# (B) descriptor HIGH vs LOW means + Cohen's d
ds = [cohens_d(hi[k].values, lo[k].values) for k,_ in descr]
y = np.arange(len(descr))
ax[1].barh(y, ds, color=["#C44E52" if v>=0 else "#4C72B0" for v in ds])
ax[1].axvspan(-0.2, 0.2, color="0.85", zorder=0)
ax[1].set_yticks(y); ax[1].set_yticklabels([n for _,n in descr]); ax[1].invert_yaxis()
ax[1].set_xlabel("Cohen's d (HIGH − LOW)"); ax[1].set_title("B · Descriptor separation")
for yi,v in zip(y,ds): ax[1].text(v, yi, f" {v:+.2f}", va="center", fontsize=9)
# (C) mean pupil trace
t = np.arange(pH.shape[1])
ax[2].plot(t, pH.mean(0), color="#C44E52", label="HIGH")
ax[2].plot(t, pL.mean(0), color="#4C72B0", label="LOW")
ax[2].set_title("C · Mean pupil trace"); ax[2].set_xlabel("sample (@120 Hz)"); ax[2].set_ylabel("pupil")
ax[2].legend(frameon=False)
fig.suptitle("Eye-tracking behavioural phenotype of engagement (385 epochs, 42 subjects)", y=1.02)
fig.tight_layout()
out = P8 / "figures/regen"; out.mkdir(exist_ok=True)
for ext in ("png","pdf"): fig.savefig(out / f"fig_et_phenotype.{ext}", dpi=200, bbox_inches="tight")
print("saved", out / "fig_et_phenotype.png", "| Cohen d:", {n:round(v,2) for (_,n),v in zip(descr,ds)})
