#!/usr/bin/env python3
"""
Regenerate fig_roi_timecourse from committed real data.
Source : results/case_study/roi_timecourse.json  (real ROI-saliency vectors for
         the HIGH exemplar S24 and the LOW exemplar S30)
Output : results/figures/fig_roi_timecourse.{png,pdf}
Run    : python figures/gen_fig_roi_timecourse.py   (repository root)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P8 = Path(__file__).resolve().parents[2]   # repository root
d = json.load(open(P8 / "results/case_study/roi_timecourse.json"))
H, L = d["HIGH"], d["LOW"]
roiH, roiL = np.asarray(H["roi_vector"]), np.asarray(L["roi_vector"])
n = len(roiH)
x = np.arange(n)

fig, ax = plt.subplots(figsize=(8.5, 4.2))
w = 0.4
ax.bar(x - w/2, roiH, w, label=f"HIGH — {H['subj']} (epoch {H['rep']})", color="#C44E52")
ax.bar(x + w/2, roiL, w, label=f"LOW — {L['subj']} (epoch {L['rep']})",  color="#4C72B0")
ax.set_xticks(x); ax.set_xticklabels([str(i + 1) for i in range(n)], rotation=0, fontsize=9)
ax.set_xlabel("Grid cell (1-5 upper row, 6-10 lower row)")
ax.set_ylabel("Gaze-sample share")
ax.set_title("ROI-saliency distribution for the case-study exemplars")
ax.legend(frameon=False)
fig.tight_layout()
for out in (P8 / "paper/figures", P8 / "results/figures"):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_roi_timecourse.{ext}", dpi=200, bbox_inches="tight")
print("saved fig_roi_timecourse.{png,pdf} to paper/figures and results/figures",
      "| HIGH sum=%.3f LOW sum=%.3f" % (roiH.sum(), roiL.sum()))
