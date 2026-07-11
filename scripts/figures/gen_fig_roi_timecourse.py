#!/usr/bin/env python3
"""
Regenerate fig_roi_timecourse from committed real data.
Source : results/case_study/roi_timecourse.json  (real ROI-saliency vectors for
         the HIGH exemplar S24 and the LOW exemplar S30)
Output : figures/regen/fig_roi_timecourse.{png,pdf}
Run    : python figures/gen_fig_roi_timecourse.py   (from NEUMA_PHASE8/)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P8 = Path(__file__).resolve().parents[1]
d = json.load(open(P8 / "results/case_study/roi_timecourse.json"))
H, L = d["HIGH"], d["LOW"]
roiH, roiL = np.asarray(H["roi_vector"]), np.asarray(L["roi_vector"])
n = len(roiH)
x = np.arange(n)

fig, ax = plt.subplots(figsize=(8.5, 4.2))
w = 0.4
ax.bar(x - w/2, roiH, w, label=f"HIGH — {H['subj']} (epoch {H['rep']})", color="#C44E52")
ax.bar(x + w/2, roiL, w, label=f"LOW — {L['subj']} (epoch {L['rep']})",  color="#4C72B0")
ax.set_xticks(x); ax.set_xticklabels([f"ROI {i+1}" for i in range(n)], rotation=0, fontsize=8)
ax.set_ylabel("ROI dwell / saliency")
ax.set_title("ROI-saliency distribution for the case-study exemplars")
ax.legend(frameon=False)
fig.tight_layout()
out = P8 / "figures/regen"; out.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"fig_roi_timecourse.{ext}", dpi=200)
print("saved", out / "fig_roi_timecourse.png",
      "| HIGH sum=%.3f LOW sum=%.3f" % (roiH.sum(), roiL.sum()))
