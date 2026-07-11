"""
Subject-wise LOSOCV performance bar chart (manuscript Fig., §3.5).
Each bar = one held-out subject's balanced accuracy; dashed line = cross-subject mean.
Reads the canonical per-fold CSV. Renders figures/fig_losocv.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path

PH8 = Path(__file__).resolve().parents[1]
df = pd.read_csv(PH8 / "results/losocv_metrics/losocv_repro_focal_g3p0_effective_num_37.csv")
df = df.sort_values("balanced_acc", ascending=False).reset_index(drop=True)
bal = df["balanced_acc"].values
subj = df["test_subject"].astype(str).values
mean_bal = float(np.mean(bal))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(12, 4.2))
colors = ["#2EA37A" if b >= 0.5 else "#d1495b" for b in bal]
ax.bar(range(len(bal)), bal, color=colors, edgecolor="white", linewidth=0.4)
ax.axhline(mean_bal, ls="--", color="#2F3645", lw=1.4)
ax.text(len(bal) - 0.5, mean_bal + 0.015, f"mean = {mean_bal:.3f}",
        ha="right", va="bottom", fontsize=9, color="#2F3645")
ax.axhline(0.5, ls=":", color="#999", lw=1.0)
ax.text(0.2, 0.505, "chance", fontsize=7.5, color="#999", va="bottom")
ax.set_xticks(range(len(bal)))
ax.set_xticklabels(subj, rotation=90, fontsize=6.5)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Balanced accuracy")
ax.set_xlabel("Held-out subject (sorted)")
ax.set_title(f"Subject-wise LOSOCV performance ({len(bal)} folds)",
             fontweight="bold", fontsize=12)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(PH8 / f"figures/fig_losocv.{ext}", dpi=300, bbox_inches="tight")
print(f"wrote figures/fig_losocv.* | mean bal-acc={mean_bal:.3f}, "
      f"range [{bal.min():.2f}, {bal.max():.2f}], n={len(bal)}")
