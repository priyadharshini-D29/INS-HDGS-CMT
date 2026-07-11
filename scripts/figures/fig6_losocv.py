"""
Figure (LOSOCV schematic) — leave-one-subject-out cross-validation with a held-out
validation subject for leakage-free calibration/thresholding. Single-column.
Renders figures/fig6_losocv.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

C_TR = "#cdd8ec"; C_VAL = "#E8A33D"; C_TE = "#C0504D"; C_EDGE = "#5b6472"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

N = 37                      # evaluable subjects
shown = [(0, "Fold 1"), (1, "Fold 2"), (2, "Fold 3"), None, (N-1, f"Fold {N}")]

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.set_xlim(-0.5, N + 7); ax.set_ylim(-1.5, len(shown) + 1.2); ax.axis("off")
ax.set_title("Leave-one-subject-out cross-validation (LOSOCV)", fontsize=12, fontweight="bold")

cw = 0.92
for r, item in enumerate(shown):
    y = len(shown) - 1 - r
    if item is None:                                   # ellipsis row
        for d in range(3):
            ax.text(N/2, y + 0.45 - d*0.0, "$\\vdots$", ha="center", va="center", fontsize=12)
        ax.text(N/2, y + 0.45, "$\\vdots$", ha="center", va="center", fontsize=13)
        continue
    test_i, lab = item
    val_i = (test_i + 1) % N
    for s in range(N):
        col = C_TE if s == test_i else (C_VAL if s == val_i else C_TR)
        ax.add_patch(Rectangle((s, y), cw, 0.8, fc=col, ec="white", lw=0.6))
    ax.text(-0.7, y + 0.4, lab, ha="right", va="center", fontsize=9, fontweight="bold")
    ax.text(N + 0.4, y + 0.4, f"test = S{test_i+1:02d}", ha="left", va="center", fontsize=8, color=C_TE)

# subject axis label
ax.annotate("", xy=(N, -0.5), xytext=(0, -0.5),
            arrowprops=dict(arrowstyle="-", color="#999", lw=1))
ax.text(N/2, -1.0, f"{N} subjects (one held out as test each fold)", ha="center", fontsize=8.5, color="#555")

# legend
handles = [Rectangle((0,0),1,1,fc=C_TR,ec="white"), Rectangle((0,0),1,1,fc=C_VAL,ec="white"),
           Rectangle((0,0),1,1,fc=C_TE,ec="white")]
ax.legend(handles, ["Train pool", "Validation subject", "Test subject"],
          loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8.5)

# leakage-free note
ax.text(N/2, -1.45,
        "Temperature calibration and the decision threshold are fit on the validation "
        "subject only\n→ no information from the test subject leaks into the model.",
        ha="center", va="center", fontsize=8, style="italic", color="#2b6",
        bbox=dict(boxstyle="round,pad=0.4", fc="#eef7f1", ec="#2EA37A", lw=1))

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig6_losocv.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig6_losocv.pdf and .png")
