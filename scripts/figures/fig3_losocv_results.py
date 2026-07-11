"""
Figure 3 — LOSOCV results: production INS-HDGS-CMT vs baselines.
Panel A: pooled ROC (real, from per-fold y_true/y_prob).
Panel B: per-fold metric distribution (box + jitter).
Panel C: headline metrics vs literature + baseline slots (auto-fill from ablation CSVs).

Real data now: output/metrics/repro_focal_g3p0_effective_num_37/losocv_*.csv (the 0.79 model).
Baseline bars (eeg_only / et_only / baseline_linear) fill in automatically once
results/ablation/abl_*/losocv_*.csv exist; until then they render as 'pending'.
Renders figures/fig3_losocv_results.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, ast
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, matthews_corrcoef

PH8 = Path(__file__).resolve().parents[1]
FULL_CSV = PH8 / "results/losocv_metrics/losocv_repro_focal_g3p0_effective_num_37.csv"
ABL = PH8 / "results/ablation"
LIT_REF = 0.77   # engagement-decoding literature benchmark (replace w/ exact cite)

C_FULL = "#2EA37A"; C_BASE = "#5B8DEF"; C_LIT = "#B2592E"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

def pooled_yt_yp(csv):
    d = pd.read_csv(csv); yt, yp = [], []
    for _, r in d.iterrows():
        yt += ast.literal_eval(r["y_true"]); yp += ast.literal_eval(r["y_prob"])
    return d, np.array(yt), np.array(yp)

d, yt, yp = pooled_yt_yp(FULL_CSV)
auc = roc_auc_score(yt, yp); fpr, tpr, _ = roc_curve(yt, yp)

fig = plt.figure(figsize=(13.5, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.25], wspace=0.32)

# ── Panel A: pooled ROC ───────────────────────────────────────────────────────
axA = fig.add_subplot(gs[0])
axA.plot(fpr, tpr, color=C_FULL, lw=2.4, label="INS-HDGS-CMT")
axA.plot([0, 1], [0, 1], "--", color="#999", lw=1.2)
axA.set_xlabel("False positive rate"); axA.set_ylabel("True positive rate")
axA.set_title("(A) Pooled LOSOCV ROC", fontweight="bold", fontsize=10)
axA.legend(loc="lower right", fontsize=8.5, frameon=False); axA.set_aspect("equal")
axA.grid(alpha=.25)

# ── Panel B: per-fold distribution ────────────────────────────────────────────
axB = fig.add_subplot(gs[1])
mets = [("balanced_acc_cal", "Balanced\nacc"), ("roc_auc", "ROC-AUC"),
        ("mcc_cal", "MCC"), ("accuracy_cal", "Accuracy")]
data = [d[m].values for m, _ in mets]
bp = axB.boxplot(data, patch_artist=True, widths=.6, showmeans=True,
                 meanprops=dict(marker="D", mfc="white", mec="k", ms=5))
for patch in bp["boxes"]: patch.set(facecolor=C_FULL, alpha=.35)
for i, vals in enumerate(data):
    axB.scatter(np.random.normal(i+1, .05, len(vals)), vals, s=10, color=C_FULL, alpha=.6, zorder=3)
axB.set_xticks(range(1, len(mets)+1)); axB.set_xticklabels([l for _, l in mets])
axB.set_ylim(-.4, 1.05); axB.axhline(0, color="#bbb", lw=.8)
axB.set_title(f"(B) Per-fold distribution (n={len(d)} folds)", fontweight="bold", fontsize=10)
axB.grid(axis="y", alpha=.25)

# ── Panel C: model vs baselines (auto-fill) ───────────────────────────────────
axC = fig.add_subplot(gs[2])
def metrics_of(csv):
    if not Path(csv).exists(): return None
    dd, t, p = pooled_yt_yp(csv)
    thr = dd["opt_threshold"].values
    pred = np.concatenate([(np.array(ast.literal_eval(r["y_prob"])) >= r["opt_threshold"]).astype(int)
                           for _, r in dd.iterrows()])
    return dict(bal=balanced_accuracy_score(t, pred), auc=roc_auc_score(t, p), mcc=matthews_corrcoef(t, pred))

models = [("INS-HDGS-CMT\n(full)", FULL_CSV)]
for v in ["no_graph", "no_snn", "no_et", "no_fusion_transformer", "no_neuro_symbolic"]:
    models.append((v.replace("_", "\n"), ABL / f"abl_{v}" / f"losocv_abl_{v}.csv"))

labels, bal, auc_, mcc, avail = [], [], [], [], []
for name, csv in models:
    m = metrics_of(csv); labels.append(name)
    if m: bal.append(m["bal"]); auc_.append(m["auc"]); mcc.append(m["mcc"]); avail.append(True)
    else: bal.append(0); auc_.append(0); mcc.append(0); avail.append(False)

x = np.arange(len(labels)); w = .26
axC.bar(x-w, bal,  w, label="Balanced acc", color=C_FULL, alpha=.85)
axC.bar(x,   auc_, w, label="ROC-AUC",      color=C_BASE, alpha=.85)
axC.bar(x+w, mcc,  w, label="MCC",          color="#E8A33D", alpha=.9)
for i, ok in enumerate(avail):
    if not ok: axC.text(i, .03, "pending\n(ablation)", ha="center", va="bottom", fontsize=6.8,
                        color="#999", rotation=0)
axC.set_xticks(x); axC.set_xticklabels(labels, fontsize=7.3)
axC.set_ylim(0, 1.0); axC.set_ylabel("Score")
axC.set_title("(C) Full model vs ablation baselines", fontweight="bold", fontsize=10)
axC.legend(fontsize=7.6, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.0))
axC.grid(axis="y", alpha=.25)

fig.suptitle("LOSOCV performance — INS-HDGS-CMT (37 folds)", fontsize=13, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig3_losocv_results.{ext}", dpi=300, bbox_inches="tight")
print(f"wrote figures/fig3_losocv_results.* | full AUC={auc:.3f}, "
      f"baselines available: {sum(avail)-1}/{len(models)-1}")
