"""
Figure 4 — Component (leave-one-out) ablation: importance ranking.
Reads results/ablation/abl_<variant>/losocv_*.csv (produced by the ablation study)
and the full baseline CSV; plots Δ balanced-acc and Δ MCC vs full, ranked by
importance, with Wilcoxon significance stars.

No-ops with a message until the ablation study has produced variant CSVs.
Renders figures/fig4_ablation.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, ast, sys
from pathlib import Path
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

PH8 = Path(__file__).resolve().parents[1]
FULL_CSV = PH8 / "output/metrics/repro_focal_g3p0_effective_num_37/losocv_repro_focal_g3p0_effective_num_37.csv"
ABL = PH8 / "results/ablation"
VARIANTS = ["no_snn", "no_graph", "no_neuro_symbolic", "no_et", "no_roi",
            "no_fusion_transformer", "no_contrastive", "no_mmd"]
C_HURT = "#C0504D"; C_HELP = "#2EA37A"

def per_fold(csv):
    """Return dict test_subject -> (bal_acc, mcc) using each fold's opt_threshold."""
    d = pd.read_csv(csv); out = {}
    for _, r in d.iterrows():
        yt = np.array(ast.literal_eval(r["y_true"])); yp = np.array(ast.literal_eval(r["y_prob"]))
        pred = (yp >= r["opt_threshold"]).astype(int)
        out[r["test_subject"]] = (balanced_accuracy_score(yt, pred) if len(set(yt)) > 1 else np.nan,
                                  matthews_corrcoef(yt, pred) if len(set(yt)) > 1 else np.nan)
    return out

def main():
    if not FULL_CSV.exists():
        print(f"[fig4] baseline CSV missing: {FULL_CSV}"); return
    present = [v for v in VARIANTS if (ABL / f"abl_{v}" / f"losocv_abl_{v}.csv").exists()]
    if not present:
        print(f"[fig4] no ablation variant CSVs under {ABL} yet — "
              f"figure will render once the ablation study completes."); return

    full = per_fold(FULL_CSV)
    rows = []
    for v in present:
        var = per_fold(ABL / f"abl_{v}" / f"losocv_abl_{v}.csv")
        subj = [s for s in full if s in var and not np.isnan(full[s][0]) and not np.isnan(var[s][0])]
        dbal = np.array([var[s][0] - full[s][0] for s in subj])
        dmcc = np.array([var[s][1] - full[s][1] for s in subj])
        try: _, p = wilcoxon(dbal)
        except Exception: p = np.nan
        rows.append(dict(variant=v.replace("no_", "−"), dbal=dbal.mean(), dmcc=dmcc.mean(), p=p, n=len(subj)))
    df = pd.DataFrame(rows).sort_values("dbal")   # most negative (most important) at top

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    y = np.arange(len(df))
    def stars(p): return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
    for ax, col, title in [(a1, "dbal", "Δ balanced accuracy"), (a2, "dmcc", "Δ MCC")]:
        vals = df[col].values
        ax.barh(y, vals, color=[C_HURT if v < 0 else C_HELP for v in vals], alpha=.85)
        ax.axvline(0, color="#333", lw=1)
        for i, v in enumerate(vals):
            ax.text(v + (0.005 if v >= 0 else -0.005), i, f"{v:+.3f} {stars(df['p'].values[i])}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel(f"{title}  (variant − full)")
        ax.grid(axis="x", alpha=.25)
    a1.set_yticks(y); a1.set_yticklabels(df["variant"], fontsize=10)
    a1.invert_yaxis()
    fig.suptitle("Component leave-one-out ablation — negative Δ ⇒ component HELPS (removing it hurts)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(.5, -.02, f"Paired over shared test subjects · Wilcoxon: *** p<.001, ** p<.01, * p<.05 · "
             f"baseline = production focal model (n folds = {df['n'].max()})", ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig4_ablation.{ext}", dpi=300, bbox_inches="tight")
    print(f"[fig4] wrote figures/fig4_ablation.* with {len(present)}/{len(VARIANTS)} variants")

if __name__ == "__main__":
    main()
