#!/usr/bin/env python3
"""Control ladder: paired change against the full model for every control (Section 3.4 of the
manuscript). One row per control, three panels (ROC-AUC, balanced accuracy, MCC at the
uncalibrated operating point = threshold transferred from the validation subject), mean paired
difference with a bootstrap 95 % CI and the paired Wilcoxon p (zero differences discarded, the
scipy default used by compare_component_ablation.py, so the raw p equals Table 7), over the held-out
subjects the two runs share.

Reference : results/ablation/abl_full/losocv_abl_full.csv, the revision re-run of the production
            configuration from the same code path as every ablation (the reference of Table 7 /
            compare_component_ablation.py --full-csv ... --raw).  The published production CSV
            differs from it by run-to-run variation (0.008 balanced accuracy raw, 0.034 calibrated);
            the calibrated operating point (0.5 cut-off after temperature scaling) is therefore
            NOT used for paired deltas.  Override with NEUMA_LADDER_FULL=<csv>.
Variants  : results/ablation/abl_<v>/losocv_abl_<v>.csv ; ET-LSTM from results/baselines/dl_tuned/
Output    : paper/figures/fig_control_ladder.{pdf,png}, results/figures/control_ladder.csv
Run       : python scripts/figures/fig_control_ladder.py     (repository root; CPU; needs the
            revision CSVs, i.e. run on the server or after they are pulled)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
import os
FULL = Path(os.environ.get("NEUMA_LADDER_FULL", ROOT / "results/ablation/abl_full/losocv_abl_full.csv"))
ABL = ROOT / "results/ablation"
BASE = ROOT / "results/baselines/dl_tuned"

# (label, csv, group).  Order = display order (top to bottom).
ROWS = [
    ("$-$ Spiking encoder",                       ABL / "abl_no_snn/losocv_abl_no_snn.csv",                         "Architecture ablations"),
    ("$-$ ROI guidance",                          ABL / "abl_no_roi/losocv_abl_no_roi.csv",                         "Architecture ablations"),
    ("$-$ Fusion transformer",                    ABL / "abl_no_fusion_transformer/losocv_abl_no_fusion_transformer.csv", "Architecture ablations"),
    ("$-$ Neuro-symbolic module",                 ABL / "abl_no_neuro_symbolic/losocv_abl_no_neuro_symbolic.csv",   "Architecture ablations"),
    ("$-$ Contrastive objective",                 ABL / "abl_no_contrastive/losocv_abl_no_contrastive.csv",         "Architecture ablations"),
    ("$-$ MMD alignment",                         ABL / "abl_no_mmd/losocv_abl_no_mmd.csv",                         "Architecture ablations"),
    ("Rule gate closed ($\\alpha\\equiv0$)",      ABL / "abl_ns_rule_only/losocv_abl_ns_rule_only.csv",             "Architecture ablations"),
    ("$-$ Dynamic graph pathway",                 ABL / "abl_no_graph/losocv_abl_no_graph.csv",                     "Graph topology"),
    ("Measured $\\to$ static graph (null)",       ABL / "abl_full_graph_static/losocv_abl_full_graph_static.csv",   "Graph topology"),
    ("Measured $\\to$ random graph (null)",       ABL / "abl_full_graph_random/losocv_abl_full_graph_random.csv",   "Graph topology"),
    ("$-$ Gaze sequence (ROI retained)",          ABL / "abl_no_et/losocv_abl_no_et.csv",                           "Gaze input"),
    ("$-$ All gaze input (gaze-free variant)",     ABL / "abl_eeg_only/losocv_abl_eeg_only.csv",                     "Gaze input"),
    ("ET-LSTM, gaze only (tuned baseline)",       BASE / "losocv_et_lstm.csv",                                      "External reference"),
]
METRICS = [("roc_auc", "$\\Delta$ ROC-AUC"), ("balanced_acc", "$\\Delta$ balanced accuracy"),
           ("mcc", "$\\Delta$ MCC")]
GROUP_COLOR = {"Architecture ablations": "#4c72b0", "Graph topology": "#8172b2",
               "Gaze input": "#c44e52", "External reference": "#55a868"}
RNG = np.random.default_rng(0)


def load(csv):
    d = pd.read_csv(csv)
    d = d.drop_duplicates("test_subject", keep="last").set_index("test_subject")
    return d


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    m = RNG.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    full = load(FULL)
    recs = []
    per_subject = {}                                      # (control, metric) -> paired differences
    for label, csv, group in ROWS:
        if not csv.exists():
            print(f"[skip] {label}: {csv.relative_to(ROOT)} missing"); continue
        v = load(csv)
        common = sorted(set(full.index) & set(v.index))
        for m, _ in METRICS:
            if m not in v.columns:
                print(f"[skip] {label}: no column {m}"); continue
            a = full.loc[common, m].astype(float).to_numpy()
            b = v.loc[common, m].astype(float).to_numpy()
            d = b - a                                     # variant minus full (negative = variant worse)
            lo, hi = boot_ci(d)
            try:
                p = wilcoxon(b, a, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0   # = Table 7 convention
            except ValueError:
                p = float("nan")
            recs.append(dict(control=label, group=group, metric=m, n=len(common), mean_full=a.mean(),
                             mean_variant=b.mean(), delta=d.mean(), ci_lo=lo, ci_hi=hi, wilcoxon_p=p,
                             wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum())))
            per_subject[(label, m)] = d
    df = pd.DataFrame(recs)
    (ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results/figures/control_ladder.csv", index=False)

    labels = [r[0] for r in ROWS if r[0] in set(df.control)]
    y = np.arange(len(labels))[::-1]
    n_pairs = int(df.n.min())
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.46 * len(labels) + 1.8), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    jit = np.random.default_rng(1)
    for ax, (m, title) in zip(axes, METRICS):
        sub = df[df.metric == m].set_index("control")
        alld = np.concatenate([per_subject[(l, m)] for l in labels])
        lo = min(sub.ci_lo.min(), np.percentile(alld, 2.5))
        hi = max(sub.ci_hi.max(), np.percentile(alld, 97.5))
        span = hi - lo
        xlo, xhi = lo - 0.04 * span, hi + 0.42 * span     # room on the right for the p value
        n_clip = 0
        for yi, lab in zip(y, labels):
            if lab not in sub.index:
                continue
            r = sub.loc[lab]
            c = GROUP_COLOR[r.group]
            d = per_subject[(lab, m)]                      # the individual paired differences
            yj = yi + jit.uniform(-0.22, 0.22, len(d))
            inside = (d >= xlo) & (d <= hi + 0.02 * span)
            ax.scatter(d[inside], yj[inside], s=9, color=c, alpha=0.28, lw=0, zorder=1)
            # subjects beyond the plotted range are drawn as triangles at the edge
            for side, mk, xe in ((d < xlo, "<", xlo + 0.01 * span), (d > hi + 0.02 * span, ">", hi + 0.03 * span)):
                if side.any():
                    ax.scatter(np.full(side.sum(), xe), yj[side], s=14, marker=mk, color=c, alpha=0.5, lw=0, zorder=1)
                    n_clip += int(side.sum())
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=c, lw=2.4, solid_capstyle="round", zorder=2)
            sig = r.wilcoxon_p < 0.05
            ax.plot(r.delta, yi, "o", ms=7.5, mfc=c if sig else "white", mec=c, mew=1.6, zorder=3)
            ax.text(0.985, yi, f"p={r.wilcoxon_p:.3f}" if r.wilcoxon_p >= 0.001 else "p<0.001",
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.2,
                    color="0.2" if sig else "0.45", fontweight="bold" if sig else "normal", zorder=4)
        ax.axvline(0, color="k", lw=0.9, zorder=2)
        ax.set_xlim(xlo, xhi)
        print(f"{m}: axis [{lo:+.3f}, {hi:+.3f}], {n_clip} subject points drawn at the edge")
        ax.set_ylim(-0.7, len(labels) - 0.3)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel(f"variant $-$ decoder over {n_pairs} paired subjects", fontsize=9)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=10)
    # group separators
    groups = [next(g for l, _, g in ROWS if l == lab) for lab in labels]
    for i in range(1, len(labels)):
        if groups[i] != groups[i - 1]:
            for ax in axes:
                ax.axhline(y[i] + 0.5, color="0.8", lw=0.8, ls="--")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2.4, label=g) for g, c in GROUP_COLOR.items()]
    handles += [plt.Line2D([], [], color="0.3", marker="o", mfc="0.3", lw=0, label="raw Wilcoxon $p<0.05$"),
                plt.Line2D([], [], color="0.3", marker="o", mfc="white", mew=1.4, lw=0, label="$p \\geq 0.05$"),
                plt.Line2D([], [], color="0.3", marker="o", ms=3, alpha=0.4, lw=0, label="individual subjects"),
                plt.Line2D([], [], color="0.3", marker="<", ms=4, alpha=0.6, lw=0, label="beyond axis range")]
    fig.legend(handles=handles, loc="lower center", fontsize=9, frameon=False, ncol=8, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    for out in (ROOT / "paper/figures", ROOT / "results/figures"):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "fig_control_ladder.pdf", bbox_inches="tight")
        fig.savefig(out / "fig_control_ladder.png", dpi=250, bbox_inches="tight")
    pd.set_option("display.width", 200)
    print(df[["control", "metric", "n", "mean_full", "mean_variant", "delta", "ci_lo", "ci_hi", "wilcoxon_p", "wins", "ties", "losses"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("wrote paper/figures/fig_control_ladder.{pdf,png}, results/figures/control_ladder.csv")


if __name__ == "__main__":
    main()
