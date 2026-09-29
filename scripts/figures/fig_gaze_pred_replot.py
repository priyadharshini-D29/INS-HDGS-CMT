"""
Re-draw Fig. 12 (fig_gaze_pred) from the saved case-study outputs, without the model.

case_study_high_low.py needs the fold checkpoints (server) to recompute the
predictions and the integrated-gradients attribution.  Its outputs are saved
under results/case_study/ (high_low_provenance.json: representative epoch,
p(HIGH), gaze entropy, per-input IG mass; roi_timecourse.json: the 5x2 ROI
saliency vector), and the gaze trajectory of the representative epoch is the
pipeline's own eye-tracking epoch (et_epochs_phase3d.npy, row `rep`, columns
x, y).  This script reproduces the figure from those files so the layout can be
changed without a server run.  The gaze entropy is recomputed from the epoch
and checked against the saved value, so a mismatch between the local epochs
and the ones the checkpoints saw is caught.

Run:  python scripts/figures/fig_gaze_pred_replot.py     (repository root)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "case_study"
SEG = ROOT / "src" / "data_pipeline" / "04_segmentation"
OUT = [ROOT / "paper" / "figures", ROOT / "results" / "figures" / "case_study"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.labelweight": "bold",
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10,
    "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.titlecolor": "black",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
FMT = mticker.FormatStrFormatter("%.3f")
# The connectivity input reaches the graph attention only as a binary mask, so
# integrated gradients assign it exactly zero; it is not drawn.
IGN = {"eeg_windows": "EEG", "et_seq": "Gaze", "roi_vector": "ROI"}


def _std_axis(a, xlabel=None, ylabel=None, fmt_x=False, fmt_y=True):
    if xlabel is not None:
        a.set_xlabel(xlabel)
    if ylabel is not None:
        a.set_ylabel(ylabel)
    if fmt_y:
        a.yaxis.set_major_formatter(FMT)
    if fmt_x:
        a.xaxis.set_major_formatter(FMT)


def gaze_entropy(xy: np.ndarray) -> float:
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=8)
    p = H / max(H.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def saved_probs() -> dict:
    """Per-subject held-out probabilities from the saved LOSOCV evaluation —
    the manuscript's probability definition (temperature-scaled ensemble mean,
    Section 2.7). The provenance JSON's p_high is the checkpoint re-inference
    WITHOUT the per-member temperature (the checkpoints do not store it), which
    differs on fold 28, the one fold where the temperatures were active."""
    import ast
    import pandas as pd
    d = pd.read_csv(ROOT / "results" / "ablation" / "abl_full" / "losocv_abl_full.csv")
    d = d.drop_duplicates("test_subject", keep="last").set_index("test_subject")
    return {s: [float(v) for v in ast.literal_eval(r["y_prob"])] for s, r in d.iterrows()}


def load_case(name: str, prov: dict, roi: dict, probs: dict) -> dict:
    c = prov[name]
    et = np.load(SEG / c["subj"] / "output" / "epochs" / "et_epochs_phase3d.npy", allow_pickle=True)
    xy = np.asarray(et[c["rep"]], np.float32)[:, :2]
    ent = gaze_entropy(xy)
    if abs(ent - c["gaze_entropy"]) > 1e-3:
        raise RuntimeError(f"{name}: local epoch {c['rep']} of {c['subj']} gives gaze entropy {ent:.4f}, "
                           f"saved run had {c['gaze_entropy']:.4f}; epochs differ from the analysed run")
    p_saved = probs[c["subj"]][c["rep"]]
    return dict(subj=c["subj"], xy=xy, entropy=c["gaze_entropy"], p_high=p_saved,
                p_high_untempered=c["p_high"], true=c["true"],
                ig=c["ig"], roi=np.asarray(roi[name]["roi_vector"], np.float32))


def main():
    prov = json.loads((RES / "high_low_provenance.json").read_text())
    roi = json.loads((RES / "roi_timecourse.json").read_text())
    probs = saved_probs()
    H, L = load_case("HIGH", prov, roi, probs), load_case("LOW", prov, roi, probs)

    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for col, (C, ttl) in enumerate([(H, "HIGH (S24)"), (L, "LOW (S30)")]):
        xy = C["xy"]
        ax[col, 0].plot(xy[:, 0], xy[:, 1], lw=0.6, color="#444")
        ax[col, 0].scatter(xy[:, 0], xy[:, 1], s=4, c=np.arange(len(xy)), cmap="viridis")
        ax[col, 0].set_title(f"{ttl}\nGaze trajectory (entropy={C['entropy']:.3f})")
        _std_axis(ax[col, 0], xlabel="Gaze x (norm.)", ylabel="Gaze y (norm.)", fmt_x=True, fmt_y=True)
        ax[col, 1].bar(np.arange(1, len(C["roi"]) + 1), C["roi"], color="#3f7d20"); ax[col, 1].set_xticks(np.arange(1, len(C["roi"]) + 1))
        ax[col, 1].set_title("ROI saliency vector (5×2 page-grid cells)")
        _std_axis(ax[col, 1], xlabel="Grid cell (1-5 upper row, 6-10 lower row)", ylabel="Gaze-sample share")
        ax[col, 2].bar(["LOW", "HIGH"], [1 - C["p_high"], C["p_high"]], color=["#30638e", "#d1495b"])
        ax[col, 2].set_ylim(0, 1)
        ax[col, 2].set_title(f"Prediction  (true={'HIGH' if C['true'] else 'LOW'})")
        _std_axis(ax[col, 2], xlabel="Class", ylabel="Probability")
        tot = sum(C["ig"][k] for k in IGN) or 1
        ax[col, 3].bar([IGN[k] for k in IGN], [C["ig"][k] / tot for k in IGN], color="#9b5de5")
        ax[col, 3].set_title("IG attribution (norm.)")
        _std_axis(ax[col, 3], xlabel="Modality", ylabel="Attribution (norm.)")
    fig.suptitle("INS-HDGS-CMT — gaze, ROI, decision & attribution: HIGH vs LOW", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for out in OUT:
        out.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig_gaze_pred.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    for name, C in (("HIGH", H), ("LOW", L)):
        tot = sum(C["ig"][k] for k in IGN)
        print(name, C["subj"], "p_high(saved)=%.3f" % C["p_high"],
              "p_high(untempered)=%.3f" % C["p_high_untempered"],
              "entropy=%.3f" % C["entropy"],
              {IGN[k]: round(C["ig"][k] / tot, 2) for k in IGN})
    print("wrote", ", ".join(str(o / "fig_gaze_pred.{pdf,png}") for o in OUT))


if __name__ == "__main__":
    main()
