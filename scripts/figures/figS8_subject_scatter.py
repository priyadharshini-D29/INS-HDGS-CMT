#!/usr/bin/env python3
"""Supplementary Fig. S8: per-subject view of the two headline comparisons (Sections 3.2 and 3.5).

  (A) full model vs gaze-free branch, per held-out subject ROC-AUC  (gaze dependence, subject by subject)
  (B) full model vs tuned ET-LSTM, per held-out subject ROC-AUC     (parity with a gaze-only encoder)

Points above the diagonal favour the full model. Counts of subjects above / on / below the diagonal are
printed in each panel. Same subjects as Table S14 (fold-matched).

Many folds hold only ~9 epochs, so per-subject ROC-AUC saturates: in panel B sixteen subjects share the
single point (1.00, 1.00). Subjects with identical scores are therefore drawn as ONE marker, sized by
and labelled with the number of subjects it stands for, and only uniquely placed subjects carry an id.
Labels are positioned by a greedy collision search so that none overlap or leave the axes.

Output : paper/figures/figS8_subject_scatter.{pdf,png}
Run    : python scripts/figures/figS8_subject_scatter.py     (repository root; CPU)
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "results/ablation/abl_full/losocv_abl_full.csv"
PANELS = [("A", "Gaze-free variant", ROOT / "results/ablation/abl_eeg_only/losocv_abl_eeg_only.csv", "#c44e52"),
          ("B", "ET-LSTM, gaze only (tuned)", ROOT / "results/baselines/dl_tuned/losocv_et_lstm.csv", "#55a868")]

LIM = 1.12          # axis limit: leaves room for labels on points at 1.00
AX_PT = 288.0       # approximate axes width in points, for label-size estimates
FS = 6.0            # label font size
# candidate label offsets (axes fraction), tried in order: right, left, above, below, diagonals, further out
OFFSETS = [(0.018, 0.010), (-0.018, 0.010), (0.018, -0.020), (-0.018, -0.020),
           (0.000, 0.022), (0.000, -0.030), (0.040, 0.010), (-0.040, 0.010),
           (0.040, -0.020), (-0.040, -0.020), (0.000, 0.042), (0.000, -0.050),
           (0.062, 0.010), (-0.062, 0.010), (0.062, -0.020), (-0.062, -0.020),
           (0.018, 0.034), (-0.018, 0.034), (0.018, -0.044), (-0.018, -0.044),
           (0.000, 0.062), (0.000, -0.072), (0.088, 0.010), (-0.088, 0.010),
           (0.040, 0.034), (-0.040, 0.034), (0.040, -0.044), (-0.040, -0.044)]


def load(csv):
    return pd.read_csv(csv).drop_duplicates("test_subject", keep="last").set_index("test_subject")


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def place_labels(ax, items, occupied):
    """items: (x_frac, y_frac, text, ha_anchor). Greedy placement avoiding `occupied` rectangles."""
    placed = list(occupied)
    skipped = []
    for xf, yf, text in items:
        w = len(text) * 0.60 * FS / AX_PT
        h = 1.35 * FS / AX_PT
        for dx, dy in OFFSETS:
            x0 = xf + dx if dx >= 0 else xf + dx - w
            y0 = yf + dy if dy >= 0 else yf + dy
            rect = (x0 - 0.004, y0 - 0.004, x0 + w + 0.004, y0 + h + 0.004)
            if rect[0] < 0.0 or rect[2] > 1.0 or rect[1] < 0.0 or rect[3] > 1.0:
                continue
            if any(_overlap(rect, r) for r in placed):
                continue
            ax.text(x0, y0, text, transform=ax.transAxes, fontsize=FS, color="0.30",
                    ha="left", va="bottom", zorder=4)
            placed.append(rect)
            break
        else:
            skipped.append(text)
    return skipped


def main():
    full = load(FULL)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    for ax, (tag, name, csv, col) in zip(axes, PANELS):
        if not csv.exists():
            ax.text(0.5, 0.5, f"{csv.relative_to(ROOT)}\nmissing", ha="center", va="center", transform=ax.transAxes)
            print(f"[{tag}] missing {csv}")
            continue
        v = load(csv)
        common = sorted(set(full.index) & set(v.index))
        x = v.loc[common, "roc_auc"].astype(float).to_numpy()
        y = full.loc[common, "roc_auc"].astype(float).to_numpy()

        # coincident subjects share one marker
        groups = defaultdict(list)
        for s, xi, yi in zip(common, x, y):
            groups[(round(float(xi), 3), round(float(yi), 3))].append(s)

        ax.plot([0, 1], [0, 1], "--", color="0.5", lw=0.9, zorder=1)
        ax.axhline(0.5, color="0.88", lw=0.7, zorder=0)
        ax.axvline(0.5, color="0.88", lw=0.7, zorder=0)
        ax.set_xlim(0, LIM)
        ax.set_ylim(0, LIM)
        ax.set_aspect("equal")

        gx = np.array([p[0] for p in groups])
        gy = np.array([p[1] for p in groups])
        gn = np.array([len(v_) for v_ in groups.values()])
        ax.scatter(gx, gy, s=30 + 16 * (gn - 1), color=col, alpha=0.85,
                   edgecolor="white", linewidth=0.7, zorder=3)

        # inset: the bottom strip is empty in both panels (lowest subject sits near 0.40)
        above, on, below = int((y > x).sum()), int((y == x).sum()), int((y < x).sum())
        ax.text(0.025, 0.025,
                f"decoder higher: {above} / equal: {on} / lower: {below} of {len(common)} subjects\n"
                f"mean ROC-AUC {y.mean():.3f} vs {x.mean():.3f}",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5, zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))
        inset_rect = (0.010, 0.010, 0.980, 0.115)          # keep labels clear of the inset

        items = []
        for (px, py), subs in groups.items():
            text = subs[0] if len(subs) == 1 else f"{len(subs)} subjects"
            items.append((px / LIM, py / LIM, text))
        items.sort(key=lambda it: -len(it[2]))              # place the wide count labels first
        skipped = place_labels(ax, items, [inset_rect])

        ax.set_xlabel(f"{name}: ROC-AUC per held-out subject")
        ax.set_ylabel("Decoder (EEG + gaze): ROC-AUC per held-out subject")
        ax.set_title(f"({tag}) Decoder vs {name.split(',')[0]}", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25, zorder=0)
        ties = {k: v_ for k, v_ in groups.items() if len(v_) > 1}
        print(f"[{tag}] n={len(common)} above={above} on={on} below={below} "
              f"mean full={y.mean():.3f} other={x.mean():.3f} | {len(groups)} distinct positions"
              + (f" | coincident: {[(k, len(v_)) for k, v_ in ties.items()]}" if ties else "")
              + (f" | UNPLACED LABELS: {skipped}" if skipped else ""))
    fig.tight_layout()
    for out in (ROOT / "paper/figures", ROOT / "results/figures"):
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "figS8_subject_scatter.pdf", bbox_inches="tight")
        fig.savefig(out / "figS8_subject_scatter.png", dpi=250, bbox_inches="tight")
    print("wrote paper/figures/figS8_subject_scatter.{pdf,png}")


if __name__ == "__main__":
    main()
