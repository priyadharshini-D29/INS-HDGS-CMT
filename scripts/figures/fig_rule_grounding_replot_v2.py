"""
Re-draw Fig. S7 (rule grounding) from the saved attribution file, without the model,
at print-legible type sizes (fig_rule_grounding_replot.py kept the original 16x9-in
side-by-side canvas, which renders its electrode labels at about 2.3 pt once the
figure is scaled to the supplement's text width).

Changes relative to fig_rule_grounding_replot.py:
  - electrode tick labels only on the leftmost map of each row and band labels only
    on the bottom row (the eight maps share both axes), so the labels can be large;
  - the attribution-summary text panel moves below the maps and spans the full width;
  - horizontal colourbar below the maps;
  - all font sizes raised; intrinsic canvas narrowed to ~11 in.
Content is identical: the same 8 x 19 x 5 signed maps are drawn from the same
rule_grounding_<tag>.json, and the premise strings are still checked line for line
against the saved rule_grounding_<tag>.md before the figure is written.

Run:  python scripts/figures/fig_rule_grounding_replot_v2.py            (repository root)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EXPL = ROOT / "results" / "explainability"
ARROW = {"up": "↑", "down": "↓"}


def build_text_panel(rules) -> list[str]:
    lines = []
    for r in rules:
        prem = " ∧ ".join(f"{p['electrode']}-{p['band']}{ARROW[p['direction']]}" for p in r["premises"])
        etp = f"{r['et_premise']['stream']}{ARROW[r['et_premise']['direction']]}"
        lines.append(f"R{r['rule']}: IF {prem} ∧ {etp} → {r['conclusion']}")
    return lines


def check_against_md(tag: str, rules) -> None:
    md = EXPL / f"rule_grounding_{tag}.md"
    if not md.exists():
        raise RuntimeError(f"{md.name} absent; refusing to draw unchecked premises")
    rows = {}
    for line in md.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].isdigit():
            rows[int(cells[0])] = cells
    bad = []
    for r in rules:
        row = rows.get(r["rule"])
        prem = " ∧ ".join(f"{p['electrode']}-{p['band']}{ARROW[p['direction']]}" for p in r["premises"])
        etp = f"{r['et_premise']['stream']}{ARROW[r['et_premise']['direction']]}"
        if row is None or row[1] != r["conclusion"] or row[4] != prem or row[5] != etp:
            bad.append((r["rule"], prem, etp, row))
    if bad:
        raise RuntimeError(f"premises differ from {md.name}: {bad}")
    print(f"[check] all {len(rules)} rules match {md.name} (premises, ET term, conclusion)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="abl_ns_rule_only")
    args = ap.parse_args()

    src = EXPL / f"rule_grounding_{args.tag}.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    eeg = np.asarray(d["eeg_attr"], float)          # (rules, electrodes, bands)
    chan, bands, rules = d["electrodes"], d["bands"], d["rules"]
    R = len(rules)
    panel = build_text_panel(rules)
    check_against_md(args.tag, rules)

    ncol = R // 2                                   # 4 map columns x 2 rows
    fig = plt.figure(figsize=(11.0, 13.2))
    gs = fig.add_gridspec(5, ncol,
                          height_ratios=[1.0, 1.0, 0.16, 0.05, 0.42],
                          wspace=0.10, hspace=0.30)
    vmax = np.abs(eeg).max()
    im = None
    for r in range(R):
        row, col = r % 2, r // 2
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(eeg[r], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Rule {r + 1} → {rules[r]['conclusion']}", fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(bands)))
        ax.set_yticks(range(len(chan)))
        if row == 1:
            ax.set_xticklabels([b[:1].upper() + b[1:] for b in bands], rotation=45, fontsize=11)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_yticklabels(chan, fontsize=10)
        else:
            ax.set_yticklabels([])

    caxrow = gs[2, 1:3].subgridspec(2, 1, height_ratios=[0.55, 0.45], hspace=0.0)
    cax = fig.add_subplot(caxrow[1, 0])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.set_xlabel("IG attribution of $a_r$ (signed)", fontsize=11, labelpad=6)
    cb.ax.tick_params(labelsize=10)

    axT = fig.add_subplot(gs[4, :])
    axT.axis("off")
    axT.set_title("Attribution summaries (IF–THEN shorthand)", fontweight="bold",
                  fontsize=13, loc="center")
    axT.text(0.5, 0.92, "\n".join(panel), va="top", ha="center", fontsize=13,
             family="monospace", linespacing=1.75, transform=axT.transAxes)
    fig.suptitle("Rule activations attributed to scalp electrodes × frequency bands (rule-only variant)",
                 fontweight="bold", fontsize=14, y=0.995)

    out = ROOT / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figS7_rule_grounding_r5.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out / 'figS7_rule_grounding_r5.{pdf,png}'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
