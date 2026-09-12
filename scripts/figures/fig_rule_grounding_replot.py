"""
Re-draw Fig. S7 (rule grounding) from the saved attribution file, without the model.

ground_rules_to_electrodes.py needs the fold checkpoints (server) to run integrated
gradients on the rule activations. Its output is saved as
results/explainability/rule_grounding_<tag>.json, which holds the 8 x 19 x 5 signed
attribution maps, the electrode and band names, and the grounded premises of every rule.
This script redraws the figure from that file so the layout can be corrected without a
server run. The premise strings are rebuilt from the same fields the original uses, and
the redrawn text panel is checked against the saved rule_grounding_<tag>.md line for line.

Layout fix relative to the original: the colourbar column and the text panel had no gap,
so the colourbar title and its tick labels overlapped the IF-THEN summaries. A spacer
column is inserted between them and the colourbar title is centred over its own axis.

Run:  python scripts/figures/fig_rule_grounding_replot.py            (repository root)
      python scripts/figures/fig_rule_grounding_replot.py --tag abl_full
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
    """Same premise strings as ground_rules_to_electrodes.py writes into the .md."""
    lines = []
    for r in rules:
        prem = " ∧ ".join(f"{p['electrode']}-{p['band']}{ARROW[p['direction']]}" for p in r["premises"])
        etp = f"{r['et_premise']['stream']}{ARROW[r['et_premise']['direction']]}"
        lines.append(f"R{r['rule']}: IF {prem} ∧ {etp} → {r['conclusion']}")
    return lines


def check_against_md(tag: str, rules) -> None:
    """The .md holds a table row per rule; confirm the redrawn premises match it exactly."""
    md = EXPL / f"rule_grounding_{tag}.md"
    if not md.exists():
        print(f"[check] {md.name} absent; premise strings not cross-checked")
        return
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

    fig = plt.figure(figsize=(16, 9))
    # ... heatmaps ... | colourbar | SPACER | text panel
    gs = fig.add_gridspec(2, R // 2 + 3,
                          width_ratios=[1] * (R // 2) + [0.09, 0.30, 1.6],
                          wspace=0.35, hspace=0.45)
    vmax = np.abs(eeg).max()
    im = None
    for r in range(R):
        ax = fig.add_subplot(gs[r % 2, r // 2])
        im = ax.imshow(eeg[r], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Rule {r + 1} → {rules[r]['conclusion']}", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b[:1].upper() + b[1:] for b in bands], rotation=45, fontsize=7)
        ax.set_yticks(range(len(chan)))
        ax.set_yticklabels(chan, fontsize=6)
    cax = fig.add_subplot(gs[:, R // 2])
    fig.colorbar(im, cax=cax)
    cax.set_title("IG attribution\nof $a_r$ (signed)", fontsize=8, loc="center", pad=10)
    cax.tick_params(labelsize=7)

    axT = fig.add_subplot(gs[:, R // 2 + 2])
    axT.axis("off")
    axT.set_title("Grounded IF–THEN summaries", fontweight="bold", loc="center")
    axT.text(0, 0.98, "\n".join(panel), va="top", ha="left", fontsize=10.5,
             family="monospace", linespacing=1.9, transform=axT.transAxes)
    fig.suptitle(f"Soft-rule premises projected onto scalp electrodes × frequency bands ({args.tag})",
                 fontweight="bold")

    targets = [(EXPL, f"fig_rule_grounding_{args.tag}")]
    if args.tag == "abl_ns_rule_only":                       # the figure the supplement includes
        targets.append((ROOT / "paper" / "figures", "figS7_rule_grounding"))
    for out, stem in targets:
        out.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / (stem + '.{pdf,png}')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
