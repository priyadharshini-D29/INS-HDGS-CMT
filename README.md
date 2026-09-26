# INS-HDGS-CMT — manuscript sources and per-fold results

This branch holds the sources behind the manuscript
**"What Does a Multimodal EEG–Eye-Tracking Engagement Decoder Learn?
A Leakage-Aware Audit on the NeuMa Dataset"** (under review at *Brain Informatics*):
all code, per-fold results (CSV), statistics and manuscript TeX.

- **Start here: [`REPRODUCING.md`](REPRODUCING.md)** — how to verify every reported
  number from the committed per-fold files without retraining, and how to retrain
  everything from the public NeuMa release (commands, seeds, expected run-to-run spread).
- `paper/` — manuscript and supplementary TeX (the PDFs compile from these sources).
- `results/` — per-fold metrics, held-out probabilities and statistics reports.
- `src/`, `scripts/` — pipeline, models, training and analysis code.

Binary artifacts (checkpoints, compiled PDFs, rendered figures) are not distributed;
they regenerate from the seeded scripts, and no reported statistic depends on them.
The project overview, dataset instructions and citation are on the
[`main`](../../tree/main) branch.
