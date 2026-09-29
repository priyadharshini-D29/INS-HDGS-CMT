# INS-HDGS-CMT — code, reproducibility and per-fold results

This branch holds the analysis behind the manuscript
**"What Does a Multimodal EEG–Eye-Tracking Engagement Decoder Learn?
A Leakage-Aware Audit on the NeuMa Dataset"** (under review at *Brain Informatics*):
all code, per-fold results (CSV), result tables and statistics reports.

- **Start here: [`REPRODUCING.md`](REPRODUCING.md)** — how to verify every reported
  number from the committed per-fold files without retraining, and how to retrain
  everything from the public NeuMa release (commands, seeds, expected run-to-run spread).
- `results/` — per-fold metrics, held-out probabilities and statistics reports.
- `tables/` — result tables exported from the per-fold files.
- `src/`, `scripts/` — pipeline, models, training and analysis code.

The manuscript and supplementary sources are with the journal during review and
will be deposited in this branch upon publication.
Binary artifacts (checkpoints, compiled PDFs, rendered figures) are not distributed;
they regenerate from the seeded scripts, and no reported statistic depends on them.
The project overview, dataset instructions and citation are on the
[`main`](../../tree/main) branch.
