# Cross-machine verification of the Fig. 6 / Table 6 / Table S14 statistics (2026-09-29)

The control-ladder statistics of the manuscript (Fig. 6, Table 6, Tables S13/S14) were verified
against the Brev compute instance (`brev-x1v47hbeh`, repo `~/INS-HDGS-CMT-real`, env `fignn_env`)
on 2026-09-29, before the instance window closed. Findings:

## Data identity
`results/figures/control_ladder.csv` is line-identical on the laptop and the box, and the
per-fold CSVs of `abl_full`, the twelve variants and the ET-LSTM baseline produce identical
per-fold metrics on both machines, so both hold the same runs.

## Effect sizes: bit-exact
At the manuscript's operating point (per-fold balanced accuracy and MCC from the saved held-out
probabilities thresholded at 0.5; ROC-AUC from the stored per-fold column), **all 39 paired mean
differences reproduce to four decimals on both machines and equal the values printed in
Table 6 and Table S14**, as do the matched-pairs rank-biserial correlations (e.g. gaze-free
removal r_rb = 0.96, ET-LSTM r_rb = -0.05) and the win/tie/loss counts (29/6/2, 25/5/7,
17/16/4, 8/22/7, 10/18/9, ...).

## Wilcoxon p-values: exact where ranks are untied, library-tolerant where they are not
Per-fold score differences can be equal up to one ulp across save paths; the published analysis
treated those as ties. With that tie rule (|d| < 1e-9), the current scipy (1.17) reproduces the
published p **exactly (3 d.p.)** in every cell whose nonzero |differences| are essentially
untied in rank: all twelve MCC cells (0.381, 0.232, 0.028, 0.039, 0.519, 0.528, 0.233, 0.259,
0.176, 0.083, 0.661, 0.277) and the ROC cells 0.410 (-ROI), 0.733 (-fusion), 0.011 (-graph).
The remaining ROC/balanced-accuracy cells carry heavily tied ranks, where the tie handling in
the variance of the signed-rank normal approximation differs across scipy versions; there the
published values (computed with the box's earlier scipy) and the current recomputation differ
in the second-to-third decimal:

| cell | published | scipy 1.17 |
|---|---|---|
| -Spiking, ROC / BalAcc | 0.434 / 0.460 | 0.407 / 0.448 |
| -ROI, BalAcc | 0.210 | 0.191 |
| -Fusion, BalAcc | 0.052 | 0.048 |
| -Neuro-symbolic, ROC / BalAcc | 0.360 / 0.043 | 0.338 / 0.039 |
| -Contrastive, ROC / BalAcc | 0.897 / 0.890 | 0.856 / 0.776 |
| -MMD, ROC / BalAcc | 0.443 / 0.555 | 0.466 / 0.556 |
| Rule gate closed, ROC / BalAcc / MCC | 0.018 / 0.419 / 0.233 | 0.014 / 0.429 / 0.233 |
| -Graph, BalAcc | 0.265 | 0.284 |
| Static null, ROC / BalAcc | 0.293 / 0.232 | 0.268 / 0.226 |
| Random null, ROC / BalAcc | 0.715 / 0.087 | 0.695 / 0.088 |
| -Gaze seq., ROC / BalAcc | 0.025 / 0.306 | 0.024 / 0.284 |
| ET-LSTM, ROC / BalAcc | 0.840 / 0.290 | 0.856 / 0.278 |

## Conclusions are insensitive to the difference
Under either p-set: no single-component ablation survives Holm correction on any metric
(smallest adjusted p = 0.22, -fusion on MCC), the topology nulls produce no detectable change,
the rule-gate-closed ROC-AUC cost is significant raw (0.018 published / 0.014 current), and the
gaze-input removal remains p < 0.001 throughout. The manuscript's printed values are the
original coherent computation and are retained; `scripts/figures/fig_control_ladder.py` now
implements the manuscript's operating point and tie rule, so a re-run reproduces every delta,
r_rb and W/T/L exactly and every p up to the documented tie-handling tolerance.
