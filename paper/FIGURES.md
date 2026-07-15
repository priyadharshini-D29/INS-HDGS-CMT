# Figure index — paper → file

The manuscript's figure **numbers** and the figure **filenames** in
[`figures/`](figures/) do not correspond. The filenames are historical and are
kept unchanged because `INS_HDGS_CMT_manuscript.tex` references them directly —
renaming them would break the build. This file is the authoritative mapping.

`\graphicspath{{figures/}}` is set in the manuscript, so each `\includegraphics`
resolves against [`figures/`](figures/).

| Paper | Label | Caption (abridged) | File |
|---|---|---|---|
| Fig. 1 | `fig1` | Representative synchronised multimodal (EEG, gaze, pupil) epoch, subject S24 | [`fig1_real_overlay.pdf`](figures/fig1_real_overlay.pdf) |
| Fig. 2 | `fig2` | Overview of the INS-HDGS-CMT architecture | [`fig2_Architecture.pdf`](figures/fig2_Architecture.pdf) |
| Fig. 3 | `fig3` | Dynamic EEG functional-graph construction | [`fig_graph.pdf`](figures/fig_graph.pdf) |
| Fig. 4 | `fig4` | Cross-modal fusion and neuro-symbolic reasoning | [`fig4_cross-modal.pdf`](figures/fig4_cross-modal.pdf) |
| Fig. 5 | `fig5` | Classification performance under LOSOCV | [`fig3_losocv_results.pdf`](figures/fig3_losocv_results.pdf) |
| Fig. 6 | `fig6` | EEG leakage-controlled comparison (Nemenyi critical difference) | [`fig6_combined.pdf`](figures/fig6_combined.pdf) |
| Fig. 7 | `fig7` | Engagement is encoded multivariately, not by any single marker | [`fig_concordance_depth.pdf`](figures/fig_concordance_depth.pdf) |
| Fig. 8 | `fig8` | Explainability for one held-out subject (S01) | [`fig5_explainability.pdf`](figures/fig5_explainability.pdf) |
| Fig. 9 | `fig9` | Eye-tracking phenotype of engagement (385 epochs) | [`fig_et_phenotype.pdf`](figures/fig_et_phenotype.pdf) |
| Fig. 10 | `fig10` | Gaze, ROI-saliency, decision and attribution (S24 / S30) | [`fig_gaze_pred.pdf`](figures/fig_gaze_pred.pdf) |

## Watch out

Three filenames are actively misleading — they carry a number that is **not**
their figure number:

| File | Reads as | Actually is |
|---|---|---|
| `fig3_losocv_results.pdf` | Figure 3 | **Figure 5** |
| `fig5_explainability.pdf` | Figure 5 | **Figure 8** |
| `fig_graph.pdf` | — | **Figure 3** |

## Not referenced by the manuscript

These files are present in [`figures/`](figures/) but are **not** included by
`INS_HDGS_CMT_manuscript.tex`. They are earlier or alternative versions, kept for
provenance:

- `fig_losocv.pdf`
- `fig_regions.pdf`
- `fig_roi_timecourse.png`
- `fig1_dataflow_corrected.tex`

## Regenerating

```bash
bash ../reproducibility/generate_figures.sh
```

Source generators live in [`../scripts/figures/`](../scripts/figures/).
