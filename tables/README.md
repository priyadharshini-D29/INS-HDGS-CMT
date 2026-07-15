# `tables/` — Manuscript tables

Paper tables in machine-readable (`.csv`), human-readable (`.md`) and typeset
(`.tex`) form. [`latex/`](latex/) holds the Overleaf-ready `.tex` fragments.

## Paper → file

| Paper | Label | Subject | Typeset | Data |
|---|---|---|---|---|
| Table 1 | `tab1` | NeuMa dataset characteristics | [`latex/table1_dataset.tex`](latex/table1_dataset.tex) | — |
| Table 2 | `tab2` | Implementation details / hyperparameters | [`latex/table2_implementation.tex`](latex/table2_implementation.tex) | — |
| Table 3 | `tab3` | EEG-encoder comparison (leakage-free headline) | [`latex/table3_eeg_encoders.tex`](latex/table3_eeg_encoders.tex) | [`table1_eeg_encoders.csv`](table1_eeg_encoders.csv) |
| Table 4 | `tab4` | Eye-tracking encoders | [`latex/table4_et_encoders.tex`](latex/table4_et_encoders.tex) | [`table2_et_encoders.csv`](table2_et_encoders.csv) |
| Table 5 | `tab5` | Multimodal fusion (label-coupled) | [`latex/table5_fusion.tex`](latex/table5_fusion.tex) | [`table3_fusion.csv`](table3_fusion.csv) |
| Table 6 | `tab6` | Contextual Cohen's κ vs prior NeuMa work | *manuscript only* | — |
| Table 7 | `tab7` | Component ablation | [`latex/table6_ablation.tex`](latex/table6_ablation.tex) | — |
| Table 8 | `tab8` | Proposed EEG branch vs each baseline (Wilcoxon, Holm) | [`latex/table7_significance.tex`](latex/table7_significance.tex) | [`ranks_eeg_mcc.csv`](ranks_eeg_mcc.csv) |
| Pipeline table | `tab_pipeline` | Eight-stage processing pipeline | *manuscript only* | — |

## Numbering is offset — read this

Filenames here predate the final manuscript numbering and are **not** renamed, so
that existing references stay stable. Two are actively misleading:

| File | Reads as | Actually is |
|---|---|---|
| `latex/table6_ablation.tex` | Table 6 | **Table 7** |
| `latex/table7_significance.tex` | Table 7 | **Table 8** |
| `table1_eeg_encoders.csv` | Table 1 | **Table 3** |
| `table2_et_encoders.csv` | Table 2 | **Table 4** |
| `table3_fusion.csv` | Table 3 | **Table 5** |

Supporting the supplementary material (not main-paper tables):

- `latex/table8_ig_features.tex` — most influential inputs by Integrated Gradients
- `latex/table9_case_study.tex` — HIGH vs LOW engagement case study

## Other files

- `ranks_eeg_balanced_acc.csv`, `ranks_eeg_mcc.csv`, `ranks_eeg_roc_auc.csv` —
  per-metric baseline rankings behind the Nemenyi analysis (paper Fig. 6).
- `TABLES_REPORT.md` — how each table maps to the raw result files.

## Regenerating

```bash
bash ../reproducibility/generate_tables.sh
```
