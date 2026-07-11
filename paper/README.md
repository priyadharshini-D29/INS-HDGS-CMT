# `paper/` — Manuscript sources

LaTeX sources, bibliography and compiled PDF of the manuscript.

- `INS_HDGS_CMT_manuscript.tex` — main manuscript source (Overleaf v5, definitive)
- `INS_HDGS_CMT_manuscript.pdf` — compiled PDF
- `references.bib`, `references_full.bib` — bibliography
- `sn-jnl.cls`, `sn-mathphys-num.bst` — Springer Nature journal class/style
- `figures/` — the exact figure files the manuscript PDF was built from

Build (with a TeX distribution): `latexmk -pdf INS_HDGS_CMT_manuscript.tex`
(or upload to Overleaf). LaTeX build junk is git-ignored.
