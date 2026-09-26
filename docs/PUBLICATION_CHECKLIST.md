# Publication Checklist

Pre-submission / camera-ready checklist for a Q1 journal (Brain Informatics,
Nature Portfolio, IEEE, Elsevier, Springer).

## Manuscript
- [ ] Title, authors, affiliations, ORCIDs finalised
- [ ] Abstract ≤ journal word limit
- [ ] Keywords set
- [ ] All figures referenced in text and high-resolution (≥ 300 dpi / vector)
- [ ] All tables referenced in text; units and significant figures consistent
- [ ] Statistical tests reported with effect sizes and corrected p-values
- [ ] Limitations and ethical considerations discussed
- [ ] References complete and consistently formatted (`paper/references.bib`)

## Code & data (reproducibility)
- [ ] Public repository link in the paper (this repo)
- [ ] `README.md` complete (install, train, evaluate, reproduce)
- [ ] `requirements.txt` / `environment.yml` pinned and tested from clean env
- [ ] `LICENSE` present (MIT for code; CC-BY-4.0 intended for figures/text)
- [ ] `CITATION.cff` present and valid
- [ ] Random seed, package/CUDA/Python versions documented
- [ ] Dataset access documented; **no proprietary data committed**
- [ ] One-command reproduction script works (`reproducibility/run_all.sh`)
- [ ] Reported numbers traceable to committed `results/` CSVs
- [ ] Git tag / release (`v1.0`) and archival DOI (Zenodo) minted

## Ethics & compliance
- [ ] Human-subjects data used under the original dataset's terms/IRB
- [ ] No personally identifiable information in repo or figures
- [ ] Conflicts of interest and funding statements included

## Supplementary
- [ ] Supplementary tables/figures uploaded (`supplementary/`)
- [ ] Model complexity, runtime, memory reported
- [ ] Subject-wise performance and failure cases included
- [ ] Calibration analysis included

## Archival
- [ ] Zenodo/OSF snapshot of the repository at the reported commit
- [ ] DOI badge added to `README.md` and `CITATION.cff`
