# Release Notes — v1.1.0 (2026-09-18)

**Revision update.** The repository documents now match the revised manuscript submitted to *Brain Informatics* (manuscript b2be65ae-0c55-48d8-a981-3da9d290f0e4).

## Changed
- `docs/ARCHITECTURE.md` Section 8 now reports the run used in the paper (`ins_hdgs_cmt_ch19fix`, 19-electrode montage): accuracy 78.24 %, balanced accuracy 74.75 %, MCC 0.49, ROC-AUC 0.88 (uncalibrated operating point), with the manuscript's framing (label-coupled headline; gaze-free EEG branch 0.59; controls). The pre-montage-fix numbers (accuracy 0.7976, MCC 0.5304) are marked retired.
- Superseded banners added to the historical result documents (`src/model/docs/FOCAL_ABLATION_RESULTS.md`, the metric-boosting guides) so that their exploratory numbers cannot be mistaken for the paper's results.
- `docs/REPRODUCIBILITY_CHECKLIST.md` lists the reference values a correct re-run should approach.
- `datasets/README.md` added: where to obtain NeuMa (Scientific Data descriptor; figshare raw and preprocessed deposits with DOIs) and how the pipeline expects it.
- `README.md`: dataset DOIs added.
- `docs/RELEASE_NOTES_v1.0.md`: title corrected to the submitted title; the list of included folders corrected.

## Unchanged
- Model code, configuration defaults and reproduction scripts.
