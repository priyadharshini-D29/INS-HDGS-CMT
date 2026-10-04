# -*- coding: utf-8 -*-
"""
Third-external-audit repo refresh (2026-10-04):

1. Materialise the FINAL production decoder's per-fold operating-point metrics as
   results/ablation/abl_full/losocv_abl_full_final.csv, derived from the saved
   held-out probabilities re-evaluated in results/statistics/rule_fidelity_per_fold.csv
   (gated columns; mean 0.71923 / 0.87899 / 0.44584 = the manuscript headline).
   The historical losocv_abl_full.csv (0.7475 / 0.8790 / 0.4854) is the earlier
   24-channel-loader run and is kept unchanged as a historical artifact.
2. Refresh the superseded bottom table of results/statistics/rule_fidelity.md from
   current per-fold data (final decoder vs the rule-only variant), refusing to write
   unless the recomputed deltas reproduce Supplementary Table S13, and add the
   pooled epoch-level agreement beside the fold-mean agreement.
3. Prepend a SUPERSEDED banner to results/statistics/cross_modal_contribution.md,
   whose per-variant means and balanced-accuracy/MCC rows reflect the earlier
   full-model run.
4. Write results/statistics/STATS_ENVIRONMENT.md recording the statistical software
   and Wilcoxon settings, and the precision bounds of the log-reconstructed CSVs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
STATS = ROOT / "results" / "statistics"

# ---- 1. final decoder per-fold metrics under a canonical name -----------------
pf = pd.read_csv(STATS / "rule_fidelity_per_fold.csv")
final = pd.DataFrame({
    "fold": pf["fold"], "test_subject": pf["test_subject"], "test_n": pf["n"],
    "balanced_acc": pf["balanced_acc_gated"], "roc_auc": pf["roc_auc_gated"],
    "mcc": pf["mcc_gated"],
    "source": "saved held-out probabilities re-evaluated (rule_fidelity_per_fold.csv, gated)",
}).sort_values("fold").reset_index(drop=True)
m = (final.balanced_acc.mean(), final.roc_auc.mean(), final.mcc.mean())
assert len(final) == 37 and abs(m[0] - 0.71923) < 5e-5 and abs(m[1] - 0.87899) < 5e-5, m
out1 = ROOT / "results/ablation/abl_full/losocv_abl_full_final.csv"
final.to_csv(out1, index=False)
print(f"wrote {out1.relative_to(ROOT)} | balacc {m[0]:.5f} auc {m[1]:.5f} mcc {m[2]:.5f}")

# ---- 2. refresh rule_fidelity.md ---------------------------------------------
ro = pd.read_csv(ROOT / "results/ablation/abl_ns_rule_only/losocv_abl_ns_rule_only.csv")
ro = ro.drop_duplicates("test_subject", keep="last").set_index("test_subject")
fu = final.set_index("test_subject")
common = fu.index.intersection(ro.index)
assert len(common) == 37, len(common)

# The published S13 pairing uses the saved held-out probabilities of BOTH models
# re-evaluated at the common 0.5 operating point; the rule-only variant's probability
# files are not in this tree (Brev), and its training-time CSV brackets the published
# row (balanced_acc 0.7147 uncal / 0.7335 cal vs published 0.734), so the table below
# is replaced by a superseded note rather than recomputed from mismatched inputs.

# pooled vs fold-mean agreement
n, agree = pf["n"].to_numpy(float), pf["fidelity_agreement"].to_numpy(float)
pooled = float((n * agree).sum() / n.sum())
foldmean = float(agree.mean())
r_mean = float(pf["r_margin_rule_vs_final"].mean())
print(f"agreement: fold mean {foldmean:.3f} | pooled {pooled:.3f} ({int(round((n*agree).sum()))}/{int(n.sum())}) | mean r {r_mean:.3f}")

md = (STATS / "rule_fidelity.md").read_text(encoding="utf-8")
old_line = ("Decision fidelity: argmax(R) agrees with the final decision on 39.4% of held-out epochs "
            "(fold mean; min 0%); Pearson r between the rule margin (R_HIGH−R_LOW) and the final logit margin: mean -0.691.")
new_line = (f"Decision fidelity: argmax(R) agrees with the final decision on {foldmean*100:.1f}% of held-out epochs "
            f"as a mean over folds (min 0%); pooled over the {int(n.sum())} held-out epochs the share is "
            f"{pooled*100:.1f}% ({int(round((n*agree).sum()))}/{int(n.sum())}). Pearson r between the rule margin "
            f"(R_HIGH−R_LOW) and the final logit margin: mean within-fold {r_mean:.3f}.")
if old_line in md:
    md = md.replace(old_line, new_line)
else:
    # fall back: replace the sentence by locating its prefix
    import re
    md, cnt = re.subn(r"Decision fidelity: argmax\(R\)[^\n]*", new_line, md, count=1)
    assert cnt == 1, "fidelity line not found"

old_tbl_hdr = "## Model TRAINED with α fixed at 0 (`losocv_abl_ns_rule_only.csv`, 37 paired folds)"
i = md.find(old_tbl_hdr)
assert i != -1, "bottom table header not found"
new_tbl = [old_tbl_hdr, "",
    "> **SUPERSEDED (2026-10-04).** The comparison that stood here (full 0.747 vs rule-only 0.715,",
    "> p = 0.047/0.014/0.093) paired the earlier 24-channel full run with the rule-only variant's",
    "> training-time export. The manuscript's final comparison (Supplementary Table S13) pairs the",
    "> saved held-out probabilities of the final production run (`losocv_abl_full_final.csv`;",
    "> balanced accuracy 0.719, ROC-AUC 0.879, MCC 0.446) with those of the rule-only variant",
    "> (0.734, 0.839, 0.497) at the common 0.5 operating point: Δ = +0.014 / −0.040 / +0.051,",
    "> Wilcoxon p = 0.42 / 0.018 / 0.23. The rule-only variant's per-fold probability files were",
    "> produced on the Brev instance and are not in this tree; its training-time export brackets",
    "> the published row (balanced_acc 0.7147 uncalibrated, 0.7335 calibrated).",
    ""]
md = md[:i] + "\n".join(new_tbl) + "\n"
(STATS / "rule_fidelity.md").write_text(md, encoding="utf-8", newline="\n")
print("rewrote results/statistics/rule_fidelity.md")

# ---- 3. banner on cross_modal_contribution.md --------------------------------
cm_path = STATS / "cross_modal_contribution.md"
cm = cm_path.read_text(encoding="utf-8")
banner = ("> **SUPERSEDED (2026-10-04).** The `full` row below (balanced accuracy 0.747, MCC 0.485) and every\n"
          "> balanced-accuracy/MCC comparison derived from it reflect the earlier 24-channel full-model run.\n"
          "> The manuscript's cross-modal comparisons (Supplementary Table S14) pair the variants against the\n"
          "> final production run (balanced accuracy 0.719, MCC 0.446; `results/ablation/abl_full/losocv_abl_full_final.csv`).\n"
          "> The ROC-AUC rows are numerically unaffected because both full runs reach 0.879.\n\n")
if not cm.startswith("> **SUPERSEDED"):
    cm_path.write_text(banner + cm, encoding="utf-8", newline="\n")
    print("bannered results/statistics/cross_modal_contribution.md")

# ---- 4. STATS_ENVIRONMENT.md --------------------------------------------------
env = f"""# Statistical environment and settings for the manuscript comparisons

- Paired comparisons: two-sided Wilcoxon signed-rank test, zero differences discarded
  (`scipy.stats.wilcoxon(x, zero_method="wilcox")`, default two-sided alternative and
  automatic exact/normal method selection), Holm–Bonferroni correction within each stated
  family; matched-pairs rank-biserial correlation as effect size; 95% bootstrap CIs of the
  mean paired difference (10,000 resamples). Implementation: `scripts/revision/round2_tests.py`
  and the released analysis scripts.
- SciPy on this machine when the 2026-10-04 refresh was run: {scipy.__version__}
  (numpy {np.__version__}, pandas {pd.__version__}). The original Brev runs used the
  instance's own SciPy; `VERIFICATION_2026-09-29.md` documents the version-dependent
  p-value differences observed between SciPy builds (exact vs normal-approximation paths).
- `losocv_eegnet_et.csv`, `losocv_shallow_et.csv` (engagement index) and the
  `dl_tuned_v2` fold-wise `losocv_{{eegnet,shallow,et_lstm}}.csv` are reconstructed from the
  sliced Brev training logs (`scripts/revision/reconstruct_losocv_from_logs.py`): per-fold
  metrics are exact to the 3 decimals the logs print, so paired statistics recomputed from
  them can differ from the published full-precision values in tie counts and in the third
  decimal of p-values. Their epoch-level probabilities (ECE/Brier, pooled curves) were
  computed on the Brev instance and are not reconstructable from the logs.
- The final production decoder's per-fold operating-point metrics are
  `results/ablation/abl_full/losocv_abl_full_final.csv` (mean balanced accuracy 0.71923,
  ROC-AUC 0.87899, MCC 0.44584), derived from the saved held-out probabilities; the
  historical `losocv_abl_full.csv` (0.7475/0.8790/0.4854) is the earlier 24-channel run.
- Known remaining gap: the per-fold held-out probability files of the rule-only variant
  (`abl_ns_rule_only`) and of the two engagement-index late fusions were produced on the
  Brev instance and are not in this tree, so Supplementary Table S13's rule-only row and
  the fusions' ECE/Brier cannot be recomputed here; their fold-level metrics are bracketed
  or reproduced by the files named above.
"""
(STATS / "STATS_ENVIRONMENT.md").write_text(env, encoding="utf-8", newline="\n")
print("wrote results/statistics/STATS_ENVIRONMENT.md")
