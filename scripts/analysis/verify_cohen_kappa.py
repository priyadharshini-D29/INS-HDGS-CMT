"""
Reproduce the Cohen's kappa values reported in the manuscript (Table tab:kappa)
directly from the released per-fold LOSOCV predictions. No stored kappa columns
are trusted: kappa is recomputed from y_true and y_prob with sklearn.

Run:  python analysis/verify_cohen_kappa.py
Source predictions:
  full model : results/losocv_metrics/losocv_repro_focal_g3p0_effective_num_37.csv
  EEG branch : results/ablation/abl_no_et/losocv_abl_no_et.csv
Each row stores y_true, y_prob (raw), opt_threshold, T_post, opt_threshold_cal.
"""
import csv, ast, statistics as st
import numpy as np
from sklearn.metrics import cohen_kappa_score

def kappas(fn):
    rows = list(csv.DictReader(open(fn)))
    raw, cal, yt_pool, pr_pool = [], [], [], []
    for r in rows:
        yt = np.array(ast.literal_eval(r["y_true"]))
        p  = np.array(ast.literal_eval(r["y_prob"]))
        # raw operating point: validation-tuned threshold (leakage-free)
        raw.append(cohen_kappa_score(yt, (p >= float(r["opt_threshold"])).astype(int)))
        # calibrated: standard temperature scaling + calibrated threshold
        T, thrc = float(r["T_post"]), float(r["opt_threshold_cal"])
        eps = 1e-7
        logit = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps))
        pcal = 1.0 / (1.0 + np.exp(-logit / T))
        cal.append(cohen_kappa_score(yt, (pcal >= thrc).astype(int)))
        yt_pool += list(yt); pr_pool += list((p >= 0.5).astype(int))
    return (round(st.mean(raw), 4), round(st.mean(cal), 4),
            round(cohen_kappa_score(yt_pool, pr_pool), 4), len(rows))

if __name__ == "__main__":
    for fn, name in [
        ("results/losocv_metrics/losocv_repro_focal_g3p0_effective_num_37.csv", "Full multimodal model"),
        ("results/ablation/abl_no_et/losocv_abl_no_et.csv", "EEG branch (leakage-free)")]:
        raw, cal, pooled, n = kappas(fn)
        print(f"{name:28s} (n={n})  raw={raw}  calibrated={cal}  pooled@0.5={pooled}")
