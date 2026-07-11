"""
================================================================
INS-HDGS-CMT — Construct Validity (PRIMARY): EEG connectivity ↔ engagement
================================================================
The model's most important component is the EEG functional-connectivity graph
(ablation: removing it costs Δbal-acc −0.057, Wilcoxon p=0.007). Engagement in
this dataset is therefore encoded in CONNECTIVITY structure, not regional band
power (see analysis/eeg_concordance.py, which found only weak power effects in
this delta-dominated, eyes-open paradigm).

This script tests the construct from the independent EEG modality using the SAME
quantity the model consumes: phase-locking value (PLV). If the ET-derived
HIGH/LOW labels capture a genuine engagement state, fronto-parietal PLV — a
canonical correlate of attentional engagement — should differ between HIGH and
LOW epochs.

METHOD (leakage-free, within-subject)
-------------------------------------
* Raw epochs (1500×24 @300Hz) + matching phase3d HIGH/LOW labels per subject.
* Per band (theta 4–8, alpha 8–13 Hz): zero-phase band-pass → Hilbert phase →
  PLV between every frontal×posterior channel pair; mean over pairs = the
  epoch's fronto-parietal PLV.
    frontal  : Fz, F3, F4, FC1, FC2
    posterior: Pz, Oz, O1, O2, P3, P4, P7, P8
* Within-subject epoch-level permutation test (subject-centred PLV; labels
  shuffled within subject), two-sided. Subject = structure unit. Reports Cohen's
  d and a per-subject-mean Wilcoxon as a secondary check.

Usage
-----
    python analysis/connectivity_concordance.py
================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import wilcoxon

# reuse the loader/harmonisation from the band-power script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eeg_concordance import _load_subject, _subjects, IDX, FS, OUT_DIR  # noqa: E402

FRONTAL = ["Fz", "F3", "F4", "FC1", "FC2"]
POSTERIOR = ["Pz", "Oz", "O1", "O2", "P3", "P4", "P7", "P8"]
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0)}


def _phase(epoch_band):
    """Analytic-signal phase of an already band-passed (T,C) array."""
    return np.angle(hilbert(epoch_band, axis=0))


def _bandpass(epoch, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, epoch, axis=0)


def _fronto_parietal_plv(epoch, band):
    lo, hi = BANDS[band]
    ph = _phase(_bandpass(epoch, lo, hi))           # (T, C)
    f_idx = [IDX[c] for c in FRONTAL if c in IDX and np.any(epoch[:, IDX[c]] != 0)]
    p_idx = [IDX[c] for c in POSTERIOR if c in IDX and np.any(epoch[:, IDX[c]] != 0)]
    if not f_idx or not p_idx:
        return np.nan
    plvs = []
    for fi in f_idx:
        d = ph[:, p_idx] - ph[:, [fi]]              # (T, |p|)
        plvs.append(np.abs(np.exp(1j * d).mean(axis=0)))   # (|p|,)
    return float(np.mean(np.concatenate(plvs)))


def within_subject_perm(ep, col, n_perm=20000, seed=42):
    rng = np.random.default_rng(seed)
    ep = ep.dropna(subset=[col]).copy()
    ep["c"] = ep.groupby("subject")[col].transform(lambda x: x - x.mean())
    lab = ep["label"].to_numpy(); c = ep["c"].to_numpy(); subj = ep["subject"].to_numpy()
    diff = lambda L: c[L == 1].mean() - c[L == 0].mean()
    obs = diff(lab)
    h, l = c[lab == 1], c[lab == 0]
    psd = np.sqrt(((len(h) - 1) * h.var(ddof=1) + (len(l) - 1) * l.var(ddof=1))
                  / (len(h) + len(l) - 2))
    d = float(obs / psd) if psd > 0 else np.nan
    subj_idx = {s: np.where(subj == s)[0] for s in np.unique(subj)}
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = lab.copy()
        for s, ix in subj_idx.items():
            perm[ix] = rng.permutation(lab[ix])
        null[b] = diff(perm)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)   # two-sided
    return {"observed_diff": float(obs), "cohens_d": d,
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "p_value_two_sided": float(p), "n_epochs": int(len(ep)), "n_perm": n_perm}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    epoch_rows, subj_rows = [], []
    for sid in _subjects():
        eeg, lab = _load_subject(sid)
        if eeg is None or len(np.unique(lab)) < 2:
            continue
        theta = np.array([_fronto_parietal_plv(eeg[i], "theta") for i in range(len(eeg))])
        alpha = np.array([_fronto_parietal_plv(eeg[i], "alpha") for i in range(len(eeg))])
        hi, lo = lab == 1, lab == 0
        for i in range(len(eeg)):
            epoch_rows.append({"subject": sid, "label": int(lab[i]),
                               "fp_plv_theta": theta[i], "fp_plv_alpha": alpha[i]})
        subj_rows.append({
            "subject": sid, "n_high": int(hi.sum()), "n_low": int(lo.sum()),
            "delta_fp_plv_theta": float(np.nanmean(theta[hi]) - np.nanmean(theta[lo])),
            "delta_fp_plv_alpha": float(np.nanmean(alpha[hi]) - np.nanmean(alpha[lo])),
        })

    ep = pd.DataFrame(epoch_rows)
    sdf = pd.DataFrame(subj_rows)
    ep.to_csv(OUT_DIR / "connectivity_concordance_per_epoch.csv", index=False)
    sdf.to_csv(OUT_DIR / "connectivity_concordance_per_subject.csv", index=False)

    ws_theta = within_subject_perm(ep, "fp_plv_theta")
    ws_alpha = within_subject_perm(ep, "fp_plv_alpha")
    w_theta = wilcoxon(sdf["delta_fp_plv_theta"].to_numpy())
    w_alpha = wilcoxon(sdf["delta_fp_plv_alpha"].to_numpy())

    print(f"\n{'='*64}\n  CONSTRUCT VALIDITY (PRIMARY) — fronto-parietal PLV\n{'='*64}")
    print(f"  Subjects: {len(sdf)}   Epochs: {len(ep)}")
    for name, ws, wil, sd_col in [
        ("THETA fronto-parietal PLV", ws_theta, w_theta, "delta_fp_plv_theta"),
        ("ALPHA fronto-parietal PLV", ws_alpha, w_alpha, "delta_fp_plv_alpha")]:
        npos = int((sdf[sd_col] > 0).sum())
        print(f"\n  {name}  (HIGH−LOW)")
        print(f"    within-subject perm: Δ(centred)={ws['observed_diff']:+.5f}  "
              f"d={ws['cohens_d']:+.3f}  p(2-sided)={ws['p_value_two_sided']:.4g}")
        print(f"    per-subject Wilcoxon: p={wil.pvalue:.4g}  ({npos}/{len(sdf)} subj Δ>0)")

    summary = {
        "n_subjects": len(sdf), "n_epochs": len(ep),
        "frontal": FRONTAL, "posterior": POSTERIOR, "bands": BANDS,
        "theta": {"within_subject_perm": ws_theta, "wilcoxon_p": float(w_theta.pvalue)},
        "alpha": {"within_subject_perm": ws_alpha, "wilcoxon_p": float(w_alpha.pvalue)},
    }
    with open(OUT_DIR / "connectivity_concordance.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Saved: results/validation/connectivity_concordance.json (+ CSVs)\n")


if __name__ == "__main__":
    main()
