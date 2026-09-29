"""
================================================================
INS-HDGS-CMT — Construct Validity: EEG ↔ engagement concordance
================================================================
WHY THIS EXISTS
---------------
The HIGH/LOW engagement labels are derived ENTIRELY from eye-tracking
(fixation ratio, dwell, ROI density, pupil, gaze entropy → composite →
per-subject median split; see labeling/engagement_labeler.py). Eye-tracking
is ALSO a model input. A reviewer can therefore object that the task is
circular (ET-label predicted from ET-input) and that the EEG branch may be
decorative.

This script tests the construct from the INDEPENDENT modality: if the
ET-derived label captures a genuine engagement state, then the EEG — which
plays no part in defining the label — should show the canonical
electrophysiological signatures of engagement/attention:

  * Parietal–occipital ALPHA (8–13 Hz) SUPPRESSION in HIGH vs LOW
    (alpha desynchronisation indexes cortical engagement).
  * Frontal-midline THETA (4–8 Hz) INCREASE in HIGH vs LOW
    (frontal theta indexes attentional control / sustained engagement).

METHOD (leakage-free, within-subject)
-------------------------------------
* Load raw, un-normalised epochs (eeg_epochs_phase3d.npy, 1500×24 @300Hz) and
  the matching phase3d HIGH/LOW labels per subject.
* Welch PSD per epoch per channel → absolute band power; RELATIVE band power
  = band / total(2–45 Hz) (invariant to per-channel scaling, so robust to the
  pipeline's per-subject z-scoring).
* Region of interest (2026-09-29 correction: the recordings are DSI-24; the
  earlier sets named FC1/FC2/Oz/P7/P8, which do not exist in this headset, and
  the channel index map did not match the saved epoch order — see CANONICAL):
    frontal-theta  channels: Fz, F3, F4          (variant: + F7, F8)
    post-alpha     channels: Pz, O1, O2, P3, P4  (variant: + T5, T6)
* Per subject, contrast HIGH vs LOW (mean of HIGH epochs − mean of LOW).
  Aggregate the per-subject deltas across subjects and test with a two-sided
  Wilcoxon signed-rank test (subject = unit; no pooling of epochs across
  subjects, so per-subject baseline differences cannot drive the effect).

Directional hypotheses:
    post-alpha  : HIGH < LOW   (suppression  → negative delta)
    frontal-theta: HIGH > LOW  (increase      → positive delta)

Usage
-----
    python analysis/eeg_concordance.py
================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import wilcoxon

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

REPO = Path(__file__).resolve().parents[2]
_P3_CANDIDATES = [REPO.parent / "NEUMA_PHASE3",                       # box layout
                  REPO / "src" / "data_pipeline" / "04_segmentation"]  # repo layout
PHASE3 = next((p for p in _P3_CANDIDATES if p.exists()), _P3_CANDIDATES[0])
BAD_CH_JSON = REPO / "src" / "data_pipeline" / "03_preprocessing" / "metadata" / "bad_channels.json"
OUT_DIR = REPO / "results" / "validation"

FS = 300
# TRUE saved epoch channel order — the DSI-24 hardware montage of the raw
# arrays (src/model/data/channel_harmonizer.RAW_MONTAGE_CHANNELS). Until
# 2026-09-29 this list was a generic 10-20 ordering (Fp1..FC6) that matched
# neither the saved order nor the headset's electrode names, so the region
# sets were read from the wrong columns (e.g. "Fz" read X2, "FC1" read A2).
CANONICAL = ["P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz",
             "Pz", "Fp1", "Fp2", "T3", "T5", "O1", "O2", "X3",
             "X2", "F7", "F8", "X1", "A2", "T6", "T4", "TRG"]
IDX = {c: i for i, c in enumerate(CANONICAL)}

# Primary sets = the intended regions restricted to electrodes that exist in
# this montage; the variants add the neighbouring lateral / posterior-temporal
# sites (T5/T6 are the older names of P7/P8).
FRONTAL_THETA_CH = ["Fz", "F3", "F4"]
FRONTAL_THETA_CH_VAR = ["Fz", "F3", "F4", "F7", "F8"]
POST_ALPHA_CH = ["Pz", "O1", "O2", "P3", "P4"]
POST_ALPHA_CH_VAR = ["Pz", "O1", "O2", "P3", "P4", "T5", "T6"]

BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0)}
TOTAL_BAND = (2.0, 45.0)

# epoch-file / label-dir candidates, highest priority first
CANDIDATES = [
    ("epochs/eeg_epochs_phase3d.npy", "engagement_phase3d/engagement_labels.npy"),
    ("epochs/eeg_epochs_engagement.npy", "engagement/engagement_labels.npy"),
]


def _to_raw24(eeg, sid):
    """(N, T, C_subj) -> (N, T, 24) raw-montage layout: zero-fill the channels
    removed in preprocessing (index-based, from bad_channels.json). The
    model-side harmonizer now returns 19 channels and cannot be used here."""
    n, t, c = eeg.shape
    with BAD_CH_JSON.open() as fh:
        bad = list(json.load(fh).get(sid, []))
    good = [i for i in range(24) if i not in bad]
    if len(good) != c:
        raise RuntimeError(f"{sid}: {c} channels but bad_channels.json implies {len(good)}")
    out = np.zeros((n, t, 24), dtype=np.float64)
    out[:, :, good] = eeg
    return out


def _subjects():
    return sorted(p.name for p in PHASE3.iterdir()
                  if p.is_dir() and p.name.startswith("S"))


def _load_subject(sid):
    base = PHASE3 / sid / "output"
    for eeg_rel, lab_rel in CANDIDATES:
        eeg_p, lab_p = base / eeg_rel, base / lab_rel
        if eeg_p.exists() and lab_p.exists():
            eeg = np.load(eeg_p, allow_pickle=True)
            lab = np.load(lab_p, allow_pickle=True).astype(int)
            eeg = np.stack([np.asarray(e, dtype=np.float64) for e in eeg])  # (N,1500,C)
            if eeg.shape[-1] != len(CANONICAL):      # rebuild raw-24 layout (S01)
                eeg = _to_raw24(eeg, sid)
            if eeg.shape[0] != lab.shape[0]:
                n = min(eeg.shape[0], lab.shape[0])
                eeg, lab = eeg[:n], lab[:n]
            return eeg, lab
    return None, None


def _rel_band_power(epoch):
    """epoch: (T, C). Return dict band -> (C,) relative band power."""
    # Welch over time axis; nperseg ~1s
    freqs, psd = welch(epoch, fs=FS, axis=0, nperseg=min(FS, epoch.shape[0]))
    tot_mask = (freqs >= TOTAL_BAND[0]) & (freqs < TOTAL_BAND[1])
    total = _trapz(psd[tot_mask], freqs[tot_mask], axis=0) + 1e-20  # (C,)
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        bp = _trapz(psd[m], freqs[m], axis=0)  # (C,)
        out[b] = bp / total
    return out


def _region_power(epoch, channels, band):
    rbp = _rel_band_power(epoch)[band]
    # Drop channels that are dead (all-zero → removed & zero-filled at harmonization)
    cols = [IDX[c] for c in channels if c in IDX and np.any(epoch[:, IDX[c]] != 0)]
    if not cols:
        return np.nan
    return float(np.mean(rbp[cols]))


def within_subject_perm(ep, col, direction, n_perm=20000, seed=42):
    """Within-subject epoch-level permutation test.

    Center each subject's band power (remove between-subject offset), pool all
    epochs, and use the HIGH−LOW difference of centered power as the statistic.
    Null distribution: permute HIGH/LOW labels WITHIN each subject (respects the
    subject structure and per-subject class balance). Far more powerful than the
    per-subject-mean Wilcoxon when epochs/subject is small (~6–16 here).

    direction: 'greater' (HIGH>LOW, theta) or 'less' (HIGH<LOW, alpha).
    Returns observed diff, Cohen's d, and both the one-sided p (pre-specified
    canonical direction) and the two-sided p (|null| >= |obs|), from the same
    null draws.
    """
    rng = np.random.default_rng(seed)
    ep = ep.dropna(subset=[col]).copy()
    # within-subject centering
    ep["c"] = ep.groupby("subject")[col].transform(lambda x: x - x.mean())
    lab = ep["label"].to_numpy()
    c = ep["c"].to_numpy()
    subj = ep["subject"].to_numpy()

    def _diff(labels):
        return c[labels == 1].mean() - c[labels == 0].mean()

    obs = _diff(lab)
    # Cohen's d on centered power (pooled SD)
    h, l = c[lab == 1], c[lab == 0]
    psd = np.sqrt(((len(h) - 1) * h.var(ddof=1) + (len(l) - 1) * l.var(ddof=1))
                  / (len(h) + len(l) - 2))
    d = float(obs / psd) if psd > 0 else np.nan

    # within-subject label permutation
    subj_idx = {s: np.where(subj == s)[0] for s in np.unique(subj)}
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = lab.copy()
        for s, ix in subj_idx.items():
            perm[ix] = rng.permutation(lab[ix])
        null[b] = _diff(perm)

    if direction == "greater":
        p_one = (np.sum(null >= obs) + 1) / (n_perm + 1)
    else:
        p_one = (np.sum(null <= obs) + 1) / (n_perm + 1)
    p_two = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"observed_diff": float(obs), "cohens_d": d,
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "p_value": float(p_one), "p_value_one_sided": float(p_one),
            "p_value_two_sided": float(p_two),
            "n_epochs": int(len(ep)), "n_perm": n_perm}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    epoch_rows = []
    for sid in _subjects():
        eeg, lab = _load_subject(sid)
        if eeg is None or len(np.unique(lab)) < 2:
            continue
        hi, lo = lab == 1, lab == 0
        # per-epoch region powers
        fth = np.array([_region_power(eeg[i], FRONTAL_THETA_CH, "theta")
                        for i in range(len(eeg))])
        pal = np.array([_region_power(eeg[i], POST_ALPHA_CH, "alpha")
                        for i in range(len(eeg))])
        fth_v = np.array([_region_power(eeg[i], FRONTAL_THETA_CH_VAR, "theta")
                          for i in range(len(eeg))])
        pal_v = np.array([_region_power(eeg[i], POST_ALPHA_CH_VAR, "alpha")
                          for i in range(len(eeg))])
        for i in range(len(eeg)):
            epoch_rows.append({"subject": sid, "label": int(lab[i]),
                               "frontal_theta": fth[i], "post_alpha": pal[i],
                               "frontal_theta_var": fth_v[i], "post_alpha_var": pal_v[i]})
        rows.append({
            "subject": sid,
            "n_high": int(hi.sum()), "n_low": int(lo.sum()),
            "frontal_theta_HIGH": float(fth[hi].mean()),
            "frontal_theta_LOW": float(fth[lo].mean()),
            "delta_frontal_theta": float(fth[hi].mean() - fth[lo].mean()),
            "post_alpha_HIGH": float(pal[hi].mean()),
            "post_alpha_LOW": float(pal[lo].mean()),
            "delta_post_alpha": float(pal[hi].mean() - pal[lo].mean()),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "eeg_concordance_per_subject.csv", index=False)

    # Persist per-epoch table for the within-subject permutation test
    ep = pd.DataFrame(epoch_rows)
    ep.to_csv(OUT_DIR / "eeg_concordance_per_epoch.csv", index=False)

    d_theta = df["delta_frontal_theta"].to_numpy()
    d_alpha = df["delta_post_alpha"].to_numpy()

    # Wilcoxon signed-rank across subjects (subject = unit)
    w_theta = wilcoxon(d_theta, alternative="greater")    # H1: HIGH theta > LOW
    w_alpha = wilcoxon(d_alpha, alternative="less")        # H1: HIGH alpha < LOW
    w_theta2 = wilcoxon(d_theta)                           # two-sided
    w_alpha2 = wilcoxon(d_alpha)                           # two-sided

    n = len(df)
    n_theta_pos = int((d_theta > 0).sum())
    n_alpha_neg = int((d_alpha < 0).sum())

    print(f"\n{'='*64}\n  CONSTRUCT VALIDITY — EEG ↔ engagement concordance\n{'='*64}")
    print(f"  Subjects with both classes: {n}")
    print(f"\n  Frontal-midline THETA (Fz,F3,F4), HIGH−LOW relative power")
    print(f"    mean Δ = {d_theta.mean():+.5f}   median Δ = {np.median(d_theta):+.5f}")
    print(f"    subjects with Δ>0 (predicted): {n_theta_pos}/{n}")
    print(f"    Wilcoxon (H1: HIGH>LOW)  W={w_theta.statistic:.1f}  p={w_theta.pvalue:.4g}")
    print(f"\n  Posterior ALPHA (Pz,O1,O2,P3,P4), HIGH−LOW relative power")
    print(f"    mean Δ = {d_alpha.mean():+.5f}   median Δ = {np.median(d_alpha):+.5f}")
    print(f"    subjects with Δ<0 (suppression, predicted): {n_alpha_neg}/{n}")
    print(f"    Wilcoxon (H1: HIGH<LOW)  W={w_alpha.statistic:.1f}  p={w_alpha.pvalue:.4g}")

    # ── Primary: within-subject epoch-level permutation test ──────────────────
    ws_theta = within_subject_perm(ep, "frontal_theta", "greater")
    ws_alpha = within_subject_perm(ep, "post_alpha", "less")
    ws_theta_v = within_subject_perm(ep, "frontal_theta_var", "greater")
    ws_alpha_v = within_subject_perm(ep, "post_alpha_var", "less")
    print("\n  ── Within-subject epoch-level permutation (PRIMARY) ──")
    print(f"  Frontal theta  HIGH−LOW(centered)={ws_theta['observed_diff']:+.5f}  "
          f"d={ws_theta['cohens_d']:+.3f}  p={ws_theta['p_value']:.4g}  "
          f"(n={ws_theta['n_epochs']} epochs)")
    print(f"  Posterior alpha HIGH−LOW(centered)={ws_alpha['observed_diff']:+.5f}  "
          f"d={ws_alpha['cohens_d']:+.3f}  p={ws_alpha['p_value']:.4g}  "
          f"(n={ws_alpha['n_epochs']} epochs)")

    summary = {
        "n_subjects": n,
        "within_subject_permutation": {
            "frontal_theta_HIGH_gt_LOW": ws_theta,
            "posterior_alpha_HIGH_lt_LOW": ws_alpha,
        },
        "frontal_theta": {
            "mean_delta": float(d_theta.mean()),
            "median_delta": float(np.median(d_theta)),
            "n_positive": n_theta_pos,
            "wilcoxon_W": float(w_theta.statistic),
            "wilcoxon_p_greater": float(w_theta.pvalue),
            "wilcoxon_p_two_sided": float(w_theta2.pvalue),
        },
        "posterior_alpha": {
            "mean_delta": float(d_alpha.mean()),
            "median_delta": float(np.median(d_alpha)),
            "n_negative": n_alpha_neg,
            "wilcoxon_W": float(w_alpha.statistic),
            "wilcoxon_p_less": float(w_alpha.pvalue),
            "wilcoxon_p_two_sided": float(w_alpha2.pvalue),
        },
        "variant": {
            "frontal_theta_HIGH_gt_LOW": ws_theta_v,
            "posterior_alpha_HIGH_lt_LOW": ws_alpha_v,
            "channels": {"frontal_theta": FRONTAL_THETA_CH_VAR,
                         "posterior_alpha": POST_ALPHA_CH_VAR},
        },
        "channels": {"frontal_theta": FRONTAL_THETA_CH, "posterior_alpha": POST_ALPHA_CH},
        "bands": BANDS, "fs": FS,
        "montage": CANONICAL,
    }
    with open(OUT_DIR / "eeg_concordance.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Saved: results/validation/eeg_concordance.json (+ per-subject CSV)\n")


if __name__ == "__main__":
    main()
