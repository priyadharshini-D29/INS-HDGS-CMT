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
* Region of interest:
    frontal-theta  channels: Fz, F3, F4, FC1, FC2
    post-alpha     channels: Pz, Oz, O1, O2, P3, P4, P7, P8
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

ROOT = Path(__file__).resolve().parents[1]
PHASE3 = ROOT.parent / "NEUMA_PHASE3"
OUT_DIR = ROOT / "results/validation"

FS = 300
CANONICAL = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
             "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
             "Fz", "Cz", "Pz", "Oz", "FC1", "FC2", "FC5", "FC6"]
IDX = {c: i for i, c in enumerate(CANONICAL)}

FRONTAL_THETA_CH = ["Fz", "F3", "F4", "FC1", "FC2"]
POST_ALPHA_CH = ["Pz", "Oz", "O1", "O2", "P3", "P4", "P7", "P8"]

BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0)}
TOTAL_BAND = (2.0, 45.0)

# epoch-file / label-dir candidates, highest priority first
CANDIDATES = [
    ("epochs/eeg_epochs_phase3d.npy", "engagement_phase3d/engagement_labels.npy"),
    ("epochs/eeg_epochs_engagement.npy", "engagement/engagement_labels.npy"),
]


def _import_harmonizer():
    """Import channel_harmonizer.py directly, bypassing data/__init__ (which
    imports torch-heavy modules)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_chan_harm", ROOT / "data" / "channel_harmonizer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.harmonize_eeg_channels, mod.load_channel_names


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
            if eeg.shape[-1] != len(CANONICAL):      # harmonize subjects w/ removed channels (S01)
                harmonize_eeg_channels, load_channel_names = _import_harmonizer()
                names = load_channel_names(base / "epochs")
                eeg = np.stack([harmonize_eeg_channels(e, sid, channel_names=names, verbose=False)
                                for e in eeg]).astype(np.float64)
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
    Returns observed diff, Cohen's d, and one-sided p-value.
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
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(null <= obs) + 1) / (n_perm + 1)
    return {"observed_diff": float(obs), "cohens_d": d,
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "p_value": float(p), "n_epochs": int(len(ep)), "n_perm": n_perm}


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
        for i in range(len(eeg)):
            epoch_rows.append({"subject": sid, "label": int(lab[i]),
                               "frontal_theta": fth[i], "post_alpha": pal[i]})
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

    n = len(df)
    n_theta_pos = int((d_theta > 0).sum())
    n_alpha_neg = int((d_alpha < 0).sum())

    print(f"\n{'='*64}\n  CONSTRUCT VALIDITY — EEG ↔ engagement concordance\n{'='*64}")
    print(f"  Subjects with both classes: {n}")
    print(f"\n  Frontal-midline THETA (Fz,F3,F4,FC1,FC2), HIGH−LOW relative power")
    print(f"    mean Δ = {d_theta.mean():+.5f}   median Δ = {np.median(d_theta):+.5f}")
    print(f"    subjects with Δ>0 (predicted): {n_theta_pos}/{n}")
    print(f"    Wilcoxon (H1: HIGH>LOW)  W={w_theta.statistic:.1f}  p={w_theta.pvalue:.4g}")
    print(f"\n  Posterior ALPHA (Pz,Oz,O1,O2,P3,P4,P7,P8), HIGH−LOW relative power")
    print(f"    mean Δ = {d_alpha.mean():+.5f}   median Δ = {np.median(d_alpha):+.5f}")
    print(f"    subjects with Δ<0 (suppression, predicted): {n_alpha_neg}/{n}")
    print(f"    Wilcoxon (H1: HIGH<LOW)  W={w_alpha.statistic:.1f}  p={w_alpha.pvalue:.4g}")

    # ── Primary: within-subject epoch-level permutation test ──────────────────
    ws_theta = within_subject_perm(ep, "frontal_theta", "greater")
    ws_alpha = within_subject_perm(ep, "post_alpha", "less")
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
        },
        "posterior_alpha": {
            "mean_delta": float(d_alpha.mean()),
            "median_delta": float(np.median(d_alpha)),
            "n_negative": n_alpha_neg,
            "wilcoxon_W": float(w_alpha.statistic),
            "wilcoxon_p_less": float(w_alpha.pvalue),
        },
        "channels": {"frontal_theta": FRONTAL_THETA_CH, "posterior_alpha": POST_ALPHA_CH},
        "bands": BANDS, "fs": FS,
    }
    with open(OUT_DIR / "eeg_concordance.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Saved: results/validation/eeg_concordance.json (+ per-subject CSV)\n")


if __name__ == "__main__":
    main()
