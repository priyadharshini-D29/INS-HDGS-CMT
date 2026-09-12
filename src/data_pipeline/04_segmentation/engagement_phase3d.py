#!/usr/bin/env python3
"""
=======================================================================
NEUMA PHASE 3D — MULTIMODAL COGNITIVE ENGAGEMENT LABELING
=======================================================================
Extends Phase 3 engagement labeling by fusing EEG band-power features
with Eye-Tracking features for a richer, neuroscientifically grounded
engagement score.

EEG features (frontal channels, Welch PSD):
    theta (4–8 Hz)        : cognitive load, memory encoding  (+)
    alpha (8–13 Hz)       : idling / relaxed attention       (−)
    beta  (13–30 Hz)      : active attention                 (+)
    theta/beta ratio      : effort balance                   (+)
    frontal asymmetry     : approach motivation              (+)

ET features:
    fixation_duration     : mean absolute x-gaze position    (+)
    dwell_time            : mean absolute y-gaze position    (+)
    roi_attention         : gaze spatial spread (std)        (+)
    revisit_count         : gaze stability proxy             (+)
    gaze_entropy          : spatial entropy of x-gaze        (−)

Multimodal engagement score (all features min-max normalized):
    score = 1.0*fix_dur + 1.0*dwell + 1.0*roi + 1.0*revisit
          + 1.5*theta  + 1.2*beta  - 1.5*alpha
          + 1.0*theta_beta_ratio + 0.5*frontal_asym
          - 1.0*gaze_entropy

Threshold: global median across all stimulus epochs (cross-subject).

Usage
-----
    cd src/data_pipeline/04_segmentation
    python engagement_phase3d.py
    python engagement_phase3d.py --subjects S01 S02
    python engagement_phase3d.py --dry-run
=======================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────

PHASE3_ROOT = Path(__file__).resolve().parent   # …/src/data_pipeline/04_segmentation

EEG_SR           = 300          # Hz — from 04_segmentation/config/settings.py
FRONTAL_CH       = [0, 1, 2, 3] # first 4 channels (Fp1, Fp2, F3, F4 or similar)
STIMULUS_PATTERN = "ImagePage"

LABEL_NAMES = {0: "LOW_ENGAGEMENT", 1: "HIGH_ENGAGEMENT"}

# Engagement score weights (post min-max normalisation)
WEIGHTS = {
    "fixation_duration"  : +1.0,
    "dwell_time"         : +1.0,
    "roi_attention"      : +1.0,
    "revisit_count"      : +1.0,
    "theta"              : +1.5,
    "beta"               : +1.2,
    "alpha"              : -1.5,   # alpha is SUBTRACTED
    "theta_beta_ratio"   : +1.0,
    "frontal_asymmetry"  : +0.5,
    "gaze_entropy"       : -1.0,   # entropy is SUBTRACTED
}

FEATURE_COLS = list(WEIGHTS.keys())

# ── EEG Feature Extraction ────────────────────────────────────────────────────

def _bandpower(x: np.ndarray, fmin: float, fmax: float) -> float:
    """Welch-PSD bandpower between fmin and fmax Hz."""
    nperseg = min(256, len(x))
    f, pxx  = welch(x, fs=EEG_SR, nperseg=nperseg)
    mask    = (f >= fmin) & (f <= fmax)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(pxx[mask], f[mask])) if mask.any() else 0.0


def extract_eeg_features(eeg_epoch) -> dict | None:
    """
    Extract frontal band-power features from one EEG epoch.

    eeg_epoch : (T, C) array — time × channels (may be dtype=object)
    Returns dict or None if epoch is malformed.
    """
    eeg = np.asarray(eeg_epoch, dtype=float)
    if eeg.ndim != 2 or eeg.shape[1] < max(FRONTAL_CH) + 1:
        return None

    frontal      = eeg[:, FRONTAL_CH]          # (T, 4)
    frontal_mean = frontal.mean(axis=1)         # (T,)

    theta = _bandpower(frontal_mean, 4,  8)
    alpha = _bandpower(frontal_mean, 8,  13)
    beta  = _bandpower(frontal_mean, 13, 30)
    tbr   = theta / (beta + 1e-8)
    asym  = float(frontal[:, 0].mean() - frontal[:, -1].mean())

    return {
        "theta"            : theta,
        "alpha"            : alpha,
        "beta"             : beta,
        "theta_beta_ratio" : tbr,
        "frontal_asymmetry": asym,
    }


# ── ET Feature Extraction ─────────────────────────────────────────────────────

def extract_et_features(et_epoch) -> dict | None:
    """
    Extract engagement-relevant ET features from one ET epoch.

    et_epoch : (T, C) array — col0 = x-gaze [0,1], col1 = y-gaze [0,1]
    """
    et = np.asarray(et_epoch, dtype=float)
    if et.ndim != 2 or et.shape[1] < 2:
        return None

    x = et[:, 0]
    y = et[:, 1]

    # Drop NaN / Inf samples (blink gaps, tracker loss)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return None
    x = x[valid]
    y = y[valid]

    fix_dur = float(np.mean(np.abs(x)))
    dwell   = float(np.mean(np.abs(y)))
    roi_att = float(np.std(x))

    dx = np.abs(np.diff(x))
    revisit = int(np.sum(dx < 0.05))

    hist, _ = np.histogram(x, bins=20)
    p       = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(p * np.log2(p + 1e-12)))

    return {
        "fixation_duration" : fix_dur,
        "dwell_time"        : dwell,
        "roi_attention"     : roi_att,
        "revisit_count"     : float(revisit),
        "gaze_entropy"      : entropy,
    }


# ── Subject Data Loading ──────────────────────────────────────────────────────

def load_subject(subject_dir: Path) -> dict | None:
    """Load EEG/ET epochs + metadata for one subject, filtered to stimuli."""
    sid        = subject_dir.name
    ep_dir     = subject_dir / "output" / "epochs"
    meta_path  = subject_dir / "output" / "metadata" / "metadata.csv"

    eeg_path = ep_dir / "eeg_epochs.npy"
    et_path  = ep_dir / "et_epochs.npy"

    if not eeg_path.exists() or not et_path.exists():
        print(f"  [SKIP] {sid}: missing epoch files")
        return None

    eeg_arr = np.load(eeg_path, allow_pickle=True)
    et_arr  = np.load(et_path,  allow_pickle=True)
    n       = min(len(eeg_arr), len(et_arr))

    if meta_path.exists():
        meta_df   = pd.read_csv(meta_path).iloc[:n].reset_index(drop=True)
        stim_mask = meta_df["label"].astype(str).str.startswith(STIMULUS_PATTERN)
        stim_idx  = np.where(stim_mask.values)[0]
        if len(stim_idx) == 0:
            stim_idx = np.arange(n)
            print(f"  [INFO] {sid}: no '{STIMULUS_PATTERN}*' labels — using all {n}")
        else:
            print(f"  [INFO] {sid}: {len(stim_idx)}/{n} stimulus epochs")
        meta_df = meta_df.iloc[stim_idx].reset_index(drop=True)
    else:
        stim_idx = np.arange(n)
        meta_df  = pd.DataFrame({"label": ["ImagePage"] * n})
        print(f"  [INFO] {sid}: no metadata.csv — using all {n} epochs")

    return {
        "subject_id": sid,
        "eeg_list"  : [eeg_arr[i] for i in stim_idx],
        "et_list"   : [et_arr[i]  for i in stim_idx],
        "meta_df"   : meta_df,
        "indices"   : stim_idx,
    }


# ── Feature Extraction ────────────────────────────────────────────────────────

def build_feature_table(subject_data: list) -> pd.DataFrame:
    """Extract multimodal features for every epoch across all subjects."""
    records = []
    for sd in subject_data:
        sid = sd["subject_id"]
        for i, (eeg_ep, et_ep, meta_row) in enumerate(
            zip(sd["eeg_list"], sd["et_list"], sd["meta_df"].to_dict("records"))
        ):
            eeg_feats = extract_eeg_features(eeg_ep)
            et_feats  = extract_et_features(et_ep)
            if eeg_feats is None or et_feats is None:
                continue
            records.append({
                "subject_id" : sid,
                "epoch_idx"  : i,
                "orig_label" : meta_row.get("label", ""),
                **eeg_feats,
                **et_feats,
            })
    return pd.DataFrame(records)


# ── Engagement Scoring ────────────────────────────────────────────────────────

def compute_scores(features_df: pd.DataFrame) -> np.ndarray:
    """
    Min-max normalize all features globally, then compute weighted score.
    Global normalization → subject-independent threshold by construction.
    """
    X = features_df[FEATURE_COLS].copy().astype(float)
    X.fillna(X.mean(), inplace=True)
    X_norm = MinMaxScaler().fit_transform(X)
    X_df   = pd.DataFrame(X_norm, columns=FEATURE_COLS)
    return sum(w * X_df[col] for col, w in WEIGHTS.items()).values


# ── Saving ────────────────────────────────────────────────────────────────────

def _dist_plot(scores, labels, threshold, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[labels == 0], bins=15, alpha=0.7,
            label="LOW_ENGAGEMENT",  color="#4472C4")
    ax.hist(scores[labels == 1], bins=15, alpha=0.7,
            label="HIGH_ENGAGEMENT", color="#ED7D31")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"threshold = {threshold:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Multimodal Engagement Score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_subject(subject_dir, eeg_list, et_list, meta_df,
                 labels, scores, threshold):
    eng_dir = subject_dir / "output" / "engagement_phase3d"
    ep_dir  = subject_dir / "output" / "epochs"
    eng_dir.mkdir(parents=True, exist_ok=True)

    np.save(eng_dir / "engagement_labels.npy", labels.astype(np.int32))
    np.save(eng_dir / "engagement_scores.npy", scores.astype(np.float32))

    # Stimulus-only epoch arrays for Phase 8 dataset.py
    np.save(ep_dir / "eeg_epochs_phase3d.npy", np.array(eeg_list, dtype=object))
    np.save(ep_dir / "et_epochs_phase3d.npy",  np.array(et_list,  dtype=object))

    m = meta_df.copy()
    m["engagement_score"] = scores
    m["engagement_label"] = [LABEL_NAMES[int(l)] for l in labels]
    m["threshold"]        = threshold
    m.to_csv(eng_dir / "engagement_metadata.csv", index=False)

    _dist_plot(scores, labels, threshold,
               title=f"Multimodal Engagement — {subject_dir.name}",
               path=eng_dir / "engagement_distribution.png")


def save_global(global_dir, features_df, all_scores, all_labels,
                threshold, subject_data):
    global_dir.mkdir(parents=True, exist_ok=True)

    np.save(global_dir / "multimodal_engagement_labels.npy",
            all_labels.astype(np.int32))
    np.save(global_dir / "multimodal_engagement_scores.npy",
            all_scores.astype(np.float32))

    meta = features_df.copy()
    meta["engagement_score"] = all_scores
    meta["engagement_label"] = [LABEL_NAMES[int(l)] for l in all_labels]
    meta["threshold"]        = threshold
    meta.to_csv(global_dir / "multimodal_features.csv", index=False)

    # EEG feature means per subject
    features_df.groupby("subject_id")[FEATURE_COLS].mean().to_csv(
        global_dir / "eeg_feature_summary.csv"
    )

    # Subject summary
    records, offset = [], 0
    for sd in subject_data:
        n  = len(sd["eeg_list"])
        sl = all_labels[offset:offset + n]
        ss = all_scores[offset:offset + n]
        n_high, n_low = int((sl == 1).sum()), int((sl == 0).sum())
        records.append({
            "subject_id"   : sd["subject_id"],
            "n_epochs"     : n,
            "n_high"       : n_high,
            "n_low"        : n_low,
            "balance"      : round(min(n_high, n_low) / max(n_high, n_low, 1), 3),
            "score_mean"   : round(float(ss.mean()), 4),
            "score_std"    : round(float(ss.std()),  4),
        })
        offset += n
    pd.DataFrame(records).to_csv(global_dir / "subject_engagement_summary.csv",
                                  index=False)

    # Two-panel global plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(all_scores[all_labels == 0], bins=25, alpha=0.7,
            label="LOW_ENGAGEMENT",  color="#4472C4")
    ax.hist(all_scores[all_labels == 1], bins=25, alpha=0.7,
            label="HIGH_ENGAGEMENT", color="#ED7D31")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"global median = {threshold:.3f}")
    ax.set_title("Multimodal Engagement Score Distribution\n(EEG + ET, cross-subject)")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1]
    nh = int((all_labels == 1).sum())
    nl = int((all_labels == 0).sum())
    bars = ax.bar(["LOW", "HIGH"], [nl, nh], color=["#4472C4", "#ED7D31"])
    for bar, n in zip(bars, [nl, nh]):
        pct = 100 * n / len(all_labels)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"n={n}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_title("Class Balance")
    ax.set_ylabel("Count")

    fig.suptitle("NEUMA Phase 3D — Multimodal Cognitive Engagement",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(global_dir / "engagement_distribution.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {global_dir / 'engagement_distribution.png'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="NEUMA Phase 3D — Multimodal (EEG + ET) Engagement Labeling"
    )
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subject IDs to process (default: auto-discover all S*/)")
    p.add_argument("--frontal-ch", nargs="+", type=int, default=FRONTAL_CH,
                   help=f"EEG channel indices for frontal band powers (default: {FRONTAL_CH})")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute scores and print report without saving files")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse()

    # Allow overriding frontal channels via CLI
    global FRONTAL_CH
    if args.frontal_ch != FRONTAL_CH:
        FRONTAL_CH = args.frontal_ch

    print()
    print("=" * 64)
    print("  NEUMA PHASE 3D — MULTIMODAL ENGAGEMENT LABELING")
    print("  EEG band powers  +  Eye Tracking features")
    print("=" * 64)
    print(f"\n  EEG SR          : {EEG_SR} Hz")
    print(f"  Frontal channels: {FRONTAL_CH}")
    print(f"  Stimulus filter : {STIMULUS_PATTERN}*")

    # Discover subjects (they live directly inside 04_segmentation/)
    if args.subjects:
        subject_dirs = [PHASE3_ROOT / s for s in args.subjects]
    else:
        subject_dirs = sorted(
            d for d in PHASE3_ROOT.iterdir()
            if d.is_dir()
            and d.name.startswith("S")
            and (d / "output" / "epochs" / "eeg_epochs.npy").exists()
        )

    if not subject_dirs:
        print("[ERROR] No subject directories found. Run Phase 3 first.")
        sys.exit(1)

    print(f"\nDiscovered {len(subject_dirs)} subject(s): "
          f"{[d.name for d in subject_dirs[:5]]}{'…' if len(subject_dirs)>5 else ''}")

    # Load data
    print(f"\n{'-'*64}")
    print("  LOADING EPOCH DATA")
    print(f"{'-'*64}")

    subject_data = [load_subject(d) for d in subject_dirs]
    subject_data = [sd for sd in subject_data if sd is not None]

    if not subject_data:
        print("[ERROR] No valid subject data loaded.")
        sys.exit(1)

    # Extract features
    print(f"\n{'-'*64}")
    print("  EXTRACTING EEG + ET FEATURES")
    print(f"{'-'*64}")

    features_df = build_feature_table(subject_data)
    total_epochs = len(features_df)
    print(f"\nFeature matrix: {features_df.shape}")
    print(features_df[FEATURE_COLS].describe().round(4).to_string())

    # Filter subject_data to epochs that survived feature extraction
    # (some may have been skipped due to shape errors)
    ep_counts = features_df.groupby("subject_id").size().to_dict()
    subject_data = [sd for sd in subject_data if ep_counts.get(sd["subject_id"], 0) > 0]
    # Trim eeg_list/et_list to valid epochs
    for sd in subject_data:
        valid_mask = (features_df["subject_id"] == sd["subject_id"])
        valid_idx  = features_df[valid_mask]["epoch_idx"].values
        sd["eeg_list"] = [sd["eeg_list"][i] for i in valid_idx]
        sd["et_list"]  = [sd["et_list"][i]  for i in valid_idx]
        sd["meta_df"]  = sd["meta_df"].iloc[valid_idx].reset_index(drop=True)

    # Weights summary
    print(f"\nEngagement score weights:")
    for col, w in WEIGHTS.items():
        print(f"  {col:25s}  {w:+.1f}")

    # Compute scores
    print(f"\n{'-'*64}")
    print("  COMPUTING MULTIMODAL ENGAGEMENT SCORES")
    print(f"{'-'*64}")

    all_scores = compute_scores(features_df)
    threshold  = float(np.median(all_scores))
    all_labels = (all_scores >= threshold).astype(np.int32)

    n_high  = int((all_labels == 1).sum())
    n_low   = int((all_labels == 0).sum())
    balance = min(n_high, n_low) / max(n_high, n_low, 1)

    print(f"\n  Global median threshold : {threshold:.4f}")
    print(f"  Score range             : [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"\n  Class distribution:")
    print(f"    HIGH_ENGAGEMENT : {n_high:4d}  ({100*n_high/total_epochs:.1f}%)")
    print(f"    LOW_ENGAGEMENT  : {n_low:4d}  ({100*n_low/total_epochs:.1f}%)")
    print(f"    Balance ratio   : {balance:.3f}",
          "  OK" if balance >= 0.40 else "  [WARN] imbalanced")

    if args.dry_run:
        print("\n[DRY-RUN] No files saved.")
        return

    # Save per-subject
    print(f"\n{'-'*64}")
    print("  SAVING PER-SUBJECT OUTPUTS")
    print(f"{'-'*64}")

    offset = 0
    for sd in subject_data:
        n      = len(sd["eeg_list"])
        s_lb   = all_labels[offset:offset + n]
        s_sc   = all_scores[offset:offset + n]
        save_subject(
            PHASE3_ROOT / sd["subject_id"],
            sd["eeg_list"], sd["et_list"], sd["meta_df"],
            s_lb, s_sc, threshold,
        )
        nh, nl = int((s_lb == 1).sum()), int((s_lb == 0).sum())
        print(f"  {sd['subject_id']}: HIGH={nh}  LOW={nl}  "
              f"balance={min(nh,nl)/max(nh,nl,1):.2f}")
        offset += n

    # Save global
    print(f"\n{'-'*64}")
    print("  SAVING GLOBAL OUTPUTS")
    print(f"{'-'*64}")

    global_dir = PHASE3_ROOT / "output" / "engagement_phase3d"
    save_global(global_dir, features_df, all_scores, all_labels,
                threshold, subject_data)

    # Final report
    print()
    print("=" * 64)
    print("  PHASE 3D MULTIMODAL ENGAGEMENT COMPLETE")
    print("=" * 64)
    print(f"\n  Subjects processed    : {len(subject_data)}")
    print(f"  Total stimulus epochs : {total_epochs}")
    print(f"  Global threshold      : {threshold:.4f} (cross-subject median)")
    print(f"  HIGH_ENGAGEMENT       : {n_high}  ({100*n_high/total_epochs:.1f}%)")
    print(f"  LOW_ENGAGEMENT        : {n_low}  ({100*n_low/total_epochs:.1f}%)")
    print(f"\n  Global output         : {global_dir}")
    print( "    multimodal_engagement_labels.npy")
    print( "    multimodal_engagement_scores.npy")
    print( "    multimodal_features.csv")
    print( "    eeg_feature_summary.csv")
    print( "    subject_engagement_summary.csv")
    print( "    engagement_distribution.png")
    print( "\n  Per-subject (each S*/output/):")
    print( "    engagement_phase3d/engagement_labels.npy")
    print( "    engagement_phase3d/engagement_scores.npy")
    print( "    engagement_phase3d/engagement_metadata.csv")
    print( "    engagement_phase3d/engagement_distribution.png")
    print( "    epochs/eeg_epochs_phase3d.npy")
    print( "    epochs/et_epochs_phase3d.npy")
    print( "\n  READY FOR:")
    print( "    LOSOCV  /  INS-HDGS-CMT  /  GNN  /  SNN  /  Explainability")


if __name__ == "__main__":
    main()
