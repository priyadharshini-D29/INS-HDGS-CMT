#!/usr/bin/env python3
"""
=======================================================================
NEUMA PHASE 3 — CROSS-SUBJECT COGNITIVE ENGAGEMENT LABELING
=======================================================================
Standalone post-processor for existing Phase 3 epoch outputs.
Replaces ImagePage_1..6 labels with HIGH_ENGAGEMENT / LOW_ENGAGEMENT
using a scientifically validated, subject-independent pipeline.

Scientific basis (Rayner 1998; Holmqvist et al. 2011; Wedel & Pieters 2008):

    engagement_score =
        w1 * fixation_duration   (+) longer fixations → deeper processing
      + w2 * dwell_time          (+) time on stimulus → sustained attention
      + w3 * roi_density         (+) foveal coverage of content area
      + w4 * revisit_count       (+) re-examination → information seeking
      - w5 * gaze_entropy        (-) low entropy → focused, targeted gaze

All features are min-max normalized globally (cross-subject) before scoring.
Threshold = global median → subject-independent by construction.

Usage
-----
    cd src/data_pipeline/04_segmentation
    python engagement_labeling.py
    python engagement_labeling.py --subjects S01 S02 S03
    python engagement_labeling.py --weights 1 1 1 1 1
    python engagement_labeling.py --dry-run
=======================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ── Global constants ─────────────────────────────────────────────────────────

PHASE3_ROOT      = Path(__file__).resolve().parent
ET_SR            = 120.0
STIMULUS_PATTERN = "ImagePage"

LABEL_NAMES = {0: "LOW_ENGAGEMENT", 1: "HIGH_ENGAGEMENT"}
LABEL_MAP   = {"LOW_ENGAGEMENT": 0, "HIGH_ENGAGEMENT": 1}

DEFAULT_WEIGHTS = {
    "fixation_duration" : +1.0,
    "dwell_time"        : +1.0,
    "roi_density"       : +1.0,
    "revisit_count"     : +1.0,
    "gaze_entropy"      : -1.0,   # negative: diffuse gaze → low engagement
}

MIN_BALANCE_RATIO = 0.40   # warn if minority class < 40 %

# ── ET Feature Extraction ────────────────────────────────────────────────────

def _safe_et(ep_raw) -> np.ndarray:
    """Convert any stored ET epoch format to float (T, C) array."""
    ep = np.asarray(ep_raw, dtype=float)
    if ep.ndim == 1:
        ep = ep.reshape(-1, 1)
    return ep


def compute_gaze_entropy(et_ep, n_bins: int = 8) -> float:
    """
    Shannon entropy of the 2D gaze spatial distribution.
    High entropy = diffuse / exploratory gaze.
    Low entropy  = focused / targeted gaze (higher engagement per-area).
    """
    ep = _safe_et(et_ep)
    x  = ep[:, 0]
    y  = ep[:, 1] if ep.shape[1] > 1 else np.zeros_like(x)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return 0.0
    H, _, _ = np.histogram2d(x[ok], y[ok], bins=n_bins)
    flat = H.ravel()
    flat = flat[flat > 0].astype(np.float64)
    flat /= flat.sum()
    return float(-np.sum(flat * np.log(flat + 1e-12)))


def compute_dwell_time(et_ep) -> float:
    """Total seconds of valid (finite, non-NaN) gaze samples in the epoch."""
    ep = _safe_et(et_ep)
    x  = ep[:, 0]
    return float(np.sum(np.isfinite(x)) / ET_SR)


def compute_roi_density(et_ep, cx_frac: float = 0.60, cy_frac: float = 0.60) -> float:
    """
    Fraction of gaze samples within the central 60 % × 60 % screen region.
    Proxy for attention focused on main content (where semantic ROIs concentrate).
    """
    ep = _safe_et(et_ep)
    x  = ep[:, 0]
    y  = ep[:, 1] if ep.shape[1] > 1 else np.zeros_like(x)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return 0.0
    xv, yv = x[ok].copy(), y[ok].copy()
    for v in (xv, yv):
        r = v.max() - v.min()
        if r > 0:
            v[:] = (v - v.min()) / r
    cx, cy = 0.5, 0.5
    in_roi = (
        (xv >= cx - cx_frac / 2) & (xv <= cx + cx_frac / 2)
        & (yv >= cy - cy_frac / 2) & (yv <= cy + cy_frac / 2)
    )
    return float(in_roi.mean())


def compute_revisit_count(et_ep, grid_size: int = 4) -> int:
    """
    Number of unique spatial grid cells visited during the epoch.
    Proxy for gaze exploration breadth; more unique areas → broader
    information seeking → higher engagement with stimulus content.
    """
    ep = _safe_et(et_ep)
    x  = ep[:, 0]
    y  = ep[:, 1] if ep.shape[1] > 1 else np.zeros_like(x)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return 0
    xv, yv = x[ok], y[ok]
    x_min, x_max = xv.min(), xv.max()
    y_min, y_max = yv.min(), yv.max()
    if x_max == x_min or y_max == y_min:
        return 1
    xq = ((xv - x_min) / (x_max - x_min + 1e-8) * grid_size).astype(int).clip(0, grid_size - 1)
    yq = ((yv - y_min) / (y_max - y_min + 1e-8) * grid_size).astype(int).clip(0, grid_size - 1)
    return int(np.unique(xq * grid_size + yq).shape[0])


def extract_et_features(et_ep, meta_row: dict) -> dict:
    """
    Extract all 5 engagement features for one epoch.

    fixation_duration : from metadata mean_fix_dur (samples → seconds)
    dwell_time        : from metadata total_gaze_sec if > 0, else computed
    roi_density       : computed from ET epoch (fraction in central region)
    revisit_count     : computed from ET epoch (unique grid cells)
    gaze_entropy      : computed from ET epoch (Shannon entropy)
    """
    fix_dur_sec = float(meta_row.get("mean_fix_dur", 0.0)) / ET_SR

    meta_dwell = float(meta_row.get("total_gaze_sec", 0.0))
    dwell      = meta_dwell if meta_dwell > 0.0 else compute_dwell_time(et_ep)

    return {
        "fixation_duration" : fix_dur_sec,
        "dwell_time"        : dwell,
        "roi_density"       : compute_roi_density(et_ep),
        "revisit_count"     : compute_revisit_count(et_ep),
        "gaze_entropy"      : compute_gaze_entropy(et_ep),
    }


# ── Subject Data Loading ─────────────────────────────────────────────────────

def load_subject_data(subject_dir: Path, stimulus_pattern: str) -> dict | None:
    """
    Load ET/EEG epochs and metadata for one subject.
    Filters to stimulus-only epochs (label starts with stimulus_pattern).

    Returns dict with keys: subject_id, et_list, eeg_list, meta_df, indices
    """
    sid        = subject_dir.name
    epochs_dir = subject_dir / "output" / "epochs"
    meta_dir   = subject_dir / "output" / "metadata"

    et_path   = epochs_dir / "et_epochs.npy"
    eeg_path  = epochs_dir / "eeg_epochs.npy"
    meta_path = meta_dir   / "metadata.csv"

    if not et_path.exists() or not eeg_path.exists():
        print(f"  [SKIP] {sid}: missing epoch files")
        return None
    if not meta_path.exists():
        print(f"  [SKIP] {sid}: missing metadata.csv")
        return None

    et_arr  = np.load(et_path,  allow_pickle=True)
    eeg_arr = np.load(eeg_path, allow_pickle=True)
    meta_df = pd.read_csv(meta_path)

    # Align lengths if there is any mismatch
    if len(et_arr) != len(meta_df):
        n = min(len(et_arr), len(eeg_arr), len(meta_df))
        print(f"  [WARN] {sid}: length mismatch — truncating to {n}")
        et_arr  = et_arr[:n]
        eeg_arr = eeg_arr[:n]
        meta_df = meta_df.iloc[:n].reset_index(drop=True)

    # Stimulus filter
    if "label" in meta_df.columns:
        stim_mask = meta_df["label"].astype(str).str.startswith(stimulus_pattern).values
        stim_idx  = np.where(stim_mask)[0]
    else:
        stim_idx = np.arange(len(meta_df))

    if len(stim_idx) == 0:
        stim_idx = np.arange(len(meta_df))
        print(f"  [INFO] {sid}: no '{stimulus_pattern}*' labels — using all {len(stim_idx)} epochs")
    else:
        print(f"  [INFO] {sid}: {len(stim_idx)}/{len(meta_df)} stimulus epochs selected")

    return {
        "subject_id" : sid,
        "et_list"    : [et_arr[i]  for i in stim_idx],
        "eeg_list"   : [eeg_arr[i] for i in stim_idx],
        "meta_df"    : meta_df.iloc[stim_idx].reset_index(drop=True),
        "indices"    : stim_idx,
    }


# ── Global Feature DataFrame ─────────────────────────────────────────────────

def build_feature_dataframe(subject_data: list) -> pd.DataFrame:
    """Build cross-subject feature table for global normalization."""
    records = []
    for sd in subject_data:
        for ep, row in zip(sd["et_list"], sd["meta_df"].to_dict("records")):
            feats = extract_et_features(ep, row)
            records.append({
                "subject_id"   : sd["subject_id"],
                "epoch_idx"    : row.get("event_id", -1),
                "orig_label"   : row.get("label", ""),
                **feats,
            })
    return pd.DataFrame(records)


# ── Engagement Scoring ───────────────────────────────────────────────────────

def compute_engagement_scores(
    features_df : pd.DataFrame,
    weights     : dict | None = None,
) -> np.ndarray:
    """
    Cross-subject min-max normalize then compute weighted engagement scores.

    Normalization is global → subject-independent threshold by construction.
    NaN features are replaced with the column mean before normalization.
    """
    weights   = weights or DEFAULT_WEIGHTS
    feat_cols = list(weights.keys())
    X         = features_df[feat_cols].copy().astype(float)
    X.fillna(X.mean(), inplace=True)

    X_norm = MinMaxScaler().fit_transform(X)
    X_df   = pd.DataFrame(X_norm, columns=feat_cols)

    score = sum(w * X_df[col] for col, w in weights.items())
    return score.values


# ── Saving ───────────────────────────────────────────────────────────────────

def _plot_dist(scores, labels, threshold, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    hi = scores[labels == 1]
    lo = scores[labels == 0]
    ax.hist(lo, bins=15, alpha=0.7, label="LOW_ENGAGEMENT",  color="#4472C4")
    ax.hist(hi, bins=15, alpha=0.7, label="HIGH_ENGAGEMENT", color="#ED7D31")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"threshold = {threshold:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Engagement Score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_subject_outputs(
    subject_dir : Path,
    eeg_list    : list,
    et_list     : list,
    meta_df     : pd.DataFrame,
    labels      : np.ndarray,
    scores      : np.ndarray,
    threshold   : float,
) -> None:
    """Save per-subject engagement outputs and filtered epoch arrays."""
    eng_dir    = subject_dir / "output" / "engagement"
    epochs_dir = subject_dir / "output" / "epochs"
    eng_dir.mkdir(parents=True, exist_ok=True)

    np.save(eng_dir / "engagement_labels.npy", labels.astype(np.int32))
    np.save(eng_dir / "engagement_scores.npy", scores.astype(np.float32))

    # Stimulus-filtered epoch arrays consumed by Phase 8 dataset.py
    np.save(epochs_dir / "eeg_epochs_engagement.npy", np.array(eeg_list, dtype=object))
    np.save(epochs_dir / "et_epochs_engagement.npy",  np.array(et_list,  dtype=object))

    eng_meta = meta_df.copy()
    eng_meta["engagement_score"] = scores
    eng_meta["engagement_label"] = [LABEL_NAMES[int(l)] for l in labels]
    eng_meta["threshold"]        = threshold
    eng_meta.to_csv(eng_dir / "engagement_metadata.csv", index=False)

    _plot_dist(
        scores, labels, threshold,
        title     = f"Engagement Distribution — {subject_dir.name}",
        save_path = eng_dir / "engagement_distribution.png",
    )


def save_global_outputs(
    global_dir   : Path,
    features_df  : pd.DataFrame,
    all_scores   : np.ndarray,
    all_labels   : np.ndarray,
    threshold    : float,
    subject_data : list,
) -> None:
    """Save global cross-subject engagement outputs."""
    global_dir.mkdir(parents=True, exist_ok=True)

    np.save(global_dir / "engagement_labels.npy", all_labels.astype(np.int32))
    np.save(global_dir / "engagement_scores.npy", all_scores.astype(np.float32))

    meta = features_df.copy()
    meta["engagement_score"] = all_scores
    meta["engagement_label"] = [LABEL_NAMES[int(l)] for l in all_labels]
    meta["threshold"]        = threshold
    meta.to_csv(global_dir / "engagement_metadata.csv", index=False)

    # Per-subject summary
    records, offset = [], 0
    for sd in subject_data:
        n       = len(sd["et_list"])
        sl      = all_labels[offset:offset + n]
        ss      = all_scores[offset:offset + n]
        n_high  = int((sl == 1).sum())
        n_low   = int((sl == 0).sum())
        records.append({
            "subject_id"   : sd["subject_id"],
            "n_epochs"     : n,
            "n_high"       : n_high,
            "n_low"        : n_low,
            "balance_ratio": round(min(n_high, n_low) / max(n_high, n_low, 1), 3),
            "score_mean"   : round(float(ss.mean()), 4),
            "score_std"    : round(float(ss.std()),  4),
            "score_min"    : round(float(ss.min()),  4),
            "score_max"    : round(float(ss.max()),  4),
        })
        offset += n
    pd.DataFrame(records).to_csv(global_dir / "subject_engagement_summary.csv", index=False)

    # Two-panel global plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    hi = all_scores[all_labels == 1]
    lo = all_scores[all_labels == 0]
    ax.hist(lo, bins=25, alpha=0.7, label="LOW_ENGAGEMENT",  color="#4472C4")
    ax.hist(hi, bins=25, alpha=0.7, label="HIGH_ENGAGEMENT", color="#ED7D31")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"Global median = {threshold:.3f}")
    ax.set_title("Global Engagement Score Distribution\n(cross-subject, all stimuli)")
    ax.set_xlabel("Engagement Score")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1]
    n_high_g = int((all_labels == 1).sum())
    n_low_g  = int((all_labels == 0).sum())
    bars = ax.bar(["LOW_ENGAGEMENT", "HIGH_ENGAGEMENT"], [n_low_g, n_high_g],
                  color=["#4472C4", "#ED7D31"])
    for bar, n in zip(bars, [n_low_g, n_high_g]):
        pct = 100 * n / len(all_labels)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"n={n}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_title("Class Balance")
    ax.set_ylabel("Count")

    fig.suptitle("NEUMA Phase 3 — Cognitive Engagement Labels", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(global_dir / "engagement_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {global_dir / 'engagement_distribution.png'}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="NEUMA Phase 3 — Cross-Subject Cognitive Engagement Labeling"
    )
    p.add_argument(
        "--subjects", nargs="+", default=None,
        help="Subject IDs to process (e.g. S01 S02). Default: auto-discover all S*/.",
    )
    p.add_argument(
        "--stimulus-pattern", default=STIMULUS_PATTERN,
        help=f"Label prefix that identifies stimulus epochs (default: {STIMULUS_PATTERN})",
    )
    p.add_argument(
        "--weights", nargs=5, type=float, default=None,
        metavar=("w_fix", "w_dwell", "w_roi", "w_revisit", "w_entropy"),
        help="Feature weights [fix_duration dwell roi_density revisit entropy]. "
             "Entropy sign is always negated. Default: 1 1 1 1 1",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print results without saving any files",
    )
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    print()
    print("=" * 64)
    print("  NEUMA PHASE 3 — COGNITIVE ENGAGEMENT LABELING")
    print("=" * 64)

    # Discover subjects
    if args.subjects:
        subject_dirs = [PHASE3_ROOT / s for s in args.subjects]
        missing = [d for d in subject_dirs if not d.exists()]
        if missing:
            print(f"[ERROR] Subject directories not found: {[d.name for d in missing]}")
            sys.exit(1)
    else:
        subject_dirs = sorted(
            d for d in PHASE3_ROOT.iterdir()
            if d.is_dir()
            and d.name.startswith("S")
            and (d / "output" / "epochs" / "et_epochs.npy").exists()
        )

    if not subject_dirs:
        print("[ERROR] No subject directories found. Run Phase 3 first.")
        sys.exit(1)

    print(f"\nDiscovered {len(subject_dirs)} subject(s):")
    for d in subject_dirs:
        print(f"  {d.name}")

    # Load per-subject data
    print(f"\n{'-' * 64}")
    print("  LOADING EPOCH DATA")
    print(f"{'-' * 64}")

    subject_data = [
        load_subject_data(d, args.stimulus_pattern) for d in subject_dirs
    ]
    subject_data = [sd for sd in subject_data if sd is not None]

    if not subject_data:
        print("[ERROR] No valid subject data loaded.")
        sys.exit(1)

    total_epochs = sum(len(sd["et_list"]) for sd in subject_data)
    print(f"\nTotal stimulus epochs: {total_epochs}  ({len(subject_data)} subjects)")

    # Build feature table
    print(f"\n{'-' * 64}")
    print("  EXTRACTING ET ENGAGEMENT FEATURES")
    print(f"{'-' * 64}")
    features_df = build_feature_dataframe(subject_data)
    print(f"\nFeature matrix: {features_df.shape}")
    print(features_df[list(DEFAULT_WEIGHTS.keys())].describe().round(4).to_string())

    # Weights
    if args.weights:
        w = args.weights
        weights = {
            "fixation_duration" : +w[0],
            "dwell_time"        : +w[1],
            "roi_density"       : +w[2],
            "revisit_count"     : +w[3],
            "gaze_entropy"      : -abs(w[4]),   # entropy is always subtracted
        }
    else:
        weights = DEFAULT_WEIGHTS.copy()

    print(f"\nEngagement score weights:")
    for k, v in weights.items():
        print(f"  {k:25s} {v:+.1f}")

    # Global scoring
    print(f"\n{'-' * 64}")
    print("  COMPUTING CROSS-SUBJECT ENGAGEMENT SCORES")
    print(f"{'-' * 64}")

    all_scores = compute_engagement_scores(features_df, weights)
    threshold  = float(np.median(all_scores))
    all_labels = (all_scores >= threshold).astype(np.int32)

    n_high  = int((all_labels == 1).sum())
    n_low   = int((all_labels == 0).sum())
    balance = min(n_high, n_low) / max(n_high, n_low, 1)

    print(f"\nGlobal median threshold : {threshold:.4f}")
    print(f"Score range             : [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"\nClass distribution:")
    print(f"  HIGH_ENGAGEMENT : {n_high:5d}  ({100*n_high/len(all_labels):.1f}%)")
    print(f"  LOW_ENGAGEMENT  : {n_low:5d}  ({100*n_low/len(all_labels):.1f}%)")
    print(f"  Balance ratio   : {balance:.3f}", end="  ")
    print("OK" if balance >= MIN_BALANCE_RATIO else "[WARN] imbalanced — consider adjusting weights")

    if args.dry_run:
        print("\n[DRY-RUN] No files saved.")
        return

    # Per-subject saves
    print(f"\n{'-' * 64}")
    print("  SAVING PER-SUBJECT OUTPUTS")
    print(f"{'-' * 64}")

    offset = 0
    for sd in subject_data:
        n       = len(sd["et_list"])
        s_sc    = all_scores[offset:offset + n]
        s_lb    = all_labels[offset:offset + n]
        n_h     = int((s_lb == 1).sum())
        n_l     = int((s_lb == 0).sum())
        save_subject_outputs(
            subject_dir = PHASE3_ROOT / sd["subject_id"],
            eeg_list    = sd["eeg_list"],
            et_list     = sd["et_list"],
            meta_df     = sd["meta_df"],
            labels      = s_lb,
            scores      = s_sc,
            threshold   = threshold,
        )
        print(f"  {sd['subject_id']}: HIGH={n_h}  LOW={n_l}  balance={min(n_h,n_l)/max(n_h,n_l,1):.2f}")
        offset += n

    # Global saves
    print(f"\n{'-' * 64}")
    print("  SAVING GLOBAL OUTPUTS")
    print(f"{'-' * 64}")

    global_dir = PHASE3_ROOT / "output" / "engagement"
    save_global_outputs(global_dir, features_df, all_scores, all_labels, threshold, subject_data)

    # Final report
    print()
    print("=" * 64)
    print("  ENGAGEMENT LABELING COMPLETE")
    print("=" * 64)
    print(f"\n  Subjects processed    : {len(subject_data)}")
    print(f"  Total stimulus epochs : {total_epochs}")
    print(f"  Global threshold      : {threshold:.4f}  (cross-subject median)")
    print(f"  HIGH_ENGAGEMENT       : {n_high}  ({100*n_high/len(all_labels):.1f}%)")
    print(f"  LOW_ENGAGEMENT        : {n_low}  ({100*n_low/len(all_labels):.1f}%)")
    print(f"\n  Global output dir     : {global_dir}")
    print( "    engagement_labels.npy")
    print( "    engagement_scores.npy")
    print( "    engagement_metadata.csv")
    print( "    engagement_distribution.png")
    print( "    subject_engagement_summary.csv")
    print( "\n  Per-subject (each S*/output/):")
    print( "    engagement/engagement_labels.npy")
    print( "    engagement/engagement_scores.npy")
    print( "    engagement/engagement_metadata.csv")
    print( "    engagement/engagement_distribution.png")
    print( "    epochs/eeg_epochs_engagement.npy")
    print( "    epochs/et_epochs_engagement.npy")
    print( "\n  READY FOR:")
    print( "    LOSOCV  /  INS-HDGS-CMT  /  GNN  /  SNN  /  Explainability")


if __name__ == "__main__":
    main()
