"""
==============================================================
NEUMA PHASE 3 — EVENT SEGMENTATION + ENGAGEMENT LABELING
EEG + Eye Tracking multimodal epoch pipeline
==============================================================
Timestamps are used for all alignment.
ET is NEVER globally resampled to EEG frequency.
Blink NaN gaps are preserved.

Phase 3 now computes cognitive engagement labels:
  HIGH_ENGAGEMENT (1) — score >= subject-level median
  LOW_ENGAGEMENT  (0) — score <  subject-level median

For cross-subject (subject-independent) thresholding run:
  python engagement_labeling.py   (after all subjects are done)
==============================================================
"""

# ── Parse subject FIRST — must precede all path-sensitive imports ─────────────
import sys
import os
import argparse

_parser = argparse.ArgumentParser(description="NEUMA Phase 3 — Event Segmentation")
_parser.add_argument(
    "--subject",
    type=str,
    default=os.environ.get("NEUMA_SUBJECT", "S01"),
    help="Subject ID (e.g. S01)",
)
_parser.add_argument(
    "--skip-plots",
    action="store_true",
    help="Skip visualisation plots",
)
_args      = _parser.parse_args()
SUBJECT    = _args.subject
SKIP_PLOTS = _args.skip_plots

os.environ["NEUMA_SUBJECT"] = SUBJECT       # set before config.settings is imported
print(f"[INFO] Processing subject: {SUBJECT}")

# ── Now safe to set up path and import project modules ────────────────────────
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Universal UTF-8 / Unicode-safe I/O ────────────────────────────────────────
import sys as _sys, os as _os
_os.environ.setdefault("PYTHONUTF8", "1")
import io as _io
try:
    if isinstance(_sys.stdout, _io.TextIOWrapper):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(_sys.stderr, _io.TextIOWrapper):
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_REPL = {"→":"->","←":"<-","✔":"[OK]","✘":"[X]","─":"-","│":"|","═":"=",
         "┌":"+","┐":"+","└":"+","┘":"+","├":"+","┤":"+","█":"#","░":".","►":">","±":"+/-","∞":"inf"}

def sanitize_text(t):
    t = str(t)
    for o, n in _REPL.items(): t = t.replace(o, n)
    return t

def safe_print(*args, **kwargs):
    sep_c = kwargs.pop("sep", " ")
    text  = sanitize_text(sep_c.join(str(a) for a in args))
    try:    print(text, **kwargs)
    except UnicodeEncodeError: print(text.encode("ascii","ignore").decode(), **kwargs)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.settings import (
    OUTPUT_EPOCHS_DIR,
    OUTPUT_PLOTS_DIR,
    OUTPUT_METADATA_DIR,
    MARKERS_CSV,
    XDF_DIR,
)

from loaders.load_clean_data import load_clean_data
from loaders.load_markers    import load_markers_csv
from loaders.load_roi        import load_roi_boxes_npy

from utils.export_markers import extract_markers_from_xdf
from utils.label_mapper import normalize_event_label

from segmentation.extract_events import extract_events

from segmentation.synchronize_modalities import (
    trim_to_overlap
)

from segmentation.extract_eeg_epochs import (
    extract_all_eeg_epochs
)

from segmentation.extract_et_epochs import (
    extract_all_et_epochs
)

from segmentation.validate_epochs import (
    validate_epoch_pair
)

from roi.fixation_detector import (
    detect_fixations,
    fixation_summary
)

from roi.roi_mapper import (
    map_epoch_gaze_to_roi
)

from visualization.plot_eeg_epoch import (
    plot_eeg_epoch
)

from visualization.plot_et_epoch import (
    plot_et_epoch
)

from visualization.plot_epoch_summary import (
    plot_epoch_summary,
    plot_fixation_density
)


# ============================================================
# ENGAGEMENT FEATURE EXTRACTION HELPERS
# ============================================================

_ET_SR         = 120.0   # Hz — must match config
_ENG_GRID_SIZE = 4       # grid cells per axis for revisit count
_ENG_N_BINS    = 8       # histogram bins for gaze entropy

_LABEL_NAMES = {0: "LOW_ENGAGEMENT", 1: "HIGH_ENGAGEMENT"}

def _safe_et(ep_raw) -> np.ndarray:
    ep = np.asarray(ep_raw, dtype=float)
    return ep.reshape(-1, 1) if ep.ndim == 1 else ep

def _et_gaze_entropy(et_ep, n_bins: int = _ENG_N_BINS) -> float:
    """Shannon entropy of 2D gaze spatial distribution."""
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

def _et_roi_density(et_ep, cx_frac: float = 0.60, cy_frac: float = 0.60) -> float:
    """Fraction of gaze in the central content region (ROI proxy)."""
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
    in_roi = (
        (xv >= 0.5 - cx_frac / 2) & (xv <= 0.5 + cx_frac / 2)
        & (yv >= 0.5 - cy_frac / 2) & (yv <= 0.5 + cy_frac / 2)
    )
    return float(in_roi.mean())

def _et_revisit_count(et_ep, grid_size: int = _ENG_GRID_SIZE) -> int:
    """Number of unique spatial grid cells visited (exploration breadth)."""
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

def _score_engagement(feature_rows: list) -> tuple:
    """
    Compute min-max normalised engagement scores from a list of feature dicts.
    Returns (scores: ndarray, threshold: float, labels: ndarray).
    """
    feat_cols = ["fixation_duration_sec", "dwell_time_sec",
                 "roi_density", "revisit_count", "gaze_entropy"]
    weights   = [+1.0, +1.0, +1.0, +1.0, -1.0]   # entropy is subtracted

    df = pd.DataFrame(feature_rows)[feat_cols].astype(float).fillna(0.0)
    X  = MinMaxScaler().fit_transform(df.values)

    scores = sum(w * X[:, i] for i, w in enumerate(weights))
    thresh = float(np.median(scores))
    labels = (scores >= thresh).astype(np.int32)
    return np.asarray(scores, dtype=np.float32), thresh, labels


# ============================================================
# MAIN
# ============================================================

def main():

    # Create subject-specific output directories
    for d in [OUTPUT_EPOCHS_DIR, OUTPUT_PLOTS_DIR, OUTPUT_METADATA_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print(f" NEUMA PHASE 3 — EVENT SEGMENTATION ({SUBJECT})")
    print("==============================")

    # --------------------------------------------------------
    # 1. LOAD CLEAN DATA
    # --------------------------------------------------------

    eeg, et, eeg_ts, et_ts = load_clean_data(SUBJECT)

    # --------------------------------------------------------
    # 2. TEMPORAL SYNCHRONIZATION
    # --------------------------------------------------------

    eeg, eeg_ts, et, et_ts = trim_to_overlap(
        eeg,
        eeg_ts,
        et,
        et_ts
    )

    # --------------------------------------------------------
    # 3. LOAD MARKERS
    # --------------------------------------------------------

    if not MARKERS_CSV.exists():
        print("\n[INFO] markers.csv not found — extracting from XDF...")
        MARKERS_CSV.parent.mkdir(parents=True, exist_ok=True)
        xdf_path = XDF_DIR / f"{SUBJECT}.xdf"
        extract_markers_from_xdf(xdf_path, out_path=MARKERS_CSV)

    markers_df = load_markers_csv(MARKERS_CSV)

    events = extract_events(markers_df)

    if len(events) == 0:

        print(
            "\n[STOP] No valid events found."
        )

        return

    # --------------------------------------------------------
    # 4. EEG EPOCH EXTRACTION
    # --------------------------------------------------------

    eeg_epochs, eeg_valid_idx = (
        extract_all_eeg_epochs(
            eeg,
            eeg_ts,
            events
        )
    )

    # --------------------------------------------------------
    # 5. ET EPOCH EXTRACTION
    # --------------------------------------------------------

    et_epochs, both_valid_idx = (
        extract_all_et_epochs(
            et,
            et_ts,
            events,
            eeg_valid_idx
        )
    )

    # --------------------------------------------------------
    # 6. VALIDATION + FIXATIONS + ROI
    # --------------------------------------------------------

    final_eeg = []
    final_et = []
    final_labels = []

    metadata = []

    all_fixations = []

    # --------------------------------------------------------
    # OPTIONAL ROI LOADING
    # --------------------------------------------------------

    try:

        roi_boxes = load_roi_boxes_npy()

    except FileNotFoundError:

        roi_boxes = None

        print(
            "\n[INFO] ROI boxes not found "
            "— skipping ROI statistics"
        )

    # --------------------------------------------------------
    # PROCESS EPOCHS
    # --------------------------------------------------------

    for rank, idx in enumerate(both_valid_idx):

        eeg_ep = eeg_epochs[rank]

        et_ep = et_epochs[rank]

        ev = events[idx]

        clean_label = normalize_event_label(ev["label"])

        # ----------------------------------------------------
        # VALIDATE EPOCH
        # ----------------------------------------------------

        if not validate_epoch_pair(
            eeg_ep,
            et_ep,
            event_id=idx
        ):
            continue

        # ----------------------------------------------------
        # FIXATION DETECTION
        # ----------------------------------------------------

        fx = detect_fixations(
            et_ep
        )

        fx_info = fixation_summary(fx)

        all_fixations.append(fx)

        # ----------------------------------------------------
        # ROI DWELL TIME
        # ----------------------------------------------------

        dwell_info = {}

        if roi_boxes is not None:

            image_shape = (1688, 3000)

            hit_array = map_epoch_gaze_to_roi(
                et_ep,
                roi_boxes,
                image_shape
            )

            # =================================================
            # VALID ROI HITS
            # =================================================

            valid_hits = (
                (hit_array != "background")
                &
                (hit_array != "invalid")
            )

            total_gaze_sec = (
                np.sum(valid_hits) / 120
            )

            # =================================================
            # ROI DWELL
            # =================================================

            roi_dwell = {}

            unique_rois = np.unique(hit_array)

            for roi_name in unique_rois:

                if roi_name in [
                    "background",
                    "invalid"
                ]:
                    continue

                dwell_sec = (
                    np.sum(hit_array == roi_name)
                    / 120
                )

                roi_dwell[
                    str(roi_name)
                ] = float(dwell_sec)

            # =================================================
            # TOP ROI
            # =================================================

            top_roi = "none"

            if len(roi_dwell) > 0:

                top_roi = max(
                    roi_dwell,
                    key=roi_dwell.get
                )

            # =================================================
            # DEBUG
            # =================================================

            print("\n===== ROI STATISTICS =====")

            for k, v in roi_dwell.items():

                print(
                    f"{k}: {v:.3f} sec"
                )

            print(
                f"Top ROI         : {top_roi}"
            )

            print(
                f"Total Gaze Time : "
                f"{total_gaze_sec:.3f} sec"
            )

            dwell_info = {

                "top_roi":
                    top_roi,

                "total_gaze_seconds":
                    total_gaze_sec,

                "roi_dwell":
                    roi_dwell
            }

        # ----------------------------------------------------
        # ENGAGEMENT FEATURES (inline per epoch)
        # ----------------------------------------------------

        gaze_entropy  = _et_gaze_entropy(et_ep)
        roi_density   = _et_roi_density(et_ep)
        revisit_count = _et_revisit_count(et_ep)

        # dwell_time: prefer ROI-based if available, else valid-gaze seconds
        meta_dwell    = dwell_info.get("total_gaze_seconds", 0.0)
        dwell_time_s  = (
            meta_dwell if meta_dwell > 0.0
            else float(np.sum(np.isfinite(np.asarray(et_ep, dtype=float)[:, 0])) / _ET_SR)
        )
        fix_dur_s = round(fx_info["mean_duration"] / _ET_SR, 4)

        # ----------------------------------------------------
        # SAVE TEMP
        # ----------------------------------------------------

        final_eeg.append(eeg_ep)

        final_et.append(et_ep)

        final_labels.append(clean_label)

        metadata.append({

            "event_id":
                idx,

            "timestamp":
                ev["timestamp"],

            "label":
                clean_label,

            "eeg_samples":
                len(eeg_ep),

            "et_samples":
                len(et_ep),

            "fixation_count":
                fx_info["count"],

            "mean_fix_dur":
                round(
                    fx_info["mean_duration"],
                    2
                ),

            "top_roi":
                dwell_info.get(
                    "top_roi",
                    "none"
                ),

            "total_gaze_sec":
                round(
                    dwell_info.get(
                        "total_gaze_seconds",
                        0.0
                    ),
                    3
                ),

            # --- Engagement features -------------------------
            "fixation_duration_sec" : fix_dur_s,
            "dwell_time_sec"        : round(dwell_time_s, 4),
            "roi_density"           : round(roi_density, 4),
            "revisit_count"         : revisit_count,
            "gaze_entropy"          : round(gaze_entropy, 4),
        })

    print(
        f"\n===== FINAL EPOCH COUNT : "
        f"{len(final_eeg)} ====="
    )

    if len(final_eeg) == 0:

        print(
            "\n[STOP] No epochs passed validation."
        )

        return

    # ========================================================
    # 7. REMOVE INCOMPLETE EPOCHS
    # ========================================================

    EXPECTED_EEG = 1500
    EXPECTED_ET = 600

    valid_idx = []

    for i in range(len(final_eeg)):

        eeg_ok = (
            final_eeg[i].shape[0]
            == EXPECTED_EEG
        )

        et_ok = (
            final_et[i].shape[0]
            == EXPECTED_ET
        )

        if eeg_ok and et_ok:

            valid_idx.append(i)

    removed = (
        len(final_eeg)
        - len(valid_idx)
    )

    final_eeg = [
        final_eeg[i]
        for i in valid_idx
    ]

    final_et = [
        final_et[i]
        for i in valid_idx
    ]

    final_labels = [
        final_labels[i]
        for i in valid_idx
    ]

    meta_df = pd.DataFrame(metadata)

    meta_df = meta_df.iloc[
        valid_idx
    ].reset_index(drop=True)

    print("\n===== EPOCH FILTERING =====")

    print(
        f"Removed incomplete epochs : "
        f"{removed}"
    )

    print(
        f"Remaining valid epochs    : "
        f"{len(final_eeg)}"
    )

    # ========================================================
    # 8. SAVE OUTPUTS
    # ========================================================

    eeg_arr = np.array(
        final_eeg,
        dtype=object
    )

    et_arr = np.array(
        final_et,
        dtype=object
    )

    lbl_arr = np.array(final_labels)

    np.save(
        OUTPUT_EPOCHS_DIR / "eeg_epochs.npy",
        eeg_arr
    )

    np.save(
        OUTPUT_EPOCHS_DIR / "et_epochs.npy",
        et_arr
    )

    np.save(
        OUTPUT_EPOCHS_DIR / "labels.npy",
        lbl_arr
    )

    meta_df.to_csv(
        OUTPUT_METADATA_DIR / "metadata.csv",
        index=False
    )

    print("\nSaved:")

    print(
        f"  {OUTPUT_EPOCHS_DIR}/eeg_epochs.npy"
    )

    print(
        f"  {OUTPUT_EPOCHS_DIR}/et_epochs.npy"
    )

    print(
        f"  {OUTPUT_EPOCHS_DIR}/labels.npy"
    )

    print(
        f"  {OUTPUT_METADATA_DIR}/metadata.csv"
    )

    # ========================================================
    # 8B. ENGAGEMENT LABELING (subject-level median threshold)
    # ========================================================
    # For cross-subject (publication-grade) threshold run
    # engagement_labeling.py after all subjects are processed.

    print("\n==============================")
    print(f" ENGAGEMENT LABELING ({SUBJECT})")
    print("==============================")

    # Filter to stimulus epochs (ImagePage_*) for engagement scoring
    stim_mask = meta_df["label"].astype(str).str.startswith("ImagePage")
    stim_idx  = stim_mask[stim_mask].index.tolist()

    if len(stim_idx) == 0:
        print("[INFO] No ImagePage epochs — skipping engagement scoring")
    else:
        eng_feature_rows = meta_df.iloc[stim_idx].to_dict("records")
        eng_scores, eng_thresh, eng_labels = _score_engagement(eng_feature_rows)

        stim_eeg = [final_eeg[i] for i in stim_idx]
        stim_et  = [final_et[i]  for i in stim_idx]

        eng_dir = Path(OUTPUT_EPOCHS_DIR).parent / "engagement"
        eng_dir.mkdir(parents=True, exist_ok=True)

        np.save(eng_dir / "engagement_labels.npy", eng_labels.astype(np.int32))
        np.save(eng_dir / "engagement_scores.npy", eng_scores)

        np.save(Path(OUTPUT_EPOCHS_DIR) / "eeg_epochs_engagement.npy",
                np.array(stim_eeg, dtype=object))
        np.save(Path(OUTPUT_EPOCHS_DIR) / "et_epochs_engagement.npy",
                np.array(stim_et, dtype=object))

        eng_meta = meta_df.iloc[stim_idx].copy().reset_index(drop=True)
        eng_meta["engagement_score"] = eng_scores
        eng_meta["engagement_label"] = [_LABEL_NAMES[int(l)] for l in eng_labels]
        eng_meta["threshold"]        = eng_thresh
        eng_meta.to_csv(eng_dir / "engagement_metadata.csv", index=False)

        n_high = int((eng_labels == 1).sum())
        n_low  = int((eng_labels == 0).sum())
        print(f"\nStimulus epochs         : {len(stim_idx)}")
        print(f"Subject median threshold: {eng_thresh:.4f}")
        print(f"HIGH_ENGAGEMENT         : {n_high}")
        print(f"LOW_ENGAGEMENT          : {n_low}")
        print(f"Balance ratio           : {min(n_high, n_low)/max(n_high, n_low, 1):.3f}")
        print(f"\nSaved to: {eng_dir}")
        print( "  engagement_labels.npy")
        print( "  engagement_scores.npy")
        print( "  engagement_metadata.csv")
        print( "  epochs/eeg_epochs_engagement.npy")
        print( "  epochs/et_epochs_engagement.npy")

    # ========================================================
    # 9. VALIDATION PLOTS
    # ========================================================

    print("\nGenerating validation plots...")

    plot_eeg_epoch(
        final_eeg[0],
        event_id=0,
        save_path=(
            OUTPUT_PLOTS_DIR
            / "sample_eeg_epoch.png"
        )
    )

    plot_et_epoch(
        final_et[0],
        event_id=0,
        save_path=(
            OUTPUT_PLOTS_DIR
            / "sample_et_epoch.png"
        )
    )

    plot_epoch_summary(
        meta_df,
        save_path=(
            OUTPUT_PLOTS_DIR
            / "epoch_summary.png"
        )
    )

    plot_fixation_density(
        all_fixations,
        save_path=(
            OUTPUT_PLOTS_DIR
            / "fixation_density.png"
        )
    )

    # ========================================================
    # 10. SUMMARY
    # ========================================================

    print("\n==============================")
    print(" PHASE 3 COMPLETE")
    print("==============================")

    print(
        f"EEG epochs : {len(final_eeg)}"
    )

    print(
        f"ET  epochs : {len(final_et)}"
    )

    print(
        f"Labels     : {len(final_labels)}"
    )

    print("\nLabel distribution:")

    print(
        meta_df["label"]
        .value_counts()
        .to_string()
    )

    print("\nEpoch statistics:")

    print(
        meta_df[
            [
                "eeg_samples",
                "et_samples",
                "fixation_count",
                "mean_fix_dur",
                "total_gaze_sec"
            ]
        ]
        .describe()
        .round(2)
        .to_string()
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()