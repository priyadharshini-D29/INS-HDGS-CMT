"""
================================================================
NEUMA Phase 8 — Multi-Subject Data Aggregator
================================================================
Scans data_pipeline/04_segmentation/S*/output/epochs/ for per-subject
epoch files and merges them into a single pooled dataset at
data_pipeline/04_segmentation/output/epochs/ with a subject_ids.npy index.

This enables the Phase 8 LOSOCV pipeline (Option B isolation):
each fold filters the pooled arrays by subject_ids.npy instead
of needing separate directories per subject.

Run once after all subjects have been processed by 04_segmentation:

    python src/model/build_subject_pool.py
    python src/model/build_subject_pool.py --dry-run   # preview only
    python src/model/build_subject_pool.py --verify    # check output
    python src/model/build_subject_pool.py --output-dir /custom/path

After running, confirm isolation:
    python src/model/main.py --audit-quick
================================================================
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

# ── Path resolution ──────────────────────────────────────────────────────────
# _HERE.parent (src/) is the shared ancestor of both src/model/ and
# src/data_pipeline/ -- still correct after the reorg, just the sibling
# folder name changed (NEUMA_PHASE3 -> data_pipeline/04_segmentation).
_HERE       = Path(__file__).resolve().parent
PHASE3_DIR  = _HERE.parent / "data_pipeline" / "04_segmentation"
DEFAULT_OUT = PHASE3_DIR / "output" / "epochs"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_npy(path: Path, allow_pickle: bool = True):
    arr = np.load(path, allow_pickle=allow_pickle)
    # Unwrap object arrays that wrap a regular array
    if arr.dtype == object and arr.ndim == 0:
        arr = arr.item()
    return arr


def _to_3d(arr, label: str):
    """
    Ensure array is 3-D (N, T, C).
    Handles:
      - already (N, T, C) float arrays
      - object arrays of 2-D variable-length arrays → pad to max length
    """
    if arr.ndim == 3:
        return arr.astype(np.float32)

    if arr.ndim == 1 or arr.dtype == object:
        items = [np.asarray(a, dtype=np.float32) for a in arr]
        T_max = max(a.shape[0] for a in items)
        C     = items[0].shape[1] if items[0].ndim == 2 else 1
        out   = np.zeros((len(items), T_max, C), dtype=np.float32)
        for i, a in enumerate(items):
            t = min(a.shape[0], T_max)
            out[i, :t] = a[:t]
        print(f"    [{label}] padded ragged array → {out.shape}")
        return out

    raise ValueError(f"[{label}] unexpected array shape: {arr.shape}")


# ── Main Aggregation ──────────────────────────────────────────────────────────

def aggregate(
    phase3_dir : Path = PHASE3_DIR,
    out_dir    : Path = DEFAULT_OUT,
    dry_run    : bool = False,
    verbose    : bool = True,
) -> dict:
    """
    Scan phase3_dir/S*/output/epochs/ and pool all subjects.

    Returns
    -------
    summary : dict with keys 'n_subjects', 'n_epochs', 'subjects_found'
    """
    phase3_dir = Path(phase3_dir)
    out_dir    = Path(out_dir)

    # ── Discover per-subject directories ─────────────────────────────────────
    pattern  = str(phase3_dir / "S*" / "output" / "epochs")
    subj_dirs = sorted(glob.glob(pattern))

    if not subj_dirs:
        print(
            f"\n[ERROR] No per-subject epoch directories found.\n"
            f"  Pattern searched: {pattern}\n"
            f"  Expected layout :\n"
            f"    {phase3_dir}/S01/output/epochs/eeg_epochs.npy\n"
            f"    {phase3_dir}/S02/output/epochs/eeg_epochs.npy\n"
            f"    ...\n"
            f"  Run Phase 3 separately for each subject first."
        )
        return {"n_subjects": 0, "n_epochs": 0, "subjects_found": []}

    print(f"\n{'='*60}")
    print(f"  Multi-Subject Aggregator")
    print(f"{'='*60}")
    print(f"  Phase 3 root : {phase3_dir}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Subjects found: {len(subj_dirs)}")

    # ── Load per-subject data ─────────────────────────────────────────────────
    all_eeg     : list = []
    all_et      : list = []
    all_labels  : list = []
    all_subj_ids: list = []
    subjects_loaded : list = []

    for sdir in subj_dirs:
        sdir = Path(sdir)
        # Subject ID = name of the S* directory (e.g. "S01")
        sid = sdir.parents[1].name

        eeg_path = sdir / "eeg_epochs.npy"
        et_path  = sdir / "et_epochs.npy"
        lbl_path = sdir / "labels.npy"

        if not eeg_path.exists():
            print(f"  [SKIP] {sid}: eeg_epochs.npy not found at {sdir}")
            continue

        if verbose:
            print(f"\n  Loading {sid} from {sdir}")

        eeg = _load_npy(eeg_path)
        et  = _load_npy(et_path)  if et_path.exists()  else None
        lbl = _load_npy(lbl_path) if lbl_path.exists() else None

        # Force 3-D (N, T, C)
        eeg = _to_3d(eeg, f"{sid}/EEG")
        N   = len(eeg)

        if et is not None:
            et = _to_3d(et, f"{sid}/ET")
            if len(et) != N:
                print(f"    [WARN] {sid}: ET length {len(et)} ≠ EEG length {N} — padding/truncating")
                if len(et) > N:
                    et = et[:N]
                else:
                    pad = np.zeros((N - len(et), et.shape[1], et.shape[2]),
                                   dtype=np.float32)
                    et = np.concatenate([et, pad], axis=0)
        else:
            # Synthesise zero ET if not present
            et = np.zeros((N, eeg.shape[1], 3), dtype=np.float32)
            print(f"    [INFO] {sid}: no ET file — using zeros")

        if lbl is None:
            lbl = np.array([f"epoch_{i}" for i in range(N)])
            print(f"    [INFO] {sid}: no labels file — using synthetic labels")
        else:
            lbl = np.asarray(lbl)
            if len(lbl) != N:
                print(f"    [WARN] {sid}: labels length {len(lbl)} ≠ EEG length {N}")
                lbl = lbl[:N] if len(lbl) > N else np.pad(
                    lbl.astype(str), (0, N - len(lbl)),
                    constant_values="unknown"
                )

        all_eeg.append(eeg)
        all_et.append(et)
        all_labels.append(lbl)
        all_subj_ids.extend([sid] * N)
        subjects_loaded.append(sid)

        if verbose:
            from collections import Counter
            print(f"    EEG  : {eeg.shape}")
            print(f"    ET   : {et.shape}")
            print(f"    Labels ({N}): {Counter(str(l) for l in lbl)}")

    if not subjects_loaded:
        print("\n[ERROR] No subject data loaded — aborting.")
        return {"n_subjects": 0, "n_epochs": 0, "subjects_found": []}

    # ── Concatenate ───────────────────────────────────────────────────────────
    # Pad channels to the maximum across subjects if needed
    C_eeg_max = max(a.shape[2] for a in all_eeg)
    C_et_max  = max(a.shape[2] for a in all_et)
    T_eeg_max = max(a.shape[1] for a in all_eeg)
    T_et_max  = max(a.shape[1] for a in all_et)

    def _pad_to(arr, T, C):
        N, t, c = arr.shape
        out = np.zeros((N, T, C), dtype=np.float32)
        out[:, :t, :c] = arr
        return out

    X_eeg = np.concatenate([_pad_to(a, T_eeg_max, C_eeg_max) for a in all_eeg], axis=0)
    X_et  = np.concatenate([_pad_to(a, T_et_max,  C_et_max)  for a in all_et],  axis=0)
    y     = np.concatenate(all_labels)
    sids  = np.array(all_subj_ids)

    n_total = len(X_eeg)
    print(f"\n{'─'*60}")
    print(f"  Pooled dataset:")
    print(f"    Subjects  : {len(subjects_loaded)}  {subjects_loaded}")
    print(f"    Epochs    : {n_total}")
    print(f"    EEG shape : {X_eeg.shape}")
    print(f"    ET  shape : {X_et.shape}")
    print(f"    Labels    : {y.shape}")
    print(f"    SubjectIDs: {sids.shape}  unique={np.unique(sids).tolist()}")

    # ── Verify no subject appears in both train and test for any fold ─────────
    _verify_isolation(sids, subjects_loaded)

    if dry_run:
        print(f"\n  [DRY RUN] Would save to {out_dir}/ — nothing written.")
        return {
            "n_subjects"    : len(subjects_loaded),
            "n_epochs"      : n_total,
            "subjects_found": subjects_loaded,
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "eeg_epochs.npy",   X_eeg)
    np.save(out_dir / "et_epochs.npy",    X_et)
    np.save(out_dir / "labels.npy",       y)
    np.save(out_dir / "subject_ids.npy",  sids)

    print(f"\n  Saved to {out_dir}/")
    print(f"    eeg_epochs.npy   {X_eeg.shape}")
    print(f"    et_epochs.npy    {X_et.shape}")
    print(f"    labels.npy       {y.shape}")
    print(f"    subject_ids.npy  {sids.shape}")
    print(f"\n  Run audit to confirm isolation:")
    print(f"    python src/model/main.py --audit-quick")

    return {
        "n_subjects"    : len(subjects_loaded),
        "n_epochs"      : n_total,
        "subjects_found": subjects_loaded,
    }


def _verify_isolation(sids: np.ndarray, subjects: list):
    """
    Confirm that for every possible LOSOCV fold, train and test subjects
    are disjoint (they always are by construction, but verify the IDs
    in sids match the discovered subject list).
    """
    unique_in_sids = set(sids.tolist())
    unique_expected = set(subjects)

    missing  = unique_expected - unique_in_sids
    extra    = unique_in_sids   - unique_expected

    if missing:
        print(f"  [WARN] Subject IDs expected but missing from sids: {missing}")
    if extra:
        print(f"  [WARN] Extra subject IDs in sids not in subject list: {extra}")

    # Check per-fold: any subject that would appear in both train and test?
    # By construction (we assign sids[i] = sid for all i from that subject)
    # this cannot happen, but we verify counts are reasonable.
    from collections import Counter
    counts = Counter(sids.tolist())
    print(f"\n  Epochs per subject (for isolation verification):")
    for sid, cnt in sorted(counts.items()):
        print(f"    {sid}: {cnt} epochs")

    if len(unique_in_sids) < 2:
        print(
            "  [WARN] Only one unique subject ID — LOSOCV will use "
            "within-subject 70/30 split, not true cross-subject evaluation."
        )
    else:
        print(f"\n  [OK] {len(unique_in_sids)} subjects — LOSOCV isolation is valid.")


# ── Verification Pass ─────────────────────────────────────────────────────────

def verify_output(out_dir: Path = DEFAULT_OUT):
    """Load the saved pooled files and verify structure."""
    out_dir = Path(out_dir)
    print(f"\n{'='*60}")
    print(f"  Verifying pooled output at {out_dir}/")
    print(f"{'='*60}")

    required = ["eeg_epochs.npy", "labels.npy", "subject_ids.npy"]
    for fname in required:
        fpath = out_dir / fname
        if not fpath.exists():
            print(f"  [MISSING] {fname}")
        else:
            arr = np.load(fpath, allow_pickle=True)
            print(f"  [OK] {fname}  shape={arr.shape}  dtype={arr.dtype}")

    sid_path = out_dir / "subject_ids.npy"
    if sid_path.exists():
        sids = np.load(sid_path, allow_pickle=True)
        from collections import Counter
        counts = Counter(str(s) for s in sids)
        print(f"\n  Subject distribution:")
        for sid, cnt in sorted(counts.items()):
            print(f"    {sid}: {cnt} epochs")

        n_subjects = len(counts)
        if n_subjects >= 2:
            print(f"\n  [READY] {n_subjects} subjects — Phase 8 LOSOCV will work correctly.")
        else:
            print(f"\n  [WARN] Only {n_subjects} subject — need ≥2 for LOSOCV.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Aggregate per-subject Phase 3 outputs into a pooled LOSOCV dataset"
    )
    p.add_argument(
        "--phase3-dir", default=str(PHASE3_DIR),
        help="Root of data_pipeline/04_segmentation (default: auto-detected sibling)"
    )
    p.add_argument(
        "--output-dir", default=str(DEFAULT_OUT),
        help="Where to write the pooled files (default: data_pipeline/04_segmentation/output/epochs/)"
    )
    p.add_argument("--dry-run",  action="store_true",
                   help="Preview without writing anything")
    p.add_argument("--verify",   action="store_true",
                   help="Verify existing output and exit")
    p.add_argument("--quiet",    action="store_true",
                   help="Suppress per-subject loading output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.verify:
        verify_output(Path(args.output_dir))
        sys.exit(0)

    result = aggregate(
        phase3_dir = Path(args.phase3_dir),
        out_dir    = Path(args.output_dir),
        dry_run    = args.dry_run,
        verbose    = not args.quiet,
    )

    if result["n_subjects"] == 0:
        sys.exit(1)
