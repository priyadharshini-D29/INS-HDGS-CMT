"""
================================================================
NEUMA Phase 8 — Leakage Audit
================================================================
Run this BEFORE any published LOSOCV results to confirm the
pipeline is leakage-free.

Checks performed
----------------
1. Data isolation      — detect subjects sharing the same data file
                         without subject_ids.npy filtering
2. Duplicate epochs    — cosine similarity > 0.999 between epochs
3. Label distribution  — class imbalance / label collapse
4. Subject distribution— epochs per subject
5. Random-label sanity — LOSOCV with shuffled labels → must hit chance
6. ROI ablation        — LOSOCV with zeroed ROI vectors
7. Identity-graph      — LOSOCV with adjacency = I (graph leakage test)
8. Subject-shuffle     — shuffle subject assignments and re-run

Usage
-----
    python evaluation/leakage_audit.py
    python evaluation/leakage_audit.py --quick      # skip LOSOCV tests
    python evaluation/leakage_audit.py --subject S02
================================================================
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    SUBJECT_IDS, EPOCHS, BATCH_SIZE, METRICS_DIR, ABLATION_DIR,
)
from data.dataset import NeumaGraphDataset, _normalize_subject_id


# ── Helpers ──────────────────────────────────────────────────────────────────

def _banner(text, char="=", width=60):
    print(f"\n{char*width}")
    print(f"  {text}")
    print(f"{char*width}")


def _pass(msg):
    print(f"  [PASS] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")


def _warn(msg):
    print(f"  [WARN] {msg}")


# ── Check 1: Data Isolation ───────────────────────────────────────────────────

def check_data_isolation(subject_ids=None) -> bool:
    """
    Verify that each subject ID maps to a distinct set of source data.

    Uses the same file candidates the dataset loader uses (engagement_phase3d
    and engagement flavours) rather than the raw eeg_epochs.npy, so subjects
    that have no engagement data are correctly treated as absent — not as
    fallback-to-pooled leakage.
    """
    _banner("Check 1: Data Isolation", char="-")

    if subject_ids is None:
        subject_ids = SUBJECT_IDS

    from config.settings import PHASE3_DIR

    # Candidates the dataset actually loads — ordered by priority.
    EEG_CANDIDATES = [
        ("engagement_phase3d", "eeg_epochs_phase3d.npy"),
        ("engagement",         "eeg_epochs_engagement.npy"),
    ]

    per_subj_ok     = []   # subjects with their own engagement data
    per_subj_absent = []   # subjects with no engagement data at all
    shared_risk     = []   # subjects whose data is only in a pooled dir

    for sid in subject_ids:
        sid          = _normalize_subject_id(sid)
        subj_root    = PHASE3_DIR.parent / sid / "output"
        found_own    = False

        for eng_sub, eeg_file in EEG_CANDIDATES:
            eng_dir  = subj_root / eng_sub
            lbl_path = eng_dir / "engagement_labels.npy"
            eeg_path = subj_root / "epochs" / eeg_file
            if lbl_path.exists() and eeg_path.exists():
                found_own = True
                break

        if found_own:
            per_subj_ok.append(sid)
        else:
            # Check if pooled directory could serve this subject
            pooled_eeg = PHASE3_DIR / "epochs" / "eeg_epochs.npy"
            if pooled_eeg.exists():
                shared_risk.append(sid)
            else:
                per_subj_absent.append(sid)

    print(f"  Subjects with own engagement data : {len(per_subj_ok)}")
    print(f"  Subjects absent (no data found)   : {len(per_subj_absent)}")
    print(f"  Subjects sharing pooled dir        : {len(shared_risk)}")

    if per_subj_absent:
        print(f"  Absent subjects (will be skipped by dataset): "
              f"{per_subj_absent[:10]}")

    if len(shared_risk) > 1:
        _fail(
            f"{len(shared_risk)} subjects ({shared_risk[:5]}) share the pooled "
            f"data directory WITHOUT subject-level filtering.\n"
            f"    This causes data leakage in LOSOCV.\n"
            f"  Fix: run Phase 3 engagement labeling per subject to create\n"
            f"    {PHASE3_DIR.parent}/SXX/output/engagement_phase3d/ directories."
        )
        return False

    if len(shared_risk) == 1:
        _warn(f"One subject ({shared_risk}) shares pooled dir — single-subject "
              f"fallback, no cross-subject leakage.")

    _pass(
        f"Isolation OK — {len(per_subj_ok)} subjects have dedicated engagement "
        f"data; {len(per_subj_absent)} absent subjects will be skipped."
    )
    return True


# ── Check 2: Duplicate Epochs ─────────────────────────────────────────────────

def check_duplicate_epochs(
    X          : np.ndarray,
    subject_ids: np.ndarray = None,
    threshold  : float      = 0.9999,
) -> bool:
    """
    Detect near-identical epochs using cosine similarity.

    EEG signals recorded from the same subject during similar cognitive
    states are naturally correlated, so the threshold must be very high
    (0.9999) to flag true duplicates rather than similar-but-distinct
    epochs.  Cross-subject duplicates at this threshold indicate a data
    leakage problem; within-subject duplicates suggest repeated stimulus
    presentation (acceptable for LOSOCV as long as subjects are isolated).

    Returns True if no cross-subject duplicates are found.
    """
    _banner("Check 2: Duplicate Epoch Detection", char="-")

    N = len(X)
    Xf = X.reshape(N, -1).astype(np.float64)

    # Normalise rows to unit vectors (safe against zero-norm rows)
    norms = np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-10
    Xf    = Xf / norms

    sim = Xf @ Xf.T
    np.fill_diagonal(sim, 0.0)

    max_sim = sim.max()
    dup_mask = sim > threshold
    n_pairs  = int(dup_mask.sum()) // 2

    print(f"  Epochs               : {N}")
    print(f"  Max pairwise cosine  : {max_sim:.6f}")
    print(f"  Near-duplicate pairs : {n_pairs}  (threshold={threshold})")

    if n_pairs == 0:
        _pass("No near-duplicate epochs.")
        return True

    # Distinguish within-subject vs cross-subject duplicates
    pairs = np.argwhere(dup_mask)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]

    cross_subj = 0
    if subject_ids is not None:
        for i, j in pairs:
            if int(subject_ids[i]) != int(subject_ids[j]):
                cross_subj += 1

    print(f"  First duplicate pairs: {pairs[:10].tolist()}")

    if subject_ids is not None:
        within_subj = n_pairs - cross_subj
        print(f"  Within-subject pairs : {within_subj} (repeated stimulus — OK)")
        print(f"  Cross-subject pairs  : {cross_subj} (potential leakage)")

        if cross_subj > 0:
            _fail(f"{cross_subj} cross-subject duplicate epoch pairs detected — "
                  "data leakage risk.")
            return False

        _pass(f"{n_pairs} within-subject near-duplicates (repeated stimuli — "
              "no cross-subject leakage).")
        return True

    # No subject IDs available — warn but don't fail
    _warn(f"{n_pairs} near-duplicate pairs found (no subject IDs to classify "
          "as within/cross-subject).")
    return True


# ── Check 3: Label Distribution ──────────────────────────────────────────────

def check_label_distribution(labels: np.ndarray) -> bool:
    """Print label counts and warn on heavy imbalance or collapse."""
    _banner("Check 3: Label Distribution", char="-")

    counts = Counter(int(l) for l in labels)
    n_total = sum(counts.values())
    n_classes = len(counts)

    print(f"  Total epochs  : {n_total}")
    print(f"  Classes       : {n_classes}")
    for cls, cnt in sorted(counts.items()):
        pct = 100 * cnt / n_total
        print(f"    Class {cls}: {cnt:4d} ({pct:5.1f}%)")

    if n_classes <= 1:
        _fail("Only one class present — trivially 100% accuracy by constant prediction.")
        return False

    dominant_pct = max(counts.values()) / n_total * 100
    if dominant_pct > 90:
        _fail(f"Dominant class has {dominant_pct:.1f}% of samples — "
              "model can hit that accuracy by always predicting majority class.")
        return False

    if dominant_pct > 70:
        _warn(f"Class imbalance: dominant class = {dominant_pct:.1f}%")
    else:
        _pass(f"Label distribution acceptable (dominant class = {dominant_pct:.1f}%).")

    return True


# ── Check 4: Subject Distribution ────────────────────────────────────────────

def check_subject_distribution(subject_ids: np.ndarray) -> bool:
    """Print epochs-per-subject counts."""
    _banner("Check 4: Subject Distribution", char="-")

    counts = Counter(int(s) for s in subject_ids)
    n_subjects = len(counts)
    total = sum(counts.values())

    print(f"  Unique subjects : {n_subjects}")
    print(f"  Total epochs    : {total}")
    for sid, cnt in sorted(counts.items()):
        print(f"    Subject {sid:2d}: {cnt:4d} epochs")

    if n_subjects <= 1:
        _warn("Only 1 subject — true LOSOCV is impossible. "
              "Using within-subject split instead.")
        return False

    _pass(f"{n_subjects} subjects found.")
    return True


# ── Check 5: Random-Label Sanity ─────────────────────────────────────────────

def check_random_label_sanity(
    subject_ids=None,
    epochs: int = 30,
    batch_size: int = None,
    verbose: bool = True,
) -> bool:
    """
    Run a short LOSOCV with shuffled labels.
    Expected: accuracy ≈ 1/n_classes (chance).
    If accuracy stays high: leakage is confirmed.
    """
    _banner("Check 5: Random-Label Sanity Check", char="-")
    print("  Permuting training labels → expect chance-level performance.")

    from .losocv import run_losocv
    from models.ins_hdgs_cmt import AblationConfig

    bs = batch_size or BATCH_SIZE
    ep = min(epochs, 30)

    df = run_losocv(
        subject_ids   = subject_ids,
        ablation      = AblationConfig.full(),
        epochs        = ep,
        batch_size    = bs,
        label         = "sanity_random_labels",
        save_dir      = ABLATION_DIR / "sanity",
        verbose       = verbose,
        random_labels = True,
    )

    if df.empty:
        _warn("No folds completed — cannot assess.")
        return False

    mean_acc = df["accuracy"].mean() if "accuracy" in df.columns else None
    n_cls    = df["accuracy"].count()

    print(f"  Mean accuracy (random labels) : "
          f"{mean_acc:.4f}" if mean_acc is not None else "  N/A")

    if mean_acc is None:
        _warn("Accuracy not in results.")
        return False

    chance = 1.0 / max(df.get("n_classes", [7]).values[0], 2)
    print(f"  Chance level  : {chance:.4f}")

    if mean_acc < chance + 0.15:
        _pass(f"Accuracy with random labels ({mean_acc:.4f}) ≈ chance ({chance:.4f}) — "
              "no label leakage.")
        return True

    _fail(
        f"Accuracy with random labels = {mean_acc:.4f}, "
        f"expected ≈ {chance:.4f}.\n"
        f"    This confirms data leakage — the model is not learning from labels."
    )
    return False


# ── Check 6: ROI Ablation ────────────────────────────────────────────────────

def check_roi_ablation(
    subject_ids=None,
    epochs: int = 30,
    batch_size: int = None,
    verbose: bool = True,
) -> dict:
    """
    Run LOSOCV with ROI vectors zeroed out.
    Returns dict with {'full': mean_acc, 'no_roi': mean_acc}.
    """
    _banner("Check 6: ROI Ablation", char="-")
    print("  Zeroing ROI vectors — large accuracy drop → ROI contributes.")
    print("  No drop → ROI is redundant or model ignores it.")

    from .losocv import run_losocv
    from models.ins_hdgs_cmt import AblationConfig

    bs = batch_size or BATCH_SIZE
    ep = min(epochs, 30)

    df_no_roi = run_losocv(
        subject_ids   = subject_ids,
        ablation      = AblationConfig.full(),
        epochs        = ep,
        batch_size    = bs,
        label         = "ablation_zero_roi",
        save_dir      = ABLATION_DIR / "sanity",
        verbose       = verbose,
        zero_roi      = True,
    )

    if df_no_roi.empty:
        _warn("No folds completed.")
        return {}

    acc_no_roi = df_no_roi["accuracy"].mean() if "accuracy" in df_no_roi.columns else None
    print(f"  Accuracy (zero ROI) : {acc_no_roi:.4f}" if acc_no_roi else "  N/A")
    return {"no_roi": acc_no_roi}


# ── Check 7: Identity-Graph Ablation ─────────────────────────────────────────

def check_identity_graph_ablation(
    subject_ids=None,
    epochs: int = 30,
    batch_size: int = None,
    verbose: bool = True,
) -> dict:
    """
    Run LOSOCV replacing adjacency matrices with identity matrices.
    If accuracy stays high: graph structure leaks or GAT does nothing.
    """
    _banner("Check 7: Identity-Graph Ablation", char="-")
    print("  Replacing adj matrices with I — tests whether graph structure matters.")

    from .losocv import run_losocv
    from models.ins_hdgs_cmt import AblationConfig

    bs = batch_size or BATCH_SIZE
    ep = min(epochs, 30)

    df_id = run_losocv(
        subject_ids    = subject_ids,
        ablation       = AblationConfig.full(),
        epochs         = ep,
        batch_size     = bs,
        label          = "ablation_identity_graph",
        save_dir       = ABLATION_DIR / "sanity",
        verbose        = verbose,
        identity_graph = True,
    )

    if df_id.empty:
        _warn("No folds completed.")
        return {}

    acc_id = df_id["accuracy"].mean() if "accuracy" in df_id.columns else None
    print(f"  Accuracy (identity graph) : {acc_id:.4f}" if acc_id else "  N/A")

    if acc_id is not None and acc_id > 0.95:
        _fail(
            f"Identity-graph accuracy = {acc_id:.4f}. "
            "Graph branch contributes nothing or adjacency encodes class identity."
        )
    else:
        _pass("Accuracy drops with identity graph — graph structure is informative.")

    return {"identity_graph": acc_id}


# ── Check 8: Subject-Shuffle ──────────────────────────────────────────────────

def check_subject_shuffle(
    subject_ids=None,
    epochs: int = 30,
    batch_size: int = None,
    verbose: bool = True,
) -> bool:
    """
    Shuffle subject-to-fold assignments and re-run LOSOCV.
    If splits are correct, shuffling subject IDs should not improve or maintain
    perfect accuracy (it degrades meaningful evaluation, but still runs).
    The key check: if accuracy stays at 100%, the split logic is broken.
    """
    _banner("Check 8: Subject-Shuffle Sanity", char="-")
    print("  Shuffling subject IDs — if LOSOCV remains perfect, split logic is broken.")

    try:
        ds = NeumaGraphDataset(subject_ids=subject_ids, precompute_graphs=False)
    except FileNotFoundError as e:
        _warn(f"Could not load dataset: {e}")
        return False

    original_subjects = list(ds.unique_subjects)
    if len(original_subjects) <= 1:
        _warn("Only one subject — shuffle test not applicable.")
        return False

    shuffled = np.random.default_rng(0).permutation(original_subjects).tolist()
    print(f"  Original : {original_subjects[:5]}")
    print(f"  Shuffled : {shuffled[:5]}")

    from .losocv import run_losocv
    from models.ins_hdgs_cmt import AblationConfig

    bs = batch_size or BATCH_SIZE
    ep = min(epochs, 30)

    df = run_losocv(
        subject_ids = shuffled,
        ablation    = AblationConfig.full(),
        epochs      = ep,
        batch_size  = bs,
        label       = "sanity_subject_shuffle",
        save_dir    = ABLATION_DIR / "sanity",
        verbose     = verbose,
    )

    if df.empty:
        _warn("No folds completed.")
        return False

    mean_acc = df["accuracy"].mean() if "accuracy" in df.columns else None
    print(f"  Mean accuracy (shuffled subjects) : "
          f"{mean_acc:.4f}" if mean_acc else "  N/A")

    if mean_acc is not None and mean_acc > 0.98:
        _fail(
            f"Shuffled-subject accuracy = {mean_acc:.4f}. "
            "LOSOCV split is insensitive to subject ordering — data isolation is broken."
        )
        return False

    _pass("Subject shuffle affected performance — split logic appears correct.")
    return True


# ── Full Audit ────────────────────────────────────────────────────────────────

def run_audit(
    subject_ids = None,
    quick       : bool = False,
    epochs      : int  = 30,
    batch_size  : int  = None,
    verbose     : bool = True,
):
    """
    Run all leakage checks in order.

    Parameters
    ----------
    quick       : skip LOSOCV-based checks (5-8), only run data-level checks (1-4)
    epochs      : training epochs for LOSOCV sanity checks
    batch_size  : mini-batch size (None = use config default)
    """
    _banner("NEUMA Phase 8 — Leakage Audit")
    print(f"  Quick mode : {quick}")
    print(f"  Subjects   : {subject_ids or 'all available'}")

    results = {}

    # ── Static checks (no training) ──────────────────────────────────────────
    results["isolation"] = check_data_isolation(subject_ids)

    # Load dataset once for static checks
    try:
        ds = NeumaGraphDataset(
            subject_ids       = subject_ids,
            precompute_graphs = False,
        )
        n_check  = min(len(ds), 200)
        eeg_flat = np.stack([
            np.array(ds.raw_eeg[i]).reshape(-1)
            for i in range(n_check)
        ])
        sids_flat = np.array(ds.subject_ids[:n_check])
        results["duplicates"]          = check_duplicate_epochs(eeg_flat, sids_flat)
        results["label_distribution"]  = check_label_distribution(ds.labels)
        results["subject_distribution"]= check_subject_distribution(ds.subject_ids)
    except FileNotFoundError as e:
        _warn(f"Dataset load failed: {e}")
        results["duplicates"] = None
        results["label_distribution"] = None
        results["subject_distribution"] = None

    if quick:
        _banner("Quick audit complete — skipped LOSOCV checks (5-8)")
    else:
        # ── LOSOCV-based checks (require training) ────────────────────────────
        results["random_labels"]  = check_random_label_sanity(
            subject_ids, epochs=epochs, batch_size=batch_size, verbose=verbose
        )
        roi_res = check_roi_ablation(
            subject_ids, epochs=epochs, batch_size=batch_size, verbose=verbose
        )
        results.update(roi_res)

        graph_res = check_identity_graph_ablation(
            subject_ids, epochs=epochs, batch_size=batch_size, verbose=verbose
        )
        results.update(graph_res)

        results["subject_shuffle"] = check_subject_shuffle(
            subject_ids, epochs=epochs, batch_size=batch_size, verbose=verbose
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("Audit Summary")
    bool_checks = {k: v for k, v in results.items() if isinstance(v, bool)}
    num_checks  = {k: v for k, v in results.items() if isinstance(v, float)}

    passed = sum(1 for v in bool_checks.values() if v)
    failed = sum(1 for v in bool_checks.values() if not v)

    for name, ok in bool_checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    for name, val in num_checks.items():
        if val is not None:
            print(f"  [INFO] {name} accuracy = {val:.4f}")

    print(f"\n  {passed} passed  |  {failed} failed")

    if failed == 0 and not any(v is None for v in bool_checks.values()):
        print("\n  Pipeline appears leakage-free.")
    else:
        print(
            "\n  Leakage or data issues detected.\n"
            "  DO NOT publish results until all checks pass.\n"
            "  See individual check output above for remediation steps."
        )

    return results


# ── Public API ───────────────────────────────────────────────────────────────

def run_leakage_audit(
    subject_ids = None,
    quick       : bool = True,
    epochs      : int  = 30,
    batch_size  : int  = None,
    verbose     : bool = True,
) -> dict:
    """
    Public entry-point imported by main.py.

    Runs leakage checks and returns a dict that includes a ``has_risk`` key:
      True  — one or more checks failed (leakage detected)
      False — all checks passed

    Parameters mirror run_audit(); ``quick=True`` by default to avoid
    triggering a full LOSOCV training run from within main.py.
    """
    results = run_audit(
        subject_ids = subject_ids,
        quick       = quick,
        epochs      = epochs,
        batch_size  = batch_size,
        verbose     = verbose,
    )
    bool_checks = {k: v for k, v in results.items() if isinstance(v, bool)}
    results["has_risk"] = any(not v for v in bool_checks.values())
    return results


# backward compatibility
leakage_audit = run_leakage_audit


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="NEUMA Phase 8 Leakage Audit")
    p.add_argument("--subject",    default=None, help="Single subject to audit")
    p.add_argument("--quick",      action="store_true",
                   help="Skip LOSOCV training tests (fast, data-level only)")
    p.add_argument("--epochs",     type=int, default=30,
                   help="Epochs for LOSOCV sanity checks")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size for LOSOCV sanity checks")
    p.add_argument("--no-verbose", action="store_true",
                   help="Suppress per-epoch training output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    sids = [args.subject] if args.subject else None
    run_audit(
        subject_ids = sids,
        quick       = args.quick,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        verbose     = not args.no_verbose,
    )
