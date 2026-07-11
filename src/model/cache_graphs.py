"""
================================================================
INS-HDGS-CMT — Graph Pre-Caching Script
================================================================
Run ONCE before starting LOSOCV training to pre-compute and
save all epoch graphs to disk.

After this runs, every fold worker loads cached .pt tensors
instead of calling Welch + Pearson from scratch.  With 32 folds
this saves 31/32 of all graph computation (and ~31 × 3-5 min
of CPU overhead spread across the parallel fold workers).

Usage
-----
  python cache_graphs.py                  # cache all subjects
  python cache_graphs.py --subjects S01,S02,S33
  python cache_graphs.py --force          # recompute even if cached
  python cache_graphs.py --workers 8      # parallel subjects (CPU)

Outputs
-------
  data_pipeline/04_segmentation/S##/output/graph_cache/<cache_key>/ep{NNNN}.pt

The cache_key is an 8-char hash of (n_windows, conn_method,
threshold, fs_eeg) — if you change any graph parameter, the
old cache is silently ignored and a new one is written.
================================================================
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# data/dataset.py imports `model.inference...` as an absolute package path,
# so the *parent* of this directory (not just this directory itself) must
# be importable too -- mirrors the pattern in main.py / sensitivity_sweep.py.
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))


def _cache_one_subject(args_tuple) -> tuple[str, int, float]:
    """Worker: compute + save graphs for all epochs of one subject."""
    sid, force, n_windows, conn_method, threshold, fs_eeg = args_tuple

    import numpy as np
    from data.dataset import (
        NeumaGraphDataset, _normalize_subject_id,
        load_cached_graph, save_cached_graph, _graph_cache_key,
    )

    cache_key = _graph_cache_key(n_windows, conn_method, threshold, fs_eeg)

    # Build a single-subject dataset (no graph precompute — we handle it here)
    try:
        ds = NeumaGraphDataset(
            subject_ids       = [sid],
            precompute_graphs = False,   # don't precompute in __init__
            use_disk_cache    = False,   # we control cache below
        )
    except Exception as e:
        return sid, 0, 0.0

    n = len(ds)
    t0 = time.perf_counter()
    written = 0

    for idx in range(n):
        subj_dir = ds._subj_dirs[idx]
        if subj_dir is None:
            continue
        cache_path_exists = load_cached_graph(subj_dir, idx, cache_key) is not None
        if cache_path_exists and not force:
            continue
        graph = ds._compute_graph(idx)
        save_cached_graph(subj_dir, idx, cache_key, graph)
        written += 1

    elapsed = time.perf_counter() - t0
    return sid, written, elapsed


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-compute and cache epoch graphs to disk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subjects", type=str, default=None,
                   help="Comma-separated subject IDs (default: all configured)")
    p.add_argument("--force", action="store_true",
                   help="Recompute and overwrite even if cache already exists")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel subject workers (CPU processes)")
    p.add_argument("--n-windows", type=int, default=None, dest="n_windows")
    p.add_argument("--conn-method", type=str, default=None, dest="conn_method")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--fs-eeg", type=float, default=None, dest="fs_eeg")
    args = p.parse_args()

    from config.settings import (
        SUBJECT_IDS, N_WINDOWS, CONN_METHOD, CONN_THRESHOLD, EEG_SR,
    )
    from data.dataset import _graph_cache_key

    n_windows   = args.n_windows   or N_WINDOWS
    conn_method = args.conn_method or CONN_METHOD
    threshold   = args.threshold   if args.threshold is not None else CONN_THRESHOLD
    fs_eeg      = args.fs_eeg      or EEG_SR

    cache_key = _graph_cache_key(n_windows, conn_method, threshold, fs_eeg)

    subject_ids = (
        [s.strip() for s in args.subjects.split(",")]
        if args.subjects else list(SUBJECT_IDS)
    )

    print(f"\n  [cache_graphs] Subjects : {len(subject_ids)}")
    print(f"  [cache_graphs] Cache key: {cache_key}")
    print(f"  [cache_graphs] Workers  : {args.workers}")
    print(f"  [cache_graphs] Force    : {args.force}")
    print()

    worker_args = [
        (sid, args.force, n_windows, conn_method, threshold, fs_eeg)
        for sid in subject_ids
    ]

    t_total = time.perf_counter()
    results = []

    if args.workers == 1:
        for wa in worker_args:
            results.append(_cache_one_subject(wa))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_cache_one_subject, wa): wa[0]
                       for wa in worker_args}
            for fut in as_completed(futures):
                results.append(fut.result())

    results.sort(key=lambda r: r[0])
    total_written = 0
    for sid, written, elapsed in results:
        status = f"{written} graphs in {elapsed:.1f}s" if written else "already cached"
        print(f"  {sid}: {status}")
        total_written += written

    elapsed_total = time.perf_counter() - t_total
    print(f"\n  Done. {total_written} graphs written in {elapsed_total:.1f}s")
    print(f"  Cache key: {cache_key}")
    print()


if __name__ == "__main__":
    main()
