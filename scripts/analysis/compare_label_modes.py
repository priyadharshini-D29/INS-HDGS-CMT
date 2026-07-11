"""
Compare global vs subject-median engagement labels before LOSOCV training.

Run from NEUMA_PHASE8:
    python analysis/compare_label_modes.py

The script avoids graph precomputation, so it is a quick sanity check for
single-class held-out subjects and per-subject class balance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config.settings import SUBJECT_IDS
from data.dataset import NeumaGraphDataset, _normalize_subject_id


def summarize_mode(label_mode: str) -> list[dict]:
    rows = []
    for sid in [_normalize_subject_id(s) for s in SUBJECT_IDS]:
        try:
            ds = NeumaGraphDataset(
                subject_ids=[sid],
                precompute_graphs=False,
                label_mode=label_mode,
            )
        except FileNotFoundError:
            rows.append({
                "subject": sid,
                "n": 0,
                "low": 0,
                "high": 0,
                "valid_fold": False,
                "status": "missing",
            })
            continue

        counts = np.bincount(ds.labels, minlength=2)
        low, high = int(counts[0]), int(counts[1])
        rows.append({
            "subject": sid,
            "n": int(len(ds)),
            "low": low,
            "high": high,
            "valid_fold": low > 0 and high > 0,
            "status": "ok" if low > 0 and high > 0 else "single_class",
        })
    return rows


def print_table(label_mode: str, rows: list[dict]) -> None:
    valid = sum(1 for r in rows if r["valid_fold"])
    present = sum(1 for r in rows if r["n"] > 0)
    print(f"\nLABEL_MODE={label_mode}  present={present}  valid_losocv_folds={valid}")
    print("subject   n   LOW  HIGH  status")
    print("------- --- ----- -----  ------------")
    for r in rows:
        print(
            f"{r['subject']:>7} {r['n']:>3} {r['low']:>5} "
            f"{r['high']:>5}  {r['status']}"
        )


def main() -> None:
    for mode in ("global", "subject_median"):
        rows = summarize_mode(mode)
        print_table(mode, rows)


if __name__ == "__main__":
    main()
