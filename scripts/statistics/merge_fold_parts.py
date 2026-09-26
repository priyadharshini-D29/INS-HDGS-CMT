#!/usr/bin/env python
"""Merge the per-part outputs of baselines/run_baselines.py --folds A-B into the
plain per-model files that every analysis script reads.

    python scripts/revision/merge_fold_parts.py --out-dir results/baselines/dl_tuned --models eegnet_et shallow_et
    python scripts/revision/merge_fold_parts.py --out-dir results/label_product/baselines/dl_tuned --models all

For each model it concatenates losocv_<m>.partA-B.csv, hparams_<m>.partA-B.csv and
fold_probs/probs_<m>.partA-B.csv (sorted by fold), refuses to merge if two parts
contain the same fold, reports the folds covered, writes losocv_<m>.csv,
hparams_<m>.csv, fold_probs/probs_<m>.csv and summary_<m>.csv, and leaves the part
files in place.  --expected N warns when fewer than N folds are present.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

PART = re.compile(r"^losocv_(?P<model>.+?)\.part(?P<lo>\d+)-(?P<hi>\d+)\.csv$")


def merge_one(out_dir: Path, model: str, expected: int | None, force: bool) -> bool:
    parts = sorted(out_dir.glob(f"losocv_{model}.part*.csv"))
    if not parts:
        print(f"[merge] {model}: no part files in {out_dir}"); return False
    final = out_dir / f"losocv_{model}.csv"
    if final.exists() and not force:
        print(f"[merge] {model}: {final.name} already exists (use --force to overwrite)"); return False

    def cat(paths):
        frames = [pd.read_csv(p) for p in paths if p.exists() and p.stat().st_size > 0]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    df = cat(parts)
    if df.empty:
        print(f"[merge] {model}: parts are empty"); return False
    dup = df["fold"][df["fold"].duplicated()].tolist()
    if dup:
        raise SystemExit(f"[merge] {model}: fold(s) {sorted(set(dup))} appear in more than one part; fix the ranges first")
    df = df.sort_values("fold").reset_index(drop=True)
    suffixes = [PART.match(p.name).group(0).split(".part")[1][:-4] for p in parts]
    hp = cat([out_dir / f"hparams_{model}.part{s}.csv" for s in suffixes])
    pr = cat([out_dir / "fold_probs" / f"probs_{model}.part{s}.csv" for s in suffixes])
    if not hp.empty: hp = hp.sort_values("fold").reset_index(drop=True)
    if not pr.empty: pr = pr.sort_values(["fold"]).reset_index(drop=True)

    df.to_csv(final, index=False)
    if not hp.empty: hp.to_csv(out_dir / f"hparams_{model}.csv", index=False)
    if not pr.empty: pr.to_csv(out_dir / "fold_probs" / f"probs_{model}.csv", index=False)
    means = {k: float(np.nanmean(df[k])) for k in ["balanced_acc", "f1", "roc_auc", "mcc", "accuracy", "ece",
                                                    "balanced_acc_cal", "mcc_cal", "accuracy_cal"] if k in df}
    means.update(model=model, n_folds=len(df), seconds=float("nan"), regime="tuned")
    pd.DataFrame([means]).to_csv(out_dir / f"summary_{model}.csv", index=False)
    folds = df["fold"].tolist()
    msg = f"[merge] {model}: {len(parts)} parts -> {len(df)} folds ({folds[0]}..{folds[-1]}) " \
          f"bal={means['balanced_acc']:.3f} auc={means['roc_auc']:.3f} mcc={means['mcc']:.3f}"
    if expected and len(df) < expected:
        msg += f"   !! only {len(df)} of {expected} evaluable folds"
    print(msg)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--expected", type=int, default=None, help="number of evaluable folds (37 on the index, 41-42 on the product label)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    models = a.models
    if models == ["all"]:
        models = sorted({PART.match(p.name).group("model") for p in out_dir.glob("losocv_*.part*.csv") if PART.match(p.name)})
    if not models:
        raise SystemExit(f"[merge] no part files under {out_dir}")
    for m in models:
        merge_one(out_dir, m, a.expected, a.force)


if __name__ == "__main__":
    main()
