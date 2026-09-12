#!/usr/bin/env python
"""
================================================================
NEUMA Phase 8 vNext — Model Selection
================================================================
Ranks every subject-invariance experiment by  MCC → Balanced Acc → F1
(never Accuracy alone, per the vNext spec) and flags each against the
success criteria:

    improved  iff   MCC↑  AND  Balanced-Acc↑  AND
                    subject-probe-acc↓  AND  hard-subject-perf↑
                    (relative to the plain-model reference)

Reference = the unmodified plain model, preferred in this order:
    si_dann_l0p0  (λ_dann=0, no MMD, zscore, no pretrain)  →
    si_norm_zscore  →  ins_hdgs_cmt_v17.

Writes output/analysis/model_selection.csv and prints the ranked table.

Usage:
    python select_model.py
================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_ablations import (
    METRICS_DIR, HARD_SUBJECTS, _load, _losocv_means, _leakage, _csv_path,
)

OUT_DIR = Path("output/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PREFERENCE = ["si_dann_l0p0", "si_norm_zscore", "ins_hdgs_cmt_v17"]


def _hard_bal_acc(df: pd.DataFrame) -> float:
    if "test_subject" not in df.columns or "balanced_acc" not in df.columns:
        return float("nan")
    sub = df[df["test_subject"].isin(HARD_SUBJECTS)]
    return float(sub["balanced_acc"].mean()) if len(sub) else float("nan")


def _collect() -> pd.DataFrame:
    rows = []
    # all subject-invariance runs + any explicit baselines named above
    labels = {d.name for d in METRICS_DIR.glob("si_*") if _csv_path(d.name).exists()}
    labels |= {l for l in REFERENCE_PREFERENCE if _csv_path(l).exists()}
    for label in sorted(labels):
        df = _load(label)
        if df is None:
            continue
        r = {"experiment": label, "n_folds": int(len(df))}
        r.update(_losocv_means(df))
        r.update(_leakage(label))
        r["hard_bal_acc"] = _hard_bal_acc(df)
        rows.append(r)
    return pd.DataFrame(rows)


def _pick_reference(df: pd.DataFrame) -> str | None:
    present = set(df["experiment"])
    for cand in REFERENCE_PREFERENCE:
        if cand in present:
            return cand
    return None


def main():
    df = _collect()
    if df.empty:
        print("[select] no experiments found under output/metrics/si_*")
        return

    # Rank: MCC → Balanced Acc → F1 (all descending). Accuracy never primary.
    df = df.sort_values(["mcc", "balanced_acc", "f1"],
                        ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    ref_label = _pick_reference(df)
    if ref_label is None:
        print("[select] WARNING: no reference baseline present yet — "
              "success criteria left blank.")
        df["meets_criteria"] = pd.NA
    else:
        ref = df[df.experiment == ref_label].iloc[0]
        # probe-acc criterion only applies where leakage metrics exist for both.
        def _improved(r):
            ok_mcc  = r["mcc"]          > ref["mcc"]
            ok_bal  = r["balanced_acc"] > ref["balanced_acc"]
            ok_hard = (np.isnan(ref["hard_bal_acc"]) or np.isnan(r["hard_bal_acc"])
                       or r["hard_bal_acc"] > ref["hard_bal_acc"])
            ok_probe = (np.isnan(ref["subject_probe_acc"]) or np.isnan(r["subject_probe_acc"])
                        or r["subject_probe_acc"] < ref["subject_probe_acc"])
            return bool(ok_mcc and ok_bal and ok_hard and ok_probe)
        df["meets_criteria"] = df.apply(_improved, axis=1)
        df.loc[df.experiment == ref_label, "meets_criteria"] = False  # ref vs itself
        print(f"[select] reference baseline = {ref_label}  "
              f"(MCC={ref['mcc']:.4f}  BalAcc={ref['balanced_acc']:.4f}  "
              f"probe={ref['subject_probe_acc']}  hard={ref['hard_bal_acc']:.4f})")

    cols = ["rank", "experiment", "mcc", "balanced_acc", "f1", "accuracy",
            "roc_auc", "subject_probe_acc", "silhouette_subject",
            "hard_bal_acc", "meets_criteria", "n_folds"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols]

    path = OUT_DIR / "model_selection.csv"
    out.to_csv(path, index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(out.to_string(index=False,
                        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
    print(f"\n[select] written → {path}")
    if "meets_criteria" in out.columns:
        winners = out[out["meets_criteria"] == True]  # noqa: E712
        if len(winners):
            print(f"[select] {len(winners)} config(s) meet ALL success criteria; "
                  f"top: {winners.iloc[0]['experiment']}")
        else:
            print("[select] no config yet meets all four success criteria.")


if __name__ == "__main__":
    main()
