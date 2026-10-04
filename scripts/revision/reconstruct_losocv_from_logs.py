# -*- coding: utf-8 -*-
"""
Reconstruct the per-fold LOSOCV result CSVs that exist only as Brev training logs.

The engagement-index late fusions (EEGNet+ET-LSTM, ShallowConvNet+ET-LSTM; round-2
run r1) and the fold-wise-label baselines (EEGNet, ShallowConvNet, ET-LSTM; run r2)
were trained on the Brev instance, whose window has closed; the per-fold result CSVs
were never downloaded, but the sliced training logs in logs/revision/ print one line
per completed fold:

  fold 07  S08 bal=0.467 auc=0.783 mcc=-0.098 | val_bal=1.00 cfg={...} ep=1

This script parses those lines into CSVs restricted to the columns the logs carry
(fold, test_subject, balanced_acc, roc_auc, mcc, val_balanced_acc, hyperparameters,
best_epoch) and REFUSES to write unless the reconstructed means reproduce the
published values (Table 4 and Table S17 of the manuscript) within rounding.
Epoch-level probabilities (and therefore ECE/Brier and pooled curves) are not in
the logs and are not reconstructable; see results/statistics/STATS_ENVIRONMENT.md.

Run:  python scripts/revision/reconstruct_losocv_from_logs.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs" / "revision"

LINE = re.compile(
    r"^\s*fold\s+(\d+)\s+(S\d+)\s+bal=([\d.+-]+)\s+auc=([\d.+-]+)\s+mcc=([\d.+-]+)"
    r"\s*\|\s*val_bal=([\d.+-]+)\s+cfg=\{(.*?)\}\s+ep=(\d+)",
    re.M,
)
CFG_KEYS = ("lr", "wd", "batch_size", "drop")

# (output csv, slice-log stem, published mean (balanced_acc, roc_auc, mcc), source table)
JOBS = [
    (ROOT / "results/baselines/dl_tuned/losocv_eegnet_et.csv",
     "round2_r1_eegnet_et", (0.73, 0.81, 0.46), "Table 4"),
    (ROOT / "results/baselines/dl_tuned/losocv_shallow_et.csv",
     "round2_r1_shallow_et", (0.72, 0.83, 0.43), "Table 4"),
    (ROOT / "results/baselines/dl_tuned_v2/losocv_eegnet.csv",
     "round2_r2_eegnet", (0.529, 0.538, 0.060), "Table S17 (fold-wise)"),
    (ROOT / "results/baselines/dl_tuned_v2/losocv_shallow.csv",
     "round2_r2_shallow", (0.600, 0.644, 0.164), "Table S17 (fold-wise)"),
    (ROOT / "results/baselines/dl_tuned_v2/losocv_et_lstm.csv",
     "round2_r2_et_lstm", (0.738, 0.859, 0.497), "Table S17 (fold-wise)"),
]


def parse_cfg(cfg: str) -> dict:
    out = {}
    for key in CFG_KEYS:
        m = re.search(rf"'{key}':\s*([^,}}]+)", cfg)
        v = m.group(1).strip() if m else ""
        out[key] = "" if v == "None" else v
    return out


def reconstruct(stem: str) -> pd.DataFrame:
    rows = {}
    logs = sorted(LOGS.glob(f"{stem}_*.log"))        # sliced logs only
    if not logs:
        raise SystemExit(f"no sliced logs for {stem}")
    for lg in logs:
        for m in LINE.finditer(lg.read_text(encoding="utf-8", errors="replace")):
            fold, subj = int(m.group(1)), m.group(2)
            cfg = parse_cfg(m.group(7))
            rows[subj] = {                            # keep the last occurrence per subject
                "fold": fold, "test_subject": subj,
                "balanced_acc": float(m.group(3)),
                "roc_auc": float(m.group(4)),
                "mcc": float(m.group(5)),
                "val_balanced_acc": float(m.group(6)),
                **cfg,
                "best_epoch": int(m.group(8)),
                "reconstructed_from_log": lg.name,
            }
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r["fold"])).reset_index(drop=True)


def main() -> None:
    staged = []
    for out_csv, stem, pub, src in JOBS:
        df = reconstruct(stem)
        got = (df.balanced_acc.mean(), df.roc_auc.mean(), df.mcc.mean())
        ok = len(df) == 37 and all(abs(g - p) <= 0.006 for g, p in zip(got, pub))
        print(f"{stem}: {len(df)} folds | bal/auc/mcc = "
              f"{got[0]:.3f}/{got[1]:.3f}/{got[2]:.3f} vs {src} {pub} -> "
              + ("OK" if ok else "MISMATCH"))
        if not ok:
            raise SystemExit(f"reconstruction of {stem} does not reproduce {src}; nothing written")
        staged.append((out_csv, df))
    for out_csv, df in staged:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"wrote {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
