#!/usr/bin/env python
"""
================================================================
NEUMA Phase 8 vNext — Ablation Aggregator
================================================================
Collects the per-config LOSOCV result CSVs from each subject-invariance family
and emits the named ablation tables required by the vNext spec:

    output/analysis/normalization_ablation.csv   (Phase 1)
    output/analysis/dann_ablation.csv            (Phase 2)
    output/analysis/mmd_ablation.csv             (Phase 3)
    output/analysis/contrastive_ablation.csv     (Phase 4)
    output/analysis/hard_subject_recovery.csv    (Phase 5)

LOSOCV metrics are read straight from each ``losocv_<label>.csv`` (mean over
the held-out-subject folds — one fold == one subject).  Leakage metrics
(subject-probe accuracy, silhouettes — Phase 6) are joined from
``output/subject_invariance/<label>/embedding_validation.json`` when present
(produced separately by subject_invariance_eval.py).  Missing runs are skipped
gracefully so this can be run at any point during the sweeps.

Usage:
    python aggregate_ablations.py
================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

METRICS_DIR  = Path("output/metrics")
SI_DIR       = Path("output/subject_invariance")
OUT_DIR      = Path("output/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HARD_SUBJECTS = ["S21", "S03", "S13", "S35", "S36"]

# Mean over folds of these per-fold metric columns (already in each CSV).
_METRIC_COLS = ["accuracy", "balanced_acc", "f1", "mcc", "roc_auc"]


def _csv_path(label: str) -> Path:
    return METRICS_DIR / label / f"losocv_{label}.csv"


def _load(label: str):
    p = _csv_path(label)
    return pd.read_csv(p) if p.exists() else None


def _losocv_means(df: pd.DataFrame) -> dict:
    return {c: float(df[c].mean()) if c in df.columns else float("nan")
            for c in _METRIC_COLS}


def _leakage(label: str) -> dict:
    """subject_probe_acc / silhouettes from embedding_validation.json if present."""
    f = SI_DIR / label / "embedding_validation.json"
    keys = ("subject_probe_acc", "silhouette_subject", "silhouette_label")
    if not f.exists():
        return {k: float("nan") for k in keys}
    ev = json.loads(f.read_text())
    return {k: (ev.get(k) if ev.get(k) is not None else float("nan")) for k in keys}


def _row(label: str, df: pd.DataFrame) -> dict:
    r = {"label": label, "n_folds": int(len(df))}
    r.update(_losocv_means(df))
    r.update(_leakage(label))
    return r


def _discover(prefix: str) -> list[str]:
    """Labels with a results CSV whose name starts with `prefix`."""
    out = []
    for d in sorted(METRICS_DIR.glob(f"{prefix}*")):
        if _csv_path(d.name).exists():
            out.append(d.name)
    return out


def _write(df: pd.DataFrame, name: str):
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(f"  ✓ {name:32s} ({len(df)} rows) → {path}")


# ── Phase 1: normalization ──────────────────────────────────────────────────

def normalization_table():
    rows = []
    for label in _discover("si_norm_"):
        df = _load(label)
        r = _row(label, df)
        r["method"] = label.replace("si_norm_", "")
        rows.append(r)
    if not rows:
        return
    out = pd.DataFrame(rows).rename(columns={"balanced_acc": "balanced_accuracy"})
    out = out[["method", "accuracy", "f1", "balanced_accuracy", "mcc",
               "roc_auc", "subject_probe_acc", "silhouette_subject",
               "silhouette_label", "n_folds"]]
    _write(out.sort_values("mcc", ascending=False), "normalization_ablation.csv")


# ── Phase 2: DANN ───────────────────────────────────────────────────────────

def dann_table():
    rows = []
    for label in _discover("si_dann_"):
        df = _load(label)
        r = _row(label, df)
        r["lambda_dann"] = (float(df["lambda_dann"].iloc[0])
                            if "lambda_dann" in df.columns else float("nan"))
        rows.append(r)
    if not rows:
        return
    out = pd.DataFrame(rows).rename(columns={"balanced_acc": "balanced_accuracy"})
    out = out[["lambda_dann", "accuracy", "balanced_accuracy", "f1", "mcc",
               "roc_auc", "subject_probe_acc", "silhouette_subject",
               "silhouette_label", "n_folds"]]
    _write(out.sort_values("lambda_dann"), "dann_ablation.csv")


# ── Phase 3: MMD ────────────────────────────────────────────────────────────

def mmd_table():
    rows = []
    for label in _discover("si_mmd_"):
        df = _load(label)
        r = _row(label, df)
        r["lambda_mmd"] = (float(df["lambda_mmd"].iloc[0])
                           if "lambda_mmd" in df.columns else float("nan"))
        r["mmd_mode"] = (df["mmd_mode"].iloc[0] if "mmd_mode" in df.columns
                         else ("class_conditional" if "_cc_" in label else "marginal"))
        rows.append(r)
    if not rows:
        return
    out = pd.DataFrame(rows).rename(columns={"balanced_acc": "balanced_accuracy"})
    out = out[["lambda_mmd", "mmd_mode", "accuracy", "balanced_accuracy", "f1",
               "mcc", "roc_auc", "subject_probe_acc", "silhouette_subject",
               "silhouette_label", "n_folds"]]
    _write(out.sort_values(["mmd_mode", "lambda_mmd"]), "mmd_ablation.csv")


# ── Phase 4: contrastive ────────────────────────────────────────────────────

def contrastive_table():
    mapping = {"si_contrastive_pretrained": "with_pretraining",
               "si_contrastive_baseline":   "without_pretraining"}
    rows = []
    for label, tag in mapping.items():
        df = _load(label)
        if df is None:
            continue
        r = _row(label, df)
        r["pretraining"] = tag
        rows.append(r)
    if not rows:
        return
    out = pd.DataFrame(rows).rename(columns={"balanced_acc": "balanced_accuracy"})
    out = out[["pretraining", "accuracy", "balanced_accuracy", "f1", "mcc",
               "roc_auc", "subject_probe_acc", "silhouette_subject",
               "silhouette_label", "n_folds"]]
    _write(out, "contrastive_ablation.csv")


# ── Phase 5: hard-subject recovery (per subject, per experiment) ─────────────

def hard_subject_table():
    rename = {"test_subject": "subject", "balanced_acc": "balanced_accuracy",
              "roc_auc": "auc"}
    want = ["accuracy", "balanced_acc", "f1", "mcc", "roc_auc"]
    blocks = []
    for d in sorted(METRICS_DIR.glob("si_*")):
        label = d.name
        df = _load(label)
        if df is None or "test_subject" not in df.columns:
            continue
        sub = df[df["test_subject"].isin(HARD_SUBJECTS)].copy()
        if sub.empty:
            continue
        keep = ["test_subject"] + [c for c in want if c in sub.columns]
        sub = sub[keep].rename(columns=rename)
        sub.insert(0, "experiment", label)
        # stable hard-subject ordering
        order = {s: i for i, s in enumerate(HARD_SUBJECTS)}
        sub = sub.sort_values("subject", key=lambda s: s.map(order))
        blocks.append(sub)
    if not blocks:
        return
    out = pd.concat(blocks, ignore_index=True)
    _write(out, "hard_subject_recovery.csv")


def main():
    print("[aggregate] writing ablation tables to", OUT_DIR)
    normalization_table()
    dann_table()
    mmd_table()
    contrastive_table()
    hard_subject_table()
    print("[aggregate] done.")


if __name__ == "__main__":
    main()
