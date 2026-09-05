#!/usr/bin/env python3
"""
=======================================================================
NEUMA — BEHAVIOURAL (PURCHASE-INTENT) LABELS FOR THE 5-s STIMULUS EPOCHS
=======================================================================
Replaces the rule-based "engagement" index (a function of the model's own
inputs) with the participant's recorded purchase decision.

Ground truth
    In the NeuMa protocol every participant browsed 6 brochure pages of
    24 products each and clicked the products they intended to buy.  The
    selected products are listed per participant in DataSource/S##.xlsx
    (column Q77: "<product code><A|B>", code 1..144 = (page-1)*24 + slot).
    The mouse-click stream reproduces this list for 752/754 products
    (results/statistics/purchase_click_validation.csv), so Q77 is used as
    the label source and the clicks only as an audit.

Window label
    Each existing stimulus epoch is the first 5 s after a page onset
    (Phase 3, PRE_TIME=0, POST_TIME=5).  Gaze samples (left eye, normalised
    screen coordinates) are projected onto the page image
    (1920x1080 screen -> 3000x1688 page, uniform scale 1.5625) and assigned
    to the product bounding boxes of Dependencies/BoundingBox_Coordinates.
        dom_product          product that received the most gaze samples
        label                1 if dom_product was selected for purchase (Q77)
        gaze_on_bought_frac  fraction of on-product gaze on later-bought products
        click_in_window      (from the audit CSV, informational)
    Epochs with fewer than --min-valid gaze samples on any product are
    dropped (they carry no product information).

Outputs (per subject, next to the Phase-3 outputs)
    output/engagement_purchase/engagement_labels.npy       int   (N,)
    output/engagement_purchase/engagement_scores.npy       float (N,)  gaze_on_bought_frac
    output/engagement_purchase/engagement_metadata.csv     per-epoch table
    output/epochs/eeg_epochs_purchase.npy, et_epochs_purchase.npy
Global
    output/engagement_purchase/summary.csv, window_labels.csv

Usage
    cd src/data_pipeline/04_segmentation
    python purchase_labeling.py [--subjects S01 S02] [--dry-run] [--min-valid 60]
The dataset loader picks these files up with NEUMA_LABEL_SOURCE=purchase.
=======================================================================
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONUTF8", "1")
HERE = Path(__file__).resolve().parent                 # .../src/data_pipeline/04_segmentation
ROOT = HERE.parents[2]                                 # repo root (DataSource/ lives here)
DATA = ROOT / "DataSource"
DEP = DATA / "Dependencies"

SCREEN_W, SCREEN_H = 1920.0, 1080.0
PAGE_W, PAGE_H = 3000.0, 1688.0
N_PAGES, N_SLOTS = 6, 24
STIMULUS_PATTERN = "ImagePage"
OUT_SUB = "engagement_purchase"


def load_boxes() -> dict[int, np.ndarray]:
    import scipy.io as sio
    boxes = {}
    for k in range(1, N_PAGES + 1):
        m = sio.loadmat(DEP / "BoundingBox_Coordinates" / f"BoundingBoxPage_{k}.mat")["ROI_list"][0]
        boxes[k] = np.array([np.asarray(b, float).ravel() for b in m])   # (24, 4) = x, y, w, h in page pixels
        assert boxes[k].shape == (N_SLOTS, 4), boxes[k].shape
    return boxes


def load_descriptions() -> np.ndarray:
    import scipy.io as sio
    d = sio.loadmat(DEP / "Leaflet_Product_Descriptions.mat")["Product_Descriptions"]   # (6, 24) object
    return np.vectorize(lambda x: str(np.asarray(x).ravel()[0]) if np.asarray(x).size else "")(d)


def bought_products(subject: str) -> set[int]:
    x = pd.ExcelFile(DATA / f"{subject}.xlsx")
    d = x.parse(x.sheet_names[0])
    cols = {str(c).strip(): c for c in d.columns}
    if "Q77" not in cols:
        raise KeyError(f"{subject}: no Q77 column in the questionnaire")
    codes = set()
    for v in d[cols["Q77"]].dropna().astype(str).str.strip():
        m = re.match(r"(\d+)", v)
        if m:
            codes.add(int(m.group(1)))
    return codes


def products_per_sample(page: int, x: np.ndarray, y: np.ndarray, boxes) -> np.ndarray:
    """normalised screen coords -> product slot 1..24 per sample (0 = not on a product)."""
    px, py = x * SCREEN_W * (PAGE_W / SCREEN_W), y * SCREEN_H * (PAGE_H / SCREEN_H)
    b = boxes[page]
    out = np.zeros(len(x), int)
    for i in range(N_SLOTS):
        m = (px >= b[i, 0]) & (px <= b[i, 0] + b[i, 2]) & (py >= b[i, 1]) & (py <= b[i, 1] + b[i, 3])
        out[m & (out == 0)] = i + 1
    return out


def label_subject(sub_dir: Path, boxes, desc, min_valid: int, click_audit: pd.DataFrame | None):
    sid = sub_dir.name
    ep_dir = sub_dir / "output" / "epochs"
    meta_path = sub_dir / "output" / "metadata" / "metadata.csv"
    if not (ep_dir / "eeg_epochs.npy").exists() or not meta_path.exists():
        return None
    eeg = np.load(ep_dir / "eeg_epochs.npy", allow_pickle=True)
    et = np.load(ep_dir / "et_epochs.npy", allow_pickle=True)
    n = min(len(eeg), len(et))
    meta = pd.read_csv(meta_path).iloc[:n].reset_index(drop=True)
    stim = np.where(meta["label"].astype(str).str.startswith(STIMULUS_PATTERN))[0]
    bought = bought_products(sid)
    rows, keep_eeg, keep_et = [], [], []
    for j, i in enumerate(stim):                       # j = index among stimulus epochs (matches phase3d)
        lbl = str(meta["label"][i])
        page = int(re.search(r"_(\d+)", lbl).group(1))
        e = np.asarray(et[i], float)
        x, y = e[:, 0], e[:, 1]
        ok = np.isfinite(x) & np.isfinite(y)
        pr = products_per_sample(page, x[ok], y[ok], boxes)
        on = pr[pr > 0]
        if len(on) < min_valid:
            rows.append(dict(subject_id=sid, epoch_idx=j, orig_epoch=int(i), page=page, kept=0, n_on_product=int(len(on))))
            continue
        vals, cnt = np.unique(on, return_counts=True)
        dom_slot = int(vals[np.argmax(cnt)])
        dom_code = (page - 1) * N_SLOTS + dom_slot
        codes = (page - 1) * N_SLOTS + on
        rows.append(dict(subject_id=sid, epoch_idx=j, orig_epoch=int(i), page=page, kept=1,
                         n_on_product=int(len(on)), dom_product=dom_code, dom_slot=dom_slot,
                         dom_desc=desc[page - 1, dom_slot - 1], dom_frac=float(cnt.max() / len(pr)),
                         gaze_on_bought_frac=float(np.isin(codes, list(bought)).mean()),
                         n_products_seen=int(len(vals)), label=int(dom_code in bought)))
        keep_eeg.append(eeg[i]); keep_et.append(et[i])
    df = pd.DataFrame(rows)
    if click_audit is not None and sid in set(click_audit.subject):
        pass  # informational only; the click audit is a separate CSV
    kept = df[df.kept == 1].reset_index(drop=True)
    return dict(sid=sid, table=df, kept=kept, eeg=keep_eeg, et=keep_et, n_bought_products=len(bought))


def save_subject(sub_dir: Path, res):
    out = sub_dir / "output" / OUT_SUB
    out.mkdir(parents=True, exist_ok=True)
    kept = res["kept"]
    np.save(out / "engagement_labels.npy", kept["label"].to_numpy(np.int64))
    np.save(out / "engagement_scores.npy", kept["gaze_on_bought_frac"].to_numpy(np.float32))
    res["table"].to_csv(out / "engagement_metadata.csv", index=False)
    ep_dir = sub_dir / "output" / "epochs"
    np.save(ep_dir / "eeg_epochs_purchase.npy", np.array(res["eeg"], dtype=object))
    np.save(ep_dir / "et_epochs_purchase.npy", np.array(res["et"], dtype=object))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--min-valid", type=int, default=60, help="min gaze samples on any product (60 = 0.5 s at 120 Hz)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    boxes, desc = load_boxes(), load_descriptions()
    audit_csv = ROOT / "results" / "statistics" / "purchase_click_validation.csv"
    audit = pd.read_csv(audit_csv) if audit_csv.exists() else None
    subs = args.subjects or sorted(d.name for d in HERE.iterdir() if d.is_dir() and re.fullmatch(r"S\d+", d.name))
    tables, summary = [], []
    for s in subs:
        res = label_subject(HERE / s, boxes, desc, args.min_valid, audit)
        if res is None:
            print(f"  [SKIP] {s}: no Phase-3 epochs"); continue
        k = res["kept"]
        summary.append(dict(subject=s, stimulus_epochs=len(res["table"]), kept=len(k), bought=int(k.label.sum()) if len(k) else 0,
                            not_bought=int((k.label == 0).sum()) if len(k) else 0, products_bought=res["n_bought_products"],
                            both_classes=int(len(k) > 0 and 0 < k.label.sum() < len(k))))
        tables.append(res["table"])
        print(f"  {s}: {len(k)}/{len(res['table'])} epochs kept, bought {int(k.label.sum()) if len(k) else 0}")
        if not args.dry_run:
            save_subject(HERE / s, res)
    S = pd.DataFrame(summary); T = pd.concat(tables, ignore_index=True)
    print("\n=== purchase-intent window labels ===")
    print(f"  epochs kept {S.kept.sum()} / {S.stimulus_epochs.sum()}   bought {S.bought.sum()} ({S.bought.sum() / max(S.kept.sum(), 1):.3f})"
          f"   subjects with both classes {S.both_classes.sum()} / {len(S)}")
    if not args.dry_run:
        g = HERE / "output" / OUT_SUB
        g.mkdir(parents=True, exist_ok=True)
        S.to_csv(g / "summary.csv", index=False); T.to_csv(g / "window_labels.csv", index=False)
        print(f"  written: {g}/summary.csv, window_labels.csv and per-subject S*/output/{OUT_SUB}/")


if __name__ == "__main__":
    main()
