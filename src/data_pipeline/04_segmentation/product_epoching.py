#!/usr/bin/env python3
"""
=======================================================================
NEUMA — PRODUCT-LEVEL EPOCHS WITH PURCHASE LABELS (NeuMa-native design)
=======================================================================
One epoch per (participant, product) that the participant actually looked
at, labelled by whether that product was selected for purchase (Q77 in
DataSource/S##.xlsx, validated against the mouse clicks 752/754).

Epoch construction (fixed 5 s so the model is unchanged: 1500 EEG samples
at 300 Hz, 600 gaze samples at 120 Hz)
  1. For every brochure-page view, gaze samples (left eye, normalised
     screen coordinates) are assigned to the 24 product boxes of that page
     (Dependencies/BoundingBox_Coordinates; screen 1920x1080 -> page
     3000x1688).
  2. Per product, contiguous on-product runs are formed (gaps <= 100 ms
     bridged); runs shorter than --min-run (0.3 s) are discarded.  Products
     with total dwell < --min-dwell (1.0 s) are skipped.
  3. The epoch is anchored 1 s before the onset of the product's longest
     eligible run: window = [onset - 1 s, onset + 4 s], kept inside the page
     view.  Anchoring is identical for both classes; for bought products
     only runs whose window ends >= 250 ms before the first click on that
     product are eligible (otherwise the next-longest run is tried), so no
     click / motor execution is inside the epoch: the label is predicted
     from activity that precedes the purchase decision.
  4. Products for which no 5-s window satisfies the constraints are skipped.

Outputs (per subject, next to the Phase-3 outputs)
  output/epochs/eeg_epochs_product.npy, et_epochs_product.npy   (object arrays)
  output/engagement_product/engagement_labels.npy               1 = bought
  output/engagement_product/engagement_scores.npy               total dwell (s)
  output/engagement_product/engagement_metadata.csv
Global: output/engagement_product/summary.csv, product_epochs.csv

Usage
  cd src/data_pipeline/04_segmentation
  python product_epoching.py [--subjects S01 S02] [--min-dwell 1.0] [--min-run 0.3] [--dry-run]
Dataset loader: NEUMA_LABEL_SOURCE=product
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
os.environ.setdefault("NEUMA_SUBJECT", "S01")          # config.settings reads it at import; overridden per call
os.environ.setdefault("NEUMA_SKIP_PLOTS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "DataSource"
sys.path.insert(0, str(HERE))

from purchase_labeling import (bought_products, load_boxes, load_descriptions,   # noqa: E402
                               products_per_sample, N_SLOTS, SCREEN_W, SCREEN_H, PAGE_W, PAGE_H)

EEG_SR, ET_SR = 300, 120
EPOCH_S, PRE_S = 5.0, 1.0
N_EEG, N_ET = int(EPOCH_S * EEG_SR), int(EPOCH_S * ET_SR)
GAP_SAMPLES = int(0.100 * ET_SR)                       # bridge gaps <= 100 ms
CLICK_MARGIN_S = 0.25                                  # bought products: window ends >= 250 ms before the click
OUT_SUB = "engagement_product"


def page_views(markers_csv: Path):
    m = pd.read_csv(markers_csv)
    views = []
    for t, lbl in zip(m["timestamp"], m["label"].astype(str)):
        if "FYLLADIO_" in lbl:
            views.append([float(t), None, int(re.search(r"FYLLADIO_(\d)", lbl).group(1))])
        elif views and views[-1][1] is None:
            views[-1][1] = float(t)
    for i in range(len(views) - 1):
        if views[i][1] is None or views[i][1] > views[i + 1][0]:
            views[i][1] = views[i + 1][0]
    return [(a, b, p) for a, b, p in views if b is not None and b > a]


def product_clicks(xdf_path: Path, views, boxes) -> dict[int, float]:
    """first click time per product code (1..144), from MouseButtons + MousePosition."""
    import pyxdf
    streams, _ = pyxdf.load_xdf(str(xdf_path), verbose=False)
    st = {s["info"]["name"][0]: s for s in streams}
    if "MouseButtons" not in st or "MousePosition" not in st:
        return {}
    bts = np.asarray(st["MouseButtons"]["time_stamps"]); bv = [str(v[0]) for v in st["MouseButtons"]["time_series"]]
    pts = np.asarray(st["MousePosition"]["time_stamps"]); pxy = np.asarray(st["MousePosition"]["time_series"], float)
    out = {}
    for t, v in zip(bts, bv):
        if "pressed" not in v:
            continue
        page = next((p for a, b, p in views if a <= t < b), 0)
        if page == 0:
            continue
        i = int(np.argmin(np.abs(pts - t)))
        x, y = pxy[i, 0] / SCREEN_W, pxy[i, 1] / SCREEN_H          # to normalised screen coords
        slot = int(products_per_sample(page, np.array([x]), np.array([y]), boxes)[0])
        if slot:
            code = (page - 1) * N_SLOTS + slot
            out.setdefault(code, float(t))
    return out


def runs_on_product(on: np.ndarray, ts: np.ndarray, min_run_s: float):
    """contiguous True runs in `on` (gaps <= GAP_SAMPLES bridged) -> list of (t_start, t_end)."""
    if not on.any():
        return []
    idx = np.where(on)[0]
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > GAP_SAMPLES + 1:
            runs.append((start, prev)); start = i
        prev = i
    runs.append((start, prev))
    out = []
    for a, b in runs:
        if ts[b] - ts[a] >= min_run_s:
            out.append((float(ts[a]), float(ts[b])))
    return out


def cut(arr, ts, t0, n):
    i0 = int(np.searchsorted(ts, t0, side="left"))
    seg = arr[i0:i0 + n]
    return seg if len(seg) == n else None


def build_subject(sid: str, boxes, desc, min_dwell, min_run, dry_run):
    from loaders.load_clean_data import load_clean_data
    from segmentation.synchronize_modalities import trim_to_overlap
    sub = HERE / sid
    markers_csv = sub / "output" / "events" / "markers.csv"
    if not markers_csv.exists():
        print(f"  [SKIP] {sid}: no markers.csv (run main_phase3.py first)"); return None
    eeg, et, eeg_ts, et_ts = load_clean_data(sid)
    eeg, eeg_ts, et, et_ts = trim_to_overlap(eeg, eeg_ts, et, et_ts)
    views = page_views(markers_csv)
    bought = bought_products(sid)
    clicks = product_clicks(DATA / f"{sid}.xdf", views, boxes)
    # gaze -> product per view
    per_product = {}                                   # code -> list of dict(run=(a,b), view=(t0,t1))
    for (t0, t1, page) in views:
        m = (et_ts >= t0) & (et_ts < t1)
        if m.sum() < 2:
            continue
        ts, x, y = et_ts[m], et[m, 0], et[m, 1]
        ok = np.isfinite(x) & np.isfinite(y)
        pr = np.zeros(len(ts), int)
        pr[ok] = products_per_sample(page, x[ok], y[ok], boxes)
        for slot in np.unique(pr[pr > 0]):
            code = (page - 1) * N_SLOTS + int(slot)
            for r in runs_on_product(pr == slot, ts, min_run):
                per_product.setdefault(code, []).append(dict(run=r, view=(t0, t1), page=page))
    rows, eeg_eps, et_eps = [], [], []
    n_skip_dwell = n_skip_window = 0
    for code, runs in sorted(per_product.items()):
        total = sum(b - a for d in runs for a, b in [d["run"]])
        if total < min_dwell:
            n_skip_dwell += 1; continue
        is_bought = code in bought
        t_click = clicks.get(code, np.inf) if is_bought else np.inf
        # eligible runs: start before the click (bought) -> longest first
        elig = sorted([d for d in runs if d["run"][0] < t_click], key=lambda d: d["run"][1] - d["run"][0], reverse=True)
        win = None
        for d in elig:
            a, (v0, v1) = d["run"][0], d["view"]
            start, end = a - PRE_S, a - PRE_S + EPOCH_S
            if start < v0:                                       # page just started: window begins at page onset (both classes)
                start, end = v0, v0 + EPOCH_S
            if end > v1:                                         # page view too short after the anchor: slide left to the page end
                end = v1; start = end - EPOCH_S
            if start < v0:
                continue
            if end > t_click - CLICK_MARGIN_S:                   # bought: the whole window must precede the click (no sliding)
                continue
            e_ep, t_ep = cut(eeg, eeg_ts, start, N_EEG), cut(et, et_ts, start, N_ET)
            if e_ep is None or t_ep is None:
                continue
            win = (start, end, d); break
        if win is None:
            n_skip_window += 1; continue
        start, end, d = win
        page, slot = d["page"], code - (d["page"] - 1) * N_SLOTS
        rows.append(dict(subject_id=sid, product=code, page=page, slot=slot, desc=desc[page - 1, slot - 1], label=int(is_bought),
                         total_dwell_s=round(total, 3), n_runs=len(runs), anchor_run_s=round(d["run"][1] - d["run"][0], 3),
                         n_views=len({d2["view"] for d2 in runs}), win_start=start, win_end=end,
                         click_after_window_s=(round(t_click - end, 3) if np.isfinite(t_click) else np.nan),
                         clicked=int(code in clicks)))
        eeg_eps.append(e_ep); et_eps.append(t_ep)
    df = pd.DataFrame(rows)
    print(f"  {sid}: {len(df)} product epochs (bought {int(df.label.sum()) if len(df) else 0}); "
          f"skipped dwell<{min_dwell}s: {n_skip_dwell}, no valid window: {n_skip_window}; clicks mapped {len(clicks)}, Q77 {len(bought)}")
    if not dry_run and len(df):
        out = sub / "output" / OUT_SUB; out.mkdir(parents=True, exist_ok=True)
        np.save(out / "engagement_labels.npy", df["label"].to_numpy(np.int64))
        np.save(out / "engagement_scores.npy", df["total_dwell_s"].to_numpy(np.float32))
        df.to_csv(out / "engagement_metadata.csv", index=False)
        ep = sub / "output" / "epochs"
        np.save(ep / "eeg_epochs_product.npy", np.array(eeg_eps, dtype=object))
        np.save(ep / "et_epochs_product.npy", np.array(et_eps, dtype=object))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--min-dwell", type=float, default=1.0)
    ap.add_argument("--min-run", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    boxes, desc = load_boxes(), load_descriptions()
    subs = args.subjects or sorted(d.name for d in HERE.iterdir() if d.is_dir() and re.fullmatch(r"S\d+", d.name))
    tables = []
    for s in subs:
        df = build_subject(s, boxes, desc, args.min_dwell, args.min_run, args.dry_run)
        if df is not None and len(df):
            tables.append(df)
    T = pd.concat(tables, ignore_index=True)
    S = T.groupby("subject_id").agg(epochs=("label", "size"), bought=("label", "sum")).reset_index()
    S["both_classes"] = (S.bought > 0) & (S.bought < S.epochs)
    print("\n=== product-level epochs ===")
    print(f"  epochs {len(T)}  bought {int(T.label.sum())} ({T.label.mean():.3f})  subjects {len(S)}  with both classes {int(S.both_classes.sum())}")
    print(f"  epochs/subject median {S.epochs.median():.0f} (min {S.epochs.min()}, max {S.epochs.max()}); "
          f"click after window end: median {T.click_after_window_s.median():.1f} s (bought only)")
    if not args.dry_run:
        g = HERE / "output" / OUT_SUB; g.mkdir(parents=True, exist_ok=True)
        S.to_csv(g / "summary.csv", index=False); T.to_csv(g / "product_epochs.csv", index=False)
        print(f"  written: {g}/summary.csv, product_epochs.csv and per-subject S*/output/{OUT_SUB}/")


if __name__ == "__main__":
    main()
