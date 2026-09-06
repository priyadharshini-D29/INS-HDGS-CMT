#!/usr/bin/env python3
"""
=======================================================================
NEUMA — POSITIVE-CONTROL EPOCHS: BROCHURE BROWSING vs RESTING STATE
=======================================================================
The engagement and purchase labels turned out not to be decodable from
5-s EEG.  Before concluding anything about the EEG branch, the pipeline
must be shown to decode a state contrast that is *known* to be present in
these recordings.  NeuMa recorded two minutes of resting-state EEG
(marker `fixation_cross`) immediately before the brochure presentation
(Neuma_RawDataset_Info.pdf, "Experimental Protocol").

Label (per 5-s epoch, same shape as every other track: 1500 EEG samples
at 300 Hz, 600 gaze samples at 120 Hz)
    0 = resting state (fixation cross, before the first brochure page)
    1 = active browsing (inside a brochure-page view)

Design choices that keep the control honest
  * classes are balanced within subject: n = min(#rest epochs, #browsing
    epochs, --max-per-class); the browsing epochs are the EARLIEST ones in
    the session, so the two classes are as close in time as the protocol
    allows (limits slow-drift / electrode-impedance confounds);
  * the first --skip-s seconds after the fixation-cross marker and after
    every page onset are discarded (marker / page-flip transients);
  * epochs never overlap and never cross a page boundary;
  * the gaze channels trivially separate the classes (fixed gaze at the
    cross), so the informative run is the gaze-free EEG variant
    (`eeg_only_mmd`); `full` is expected to be ~1.0 and only confirms the
    loader.

Outputs (per subject, next to the Phase-3 outputs)
  output/epochs/eeg_epochs_control.npy, et_epochs_control.npy   (object arrays)
  output/engagement_control/engagement_labels.npy               1 = browsing
  output/engagement_control/engagement_scores.npy               epoch onset, s from the fixation cross
  output/engagement_control/engagement_metadata.csv
Global: output/engagement_control/summary.csv

Usage
  cd src/data_pipeline/04_segmentation
  python control_epoching.py [--subjects S01 S02] [--max-per-class 24] [--skip-s 5] [--dry-run]
Dataset loader: NEUMA_LABEL_SOURCE=control
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
os.environ.setdefault("NEUMA_SUBJECT", "S01")
os.environ.setdefault("NEUMA_SKIP_PLOTS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from product_epoching import page_views, cut, EEG_SR, ET_SR, EPOCH_S, N_EEG, N_ET   # noqa: E402

OUT_SUB = "engagement_control"
MIN_REST_S = 30.0                                       # subjects with a shorter usable rest segment are skipped


def rest_window(markers_csv: Path, t_first: float):
    """(start, end) of the resting-state segment: fixation_cross marker -> first page marker."""
    m = pd.read_csv(markers_csv)
    fx = m.loc[m["label"].astype(str).str.strip() == "fixation_cross", "timestamp"]
    pages = m.loc[m["label"].astype(str).str.contains("FYLLADIO_"), "timestamp"]
    if pages.empty:
        return None
    t_page = float(pages.min())
    t0 = float(fx[fx < t_page].max()) if (fx < t_page).any() else float(t_first)
    return (t0, t_page)


def grid(t0: float, t1: float, skip_s: float):
    """non-overlapping EPOCH_S windows inside [t0 + skip_s, t1]."""
    s = t0 + skip_s
    out = []
    while s + EPOCH_S <= t1:
        out.append((s, s + EPOCH_S)); s += EPOCH_S
    return out


def build_subject(sid: str, max_per_class: int, skip_s: float, dry_run: bool):
    from loaders.load_clean_data import load_clean_data
    from segmentation.synchronize_modalities import trim_to_overlap
    sub = HERE / sid
    markers_csv = sub / "output" / "events" / "markers.csv"
    if not markers_csv.exists():
        print(f"  [SKIP] {sid}: no markers.csv (run main_phase3.py first)"); return None
    eeg, et, eeg_ts, et_ts = load_clean_data(sid)
    eeg, eeg_ts, et, et_ts = trim_to_overlap(eeg, eeg_ts, et, et_ts)
    t_first = float(max(eeg_ts[0], et_ts[0]))
    rw = rest_window(markers_csv, t_first)
    if rw is None:
        print(f"  [SKIP] {sid}: no page markers"); return None
    r0, r1 = max(rw[0], t_first), rw[1]
    if r1 - r0 < MIN_REST_S:
        print(f"  [SKIP] {sid}: resting segment only {r1 - r0:.1f} s"); return None

    cand = [dict(condition="rest", label=0, page=0, t_start=a, t_end=b) for a, b in grid(r0, r1, skip_s)]
    browse = []
    for (v0, v1, page) in sorted(page_views(markers_csv)):
        browse += [dict(condition="browse", label=1, page=page, t_start=a, t_end=b) for a, b in grid(v0, v1, skip_s)]
    browse.sort(key=lambda d: d["t_start"])                     # earliest browsing first (closest in time to rest)

    def realise(items):
        out = []
        for d in items:
            e_ep, t_ep = cut(eeg, eeg_ts, d["t_start"], N_EEG), cut(et, et_ts, d["t_start"], N_ET)
            if e_ep is None or t_ep is None or not np.isfinite(e_ep).all():
                continue
            out.append((d, e_ep, t_ep))
        return out

    rest_ep, browse_ep = realise(cand), realise(browse)
    n = min(len(rest_ep), len(browse_ep), max_per_class)
    if n < 6:
        print(f"  [SKIP] {sid}: only {len(rest_ep)} rest / {len(browse_ep)} browsing epochs"); return None
    # rest: evenly spaced over the segment; browsing: the earliest n
    ridx = np.linspace(0, len(rest_ep) - 1, n).round().astype(int)
    chosen = [rest_ep[i] for i in ridx] + browse_ep[:n]
    rows, eeg_eps, et_eps = [], [], []
    for i, (d, e_ep, t_ep) in enumerate(chosen):
        rows.append(dict(subject_id=sid, epoch_idx=i, **d, onset_from_fixation_s=round(d["t_start"] - rw[0], 2)))
        eeg_eps.append(e_ep); et_eps.append(t_ep)
    df = pd.DataFrame(rows)
    print(f"  {sid}: rest {r1 - r0:.0f} s -> {n} rest + {n} browsing epochs "
          f"(browsing epochs span {df[df.label == 1].onset_from_fixation_s.min():.0f}-{df[df.label == 1].onset_from_fixation_s.max():.0f} s after the cross)")
    if not dry_run:
        out = sub / "output" / OUT_SUB; out.mkdir(parents=True, exist_ok=True)
        np.save(out / "engagement_labels.npy", df["label"].to_numpy(np.int64))
        np.save(out / "engagement_scores.npy", df["onset_from_fixation_s"].to_numpy(np.float32))
        df.to_csv(out / "engagement_metadata.csv", index=False)
        ep = sub / "output" / "epochs"; ep.mkdir(parents=True, exist_ok=True)
        np.save(ep / "eeg_epochs_control.npy", np.array(eeg_eps, dtype=object))
        np.save(ep / "et_epochs_control.npy", np.array(et_eps, dtype=object))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--max-per-class", type=int, default=24)
    ap.add_argument("--skip-s", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    subs = args.subjects or sorted(d.name for d in HERE.iterdir() if d.is_dir() and re.fullmatch(r"S\d+", d.name))
    tables = [df for s in subs for df in [build_subject(s, args.max_per_class, args.skip_s, args.dry_run)] if df is not None]
    if not tables:
        raise SystemExit("no subject produced control epochs")
    T = pd.concat(tables, ignore_index=True)
    S = T.groupby("subject_id").agg(epochs=("label", "size"), browsing=("label", "sum")).reset_index()
    print("\n=== positive-control epochs (browsing vs rest) ===")
    print(f"  epochs {len(T)}  browsing {int(T.label.sum())} ({T.label.mean():.3f})  subjects {len(S)}; "
          f"epochs/subject median {S.epochs.median():.0f} (min {S.epochs.min()}, max {S.epochs.max()})")
    if not args.dry_run:
        g = HERE / "output" / OUT_SUB; g.mkdir(parents=True, exist_ok=True)
        S.to_csv(g / "summary.csv", index=False); T.to_csv(g / "control_epochs.csv", index=False)
        print(f"  written: {g}/summary.csv, control_epochs.csv and per-subject S*/output/{OUT_SUB}/")


if __name__ == "__main__":
    main()
