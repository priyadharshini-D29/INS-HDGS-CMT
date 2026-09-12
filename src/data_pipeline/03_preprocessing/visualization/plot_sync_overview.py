"""
plot_sync_overview.py
Generates three publication-quality figures showing:
  Fig 1 — EEG: raw vs preprocessed (4 representative channels)
  Fig 2 — ET: raw vs preprocessed (gaze X/Y + pupil)
  Fig 3 — Timestamp synchronisation (stream density + zoom + dual-axis overlay)

Usage:
    python3 visualization/plot_sync_overview.py --subject S01
    python3 visualization/plot_sync_overview.py --subject S01 --window 10 --start 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pyxdf

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # visualization/
_PH2  = _HERE.parent                             # src/data_pipeline/03_preprocessing/
# _PH2 is now nested two levels deeper than before (src/data_pipeline/
# instead of directly under the repo root), so three .parent hops are
# needed here instead of one.
_ROOT = _PH2.parent.parent.parent                # Neuma_Model/

EEG_CHANNEL_NAMES = [
    "P3","C3","F3","Fz","F4","C4","P4","Cz","Pz",
    "Fp1","Fp2","T3","T5","O1","O2","X3","X2","F7","F8","X1","A2","T6","T4","TRG"
]
ET_CHANNEL_NAMES = ["Left X","Left Y","Left Pupil","Right X","Right Y","Right Pupil"]

# ── Colour palette ────────────────────────────────────────────────────────────
RAW_COLOR   = "#E07B39"
CLEAN_COLOR = "#2C7BB6"
EEG_COLORS  = ["#E07B39","#C0392B","#8E44AD","#16A085"]
SYNC_EEG    = "#2C7BB6"
SYNC_ET     = "#E07B39"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw(subject: str):
    xdf_path = _ROOT / "DataSource" / f"{subject}.xdf"
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    identified = {}
    for s in streams:
        name = s["info"]["name"][0]
        if "EEG" in name or "WS" in name:
            identified["EEG"] = s
        elif "Tobii" in name:
            identified["ET"] = s
    eeg_stream = identified["EEG"]
    et_stream  = identified["ET"]
    return (
        np.array(eeg_stream["time_series"]),
        np.array(eeg_stream["time_stamps"]),
        np.array(et_stream["time_series"]),
        np.array(et_stream["time_stamps"]),
    )


def _load_preprocessed(subject: str):
    d = _PH2 / subject / "output" / "clean_data"
    eeg_clean  = np.load(d / "eeg_clean.npy")
    eeg_ts     = np.load(d / "eeg_timestamps.npy")
    et_clean   = np.load(d / "et_clean.npy")
    et_ts      = np.load(d / "et_timestamps.npy")
    et_synced  = np.load(d / "et_synced.npy")
    return eeg_clean, eeg_ts, et_clean, et_ts, et_synced


def _kept_channels(subject: str):
    bad_map = json.loads((_PH2 / "metadata" / "bad_channels.json").read_text())
    bad = bad_map.get(subject, [])
    good = [i for i in range(24) if i not in bad]
    return good


def _window_mask(ts, t_start, t_end):
    return (ts >= t_start) & (ts <= t_end)


def _rel(ts):
    return ts - ts[0]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — EEG raw vs preprocessed
# ─────────────────────────────────────────────────────────────────────────────

def plot_eeg_comparison(
    eeg_raw, eeg_ts_raw,
    eeg_clean, eeg_ts_clean,
    kept_ch,
    subject, win_sec=10.0, start_sec=5.0,
    out_path=None
):
    # Pick 4 representative channels: Fp1(9), Cz(7), O1(13), C3(1)
    ch_raw_indices = [9, 7, 13, 1]
    ch_labels = [EEG_CHANNEL_NAMES[i] for i in ch_raw_indices]

    # Corresponding indices in eeg_clean
    ch_clean_indices = []
    for ri in ch_raw_indices:
        if ri in kept_ch:
            ch_clean_indices.append(kept_ch.index(ri))
        else:
            ch_clean_indices.append(None)

    t0 = eeg_ts_raw[0] + start_sec
    t1 = t0 + win_sec
    m_raw   = _window_mask(eeg_ts_raw,   t0, t1)
    m_clean = _window_mask(eeg_ts_clean, t0, t1)

    t_raw   = _rel(eeg_ts_raw[m_raw])   - (start_sec)
    t_clean = _rel(eeg_ts_clean[m_clean]) - (start_sec)
    t_raw   = eeg_ts_raw[m_raw]   - eeg_ts_raw[m_raw][0]
    t_clean = eeg_ts_clean[m_clean] - eeg_ts_clean[m_clean][0]

    fig, axes = plt.subplots(
        4, 2, figsize=(16, 10),
        sharex="col", sharey="none",
        gridspec_kw={"hspace": 0.45, "wspace": 0.08}
    )
    fig.suptitle(
        f"EEG Signal: Raw vs Preprocessed — {subject}\n"
        f"Window: {start_sec:.0f}–{start_sec+win_sec:.0f} s (absolute)",
        fontsize=13, fontweight="bold", y=1.01
    )

    axes[0][0].set_title("Raw EEG", fontsize=11, fontweight="bold", color=RAW_COLOR)
    axes[0][1].set_title("Preprocessed EEG\n(bad-ch removed → CAR → bandpass 1–45 Hz → notch 50 Hz → ICA)",
                         fontsize=10, fontweight="bold", color=CLEAN_COLOR)

    for row, (ri, ci, lbl, col) in enumerate(
        zip(ch_raw_indices, ch_clean_indices, ch_labels, EEG_COLORS)
    ):
        ax_r = axes[row][0]
        ax_c = axes[row][1]

        # Raw
        sig_raw = eeg_raw[m_raw, ri]
        ax_r.plot(t_raw, sig_raw, linewidth=0.5, color=RAW_COLOR, alpha=0.85)
        ax_r.set_ylabel(f"{lbl}\n(µV)", fontsize=8)
        ax_r.yaxis.set_label_coords(-0.14, 0.5)

        # Clean
        if ci is not None:
            sig_clean = eeg_clean[m_clean, ci]
            ax_c.plot(t_clean, sig_clean, linewidth=0.5, color=CLEAN_COLOR, alpha=0.9)
        else:
            ax_c.text(0.5, 0.5, "channel removed (bad)", transform=ax_c.transAxes,
                      ha="center", va="center", color="gray", fontsize=9)

        for ax in (ax_r, ax_c):
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for col in range(2):
        axes[-1][col].set_xlabel("Time (s)", fontsize=9)

    # Sampling rate annotation
    fs_raw = round(len(eeg_ts_raw) / (eeg_ts_raw[-1] - eeg_ts_raw[0]))
    fig.text(0.25, -0.01, f"Sampling rate: {fs_raw} Hz", ha="center", fontsize=8, color="gray")
    fig.text(0.75, -0.01, f"Sampling rate: {fs_raw} Hz", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — ET raw vs preprocessed
# ─────────────────────────────────────────────────────────────────────────────

def plot_et_comparison(
    et_raw, et_ts_raw,
    et_clean, et_ts_clean,
    subject, win_sec=10.0, start_sec=5.0,
    out_path=None
):
    t0 = et_ts_raw[0] + start_sec
    t1 = t0 + win_sec
    m_raw   = _window_mask(et_ts_raw,   t0, t1)
    m_clean = _window_mask(et_ts_clean, t0, t1)

    t_raw   = et_ts_raw[m_raw]   - et_ts_raw[m_raw][0]
    t_clean = et_ts_clean[m_clean] - et_ts_clean[m_clean][0]

    # 3 rows: Gaze X (left+right), Gaze Y (left+right), Pupil (left+right)
    row_specs = [
        ("Gaze X",   [0, 3], ["Left X", "Right X"], ["#2C7BB6", "#1B5C82"]),
        ("Gaze Y",   [1, 4], ["Left Y", "Right Y"], ["#E07B39", "#9D4F1A"]),
        ("Pupil",    [2, 5], ["Left Pupil", "Right Pupil"], ["#27AE60", "#1A6E3C"]),
    ]

    fig, axes = plt.subplots(
        3, 2, figsize=(16, 9),
        sharex="col",
        gridspec_kw={"hspace": 0.42, "wspace": 0.08}
    )
    fig.suptitle(
        f"Eye Tracker Signal: Raw vs Preprocessed — {subject}\n"
        f"Window: {start_sec:.0f}–{start_sec+win_sec:.0f} s",
        fontsize=13, fontweight="bold", y=1.01
    )

    axes[0][0].set_title("Raw ET (Tobii, 120 Hz)", fontsize=11,
                         fontweight="bold", color=RAW_COLOR)
    axes[0][1].set_title("Preprocessed ET\n(linear interpolation ≤100 ms gaps → 5-sample rolling smooth)",
                         fontsize=10, fontweight="bold", color=CLEAN_COLOR)

    for row, (label, ch_idx, ch_names, colors) in enumerate(row_specs):
        ax_r = axes[row][0]
        ax_c = axes[row][1]

        for ci, ch_name, color in zip(ch_idx, ch_names, colors):
            r_sig = et_raw[m_raw, ci].astype(float)
            c_sig = et_clean[m_clean, ci].astype(float)

            ax_r.plot(t_raw,   r_sig, linewidth=0.7, color=color,
                      alpha=0.8, label=ch_name)
            ax_c.plot(t_clean, c_sig, linewidth=0.7, color=color,
                      alpha=0.9, label=ch_name)

        for ax in (ax_r, ax_c):
            ax.legend(fontsize=7, loc="upper right", framealpha=0.5)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # show NaN gaps as shaded regions on raw
        nan_mask = np.isnan(et_raw[m_raw, ch_idx[0]])
        if nan_mask.any():
            _shade_nans(ax_r, t_raw, nan_mask)

        unit = "px" if "Pupil" not in label else "mm"
        ax_r.set_ylabel(f"{label} ({unit})", fontsize=8)
        ax_r.yaxis.set_label_coords(-0.14, 0.5)

    for col in range(2):
        axes[-1][col].set_xlabel("Time (s)", fontsize=9)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.close(fig)


def _shade_nans(ax, t, nan_mask):
    in_gap = False
    g_start = None
    for i, is_nan in enumerate(nan_mask):
        if is_nan and not in_gap:
            g_start = t[i]
            in_gap = True
        elif not is_nan and in_gap:
            ax.axvspan(g_start, t[i], color="red", alpha=0.12, linewidth=0)
            in_gap = False
    if in_gap:
        ax.axvspan(g_start, t[-1], color="red", alpha=0.12, linewidth=0)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Timestamp synchronisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_sync(
    eeg_ts_raw, et_ts_raw,
    eeg_clean, eeg_ts_clean,
    et_synced, et_ts_clean,
    subject, zoom_start=5.0, zoom_len=2.0,
    eeg_ch_idx=7,    # Cz in eeg_clean
    et_ch_idx=0,     # Left X in et_synced
    out_path=None
):
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.2, 1.5, 2.0],
        hspace=0.55, wspace=0.30,
        figure=fig
    )
    fig.suptitle(
        f"EEG & ET Timestamp Synchronisation — {subject}",
        fontsize=14, fontweight="bold", y=1.01
    )

    # ── Panel A: full-session timestamp density ────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])   # full width
    t0 = min(eeg_ts_raw[0], et_ts_raw[0])

    eeg_rel = eeg_ts_raw - t0
    et_rel  = et_ts_raw  - t0

    # Rug / event lines (down-sampled for speed)
    step_eeg = max(1, len(eeg_rel) // 3000)
    step_et  = max(1, len(et_rel)  // 3000)

    ax_a.eventplot(eeg_rel[::step_eeg], lineoffsets=1.0, linelengths=0.6,
                   linewidths=0.35, colors=SYNC_EEG, label=f"EEG ({len(eeg_rel):,} samples, ~300 Hz)")
    ax_a.eventplot(et_rel[::step_et],  lineoffsets=0.0, linelengths=0.6,
                   linewidths=0.35, colors=SYNC_ET,  label=f"ET ({len(et_rel):,} samples, ~120 Hz)")

    ax_a.set_xlabel("Session time (s)", fontsize=9)
    ax_a.set_yticks([0.0, 1.0])
    ax_a.set_yticklabels(["ET stream", "EEG stream"], fontsize=8)
    ax_a.set_title("A  — Full-session timestamp coverage (down-sampled rug plot)", fontsize=10, loc="left")
    ax_a.legend(fontsize=8, loc="upper right")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Shade the zoom region
    z0 = zoom_start
    z1 = zoom_start + zoom_len
    ax_a.axvspan(z0, z1, color="gold", alpha=0.25, label="zoom window")

    # ── Panel B: zoomed window — individual sample positions ──────────────
    ax_b = fig.add_subplot(gs[1, 0])

    eeg_zoom = eeg_rel[(eeg_rel >= z0) & (eeg_rel <= z1)]
    et_zoom  = et_rel[(et_rel  >= z0) & (et_rel  <= z1)]

    ax_b.eventplot(eeg_zoom, lineoffsets=1.0, linelengths=0.5,
                   linewidths=0.7, colors=SYNC_EEG, label=f"EEG ({len(eeg_zoom)} pts)")
    ax_b.eventplot(et_zoom,  lineoffsets=0.0, linelengths=0.5,
                   linewidths=0.7, colors=SYNC_ET,  label=f"ET ({len(et_zoom)} pts)")

    ax_b.set_xlabel("Session time (s)", fontsize=9)
    ax_b.set_yticks([0.0, 1.0])
    ax_b.set_yticklabels(["ET samples", "EEG samples"], fontsize=8)
    ax_b.set_title(f"B — Zoom: {z0:.1f}–{z1:.1f} s (individual sample positions)", fontsize=10, loc="left")
    ax_b.legend(fontsize=8, loc="upper right")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # ── Panel C: inter-sample interval histograms ──────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])

    eeg_isi = np.diff(eeg_ts_raw) * 1000   # ms
    et_isi  = np.diff(et_ts_raw)  * 1000

    bins = np.linspace(0, 15, 80)
    ax_c.hist(eeg_isi, bins=bins, color=SYNC_EEG, alpha=0.65,
              label=f"EEG  median={np.median(eeg_isi):.2f} ms")
    ax_c.hist(et_isi,  bins=bins, color=SYNC_ET,  alpha=0.65,
              label=f"ET   median={np.median(et_isi):.2f} ms")
    ax_c.axvline(1000/300, color=SYNC_EEG, linestyle="--", linewidth=1.2, alpha=0.8)
    ax_c.axvline(1000/120, color=SYNC_ET,  linestyle="--", linewidth=1.2, alpha=0.8)
    ax_c.set_xlabel("Inter-sample interval (ms)", fontsize=9)
    ax_c.set_ylabel("Count", fontsize=9)
    ax_c.set_title("C — Sample-interval distributions (dashed = nominal period)", fontsize=10, loc="left")
    ax_c.legend(fontsize=8)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ── Panel D: dual-axis overlay — EEG Cz + ET Left-X synced ───────────
    ax_d = fig.add_subplot(gs[2, :])

    # Use the synced (ET resampled to EEG grid) so both share the same time axis
    t_clean_rel = eeg_ts_clean - eeg_ts_clean[0]

    # 30-second overlay window
    ov_len = 30.0
    ov_start = 5.0
    m_ov = (t_clean_rel >= ov_start) & (t_clean_rel <= ov_start + ov_len)

    t_ov     = t_clean_rel[m_ov]
    eeg_ov   = eeg_clean[m_ov, eeg_ch_idx]
    et_ov    = et_synced[m_ov, et_ch_idx]

    ax_d2 = ax_d.twinx()

    l1, = ax_d.plot(t_ov, eeg_ov, linewidth=0.5, color=SYNC_EEG, alpha=0.85,
                    label="EEG — Cz (µV)")
    l2, = ax_d2.plot(t_ov, et_ov, linewidth=0.7, color=SYNC_ET, alpha=0.85,
                     label="ET — Left Gaze X (px, synced to EEG grid)")

    ax_d.set_xlabel("Time relative to session start (s)", fontsize=9)
    ax_d.set_ylabel("EEG amplitude (µV)", color=SYNC_EEG, fontsize=9)
    ax_d2.set_ylabel("Gaze X position (px)", color=SYNC_ET, fontsize=9)
    ax_d.tick_params(axis="y", labelcolor=SYNC_EEG, labelsize=7)
    ax_d2.tick_params(axis="y", labelcolor=SYNC_ET,  labelsize=7)
    ax_d.tick_params(axis="x", labelsize=7)

    ax_d.set_title(
        "D — Synchronised overlay: EEG Cz + ET Left Gaze X (ET resampled to EEG 300 Hz grid via nearest-neighbour)",
        fontsize=10, loc="left"
    )
    ax_d.legend(handles=[l1, l2], fontsize=8, loc="upper right")
    ax_d.spines["top"].set_visible(False)
    ax_d2.spines["top"].set_visible(False)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def _available_subjects():
    return sorted(
        p.name for p in _PH2.iterdir()
        if p.is_dir() and (p / "output" / "clean_data" / "eeg_clean.npy").exists()
    )


def _run_subject(subject, window, start, outdir):
    out_dir = Path(outdir) if outdir else _PH2 / subject / "output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Subject : {subject}")
    print(f"[INFO] Output  : {out_dir}")
    print("[INFO] Loading raw XDF ...")
    eeg_raw, eeg_ts_raw, et_raw, et_ts_raw = _load_raw(subject)

    print("[INFO] Loading preprocessed .npy ...")
    eeg_clean, eeg_ts_clean, et_clean, et_ts_clean, et_synced = _load_preprocessed(subject)

    kept_ch = _kept_channels(subject)

    print("[INFO] Rendering Figure 1 — EEG raw vs preprocessed ...")
    plot_eeg_comparison(
        eeg_raw, eeg_ts_raw,
        eeg_clean, eeg_ts_clean,
        kept_ch, subject,
        win_sec=window, start_sec=start,
        out_path=out_dir / f"{subject}_fig1_eeg_raw_vs_clean.png",
    )

    print("[INFO] Rendering Figure 2 — ET raw vs preprocessed ...")
    plot_et_comparison(
        et_raw, et_ts_raw,
        et_clean, et_ts_clean,
        subject,
        win_sec=window, start_sec=start,
        out_path=out_dir / f"{subject}_fig2_et_raw_vs_clean.png",
    )

    print("[INFO] Rendering Figure 3 — Timestamp synchronisation ...")
    plot_sync(
        eeg_ts_raw, et_ts_raw,
        eeg_clean, eeg_ts_clean,
        et_synced, et_ts_clean,
        subject,
        zoom_start=start,
        zoom_len=2.0,
        out_path=out_dir / f"{subject}_fig3_timestamp_sync.png",
    )

    print(f"[DONE] {subject} — 3 figures saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Plot EEG/ET raw-vs-preprocessed + sync overview"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--subject", default=None,
                     help="Single subject ID, e.g. S01")
    grp.add_argument("--all", action="store_true",
                     help="Run for every subject that has preprocessed data")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="List of subject IDs, e.g. --subjects S01 S03 S05")
    parser.add_argument("--window",  type=float, default=10.0,
                        help="Window length in seconds (default: 10)")
    parser.add_argument("--start",   type=float, default=5.0,
                        help="Start offset in seconds (default: 5)")
    parser.add_argument("--outdir",  default=None,
                        help="Output directory override (per-subject subfolders created inside)")
    args = parser.parse_args()

    if args.all:
        subjects = _available_subjects()
        print(f"[INFO] Running all {len(subjects)} subjects: {subjects}")
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = [args.subject or "S01"]

    failed = []
    for i, subject in enumerate(subjects, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(subjects)}] {subject}")
        print(f"{'='*60}")
        try:
            outdir = str(Path(args.outdir) / subject) if args.outdir else None
            _run_subject(subject, args.window, args.start, outdir)
        except Exception as exc:
            print(f"[ERROR] {subject} failed: {exc}")
            failed.append(subject)

    print(f"\n{'='*60}")
    print(f"  Finished {len(subjects) - len(failed)}/{len(subjects)} subjects")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
