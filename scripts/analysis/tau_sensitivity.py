"""
================================================================
Connectivity-threshold (tau) sensitivity — Reviewer 2, comment 2
================================================================
Eq. (3) sparsifies each windowed |Pearson| connectivity matrix at a
fixed tau = 0.30. This script quantifies (A) how tau changes the
functional-connectivity graphs themselves, over ALL epochs of ALL
available subjects, and (B) how it changes downstream LOSOCV metrics,
by summarising the sweep runs of sensitivity_sweep.py.

(A) Graph-structure sweep (no training needed; runs in seconds)
    For tau in a grid, per window:
      density        : fraction of the C(C-1)/2 electrode pairs retained
      mean degree    : average retained edges per electrode (excl. self-loop)
      isolated nodes : electrodes with degree < MIN_EDGES (settings; 3)
      windows below  : windows with mean degree < MIN_EDGES
    Reported as mean ± SD over windows and as per-subject means, so the
    reader can see that tau=0.30 sits on the plateau of the density curve
    rather than at a cliff.

(B) Downstream sweep (training; run on the GPU server)
      cd src/model
      python sensitivity_sweep.py --sweep threshold \
          --subjects $(python -c "from config.settings import SUBJECT_IDS;print(','.join(SUBJECT_IDS))") \
          --n-ensemble 3 --epochs 120
    then re-run this script: it picks up
      src/model/output/metrics/sensitivity_threshold_summary.csv
    and merges balanced accuracy / ROC-AUC / MCC per tau into the table.

Outputs
-------
  results/sensitivity/tau_graph_density.csv        (per tau, pooled)
  results/sensitivity/tau_graph_density_per_subject.csv
  results/sensitivity/tau_sensitivity.md           (combined table)
  results/sensitivity/fig_tau_sensitivity.pdf/.png (2 panels)
================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "src" / "model"
for p in (str(MODEL), str(MODEL.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config.settings import (CONN_THRESHOLD, EEG_SR, MIN_EDGES,       # noqa: E402
                             N_WINDOWS, SUBJECT_IDS)
from data.dataset import NeumaGraphDataset                            # noqa: E402
from model.inference.graphs.connectivity import pearson_connectivity   # noqa: E402


def raw_window_connectivity(eeg_epoch: np.ndarray, n_windows: int) -> np.ndarray:
    """|Pearson r| per sliding window, exactly as graph_builder.compute_epoch_graphs
    builds the matrix that Eq. (3) thresholds (per-epoch z-score, clip ±5,
    non-overlapping windows). Returns (n_windows, C, C) with unit diagonal."""
    x = np.nan_to_num(np.asarray(eeg_epoch, np.float32))
    x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)
    x = np.clip(x, -5.0, 5.0)
    T, C = x.shape
    w = T // n_windows
    out = np.zeros((n_windows, C, C), np.float32)
    for k in range(n_windows):
        r = np.abs(pearson_connectivity(x[k * w:(k + 1) * w]))
        np.fill_diagonal(r, 1.0)
        out[k] = r
    return out

OUT = ROOT / "results" / "sensitivity"
TAUS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]


def graph_stats(ds: NeumaGraphDataset, taus):
    """Return (pooled DataFrame, per-subject DataFrame)."""
    rows, subj_rows = [], []
    W = []
    for i in range(len(ds)):
        W.append(raw_window_connectivity(ds.raw_eeg[i], N_WINDOWS))   # (n_win, C, C), |r|, diag 1
    W = np.concatenate(W, axis=0)                           # (N_windows_total, C, C)
    C = W.shape[-1]
    iu = np.triu_indices(C, k=1)
    upper = W[:, iu[0], iu[1]]                              # (Nw, C(C-1)/2)
    subj_of_window = np.repeat(ds.subject_ids, N_WINDOWS)
    names = ds.unique_subjects
    for tau in taus:
        keep = upper >= tau                                 # (Nw, P)
        density = keep.mean(axis=1)
        deg = np.zeros((keep.shape[0], C))
        for k, (a, b) in enumerate(zip(*iu)):
            deg[:, a] += keep[:, k]; deg[:, b] += keep[:, k]
        isolated = (deg < MIN_EDGES).sum(axis=1)
        rows.append(dict(tau=tau, n_windows=len(density),
                         density_mean=density.mean(), density_sd=density.std(),
                         mean_degree=deg.mean(), mean_degree_sd=deg.mean(axis=1).std(),
                         isolated_nodes_mean=isolated.mean(),
                         frac_windows_any_isolated=(isolated > 0).mean(),
                         frac_windows_mean_deg_below_min=(deg.mean(axis=1) < MIN_EDGES).mean(),
                         frac_windows_empty=(keep.sum(axis=1) == 0).mean()))
        for s_idx, s_name in enumerate(names):
            m = subj_of_window == s_idx
            subj_rows.append(dict(tau=tau, subject=s_name, density_mean=density[m].mean(),
                                  mean_degree=deg[m].mean(), isolated_nodes_mean=isolated[m].mean()))
    return pd.DataFrame(rows), pd.DataFrame(subj_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default=None, help="comma list; default all available")
    ap.add_argument("--taus", default=",".join(str(t) for t in TAUS))
    ap.add_argument("--downstream-csv", default=str(MODEL / "output" / "metrics" / "sensitivity_threshold_summary.csv"))
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    taus = [float(t) for t in args.taus.split(",")]
    subjects = args.subjects.split(",") if args.subjects else SUBJECT_IDS

    ds = NeumaGraphDataset(subject_ids=subjects, precompute_graphs=False, augment=False)
    print(f"[tau] {len(ds)} epochs from {len(ds.unique_subjects)} subjects, "
          f"{N_WINDOWS} windows/epoch, C={ds.n_eeg_ch}")
    pooled, per_subj = graph_stats(ds, taus)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(out / "tau_graph_density.csv", index=False)
    per_subj.to_csv(out / "tau_graph_density_per_subject.csv", index=False)

    # downstream sweep (if available)
    down = None
    dcsv = Path(args.downstream_csv)
    if dcsv.exists():
        down = pd.read_csv(dcsv).rename(columns={"NEUMA_CONN_THRESHOLD": "tau"})
        print(f"[tau] merged downstream sweep from {dcsv}")
    else:
        print(f"[tau] downstream sweep not found ({dcsv}); run sensitivity_sweep.py --sweep threshold first")

    # markdown
    lines = [f"# Connectivity-threshold sensitivity (tau; Eq. 3)", "",
             f"Graph statistics pooled over {int(pooled.n_windows.iloc[0])} windows "
             f"({len(ds)} epochs × {N_WINDOWS} windows, {len(ds.unique_subjects)} subjects, "
             f"{ds.n_eeg_ch} electrodes). Density = fraction of the "
             f"{ds.n_eeg_ch*(ds.n_eeg_ch-1)//2} electrode pairs retained; MIN_EDGES = {MIN_EDGES}.", "",
             "| tau | density (mean ± SD) | mean degree | isolated nodes / window | windows with any isolated node | "
             + ("BalAcc | ROC-AUC (subj) | MCC | n folds |" if down is not None else ""),
             "|---|---|---|---|---|" + ("---|---|---|---|" if down is not None else "")]
    for _, r in pooled.iterrows():
        mark = " **(paper)**" if abs(r.tau - CONN_THRESHOLD) < 1e-9 else ""
        line = (f"| {r.tau:.2f}{mark} | {r.density_mean:.3f} ± {r.density_sd:.3f} | {r.mean_degree:.2f} | "
                f"{r.isolated_nodes_mean:.2f} | {r.frac_windows_any_isolated:.3f} |")
        if down is not None:
            d = down[np.isclose(down.tau, r.tau)]
            if len(d):
                d = d.iloc[0]
                line += (f" {d.mean_balanced_acc:.3f} | {d.mean_subject_auc:.3f} | {d.mean_mcc:.3f} | {int(d.n_folds)} |")
            else:
                line += " – | – | – | – |"
        lines.append(line)
    (out / "tau_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ncol = 2 if down is not None else 1
        fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 3.8))
        axes = np.atleast_1d(axes)
        ax = axes[0]
        ax.errorbar(pooled.tau, pooled.density_mean, yerr=pooled.density_sd, marker="o", color="#2f4b8f",
                    capsize=3, label="edge density")
        ax2 = ax.twinx()
        ax2.plot(pooled.tau, pooled.isolated_nodes_mean, marker="s", color="#C0504D", label="isolated nodes / window")
        ax.axvline(CONN_THRESHOLD, ls="--", color="k", lw=1)
        ax.set_xlabel(r"connectivity threshold $\tau$"); ax.set_ylabel("edge density"); ax2.set_ylabel("isolated electrodes")
        ax.set_title("(A) Graph structure vs. $\\tau$", fontweight="bold")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
        if down is not None:
            ax = axes[1]
            ax.plot(down.tau, down.mean_balanced_acc, marker="o", label="balanced accuracy")
            ax.plot(down.tau, down.mean_subject_auc, marker="s", label="ROC-AUC (per subject)")
            ax.plot(down.tau, down.mean_mcc, marker="^", label="MCC")
            ax.axvline(CONN_THRESHOLD, ls="--", color="k", lw=1)
            ax.set_xlabel(r"connectivity threshold $\tau$"); ax.set_ylabel("LOSOCV metric")
            ax.set_title("(B) Downstream metrics vs. $\\tau$", fontweight="bold"); ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_tau_sensitivity.{ext}", dpi=300, bbox_inches="tight")
        print(f"[tau] figure → {out / 'fig_tau_sensitivity.pdf'}")
    except Exception as e:  # pragma: no cover
        print(f"[tau] figure skipped: {e}")


if __name__ == "__main__":
    main()
