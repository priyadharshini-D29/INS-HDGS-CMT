#!/usr/bin/env python3
"""Figure 3 -- Dynamic EEG functional-graph construction (real data, real Pearson).

The epoch is loaded through NeumaGraphDataset, so it passes through the same channel
harmonisation (24 raw -> 19 cortical electrodes, canonical order) and the same per-epoch
z-scoring as the training data. The adjacency is the |Pearson| matrix of one 0.5-s window
computed with the model's own pearson_connectivity, as in graph_builder.compute_epoch_graphs
and scripts/analysis/tau_sensitivity.py.

  (A) the 5-s epoch with the ten non-overlapping 0.5-s windows marked;
  (B) the z-scored signals of one electrode pair within the chosen window (one edge);
  (C) the |rho_ij(t)| adjacency of that window;
  (D) the functional graph G_t after thresholding at tau (settings.CONN_THRESHOLD = 0.30).

Output : paper/figures/fig_graph.{pdf,png} (+ results/figures/)
Run    : python scripts/figures/fig_graph.py              (repository root, server; CPU)
Env    : NEUMA_FIG3_SUBJECT (S02), NEUMA_FIG3_EPOCH (0), NEUMA_FIG3_WINDOW (auto = the window
         whose density at tau is closest to the pooled mean density; or an integer 0-9),
         NEUMA_FIG3_PAIR (Fz,Pz)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]                     # <repo>
MODEL = ROOT / "src" / "model"
for p in (str(MODEL), str(MODEL.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
from config.settings import CONN_THRESHOLD, EEG_SR, N_WINDOWS   # noqa: E402
from data.channel_harmonizer import CANONICAL_CHANNELS          # noqa: E402
from data.dataset import NeumaGraphDataset                      # noqa: E402
from graphs.connectivity import pearson_connectivity            # noqa: E402

POS = {  # approximate 10-20 scalp coordinates (x right, y front)
    "Fp1": (-.30, .92), "Fp2": (.30, .92),
    "F7": (-.82, .50), "F3": (-.40, .52), "Fz": (0, .54), "F4": (.40, .52), "F8": (.82, .50),
    "T3": (-.97, 0), "C3": (-.42, 0), "Cz": (0, 0), "C4": (.42, 0), "T4": (.97, 0),
    "T5": (-.80, -.50), "P3": (-.40, -.52), "Pz": (0, -.54), "P4": (.40, -.52), "T6": (.80, -.50),
    "O1": (-.30, -.90), "O2": (.30, -.90),
}
FS = float(EEG_SR)
TAU = float(CONN_THRESHOLD)
SUBJ = os.environ.get("NEUMA_FIG3_SUBJECT", "S02")
EPOCH = int(os.environ.get("NEUMA_FIG3_EPOCH", "0"))
WIN_ENV = os.environ.get("NEUMA_FIG3_WINDOW", "auto")
PAIR = tuple(os.environ.get("NEUMA_FIG3_PAIR", "Fz,Pz").split(","))


def window_adjacencies(eeg_epoch: np.ndarray, n_windows: int) -> np.ndarray:
    """|Pearson r| per non-overlapping window, as in tau_sensitivity.raw_window_connectivity
    (per-epoch z-score, clip +/-5). Returns (n_windows, C, C) with unit diagonal."""
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
    return out, x


def pooled_density_at_tau(default: float = 0.57) -> float:
    """Mean density at tau from results/sensitivity/tau_graph_density.csv (Sec. 2.5.2), else default."""
    csv = ROOT / "results" / "sensitivity" / "tau_graph_density.csv"
    try:
        import pandas as pd
        d = pd.read_csv(csv)
        row = d.iloc[(d["tau"] - TAU).abs().argmin()]
        return float(row["density_mean"])
    except Exception:
        return default


def main():
    ds = NeumaGraphDataset(subject_ids=[SUBJ], precompute_graphs=False, augment=False)
    ep = np.asarray(ds.raw_eeg[EPOCH], np.float32)             # (T, C) harmonised, canonical order
    T, C = ep.shape
    ch = list(CANONICAL_CHANNELS)[:C]
    assert C == len(CANONICAL_CHANNELS), f"expected {len(CANONICAL_CHANNELS)} electrodes, got {C}"
    pos = {c: POS[c] for c in ch}
    t = np.arange(T) / FS
    W, x = window_adjacencies(ep, N_WINDOWS)                     # (n_win, C, C), z-scored epoch
    w_size = T // N_WINDOWS
    iu = np.triu_indices(C, k=1)
    dens = (W[:, iu[0], iu[1]] >= TAU).mean(axis=1)             # density per window
    if WIN_ENV.strip().lower() == "auto":
        target = pooled_density_at_tau()
        WIN = int(np.argmin(np.abs(dens - target)))
    else:
        WIN = int(WIN_ENV)
    w0, w1 = WIN * w_size, (WIN + 1) * w_size
    A = W[WIN]
    rho = pearson_connectivity(x[w0:w1])                         # signed, for the annotation in (B)
    i1, i2 = ch.index(PAIR[0]), ch.index(PAIR[1])

    fig = plt.figure(figsize=(12.5, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.24)

    # (A) EEG epoch with the ten 0.5-s windows marked ------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    show = [c for c in ["Fz", "F3", "Cz", "Pz", "O1"] if c in ch]
    for i, name in enumerate(show):
        axA.plot(t, x[:, ch.index(name)] + i * 3.0, lw=0.7)
        axA.text(-0.02, i * 3.0, name, ha="right", va="center", fontsize=11,
                 transform=axA.get_yaxis_transform())
    for k in range(N_WINDOWS + 1):
        axA.axvline(k * w_size / FS, color="0.75", lw=0.6, ls=":")
    axA.axvspan(w0 / FS, w1 / FS, color="#ffd54f", alpha=0.35, lw=0)
    axA.text((w0 + w1) / 2 / FS, len(show) * 3.0 - 1.0, f"window t={WIN + 1}",
             ha="center", fontsize=10, color="#8a6d00")
    axA.set_title(f"(A) EEG epoch (5 s, {N_WINDOWS} windows of {w_size / FS:.1f} s)",
                  fontsize=14, fontweight="bold", loc="left")
    axA.set_xlabel("Time (s)", fontsize=12); axA.tick_params(labelsize=10)
    axA.set_xlim(0, T / FS)
    axA.set_yticks([]); axA.spines[["top", "right", "left"]].set_visible(False)

    # (B) two windowed z-scored signals -> one Pearson edge ------------------------------
    axB = fig.add_subplot(gs[0, 1])
    tw = t[w0:w1]
    for name, idx, col in [(PAIR[0], i1, "C0"), (PAIR[1], i2, "C2")]:
        s = x[w0:w1, idx]
        s = (s - s.mean()) / (s.std() + 1e-6)
        axB.plot(tw, s, color=col, lw=1.4, label=name)
    axB.set_title("(B) Windowed signals (z-scored)", fontsize=14, fontweight="bold", loc="left")
    axB.set_xlabel("Time (s)", fontsize=12); axB.set_ylabel("Amplitude (z)", fontsize=12)
    axB.tick_params(labelsize=10)
    axB.legend(fontsize=11, loc="upper right", frameon=False)
    cmp_sym = r"\geq" if A[i1, i2] >= TAU else "<"
    axB.text(0.02, 0.04,
             rf"$\rho_{{\mathrm{{{PAIR[0]},{PAIR[1]}}}}}(t) = {rho[i1, i2]:+.2f}$,  "
             rf"$|\rho| {cmp_sym} \tau = {TAU:.2f}$",
             transform=axB.transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    axB.spines[["top", "right"]].set_visible(False)

    # (C) |Pearson| adjacency of the window ---------------------------------------------
    axC = fig.add_subplot(gs[1, 0])
    im = axC.imshow(A, cmap="viridis", vmin=0, vmax=1)
    axC.set_title(rf"(C) $|\rho_{{ij}}(t)|$ adjacency, window t={WIN + 1}",
                  fontsize=14, fontweight="bold", loc="left")
    axC.set_xticks(range(C)); axC.set_yticks(range(C))
    axC.set_xticklabels(ch, rotation=90, fontsize=9)
    axC.set_yticklabels(ch, fontsize=9)
    cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
    cb.set_label(r"$|\rho|$ (Pearson)", fontsize=11); cb.ax.tick_params(labelsize=9)
    cb.ax.axhline(TAU, color="w", lw=1.2)

    # (D) thresholded functional graph --------------------------------------------------
    axD = fig.add_subplot(gs[1, 1])
    G = nx.Graph()
    G.add_nodes_from(ch)
    for i in range(C):
        for j in range(i + 1, C):
            if A[i, j] >= TAU:
                G.add_edge(ch[i], ch[j], w=float(A[i, j]))
    ws = np.array([G[u][v]["w"] for u, v in G.edges()]) if G.number_of_edges() else np.array([])
    if ws.size:
        nx.draw_networkx_edges(G, pos, ax=axD, width=0.3 + 2.5 * (ws - TAU) / (1 - TAU),
                               edge_color=ws, edge_cmap=plt.cm.plasma, edge_vmin=TAU, edge_vmax=1,
                               alpha=.7)
    deg = np.array([G.degree(c) for c in ch])
    nx.draw_networkx_nodes(G, pos, nodelist=ch, ax=axD, node_size=320 + 40 * deg,
                           node_color="#1f4e79", edgecolors="white", linewidths=.8)
    nx.draw_networkx_labels(G, pos, ax=axD, font_size=9, font_color="white")
    n_pairs = C * (C - 1) // 2
    axD.set_title(rf"(D) Functional graph $G_t$ ($|\rho|\geq\tau={TAU:.2f}$; "
                  rf"{G.number_of_edges()}/{n_pairs} edges)",
                  fontsize=14, fontweight="bold", loc="left")
    axD.set_axis_off(); axD.set_aspect("equal")

    for out_dir in (ROOT / "paper" / "figures", ROOT / "results" / "figures"):
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "fig_graph"
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    off = A[~np.eye(C, dtype=bool)]
    print(f"subject={SUBJ} epoch={EPOCH} ({len(ds)} epochs) C={C} window={WIN} (densities per window: "
          + " ".join(f"{d:.2f}" for d in dens) + ")")
    print(f"|rho| range [{off.min():.2f},{off.max():.2f}]  edges>= {TAU}: {G.number_of_edges()}/{n_pairs} "
          f"(density {dens[WIN]:.2f}; mean degree {deg.mean():.1f})  rho({PAIR[0]},{PAIR[1]})={rho[i1, i2]:+.3f}")
    print("wrote paper/figures/fig_graph.{pdf,png} and results/figures/fig_graph.{pdf,png}")


if __name__ == "__main__":
    main()
