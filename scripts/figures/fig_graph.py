#!/usr/bin/env python3
"""Figure 3 — Dynamic EEG graph construction (real data, real PLV).

Pipeline panels: (A) EEG epoch signals -> (B) instantaneous phase (Hilbert)
-> (C) pairwise PLV adjacency matrix -> (D) thresholded functional graph (tau=0.30),
matching Section 'Dynamic EEG Graph Construction' and Table~impl (tau=0.30).

Uses the project's own graphs/connectivity.py so the figure reflects exactly
what the model consumes. Outputs figures/fig_graph.{pdf,png}.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import networkx as nx

P8 = Path(__file__).resolve().parents[1]          # NEUMA_PHASE8
ROOT = P8.parents[0]                               # repo root
sys.path.insert(0, str(P8))
from graphs.connectivity import compute_weighted_adjacency  # noqa: E402

# Real 19-channel NeuMa scalp montage (order matches NEUMA_PHASE3_clean data).
CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
      "F7",  "F8",  "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
POS = {  # approximate 10-20 scalp coordinates (x right, y front)
    "Fp1": (-.30, .92), "Fp2": (.30, .92),
    "F7": (-.82, .50), "F3": (-.40, .52), "Fz": (0, .54), "F4": (.40, .52), "F8": (.82, .50),
    "T3": (-.97, 0), "C3": (-.42, 0), "Cz": (0, 0), "C4": (.42, 0), "T4": (.97, 0),
    "T5": (-.80, -.50), "P3": (-.40, -.52), "Pz": (0, -.54), "P4": (.40, -.52), "T6": (.80, -.50),
    "O1": (-.30, -.90), "O2": (.30, -.90),
}
FS = 300.0
TAU = 0.30
SUBJ = "S02"          # 19-channel scalp montage (cleaned)
EPOCH = 0


def load_epoch():
    a = np.asarray(np.load(P8.parents[0] / f"NEUMA_PHASE3_clean/{SUBJ}/output/epochs/eeg_epochs_engagement.npy",
                           allow_pickle=True)).astype(np.float32)
    return a[EPOCH]   # (1500, 19)


def main():
    ep = load_epoch()                                    # (T, C)
    T = ep.shape[0]
    t = np.arange(T) / FS
    plv = compute_weighted_adjacency(ep, method="plv")   # (24, 24) in [0,1]

    # 2x2 layout (A,B top; C,D bottom) so each panel is large and legible when
    # the figure is scaled to the column/page width.
    fig = plt.figure(figsize=(12.5, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.24)

    # (A) raw EEG signals — a few representative channels, first ~1 s
    axA = fig.add_subplot(gs[0, 0])
    show = ["Fz", "F3", "Cz", "Pz", "O1"]
    w = slice(0, int(1.0 * FS))
    for i, name in enumerate(show):
        c = CH.index(name)
        s = ep[w, c]
        s = (s - s.mean()) / (s.std() + 1e-6)
        axA.plot(t[w], s + i * 3.0, lw=1.1)
        axA.text(-0.02, i * 3.0, name, ha="right", va="center", fontsize=11,
                 transform=axA.get_yaxis_transform())
    axA.set_title("(A) EEG epoch", fontsize=14, fontweight="bold", loc="left")
    axA.set_xlabel("Time (s)", fontsize=12); axA.tick_params(labelsize=10)
    axA.set_yticks([]); axA.spines[["top", "right", "left"]].set_visible(False)

    # (B) instantaneous phase (Hilbert) for two channels
    axB = fig.add_subplot(gs[0, 1])
    wb = slice(0, int(0.5 * FS))
    for name, col in [("Fz", "C0"), ("Pz", "C2")]:
        ph = np.angle(hilbert(ep[:, CH.index(name)]))
        axB.plot(t[wb], ph[wb], color=col, lw=1.4, label=name)
    axB.set_title("(B) Instantaneous phase", fontsize=14, fontweight="bold", loc="left")
    axB.set_xlabel("Time (s)", fontsize=12); axB.set_ylabel("Phase (rad)", fontsize=12)
    axB.set_yticks([-np.pi, 0, np.pi]); axB.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    axB.tick_params(labelsize=10)
    axB.legend(fontsize=11, loc="upper right", frameon=False)
    axB.spines[["top", "right"]].set_visible(False)

    # (C) PLV adjacency matrix
    axC = fig.add_subplot(gs[1, 0])
    im = axC.imshow(plv, cmap="viridis", vmin=0, vmax=1)
    axC.set_title("(C) PLV adjacency", fontsize=14, fontweight="bold", loc="left")
    axC.set_xticks(range(len(CH))); axC.set_yticks(range(len(CH)))
    axC.set_xticklabels(CH, rotation=90, fontsize=9)
    axC.set_yticklabels(CH, fontsize=9)
    cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
    cb.set_label("PLV", fontsize=11); cb.ax.tick_params(labelsize=9)

    # (D) thresholded functional graph
    axD = fig.add_subplot(gs[1, 1])
    G = nx.Graph()
    for c in CH:
        G.add_node(c)
    for i in range(len(CH)):
        for j in range(i + 1, len(CH)):
            if plv[i, j] >= TAU:
                G.add_edge(CH[i], CH[j], w=float(plv[i, j]))
    ws = np.array([G[u][v]["w"] for u, v in G.edges()])
    nx.draw_networkx_edges(G, POS, ax=axD, width=0.3 + 2.5 * (ws - TAU) / (1 - TAU),
                           edge_color=ws, edge_cmap=plt.cm.plasma, edge_vmin=TAU, edge_vmax=1, alpha=.7)
    deg = np.array([G.degree(c) for c in CH])
    nx.draw_networkx_nodes(G, POS, nodelist=CH, ax=axD, node_size=320 + 40 * deg,
                           node_color="#1f4e79", edgecolors="white", linewidths=.8)
    nx.draw_networkx_labels(G, POS, ax=axD, font_size=9, font_color="white")
    axD.set_title(rf"(D) Functional graph ($\tau={TAU}$)", fontsize=14, fontweight="bold", loc="left")
    axD.set_axis_off(); axD.set_aspect("equal")

    out = P8 / "figures" / "fig_graph"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    n_edges = G.number_of_edges()
    nC = len(CH)
    print(f"subject={SUBJ} epoch={EPOCH}  PLV range [{plv[~np.eye(nC,dtype=bool)].min():.2f},"
          f"{plv[~np.eye(nC,dtype=bool)].max():.2f}]  edges>= {TAU}: {n_edges}/{nC*(nC-1)//2}")
    print(f"wrote {out.with_suffix('.pdf').relative_to(P8)} and .png")


if __name__ == "__main__":
    main()
