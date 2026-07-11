#!/usr/bin/env python3
"""
figure_panels_S01.py
====================
Generate the 13 manuscript case-study figures (Panels A–D) for ONE
representative LOSOCV sample, using REAL model intermediates — not synthetic
placeholders.

Subject / fold : S01  (fold-01 of repro_g0p0_balanced_37)
                 → best-performing fold (acc 0.933 / MCC 0.829 / bal-acc 0.875)
                   AND the user-suggested S01.
Checkpoint     : output/checkpoints/repro_g0p0_balanced_37/
                 repro_g0p0_balanced_37_fold01_e{0..4}.pt   (5-member ensemble)

Every panel is sourced from the trained network's eval-mode forward pass on the
exact preprocessed tensors fed to the model:
  A1 raw EEG          — preprocessed epoch waveform (the same epoch the band-
                        power node features are computed from)
  A2 Pearson matrix   — sample["weighted_adjs"] (the actual DynamicGAT input)
  A3 brain graph      — same adjacency, electrodes at 10-20 positions
  A4 LIF spike raster — captured from SpikingEEGEncoder surrogate-spike calls
  B1 gaze trajectory  — recorded gaze (normalised screen coords) on the brochure
  B2 fixation map     — fixations detected from the gaze stream, sized by dwell
  B3 ROI heatmap      — ET-encoder roi_attn (B, N_ROIS), the real learned ROIs
  C1 EEG←Graph attn   — out["attn"]["eeg_graph"]
  C2 EEG←ET attn      — out["attn"]["eeg_et"]
  C3 fused embedding   — out["fused"]
  D1 rule activations  — out["rule_info"]["activations"]  (8 rules)
  D2 prediction        — softmax(out["logits"]); ensemble-mean prob for headline
"""

import os
import sys
from pathlib import Path

# cap BLAS/OpenMP threads BEFORE importing numpy/torch to avoid CPU oversubscription
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

# ── Run from the project root regardless of invocation cwd ───────────────────
PHASE8 = Path("/home/nvidia/24PHD1314/Neuma_Model/NEUMA_PHASE8")
os.chdir(PHASE8)
sys.path.insert(0, str(PHASE8))
sys.path.insert(0, str(PHASE8.parent))   # so absolute `NEUMA_PHASE8.x` imports resolve

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import networkx as nx
from PIL import Image
import pandas as pd

from data.dataset import NeumaGraphDataset
from models.ins_hdgs_cmt import INS_HDGS_CMT, AblationConfig
import models.spiking_encoder as se
import main as neuma_main
from data.channel_harmonizer import CANONICAL_CHANNELS
from explainability.topomap import _DEFAULT_10_20 as ELEC_POS

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(8)            # prevent 200+ thread thrashing on CPU
DEVICE = torch.device("cpu")            # single sample — avoid GPU contention with the running ablation

SUBJECT       = "S01"
FOLD_TAG      = "repro_g0p0_balanced_37"
CKPT_DIR      = PHASE8 / "output" / "checkpoints" / FOLD_TAG
ENSEMBLE      = sorted(CKPT_DIR.glob(f"{FOLD_TAG}_fold01_e*.pt"))
BROCHURE_DIR  = Path("/home/nvidia/24PHD1314/Neuma_Model/DataSource/Dependencies/Brochure_Pages")
META_CSV      = Path(f"/home/nvidia/24PHD1314/Neuma_Model/NEUMA_PHASE3/{SUBJECT}/output/engagement_phase3d/engagement_metadata.csv")
RAW_ET_NPY    = Path(f"/home/nvidia/24PHD1314/Neuma_Model/NEUMA_PHASE3/{SUBJECT}/output/epochs/et_epochs_phase3d.npy")
RAW_EEG_NPY   = Path(f"/home/nvidia/24PHD1314/Neuma_Model/NEUMA_PHASE3/{SUBJECT}/output/epochs/eeg_epochs_phase3d.npy")

OUT = PHASE8 / "figures" / "case_study_S01"
for p in ["A", "B", "C", "D"]:
    (OUT / p).mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["LOW_ENGAGEMENT", "HIGH_ENGAGEMENT"]
EEG_SR      = 300        # Hz
ET_SR       = 120        # 600 samples / 5 s

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200,
                     "axes.titlesize": 12, "font.size": 9})


def log(msg): print(msg, flush=True)


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA + MODEL
# ════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log(" INS-HDGS-CMT — REAL case-study panels for S01 (fold-01)")
log("=" * 70)

log("\n[1] Building S01 dataset (global labels, zscore, 3-ch ET) …")
ds = NeumaGraphDataset(subject_ids=[SUBJECT], norm_mode="zscore",
                       label_mode="global", precompute_graphs=True,
                       use_disk_cache=True)
log(f"    {len(ds)} epochs loaded.")

# channel count actually fed to the model
sample0   = ds[0]
n_eeg_ch  = sample0["eeg_windows"].shape[1]
CH_NAMES  = CANONICAL_CHANNELS[:n_eeg_ch]
log(f"    EEG channels fed to model: {n_eeg_ch}  ({', '.join(CH_NAMES)})")


def build_model():
    m = neuma_main._make_model(n_eeg_ch=n_eeg_ch, n_classes=2,
                               ablation=AblationConfig.full())
    return m.to(DEVICE).eval()


def load_member(idx):
    m  = build_model()
    ck = torch.load(ENSEMBLE[idx], map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=True)
    m.eval()
    return m


def collate(sample):
    """One dataset item → batched model kwargs on DEVICE."""
    def t(x):
        x = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x, dtype=np.float32))
        return x.float().unsqueeze(0).to(DEVICE)
    return dict(
        eeg_windows  = t(sample["eeg_windows"]),
        adj_matrices = t(sample["adj_matrices"]),
        et_seq       = t(sample["et_seq"]),
        roi_vector   = t(sample["roi_vector"]),
        weighted_adjs= t(sample["weighted_adjs"]),
    )


log(f"\n[2] Loading {len(ENSEMBLE)}-member ensemble …")
members = [load_member(i) for i in range(len(ENSEMBLE))]
log(f"    loaded: {[p.name for p in ENSEMBLE]}")


# ── Ensemble pass on all epochs → pick representative sample ─────────────────
log("\n[3] Ensemble inference over all S01 epochs …")
probs_ens = np.zeros((len(ds), len(members)))
labels    = np.array([int(ds[i]["label"]) for i in range(len(ds))])
with torch.no_grad():
    for i in range(len(ds)):
        kw = collate(ds[i])
        for j, m in enumerate(members):
            out = m(**kw)
            probs_ens[i, j] = torch.softmax(out["logits"], -1)[0, 1].item()
mean_prob = probs_ens.mean(1)
pred      = (mean_prob >= 0.39).astype(int)        # opt_threshold for fold01 (from CSV)

meta = pd.read_csv(META_CSV)
pages = meta["label"].tolist()[:len(ds)]           # epoch → ImagePage_N

log("    idx  page          true  pred  mean_prob  correct")
for i in range(len(ds)):
    log(f"    {i:>3}  {pages[i]:<12}  {labels[i]}     {pred[i]}     "
        f"{mean_prob[i]:.3f}      {'Y' if pred[i]==labels[i] else 'n'}")

# representative = most-confident correctly-classified HIGH-engagement epoch
conf = np.abs(mean_prob - 0.5)
cand = [i for i in range(len(ds))
        if pred[i] == labels[i] and labels[i] == 1]
REP = max(cand, key=lambda i: conf[i]) if cand else int(np.argmax(conf))
log(f"\n    → Representative sample: epoch idx {REP}  "
    f"({pages[REP]}, true={CLASS_NAMES[labels[REP]]}, "
    f"ensemble p(HIGH)={mean_prob[REP]:.3f})")

rep_sample = ds[REP]
rep_kw     = collate(rep_sample)
rep_page   = pages[REP]


# ── Deep trace through member-0, capturing LIF spikes ────────────────────────
log("\n[4] Tracing representative sample through ensemble member e0 …")
captured_spikes = []
_orig_spike = se.surrogate_spike
def _spy(v, *a, **k):
    s = _orig_spike(v, *a, **k)
    captured_spikes.append(s.detach().cpu().numpy())
    return s
se.surrogate_spike = _spy

# capture fusion cross-attention (eeg_graph / eeg_et) — not in the eval dict
fusion_attn = {}
def _fusion_hook(mod, args, output):
    if isinstance(output, (tuple, list)) and len(output) > 1 and isinstance(output[1], dict):
        fusion_attn.update(output[1])
h = members[0].fusion.register_forward_hook(_fusion_hook)

try:
    with torch.no_grad():
        out = members[0](**rep_kw)
finally:
    se.surrogate_spike = _orig_spike
    h.remove()
log(f"    fusion attn keys: {list(fusion_attn.keys())}")

trace_prob = torch.softmax(out["logits"], -1)[0].cpu().numpy()
log(f"    member-0 p = {trace_prob}  | keys: {list(out.keys())}")
log(f"    captured {len(captured_spikes)} surrogate-spike tensors "
    f"(shapes e.g. {captured_spikes[0].shape if captured_spikes else 'none'})")


# helpers for the brochure
def load_brochure(page_label):
    f = BROCHURE_DIR / f"{page_label}.tif"
    return np.asarray(Image.open(f).convert("RGB"))

def raw_gaze(epoch_idx):
    """Recorded gaze in normalised [0,1] screen coords (left eye, cols 0,1)."""
    et = np.load(RAW_ET_NPY, allow_pickle=True).astype(np.float32)[epoch_idx]
    gx, gy = et[:, 0], et[:, 1]
    # average both eyes if right-eye cols present and valid
    if et.shape[1] >= 5:
        gx = np.nanmean(np.stack([et[:, 0], et[:, 3]]), 0)
        gy = np.nanmean(np.stack([et[:, 1], et[:, 4]]), 0)
    m = np.isfinite(gx) & np.isfinite(gy)
    return np.clip(gx[m], 0, 1), np.clip(gy[m], 0, 1)


# ════════════════════════════════════════════════════════════════════════════
# PANEL A — EEG Dynamic Graph-Spiking Branch
# ════════════════════════════════════════════════════════════════════════════
def panel_A():
    log("\n[A] EEG graph-spiking branch …")

    # ── A1: raw EEG, 2-s segment, all channels stacked ──────────────────────
    eeg_full = np.load(RAW_EEG_NPY, allow_pickle=True).astype(np.float32)[REP]   # (1500, C)
    C = min(eeg_full.shape[1], n_eeg_ch)
    seg = eeg_full[:int(2 * EEG_SR), :C]                                          # first 2 s
    t = np.arange(seg.shape[0]) / EEG_SR
    # per-channel z-score for display, then stack with fixed offset
    z = (seg - seg.mean(0)) / (seg.std(0) + 1e-6)
    dead = seg.std(0) < 1e-6                      # zero-filled (absent) channels
    off = 4.0
    fig, ax = plt.subplots(figsize=(11, 9))
    for c in range(C):
        ax.plot(t, z[:, c] + c * off, lw=0.6,
                color="0.7" if dead[c] else "C0")
    ax.set_yticks([c * off for c in range(C)])
    ax.set_yticklabels([f"{CH_NAMES[c]} (—)" if dead[c] else CH_NAMES[c]
                        for c in range(C)])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("EEG channel")
    ax.set_title(f"A1 · Preprocessed EEG epoch (2 s) — {SUBJECT} epoch {REP} ({rep_page})\n"
                 f"{C}-ch canonical montage  ({int((~dead).sum())} recorded, "
                 f"{int(dead.sum())} absent/zero-filled, grey)")
    ax.margins(x=0)
    fig.tight_layout(); fig.savefig(OUT / "A/A1_raw_eeg.png"); plt.close(fig)

    # ── A2: Pearson functional connectivity (DynamicGAT input) ──────────────
    adj = rep_sample["weighted_adjs"]
    adj = adj.numpy() if torch.is_tensor(adj) else np.asarray(adj)               # (W, C, C)
    A   = adj.mean(0)                                                            # window-mean
    np.fill_diagonal(A, np.nan)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(A, cmap="RdBu_r", vmin=np.nanmin(A), vmax=np.nanmax(A))
    ax.set_xticks(range(len(CH_NAMES))); ax.set_xticklabels(CH_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(CH_NAMES))); ax.set_yticklabels(CH_NAMES, fontsize=7)
    ax.set_title(f"A2 · Pearson functional connectivity (DynamicGAT input)\n"
                 f"{SUBJECT} epoch {REP} — mean over {adj.shape[0]} windows")
    fig.colorbar(im, ax=ax, fraction=0.046, label="weighted connectivity")
    fig.tight_layout(); fig.savefig(OUT / "A/A2_connectivity_matrix.png"); plt.close(fig)

    # ── A3: dynamic brain graph (strongest edges) ───────────────────────────
    Aplot = np.nan_to_num(adj.mean(0).copy())
    np.fill_diagonal(Aplot, 0)
    pos = {ch: ELEC_POS[ch] for ch in CH_NAMES if ch in ELEC_POS}
    nodes = list(pos.keys())
    idx   = {ch: CH_NAMES.index(ch) for ch in nodes}
    # keep top-15% strongest edges
    triu = np.array([Aplot[idx[a], idx[b]]
                     for ii, a in enumerate(nodes) for b in nodes[ii + 1:]])
    thr = np.quantile(np.abs(triu), 0.85) if triu.size else 0
    G = nx.Graph()
    for ch in nodes: G.add_node(ch)
    for ii, a in enumerate(nodes):
        for b in nodes[ii + 1:]:
            w = Aplot[idx[a], idx[b]]
            if abs(w) >= thr and thr > 0:
                G.add_edge(a, b, weight=abs(w))
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.add_patch(plt.Circle((0, 0), 1.08, fill=False, lw=1.5, color="gray"))
    ax.plot([0, -0.1, 0, 0.1, 0], [1.18, 1.06, 1.10, 1.06, 1.18], color="gray", lw=1.5)  # nose
    if G.number_of_edges():
        ews = np.array([G[a][b]["weight"] for a, b in G.edges()])
        nx.draw_networkx_edges(G, pos, ax=ax, width=1 + 4 * ews / ews.max(),
                               edge_color=ews, edge_cmap=plt.cm.viridis, alpha=0.8)
    deg = dict(G.degree())
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=420,
                           node_color=[deg[n] for n in G.nodes()], cmap="autumn")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    ax.set_title(f"A3 · Dynamic brain graph (top-15% edges)\n{SUBJECT} epoch {REP}")
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "A/A3_brain_graph.png"); plt.close(fig)

    # ── A4: LIF spike raster ────────────────────────────────────────────────
    # surrogate-spike calls = n_lif_layers × time_steps ; each (B, D) binary.
    if captured_spikes:
        arr = np.stack([s[0] for s in captured_spikes], 0)        # (n_calls, D)
        # group into the first LIF layer (first SNN_TIME_STEPS calls)
        T = se.__dict__.get("SNN_TIME_STEPS", None)
        raster = arr.T                                            # (D neurons, n_calls time)
        fig, ax = plt.subplots(figsize=(11, 7))
        ys, xs = np.where(raster > 0.5)
        ax.scatter(xs, ys, s=6, marker="|", color="k")
        ax.set_xlabel("LIF simulation step")
        ax.set_ylabel("Neuron index")
        ax.set_title(f"A4 · LIF spike raster (SpikingEEGEncoder) — {SUBJECT} epoch {REP}\n"
                     f"{raster.shape[0]} neurons × {raster.shape[1]} steps · "
                     f"firing rate {raster.mean():.3f}")
        ax.set_xlim(-0.5, raster.shape[1] - 0.5)
        ax.margins(y=0.01)
        fig.tight_layout(); fig.savefig(OUT / "A/A4_spike_raster.png"); plt.close(fig)
    else:
        log("    A4 skipped — no spikes captured (SNN branch inactive?)")
    log("    Panel A done.")


# ════════════════════════════════════════════════════════════════════════════
# PANEL B — Eye-Tracking Branch
# ════════════════════════════════════════════════════════════════════════════
def panel_B():
    log("\n[B] Eye-tracking branch …")
    img = load_brochure(rep_page)
    H, W = img.shape[:2]
    gx, gy = raw_gaze(REP)
    px, py = gx * W, gy * H
    n = len(px)

    # ── B1: gaze trajectory ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9 * H / W))
    ax.imshow(img)
    ax.plot(px, py, "-", color="cyan", lw=1.0, alpha=0.6)
    sc = ax.scatter(px, py, c=np.arange(n), cmap="plasma", s=18, zorder=3)
    ax.scatter(px[0], py[0], marker="o", s=130, facecolor="none",
               edgecolor="lime", lw=2, label="start")
    ax.scatter(px[-1], py[-1], marker="s", s=130, facecolor="none",
               edgecolor="red", lw=2, label="end")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="gaze sample (time →)")
    ax.set_title(f"B1 · Gaze trajectory on {rep_page} — {SUBJECT} epoch {REP}")
    ax.legend(loc="upper right"); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "B/B1_gaze_trajectory.png"); plt.close(fig)

    # ── B2: fixation map (dispersion-based) ─────────────────────────────────
    # velocity in normalised units/sample; low velocity ⇒ fixation
    vel = np.hypot(np.diff(gx, prepend=gx[0]), np.diff(gy, prepend=gy[0]))
    is_fix = vel < np.quantile(vel, 0.6)
    fixations = []          # (cx, cy, dur_ms)
    i = 0
    while i < n:
        if is_fix[i]:
            j = i
            while j < n and is_fix[j]:
                j += 1
            if j - i >= 3:                     # >= ~25 ms
                fixations.append((gx[i:j].mean(), gy[i:j].mean(),
                                  (j - i) / ET_SR * 1000))
            i = j
        else:
            i += 1
    fig, ax = plt.subplots(figsize=(9, 9 * H / W))
    ax.imshow(img)
    if fixations:
        durs = np.array([f[2] for f in fixations])
        for k, (cx, cy, d) in enumerate(fixations):
            r = 18 + 90 * d / durs.max()
            ax.add_patch(Circle((cx * W, cy * H), r, color="orange",
                                 alpha=0.45, ec="darkorange", lw=1.5))
            ax.text(cx * W, cy * H, str(k + 1), ha="center", va="center",
                    fontsize=8, weight="bold")
    ax.set_title(f"B2 · Fixation map ({len(fixations)} fixations, circle ∝ dwell) — "
                 f"{SUBJECT} epoch {REP}")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "B/B2_fixation_map.png"); plt.close(fig)

    # ── B3: ROI attention heatmap (real et_roi_attn) ────────────────────────
    roi = out.get("et_roi_attn")
    if roi is not None:
        roi = roi[0].detach().cpu().numpy()                     # (N_ROIS,)
        roi = roi - roi.min()
        roi = roi / (roi.max() + 1e-9)
        # arrange N_ROIS over the page as a 2×5 grid (learned attention regions)
        nrow, ncol = 2, int(np.ceil(len(roi) / 2))
        grid = np.zeros((nrow, ncol))
        for idx, v in enumerate(roi):
            grid[idx // ncol, idx % ncol] = v
        fig, ax = plt.subplots(figsize=(9, 9 * H / W))
        ax.imshow(img)
        ax.imshow(grid, cmap="jet", alpha=0.45, extent=[0, W, H, 0],
                  interpolation="bilinear", aspect="auto")
        for idx, v in enumerate(roi):
            r, c = idx // ncol, idx % ncol
            ax.text((c + 0.5) * W / ncol, (r + 0.5) * H / nrow,
                    f"R{idx}\n{v:.2f}", ha="center", va="center",
                    color="white", fontsize=8, weight="bold")
        ax.set_title(f"B3 · ROI attention (ET-encoder, {len(roi)} learned ROIs) — "
                     f"{SUBJECT} epoch {REP}")
        ax.axis("off")
        fig.tight_layout(); fig.savefig(OUT / "B/B3_roi_attention.png"); plt.close(fig)
    else:
        log("    B3 skipped — et_roi_attn not returned")
    log("    Panel B done.")


# ════════════════════════════════════════════════════════════════════════════
# PANEL C — NeuroFusion Transformer
# ════════════════════════════════════════════════════════════════════════════
def panel_C():
    log("\n[C] NeuroFusion transformer …")
    attn = fusion_attn or {}

    def plot_attn(key, fname, title):
        A = attn.get(key)
        if A is None:
            log(f"    {title} skipped — attn['{key}'] is None")
            return
        A = A[0].detach().cpu().numpy()
        A = np.atleast_2d(A)
        nq, nk = A.shape
        kind = "Graph" if "graph" in key else "ET"
        fig, ax = plt.subplots(figsize=(7.5, 2.6 if nq == 1 else 5.5))
        im = ax.imshow(A, cmap="magma", aspect="auto", vmin=0)
        ax.set_xticks(range(nk))
        ax.set_xticklabels([f"{kind}\nwin{j}" if nk == 10 else f"k{j}"
                            for j in range(nk)], fontsize=8)
        ax.set_yticks(range(nq))
        ax.set_yticklabels(["EEG"] if nq == 1 else [f"q{i}" for i in range(nq)])
        ax.set_xlabel(f"{kind} key tokens"); ax.set_ylabel("EEG query")
        ax.set_title(title)
        for j in range(nk):
            ax.text(j, 0, f"{A[0, j]:.2f}", ha="center", va="center",
                    color="w" if A[0, j] < A.max() * 0.6 else "k", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
        fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)

    plot_attn("eeg_graph", "C/C1_xattn_eeg_graph.png",
              f"C1 · Cross-attention  EEG ← Graph — {SUBJECT} epoch {REP}")
    plot_attn("eeg_et", "C/C2_xattn_eeg_et.png",
              f"C2 · Cross-attention  EEG ← ET — {SUBJECT} epoch {REP}")

    # ── C3: fused multimodal representation ─────────────────────────────────
    fused = out["fused"][0].detach().cpu().numpy()
    D = fused.size
    side = int(np.ceil(np.sqrt(D)))
    pad = np.full(side * side, np.nan); pad[:D] = fused
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pad.reshape(side, side), cmap="coolwarm")
    ax.set_title(f"C3 · Fused multimodal representation ({D}-d) — {SUBJECT} epoch {REP}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, label="activation")
    fig.tight_layout(); fig.savefig(OUT / "C/C3_fused_representation.png"); plt.close(fig)
    log("    Panel C done.")


# ════════════════════════════════════════════════════════════════════════════
# PANEL D — Neuro-Symbolic Decision
# ════════════════════════════════════════════════════════════════════════════
def panel_D():
    log("\n[D] Neuro-symbolic decision …")
    ra = out.get("rule_act")
    has_rules = float(out.get("has_rules", torch.tensor(1.0)).item()) > 0.5

    # ── D1: rule activations ────────────────────────────────────────────────
    if ra is not None and has_rules:
        acts = ra[0].detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(8.5, 5))
        bars = ax.bar(range(len(acts)), acts,
                      color=plt.cm.viridis(acts / (acts.max() + 1e-9)))
        ax.set_xticks(range(len(acts)))
        ax.set_xticklabels([f"R{i+1}" for i in range(len(acts))])
        ax.set_xlabel("Neuro-symbolic rule"); ax.set_ylabel("Activation strength")
        ax.set_title(f"D1 · Rule activations ({len(acts)} rules) — {SUBJECT} epoch {REP}")
        for b, v in zip(bars, acts):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "D/D1_rule_activations.png"); plt.close(fig)
    else:
        log("    D1 skipped — rule_info not returned")

    # ── D2: prediction output ───────────────────────────────────────────────
    p_member = trace_prob                       # member-0 softmax
    p_ens    = mean_prob[REP]                   # ensemble p(HIGH)
    true     = int(labels[REP])
    pred_lbl = int(p_ens >= 0.39)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    bars = ax.bar(CLASS_NAMES, [1 - p_ens, p_ens],
                  color=["#4C72B0", "#C44E52"])
    ax.set_ylim(0, 1); ax.set_ylabel("Ensemble probability")
    ax.set_title("D2 · Class probabilities")
    for b, v in zip(bars, [1 - p_ens, p_ens]):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax = axes[1]; ax.axis("off")
    ok = "✓ CORRECT" if pred_lbl == true else "✗ WRONG"
    txt = (f"Subject / fold : {SUBJECT}  (fold-01)\n"
           f"Epoch / page   : {REP}  ({rep_page})\n\n"
           f"Ground truth   : {CLASS_NAMES[true]}\n"
           f"Predicted      : {CLASS_NAMES[pred_lbl]}   {ok}\n\n"
           f"p(LOW)         : {1-p_ens:.3f}\n"
           f"p(HIGH)        : {p_ens:.3f}\n"
           f"threshold      : 0.39 (fold-01 opt)\n\n"
           f"member-0 p     : LOW {p_member[0]:.3f} / HIGH {p_member[1]:.3f}\n"
           f"ensemble       : 5 members")
    ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=11,
            bbox=dict(boxstyle="round", fc="#f3f3f3", ec="gray"))
    fig.suptitle(f"D2 · Engagement prediction — {SUBJECT} epoch {REP}", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "D/D2_prediction.png", bbox_inches="tight"); plt.close(fig)
    log("    Panel D done.")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    panel_A()
    panel_B()
    panel_C()
    panel_D()

    # provenance record
    rec = dict(subject=SUBJECT, fold="fold-01", experiment=FOLD_TAG,
               representative_epoch=int(REP), page=rep_page,
               ground_truth=CLASS_NAMES[int(labels[REP])],
               ensemble_p_high=float(mean_prob[REP]),
               member0_p=[float(x) for x in trace_prob],
               n_eeg_channels=int(n_eeg_ch), channels=CH_NAMES,
               ensemble_members=[p.name for p in ENSEMBLE])
    (OUT / "provenance.json").write_text(json.dumps(rec, indent=2))
    log(f"\n✓ All panels written to {OUT}")
    log(json.dumps(rec, indent=2))
