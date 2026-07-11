"""
Generate REAL figures for every heading of the
"Cross-Modal Fusion and Neuro-Symbolic Reasoning" diagram (Fig 4),
using the trained local checkpoint + real S01 data (no synthetic placeholders).

Source of truth:
  checkpoint : src/model/output/checkpoints/ins_hdgs_cmt/ins_hdgs_cmt_single.pt
  data       : src/data_pipeline/04_segmentation/S01/output/graph_cache/*/ep*.pt
"""
import os, sys, glob
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(r"D:\Neuma_Research")
MODEL  = ROOT / "src" / "model"
CKPT   = MODEL / "output" / "checkpoints" / "ins_hdgs_cmt" / "ins_hdgs_cmt_single.pt"
CACHE  = sorted(glob.glob(str(ROOT / "src/data_pipeline/04_segmentation/S01/output/graph_cache/*/ep*.pt")))
OUT    = ROOT / "experiments" / "NEUMA_PHASE8" / "figures" / "fig4_fusion_realdata"
OUT.mkdir(parents=True, exist_ok=True)

os.chdir(MODEL)
sys.path.insert(0, str(MODEL))
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
DEV = torch.device("cpu")

from models.ins_hdgs_cmt import INS_HDGS_CMT, AblationConfig
import main as neuma_main

CLASS = ["LOW_ENGAGEMENT", "HIGH_ENGAGEMENT"]

# ── build model + load real weights ──────────────────────────────────────────
N_EEG = 19
model = neuma_main._make_model(n_eeg_ch=N_EEG, n_classes=2, ablation=AblationConfig.full()).to(DEV).eval()
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
print(f"[load] missing={len(missing)} unexpected={len(unexpected)} (epoch {ck.get('epoch')}, best_val_bal {ck.get('best_val_bal')})")

def collate(d):
    f = lambda x: x.float().unsqueeze(0).to(DEV)
    return dict(eeg_windows=f(d["eeg_windows"]), adj_matrices=f(d["adj_matrices"]),
                et_seq=f(d["et_seq"]), roi_vector=f(d["roi_vector"]),
                weighted_adjs=f(d["weighted_adjs"]))

# ── pick the most-confident sample as the representative ─────────────────────
samples = [torch.load(p, map_location="cpu", weights_only=False) for p in CACHE]
with torch.no_grad():
    phigh = []
    for s in samples:
        o = model(**collate(s))
        phigh.append(torch.softmax(o["logits"], -1)[0, 1].item())
phigh = np.array(phigh)
REP = int(np.argmax(np.abs(phigh - 0.5)))
print(f"[rep] epoch {REP}  p(HIGH)={phigh[REP]:.3f}  (of {len(samples)} real S01 epochs)")

# ── capture intermediates on the representative sample ───────────────────────
cap = {}
def pre_fusion(mod, args, kwargs):
    cap["eeg_emb"]   = args[0].detach()[0].cpu().numpy()
    cap["graph_emb"] = args[1].detach()[0].cpu().numpy()
    cap["et_emb"]    = (args[2].detach()[0].cpu().numpy() if args[2] is not None else None)
def post_fusion(mod, args, output):
    fused, attn = output
    cap["fused"] = fused.detach()[0].cpu().numpy()
    cap["attn_eeg_graph"] = attn["eeg_graph"].detach()[0].cpu().numpy() if attn.get("eeg_graph") is not None else None
    cap["attn_eeg_et"]    = attn["eeg_et"].detach()[0].cpu().numpy()    if attn.get("eeg_et")    is not None else None
def post_gate(mod, args, output):
    cap["gate"] = output.detach()[0].cpu().numpy()

h1 = model.fusion.register_forward_pre_hook(pre_fusion, with_kwargs=True)
h2 = model.fusion.register_forward_hook(post_fusion)
h3 = model.fusion.gate.register_forward_hook(post_gate)

with torch.no_grad():
    out = model(**collate(samples[REP]))
for h in (h1, h2, h3): h.remove()

# self-attention over the 3 modality tokens — recomputed manually from the real
# captured embeddings (PyTorch's TransformerEncoder fast-path hides attn weights).
with torch.no_grad():
    fu = model.fusion
    mod = fu.mod_embed.weight                                   # (3, D)
    toks = torch.stack([torch.as_tensor(cap["eeg_emb"]) + mod[0],
                        torch.as_tensor(cap["graph_emb"]) + mod[1],
                        torch.as_tensor(cap["et_emb"]) + mod[2]], 0).unsqueeze(0).float()  # (1,3,D)
    x = toks
    last_w = None
    for layer in fu.self_attn_encoder.layers:                   # replicate norm_first layer
        h = layer.norm1(x)
        attn_out, w = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + layer.dropout1(attn_out)
        x = x + layer._ff_block(layer.norm2(x))
        last_w = w
    if last_w is not None:
        cap["self_attn"] = last_w[0].cpu().numpy()

# neuro-symbolic pieces (recompute agg from the real rule layer on the real fused vec)
rl = model.rule_layer
with torch.no_grad():
    z = torch.as_tensor(cap["fused"]).float().unsqueeze(0)
    keys = rl.key_proj(z)
    sim  = torch.einsum("rd,bd->br", rl.rule_queries, keys) / (rl.embed_dim ** 0.5)
    acts = torch.softmax(sim / rl.temperature, -1)[0].cpu().numpy()          # μ  fuzzy membership (8,)
    rlog = torch.stack([hd(keys) for hd in rl.rule_heads], 1)                 # (1,8,2)
    votes = torch.softmax(rlog, -1)[0].cpu().numpy()                          # r  per-rule vote (8,2)
    agg   = torch.softmax((torch.as_tensor(acts)[:,None]*rlog[0]).sum(0), -1).cpu().numpy()  # h (2,)
p_pred = torch.softmax(out["logits"], -1)[0].cpu().numpy()                    # final (2,)

# ── plotting helpers ─────────────────────────────────────────────────────────
plt.rcParams.update({"savefig.dpi": 200, "font.size": 11, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "figure.autolayout": True})
MODS = ["EEG", "Graph", "ET"]
def vec_heat(v, title, fname, cbar="activation", cmap="coolwarm"):
    D = v.size; side = int(np.ceil(np.sqrt(D)))
    pad = np.full(side*side, np.nan); pad[:D] = v
    fig, ax = plt.subplots(figsize=(5.4, 5))
    im = ax.imshow(pad.reshape(side, side), cmap=cmap)
    ax.set_title(f"{title}\n(real, {D}-d · S01 epoch {REP})"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, label=cbar)
    fig.savefig(OUT/fname); plt.close(fig)

# 01–03  embeddings
vec_heat(cap["eeg_emb"],   "01 · EEG Embedding (graph-GAT + LIF)", "01_eeg_embedding.png")
vec_heat(cap["graph_emb"], "02 · Graph Embedding (dyn. connectivity)", "02_graph_embedding.png")
vec_heat(cap["et_emb"],    "03 · ET Embedding (eye-tracking attn)", "03_et_embedding.png")

# 04  self-attention over the three modality tokens
if "self_attn" in cap:
    A = np.atleast_2d(cap["self_attn"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(A, cmap="magma", vmin=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(MODS); ax.set_yticks(range(3)); ax.set_yticklabels(MODS)
    ax.set_xlabel("key token"); ax.set_ylabel("query token")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                    color="w" if A[i,j] < A.max()*0.6 else "k", fontsize=10)
    ax.set_title(f"04 · Self-Attention over 3 modality tokens\n(real · S01 epoch {REP})")
    fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
    fig.savefig(OUT/"04_self_attention.png"); plt.close(fig)

# 05 / 06  directed cross-attention (EEG query over Graph / ET windows)
def cross_bar(a, kind, fname, idx):
    a = np.atleast_2d(a).ravel()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(range(len(a)), a, color="#6A4C93")
    ax.set_xticks(range(len(a))); ax.set_xticklabels([f"{kind}\nwin{j}" for j in range(len(a))], fontsize=8)
    ax.set_ylabel("attention weight"); ax.set_xlabel(f"{kind} key tokens")
    ax.set_title(f"{idx} · Cross-Attention  EEG → {kind}\n(real · S01 epoch {REP})")
    for j,v in enumerate(a): ax.text(j, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.savefig(OUT/fname); plt.close(fig)
if cap.get("attn_eeg_graph") is not None: cross_bar(cap["attn_eeg_graph"], "Graph", "05_crossattn_eeg_graph.png", "05")
if cap.get("attn_eeg_et")    is not None: cross_bar(cap["attn_eeg_et"],    "ET",    "06_crossattn_eeg_et.png",   "06")

# 07  gated residual fusion — the learned gate vector
vec_heat(cap["gate"], "07 · Gated Residual Fusion (learned gate σ)", "07_gated_fusion.png",
         cbar="gate value ∈[0,1]", cmap="viridis")

# 08  joint multimodal representation Z
vec_heat(cap["fused"], "08 · Joint Multimodal Representation Z", "08_joint_representation.png")

# 09  fuzzy membership μ
fig, ax = plt.subplots(figsize=(7.6, 4))
ax.bar(range(len(acts)), acts, color=plt.cm.viridis(acts/(acts.max()+1e-9)))
ax.set_xticks(range(len(acts))); ax.set_xticklabels([f"R{i+1}" for i in range(len(acts))])
ax.set_ylabel("membership μ ∈[0,1]"); ax.set_xlabel("learned rule premise")
ax.set_title(f"09 · Fuzzy Membership (premise match)\n(real · S01 epoch {REP})")
for i,v in enumerate(acts): ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
fig.savefig(OUT/"09_fuzzy_membership.png"); plt.close(fig)

# 10  differentiable rules (8 x 2 votes)
fig, ax = plt.subplots(figsize=(5.6, 5))
im = ax.imshow(votes, cmap="magma", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(2)); ax.set_xticklabels(CLASS, rotation=20, ha="right")
ax.set_yticks(range(len(votes))); ax.set_yticklabels([f"R{i+1}" for i in range(len(votes))])
for i in range(votes.shape[0]):
    for j in range(votes.shape[1]):
        ax.text(j, i, f"{votes[i,j]:.2f}", ha="center", va="center",
                color="w" if votes[i,j] < 0.6 else "k", fontsize=9)
ax.set_title(f"10 · Differentiable Rules (×8) — per-rule vote\n(real · S01 epoch {REP})")
fig.colorbar(im, ax=ax, fraction=0.046, label="vote probability")
fig.savefig(OUT/"10_differentiable_rules.png"); plt.close(fig)

# 11  constraint aggregation h
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.bar(CLASS, agg, color=["#4C72B0", "#C44E52"])
ax.set_ylim(0,1); ax.set_ylabel("aggregated constraint h ∈[0,1]")
for i,v in enumerate(agg): ax.text(i, v+0.02, f"{v:.3f}", ha="center", fontsize=11)
ax.set_title(f"11 · Constraint Aggregation\n(real · S01 epoch {REP})")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
fig.savefig(OUT/"11_constraint_aggregation.png"); plt.close(fig)

# 12  consumer engagement prediction
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.bar(CLASS, p_pred, color=["#4C72B0", "#C44E52"])
ax.set_ylim(0,1); ax.set_ylabel("P(class)")
for i,v in enumerate(p_pred): ax.text(i, v+0.02, f"{v:.3f}", ha="center", fontsize=11)
ax.set_title(f"12 · Consumer Engagement Prediction\npred={CLASS[int(p_pred[1]>=0.5)]} (real · S01 epoch {REP})")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
fig.savefig(OUT/"12_prediction.png"); plt.close(fig)

print("saved", len(list(OUT.glob('*.png'))), "figures to", OUT)
for p in sorted(OUT.glob("*.png")): print("  ", p.name)
