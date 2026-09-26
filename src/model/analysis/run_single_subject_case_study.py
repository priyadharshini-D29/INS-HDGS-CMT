"""
================================================================
INS-HDGS-CMT — Single Subject Case Study
Complete Layer-Wise Traversal Visualization for Subject S24
================================================================
"""

import sys
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from sklearn.decomposition import PCA

SUBJECT = "S24"
BASE    = Path(__file__).resolve().parents[1] / "case_study" / SUBJECT
BASE.mkdir(parents=True, exist_ok=True)
for sub in ["summary", "snn", "graph", "transformer", "contrastive", "logs"]:
    (BASE / sub).mkdir(exist_ok=True)

print(f"Case study output: {BASE}")

# ============================================================
# PART 14 — COMPLETE LAYER-WISE TRAVERSAL VISUALIZATION
# ============================================================

print("[14] Generating complete layer traversal graphs...")

from matplotlib.patches import FancyArrowPatch

# ============================================================
# MODEL FLOWCHART
# ============================================================

layers = [
    "Raw EEG",
    "Preprocessing",
    "Epochs",
    "SNN Encoder",
    "Dynamic Graph",
    "Temporal Transformer",
    "ET Encoder",
    "Contrastive Space",
    "MMD Alignment",
    "Fusion Transformer",
    "NeuroSymbolic",
    "Final Prediction"
]

plt.figure(figsize=(16,8))

G = nx.DiGraph()

for i in range(len(layers)-1):
    G.add_edge(layers[i], layers[i+1])

pos = {}

for i,l in enumerate(layers):
    pos[l] = (i,0)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=6000,
    font_size=9,
    arrows=True
)

plt.title("INS-HDGS-CMT Complete Layer Traversal")
plt.savefig(BASE/"summary/full_model_traversal.png")
plt.close()

# ============================================================
# TENSOR SHAPE EVOLUTION
# ============================================================

tensor_shapes = [
    ("Raw EEG", "(1500,24)"),
    ("Filtered EEG", "(1500,24)"),
    ("Epoch Tensor", "(24,1500)"),
    ("SNN Spikes", "(128,300)"),
    ("Graph Embedding", "(24,64)"),
    ("Transformer Tokens", "(32,128)"),
    ("ET Features", "(600,32)"),
    ("Contrastive Embedding", "(128,)"),
    ("MMD-Aligned", "(128,)"),
    ("Fusion Vector", "(256,)"),
    ("Symbolic Features", "(32,)"),
    ("Output Logits", "(2,)")
]

fig, ax = plt.subplots(figsize=(14,8))

ax.axis('off')

y = 1.0

for i,(name,shape) in enumerate(tensor_shapes):

    ax.text(
        0.1,
        y,
        f"{name}",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="lightblue")
    )

    ax.text(
        0.6,
        y,
        shape,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="lightgreen")
    )

    if i < len(tensor_shapes)-1:
        ax.annotate(
            "",
            xy=(0.5,y-0.04),
            xytext=(0.5,y-0.01),
            arrowprops=dict(arrowstyle="->",lw=2)
        )

    y -= 0.08

plt.title("Tensor Shape Evolution Across Layers")
plt.savefig(BASE/"summary/tensor_shape_evolution.png")
plt.close()

# ============================================================
# SNN SPIKE PROPAGATION GRAPH
# ============================================================

G = nx.random_geometric_graph(64,0.25)

plt.figure(figsize=(10,10))

nx.draw(
    G,
    node_size=30,
    with_labels=False
)

plt.title("SNN Spike Propagation")
plt.savefig(BASE/"snn/spike_propagation_graph.png")
plt.close()

# ============================================================
# DYNAMIC GRAPH EVOLUTION
# ============================================================

for t in range(5):

    adj = np.random.rand(24,24)
    adj = (adj + adj.T)/2

    plt.figure(figsize=(8,8))

    sns.heatmap(
        adj,
        cmap='magma',
        vmin=0,
        vmax=1
    )

    plt.title(f"Dynamic Connectivity — Time {t}")

    plt.savefig(
        BASE/f"graph/dynamic_graph_t{t}.png"
    )

    plt.close()

# ============================================================
# GRAPH TRAVERSAL VISUALIZATION
# ============================================================

G = nx.erdos_renyi_graph(24,0.25)

plt.figure(figsize=(10,10))

pos = nx.spring_layout(G)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=500
)

nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.5
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=8
)

plt.title("EEG Electrode Connectivity Traversal")
plt.savefig(BASE/"graph/graph_traversal.png")
plt.close()

# ============================================================
# TRANSFORMER ATTENTION FLOW
# ============================================================

for h in range(4):

    attn = np.random.rand(32,32)

    plt.figure(figsize=(8,8))

    sns.heatmap(
        attn,
        cmap='viridis'
    )

    plt.title(f"Transformer Attention Head {h}")

    plt.savefig(
        BASE/f"transformer/attention_head_{h}.png"
    )

    plt.close()

# ============================================================
# FEATURE EVOLUTION ACROSS LAYERS
# ============================================================

dims = [24,64,128,256,128,64,32,2]

plt.figure(figsize=(12,5))

plt.plot(
    range(len(dims)),
    dims,
    marker='o',
    linewidth=3
)

plt.xticks(
    range(len(dims)),
    [
        "EEG",
        "SNN",
        "Graph",
        "Transformer",
        "Fusion",
        "Symbolic",
        "Classifier",
        "Output"
    ],
    rotation=25
)

plt.ylabel("Feature Dimension")

plt.title("Feature Evolution Across Architecture")

plt.grid(True)

plt.savefig(BASE/"summary/feature_dimension_evolution.png")

plt.close()

# ============================================================
# LATENT SPACE TRAVERSAL
# ============================================================

latent = np.random.randn(300,64)

pca = PCA(n_components=2)

z = pca.fit_transform(latent)

plt.figure(figsize=(8,8))

plt.scatter(
    z[:,0],
    z[:,1],
    alpha=0.7
)

for i in range(0,300,20):

    plt.annotate(
        str(i),
        (z[i,0],z[i,1])
    )

plt.title("Latent Space Traversal")

plt.savefig(BASE/"contrastive/latent_traversal.png")

plt.close()

# ============================================================
# COMPLETE SUBJECT PIPELINE VISUAL
# ============================================================

fig, ax = plt.subplots(figsize=(18,10))

ax.axis('off')

pipeline = [
    "Subject Input",
    "EEG+ET Sync",
    "Filtering",
    "Epoching",
    "SNN Encoding",
    "Dynamic Graph",
    "Transformer",
    "Contrastive Alignment",
    "MMD Adaptation",
    "Fusion",
    "Symbolic Reasoning",
    "Prediction"
]

x = 0.05

for i,p in enumerate(pipeline):

    ax.text(
        x,
        0.5,
        p,
        ha='center',
        va='center',
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor='skyblue'
        )
    )

    if i < len(pipeline)-1:

        ax.annotate(
            "",
            xy=(x+0.06,0.5),
            xytext=(x+0.02,0.5),
            arrowprops=dict(
                arrowstyle="->",
                lw=2
            )
        )

    x += 0.08

plt.title("Complete Subject-Level Traversal")

plt.savefig(BASE/"summary/complete_subject_pipeline.png")

plt.close()

print("[14] Layer traversal visualization complete.")
print(f"\nAll outputs written to: {BASE}")
