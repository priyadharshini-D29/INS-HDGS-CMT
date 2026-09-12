"""
================================================================
NEUMA Phase 8C — Dynamic Temporal Graph Attention Network
================================================================
Processes temporally-windowed EEG connectivity graphs using
a two-layer GAT per window, then aggregates window embeddings
with either a Transformer encoder or a BiLSTM.

Input
-----
    node_feats   : (B, W, C, F)   band-power features per window
    adj_matrices : (B, W, C, C)   per-window adjacency
                   (should already be ROI-modulated before this layer)

Processing pipeline
-------------------
    1. Reshape to (B×W, C, F) and (B×W, C, C)
    2. Two-layer GATEncoder with multi-head attention
       → node_emb : (B×W, C, D)
    3. Mean-pool nodes  → window_emb : (B, W, D)
    4. Temporal aggregation:
         a) Transformer encoder (default)  — global temporal context
         b) BiLSTM                         — sequential dynamics
    5. Mean-pool time  → graph_emb : (B, D)
    6. LayerNorm

Output
------
    graph_emb  : (B, embed_dim)  — final dynamic graph embedding
    attn       : dict with per-layer attention matrices for GradCAM

Architecture notes
------------------
The per-window GATEncoder is shared (weight-tied) across all W
windows, following the parameter-efficient dynamic graph learning
paradigm of Temporal Graph Networks (Rossi et al., 2020).

For temporal aggregation, the Transformer encoder is preferred
because it captures long-range temporal dependencies with O(W²)
complexity (W=10 is small). BiLSTM is available as an ablation.
================================================================
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .gat_encoder import GATEncoder


class DynamicGAT(nn.Module):
    """
    Dynamic Temporal Graph Attention Network.

    Parameters
    ----------
    in_dim          : node feature dimension  (5 for band-power features)
    embed_dim       : output embedding dimension
    gat_heads       : number of attention heads in GAT Layer 1
    gat_head_dim    : per-head dimension in GAT Layer 1
    temporal_mode   : "transformer" | "bilstm"
    n_windows       : number of temporal windows  (W)
    t_nhead         : heads in the Transformer encoder
    t_layers        : Transformer encoder depth
    t_ff_dim        : Transformer feedforward dimension
    lstm_layers     : BiLSTM depth  (used when temporal_mode="bilstm")
    dropout         : dropout probability
    """

    def __init__(
        self,
        in_dim        : int   = 5,
        embed_dim     : int   = 128,
        gat_heads     : int   = 4,
        gat_head_dim  : int   = 32,
        temporal_mode : Literal["transformer", "bilstm"] = "transformer",
        n_windows     : int   = 10,
        t_nhead       : int   = 4,
        t_layers      : int   = 2,
        t_ff_dim      : int   = 256,
        lstm_layers   : int   = 2,
        dropout       : float = 0.20,
    ):
        super().__init__()

        self.temporal_mode = temporal_mode
        self.embed_dim     = embed_dim

        # ── Shared (weight-tied) GAT across all windows ──────────────────────
        self.gat = GATEncoder(
            in_dim   = in_dim,
            head_dim = gat_head_dim,
            heads    = gat_heads,
            out_dim  = embed_dim,
            dropout  = dropout,
        )

        # ── Temporal aggregation ─────────────────────────────────────────────
        if temporal_mode == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model        = embed_dim,
                nhead          = t_nhead,
                dim_feedforward= t_ff_dim,
                dropout        = dropout,
                batch_first    = True,
                norm_first     = True,    # pre-norm for stability
            )
            self.temporal = nn.TransformerEncoder(
                enc_layer, num_layers=t_layers,
                enable_nested_tensor=False,
            )
            # Learnable temporal positional embeddings for W windows
            self.pos_embed = nn.Embedding(n_windows, embed_dim)

        elif temporal_mode == "bilstm":
            self.temporal = nn.LSTM(
                input_size   = embed_dim,
                hidden_size  = embed_dim // 2,    # bidirectional → full embed_dim
                num_layers   = lstm_layers,
                batch_first  = True,
                bidirectional= True,
                dropout      = dropout if lstm_layers > 1 else 0.0,
            )
            self.pos_embed = None
        else:
            raise ValueError(
                f"temporal_mode must be 'transformer' or 'bilstm', got {temporal_mode!r}"
            )

        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Learned attention pooling over electrode nodes (replaces mean-pool)
        self.node_pool = nn.Linear(embed_dim, 1, bias=False)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        node_feats  : torch.Tensor,   # (B, W, C, F)
        adj_matrices: torch.Tensor,   # (B, W, C, C)
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        node_feats   : (B, W, C, F)   node feature tensor
        adj_matrices : (B, W, C, C)   adjacency matrices (ROI-modulated or raw)

        Returns
        -------
        graph_emb    : (B, embed_dim)
        attn         : dict with GAT attention weights
                       {'l1_attn': (B*W, C, C), 'l2_attn': (B*W, C, C)}
        window_seq   : (B, W, embed_dim)  per-window embeddings after temporal encoder
        """
        B, W, C, F = node_feats.shape

        # ── GAT: process all (B×W) graphs in one vectorised pass ────────────
        x_bw   = node_feats.view(B * W, C, F)
        adj_bw = adj_matrices.view(B * W, C, C)

        node_emb, gat_attn = self.gat(x_bw, adj_bw)     # (B*W, C, D)

        # Attention pooling over electrode nodes — preserves spatial structure
        # softmax over C electrodes, weighted sum → (B*W, D)
        pool_attn  = torch.softmax(self.node_pool(node_emb), dim=1)  # (B*W, C, 1)
        window_emb = (node_emb * pool_attn).sum(dim=1)               # (B*W, D)
        window_emb = window_emb.view(B, W, self.embed_dim)           # (B, W, D)

        # ── Temporal aggregation ─────────────────────────────────────────────
        if self.temporal_mode == "transformer":
            # Positional embeddings for W time-window positions
            pos_ids    = torch.arange(W, device=window_emb.device)
            window_emb = window_emb + self.pos_embed(pos_ids)   # (B, W, D)
            temp_out   = self.temporal(window_emb)               # (B, W, D)

        else:  # bilstm
            temp_out, _ = self.temporal(window_emb)              # (B, W, D)

        # Mean-pool over windows → single graph embedding
        graph_emb = temp_out.mean(dim=1)                        # (B, D)
        graph_emb = self.norm(graph_emb)
        graph_emb = self.dropout(graph_emb)

        # temp_out (B, W, D): window sequence for cross-modal fusion cross-attention
        return graph_emb, gat_attn, temp_out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_window_embeddings(
        self,
        node_feats  : torch.Tensor,
        adj_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return per-window embeddings (B, W, D) before temporal pooling.

        Useful for visualising temporal dynamics of the graph embedding.
        """
        B, W, C, F = node_feats.shape
        x_bw = node_feats.view(B * W, C, F)
        a_bw = adj_matrices.view(B * W, C, C)
        emb, _ = self.gat(x_bw, a_bw)
        pool_attn = torch.softmax(self.node_pool(emb), dim=1)  # (B*W, C, 1)
        return (emb * pool_attn).sum(dim=1).view(B, W, self.embed_dim)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
