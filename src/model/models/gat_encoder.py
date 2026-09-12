"""
================================================================
NEUMA Phase 8 — Graph Attention Network Encoder
================================================================
Pure-PyTorch implementation of a two-layer GAT that processes
batched dynamic electrode graphs.

Attention equation (Veličković et al., 2018):

  e_ij = LeakyReLU( a^T [W h_i ‖ W h_j] )
  α_ij = softmax_j( e_ij )
  h_i' = Σ_j α_ij · W h_j

Layer 1: 4-head with concat  →  4 × 32 = 128 dims
Layer 2: 1-head              →  128 dims  (+ residual)

Outputs learned electrode embeddings and per-head attention
weight matrices for downstream explainability.
================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Single-Head GAT Layer ────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    One head of a Graph Attention Layer.

    Operates on batched graphs: (B, N, F_in) with adjacency (B, N, N).
    Also accepts unbatched (N, F_in) with (N, N) via auto-squeeze logic.
    """

    def __init__(
        self,
        in_dim          : int,
        out_dim         : int,
        dropout         : float = 0.20,
        negative_slope  : float = 0.20,
    ):
        super().__init__()
        self.out_dim = out_dim

        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # Attention parameter: a ∈ R^{2 * out_dim}
        self.a = nn.Parameter(torch.empty(1, 1, 2 * out_dim))
        nn.init.xavier_normal_(self.a)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Parameters
        ----------
        x   : (B, N, F_in) or (N, F_in)
        adj : (B, N, N)    or (N, N)   — binary/weighted edge mask

        Returns
        -------
        out   : (B, N, out_dim)
        alpha : (B, N, N)   — attention coefficients
        """
        squeeze = x.dim() == 2
        if squeeze:
            x   = x.unsqueeze(0)
            adj = adj.unsqueeze(0)

        B, N, _ = x.shape
        Wh = self.W(x)                                           # (B, N, out_dim)

        # Build all-pairs concatenations [W*h_i ‖ W*h_j]
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)           # (B, N, N, D)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)           # (B, N, N, D)
        cat  = torch.cat([Wh_i, Wh_j], dim=-1)                 # (B, N, N, 2D)

        # Scalar attention logit: e_ij = LeakyReLU(a^T cat)
        e = (cat * self.a).sum(-1)                              # (B, N, N)
        e = self.leaky_relu(e)

        # Mask non-edges (set to -∞ before softmax)
        e = e.masked_fill(adj == 0, float("-inf"))

        alpha = torch.softmax(e, dim=-1)                        # (B, N, N)
        alpha = torch.nan_to_num(alpha, nan=0.0)                # isolated nodes → 0
        alpha = self.dropout(alpha)

        out = torch.bmm(alpha, Wh)                              # (B, N, out_dim)

        if squeeze:
            out   = out.squeeze(0)
            alpha = alpha.squeeze(0)

        return out, alpha


# ── Multi-Head + Two-Layer GATEncoder ────────────────────────────────────────

class GATEncoder(nn.Module):
    """
    Two-layer Graph Attention Encoder with:
    • Layer 1: H-head attention, concat  →  H × head_dim
    • Layer 2: 1-head attention          →  out_dim
    • Residual connection on Layer 2
    • Layer Normalisation after each layer
    • ELU activation

    Returns
    -------
    node_emb : (B, N, out_dim)
    attn     : dict with 'l1_attn' (B, N, N) and 'l2_attn' (B, N, N)
    """

    def __init__(
        self,
        in_dim   : int   = 5,
        head_dim : int   = 32,
        heads    : int   = 4,
        out_dim  : int   = 128,
        dropout  : float = 0.20,
    ):
        super().__init__()

        l1_out = heads * head_dim   # 4 × 32 = 128

        # Layer 1: multi-head, concatenated
        self.l1_heads = nn.ModuleList(
            [GATLayer(in_dim, head_dim, dropout) for _ in range(heads)]
        )
        self.norm1  = nn.LayerNorm(l1_out)

        # Layer 2: single head, residual
        self.l2     = GATLayer(l1_out, out_dim, dropout)
        self.norm2  = nn.LayerNorm(out_dim)

        # Residual projection (l1_out → out_dim)
        self.res_proj = nn.Linear(l1_out, out_dim, bias=False)

        self.act     = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        x   : (B, N, in_dim)  or (N, in_dim)
        adj : (B, N, N)        or (N, N)
        """
        squeeze = x.dim() == 2
        if squeeze:
            x   = x.unsqueeze(0)
            adj = adj.unsqueeze(0)

        # ── Layer 1 ──────────────────────────────────────────────
        head_outs, l1_alphas = [], []
        for head in self.l1_heads:
            h, a = head(x, adj)
            head_outs.append(h)
            l1_alphas.append(a)

        x1 = torch.cat(head_outs, dim=-1)                      # (B, N, l1_out)
        x1 = self.act(self.norm1(x1))
        x1 = self.dropout(x1)

        # ── Layer 2 with residual ─────────────────────────────────
        x2, l2_alpha = self.l2(x1, adj)
        x2 = x2 + self.res_proj(x1)
        x2 = self.norm2(x2)                                    # (B, N, out_dim)

        avg_l1_alpha = torch.stack(l1_alphas, 0).mean(0)

        if squeeze:
            x2          = x2.squeeze(0)
            avg_l1_alpha= avg_l1_alpha.squeeze(0)
            l2_alpha    = l2_alpha.squeeze(0)

        attn = {"l1_attn": avg_l1_alpha, "l2_attn": l2_alpha}
        return x2, attn
