"""
================================================================
NEUMA Phase 8C — NeuroFusion Transformer
================================================================
Cross-modal attention fusion of three parallel neural streams:
  EEG embedding    (spatial graph dynamics)
  Graph embedding  (dynamic connectivity topology)
  ET embedding     (visual attention and gaze behaviour)

Architecture
============

Stage 1 — Modality-Aware Tokenisation
  Each modality embedding (B, D) receives a learnable type
  embedding, producing three modal tokens (B, 3, D).

Stage 2 — Intra-modal Self-Attention
  TransformerEncoder over all three tokens jointly captures
  cross-modal contextual relationships.

Stage 3 — Bilinear Cross-Modal Attention
  EEG queries both Graph and ET context independently:
    EEG ← Graph:  CrossAttn(Q=EEG, K=Graph, V=Graph)
    EEG ← ET:     CrossAttn(Q=EEG, K=ET,    V=ET)
  These provide directed information flow: the EEG embedding
  is enriched by connectivity and visual attention context.

Stage 4 — Gated Residual Fusion
  g = σ(W_g [eeg ‖ cross_graph ‖ cross_et])
  fused = LayerNorm(g ⊙ W_f [eeg ‖ cross_graph ‖ cross_et])

  The gating mechanism selectively passes modality information
  based on the current input, preventing fusion collapse when
  one modality is uninformative.

Output
======
  fused         : (B, D)     fused multimodal embedding
  attn_weights  : dict       attention maps for visualisation

Explainability
==============
All attention matrices are returned and detached, enabling:
  • Cross-modal attention heatmaps (Stage 3)
  • Self-attention over modal tokens (Stage 2)
  Hooks registered by EEGGradCAM will fire on Stage 3 layers.
================================================================
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuroFusionTransformer(nn.Module):
    """
    Cross-modal fusion transformer for EEG + Graph + ET embeddings.

    Parameters
    ----------
    embed_dim    : embedding dimension for all modalities
    num_heads    : attention heads (for both self- and cross-attention)
    num_layers   : Transformer encoder depth  (self-attention stage)
    ff_dim       : feedforward hidden dimension in the Transformer
    dropout      : dropout rate
    n_modalities : number of input modalities  (default 3: EEG, Graph, ET)
    """

    def __init__(
        self,
        embed_dim   : int   = 128,
        num_heads   : int   = 4,
        num_layers  : int   = 2,
        ff_dim      : int   = 256,
        dropout     : float = 0.10,
        n_modalities: int   = 3,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.n_modalities = n_modalities

        # ── Stage 1: Learnable modality-type embeddings ──────────────────────
        self.mod_embed = nn.Embedding(n_modalities, embed_dim)
        nn.init.normal_(self.mod_embed.weight, std=0.02)

        # ── Stage 2: Self-attention over all modal tokens ────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model        = embed_dim,
            nhead          = num_heads,
            dim_feedforward= ff_dim,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,    # pre-LayerNorm for training stability
        )
        self.self_attn_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ── Stage 3: Directed cross-modal attention ──────────────────────────
        # EEG ← Graph context
        self.cross_eeg_graph = nn.MultiheadAttention(
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            dropout    = dropout,
            batch_first= True,
        )
        # EEG ← ET context
        self.cross_eeg_et = nn.MultiheadAttention(
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            dropout    = dropout,
            batch_first= True,
        )

        # ── Stage 4: Gated residual fusion ───────────────────────────────────
        gate_in = embed_dim * n_modalities   # concat of 3 modal embeddings
        self.gate = nn.Sequential(
            nn.Linear(gate_in, embed_dim, bias=True),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Linear(gate_in, embed_dim, bias=True)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.dropout     = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        eeg_emb    : torch.Tensor,                       # (B, D)
        graph_emb  : torch.Tensor,                       # (B, D)
        et_emb     : Optional[torch.Tensor],             # (B, D) or None
        graph_seq  : Optional[torch.Tensor] = None,      # (B, W, D) window sequence
        et_seq_emb : Optional[torch.Tensor] = None,      # (B, W, D) ET window sequence
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        eeg_emb    : (B, D)     EEG graph-level embedding
        graph_emb  : (B, D)     Dynamic GAT embedding
        et_emb     : (B, D)     Eye-tracking embedding  (None → zeros)
        graph_seq  : (B, W, D)  Per-window graph embeddings — makes Stage-3
                                cross-attention attend over W real tokens
                                instead of a single collapsed vector.
        et_seq_emb : (B, W, D)  Per-window ET embeddings (same purpose for ET branch)

        Returns
        -------
        fused        : (B, D)
        attn_weights : dict
            'eeg_graph'  : Stage-3 EEG←Graph attention weights
            'eeg_et'     : Stage-3 EEG←ET attention weights
        """
        B, D   = eeg_emb.shape
        device = eeg_emb.device

        # ET fallback
        if et_emb is None:
            et_emb = torch.zeros_like(eeg_emb)

        # ── Stage 1: modality-type embeddings ────────────────────────────────
        mod_ids   = torch.arange(self.n_modalities, device=device)
        mod_embs  = self.mod_embed(mod_ids)     # (3, D)

        eeg_tok   = eeg_emb   + mod_embs[0]    # (B, D)
        graph_tok = graph_emb + mod_embs[1]    # (B, D)
        et_tok    = et_emb    + mod_embs[2]    # (B, D)

        tokens = torch.stack([eeg_tok, graph_tok, et_tok], dim=1)  # (B, 3, D)

        # ── Stage 2: self-attention over modal tokens ────────────────────────
        tokens  = self.self_attn_encoder(tokens)       # (B, 3, D)
        eeg_out = tokens[:, 0:1, :]                    # (B, 1, D)

        # ── Stage 3: directed cross-modal attention ──────────────────────────
        # Use window sequences as K/V when available — EEG queries W real tokens
        # instead of a single collapsed vector (fixes the length-1 no-op bug).
        if graph_seq is not None:
            # mod_embs[1] is (D,) → reshape to (1, 1, D) for broadcast over (B, W, D)
            graph_kv = graph_seq + mod_embs[1].view(1, 1, -1)   # (B, W, D)
        else:
            graph_kv = tokens[:, 1:2, :]                         # (B, 1, D) fallback

        if et_seq_emb is not None:
            et_kv = et_seq_emb + mod_embs[2].view(1, 1, -1)     # (B, W, D)
        else:
            et_kv = tokens[:, 2:3, :]                           # (B, 1, D) fallback

        cross_g, attn_eg = self.cross_eeg_graph(
            query=eeg_out, key=graph_kv, value=graph_kv,
        )                                                      # (B, 1, D)

        cross_e, attn_ee = self.cross_eeg_et(
            query=eeg_out, key=et_kv, value=et_kv,
        )                                                      # (B, 1, D)

        cross_g = cross_g.squeeze(1)    # (B, D)
        cross_e = cross_e.squeeze(1)    # (B, D)
        eeg_t   = eeg_out.squeeze(1)    # (B, D)

        # ── Stage 4: gated residual fusion ───────────────────────────────────
        concat = torch.cat([eeg_t, cross_g, cross_e], dim=-1)  # (B, 3D)

        gate  = self.gate(concat)               # (B, D)
        fused = self.fusion_proj(concat)        # (B, D)
        fused = self.fusion_norm(fused * gate)
        fused = self.dropout(fused)

        attn_weights = {
            "eeg_graph": attn_eg.detach() if attn_eg is not None else None,
            "eeg_et"   : attn_ee.detach() if attn_ee is not None else None,
        }

        return fused, attn_weights

    # ── Ablation: EEG-only mode ───────────────────────────────────────────────

    def forward_eeg_only(self, eeg_emb: torch.Tensor) -> tuple:
        """
        Bypass cross-modal attention and use EEG embedding directly.
        Used in ablation studies when graph or ET branch is disabled.
        """
        zeros = torch.zeros_like(eeg_emb)
        return self.forward(eeg_emb, zeros, zeros)
