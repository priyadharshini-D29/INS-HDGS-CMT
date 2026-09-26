"""
================================================================
NEUMA Phase 8 — Bidirectional Cross-Modal NeuroFusion Attention
================================================================
Implements bidirectional cross-attention between EEG and ET
embeddings using the scaled dot-product attention formula:

  Attention(Q, K, V) = softmax(Q K^T / √d_k) V

  EEG → ET : Q = EEG,  K = V = ET   →  EEG attends to ET
  ET  → EEG: Q = ET,   K = V = EEG  →  ET  attends to EEG

The two outputs are summed + normalised to produce the final
fused representation used for classification.
================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """
    Bidirectional cross-modal attention fusion.

    Parameters
    ----------
    embed_dim  : embedding dimension (D) — must be same for EEG and ET
    num_heads  : number of attention heads
    dropout    : attention dropout
    """

    def __init__(
        self,
        embed_dim  : int   = 128,
        num_heads  : int   = 4,
        dropout    : float = 0.10,
    ):
        super().__init__()

        # EEG queries ET
        self.eeg_to_et = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )

        # ET queries EEG
        self.et_to_eeg = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )

        self.norm_eeg = nn.LayerNorm(embed_dim)
        self.norm_et  = nn.LayerNorm(embed_dim)

        # Gated fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.out_proj    = nn.Linear(embed_dim, embed_dim)
        self.out_norm    = nn.LayerNorm(embed_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(
        self,
        eeg : torch.Tensor,
        et  : torch.Tensor,
    ):
        """
        Parameters
        ----------
        eeg : (B, 1, D)  EEG embedding (sequence length 1 for cross-attn)
        et  : (B, 1, D)  ET  embedding

        Returns
        -------
        fused       : (B, D)  fused representation
        attn_weights: dict with 'eeg_to_et' and 'et_to_eeg' weights (B, 1, 1)
        """
        # EEG queries ET (EEG attends where ET provides information)
        eeg_fused, w_e2t = self.eeg_to_et(query=eeg, key=et, value=et)
        eeg_fused = self.norm_eeg(eeg + eeg_fused)             # residual

        # ET queries EEG (ET attends where EEG provides information)
        et_fused, w_t2e  = self.et_to_eeg(query=et, key=eeg, value=eeg)
        et_fused  = self.norm_et(et + et_fused)                # residual

        # Gated combination
        eeg_vec = eeg_fused.squeeze(1)                         # (B, D)
        et_vec  = et_fused.squeeze(1)                          # (B, D)

        gate   = self.fusion_gate(torch.cat([eeg_vec, et_vec], dim=-1))  # (B, D)
        fused  = gate * eeg_vec + (1 - gate) * et_vec          # (B, D)
        fused  = self.dropout(self.out_proj(fused))
        fused  = self.out_norm(fused)

        attn_weights = {
            "eeg_to_et": w_e2t,   # (B, 1, 1) for single-step input
            "et_to_eeg": w_t2e,
        }
        return fused, attn_weights
