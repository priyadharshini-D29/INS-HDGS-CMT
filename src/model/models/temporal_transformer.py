"""
================================================================
NEUMA Phase 8 — Temporal Transformer Encoder
================================================================
Learns temporal EEG dynamics over the sequence of per-window
graph embeddings using sinusoidal positional encoding and
multi-head self-attention blocks.

Input  : (B, W, D) — batch × n_windows × embed_dim
Output : (B, W, D) — attended temporal representations
================================================================
"""

import math
import torch
import torch.nn as nn


# ── Sinusoidal Positional Encoding ───────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    """

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.10):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position  = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term  = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                    # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Temporal Transformer ─────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Stack of Transformer encoder layers operating on the window
    sequence produced by the GAT encoder.

    The output is mean-pooled over the temporal axis to obtain a
    fixed-size EEG representation per epoch.
    """

    def __init__(
        self,
        d_model    : int   = 128,
        nhead      : int   = 4,
        num_layers : int   = 2,
        dim_ff     : int   = 256,
        dropout    : float = 0.10,
        max_len    : int   = 200,
    ):
        super().__init__()

        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_ff,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,          # pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, W, D)   graph-embedding sequence over W windows

        Returns
        -------
        out : (B, W, D)  attended representations (caller pools)
        """
        x   = self.pos_enc(x)
        out = self.encoder(x)
        out = self.norm(out)
        return out
