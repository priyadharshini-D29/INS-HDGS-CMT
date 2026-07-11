"""
================================================================
NEUMA Phase 8C — Eye-Tracking Encoders
================================================================
Two encoders for eye-tracking sequences:

ETEncoder  (original)
  BiLSTM + projection
  Input  : (B, T_et, C_et)
  Output : (B, D_embed)

ETAttentionEncoder  (Phase 8C)
  MLP feature extraction → Self-Attention → ROI attention vector
  Input  : (B, T_et, C_et)
  Outputs: (B, D_embed)  embedding
           (B, N_roi)    ROI attention vector  (for ROI modulation)

The ETAttentionEncoder is recommended when ROI attention vectors
are needed as direct outputs (e.g. for ROIGraphModulation input).
================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ETEncoder(nn.Module):
    """
    Bidirectional LSTM + linear projection for ET sequences.

    Architecture
    ------------
    BiLSTM  : C_et → 2 × hidden_dim  (forward + backward)
    Project : 2 × hidden_dim → embed_dim
    LayerNorm + Dropout
    """

    def __init__(
        self,
        input_dim   : int   = 3,
        hidden_dim  : int   = 64,
        embed_dim   : int   = 128,
        num_layers  : int   = 2,
        dropout     : float = 0.20,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        self.proj    = nn.Linear(2 * hidden_dim, embed_dim)
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T_et, C_et)

        Returns
        -------
        emb : (B, embed_dim)
        """
        # Replace any remaining NaNs (blink gaps)
        x = torch.nan_to_num(x, nan=0.0)

        _, (h, _) = self.lstm(x)            # h: (2*L, B, hidden_dim)

        # Take last-layer forward and backward hidden states
        h_fwd = h[-2]                       # (B, hidden_dim)
        h_bwd = h[-1]                       # (B, hidden_dim)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*hidden_dim)

        emb = self.act(self.proj(h_cat))
        emb = self.norm(emb)
        emb = self.dropout(emb)
        return emb


# ── ETAttentionEncoder (Phase 8C) ─────────────────────────────────────────────

class ETAttentionEncoder(nn.Module):
    """
    Eye-Tracking Attention Encoder with self-attention and ROI vector output.

    Architecture
    ------------
    1. Per-timestep MLP feature extractor
       (B, T, C_et) → (B, T, hidden_dim)
    2. Positional encoding (sinusoidal)
    3. Multi-head self-attention over the ET sequence
       captures fixation sequences, saccade patterns, pupil dynamics
    4. CLS-token aggregation  → ET embedding  (B, embed_dim)
    5. ROI attention projection → ROI attention vector  (B, n_rois)

    The ROI attention vector can be used directly as input to
    ROIGraphModulation for electrode-level graph modulation.

    Parameters
    ----------
    input_dim  : ET feature channels  (e.g. 3: gaze_x, gaze_y, pupil)
    hidden_dim : internal MLP/attention dimension
    embed_dim  : final embedding output dimension
    n_rois     : size of the ROI attention vector output
    n_heads    : self-attention heads
    n_layers   : attention layer depth
    max_len    : maximum sequence length for positional encoding
    dropout    : dropout rate
    """

    def __init__(
        self,
        input_dim  : int   = 3,
        hidden_dim : int   = 64,
        embed_dim  : int   = 128,
        n_rois     : int   = 10,
        n_heads    : int   = 4,
        n_layers   : int   = 2,
        max_len    : int   = 600,
        n_windows  : int   = 10,
        dropout    : float = 0.20,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim  = embed_dim
        self.n_windows  = n_windows

        # ── Per-timestep MLP feature extractor ───────────────────────────────
        self.feat_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # ── Learnable CLS token (prepended to sequence) ───────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Sinusoidal positional encoding ────────────────────────────────────
        self.register_buffer(
            "pos_enc",
            self._build_pos_enc(max_len + 1, hidden_dim),
        )

        # ── Multi-head self-attention encoder ─────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model        = hidden_dim,
            nhead          = n_heads,
            dim_feedforward= hidden_dim * 4,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # ── CLS → ET embedding ────────────────────────────────────────────────
        self.emb_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # ── ET embedding → ROI attention vector ───────────────────────────────
        self.roi_proj = nn.Sequential(
            nn.Linear(embed_dim, n_rois),
            nn.Softmax(dim=-1),   # normalised attention distribution over ROIs
        )

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_pos_enc(max_len: int, d: int) -> torch.Tensor:
        """Build sinusoidal positional encoding table (max_len, d)."""
        pos = torch.arange(max_len).unsqueeze(1).float()
        dim = torch.arange(0, d, 2).float()
        div = torch.exp(dim * -(torch.log(torch.tensor(10000.0)) / d))
        enc = torch.zeros(max_len, d)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        return enc.unsqueeze(0)   # (1, max_len, d)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,   # (B, T_et, C_et)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T_et, C_et)   raw ET sequence

        Returns
        -------
        emb          : (B, embed_dim)      ET embedding (from CLS token)
        roi_attn     : (B, n_rois)         ROI attention vector
        window_seq   : (B, n_windows, D)   per-window ET embeddings for fusion
        """
        B, T, _ = x.shape

        # Replace NaN (blink gaps)
        x = torch.nan_to_num(x, nan=0.0)

        # Per-timestep MLP
        h = self.feat_mlp(x)            # (B, T, hidden_dim)
        h = self.dropout(h)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, hidden_dim)
        h   = torch.cat([cls, h], dim=1)          # (B, T+1, hidden_dim)

        # Add positional encoding (broadcast over batch)
        T1  = h.size(1)
        h   = h + self.pos_enc[:, :T1, :]         # (B, T+1, hidden_dim)

        # Self-attention
        h   = self.transformer(h)                  # (B, T+1, hidden_dim)

        # CLS token → global ET embedding
        cls_out  = h[:, 0, :]                      # (B, hidden_dim)
        emb      = self.emb_proj(cls_out)          # (B, embed_dim)
        roi_attn = self.roi_proj(emb)              # (B, n_rois)

        # Per-window ET sequence — chunk the T per-timestep tokens into n_windows
        # Each window pools wlen consecutive ET timesteps → (B, n_windows, embed_dim)
        h_seq  = h[:, 1:, :]                       # (B, T, hidden_dim) — drop CLS
        W      = self.n_windows
        wlen   = T // W
        h_trim = h_seq[:, : W * wlen, :]           # (B, W*wlen, hidden_dim)
        h_wins = h_trim.view(B, W, wlen, self.hidden_dim).mean(dim=2)  # (B, W, hidden_dim)
        window_seq = self.emb_proj(h_wins)         # (B, W, embed_dim)

        return emb, roi_attn, window_seq
