"""
================================================================
NEUMA Phase 8C — Multi-Scale EEG Temporal Encoder
================================================================
Processes raw EEG time series with a multi-scale temporal CNN
architecture incorporating:

  • Multi-scale dilated convolutions  (d = 1, 2, 4, 8)
  • Residual connections              (He et al., 2016)
  • Squeeze-and-Excitation attention  (Hu et al., 2018)
  • Depthwise-separable convolutions  (MobileNet-style)
  • Batch normalisation + GELU

Input:  (B, C, S)   C EEG channels, S time samples
Output: (B, D)      fixed-dimension temporal embedding

Design rationale
----------------
EEG signals contain rhythmic activity across multiple timescales:
  δ (1–4 Hz):   d=8 captures slow-wave dynamics
  θ (4–8 Hz):   d=4 captures hippocampal theta
  α (8–13 Hz):  d=2 captures idle/visual alpha
  β/γ:          d=1 captures fast cortical dynamics

Each dilation scale is processed in parallel by a separate
residual branch, then channel-concatenated and projected.
The SE block learns which temporal scales are most informative
for the current stimulus/cognitive state.

This encoder is complementary to the GAT branch in RD-GANet:
the GAT branch captures inter-electrode spatial topology while
this encoder captures within-electrode temporal dynamics.
================================================================
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Squeeze-and-Excitation Channel Attention ─────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for 1-D temporal feature maps.

    Learns per-channel importance weights by:
      1. Squeeze: global average pooling  (B, C, T) → (B, C)
      2. Excitation: two-layer MLP with sigmoid gate
      3. Scale: element-wise multiply back to (B, C, T)

    Parameters
    ----------
    channels  : number of feature channels
    reduction : bottleneck reduction ratio (default 4×)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, T) → (B, C, T)"""
        s = self.pool(x).squeeze(-1)        # (B, C)
        s = self.excite(s).unsqueeze(-1)    # (B, C, 1)
        return x * s


# ── Residual Dilated Temporal Block ─────────────────────────────────────────

class ResidualDilatedBlock(nn.Module):
    """
    Residual temporal block with dilated convolution + SE attention.

    Architecture (pre-activation residual):
        x → BN → GELU → DepthwiseConv1d(dilation=d) → BN → GELU → Conv1d(1) → SE → + x

    Using depthwise-separable convolutions reduces parameter count
    while maintaining receptive field coverage.

    Parameters
    ----------
    channels  : number of feature channels
    dilation  : dilation factor for the first convolution
    kernel    : temporal kernel size (default 5 for ~3 electrode distances)
    reduction : SE reduction ratio
    """

    def __init__(
        self,
        channels : int,
        dilation : int = 1,
        kernel   : int = 5,
        reduction: int = 4,
    ):
        super().__init__()
        pad = dilation * (kernel - 1) // 2

        # Depthwise dilated conv: one filter per channel
        self.dw_conv = nn.Conv1d(
            channels, channels, kernel,
            dilation=dilation, padding=pad,
            groups=channels, bias=False,
        )
        # Pointwise (1×1): mix channels
        self.pw_conv = nn.Conv1d(channels, channels, 1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.bn3 = nn.BatchNorm1d(channels)
        self.se  = SEBlock(channels, reduction)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, T) → (B, C, T)  (same shape residual)"""
        residual = x

        # Pre-activation residual path
        h = self.act(self.bn1(x))
        h = self.dw_conv(h)
        h = self.act(self.bn2(h))
        h = self.pw_conv(h)
        h = self.bn3(h)
        h = self.se(h)

        return h + residual


# ── Multi-Scale EEG Temporal Encoder ────────────────────────────────────────

class EEGTemporalEncoder(nn.Module):
    """
    Multi-scale temporal CNN encoder for raw EEG epochs.

    Processes (B, C, S) → (B, embed_dim) by:
      1. Stem conv: channel projection + initial temporal smoothing
      2. Parallel multi-scale dilated residual branches
      3. Channel concatenation + 1×1 projection
      4. Global average pooling (temporal → scalar)
      5. LayerNorm + Dropout

    Parameters
    ----------
    n_channels  : number of EEG electrodes  (C)
    embed_dim   : output embedding dimension  (D)
    n_samples   : expected time samples  (S)  — only for documentation
    dilations   : tuple of dilation values for multi-scale branches
    stem_dim    : channels after stem projection  (embed_dim // 2 by default)
    kernel      : temporal kernel size for dilated blocks
    se_reduction: SE block channel reduction ratio
    dropout     : dropout rate on final embedding
    """

    def __init__(
        self,
        n_channels  : int               = 24,
        embed_dim   : int               = 128,
        n_samples   : int               = 1500,
        dilations   : Sequence[int]     = (1, 2, 4, 8),
        stem_dim    : int | None        = None,
        kernel      : int               = 5,
        se_reduction: int               = 4,
        dropout     : float             = 0.20,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        mid_dim          = stem_dim or (embed_dim // 2)
        n_scales         = len(dilations)

        # ── Stem: project EEG channels → mid_dim feature maps ───────────────
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, mid_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Conv1d(mid_dim, mid_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
        )

        # ── Parallel multi-scale dilated residual branches ──────────────────
        # Each branch: one or two residual blocks at a specific dilation
        self.scale_branches = nn.ModuleList([
            nn.Sequential(
                ResidualDilatedBlock(mid_dim, dilation=d, kernel=kernel, reduction=se_reduction),
                ResidualDilatedBlock(mid_dim, dilation=d, kernel=kernel, reduction=se_reduction),
            )
            for d in dilations
        ])

        # Global SE block on the concatenated multi-scale features
        concat_dim = mid_dim * n_scales
        self.global_se = SEBlock(concat_dim, reduction=se_reduction)

        # ── 1×1 projection: concat_dim → embed_dim ──────────────────────────
        self.project = nn.Sequential(
            nn.Conv1d(concat_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # ── Global temporal pooling ──────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ── Output head ──────────────────────────────────────────────────────
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, S)   raw EEG epoch  (C channels, S samples)

        Returns
        -------
        emb : (B, embed_dim)
        """
        # Stem: shared feature extraction
        h = self.stem(x)                               # (B, mid_dim, S)

        # Parallel multi-scale branches
        scale_outs = [branch(h) for branch in self.scale_branches]
        h = torch.cat(scale_outs, dim=1)               # (B, mid_dim*n_scales, S)

        # Global SE on concatenated multi-scale features
        h = self.global_se(h)

        # Project to embed_dim
        h = self.project(h)                            # (B, embed_dim, S)

        # Global average pooling over time
        h = self.pool(h).squeeze(-1)                   # (B, embed_dim)

        h = self.norm(h)
        h = self.dropout(h)
        return h

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return temporal feature maps (B, embed_dim, S) before pooling.

        Useful for temporal saliency analysis: which time points were
        most informative for the final embedding.
        """
        h = self.stem(x)
        scale_outs = [branch(h) for branch in self.scale_branches]
        h = torch.cat(scale_outs, dim=1)
        h = self.global_se(h)
        return self.project(h)     # (B, embed_dim, S)

    def get_scale_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Return per-scale feature maps before concatenation.

        Returns list of (B, mid_dim, S) tensors — one per dilation scale.
        """
        h = self.stem(x)
        return [branch(h) for branch in self.scale_branches]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
