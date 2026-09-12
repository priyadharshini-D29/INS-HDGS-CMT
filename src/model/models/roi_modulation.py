"""
================================================================
NEUMA Phase 8C — ROI-Guided Graph Modulation
================================================================
Implements dynamic, learnable modulation of EEG adjacency
matrices guided by eye-tracking ROI dwell-time vectors:

    A_roi = A ⊙ (1 + α · outerproduct(roi_elec, roi_elec))

Where:
    roi_elec = σ(W_proj · roi_vec)   ∈ R^C  (projected to electrode space)
    α        = learnable scalar        ∈ R
    A        = adjacency matrix        ∈ R^{C×C}

The outer product of electrode-level ROI attentions forms a
C×C modulation mask that selectively amplifies edges between
electrodes that both attend to the same ROI regions.

The modulated adjacency is then symmetrically normalized:
    Â = D^{-1/2} A_roi D^{-1/2}

so it can be directly used in the GATEncoder without gradient
explosion from large adjacency weights.

This layer supports:
    • Batched inputs:  (B, W, C, C) windowed adjacency
    • Single inputs:   (B, C, C)    static adjacency
    • DataParallel:    no shared state beyond learnable params
================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIGraphModulation(nn.Module):
    """
    ROI-Guided Graph Modulation layer.

    Modulates EEG connectivity graphs based on visual attention
    captured by eye-tracking ROI dwell-time vectors.

    The modulation is inspired by the biological coupling between
    visual attention (V4/IT cortex) and EEG oscillatory dynamics:
    when gaze dwells on a semantic ROI, electrodes projecting to
    those visual areas should have amplified mutual connectivity.

    Parameters
    ----------
    n_rois     : ROI vector dimension  (number of screen ROI regions)
    n_channels : EEG electrode count   (C)
    dropout    : dropout on the ROI projection
    eps        : numerical stability for degree normalisation

    Attributes
    ----------
    alpha     : learnable scalar controlling modulation strength
    roi_proj  : linear projection  R^{N_roi} → R^C
    """

    def __init__(
        self,
        n_rois    : int   = 10,
        n_channels: int   = 24,
        dropout   : float = 0.10,
        eps       : float = 1e-8,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.eps        = eps

        # Learnable modulation strength  (initialised to 0.5 → moderate effect)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Project ROI dwell-time vector to electrode-level attention weights
        self.roi_proj = nn.Sequential(

            nn.Linear(n_rois, n_channels),

            nn.Sigmoid(),
        )   

        self.dropout = nn.Dropout(dropout)

    # ── Core modulation ──────────────────────────────────────────────────────

    def forward(
        self,
        adj     : torch.Tensor,
        roi_vec,
    ) -> torch.Tensor:

        import ast
        import numpy as np

        windowed = (adj.dim() == 4)

        # -----------------------------------------------------
        # SAFETY TYPE COERCION
        # -----------------------------------------------------

        # Case 1: string input
        if isinstance(roi_vec, str):
            try:
                roi_vec = ast.literal_eval(roi_vec)
            except Exception:
                roi_vec = [
                    float(x.strip())
                    for x in roi_vec.split(",")
                ]

        # Case 2: list / tuple
        if isinstance(roi_vec, (list, tuple)):
            roi_vec = torch.tensor(
                roi_vec,
                dtype=torch.float32,
                device=adj.device,
            )

        # Case 3: numpy array
        if isinstance(roi_vec, np.ndarray):
            roi_vec = torch.tensor(
                roi_vec,
                dtype=torch.float32,
                device=adj.device,
            )

        # Final validation
        if not torch.is_tensor(roi_vec):
            raise TypeError(
                f"roi_vec must be tensor/list/array, got {type(roi_vec)}"
            )

        # -----------------------------------------------------
        # Ensure correct dtype/device
        # -----------------------------------------------------

        roi_vec = roi_vec.to(
            device=adj.device,
            dtype=torch.float32,
        )

        # -----------------------------------------------------
        # Ensure batch dimension
        # -----------------------------------------------------

        if roi_vec.dim() == 1:
            roi_vec = roi_vec.unsqueeze(0)

        # -----------------------------------------------------
        # Validate ROI dimension
        # -----------------------------------------------------

        expected_rois = self.roi_proj[0].in_features

        if roi_vec.shape[-1] != expected_rois:
            raise ValueError(
                f"Expected ROI dim {expected_rois}, "
                f"got {roi_vec.shape[-1]}"
            )

        # -----------------------------------------------------
        # ROI Projection
        # -----------------------------------------------------

        roi_elec = self.dropout(
            self.roi_proj(roi_vec)
        )  # (B, C)

        # -----------------------------------------------------
        # Electrode-pair modulation mask
        # -----------------------------------------------------

        roi_mask = torch.bmm(
            roi_elec.unsqueeze(2),
            roi_elec.unsqueeze(1),
        )  # (B, C, C)

        # Broadcast for windowed adjacency
        if windowed:
            roi_mask = roi_mask.unsqueeze(1)

        # -----------------------------------------------------
        # Graph modulation
        # -----------------------------------------------------

        A_roi = adj * (1.0 + self.alpha * roi_mask)

        # -----------------------------------------------------
        # Symmetric normalization
        # -----------------------------------------------------

        A_roi = self._sym_normalize(A_roi)

        return A_roi

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _sym_normalize(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Symmetric normalisation:  Â = D^{-1/2} A D^{-1/2}

        Preserves graph structure while bounding spectral radius.
        Applied along the last two dimensions so it works for both
        (B, C, C) and (B, W, C, C) shapes.
        """
        # Degree vector:  d_i = Σ_j A_ij
        deg = adj.sum(dim=-1, keepdim=True)             # (..., C, 1)
        deg_inv_sqrt = (deg + self.eps).pow(-0.5)       # (..., C, 1)

        # Â = D^{-1/2} A D^{-1/2}
        adj_norm = deg_inv_sqrt * adj * deg_inv_sqrt.transpose(-1, -2)
        return adj_norm

    # ── Interpretability helpers ──────────────────────────────────────────────

    @torch.no_grad()
    def roi_importance_map(
        self,
        roi_vec : torch.Tensor,   # (B, N_rois) or (N_rois,)
    ) -> torch.Tensor:
        """
        Return the (B, C, C) modulation mask without applying it to an adjacency.

        Useful for visualising which electrode pairs are amplified by
        the current ROI attention pattern.
        """
        if roi_vec.dim() == 1:
            roi_vec = roi_vec.unsqueeze(0)
        roi_elec = self.roi_proj(roi_vec)       # (B, C)
        roi_mask = torch.bmm(
            roi_elec.unsqueeze(2),
            roi_elec.unsqueeze(1),
        )                                       # (B, C, C)
        return roi_mask.squeeze(0) if roi_mask.size(0) == 1 else roi_mask

    @torch.no_grad()
    def electrode_roi_weights(
        self,
        roi_vec : torch.Tensor,   # (N_rois,) or (B, N_rois)
    ) -> torch.Tensor:
        """
        Return per-electrode ROI attention weights  (C,) or (B, C).

        Useful for electrode-level saliency maps showing which electrodes
        are most modulated by the current visual attention pattern.
        """
        single = roi_vec.dim() == 1
        if single:
            roi_vec = roi_vec.unsqueeze(0)
        w = self.roi_proj(roi_vec)              # (B, C)
        return w.squeeze(0) if single else w
