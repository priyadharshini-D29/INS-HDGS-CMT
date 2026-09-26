"""
================================================================
NEUMA Phase 8 — ROI-Guided EEG Attention Modulation
================================================================
Uses the ROI dwell-time vector r_t (computed from ET gaze) to
modulate EEG graph embeddings via element-wise gating:

  h̃_t = h_t ⊙ σ(W_r r_t + b_r)

This creates ROI-conditioned neural attention, where regions
of high visual interest selectively amplify EEG features.

Additionally predicts a soft ROI distribution for auxiliary
multi-task supervision (L_roi).
================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIAttention(nn.Module):
    """
    ROI-guided gating module.

    Parameters
    ----------
    eeg_dim   : EEG embedding dimension (D)
    roi_dim   : ROI vector dimension (N_rois)
    hidden_dim: intermediate projection size
    """

    def __init__(
        self,
        eeg_dim    : int   = 128,
        roi_dim    : int   = 10,
        hidden_dim : int   = 64,
        dropout    : float = 0.10,
    ):
        super().__init__()

        # Gate: W_r r_t + b_r  →  σ(·)  →  (D,)
        self.gate = nn.Sequential(
            nn.Linear(roi_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, eeg_dim),
        )

        # Auxiliary ROI predictor (for L_roi)
        self.roi_predictor = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, roi_dim),
        )

        self.norm    = nn.LayerNorm(eeg_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h         : torch.Tensor,
        roi_vector: torch.Tensor,
    ):
        """
        Parameters
        ----------
        h          : (B, D)      EEG embedding
        roi_vector : (B, N_rois) normalised dwell-time vector

        Returns
        -------
        h_tilde    : (B, D)    ROI-modulated EEG embedding
        roi_logits : (B, N_rois)  soft ROI distribution for L_roi
        gate_weights:(B, D)   sigmoid gate (for explainability)
        """
        gate = torch.sigmoid(self.gate(roi_vector))             # (B, D)
        h_tilde = self.norm(h * gate)                          # (B, D)
        h_tilde = self.dropout(h_tilde)

        roi_logits = self.roi_predictor(h_tilde)               # (B, N_rois)

        return h_tilde, roi_logits, gate
