"""
================================================================
NEUMA Phase 8C — Multimodal Contrastive Alignment
================================================================
Implements contrastive losses for EEG ↔ ET embedding alignment:

NT-Xent (SimCLR / InfoNCE):
    L = -log [
        exp(sim(z_i, z_j) / τ)
        / Σ_k exp(sim(z_i, z_k) / τ)
    ]

InfoNCE with hard-negative mining:
    Weights negatives by their cosine similarity to the anchor
    so that hard negatives (close in embedding space) are
    emphasised more than easy ones.

Positive pairs : (z_eeg_i, z_et_i)  — same stimulus, same subject
Negative pairs : all other in-batch pairs

Both classes are exported. InfoNCELoss is the recommended default
for publication experiments as it better exploits hard negatives.
================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Symmetrised NT-Xent contrastive loss for two views z1 (EEG) and z2 (ET).

    Parameters
    ----------
    temperature : τ  (default 0.07 following SimCLR)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z1 : (B, D)   EEG embeddings (need not be pre-normalised)
        z2 : (B, D)   ET  embeddings

        Returns
        -------
        loss : scalar NT-Xent loss
        """
        B = z1.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        # L2-normalise
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate all embeddings: [z1_0..z1_{B-1}, z2_0..z2_{B-1}]
        z   = torch.cat([z1, z2], dim=0)                       # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature              # (2B, 2B)

        # Mask self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim  = sim.masked_fill(mask, float("-inf"))

        # Positive pair indices:
        #   z1[i] → positive is z2[i] = index B+i
        #   z2[i] → positive is z1[i] = index i
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B,     device=z.device),
        ])                                                      # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss


# ── InfoNCE with Hard-Negative Mining ────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss (Oord et al., 2018) with optional hard-negative weighting.

    Extends NT-Xent by up-weighting negatives that are close in the
    embedding space (hard negatives).  The weighting is controlled by
    *hard_neg_weight*:
      • 0.0 → standard NT-Xent (uniform negatives)
      • 1.0 → fully hard-negative weighted

    Formulation
    -----------
    For each positive pair (z_eeg_i, z_et_i):

        w_k = exp(sim(z_eeg_i, z_et_k) / τ)       [negative weights]
        w_k normalised so Σ w_k = B-1

        L_i = -sim(z_eeg_i, z_et_i) / τ
              + log Σ_k w_k * exp(sim(z_eeg_i, z_et_k) / τ)

    Parameters
    ----------
    temperature     : τ in the InfoNCE formulation
    hard_neg_weight : λ ∈ [0, 1] for hard-negative emphasis
    """

    def __init__(
        self,
        temperature     : float = 0.07,
        hard_neg_weight : float = 0.5,
    ):
        super().__init__()
        self.temperature      = temperature
        self.hard_neg_weight  = hard_neg_weight

    def forward(self, z_eeg: torch.Tensor, z_et: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_eeg : (B, D)   EEG embeddings
        z_et  : (B, D)   ET  embeddings

        Returns
        -------
        loss : scalar InfoNCE loss
        """
        B = z_eeg.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z_eeg.device, requires_grad=True)

        # L2-normalise both views
        z1 = F.normalize(z_eeg, dim=1)   # (B, D)
        z2 = F.normalize(z_et,  dim=1)   # (B, D)

        # Pairwise cosine similarities: (B, B)
        sim_12 = torch.mm(z1, z2.T) / self.temperature   # z_eeg_i · z_et_j
        sim_21 = sim_12.T                                  # z_et_i · z_eeg_j

        # --- Hard-negative weighting ---
        if self.hard_neg_weight > 0.0:
            # Create negative mask: off-diagonal
            neg_mask = (~torch.eye(B, dtype=torch.bool, device=z1.device)).float()

            # Similarity-based weights for negatives (higher sim → harder)
            neg_weights = torch.exp(sim_12.detach() * self.hard_neg_weight)
            neg_weights = neg_weights * neg_mask             # zero out diagonal
            # Normalise so weights sum to (B-1) per row
            row_sum = neg_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            neg_weights = neg_weights * (B - 1) / row_sum

            # Mix hard-negative weights into the similarity matrix
            eye = torch.eye(B, device=z1.device)
            sim_12_w = sim_12 * (eye + neg_weights * (1 - eye))
            sim_21_w = sim_21 * (eye + neg_weights.T * (1 - eye))
        else:
            sim_12_w = sim_12
            sim_21_w = sim_21

        # Standard cross-entropy on positive pair targets
        labels = torch.arange(B, device=z1.device)
        loss = (
            F.cross_entropy(sim_12_w, labels) +
            F.cross_entropy(sim_21_w, labels)
        ) / 2.0

        return loss
