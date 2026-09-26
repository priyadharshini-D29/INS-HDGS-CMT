"""
================================================================
NEUMA Phase 8 — Maximum Mean Discrepancy (MMD) Domain Adaptation
================================================================
Implements subject-invariant representation learning via MMD:

  L_MMD = ‖ 1/n_s Σ_i φ(x_i^s) − 1/n_t Σ_j φ(x_j^t) ‖²

where φ is a feature map induced by the RBF kernel:

  k(x, y) = exp(−γ ‖x − y‖²)

The loss encourages the model to produce subject-agnostic
embeddings, improving Leave-One-Subject-Out performance.
================================================================
"""

import torch
import torch.nn as nn


# ── Kernel Functions ─────────────────────────────────────────────────────────

def _rbf_kernel(
    x     : torch.Tensor,
    y     : torch.Tensor,
    gamma : float = 1.0,
) -> torch.Tensor:
    """
    Compute Gaussian RBF kernel matrix.

    k(x_i, y_j) = exp(−γ ‖x_i − y_j‖²)

    Parameters
    ----------
    x : (N, D)
    y : (M, D)

    Returns
    -------
    K : (N, M)
    """
    dist_sq = torch.cdist(x, y, p=2) ** 2          # (N, M)
    return torch.exp(-gamma * dist_sq)


def _multi_rbf_kernel(
    x       : torch.Tensor,
    y       : torch.Tensor,
    gammas  = (0.1, 0.5, 1.0, 2.0, 5.0),
) -> torch.Tensor:
    """Multi-scale RBF kernel (sum over bandwidths)."""
    K = sum(_rbf_kernel(x, y, g) for g in gammas)
    return K / len(gammas)


# ── MMD Loss ─────────────────────────────────────────────────────────────────

def mmd_loss(
    source  : torch.Tensor,
    target  : torch.Tensor,
    gammas  = (0.1, 0.5, 1.0, 2.0, 5.0),
) -> torch.Tensor:
    """
    Compute unbiased MMD² between source and target distributions.

    MMD² = E[k(xs,xs)] + E[k(xt,xt)] − 2 E[k(xs,xt)]

    Parameters
    ----------
    source : (N_s, D)
    target : (N_t, D)

    Returns
    -------
    mmd² : scalar (non-negative)
    """
    if source.size(0) < 2 or target.size(0) < 2:
        return torch.tensor(0.0, device=source.device, requires_grad=True)

    K_ss = _multi_rbf_kernel(source, source, gammas)
    K_tt = _multi_rbf_kernel(target, target, gammas)
    K_st = _multi_rbf_kernel(source, target, gammas)

    # Unbiased estimator: exclude diagonal of K_ss and K_tt
    n_s, n_t = source.size(0), target.size(0)
    mask_ss  = ~torch.eye(n_s, dtype=torch.bool, device=source.device)
    mask_tt  = ~torch.eye(n_t, dtype=torch.bool, device=target.device)

    mmd = (
        K_ss[mask_ss].mean()
        + K_tt[mask_tt].mean()
        - 2.0 * K_st.mean()
    )
    return torch.clamp(mmd, min=0.0)


class MMDLoss(nn.Module):
    """Wrapper for MMD loss usable in multi-task loss composition."""

    def __init__(self, gammas=(0.1, 0.5, 1.0, 2.0, 5.0)):
        super().__init__()
        self.gammas = gammas

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mmd_loss(source, target, self.gammas)
