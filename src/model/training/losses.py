"""
================================================================
INS-HDGS-CMT — Multi-Task Loss
================================================================
Engagement-aware loss:

  L_total = λ1 L_focal        (binary engagement, class-imbalance safe)
           + λ2 L_contrast    (EEG ↔ ET modality alignment)
           + λ3 L_roi         (ROI spatial attention supervision)
           + λ4 L_conn        (graph sparsity regularisation)
           + λ5 L_mmd         (cross-subject domain adaptation)
           + λ6 L_rules       (neuro-symbolic diversity + sparsity)

Focal loss (Lin et al., 2017) is used instead of standard cross-
entropy to handle the inherent class imbalance in engagement data.
================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional


def compute_alpha_weights(
    labels    : "np.ndarray",
    strategy  : str  = "balanced",
    n_classes : int  = 2,
    beta      : float = None,          # for effective_num; auto if None
) -> "np.ndarray":
    """
    Compute per-class weights for the focal loss from training labels.

    Strategies
    ----------
    balanced        sklearn inverse-frequency  n_total / (n_cls * n_c)
    inv_freq        raw inverse frequency       1 / n_c  (normalised)
    effective_num   Cui et al. CVPR 2019        (1-β^n) / (1-β)
    sqrt_inv_freq   square-root inverse         1 / √n_c  (normalised)
    none            equal weights               1.0 for all classes

    Returns np.ndarray of shape (n_classes,) with mean≈1 for stable loss scale.
    """
    labels  = np.asarray(labels, dtype=int)
    counts  = np.bincount(labels, minlength=n_classes).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)      # avoid div/0
    n_total = float(len(labels))

    if strategy == "balanced":
        w = n_total / (n_classes * counts)
    elif strategy == "inv_freq":
        w = 1.0 / counts
        w = w / w.mean()                               # mean-normalise
    elif strategy == "effective_num":
        if beta is None:
            beta = (n_total - 1.0) / n_total
        eff_n = (1.0 - beta ** counts) / (1.0 - beta)
        w = 1.0 / eff_n
        w = w / w.mean()
    elif strategy == "sqrt_inv_freq":
        w = 1.0 / np.sqrt(counts)
        w = w / w.mean()
    elif strategy == "none":
        w = np.ones(n_classes, dtype=float)
    else:
        raise ValueError(f"Unknown alpha_strategy: {strategy!r}. "
                         "Choose balanced | inv_freq | effective_num | "
                         "sqrt_inv_freq | none")
    return w.astype(np.float32)


def temporal_consistency_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Penalise large prediction jumps between consecutive samples in a mini-batch.
    Acts as a smoothness regulariser; batch order is random so this is
    mild — it prevents the model from being arbitrarily inconsistent.
    """
    if logits.shape[0] < 2:
        return torch.tensor(0.0, device=logits.device)
    p1 = F.softmax(logits[:-1], dim=-1)
    p2 = F.softmax(logits[1:],  dim=-1)
    return F.mse_loss(p1, p2)


class MultiTaskLoss(nn.Module):
    """
    Wraps INS_HDGS_CMT.compute_loss() with configurable weights.

    Parameters
    ----------
    lambda_cls           : focal classification loss weight
    lambda_contrast      : contrastive alignment loss weight
    lambda_roi           : ROI supervision loss weight
    lambda_connectivity  : graph sparsity regularisation weight
    lambda_mmd           : MMD domain adaptation weight
    lambda_rules         : neuro-symbolic rule regularisation weight
    class_weights        : (N_classes,) tensor for weighted focal loss
    focal_alpha          : focal loss α (minority-class upweight)
    focal_gamma          : focal loss γ (hard-example focusing exponent)
    """

    def __init__(
        self,
        lambda_cls          : float = 1.00,
        lambda_contrast     : float = 0.30,
        lambda_roi          : float = 0.20,
        lambda_connectivity : float = 0.10,
        lambda_mmd          : float = 0.05,
        lambda_rules        : float = 0.02,
        lambda_dann         : float = 0.10,
        class_weights       : Optional[torch.Tensor] = None,
        focal_alpha         : float = 0.25,
        focal_gamma         : float = 2.00,
        mmd_mode            : str   = "marginal",
    ):
        super().__init__()
        self.lambda_cls          = lambda_cls
        self.lambda_contrast     = lambda_contrast
        self.lambda_roi          = lambda_roi
        self.lambda_connectivity = lambda_connectivity
        self.lambda_mmd          = lambda_mmd
        self.lambda_rules        = lambda_rules
        self.lambda_dann         = lambda_dann
        self.class_weights       = class_weights
        self.focal_alpha         = focal_alpha
        self.focal_gamma         = focal_gamma
        # MMD variant: "marginal" (subject-vs-rest) or "class_conditional"
        # (align HIGH-HIGH and LOW-LOW across subjects). Consumed by the
        # Trainer, which computes the actual MMD term on eeg_emb during
        # training (compute_loss's training path keeps MMD at 0).
        self.mmd_mode            = mmd_mode

    def forward(
        self,
        model         ,
        out           : Dict[str, Any],
        labels        : torch.Tensor,
        roi_vector    : torch.Tensor,
        weighted_adjs : Optional[torch.Tensor] = None,
        source_emb    : Optional[torch.Tensor] = None,
        target_emb    : Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cw  = (self.class_weights.to(labels.device)
               if self.class_weights is not None else None)
        raw = model.module if hasattr(model, "module") else model

        # Auxiliary loss warm-up ramp (set by Trainer.fit() each epoch).
        # Ramps auxiliary λ values from 0→full over the first 40% of training
        # so classification loss dominates early learning.
        ramp = getattr(self, "_aux_ramp", 1.0)

        return raw.compute_loss(
            out              = out,
            labels           = labels,
            roi_vector       = roi_vector,
            weighted_adjs    = weighted_adjs,
            source_emb       = source_emb,
            target_emb       = target_emb,
            lambda_cls       = self.lambda_cls,
            lambda_contrast  = self.lambda_contrast  * ramp,
            lambda_roi       = self.lambda_roi        * ramp,
            lambda_conn      = self.lambda_connectivity * ramp,
            lambda_mmd       = self.lambda_mmd        * ramp,
            lambda_rules     = self.lambda_rules      * ramp,
            class_weights    = cw,
            focal_alpha      = self.focal_alpha,
            focal_gamma      = self.focal_gamma,
        )
