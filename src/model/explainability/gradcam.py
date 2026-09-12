"""
================================================================
NEUMA Phase 8 — EEG Grad-CAM Explainability  (multi-GPU safe)
================================================================
Computes electrode-level saliency via Gradient-weighted Class
Activation Mapping (Grad-CAM):

  w_c = (1/N) Σ_i (∂y_c / ∂A_i)
  CAM_c = ReLU( Σ_i w_c · A_i )

where A_i are node activations from the final GAT layer.

DataParallel compatibility
--------------------------
When the model is wrapped in nn.DataParallel, hook registration and
forward/backward passes must use the *raw* (unwrapped) model so that:
  • Hooks fire on a single replica, not once per GPU replica.
  • model.module.gat.l2 is the actual layer (requirement 8).

The generate() method always runs on a *single sample* (no batch
scatter), so DataParallel only replicate overhead is avoided by
temporarily running in single-GPU mode via the raw model.
================================================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGGradCAM:
    """
    Grad-CAM attribution for RD-GANet electrode importance.

    Handles nn.DataParallel transparently: hooks are always registered
    on model.module.gat.l2 (the actual GATEncoder layer), not on the
    DataParallel wrapper.
    """

    def __init__(self, model: nn.Module):
        # Store the outer (possibly DP-wrapped) model for bookkeeping
        self.model = model

        # Requirement 8: always work with the raw underlying model for
        # hook registration and Grad-CAM forward/backward.
        self._raw = model.module if isinstance(model, nn.DataParallel) else model

        self._acts    = None
        self._grads   = None
        self._handles : list = []

        self._register_hooks()

    # =========================================================
    # Hook Registration
    # =========================================================

    def _register_hooks(self) -> None:
        """Register forward + backward hooks on the final GAT layer."""
        # INS_HDGS_CMT full config: uses dynamic_gat.gat.l2
        # Fallback: legacy gat.l2 for old-style models
        raw = self._raw
        if hasattr(raw, "dynamic_gat") and raw.dynamic_gat is not None:
            target = raw.dynamic_gat.gat.l2
        elif hasattr(raw, "gat") and raw.gat is not None and hasattr(raw.gat, "l2"):
            target = raw.gat.l2
        else:
            raise AttributeError(
                "Cannot find hookable GAT layer. "
                "Expected model.dynamic_gat.gat.l2 or model.gat.l2"
            )

        def _fwd(module, inp, out):
            # out = (node_emb, alpha)
            self._acts = out[0]

        def _bwd(module, grad_in, grad_out):
            self._grads = grad_out[0]

        self._handles.append(target.register_forward_hook(_fwd))
        self._handles.append(target.register_full_backward_hook(_bwd))

    # =========================================================
    # Remove Hooks
    # =========================================================

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # =========================================================
    # Generate Grad-CAM
    # =========================================================

    def generate(
        self,
        batch        : dict,
        target_class : Optional[int] = None,
        device       : torch.device  = torch.device("cpu"),
    ) -> dict:
        """
        Compute electrode-level Grad-CAM for one sample.

        DataParallel note: Grad-CAM is run on the *raw* model (single-GPU)
        to ensure hooks fire exactly once and gradient flow is unambiguous.
        cudnn deterministic mode is disabled to allow LSTM backward.

        Parameters
        ----------
        batch        : dict of tensors (from dataset.__getitem__)
        target_class : class index to explain; None → predicted class
        device       : target device

        Returns
        -------
        dict with keys: cam, connectivity, gate, pred_class, pred_prob
        """
        # ----------------------------------------------------------
        # Grad-CAM + RNN: disable cuDNN benchmarking to allow backward
        # ----------------------------------------------------------
        torch.backends.cudnn.enabled = False

        # ----------------------------------------------------------
        # Move tensors to device
        # ----------------------------------------------------------
        eeg_win  = batch["eeg_windows"].to(device)
        adj_mat  = batch["adj_matrices"].to(device)
        et_seq   = batch["et_seq"].to(device)
        roi_vec  = batch["roi_vector"].to(device)
        wadj     = batch.get("weighted_adjs")
        if wadj is not None:
            wadj = wadj.to(device)

        # Add batch dim only if missing
        if eeg_win.dim() == 3:
            eeg_win = eeg_win.unsqueeze(0)
        if adj_mat.dim() == 3:
            adj_mat = adj_mat.unsqueeze(0)
        if et_seq.dim() == 2:
            et_seq = et_seq.unsqueeze(0)
        if roi_vec.dim() == 1:
            roi_vec = roi_vec.unsqueeze(0)
        if wadj is not None and wadj.dim() == 3:
            wadj = wadj.unsqueeze(0)

        # ----------------------------------------------------------
        # Run on raw model (no DataParallel scatter for single sample)
        # ----------------------------------------------------------
        raw = self._raw
        raw.train()                        # required for gradient flow
        for p in raw.parameters():
            p.requires_grad = True
        raw.zero_grad()

        out = raw(
            eeg_windows   = eeg_win,
            adj_matrices  = adj_mat,
            et_seq        = et_seq,
            roi_vector    = roi_vec,
            weighted_adjs = wadj,
        )

        logits = out["logits"]
        probs  = F.softmax(logits, dim=1)

        # Select target class
        if target_class is None:
            target_class = int(probs.argmax(dim=1).item())
        pred_prob = float(probs[0, target_class].item())

        # Backward
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # ----------------------------------------------------------
        # Validate hooks
        # ----------------------------------------------------------
        if self._acts is None or self._grads is None:
            return {
                "cam"        : None,
                "connectivity": None,
                "gate"       : None,
                "pred_class" : target_class,
                "pred_prob"  : pred_prob,
            }

        # ----------------------------------------------------------
        # CAM computation
        # ----------------------------------------------------------
        acts   = self._acts.detach().cpu()    # (B*W, C, D)
        grads  = self._grads.detach().cpu()

        # Global average pooling over feature dimension → channel weights
        weights = grads.mean(dim=-1)          # (B*W, C)

        # Weighted sum of activations
        cam_bw  = (weights.unsqueeze(-1) * acts).sum(dim=-1)  # (B*W, C)
        cam_bw  = F.relu(cam_bw)

        # Mean over windows → (C,)
        cam = cam_bw.mean(dim=0).numpy()

        # Normalise to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)

        # ----------------------------------------------------------
        # Connectivity importance
        # ----------------------------------------------------------
        adj_np = (
            adj_mat.squeeze(0).mean(dim=0).detach().cpu().numpy()
        )
        connectivity = np.outer(cam, cam) * adj_np

        # ----------------------------------------------------------
        # ROI gate
        # ----------------------------------------------------------
        gate = None
        if out.get("gate") is not None:
            gate = out["gate"][0].detach().cpu().numpy()

        return {
            "cam"         : cam,
            "connectivity": connectivity,
            "gate"        : gate,
            "pred_class"  : target_class,
            "pred_prob"   : pred_prob,
        }
