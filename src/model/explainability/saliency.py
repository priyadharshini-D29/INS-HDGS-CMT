"""
================================================================
NEUMA Phase 8 — Electrode Saliency Maps
================================================================
Computes input-gradient saliency w.r.t. EEG node features.

  saliency_c = ‖ ∂y / ∂x_c ‖   (L2 over feature dim)

This measures how sensitively the prediction changes when
the band-power features of electrode c are perturbed.
================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F


class ElectrodeSaliency:
    """
    Input-gradient saliency for EEG channel importance.

    Parameters
    ----------
    model  : RDGANet instance
    device : computation device
    """

    def __init__(self, model, device=None):
        self.model  = model
        self.device = device or torch.device("cpu")

    @torch.enable_grad()
    def compute(
        self,
        batch        : dict,
        target_class : int = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        batch        : single sample dict (keys: eeg_windows, adj_matrices, …)
        target_class : class to explain (None → predicted)

        Returns
        -------
        saliency : (C,)  per-electrode importance, normalised [0, 1]
        """
        self.model.eval()
        device = self.device

        eeg_win = batch["eeg_windows"].unsqueeze(0).to(device).float()
        eeg_win.requires_grad_(True)

        adj_mat = batch["adj_matrices"].unsqueeze(0).to(device)
        et_seq  = batch["et_seq"].unsqueeze(0).to(device)
        roi_vec = batch["roi_vector"].unsqueeze(0).to(device)

        out = self.model(
            eeg_windows  = eeg_win,
            adj_matrices = adj_mat,
            et_seq       = et_seq,
            roi_vector   = roi_vec,
        )

        logits = out["logits"]
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Gradient w.r.t. (B=1, W, C, F_node)
        grad = eeg_win.grad.detach().cpu()              # (1, W, C, 5)
        saliency = grad.squeeze(0)                      # (W, C, 5)

        # L2 norm over feature dim, then average over windows
        saliency = saliency.norm(dim=-1).mean(dim=0)    # (C,)
        saliency = saliency.numpy()

        # Normalise
        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        return saliency

    def compute_band_saliency(self, batch, target_class=None) -> np.ndarray:
        """
        Return per-band saliency averaged over electrodes.

        Returns
        -------
        band_saliency : (5,)  [delta, theta, alpha, beta, gamma]
        """
        self.model.eval()
        device = self.device

        eeg_win = batch["eeg_windows"].unsqueeze(0).to(device).float()
        eeg_win.requires_grad_(True)

        adj_mat = batch["adj_matrices"].unsqueeze(0).to(device)
        et_seq  = batch["et_seq"].unsqueeze(0).to(device)
        roi_vec = batch["roi_vector"].unsqueeze(0).to(device)

        out = self.model(
            eeg_windows  = eeg_win,
            adj_matrices = adj_mat,
            et_seq       = et_seq,
            roi_vector   = roi_vec,
        )

        logits = out["logits"]
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        grad = eeg_win.grad.detach().cpu()              # (1, W, C, 5)
        band_sal = grad.squeeze(0).abs().mean(dim=(0, 1))  # (5,)
        band_sal = band_sal.numpy()

        b_min, b_max = band_sal.min(), band_sal.max()
        if b_max > b_min:
            band_sal = (band_sal - b_min) / (b_max - b_min)

        return band_sal
