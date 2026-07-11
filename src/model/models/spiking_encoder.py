"""
================================================================
INS-HDGS-CMT — EEG Spiking Neural Network Encoder
================================================================
Implements a multi-layer Leaky Integrate-and-Fire (LIF) spiking
neural network for temporal EEG encoding.

Scientific rationale
--------------------
Spiking neurons capture temporal firing dynamics analogous to
cortical neural activity. For cognitive engagement decoding:

  • Frontal theta bursts → SNN fire-rate ↑
  • Alpha suppression   → sustained depolarisation
  • Beta engagement     → rhythmic spike trains

Surrogate gradient
------------------
The Heaviside spike function is non-differentiable.
We use the fast sigmoid surrogate (Zenke & Ganguli, 2021):

  dH/dV ≈ 1 / (β·|V - threshold| + 1)²

This enables standard back-propagation through spike events.

Architecture
------------
  EEG (B, C, S)  →  Linear project  →  LIF × n_layers  →  pool  →  (B, D)

Each LIF layer:
  V[t] = decay · V[t-1] + W·x[t]       (integrate)
  S[t] = H(V[t] - threshold)            (fire)
  V[t] = V[t] · (1 - S[t])             (reset)
================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Surrogate Gradient ────────────────────────────────────────────────────────

class _SurrogateHeaviside(torch.autograd.Function):
    """
    Forward: Heaviside step function  H(v - threshold)
    Backward: Fast-sigmoid surrogate gradient  (Zenke & Ganguli, 2021)
    """
    @staticmethod
    def forward(ctx, v: torch.Tensor, threshold: float, beta: float):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        ctx.beta      = beta
        return (v >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v,          = ctx.saved_tensors
        threshold   = ctx.threshold
        beta        = ctx.beta
        sg          = grad_output / (beta * (v - threshold).abs() + 1.0).pow(2)
        return sg, None, None


def surrogate_spike(v: torch.Tensor,
                    threshold: float = 1.0,
                    beta: float = 5.0) -> torch.Tensor:
    return _SurrogateHeaviside.apply(v, threshold, beta)


# ── LIF Neuron Layer ──────────────────────────────────────────────────────────

class LIFLayer(nn.Module):
    """
    Vectorised Leaky Integrate-and-Fire layer.

    Parameters
    ----------
    in_dim     : input feature dimension
    out_dim    : number of LIF neurons
    decay      : membrane potential decay constant τ (0 < τ < 1)
    threshold  : spike threshold V_th
    beta       : surrogate gradient sharpness
    dropout    : dropout on synaptic inputs
    """

    def __init__(
        self,
        in_dim    : int,
        out_dim   : int,
        decay     : float = 0.90,
        threshold : float = 1.00,
        beta      : float = 5.00,
        dropout   : float = 0.20,
    ):
        super().__init__()
        self.synapse   = nn.Linear(in_dim, out_dim, bias=True)
        self.decay     = decay
        self.threshold = threshold
        self.beta      = beta
        self.dropout   = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(out_dim)

    def forward(
        self,
        x_seq    : torch.Tensor,    # (B, T, in_dim)
        v_init   : torch.Tensor = None,
    ):
        """
        Process a temporal sequence through LIF dynamics.

        Returns
        -------
        spikes  : (B, T, out_dim)  spike train
        v_final : (B, out_dim)     final membrane potential
        """
        B, T, _ = x_seq.shape
        D       = self.synapse.out_features

        v = v_init if v_init is not None else torch.zeros(B, D, device=x_seq.device)

        spike_list = []
        for t in range(T):
            i_t = self.dropout(self.synapse(x_seq[:, t, :]))   # (B, D)
            v   = self.decay * v + i_t                          # integrate
            s   = surrogate_spike(v, self.threshold, self.beta) # fire
            v   = v * (1.0 - s)                                 # hard reset
            spike_list.append(s)

        spikes = torch.stack(spike_list, dim=1)                 # (B, T, D)
        return self.norm(spikes), v


# ── SNN EEG Encoder ──────────────────────────────────────────────────────────

class SpikingEEGEncoder(nn.Module):
    """
    Multi-layer SNN encoder for temporal EEG engagement decoding.

    Input  : (B, C, S)   — C channels, S time samples (raw or band-filtered)
    Output : (B, embed_dim)

    Pipeline
    --------
    1. Project C channels → hidden_dim at each time step
    2. n_layers × LIF (surrogate-gradient)
    3. Temporal aggregation: mean over time + max over time → concat
    4. Linear projection to embed_dim
    """

    def __init__(
        self,
        n_channels   : int,
        embed_dim    : int   = 128,
        hidden_dim   : int   = 128,
        n_layers     : int   = 2,
        time_steps   : int   = 10,      # number of SNN simulation steps
        decay        : float = 0.90,
        threshold    : float = 1.00,
        beta         : float = 5.00,
        dropout      : float = 0.20,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.time_steps = time_steps

        # Step 1: temporal down-sample + project to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Step 2: LIF stack
        dims   = [hidden_dim] + [hidden_dim] * n_layers
        layers = []
        for i in range(n_layers):
            layers.append(LIFLayer(dims[i], dims[i + 1],
                                   decay=decay, threshold=threshold,
                                   beta=beta, dropout=dropout))
        self.lif_layers = nn.ModuleList(layers)

        # Step 3+4: aggregate → project
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, S)  raw EEG (channels × samples)

        Returns
        -------
        emb : (B, embed_dim)
        """
        B, C, S = x.shape

        # Subsample / fold time axis into SNN time steps
        if S != self.time_steps:
            # Temporal average pooling to reduce to time_steps
            x = F.adaptive_avg_pool1d(x, self.time_steps)   # (B, C, T)

        x_seq = x.permute(0, 2, 1)    # (B, T, C)

        # Input projection at each time step
        x_seq = self.input_proj(x_seq)    # (B, T, hidden_dim)

        # LIF stack
        spikes = x_seq
        for lif in self.lif_layers:
            spikes, _ = lif(spikes)

        # Temporal aggregation: mean + max
        mean_pool = spikes.mean(dim=1)     # (B, D)
        max_pool  = spikes.max(dim=1).values
        agg       = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2D)

        return self.out_proj(agg)          # (B, embed_dim)

    @property
    def fire_rates(self) -> None:
        """Fire-rate diagnostics (for visualisation, not used in forward)."""
        pass
