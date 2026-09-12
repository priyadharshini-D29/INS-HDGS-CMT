"""
================================================================
NEUMA Phase 8 vNext — EEG Self-Supervised Augmentations
================================================================
Stochastic augmentations for contrastive (SimCLR / NT-Xent) pretraining of
the EEG encoder.  Operate on the band-power node-feature tensor that the EEG
branch consumes:

    eeg_windows : (B, W, C, F)
        W = number of temporal windows   (temporal axis)
        C = number of EEG channels        (channel axis)
        F = number of frequency bands = 5 (delta, theta, alpha, beta, gamma)

Two independent calls to `augment_view` produce a positive pair (same sample,
different augmentations).  The graph adjacency is NOT touched here — we augment
node features only and keep the precomputed connectivity fixed (standard SimCLR
practice: perturb the representation, not the topology).

All ops are batch-wise, vectorised, and differentiable-safe (used under
no-grad on the input but the encoder still backprops normally).
================================================================
"""

from __future__ import annotations

import torch


# ── Individual augmentations ─────────────────────────────────────────────────

def temporal_mask(x: torch.Tensor, max_frac: float = 0.4) -> torch.Tensor:
    """Zero a random contiguous span of windows (the W/temporal axis).

    A contiguous span better mimics a dropped temporal segment than scattered
    windows.  Each sample in the batch gets its own random span.
    """
    B, W, C, F = x.shape
    out = x.clone()
    span = int(round(W * max_frac))
    if span < 1:
        return out
    for b in range(B):
        length = int(torch.randint(1, span + 1, (1,)).item())
        start  = int(torch.randint(0, W - length + 1, (1,)).item())
        out[b, start:start + length] = 0.0
    return out


def channel_dropout(x: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """Independently zero each channel (C axis) with probability `p`.

    A dropped channel is zeroed across all windows and bands, simulating a
    bad/absent electrode.
    """
    B, W, C, F = x.shape
    keep = (torch.rand(B, 1, C, 1, device=x.device) > p).to(x.dtype)
    return x * keep


def gaussian_noise(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Add zero-mean Gaussian noise scaled by the per-sample feature std."""
    if sigma <= 0:
        return x
    std = x.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    return x + torch.randn_like(x) * (sigma * std)


def freq_mask(x: torch.Tensor, max_bands: int = 2) -> torch.Tensor:
    """Zero a random subset of frequency bands (the last/F axis).

    With F=5 bands this drops 1..max_bands bands per sample, the SpecAugment
    analogue for band-power features.
    """
    B, W, C, F = x.shape
    out = x.clone()
    k = min(max_bands, F - 1)
    if k < 1:
        return out
    for b in range(B):
        n_drop = int(torch.randint(1, k + 1, (1,)).item())
        bands  = torch.randperm(F, device=x.device)[:n_drop]
        out[b, :, :, bands] = 0.0
    return out


# ── Composed view ────────────────────────────────────────────────────────────

def augment_view(
    x: torch.Tensor,
    p_temporal: float = 0.5,
    p_channel : float = 0.5,
    p_noise   : float = 0.5,
    p_freq    : float = 0.5,
) -> torch.Tensor:
    """Compose the four augmentations, each applied with its own probability.

    Two independent calls on the same `x` yield a SimCLR positive pair.
    Gaussian noise is applied last so masked-out regions can still receive a
    small perturbation (keeps the two views from collapsing to identical zeros).
    """
    out = x
    if torch.rand(1).item() < p_temporal:
        out = temporal_mask(out)
    if torch.rand(1).item() < p_channel:
        out = channel_dropout(out)
    if torch.rand(1).item() < p_freq:
        out = freq_mask(out)
    if torch.rand(1).item() < p_noise:
        out = gaussian_noise(out)
    return out


if __name__ == "__main__":
    # Sanity check: shape preserved, output differs, no NaNs.
    torch.manual_seed(0)
    x = torch.randn(8, 10, 24, 5)
    for fn in (temporal_mask, channel_dropout, gaussian_noise, freq_mask):
        y = fn(x)
        assert y.shape == x.shape, f"{fn.__name__} changed shape: {y.shape}"
        assert torch.isfinite(y).all(), f"{fn.__name__} produced non-finite values"
        print(f"{fn.__name__:18s} ok  Δmean={float((y - x).abs().mean()):.4f}")
    v1, v2 = augment_view(x), augment_view(x)
    assert v1.shape == x.shape and v2.shape == x.shape
    assert torch.isfinite(v1).all() and torch.isfinite(v2).all()
    assert float((v1 - v2).abs().mean()) > 0, "views identical"
    print(f"augment_view       ok  Δ(v1,v2)={float((v1 - v2).abs().mean()):.4f}")
    print("eeg_augment self-test passed.")
