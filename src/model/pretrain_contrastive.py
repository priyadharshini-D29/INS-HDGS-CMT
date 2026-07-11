#!/usr/bin/env python
"""
================================================================
NEUMA Phase 8 vNext — Phase 4: EEG Self-Supervised Pretraining
================================================================
SimCLR-style contrastive pretraining of the INS-HDGS-CMT *EEG encoder only*.

For each EEG sample we build two augmented views (temporal mask, channel
dropout, gaussian noise, frequency mask — see models/eeg_augment.py), encode
both through ``model.encode_eeg`` + a projection head, and minimise the
symmetrised NT-Xent loss (τ=0.07).  Only the EEG-branch parameters and the
projection head are optimised; ET / fusion / classifier heads are untouched.

The learned EEG-branch weights are saved to
    output/checkpoints/ssl_eeg_encoder.pt
and later transferred into a fresh model via ``main.py --pretrained-eeg``.

This is label-free: subjects are pooled and engagement labels are ignored.

Usage
-----
    python pretrain_contrastive.py --epochs 100 --batch-size 256
    python pretrain_contrastive.py --epochs 2          # smoke test
================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    DEVICE, EMBED_DIM, OUTPUT_DIR, SUBJECT_IDS,
)
from data.dataset import NeumaGraphDataset, collate_fn
from evaluation.losocv import _make_model
from models.contrastive import NTXentLoss
from models.eeg_augment import augment_view

# EEG-branch submodule prefixes — exactly the parameters encode_eeg uses, and
# the keys that get saved / transferred.  Mirrors ins_hdgs_cmt.encode_eeg.
EEG_PREFIXES = ("dynamic_gat.", "gat.", "snn_encoder.", "snn_band_weights", "eeg_merge.")

CKPT_PATH = OUTPUT_DIR / "checkpoints" / "ssl_eeg_encoder.pt"


class ProjectionHead(nn.Module):
    """SimCLR projection head: D → D → proj_dim (used only during pretraining)."""

    def __init__(self, dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


def _eeg_branch_params(model):
    """Yield (name, param) for EEG-branch params only."""
    for name, p in model.named_parameters():
        if name.startswith(EEG_PREFIXES):
            yield name, p


def _eeg_branch_state(model):
    """state_dict slice for the EEG branch (the transfer payload)."""
    return {k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if k.startswith(EEG_PREFIXES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=100)
    ap.add_argument("--batch-size", type=int,   default=256)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--out",        type=str,   default=str(CKPT_PATH))
    ap.add_argument("--num-workers", type=int,  default=4)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ssl] device={DEVICE}  epochs={args.epochs}  bs={args.batch_size}  τ={args.temperature}")

    # ── Data: all subjects pooled, label-free ──────────────────────────────
    ds = NeumaGraphDataset(
        subject_ids=SUBJECT_IDS,
        precompute_graphs=True,
        augment=False,            # augmentation happens on-the-fly below
        norm_mode="zscore",       # match the baseline normalization
    )
    print(f"[ssl] pooled dataset: {len(ds)} epochs, "
          f"n_eeg_ch={ds.n_eeg_ch}, n_classes={ds.n_classes}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model: full architecture, but we only use/optimise the EEG branch ───
    model = _make_model(ds.n_eeg_ch, ds.n_classes).to(DEVICE)
    proj  = ProjectionHead(EMBED_DIM).to(DEVICE)

    params = [p for _, p in _eeg_branch_params(model)] + list(proj.parameters())
    n_eeg_params = sum(p.numel() for _, p in _eeg_branch_params(model))
    print(f"[ssl] optimising {n_eeg_params:,} EEG-branch params + projection head")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = NTXentLoss(temperature=args.temperature)

    model.train()
    proj.train()
    for epoch in range(1, args.epochs + 1):
        t0, total, nb = time.perf_counter(), 0.0, 0
        for batch in loader:
            eeg = batch["eeg_windows"].to(DEVICE, non_blocking=True)   # (B,W,C,5)
            adj = batch["adj_matrices"].to(DEVICE, non_blocking=True)  # (B,W,C,C)

            v1 = augment_view(eeg)
            v2 = augment_view(eeg)
            z1 = proj(model.encode_eeg(v1, adj))
            z2 = proj(model.encode_eeg(v2, adj))
            loss = criterion(z1, z2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            total += float(loss.item()); nb += 1
        sched.step()
        print(f"[ssl] epoch {epoch:3d}/{args.epochs}  "
              f"nt_xent={total / max(nb, 1):.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  t={time.perf_counter() - t0:.1f}s",
              flush=True)

    payload = _eeg_branch_state(model)
    torch.save(payload, out_path)
    print(f"[ssl] saved {len(payload)} EEG-branch tensors → {out_path}")


if __name__ == "__main__":
    main()
