"""
================================================================
INS-HDGS-CMT — SNN spike-sparsity & energy-proxy measurement
================================================================
The component ablation shows the LIF spiking encoder does not change accuracy
(Δ balanced accuracy ≈ 0). Its justification is therefore *efficiency*, not
accuracy. This script measures that efficiency directly.

What it does
------------
1. Rebuilds the trained ``SpikingEEGEncoder`` (weights from a full-model
   checkpoint) and runs real NeuMa EEG epochs through it.
2. Records the binary spike trains of every LIF layer (by wrapping the
   surrogate-spike op) and reports the mean firing rate (sparsity).
3. Converts firing rate into an energy proxy: the spiking (event-driven)
   synaptic operations require accumulate (AC) ops, whereas an equivalent dense
   ANN of the same shape requires multiply-accumulate (MAC) ops. Using 45 nm
   CMOS energies (Horowitz, ISSCC 2014): E_MAC = 4.6 pJ, E_AC = 0.9 pJ.

   E_SNN(spiking layers) = rho * MACs * E_AC      (only firing neurons compute)
   E_ANN(same layers)    =        MACs * E_MAC
   ratio                 = rho * E_AC / E_MAC

Outputs (results/statistics/):
  snn_energy.json / .md

Usage
-----
  python analysis/snn_energy.py
  python analysis/snn_energy.py --max-epochs 300 --device cpu
================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
PHASE8 = HERE.parents[1]
ROOT = PHASE8.parent
for p in (str(PHASE8), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import models.spiking_encoder as se                       # noqa: E402
from models.spiking_encoder import SpikingEEGEncoder      # noqa: E402
from data.dataset import NeumaGraphDataset                # noqa: E402
from config.settings import (                             # noqa: E402
    EMBED_DIM, SNN_HIDDEN_DIM, SNN_TIME_STEPS, SNN_DECAY, SNN_THRESHOLD,
)

OUT = PHASE8 / "results" / "statistics"
E_MAC_PJ = 4.6    # 45 nm CMOS, 32-bit MAC (Horowitz 2014)
E_AC_PJ = 0.9     # 45 nm CMOS, 32-bit AC

# ── spike recorder: wrap the module-level surrogate_spike so LIFLayer.forward
#    (which calls it by name) feeds us the binary spikes without code changes ──
_REC = {"spikes": 0.0, "elems": 0}
_orig_spike = se.surrogate_spike


def _recording_spike(v, threshold=1.0, beta=5.0):
    s = _orig_spike(v, threshold, beta)
    _REC["spikes"] += float(s.sum().item())
    _REC["elems"] += int(s.numel())
    return s


def _load_encoder(ckpt_path, n_channels, device):
    enc = SpikingEEGEncoder(
        n_channels=n_channels, embed_dim=EMBED_DIM, hidden_dim=SNN_HIDDEN_DIM,
        n_layers=2, time_steps=SNN_TIME_STEPS, decay=SNN_DECAY,
        threshold=SNN_THRESHOLD)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    sub = {k[len("snn_encoder."):]: v for k, v in sd.items()
           if k.startswith("snn_encoder.")}
    missing, unexpected = enc.load_state_dict(sub, strict=False)
    print(f"[snn] loaded {len(sub)} snn_encoder tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return enc.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    device = torch.device(args.device)

    ds = NeumaGraphDataset(precompute_graphs=False, augment=False)
    X = np.stack([np.asarray(e, np.float32).T for e in ds.raw_eeg])   # (N,C,S)
    if args.max_epochs:
        X = X[:args.max_epochs]
    n_channels = X.shape[1]
    print(f"[snn] EEG tensor {X.shape}  (C={n_channels}, S={X.shape[2]})")

    ckpt = sorted((PHASE8 / "output" / "checkpoints" /
                   "repro_focal_g3p0_effective_num_37").glob("*fold01*.pt"))[0]
    print(f"[snn] checkpoint: {ckpt.name}")
    enc = _load_encoder(ckpt, n_channels, device)

    # install recorder
    se.surrogate_spike = _recording_spike
    try:
        with torch.no_grad():
            xb = torch.as_tensor(X, dtype=torch.float32, device=device)
            for i in range(0, len(xb), 64):
                enc(xb[i:i + 64])
    finally:
        se.surrogate_spike = _orig_spike   # restore

    firing_rate = _REC["spikes"] / max(_REC["elems"], 1)

    # MACs of the LIF synapses (the spiking, event-driven portion)
    lif_macs = sum(l.synapse.in_features * l.synapse.out_features
                   for l in enc.lif_layers) * SNN_TIME_STEPS
    # input_proj is a dense (non-spiking) layer driven by continuous input → MACs
    proj_macs = n_channels * SNN_HIDDEN_DIM * SNN_TIME_STEPS

    e_snn_lif = firing_rate * lif_macs * E_AC_PJ
    e_ann_lif = lif_macs * E_MAC_PJ
    ratio_lif = (firing_rate * E_AC_PJ) / E_MAC_PJ
    # whole-encoder (input_proj counted as MAC in both):
    e_snn_total = proj_macs * E_MAC_PJ + e_snn_lif
    e_ann_total = (proj_macs + lif_macs) * E_MAC_PJ

    res = {
        "checkpoint": ckpt.name, "n_epochs_measured": int(len(X)),
        "n_channels": int(n_channels), "time_steps": SNN_TIME_STEPS,
        "mean_firing_rate": firing_rate,
        "sparsity": 1.0 - firing_rate,
        "lif_synapse_macs": int(lif_macs),
        "input_proj_macs": int(proj_macs),
        "E_MAC_pJ": E_MAC_PJ, "E_AC_pJ": E_AC_PJ,
        "energy_pJ_snn_lif": e_snn_lif, "energy_pJ_ann_lif": e_ann_lif,
        "energy_ratio_lif_snn_over_ann": ratio_lif,
        "energy_pJ_snn_encoder": e_snn_total,
        "energy_pJ_ann_encoder": e_ann_total,
        "energy_ratio_encoder_snn_over_ann": e_snn_total / e_ann_total,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snn_energy.json").write_text(json.dumps(res, indent=2))

    md = [
        "# SNN spike-sparsity & energy proxy\n",
        f"Trained LIF encoder (`{ckpt.name}`), {len(X)} EEG epochs, "
        f"{SNN_TIME_STEPS} simulation steps.\n",
        f"- **Mean firing rate:** {firing_rate*100:.2f}%  "
        f"(sparsity {100*(1-firing_rate):.2f}% — neurons silent most steps).",
        f"- **LIF synaptic MACs / inference:** {lif_macs:,} "
        f"(event-driven in the SNN).",
        f"- **Energy (LIF layers), 45 nm proxy:** SNN {e_snn_lif:,.0f} pJ vs "
        f"ANN {e_ann_lif:,.0f} pJ → **{ratio_lif*100:.1f}%** of ANN "
        f"({1/ratio_lif:.1f}× lower).",
        f"- **Energy (whole encoder, dense input-projection counted as MAC in "
        f"both):** SNN {e_snn_total:,.0f} pJ vs ANN {e_ann_total:,.0f} pJ → "
        f"{100*e_snn_total/e_ann_total:.1f}% of ANN.",
        "",
        "Energies: E_MAC=4.6 pJ, E_AC=0.9 pJ (45 nm CMOS, Horowitz ISSCC 2014). "
        "The spiking layers compute only when a presynaptic neuron fires, so "
        "their cost scales with the firing rate; the ratio is "
        "rho·E_AC/E_MAC. This quantifies the efficiency rationale for the LIF "
        "encoder independently of its (neutral) effect on accuracy.",
    ]
    (OUT / "snn_energy.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
