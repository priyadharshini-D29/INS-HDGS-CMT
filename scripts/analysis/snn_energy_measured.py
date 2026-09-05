"""
================================================================
Measured cost of the spiking front-end (Reviewer 2, comment 4)
================================================================
The manuscript's "2.1 % of the energy of an equivalent dense network" is
an operation-count estimate under a 45-nm CMOS energy model, valid only
for event-driven (neuromorphic) hardware. The reviewer asks for physical
measurements. Neuromorphic hardware is not available to us, so this
script reports what CAN be measured on commodity hardware, and states
plainly that on GPUs/CPUs the LIF layers give no energy advantage:

  1. spike sparsity of the trained LIF layers on real epochs (as before)
  2. wall-clock latency of (a) the trained SpikingEEGEncoder and (b) a
     dense ANN of identical layer widths (LIF -> Linear+GELU), batch 1
     and batch 32, on the current device
  3. if an NVIDIA GPU is used: mean board power (nvidia-smi, sampled at
     10 Hz) during a sustained loop, hence measured energy per inference
     (J) for the spiking and the dense encoder
  4. the analytic neuromorphic estimate (AC vs MAC operation counts),
     restated alongside, so the reader sees both numbers side by side.

Outputs: results/statistics/snn_energy_measured.{json,md}

Usage
-----
  cd src/model
  python ../../scripts/analysis/snn_energy_measured.py --ckpt output/checkpoints/<label>/<label>_fold01_e0.pt \
      [--device cuda] [--n-iter 300]
================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "src" / "model"
for p in (str(MODEL), str(MODEL.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import models.spiking_encoder as se                                       # noqa: E402
from config.settings import (EMBED_DIM, SNN_DECAY, SNN_HIDDEN_DIM,        # noqa: E402
                             SNN_THRESHOLD, SNN_TIME_STEPS, SUBJECT_IDS)
from data.dataset import NeumaGraphDataset                                # noqa: E402
from models.spiking_encoder import SpikingEEGEncoder                      # noqa: E402

OUT = ROOT / "results" / "statistics"
E_MAC_PJ, E_AC_PJ = 4.6, 0.9   # Horowitz, ISSCC 2014 (45 nm)


class DenseTwin(nn.Module):
    """Same widths as SpikingEEGEncoder, LIF layers replaced by Linear+GELU+LayerNorm."""
    def __init__(self, snn: SpikingEEGEncoder):
        super().__init__()
        self.input_proj = snn.input_proj
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(l.synapse.in_features, l.synapse.out_features),
                                                   nn.GELU(), nn.LayerNorm(l.synapse.out_features))
                                     for l in snn.lif_layers])
        self.out_proj = snn.out_proj

    def forward(self, x):                       # (B, C, S) -> (B, D)
        h = self.input_proj(x.transpose(1, 2))  # (B, S, H)
        for l in self.layers:
            h = l(h)
        return self.out_proj(torch.cat([h.mean(1), h.max(1).values], dim=-1))


class PowerSampler(threading.Thread):
    def __init__(self, period=0.1):
        super().__init__(daemon=True); self.period = period; self.samples = []; self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                              text=True, timeout=2)
                self.samples.append(float(out.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(self.period)

    def stop(self):
        self._stop.set(); self.join(timeout=2)
        return float(np.mean(self.samples)) if self.samples else float("nan")


def bench(model, x, n_iter, device, sample_power):
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ps = PowerSampler() if sample_power else None
        if ps: ps.start()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        power = ps.stop() if ps else float("nan")
    lat_ms = dt / n_iter * 1000
    return dict(latency_ms=lat_ms, power_w=power, energy_mJ=(power * dt / n_iter * 1000) if np.isfinite(power) else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="full-model checkpoint (snn_encoder.* weights); random init if omitted")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    subjects = args.subjects.split(",") if args.subjects else SUBJECT_IDS
    ds = NeumaGraphDataset(subject_ids=subjects, precompute_graphs=True, augment=False)
    # the model feeds the SNN a band-weighted proxy of shape (B, C, W) built from the
    # node features; reproduce that input with the same softmax band weights (init values)
    band_w = torch.softmax(torch.tensor([0.05, 0.40, 0.35, 0.10, 0.10]), 0)
    X = torch.stack([(ds[i]["eeg_windows"] * band_w).sum(-1).permute(1, 0) for i in range(len(ds))])   # (N, C, W)
    X = torch.nn.functional.layer_norm(X, X.shape[-1:])
    C = X.shape[1]

    snn = SpikingEEGEncoder(n_channels=C, embed_dim=EMBED_DIM, hidden_dim=SNN_HIDDEN_DIM, n_layers=2,
                            time_steps=SNN_TIME_STEPS, decay=SNN_DECAY, threshold=SNN_THRESHOLD)
    loaded = 0
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        sub = {k.split("snn_encoder.", 1)[1]: v for k, v in sd.items() if "snn_encoder." in k}
        loaded = len(sub) - len(snn.load_state_dict(sub, strict=False).missing_keys)
    dense = DenseTwin(snn)
    snn.to(device).eval(); dense.to(device).eval()

    # 1. sparsity
    rec = {"s": 0.0, "n": 0}
    orig = se.surrogate_spike
    def _rec(v, threshold=1.0, beta=5.0):
        s = orig(v, threshold, beta); rec["s"] += float(s.sum()); rec["n"] += s.numel(); return s
    se.surrogate_spike = _rec
    try:
        with torch.no_grad():
            for i in range(0, len(X), 64):
                snn(X[i:i + 64].to(device))
    finally:
        se.surrogate_spike = orig
    rho = rec["s"] / max(rec["n"], 1)

    # 2/3. latency + power
    res = {}
    sample_power = device.type == "cuda" and shutil.which("nvidia-smi") is not None
    for bs in (1, 32):
        xb = X[:bs].to(device)
        res[f"snn_b{bs}"] = bench(snn, xb, args.n_iter, device, sample_power)
        res[f"dense_b{bs}"] = bench(dense, xb, args.n_iter, device, sample_power)
    idle_power = float("nan")
    if sample_power:
        ps = PowerSampler(); ps.start(); time.sleep(2.0); idle_power = ps.stop()

    # 4. analytic (neuromorphic) estimate
    lif_macs = sum(l.synapse.in_features * l.synapse.out_features for l in snn.lif_layers) * SNN_TIME_STEPS
    e_snn = rho * lif_macs * E_AC_PJ; e_ann = lif_macs * E_MAC_PJ
    analytic = dict(firing_rate=rho, lif_macs_per_inference=int(lif_macs), E_snn_pJ=e_snn, E_ann_pJ=e_ann,
                    ratio=e_snn / e_ann)

    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    report = dict(device=str(device), device_name=dev_name, checkpoint=args.ckpt, n_snn_tensors_loaded=loaded,
                  n_epochs=len(X), n_iter=args.n_iter, idle_power_w=idle_power, measured=res, analytic=analytic)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out / "snn_energy_measured.json", "w"), indent=1)

    lines = [f"# Spiking front-end: measured cost on commodity hardware ({dev_name})", "",
             f"Trained LIF firing rate on {len(X)} real epochs: {rho*100:.1f}% (sparsity {100-rho*100:.1f}%).", "",
             "| encoder | batch | latency / inference (ms) | board power (W) | energy / inference (mJ) |", "|---|---|---|---|---|"]
    for k, v in res.items():
        name, bs = k.rsplit("_b", 1)
        lines.append(f"| {'spiking (LIF)' if name == 'snn' else 'dense twin (Linear+GELU)'} | {bs} | {v['latency_ms']:.3f} | "
                     f"{v['power_w']:.1f} | {v['energy_mJ']:.4f} |")
    if np.isfinite(idle_power):
        lines.append(f"\nIdle board power: {idle_power:.1f} W (subtract for dynamic energy).")
    lines += ["", f"Analytic neuromorphic estimate (event-driven AC vs. dense MAC, 45 nm): "
              f"{analytic['ratio']*100:.1f}% of the dense energy for the LIF layers "
              f"({lif_macs:,} synaptic operations/inference).", "",
              "Interpretation: on a GPU/CPU every LIF time-step still executes dense matrix products, so the "
              "spiking encoder is not cheaper than its dense twin on this hardware (the table above measures "
              "this directly). The sparsity-based figure is a *projection* for event-driven neuromorphic "
              "processors and is reported only as such."]
    (out / "snn_energy_measured.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
