"""
================================================================
INS-HDGS-CMT — Integrated-Gradients feature attribution
================================================================
Fills Table~\\ref{tab:features}: per-feature importance of the full model,
computed with Integrated Gradients (Sundararajan et al., 2017) over the model's
ACTUAL inputs, under LOSOCV (each fold's trained checkpoint attributes its own
held-out subject's epochs — leakage-free).

Inputs and the honest feature groups they support
--------------------------------------------------
  eeg_windows  (B,W,24,5)  per-window per-electrode band-power
       bands  = [delta, theta, alpha, beta, gamma]
       regions= frontal {Fz,F3,F4,FC1,FC2} / posterior {Pz,Oz,O1,O2,P3,P4,P7,P8}
       → "Frontal theta power", "Posterior alpha power"
  weighted_adjs(B,W,24,24) functional connectivity
       → "Frontal functional connectivity" (frontal-frontal edges)
  et_seq       (B,600,3)   raw gaze  [gaze-x, gaze-y, pupil]
       → "Gaze position", "Pupil dynamics"
  roi_vector   (B,W)       ROI saliency  → "ROI saliency"

We DO NOT attribute to engineered ET features (dwell time, fixation duration):
those are not model inputs (they define the labels), so attributing IG to them
would be unfounded.

Importance = mean |IG| per input element within each group, averaged over folds
(epoch-weighted), then normalised to sum to 1 across the reported groups.

Outputs (results/statistics/):
  feature_importance_ig.csv / .md   (drop-in numbers for tab:features)

Usage
-----
  python analysis/integrated_gradients.py --device cuda --steps 16
  python analysis/integrated_gradients.py --max-folds 5    # quick check
================================================================
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve()
PHASE8 = HERE.parents[1]
ROOT = PHASE8.parent
for p in (str(PHASE8), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.dataset import NeumaGraphDataset                      # noqa: E402
from data.channel_harmonizer import CANONICAL_CHANNELS          # noqa: E402
from models.ins_hdgs_cmt import AblationConfig                  # noqa: E402
from evaluation.losocv import _make_model                       # noqa: E402
from config.settings import ET_INPUT_DIM                        # noqa: E402

CKPT_DIR = PHASE8 / "output" / "checkpoints" / "repro_focal_g3p0_effective_num_37"
FULL_CSV = PHASE8 / "results" / "losocv_metrics" / "losocv_repro_focal_g3p0_effective_num_37.csv"
OUT = PHASE8 / "results" / "statistics"

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
FRONTAL = ["Fz", "F3", "F4", "F7", "F8"]
POSTERIOR = ["Pz", "O1", "O2", "P3", "P4", "T5", "T6"]
F_IDX = [CANONICAL_CHANNELS.index(c) for c in FRONTAL]
P_IDX = [CANONICAL_CHANNELS.index(c) for c in POSTERIOR]
THETA, ALPHA = BANDS.index("theta"), BANDS.index("alpha")

# attributed input tensors (those that admit a clean zero baseline)
ATTR_KEYS = ["eeg_windows", "et_seq", "roi_vector", "weighted_adjs"]


def _fold_subject_map():
    df = pd.read_csv(FULL_CSV)
    return {int(r["fold"]): str(r["test_subject"]) for _, r in df.iterrows()}


def _batch_for_subject(subj):
    ds = NeumaGraphDataset(subject_ids=[subj], precompute_graphs=True, augment=False)
    if len(ds) == 0:
        return None
    return next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False)))


def _integrated_gradients(model, batch, device, steps):
    """Return dict input_key -> mean|IG| tensor (same shape as input, batch-summed)."""
    base_inputs = {k: batch[k].to(device).float() for k in ATTR_KEYS}
    # non-attributed args kept fixed at their true value
    fixed = {"adj_matrices": batch["adj_matrices"].to(device).float()}
    # MEAN baseline (per-feature average over the epochs) — a neutral, on-manifold
    # reference. A zero baseline is off-manifold (e.g. log-band-power 0) and makes
    # IG magnitudes scale with raw input size rather than importance.
    baselines = {k: v.mean(dim=0, keepdim=True).expand_as(v).contiguous()
                 for k, v in base_inputs.items()}

    accum = {k: torch.zeros_like(v) for k, v in base_inputs.items()}
    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    for a in alphas:
        xs = {k: (baselines[k] + a * (base_inputs[k] - baselines[k])).requires_grad_(True)
              for k in base_inputs}
        out = model(eeg_windows=xs["eeg_windows"], adj_matrices=fixed["adj_matrices"],
                    et_seq=xs["et_seq"], roi_vector=xs["roi_vector"],
                    weighted_adjs=xs["weighted_adjs"])
        logits = out["logits"] if isinstance(out, dict) else out
        target = logits[:, 1].sum()                       # HIGH_ENGAGEMENT logit
        grads = torch.autograd.grad(target, list(xs.values()),
                                    retain_graph=False, allow_unused=True)
        for k, g in zip(base_inputs, grads):
            if g is not None and torch.isfinite(g).all():
                accum[k] = accum[k] + g.detach()
    ig = {}
    for k in base_inputs:
        avg_grad = accum[k] / steps
        ig[k] = ((base_inputs[k] - baselines[k]) * avg_grad).abs().detach().cpu()
    return ig


def _group_density(ig):
    """mean |IG| per element for each honest feature group, from one batch."""
    eeg = ig["eeg_windows"]      # (B,W,24,5)
    adj = ig["weighted_adjs"]    # (B,W,24,24)
    et = ig["et_seq"]            # (B,600,3)
    roi = ig["roi_vector"]       # (B,W)
    ff = np.ix_(range(adj.shape[0]), range(adj.shape[1]), F_IDX, F_IDX)
    g = {
        "Frontal theta power":            eeg[:, :, F_IDX, THETA].mean().item(),
        "Posterior alpha power":          eeg[:, :, P_IDX, ALPHA].mean().item(),
        "Frontal functional connectivity": adj[ff].mean().item(),
        "Gaze position":                  et[:, :, :2].mean().item(),
        "Pupil dynamics":                 et[:, :, 2].mean().item() if et.shape[2] > 2 else float("nan"),
        "ROI saliency":                   roi.mean().item(),
    }
    return g, et.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    fmap = _fold_subject_map()
    ckpts = sorted(glob.glob(str(CKPT_DIR / "*_e0.pt")))
    if args.max_folds:
        ckpts = ckpts[:args.max_folds]
    print(f"[ig] {len(ckpts)} fold checkpoints, device={device}, steps={args.steps}")

    rows, weights = [], []
    for ck in ckpts:
        m = re.search(r"_fold(\d+)_e0", ck)
        fold = int(m.group(1))
        subj = fmap.get(fold)
        if subj is None:
            continue
        batch = _batch_for_subject(subj)
        if batch is None:
            continue
        model = _make_model(n_eeg_ch=len(CANONICAL_CHANNELS), n_et_ch=ET_INPUT_DIM, n_classes=2,
                            ablation=AblationConfig.full()).to(device).eval()
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd, strict=False)
        ig = _integrated_gradients(model, batch, device, args.steps)
        g, n = _group_density(ig)
        rows.append(g); weights.append(n)
        print(f"  fold{fold:02d} {subj}: n={n}  "
              f"Ftheta={g['Frontal theta power']:.2e} Palpha={g['Posterior alpha power']:.2e} "
              f"pupil={g['Pupil dynamics']:.2e}", flush=True)

    df = pd.DataFrame(rows)
    w = np.asarray(weights, float)
    dens = {c: float(np.average(df[c], weights=w)) for c in df.columns}
    total = sum(v for v in dens.values() if v == v)
    norm = {c: (v / total if v == v else float("nan")) for c, v in dens.items()}

    INTERP = {
        "Frontal theta power": ("EEG", "Attentional control / engagement"),
        "Posterior alpha power": ("EEG", "Visual attention allocation (alpha suppression)"),
        "Frontal functional connectivity": ("EEG", "Fronto-cortical network integration"),
        "Gaze position": ("ET", "Overt fixation location"),
        "Pupil dynamics": ("ET", "Arousal / cognitive load"),
        "ROI saliency": ("ROI", "Attended stimulus region"),
    }
    out_rows = []
    for c in df.columns:
        mod, interp = INTERP[c]
        out_rows.append(dict(modality=mod, feature=c,
                             importance=round(norm[c], 4),
                             mean_abs_ig=dens[c], interpretation=interp))
    res = pd.DataFrame(out_rows).sort_values("importance", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "feature_importance_ig.csv", index=False)

    md = ["# Integrated-Gradients feature importance (LOSOCV)\n",
          f"Attribution to HIGH_ENGAGEMENT logit, {len(rows)} folds, "
          f"{int(w.sum())} epochs, {args.steps}-step IG, mean (per-feature) baseline. "
          "Importance = normalised mean |IG| per input element.\n",
          "| Modality | Feature | Importance | Interpretation |",
          "|---|---|---|---|"]
    for _, r in res.iterrows():
        md.append(f"| {r['modality']} | {r['feature']} | {r['importance']:.3f} | {r['interpretation']} |")
    (OUT / "feature_importance_ig.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[ig] → {OUT/'feature_importance_ig.csv'}")


if __name__ == "__main__":
    main()
