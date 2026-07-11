"""
Data-grounded characterisation of the 8 learned neuro-symbolic rules.

For a trained checkpoint, run the model over real epochs and, for each rule,
measure what it ACTUALLY responds to:
  • mean activation on HIGH vs LOW engagement epochs  (-> which class it serves)
  • correlation of its activation with interpretable ET scalars
        pupil_mean       (arousal / interest)
        gaze_dispersion  (scattered gaze -> disengagement proxy)
  • how often it is the top-firing rule (dominance)

This produces honest, defensible rule descriptions. It does NOT invent
neurophysiological names for latent dimensions.

Usage:
  CUDA_VISIBLE_DEVICES="" python analysis/ground_neurosymbolic_rules.py \
      <checkpoint.pt> <subjects e.g. S24,S01> [out.json]
"""
import sys, json, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch as _t
_t.set_num_threads(4)

from data.dataset import NeumaGraphDataset, build_dataloaders, _normalize_subject_id
from models.ins_hdgs_cmt import AblationConfig
from training.train_rdganet import _make_model

CKPT     = sys.argv[1]
SUBJECTS = [_normalize_subject_id(s.strip()) for s in sys.argv[2].split(",")]
OUT      = sys.argv[3] if len(sys.argv) > 3 else "neurosymbolic_rules_grounded.json"
DEV      = "cpu"

# ── data ──────────────────────────────────────────────────────────────────────
ds = NeumaGraphDataset(subject_ids=SUBJECTS, precompute_graphs=True)
_, loader = build_dataloaders(ds, ds, batch_size=min(32, len(ds)))

# ── model ─────────────────────────────────────────────────────────────────────
model = _make_model(ds.n_eeg_ch, ds.n_classes, AblationConfig.full(), ds.n_et_ch)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
model.load_state_dict(sd, strict=False)
model.eval().to(DEV)

acts, labels, pupil, disp = [], [], [], []
with torch.no_grad():
    for b in loader:
        out = model(
            eeg_windows  = b["eeg_windows"].to(DEV).float(),
            adj_matrices = b["adj_matrices"].to(DEV).float(),
            et_seq       = b["et_seq"].to(DEV).float(),
            roi_vector   = b["roi_vector"].to(DEV).float(),
            weighted_adjs= b.get("weighted_adjs").to(DEV).float() if "weighted_adjs" in b else None,
        )
        ra = out["rule_act"]
        if ra is None:
            print("No rule activations (use_neuro_symbolic off?)"); sys.exit(1)
        acts.append(ra.cpu().numpy())
        labels.append(b["label"].cpu().numpy())
        et = b["et_seq"].cpu().numpy()                       # (B, T, 3): x,y,pupil
        pupil.append(np.nanmean(et[:, :, 2], axis=1))
        disp.append(np.nanstd(et[:, :, 0], axis=1) + np.nanstd(et[:, :, 1], axis=1))

A   = np.concatenate(acts)          # (N, 8)
y   = np.concatenate(labels)        # (N,)
pup = np.concatenate(pupil)
dsp = np.concatenate(disp)
N, R = A.shape
hi, lo = (y == 1), (y == 0)

def corr(a, b):
    if a.std() < 1e-9 or np.nanstd(b) < 1e-9: return 0.0
    m = np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])

top = A.argmax(axis=1)              # which rule fires most per epoch
rules = []
for r in range(R):
    a_hi, a_lo = A[hi, r].mean(), A[lo, r].mean()
    serves = "HIGH" if a_hi > a_lo else "LOW"
    rules.append({
        "rule": r + 1,
        "mean_act_HIGH": round(float(a_hi), 4),
        "mean_act_LOW":  round(float(a_lo), 4),
        "responds_to":   serves,
        "selectivity":   round(float(a_hi - a_lo), 4),
        "corr_pupil":    round(corr(A[:, r], pup), 3),
        "corr_gaze_dispersion": round(corr(A[:, r], dsp), 3),
        "dominant_in_pct_epochs": round(float((top == r).mean() * 100), 1),
    })

report = {
    "checkpoint": CKPT, "subjects": SUBJECTS, "n_epochs": int(N),
    "class_counts": {"HIGH": int(hi.sum()), "LOW": int(lo.sum())},
    "rules": rules,
}
with open(OUT, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{N} epochs ({int(hi.sum())} HIGH / {int(lo.sum())} LOW) from {SUBJECTS}")
print(f"{'Rule':>4} | {'act_HIGH':>8} {'act_LOW':>8} | {'serves':>6} | {'r(pupil)':>8} {'r(disp)':>8} | {'dom%':>5}")
print("-" * 70)
for r in rules:
    print(f"{r['rule']:>4} | {r['mean_act_HIGH']:>8.4f} {r['mean_act_LOW']:>8.4f} | "
          f"{r['responds_to']:>6} | {r['corr_pupil']:>8.3f} {r['corr_gaze_dispersion']:>8.3f} | "
          f"{r['dominant_in_pct_epochs']:>5.1f}")
print(f"\nSaved -> {OUT}")
