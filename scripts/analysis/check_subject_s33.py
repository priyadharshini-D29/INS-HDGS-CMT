"""
Diagnostic script for S33 — subject with accuracy=0.0 in LOSOCV.
Checks: label distribution, NaN/zero epochs, class separability.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import NeumaGraphDataset

print("\nLoading S33 ...\n")

try:
    ds = NeumaGraphDataset(subject_ids=["S33"], precompute_graphs=False)
except Exception as e:
    print(f"[FATAL] Could not load S33: {e}")
    sys.exit(1)

print("=" * 60)
print(" DATASET INFO")
print("=" * 60)
print(f"\nTotal epochs  : {len(ds)}")

labels = ds.labels
print(f"Unique labels : {np.unique(labels)}")
print("\nClass counts:")
for u in np.unique(labels):
    print(f"  Class {u}: {(labels == u).sum()}")

if len(np.unique(labels)) < 2:
    print("\n[WARNING] Only ONE class present — label generation failed for S33!")

# ── EEG quality checks ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" EEG QUALITY CHECKS")
print("=" * 60)

nan_epochs  = 0
zero_epochs = 0
flat_ch_counts = []

for i in range(len(ds)):
    eeg = np.array(ds.raw_eeg[i], dtype=np.float32)  # (T, C)
    if np.isnan(eeg).any():
        nan_epochs += 1
    if np.all(eeg == 0):
        zero_epochs += 1
    flat = (eeg.std(axis=0) < 1e-6).sum()
    flat_ch_counts.append(flat)

print(f"\nNaN epochs    : {nan_epochs}")
print(f"All-zero epoch: {zero_epochs}")
print(f"Flat channels : min={min(flat_ch_counts)}  "
      f"max={max(flat_ch_counts)}  "
      f"mean={np.mean(flat_ch_counts):.1f}")

# ── Label distribution per epoch ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(" LABEL SEQUENCE")
print("=" * 60)
print(f"\n{labels.tolist()}")

# ── EEG amplitude stats per class ────────────────────────────────────────────
print("\n" + "=" * 60)
print(" EEG AMPLITUDE BY CLASS")
print("=" * 60)
for cls in np.unique(labels):
    idx = np.where(labels == cls)[0]
    vals = np.concatenate([ds.raw_eeg[i].flatten() for i in idx])
    print(f"\n  Class {cls}:  mean={vals.mean():.4f}  "
          f"std={vals.std():.4f}  "
          f"min={vals.min():.4f}  "
          f"max={vals.max():.4f}")

print("\nDone.\n")
