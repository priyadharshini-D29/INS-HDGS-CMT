"""
Collect per-fold per-sample probabilities and raw logits from v17 checkpoints.

Runs GPU inference only (no training).  Saves one .npz per fold under
  output/fold_probs/fold{fold_no:02d}_{test_subj}.npz

Each file contains:
  y_true          : (N_test,)   int   ground-truth labels
  y_prob          : (N_test,)   float averaged ensemble P(HIGH) [per-member T applied]
  avg_logits      : (N_test, 2) float averaged raw (pre-T) logits
  val_y_true      : (N_val,)    int
  val_y_prob      : (N_val,)    float
  val_avg_logits  : (N_val, 2)  float
  T_per_member    : (N_ENS,)    float per-member calibration temperatures
"""

import os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import torch
import multiprocessing

# ── Config ────────────────────────────────────────────────────────────────────
LABEL      = "ins_hdgs_cmt_v17"
OUT_DIR    = Path("output/fold_probs")
CKPT_DIR   = Path("output/checkpoints") / LABEL
N_GPUS     = torch.cuda.device_count()

from config.settings import (
    SUBJECT_IDS, LR, WEIGHT_DECAY, PATIENCE, N_ENSEMBLE, RANDOM_SEED,
    BATCH_SIZE, IMBALANCE_SKIP_RATIO,
)


def _collect_fold(args):
    gpu_id, fold_no, test_subj, val_subj, train_subs, out_dir, label = args

    import os, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch, numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from config.settings import LR, WEIGHT_DECAY, PATIENCE, N_ENSEMBLE, RANDOM_SEED, BATCH_SIZE
    from data.dataset    import NeumaGraphDataset, build_dataloaders
    from training.trainer import Trainer
    from utils.gpu import _build_raw_model

    # ── Load datasets ─────────────────────────────────────────────────────────
    try:
        train_ds = NeumaGraphDataset(subject_ids=train_subs, precompute_graphs=True, augment=False)
        val_ds   = NeumaGraphDataset(subject_ids=[val_subj],  precompute_graphs=True)
        test_ds  = NeumaGraphDataset(subject_ids=[test_subj], precompute_graphs=True)
    except Exception as e:
        print(f"  [SKIP] Fold {fold_no} ({test_subj}): {e}", flush=True)
        return None

    if len(test_ds) < 2 or len(np.unique(test_ds.labels)) < 2:
        print(f"  [SKIP] Fold {fold_no} ({test_subj}): single-class or too small", flush=True)
        return None

    eff_bs = min(BATCH_SIZE, len(train_ds))
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size=eff_bs)
    _,            test_loader= build_dataloaders(train_ds, test_ds, batch_size=eff_bs)

    ckpt_dir_p = Path(out_dir).parent / "checkpoints" / label
    ckpt_paths = [ckpt_dir_p / f"{label}_fold{fold_no:02d}_e{ei}.pt"
                  for ei in range(N_ENSEMBLE)]
    if not all(p.exists() for p in ckpt_paths):
        print(f"  [SKIP] Fold {fold_no} ({test_subj}): checkpoints missing", flush=True)
        return None

    ensemble_tst_probs  = []
    ensemble_tst_logits = []
    ensemble_val_probs  = []
    ensemble_val_logits = []
    T_per_member        = []
    y_true_final        = None
    val_y_true_final    = None

    for ei, cp in enumerate(ckpt_paths):
        raw = _build_raw_model(train_ds, None).to(device)
        sd  = torch.load(cp, map_location=device, weights_only=False)
        raw.load_state_dict(
            sd["model_state_dict"] if isinstance(sd, dict) and "model_state_dict" in sd else sd
        )
        trainer = Trainer(model=raw, device=device, loss_fn=None,
                          lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE,
                          ckpt_path=cp, use_dp=False)
        trainer.calibrate_temperature(val_loader)
        T_per_member.append(trainer.temperature)
        raw.eval()

        # Test set
        probs_i = []; logits_i = []; ytrue_i = []
        with torch.no_grad():
            for batch in test_loader:
                bd = {k: v.to(device) for k, v in batch.items()}
                out = raw(eeg_windows=bd["eeg_windows"], adj_matrices=bd["adj_matrices"],
                          et_seq=bd["et_seq"], roi_vector=bd["roi_vector"],
                          weighted_adjs=bd["weighted_adjs"])
                p   = torch.softmax(out["logits"] / trainer.temperature, dim=1)[:, 1]
                probs_i.extend(p.cpu().numpy().tolist())
                logits_i.extend(out["logits"].cpu().numpy().tolist())
                ytrue_i.extend(bd["label"].cpu().numpy().tolist())
        ensemble_tst_probs.append(np.array(probs_i))
        ensemble_tst_logits.append(np.array(logits_i))
        if y_true_final is None:
            y_true_final = np.array(ytrue_i)

        # Val set
        vprobs_i = []; vlogits_i = []; vyt_i = []
        with torch.no_grad():
            for batch in val_loader:
                bd = {k: v.to(device) for k, v in batch.items()}
                out = raw(eeg_windows=bd["eeg_windows"], adj_matrices=bd["adj_matrices"],
                          et_seq=bd["et_seq"], roi_vector=bd["roi_vector"],
                          weighted_adjs=bd["weighted_adjs"])
                vp   = torch.softmax(out["logits"] / trainer.temperature, dim=1)[:, 1]
                vprobs_i.extend(vp.cpu().numpy().tolist())
                vlogits_i.extend(out["logits"].cpu().numpy().tolist())
                vyt_i.extend(bd["label"].cpu().numpy().tolist())
        ensemble_val_probs.append(np.array(vprobs_i))
        ensemble_val_logits.append(np.array(vlogits_i))
        if val_y_true_final is None:
            val_y_true_final = np.array(vyt_i)

    save_path = Path(out_dir) / f"fold{fold_no:02d}_{test_subj}.npz"
    np.savez(save_path,
             y_true         = y_true_final,
             y_prob         = np.stack(ensemble_tst_probs).mean(axis=0),
             avg_logits     = np.stack(ensemble_tst_logits).mean(axis=0),
             val_y_true     = val_y_true_final,
             val_y_prob     = np.stack(ensemble_val_probs).mean(axis=0),
             val_avg_logits = np.stack(ensemble_val_logits).mean(axis=0),
             T_per_member   = np.array(T_per_member),
             fold_no        = fold_no,
             test_subj      = np.array(test_subj),
    )
    print(f"  [GPU{gpu_id}] Fold {fold_no:02d} ({test_subj}): saved  "
          f"test_n={len(y_true_final)}  val_n={len(val_y_true_final)}", flush=True)
    return fold_no


def _gpu_worker(gpu_fold_list):
    """Module-level so spawn can pickle it."""
    for args in gpu_fold_list:
        _collect_fold(args)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build fold structure (same as LOSOCV — same RNG seed)
    rng = np.random.default_rng(RANDOM_SEED)
    available = list(SUBJECT_IDS)
    fold_args = []
    for fold_idx, test_subj in enumerate(available):
        fold_no = fold_idx + 1
        candidate = [s for s in available if s != test_subj]
        rng.shuffle(candidate)
        if len(candidate) < 2:
            continue
        val_subj   = candidate[0]
        train_subs = candidate[1:]
        fold_args.append((
            fold_idx % N_GPUS,
            fold_no, test_subj, val_subj, train_subs,
            str(OUT_DIR), LABEL,
        ))

    print(f"Collecting {len(fold_args)} folds across {N_GPUS} GPUs …")

    from collections import defaultdict
    by_gpu = defaultdict(list)
    for args in fold_args:
        by_gpu[args[0]].append(args)

    ctx = multiprocessing.get_context("spawn")
    procs = []
    for gpu_id, gpu_args in by_gpu.items():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        p = ctx.Process(target=_gpu_worker, args=(gpu_args,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    saved = list(OUT_DIR.glob("fold*.npz"))
    print(f"\nDone. Saved {len(saved)} fold files under {OUT_DIR}")
