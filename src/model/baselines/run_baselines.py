"""
================================================================
INS-HDGS-CMT — DL BASELINE LOSOCV RUNNER  (with per-architecture tuning)
================================================================
Benchmarks the sixteen deep-learning baselines against INS-HDGS-CMT
under the *identical* protocol:

  • same NeumaGraphDataset   → identical epochs, normalisation, labels
  • same LOSOCV folds        → leave-one-subject-out; one held-out
                               validation subject per fold (same draw rule)
  • same metrics             → training.metrics.compute_metrics
  • same leakage-free post-hoc calibration (temperature + threshold on
                               the validation subject; never the test subject)

Two training regimes
--------------------
  fixed   (paper v1)  : one common budget for every architecture —
                        Adam, lr 1e-3, wd 1e-4, batch 32, 100 epochs,
                        no early stopping.  `--epochs 100`
  tuned   (revision)  : Reviewer 2, comment 3.  For every fold and every
                        architecture, `--tune N` configurations are drawn
                        from the architecture's search space (learning
                        rate, weight decay, dropout, batch size), each is
                        trained with early stopping on the validation
                        subject's balanced accuracy (patience
                        `--patience`, max `--epochs`), and the
                        configuration with the best validation balanced
                        accuracy is the one applied to the test subject.
                        This is the same nested, validation-subject-based
                        selection and early-stopping rule used for the
                        proposed model, so the comparison is matched.
                        The selected hyper-parameters are written per fold.

Usage
-----
  cd src/model
  python baselines/run_baselines.py --models eegnet,shallow --epochs 100          # fixed budget
  python baselines/run_baselines.py --models all --tune 12 --epochs 250 --patience 30 --device cuda
  python baselines/run_baselines.py --models eegnet --tune 4 --max-folds 2         # smoke test

Outputs (results/baselines/dl[/tuned]/)
  losocv_<model>.csv, fold_probs/probs_<model>.csv, hparams_<model>.csv,
  summary_<tag>.csv
================================================================
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── path setup: <repo>/src/model/{config,data,training,baselines,...}
HERE = Path(__file__).resolve()
MODEL = HERE.parents[1]            # .../src/model
SRC = MODEL.parent                 # .../src     (for `model.inference...` imports)
ROOT = SRC.parent                  # repo root
for p in (str(MODEL), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.dataset import NeumaGraphDataset            # noqa: E402
from training.metrics import compute_metrics          # noqa: E402
from baselines.baseline_models import MODEL_REGISTRY  # noqa: E402
from evaluation.losocv import (                       # noqa: E402
    _find_optimal_threshold, _calibrate_temperature_posthoc,
)
from evaluation.val_split import pick_validation_subject   # noqa: E402
from evaluation import foldwise_label as _fwl               # noqa: E402

# ── per-architecture search spaces (random search; Reviewer 2, comment 3) ──────
# Every architecture searches learning rate, weight decay and batch size;
# dropout is searched where the constructor exposes `drop`.
DEFAULT_SPACE = dict(
    lr=[3e-4, 5e-4, 1e-3, 2e-3, 3e-3],
    wd=[0.0, 1e-5, 1e-4, 1e-3],
    batch_size=[16, 32],
    drop=[0.25, 0.4, 0.5],
)
SPACE_OVERRIDES = {
    # transformers are lr-sensitive: shift the grid down and widen dropout
    "eeg_transformer": dict(lr=[1e-4, 3e-4, 5e-4, 1e-3], drop=[0.1, 0.2, 0.3, 0.5]),
    "et_transformer":  dict(lr=[1e-4, 3e-4, 5e-4, 1e-3], drop=[0.1, 0.2, 0.3, 0.5]),
    "dual_transformer": dict(lr=[1e-4, 3e-4, 5e-4, 1e-3], drop=[0.1, 0.2, 0.3, 0.5]),
    "cross_attention": dict(lr=[1e-4, 3e-4, 5e-4, 1e-3], drop=[0.1, 0.2, 0.3, 0.5]),
    "mm_transformer":  dict(lr=[1e-4, 3e-4, 5e-4, 1e-3], drop=[0.1, 0.2, 0.3, 0.5]),
    "dynamicgat_et":   dict(lr=[1e-4, 3e-4, 5e-4, 1e-3], drop=[0.1, 0.2, 0.3, 0.5]),
    "gat":             dict(drop=[0.1, 0.2, 0.3, 0.5]),
    "brain_gcn":       dict(drop=[0.25, 0.4, 0.5]),
}


def search_space(mname: str) -> dict:
    sp = dict(DEFAULT_SPACE)
    sp.update(SPACE_OVERRIDES.get(mname, {}))
    return sp


def sample_configs(mname: str, n: int, seed: int) -> list[dict]:
    """First config = the paper's fixed budget (so tuning can only help);
    the rest are random draws from the architecture's search space."""
    rng = np.random.RandomState(seed)
    sp = search_space(mname)
    cfgs = [dict(lr=1e-3, wd=1e-4, batch_size=32, drop=None)]
    seen = {json.dumps(cfgs[0], sort_keys=True)}
    while len(cfgs) < n:
        c = {k: (float(rng.choice(v)) if k != "batch_size" else int(rng.choice(v))) for k, v in sp.items()}
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key); cfgs.append(c)
    return cfgs


# ── feature engineering for the multimodal MLP baseline ─────────────────────────

_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]  # delta..gamma


def _bandpower_feats(eeg, fs):
    T, C = eeg.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(eeg, axis=0)) ** 2) / T
    feats = []
    for lo, hi in _BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = psd[m].sum(axis=0) if m.any() else np.zeros(C)
        feats.append(np.log(bp + 1e-8))
    return np.concatenate(feats)


def _et_feats(et):
    et = np.nan_to_num(np.asarray(et, dtype=np.float32))
    return np.concatenate([et.mean(0), et.std(0)])


def build_arrays(ds, fs_eeg):
    X_eeg = np.stack([np.asarray(e, np.float32).T for e in ds.raw_eeg])[:, None, :, :]   # (N,1,C,T)
    y = np.asarray(ds.labels, np.int64)
    subj = np.asarray(ds.subject_ids, np.int64)
    feats = np.stack([np.concatenate([_bandpower_feats(np.asarray(e, np.float32), fs_eeg), _et_feats(et)])
                      for e, et in zip(ds.raw_eeg, ds.raw_et)]).astype(np.float32)
    return X_eeg, feats, y, subj


def build_et_arrays(ds):
    return np.stack([np.nan_to_num(np.asarray(et, np.float32)) for et in ds.raw_et])


def _node_feats(eeg, fs):
    T, C = eeg.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(eeg, axis=0)) ** 2) / T
    out = np.zeros((C, len(_BANDS)), dtype=np.float32)
    for j, (lo, hi) in enumerate(_BANDS):
        m = (freqs >= lo) & (freqs < hi)
        out[:, j] = (np.log(psd[m].sum(0) + 1e-8) if m.any() else np.zeros(C))
    return out


def _adjacency(eeg):
    A = np.corrcoef(np.asarray(eeg, np.float32), rowvar=False)
    return np.abs(np.nan_to_num(A, nan=0.0)).astype(np.float32)


def build_graph_arrays(ds, fs_eeg):
    Xnode = np.stack([_node_feats(np.asarray(e, np.float32), fs_eeg) for e in ds.raw_eeg]).astype(np.float32)
    Adj = np.stack([_adjacency(e) for e in ds.raw_eeg]).astype(np.float32)
    return Xnode, Adj


# ── model construction ──────────────────────────────────────────────────────────

def make_model(spec, kind, shapes, n_classes, drop):
    cls = spec["cls"]
    kw = {}
    if kind == "eeg":
        kw = dict(n_chans=shapes["C"], n_times=shapes["T"], n_classes=n_classes)
    elif kind == "feat":
        kw = dict(in_dim=shapes["F"], n_classes=n_classes)
    elif kind == "graph":
        kw = dict(n_nodes=shapes["C"], node_dim=shapes["node_dim"], n_classes=n_classes)
    elif kind == "et":
        kw = dict(in_dim=shapes["et_dim"], n_classes=n_classes)
    elif kind == "multimodal":
        kw = dict(n_chans=shapes["C"], n_times=shapes["T"], et_dim=shapes["et_dim"], n_classes=n_classes)
    if drop is not None and "drop" in inspect.signature(cls.__init__).parameters:
        kw["drop"] = drop
    return cls(**kw)


def _as_list(X):
    return list(X) if isinstance(X, (tuple, list)) else [X]


@torch.no_grad()
def _logits(model, X_list, device):
    model.eval()
    return model(*[torch.as_tensor(x, dtype=torch.float32, device=device) for x in X_list]).cpu().numpy()


def _bal_acc(y, logits):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y, logits.argmax(1))


def train_fold(model, Xtr, ytr, *, device, epochs, lr, wd, batch_size, n_classes, seed,
               Xval=None, yval=None, patience=None):
    """Train one model. With patience and a validation set, early-stop on the
    validation subject's balanced accuracy and restore the best weights;
    otherwise train for exactly `epochs` (fixed budget). Returns
    (model, best_epoch)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    Xtr_list = [torch.as_tensor(x, dtype=torch.float32) for x in _as_list(Xtr)]
    ytr_t = torch.as_tensor(ytr, dtype=torch.long)
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.clip(counts, 1, None))
    crit = nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n = len(ytr_t); bs = min(batch_size, n)
    best_state, best_val, best_ep, no_imp = None, -1.0, epochs, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2 and n >= 2:
                # a trailing single-sample batch makes BatchNorm1d raise in train
                # mode ("Expected more than 1 value per channel"): this is what
                # killed the phase3d fusion_mlp baseline (2026-09-08).  Skip it;
                # the sample is a different one every epoch (perm is reshuffled).
                continue
            xb = [x[idx].to(device) for x in Xtr_list]
            opt.zero_grad()
            loss = crit(model(*xb), ytr_t[idx].to(device))
            loss.backward(); opt.step()
        if patience is not None and Xval is not None:
            vb = _bal_acc(yval, _logits(model, _as_list(Xval), device))
            if vb > best_val + 1e-6:
                best_val, best_ep, no_imp = vb, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 1
                if ep >= 20 and no_imp >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep


# ── main LOSOCV loop ────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device)
    out_dir = ROOT / "results" / "baselines" / ("dl_tuned" if args.tune else "dl")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = out_dir / "checkpoints"; ckpt_root.mkdir(exist_ok=True)
    prob_root = out_dir / "fold_probs"; prob_root.mkdir(exist_ok=True)

    print("[baseline] building dataset (graph-free) …", flush=True)
    ds = NeumaGraphDataset(precompute_graphs=False, augment=False)
    from config.settings import EEG_SR
    X_eeg, feats, y, subj = build_arrays(ds, EEG_SR)
    X_et = build_et_arrays(ds)
    Xnode, Adj = build_graph_arrays(ds, EEG_SR)
    n_classes = int(ds.n_classes)
    sid_names = ds.unique_subjects
    subjects = list(range(len(sid_names)))
    if args.max_folds:
        subjects = subjects[:args.max_folds]
    shapes = dict(C=X_eeg.shape[2], T=X_eeg.shape[3], F=feats.shape[1], node_dim=Xnode.shape[2], et_dim=X_et.shape[2])
    print(f"[baseline] N={len(y)} epochs  C={shapes['C']}  T={shapes['T']}  subjects={len(sid_names)}  "
          f"regime={'tuned(n=%d)' % args.tune if args.tune else 'fixed'}", flush=True)

    # ── validation-subject rule and label construction (opt-in; defaults unchanged) ──
    # NEUMA_VAL_RULE=matched: evaluation/val_split.py, deterministic from (seed, test
    # subject) and restricted to subjects with both classes, so every baseline holds out
    # the same validation subject as the proposed model in the same fold (same --seed).
    # NEUMA_LABEL_FOLDWISE=1: the label of every epoch is re-derived per fold with the
    # scaler and median fitted on that fold's training subjects only (same helper as
    # evaluation/losocv.py); the stored labels are used only for the single-class skip.
    val_rule = os.environ.get("NEUMA_VAL_RULE", "legacy").strip().lower() or "legacy"
    if val_rule not in ("legacy", "matched"):
        raise ValueError(f"NEUMA_VAL_RULE={val_rule!r}; expected legacy|matched")
    eligible = {sid_names[u] for u in range(len(sid_names)) if len(np.unique(y[subj == u])) >= 2}
    foldwise = _fwl.enabled()
    label_table = _fwl.load_label_table() if foldwise else None
    print(f"[baseline] validation-subject rule: {val_rule}"
          + (f" (seed {args.seed}; single-class subjects excluded: "
             f"{sorted(set(sid_names) - eligible)})" if val_rule == "matched" else ""), flush=True)
    print(f"[baseline] label: {'FOLD-WISE (scaler + median fitted per fold on the training subjects)' if foldwise else 'stored'}",
          flush=True)

    models = (list(MODEL_REGISTRY) if args.models == "all"
              else [m.strip() for m in args.models.split(",") if m.strip()])
    summary = []
    for mname in models:
        if mname not in MODEL_REGISTRY:
            print(f"[baseline] !! unknown model '{mname}' — skipping"); continue
        spec = MODEL_REGISTRY[mname]; kind = spec["kind"]
        rows, prob_rows, hp_rows = [], [], []
        t0 = time.time()
        print(f"\n========== {mname} ==========", flush=True)
        for fold_idx, s in enumerate(subjects):
            te = subj == s
            if len(np.unique(y[te])) < 2:      # single-class test subject: skipped, as for the full model
                continue
            cand = [u for u in range(len(sid_names)) if u != s]
            if val_rule == "matched":
                val_s = sid_names.index(pick_validation_subject(
                    sid_names[s], [sid_names[u] for u in cand], eligible, args.seed))
            else:
                _rng = np.random.RandomState(args.seed + fold_idx); _rng.shuffle(cand)
                val_s = cand[0]
            va = subj == val_s
            tr = (~te) & (subj != val_s)
            if te.sum() == 0 or tr.sum() == 0 or va.sum() == 0:
                continue

            # labels used in this fold: stored, or re-derived from the training subjects only
            y_f = y
            if foldwise:
                y_f = _fwl.fold_vector(ds, [sid_names[u] for u in cand if u != val_s], label_table)
                print(f"  fold {fold_idx + 1:02d} {sid_names[s]:>4s}: fold-wise label changed "
                      f"{int((y_f != y).sum())}/{len(y)} epochs "
                      f"(train {int((y_f[tr] != y[tr]).sum())}, val {int((y_f[va] != y[va]).sum())}, "
                      f"test {int((y_f[te] != y[te]).sum())})", flush=True)
                if len(np.unique(y_f[te])) < 2:
                    print(f"  fold {fold_idx + 1:02d} {sid_names[s]:>4s}: single-class under the fold-wise label - skipped",
                          flush=True)
                    continue

            # ── modality-specific tensors (train statistics only) ────────────
            if kind == "eeg":
                Xtr, Xval, Xte = X_eeg[tr], X_eeg[va], X_eeg[te]
            elif kind == "feat":
                mu = feats[tr].mean(0, keepdims=True); sd = feats[tr].std(0, keepdims=True) + 1e-8
                Xtr, Xval, Xte = (feats[tr] - mu) / sd, (feats[va] - mu) / sd, (feats[te] - mu) / sd
            elif kind == "graph":
                mu = Xnode[tr].mean((0, 1), keepdims=True); sd = Xnode[tr].std((0, 1), keepdims=True) + 1e-8
                pack = lambda m: np.concatenate([(Xnode[m] - mu) / sd, Adj[m]], axis=-1)
                Xtr, Xval, Xte = pack(tr), pack(va), pack(te)
            elif kind == "et":
                mu = X_et[tr].reshape(-1, X_et.shape[2]).mean(0); sd = X_et[tr].reshape(-1, X_et.shape[2]).std(0) + 1e-8
                Xtr, Xval, Xte = (X_et[tr] - mu) / sd, (X_et[va] - mu) / sd, (X_et[te] - mu) / sd
            elif kind == "multimodal":
                mu = X_et[tr].reshape(-1, X_et.shape[2]).mean(0); sd = X_et[tr].reshape(-1, X_et.shape[2]).std(0) + 1e-8
                Xtr = (X_eeg[tr], (X_et[tr] - mu) / sd); Xval = (X_eeg[va], (X_et[va] - mu) / sd)
                Xte = (X_eeg[te], (X_et[te] - mu) / sd)
            else:
                raise ValueError(kind)

            # ── nested selection on the validation subject ───────────────────
            cfgs = sample_configs(mname, args.tune, args.seed + fold_idx) if args.tune else \
                [dict(lr=args.lr, wd=args.wd, batch_size=args.batch_size, drop=None)]
            best = None
            for ci, cfg in enumerate(cfgs):
                model = make_model(spec, kind, shapes, n_classes, cfg["drop"])
                model, best_ep = train_fold(
                    model, Xtr, y_f[tr], device=device, epochs=args.epochs, lr=cfg["lr"], wd=cfg["wd"],
                    batch_size=cfg["batch_size"], n_classes=n_classes, seed=args.seed,
                    Xval=Xval, yval=y_f[va], patience=args.patience if (args.tune or args.early_stop) else None)
                val_logits = _logits(model, _as_list(Xval), device)
                vb = _bal_acc(y_f[va], val_logits)
                if best is None or vb > best["val_bal"] + 1e-9:
                    best = dict(model=model, cfg=cfg, cfg_idx=ci, val_bal=vb, best_ep=best_ep, val_logits=val_logits)
            model, cfg = best["model"], best["cfg"]
            test_logits = _logits(model, _as_list(Xte), device)
            prob = torch.softmax(torch.as_tensor(test_logits), 1).numpy(); pred = prob.argmax(1)

            m = compute_metrics(y_f[te], pred, prob, n_classes=n_classes)
            m.update(fold=fold_idx + 1, test_subject=sid_names[s], train_n=int(tr.sum()), test_n=int(te.sum()))
            from scipy.special import softmax as _sp_softmax
            T_post, val_prob_cal = _calibrate_temperature_posthoc(best["val_logits"], y_f[va])
            tst_prob_cal = _sp_softmax(test_logits / T_post, axis=1)[:, 1]
            thr = _find_optimal_threshold(val_prob_cal, y_f[va])
            pred_cal = (tst_prob_cal >= thr).astype(int)
            m_cal = compute_metrics(y_f[te], pred_cal, np.stack([1 - tst_prob_cal, tst_prob_cal], 1), n_classes=n_classes)
            m.update({f"{k}_cal": v for k, v in m_cal.items()})
            m.update(T_post=round(float(T_post), 4), opt_threshold_cal=round(float(thr), 4),
                     val_subject=sid_names[val_s], val_balanced_acc=round(float(best["val_bal"]), 4),
                     lr=cfg["lr"], wd=cfg["wd"], batch_size=cfg["batch_size"], drop=cfg["drop"],
                     best_epoch=best["best_ep"], n_configs=len(cfgs))
            rows.append(m)
            hp_rows.append(dict(fold=fold_idx + 1, test_subject=sid_names[s], val_subject=sid_names[val_s],
                                cfg_idx=best["cfg_idx"], **cfg, best_epoch=best["best_ep"], val_balanced_acc=best["val_bal"]))
            fno = fold_idx + 1
            torch.save({"model_state_dict": model.state_dict(), "model": mname, "kind": kind, "fold": fno,
                        "test_subject": sid_names[s], "cfg": cfg, "metrics": m}, ckpt_root / f"{mname}_fold{fno:02d}.pt")
            for yt, yp, pp in zip(y_f[te], pred, prob[:, 1]):
                prob_rows.append(dict(fold=fno, test_subject=sid_names[s], y_true=int(yt), y_pred=int(yp), p1=float(pp)))
            print(f"  fold {fno:02d} {sid_names[s]:>4s} bal={m['balanced_acc']:.3f} auc={m['roc_auc']:.3f} "
                  f"mcc={m['mcc']:+.3f} | val_bal={best['val_bal']:.2f} cfg={cfg} ep={best['best_ep']}", flush=True)

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"losocv_{mname}.csv", index=False)
        pd.DataFrame(prob_rows).to_csv(prob_root / f"probs_{mname}.csv", index=False)
        pd.DataFrame(hp_rows).to_csv(out_dir / f"hparams_{mname}.csv", index=False)
        dt = time.time() - t0
        means = {k: float(np.nanmean(df[k])) for k in ["balanced_acc", "f1", "roc_auc", "mcc", "accuracy", "ece",
                                                        "balanced_acc_cal", "mcc_cal", "accuracy_cal"] if k in df}
        means.update(model=mname, n_folds=len(df), seconds=round(dt, 1), regime="tuned" if args.tune else "fixed")
        summary.append(means)
        print(f"[baseline] {mname}: bal={means['balanced_acc']:.3f} auc={means['roc_auc']:.3f} mcc={means['mcc']:.3f} "
              f"({len(df)} folds, {dt:.0f}s)", flush=True)

    if summary:
        tag = "_".join(models) if len(models) <= 3 else "multi"
        pd.DataFrame(summary).to_csv(out_dir / f"summary_{tag}.csv", index=False)
        print(pd.DataFrame(summary).to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description="DL baseline LOSOCV runner (fixed budget or tuned)")
    p.add_argument("--models", default="eegnet", help="comma list or 'all': " + ", ".join(MODEL_REGISTRY))
    p.add_argument("--epochs", type=int, default=100, help="fixed budget, or max epochs when tuning/early stopping")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tune", type=int, default=0, help="number of hyper-parameter configurations per fold (0 = fixed budget)")
    p.add_argument("--early-stop", action="store_true", help="early stopping on the validation subject without a search")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
