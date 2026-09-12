"""
================================================================
Where does the EEG signal die?  Stage-wise linear probe of the
model's own EEG input path
================================================================
The gaze-free EEG branch fails a rest-vs-browsing positive control (AUC 0.54)
that a 6-feature linear probe on whole-head log band power passes (AUC 0.79).
Before any component of the model sees the EEG, the signal is transformed by:

    S0  loader:        per-subject, per-channel z-score            (dataset.py)
    S1  graph builder: per-epoch, per-channel z-score, clip ±5,
                       then 10 x (19 electrodes x 5 bands) band power  (graph_builder.py)
    S2  model:         per-window LayerNorm over (electrodes x bands) (ins_hdgs_cmt.py)
    S3  SNN proxy:     softmax-band-weighted sum of S2 -> (19 x 10)   (the ONLY input the
                       spiking encoder ever receives; raw EEG is never wired in)

This script recomputes exactly those tensors from the epochs on disk and fits
the same subject-grouped logistic-regression probe at every stage.  The stage
where the AUC collapses is the stage that destroys the information.

Two reference rows are added:
    R0  absolute whole-head log band power of the raw (un-normalised) epoch
    R1  the same after the loader's per-subject z-score only (S0)

Usage (any label track that has pre-cut epochs; the control track is the point):
    python scripts/analysis/eeg_input_probe.py --label-source control
Outputs: results/label_<src>/statistics/eeg_input_probe.{csv,md}
================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "src" / "data_pipeline" / "04_segmentation"
for p in (ROOT / "src" / "model", ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scipy.signal import welch                                            # noqa: E402
from data.channel_harmonizer import harmonize_eeg_channels, load_channel_names  # noqa: E402
from model.inference.graphs.graph_builder import compute_epoch_graphs, _BANDS   # noqa: E402

EEG_SR = 300.0
N_WINDOWS = 10
SNN_INIT_W = np.array([0.05, 0.40, 0.35, 0.10, 0.10])                    # ins_hdgs_cmt.py snn_band_weights init
SRC = {"phase3d": ("engagement_phase3d", "phase3d"), "purchase": ("engagement_purchase", "purchase"),
       "product": ("engagement_product", "product"), "control": ("engagement_control", "control")}


def log_bandpower_whole_head(eeg):
    f, pxx = welch(eeg, fs=EEG_SR, nperseg=min(256, len(eeg)), axis=0)     # (F, C)
    out = []
    for lo, hi in _BANDS.values():
        m = (f >= lo) & (f <= hi)
        integ = getattr(np, "trapezoid", None) or np.trapz
        out.append(np.log(integ(pxx[m], f[m], axis=0).mean() + 1e-12))
    tot = np.exp(out).sum()
    return np.array(out + [np.exp(out[2]) / (tot + 1e-12)])                # 5 log bands + relative alpha


def layer_norm_windows(nf):
    """model Step 3a: LayerNorm over the last two dims (C, 5) per window."""
    mu = nf.mean(axis=(1, 2), keepdims=True); sd = nf.std(axis=(1, 2), keepdims=True) + 1e-5
    return (nf - mu) / sd


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def load_track(src):
    sub_dir, suffix = SRC[src]
    recs, eegs = [], []
    subjects = sorted(d.name for d in SEG.iterdir() if d.is_dir() and d.name.startswith("S"))
    for s in subjects:
        d = SEG / s / "output"
        meta = d / sub_dir / "engagement_metadata.csv"
        lab = d / sub_dir / "engagement_labels.npy"
        ep = d / "epochs" / (f"eeg_epochs_{suffix}.npy" if src != "phase3d" else "eeg_epochs_phase3d.npy")
        if not (lab.exists() and ep.exists()):
            continue
        y = np.load(lab); eeg = np.load(ep, allow_pickle=True)
        if len(y) != len(eeg):
            continue
        names = load_channel_names(d / "epochs")
        for i in range(len(y)):
            e = harmonize_eeg_channels(np.asarray(eeg[i], dtype=np.float32), s, channel_names=names, verbose=False)
            recs.append(dict(subject_id=s, epoch_idx=i, label=int(y[i]))); eegs.append(e)
    return pd.DataFrame(recs), eegs


def stage_features(meta, eegs):
    """returns {stage_name: (N, F) array} computed exactly as the pipeline does."""
    N = len(eegs)
    R0 = np.stack([log_bandpower_whole_head(e) for e in eegs])
    # S0 loader: per-subject per-channel z-score
    eeg_s0 = [None] * N
    for s, idx in meta.groupby("subject_id").indices.items():
        cat = np.concatenate([eegs[i] for i in idx], axis=0)
        ctr, scl = cat.mean(axis=0, keepdims=True), cat.std(axis=0, keepdims=True) + 1e-8
        for i in idx:
            eeg_s0[i] = (eegs[i] - ctr) / scl
    R1 = np.stack([log_bandpower_whole_head(e) for e in eeg_s0])
    # S1 graph builder node features (includes its own per-epoch z-score)
    NF = np.stack([compute_epoch_graphs(e, n_windows=N_WINDOWS, fs=EEG_SR, conn_method="pearson", threshold=0.30)[0]
                   for e in eeg_s0])                                       # (N, W, C, 5)
    S1 = np.log(NF.mean(axis=1) + 1e-12).reshape(N, -1)                    # epoch-mean node features, 19x5
    S1_wh = np.log(NF.mean(axis=(1, 2)) + 1e-12)                           # whole-head, 5
    NF_ln = np.stack([layer_norm_windows(nf) for nf in NF])                # S2
    S2 = NF_ln.mean(axis=1).reshape(N, -1)
    proxy = (NF_ln * softmax(SNN_INIT_W)).sum(axis=-1)                     # (N, W, C)  S3
    S3 = proxy.mean(axis=1)                                                # (N, C)
    S3_full = proxy.reshape(N, -1)                                         # (N, C*W)
    return {
        "R0 raw epoch: whole-head log band power (5) + rel. alpha": R0,
        "R1 after loader per-subject z-score: same features": R1,
        "S1 graph-builder node features, epoch mean, 19 electrodes x 5 bands (log)": S1,
        "S1 graph-builder node features, whole-head mean (5 bands, log)": S1_wh,
        "S2 after model per-window LayerNorm: epoch mean, 19 x 5": S2,
        "S3 SNN proxy (band-weighted sum of S2): electrode means (19)": S3,
        "S3 SNN proxy, all 10 windows x 19 electrodes": S3_full,
    }


def loso(meta, X):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y, sid = meta.label.to_numpy(), meta.subject_id.to_numpy()
    ev = [s for s, g in meta.groupby("subject_id") if g.label.nunique() > 1]
    P = np.full(len(y), np.nan); aucs = []
    for s in ev:
        tr, te = sid != s, sid == s
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X[tr], y[tr])
        P[te] = clf.predict_proba(X[te])[:, 1]; aucs.append(roc_auc_score(y[te], P[te]))
    m = ~np.isnan(P)
    return dict(pooled_auc=roc_auc_score(y[m], P[m]), fold_auc=float(np.mean(aucs)), fold_auc_sd=float(np.std(aucs)),
                pooled_balacc=balanced_accuracy_score(y[m], (P[m] >= 0.5).astype(int)), n_folds=len(aucs), n_feat=X.shape[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-source", default="control", choices=list(SRC))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    meta, eegs = load_track(args.label_source)
    if meta.empty:
        raise SystemExit(f"no epochs found for label source {args.label_source}")
    print(f"[probe] {len(meta)} epochs, {meta.subject_id.nunique()} subjects, positives {meta.label.mean():.3f}; "
          f"EEG shape {eegs[0].shape}")
    rows = []
    for name, X in stage_features(meta, eegs).items():
        r = loso(meta, np.nan_to_num(X)); r["stage"] = name; rows.append(r)
        print(f"  {name:78s} pooled AUC {r['pooled_auc']:.3f}  fold AUC {r['fold_auc']:.3f}±{r['fold_auc_sd']:.3f}  "
              f"BalAcc {r['pooled_balacc']:.3f}  ({r['n_feat']} feat)")
    out = Path(args.out_dir) if args.out_dir else ROOT / "results" / (
        "" if args.label_source == "phase3d" else f"label_{args.label_source}") / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[["stage", "n_feat", "pooled_auc", "fold_auc", "fold_auc_sd", "pooled_balacc", "n_folds"]]
    df.to_csv(out / "eeg_input_probe.csv", index=False)
    lines = [f"# Stage-wise linear probe of the model's EEG input path ({args.label_source} label)", "",
             f"{len(meta)} epochs, {meta.subject_id.nunique()} subjects, LOSO logistic regression (C=1).", "",
             "| stage | features | pooled AUC | fold AUC (mean ± SD) | pooled BalAcc |", "|---|---|---|---|---|"]
    lines += [f"| {r.stage} | {r.n_feat} | {r.pooled_auc:.3f} | {r.fold_auc:.3f} ± {r.fold_auc_sd:.3f} | {r.pooled_balacc:.3f} |"
              for r in df.itertuples()]
    lines += ["", "Reading: the first stage whose AUC falls well below R0/R1 is the transformation that removes the "
              "information the deep model would need; S3 is the only input the spiking encoder receives."]
    (out / "eeg_input_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[probe] written {out / 'eeg_input_probe.md'}")


if __name__ == "__main__":
    main()
