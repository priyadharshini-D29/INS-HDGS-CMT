"""
================================================================
Subject-Invariance Experiment Evaluation Harness
================================================================
Cross-cutting evaluation for the subject-invariance ablation
campaign (Phases 5 + 6 + model selection).  Given a finished
LOSOCV run <label>, it produces, under
output/subject_invariance/<label>/ :

  PHASE 5  hard_subject_tracking.csv   (S21,S03,S13,S35,S36)
           per_subject_metrics.csv
  PHASE 6  embeddings.npy / subject_ids.npy / labels.npy
           umap_labels.png  umap_subjects.png
           tsne_labels.png  tsne_subjects.png
           embedding_validation.json   (probe acc, silhouettes)
  SUMMARY  summary.json
           {mcc, balanced_acc, f1, accuracy, roc_auc,
            subject_probe_acc, subject_probe_chance,
            silhouette_subject, silhouette_label,
            hard_bal_acc_mean, hard_mcc_mean}

Model selection across experiments: PRIMARY MCC → BalAcc → F1.

Usable two ways:
  CLI:     python subject_invariance_eval.py --label <run_label>
  Library: from subject_invariance_eval import evaluate_experiment
           summary = evaluate_experiment("focal_abl_g2p0_balanced")
"""

from __future__ import annotations
import os, sys, json, glob, argparse, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

HARD_SUBJECTS = ["S21", "S03", "S13", "S35", "S36"]
METRICS_DIR   = Path("output/metrics")
CKPT_ROOT     = Path("output/checkpoints")
OUT_ROOT      = Path("output/subject_invariance")
RANDOM_STATE  = 42


# ── Phase 5: per-subject + hard-subject tracking ────────────────────────────

def _load_fold_csv(label: str) -> pd.DataFrame:
    csv = METRICS_DIR / label / f"losocv_{label}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"missing {csv}")
    return pd.read_csv(csv)


def phase5_tracking(label: str, df: pd.DataFrame, out: Path):
    # one fold == one held-out subject
    cols_map = {
        "test_subject": "subject",
        "accuracy":     "accuracy",
        "balanced_acc": "bal_acc",
        "f1":           "f1",
        "mcc":          "mcc",
        "roc_auc":      "auc",
    }
    keep = [c for c in cols_map if c in df.columns]
    per = df[keep].rename(columns=cols_map).copy()
    per.to_csv(out / "per_subject_metrics.csv", index=False)

    hard = per[per.subject.isin(HARD_SUBJECTS)].copy()
    # stable hard-subject ordering
    hard["__o"] = hard.subject.map({s: i for i, s in enumerate(HARD_SUBJECTS)})
    hard = hard.sort_values("__o").drop(columns="__o")
    hard.insert(0, "experiment", label)
    hard.to_csv(out / "hard_subject_tracking.csv", index=False)
    return per, hard


# ── Phase 6: embedding extraction + validation ──────────────────────────────

def _subject_to_fold():
    from config.settings import SUBJECT_IDS
    return {s: i + 1 for i, s in enumerate(SUBJECT_IDS)}, SUBJECT_IDS


def extract_embeddings(label: str, out: Path, recompute=False):
    emb_f, subj_f, lab_f = (out / "embeddings.npy",
                            out / "subject_ids.npy", out / "labels.npy")
    if emb_f.exists() and not recompute:
        return (np.load(emb_f), np.load(subj_f, allow_pickle=True), np.load(lab_f))

    import torch
    from data.dataset import NeumaGraphDataset
    from utils.gpu import _build_raw_model

    ckdir = CKPT_ROOT / label
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    s2f, SUBJECT_IDS = _subject_to_fold()
    E, S, Y = [], [], []
    for subj in SUBJECT_IDS:
        fold = s2f[subj]
        ck = sorted(glob.glob(str(ckdir / f"*_fold{fold:02d}_e0.pt")))
        if not ck:
            continue
        try:
            ds = NeumaGraphDataset(subject_ids=[subj], precompute_graphs=True)
        except Exception:
            continue
        if len(ds) < 2:
            continue
        model = _build_raw_model(ds, None).to(dev)
        sd = torch.load(ck[0], map_location=dev, weights_only=False)
        model.load_state_dict(sd["model_state_dict"]
                              if isinstance(sd, dict) and "model_state_dict" in sd else sd)
        model.eval()
        with torch.no_grad():
            for i in range(len(ds)):
                b = ds[i]
                b = {k: (v.unsqueeze(0).to(dev) if isinstance(v, torch.Tensor) else v)
                     for k, v in b.items()}
                o = model(eeg_windows=b["eeg_windows"], adj_matrices=b["adj_matrices"],
                          et_seq=b["et_seq"], roi_vector=b["roi_vector"],
                          weighted_adjs=b["weighted_adjs"])
                E.append(o["fused"].cpu().numpy()[0]); S.append(subj)
                Y.append(int(ds.labels[i]))
        del model
        if dev != "cpu":
            torch.cuda.empty_cache()
    E = np.asarray(E, np.float32); S = np.asarray(S, object); Y = np.asarray(Y, int)
    np.save(emb_f, E); np.save(subj_f, S); np.save(lab_f, Y)
    return E, S, Y


def _subject_probe(E, S):
    Es = StandardScaler().fit_transform(E)
    y = pd.factorize(S)[0]
    n_sub = len(np.unique(y)); chance = 1.0 / n_sub
    k = int(max(2, min(5, np.min(np.bincount(y)))))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    lr = cross_val_score(LogisticRegression(max_iter=2000), Es, y, cv=cv,
                         scoring="accuracy").mean()
    rf = cross_val_score(RandomForestClassifier(n_estimators=300, n_jobs=-1,
                         random_state=RANDOM_STATE), Es, y, cv=cv,
                         scoring="accuracy").mean()
    return float(lr), float(rf), float(chance), n_sub


def _scatter(coords, key, title, path, discrete):
    fig, ax = plt.subplots(figsize=(9, 8))
    if discrete:
        cats = list(dict.fromkeys(key))
        cmap = plt.cm.get_cmap("tab20", max(len(cats), 1))
        for i, c in enumerate(cats):
            m = np.array([k == c for k in key])
            ax.scatter(coords[m, 0], coords[m, 1], s=26, alpha=0.75, color=cmap(i),
                       label=str(c), edgecolors="k", linewidths=0.2)
        if len(cats) <= 22:
            ax.legend(fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    else:
        for v, col, nm in [(0, "#1f77b4", "LOW"), (1, "#d62728", "HIGH")]:
            m = key == v
            ax.scatter(coords[m, 0], coords[m, 1], s=28, alpha=0.7, color=col,
                       label=nm, edgecolors="k", linewidths=0.2)
        ax.legend(fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=180,
                                                        bbox_inches="tight")
    plt.close()


def phase6_embedding_validation(label: str, out: Path, make_plots=True,
                                recompute=False):
    E, S, Y = extract_embeddings(label, out, recompute=recompute)
    if len(E) < 5:
        return dict(subject_probe_acc=None, subject_probe_acc_lr=None,
                    subject_probe_chance=None, silhouette_subject=None,
                    silhouette_label=None, n_embedded=int(len(E)))
    Es = StandardScaler().fit_transform(E)
    sil_sub = float(silhouette_score(Es, pd.factorize(S)[0]))
    sil_lab = float(silhouette_score(Es, Y))
    lr, rf, chance, n_sub = _subject_probe(E, S)

    if make_plots:
        try:
            import umap
            Z = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                          n_neighbors=min(15, len(E) - 1)).fit_transform(Es)
            _scatter(Z, Y, f"UMAP labels — {label}", out / "umap_labels.png", False)
            _scatter(Z, list(S), f"UMAP subjects — {label}",
                     out / "umap_subjects.png", True)
        except Exception as e:
            print(f"   UMAP failed: {e}")
        try:
            from sklearn.manifold import TSNE
            perp = min(30, max(5, len(E) // 4))
            Z = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=perp,
                     init="pca").fit_transform(Es)
            _scatter(Z, Y, f"t-SNE labels — {label}", out / "tsne_labels.png", False)
            _scatter(Z, list(S), f"t-SNE subjects — {label}",
                     out / "tsne_subjects.png", True)
        except Exception as e:
            print(f"   t-SNE failed: {e}")

    ev = dict(subject_probe_acc=rf, subject_probe_acc_lr=lr,
              subject_probe_chance=chance, n_subjects=n_sub,
              silhouette_subject=sil_sub, silhouette_label=sil_lab,
              n_embedded=int(len(E)))
    (out / "embedding_validation.json").write_text(json.dumps(ev, indent=2))
    return ev


# ── Driver ──────────────────────────────────────────────────────────────────

def evaluate_experiment(label: str, make_plots=True, recompute=False) -> dict:
    out = OUT_ROOT / label
    out.mkdir(parents=True, exist_ok=True)
    df = _load_fold_csv(label)
    per, hard = phase5_tracking(label, df, out)

    def _mean(col):
        return float(df[col].mean()) if col in df.columns else float("nan")

    summary = dict(
        experiment        = label,
        n_folds           = int(len(df)),
        mcc               = _mean("mcc"),
        balanced_acc      = _mean("balanced_acc"),
        f1                = _mean("f1"),
        accuracy          = _mean("accuracy"),
        roc_auc           = _mean("roc_auc"),
        hard_bal_acc_mean = float(hard["bal_acc"].mean()) if len(hard) else float("nan"),
        hard_mcc_mean     = float(hard["mcc"].mean()) if "mcc" in hard else float("nan"),
        hard_n_recovered  = int((hard["bal_acc"] > 0.5).sum()) if len(hard) else 0,
    )
    # config columns if present
    for c in ["lambda_dann", "lambda_mmd", "mmd_mode", "norm_mode",
              "focal_gamma", "alpha_strategy"]:
        if c in df.columns:
            summary[c] = df[c].iloc[0]

    ev = phase6_embedding_validation(label, out, make_plots=make_plots,
                                     recompute=recompute)
    summary.update(ev)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _fmt(s):
    return (f"MCC={s['mcc']:.4f}  BalAcc={s['balanced_acc']:.4f}  "
            f"F1={s['f1']:.4f}  hardBalAcc={s['hard_bal_acc_mean']:.4f}  "
            f"probe={s.get('subject_probe_acc')}  "
            f"sil(subj)={s.get('silhouette_subject')}  "
            f"sil(lab)={s.get('silhouette_label')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="run label (one or comma-list)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    labels = [x.strip() for x in args.label.split(",") if x.strip()]
    rows = []
    for lab in labels:
        print(f"\n=== {lab} ===")
        try:
            s = evaluate_experiment(lab, make_plots=not args.no_plots,
                                    recompute=args.recompute)
            print("  " + _fmt(s))
            rows.append(s)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
    if len(rows) > 1:
        rk = sorted(rows, key=lambda r: (r["mcc"], r["balanced_acc"], r["f1"]),
                    reverse=True)
        print("\nRanking (MCC → BalAcc → F1):")
        for i, r in enumerate(rk, 1):
            print(f"  #{i} {r['experiment']:<34} " + _fmt(r))


if __name__ == "__main__":
    main()
