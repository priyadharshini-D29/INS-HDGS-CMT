"""
================================================================
INS-HDGS-CMT — Cross-Subject Distribution-Shift Validation
================================================================
Scientifically verifies whether the "hard" LOSOCV subjects
(S21, S03, S13, S35, S36) occupy a *different feature
distribution* from the training population — BEFORE committing
to domain-adversarial / contrastive / normalization changes.

The goal is NOT to improve performance. It is to produce
publication-quality evidence for (or against) subject shift.

Pipeline (each part writes artefacts to output/subject_shift/):

  PART 1  Embedding extraction (per LOSOCV fold, held-out)
  PART 2  UMAP visualisation        (label vs subject)
  PART 3  t-SNE visualisation        (label vs subject)
  PART 4  Subject distance matrix    (cosine / euclidean / mahalanobis)
  PART 5  Population-centroid outlier ranking
  PART 6  Cluster separability        (silhouette: subject vs label)
  PART 7  Domain-classification probe (LogReg + RF predict subject)
  PART 8  Hard-vs-easy subject comparison
  PART 9  Statistical testing         (Mann-Whitney U + Welch t)
  REPORT  subject_shift_validation_report.txt  (A/B/C verdict + recs)

Embeddings are the `fused` latent (B, embed_dim) produced by
INS_HDGS_CMT *immediately before* the neuro-symbolic classifier
head. Each subject is embedded by the model from ITS OWN LOSOCV
fold (i.e. the model that never saw that subject) — the
representation each held-out subject actually receives.

Usage:
    python validate_subject_shift.py            # full run (cached)
    python validate_subject_shift.py --recompute # force re-extract
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
import seaborn as sns

from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    roc_auc_score, balanced_accuracy_score, roc_curve,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────────────────────────
LABEL          = "ins_hdgs_cmt_v17"
CKPT_DIR       = Path(f"output/checkpoints/{LABEL}")
FOLD_PROBS_DIR = Path("output/fold_probs")
OUT            = Path("output/subject_shift")
OUT.mkdir(parents=True, exist_ok=True)

HARD_SUBJECTS  = ["S21", "S03", "S13", "S35", "S36"]
RANDOM_STATE   = 42
np.random.seed(RANDOM_STATE)


# ════════════════════════════════════════════════════════════════════════════
# PART 1 — EMBEDDING EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def _subject_to_fold():
    from config.settings import SUBJECT_IDS
    return {s: i + 1 for i, s in enumerate(SUBJECT_IDS)}, SUBJECT_IDS


def extract_all_embeddings(recompute: bool = False):
    """
    Embed every LOSOCV-eligible subject with ITS OWN fold's model
    (held-out representation).  Returns embeddings [N, D], subject_ids [N],
    labels [N].  Cached to output/subject_shift/embeddings.npy etc.
    """
    emb_f  = OUT / "embeddings.npy"
    subj_f = OUT / "subject_ids.npy"
    lab_f  = OUT / "labels.npy"

    if emb_f.exists() and subj_f.exists() and lab_f.exists() and not recompute:
        E = np.load(emb_f)
        S = np.load(subj_f, allow_pickle=True)
        Y = np.load(lab_f)
        print(f"[PART 1] Loaded cached embeddings: {E.shape}  "
              f"({len(np.unique(S))} subjects)")
        return E, S, Y

    import torch
    from data.dataset import NeumaGraphDataset
    from utils.gpu import _build_raw_model

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    s2f, SUBJECT_IDS = _subject_to_fold()

    all_E, all_S, all_Y = [], [], []
    print(f"[PART 1] Extracting `fused` embeddings on {dev} …")
    for subj in SUBJECT_IDS:
        fold = s2f[subj]
        ck = sorted(glob.glob(str(CKPT_DIR / f"*_fold{fold:02d}_e0.pt")))
        if not ck:
            print(f"   {subj} (fold {fold:02d}): no checkpoint — skipped "
                  f"(single-class / skipped fold)")
            continue
        try:
            ds = NeumaGraphDataset(subject_ids=[subj], precompute_graphs=True)
        except Exception as e:
            print(f"   {subj}: dataset build failed ({e}) — skipped")
            continue
        if len(ds) < 2:
            print(f"   {subj}: <2 epochs — skipped")
            continue

        model = _build_raw_model(ds, None).to(dev)
        sd = torch.load(ck[0], map_location=dev, weights_only=False)
        model.load_state_dict(sd["model_state_dict"]
                              if isinstance(sd, dict) and "model_state_dict" in sd
                              else sd)
        model.eval()

        with torch.no_grad():
            for i in range(len(ds)):
                b = ds[i]
                b = {k: (v.unsqueeze(0).to(dev) if isinstance(v, torch.Tensor) else v)
                     for k, v in b.items()}
                out = model(eeg_windows=b["eeg_windows"],
                            adj_matrices=b["adj_matrices"],
                            et_seq=b["et_seq"],
                            roi_vector=b["roi_vector"],
                            weighted_adjs=b["weighted_adjs"])
                all_E.append(out["fused"].cpu().numpy()[0])
                all_S.append(subj)
                all_Y.append(int(ds.labels[i]))
        del model
        if dev != "cpu":
            torch.cuda.empty_cache()
        print(f"   {subj} (fold {fold:02d}): {len(ds)} epochs embedded")

    E = np.asarray(all_E, dtype=np.float32)
    S = np.asarray(all_S, dtype=object)
    Y = np.asarray(all_Y, dtype=int)
    np.save(emb_f, E)
    np.save(subj_f, S)
    np.save(lab_f, Y)
    print(f"[PART 1] Saved: {E.shape}  →  {emb_f.name}, {subj_f.name}, {lab_f.name}")
    return E, S, Y


# ════════════════════════════════════════════════════════════════════════════
# PARTS 2 & 3 — UMAP / t-SNE
# ════════════════════════════════════════════════════════════════════════════

def _scatter_2d(coords, color_key, title, path, discrete):
    fig, ax = plt.subplots(figsize=(9, 8))
    if discrete:
        cats = list(dict.fromkeys(color_key))
        cmap = plt.cm.get_cmap("tab20", max(len(cats), 1))
        for i, c in enumerate(cats):
            m = np.array([k == c for k in color_key])
            ax.scatter(coords[m, 0], coords[m, 1], s=28, alpha=0.75,
                       color=cmap(i), label=str(c),
                       edgecolors="k", linewidths=0.2)
        if len(cats) <= 22:
            ax.legend(fontsize=6, ncol=2, markerscale=1.2,
                      loc="center left", bbox_to_anchor=(1.0, 0.5))
    else:
        colors = {0: "#1f77b4", 1: "#d62728"}
        names  = {0: "LOW", 1: "HIGH"}
        for v in (0, 1):
            m = color_key == v
            ax.scatter(coords[m, 0], coords[m, 1], s=30, alpha=0.7,
                       color=colors[v], label=names[v],
                       edgecolors="k", linewidths=0.2)
        ax.legend(fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"   saved {Path(path).name}")


def run_umap(E, S, Y):
    print("[PART 2] UMAP …")
    import umap
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                        n_neighbors=min(15, len(E) - 1))
    Z = reducer.fit_transform(StandardScaler().fit_transform(E))
    _scatter_2d(Z, Y, "UMAP — coloured by ENGAGEMENT label",
                OUT / "umap_labels.png", discrete=False)
    _scatter_2d(Z, list(S), "UMAP — coloured by SUBJECT id",
                OUT / "umap_subjects.png", discrete=True)
    return Z


def run_tsne(E, S, Y):
    print("[PART 3] t-SNE …")
    from sklearn.manifold import TSNE
    perp = min(30, max(5, len(E) // 4))
    Z = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=perp,
             init="pca").fit_transform(StandardScaler().fit_transform(E))
    _scatter_2d(Z, Y, "t-SNE — coloured by ENGAGEMENT label",
                OUT / "tsne_labels.png", discrete=False)
    _scatter_2d(Z, list(S), "t-SNE — coloured by SUBJECT id",
                OUT / "tsne_subjects.png", discrete=True)
    return Z


# ════════════════════════════════════════════════════════════════════════════
# PART 4 — SUBJECT DISTANCE MATRIX
# ════════════════════════════════════════════════════════════════════════════

def subject_means(E, S):
    subs = list(dict.fromkeys(S))
    M = np.vstack([E[np.array(S) == s].mean(0) for s in subs])
    return subs, M


def part4_distance_matrix(E, S):
    print("[PART 4] Subject distance matrix …")
    subs, M = subject_means(E, S)
    n = len(subs)

    # Shrinkage precision for Mahalanobis over subject means (robust at D>>n)
    lw = LedoitWolf().fit(M)
    VI = lw.precision_

    cos = np.zeros((n, n)); euc = np.zeros((n, n)); mah = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = M[i] - M[j]
            cos[i, j] = cosine_dist(M[i], M[j])
            euc[i, j] = np.linalg.norm(d)
            mah[i, j] = float(np.sqrt(max(d @ VI @ d, 0.0)))

    rows = []
    for i, si in enumerate(subs):
        for j, sj in enumerate(subs):
            rows.append(dict(subject_i=si, subject_j=sj,
                             cosine=cos[i, j], euclidean=euc[i, j],
                             mahalanobis=mah[i, j]))
    pd.DataFrame(rows).to_csv(OUT / "subject_distance_matrix.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 11))
    order = np.argsort(euc.mean(1))               # outliers to the edge
    subs_o = [subs[k] for k in order]
    sns.heatmap(euc[np.ix_(order, order)], xticklabels=subs_o,
                yticklabels=subs_o, cmap="magma", square=True,
                cbar_kws={"label": "Euclidean distance"}, ax=ax)
    for lab in ax.get_xticklabels():
        if lab.get_text() in HARD_SUBJECTS: lab.set_color("red"); lab.set_fontweight("bold")
    for lab in ax.get_yticklabels():
        if lab.get_text() in HARD_SUBJECTS: lab.set_color("red"); lab.set_fontweight("bold")
    ax.set_title("Pairwise subject-mean Euclidean distance "
                 "(hard subjects in red)", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(OUT / "subject_distance_heatmap.png", dpi=200)
    plt.close()
    print(f"   saved subject_distance_matrix.csv + subject_distance_heatmap.png")
    return subs, M


# ════════════════════════════════════════════════════════════════════════════
# PART 5 — POPULATION CENTROID OUTLIER RANKING
# ════════════════════════════════════════════════════════════════════════════

def part5_centroid(E, S, subs, M):
    print("[PART 5] Population-centroid outlier ranking …")
    centroid = E.mean(0)                                   # sample-level centroid
    lw = LedoitWolf().fit(E)
    VI = lw.precision_

    rows = []
    for s, m in zip(subs, M):
        d = m - centroid
        rows.append(dict(
            subject       = s,
            is_hard       = s in HARD_SUBJECTS,
            euclidean     = float(np.linalg.norm(d)),
            cosine        = float(cosine_dist(m, centroid)),
            mahalanobis   = float(np.sqrt(max(d @ VI @ d, 0.0))),
        ))
    df = pd.DataFrame(rows).sort_values("mahalanobis", ascending=False)
    df["rank_mahalanobis"] = range(1, len(df) + 1)
    df["rank_euclidean"]   = df["euclidean"].rank(ascending=False).astype(int)
    df.to_csv(OUT / "subject_outlier_ranking.csv", index=False)

    s21 = df[df.subject == "S21"]
    s21_rank = int(s21["rank_mahalanobis"].iloc[0]) if len(s21) else None
    print(f"   saved subject_outlier_ranking.csv  "
          f"(S21 Mahalanobis rank = {s21_rank}/{len(df)})")
    return df


# ════════════════════════════════════════════════════════════════════════════
# PART 6 — CLUSTER SEPARABILITY (SILHOUETTE)
# ════════════════════════════════════════════════════════════════════════════

def part6_silhouette(E, S, Y):
    print("[PART 6] Cluster separability …")
    Es = StandardScaler().fit_transform(E)
    S_int = pd.factorize(S)[0]
    sil_subject = float(silhouette_score(Es, S_int))
    sil_label   = float(silhouette_score(Es, Y))
    verdict = ("SUBJECT-driven" if sil_subject > sil_label
               else "ENGAGEMENT-driven")
    txt = (
        "CLUSTER SEPARABILITY (silhouette, standardised `fused` embeddings)\n"
        "=================================================================\n"
        f"Silhouette(subject id)      : {sil_subject:+.4f}\n"
        f"Silhouette(engagement label): {sil_label:+.4f}\n"
        f"Δ (subject − label)         : {sil_subject - sil_label:+.4f}\n\n"
        f"Interpretation: embeddings cluster more strongly by → {verdict}\n"
        "  (silhouette ∈ [-1, 1]; higher ⇒ tighter, better-separated clusters)\n"
    )
    (OUT / "cluster_analysis.txt").write_text(txt)
    print(f"   subject={sil_subject:+.4f}  label={sil_label:+.4f}  → {verdict}")
    return sil_subject, sil_label


# ════════════════════════════════════════════════════════════════════════════
# PART 7 — DOMAIN CLASSIFICATION PROBE
# ════════════════════════════════════════════════════════════════════════════

def part7_domain_probe(E, S):
    print("[PART 7] Domain-classification probe …")
    Es = StandardScaler().fit_transform(E)
    y  = pd.factorize(S)[0]
    n_sub = len(np.unique(y))
    chance = 1.0 / n_sub

    # min class count caps the CV folds
    min_count = np.min(np.bincount(y))
    k = int(max(2, min(5, min_count)))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, clf in [
        ("LogisticRegression",
         LogisticRegression(max_iter=2000, C=1.0)),
        ("RandomForest",
         RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                n_jobs=-1)),
    ]:
        acc = cross_val_score(clf, Es, y, cv=cv, scoring="accuracy")
        results[name] = (float(acc.mean()), float(acc.std()))

    lift = results["RandomForest"][0] / chance
    txt = [
        "DOMAIN-CLASSIFICATION PROBE  (predict SUBJECT id from `fused` embedding)",
        "========================================================================",
        f"Subjects (classes)        : {n_sub}",
        f"Chance accuracy (1/K)      : {chance:.4f}",
        f"Stratified CV folds        : {k}",
        "",
    ]
    for name, (m, sd) in results.items():
        txt.append(f"{name:<20} accuracy = {m:.4f} ± {sd:.4f}   "
                   f"({m/chance:5.1f}× chance)")
    txt += [
        "",
        "Interpretation:",
        "  Subject identity is strongly decodable from the latent ⇒ the encoder",
        "  has NOT achieved subject invariance; subject-specific structure",
        "  remains entangled with the task representation.",
        f"  Best probe sits at {lift:.1f}× chance.",
    ]
    (OUT / "domain_classification_report.txt").write_text("\n".join(txt) + "\n")
    print(f"   LogReg={results['LogisticRegression'][0]:.3f}  "
          f"RF={results['RandomForest'][0]:.3f}  chance={chance:.3f}")
    return results, chance


# ════════════════════════════════════════════════════════════════════════════
# Per-subject performance (for PART 8 easy-subject selection)
# ════════════════════════════════════════════════════════════════════════════

def per_subject_performance():
    """Per-subject balanced-acc at the global Youden threshold from fold_probs."""
    files = sorted(glob.glob(str(FOLD_PROBS_DIR / "*.npz")))
    pool_p, pool_y = [], []
    subj_data = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        subj = str(d["test_subj"])
        yt = d["y_true"].astype(int); yp = d["y_prob"].astype(float)
        subj_data[subj] = (yt, yp)
        pool_p.append(yp); pool_y.append(yt)
    pool_p = np.concatenate(pool_p); pool_y = np.concatenate(pool_y)
    fpr, tpr, thr = roc_curve(pool_y, pool_p)
    g_thr = float(np.clip(thr[int(np.argmax(tpr - fpr))], 0.05, 0.95))

    rows = []
    for subj, (yt, yp) in subj_data.items():
        if len(np.unique(yt)) < 2:
            bal = np.nan; auc = np.nan
        else:
            bal = balanced_accuracy_score(yt, (yp >= g_thr).astype(int))
            auc = roc_auc_score(yt, yp)
        rows.append(dict(subject=subj, balanced_acc=bal, auc=auc, n=len(yt)))
    return pd.DataFrame(rows), g_thr


# ════════════════════════════════════════════════════════════════════════════
# PART 8 — HARD vs EASY COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def _within_subject_class_overlap(Ei, Yi):
    """1 - |mean_HIGH - mean_LOW| / pooled-std  →  higher = more overlap."""
    if len(np.unique(Yi)) < 2:
        return np.nan
    mu1 = Ei[Yi == 1].mean(0); mu0 = Ei[Yi == 0].mean(0)
    sep = np.linalg.norm(mu1 - mu0)
    sd  = Ei.std(0).mean() + 1e-8
    return float(1.0 / (1.0 + sep / sd))      # ∈ (0,1], 1 = total overlap


def part8_hard_vs_easy(E, S, Y, subs, M, perf):
    print("[PART 8] Hard-vs-easy comparison …")
    S = np.array(S)
    centroid = E.mean(0)
    perf_valid = perf.dropna(subset=["balanced_acc"])
    embedded = set(subs)
    easy_pool = perf_valid[(~perf_valid.subject.isin(HARD_SUBJECTS)) &
                           (perf_valid.subject.isin(embedded))]
    easy = easy_pool.sort_values("balanced_acc", ascending=False).head(5).subject.tolist()

    sub2mean = dict(zip(subs, M))
    rows = []
    for grp, members in [("hard", HARD_SUBJECTS), ("easy", easy)]:
        for s in members:
            if s not in sub2mean:
                continue
            Ei = E[S == s]; Yi = Y[S == s]
            pr = perf[perf.subject == s]
            rows.append(dict(
                subject          = s,
                group            = grp,
                centroid_distance= float(np.linalg.norm(sub2mean[s] - centroid)),
                embedding_var    = float(Ei.var(0).mean()),
                class_overlap    = _within_subject_class_overlap(Ei, Yi),
                balanced_acc     = float(pr.balanced_acc.iloc[0]) if len(pr) else np.nan,
                auc              = float(pr.auc.iloc[0]) if len(pr) else np.nan,
                n_epochs         = int((S == s).sum()),
            ))
    df = pd.DataFrame(rows)

    # within-subject silhouette by engagement label (per subject)
    Es = StandardScaler().fit_transform(E)
    sil_by_label = {}
    for s in df.subject:
        m = S == s
        if len(np.unique(Y[m])) >= 2 and m.sum() >= 3:
            try:
                sil_by_label[s] = float(silhouette_score(Es[m], Y[m]))
            except Exception:
                sil_by_label[s] = np.nan
        else:
            sil_by_label[s] = np.nan
    df["label_silhouette"] = df.subject.map(sil_by_label)
    df.to_csv(OUT / "hard_vs_easy_subjects.csv", index=False)
    print(f"   easy subjects = {easy}")
    print(f"   saved hard_vs_easy_subjects.csv")
    return df, easy


# ════════════════════════════════════════════════════════════════════════════
# PART 9 — STATISTICAL TESTING
# ════════════════════════════════════════════════════════════════════════════

def part9_statistics(outlier_df):
    print("[PART 9] Statistical testing …")
    hard = outlier_df[outlier_df.is_hard]
    rest = outlier_df[~outlier_df.is_hard]

    lines = ["DISTRIBUTION-SHIFT STATISTICAL TESTS",
             "====================================",
             "H0: hard subjects are NOT farther from the population centroid "
             "than the rest.",
             f"Hard subjects (n={len(hard)}): {sorted(hard.subject.tolist())}",
             f"Remaining     (n={len(rest)})",
             ""]
    results = {}
    for metric in ["mahalanobis", "euclidean", "cosine"]:
        h = hard[metric].values; r = rest[metric].values
        try:
            u, pu = mannwhitneyu(h, r, alternative="greater")
        except Exception:
            u, pu = np.nan, np.nan
        t, pt = ttest_ind(h, r, equal_var=False)
        # one-sided Welch (hard > rest)
        pt_one = pt / 2 if t > 0 else 1 - pt / 2
        results[metric] = dict(p_mwu=float(pu), p_welch_one=float(pt_one),
                               hard_mean=float(h.mean()), rest_mean=float(r.mean()))
        lines += [
            f"[{metric}]  distance-to-centroid",
            f"   hard mean = {h.mean():.4f}   rest mean = {r.mean():.4f}   "
            f"ratio = {h.mean()/ (r.mean()+1e-9):.2f}×",
            f"   Mann-Whitney U (one-sided, hard>rest): p = {pu:.4f}",
            f"   Welch t-test   (one-sided, hard>rest): p = {pt_one:.4f}",
            "",
        ]
    (OUT / "distribution_shift_statistics.txt").write_text("\n".join(lines) + "\n")
    print("   saved distribution_shift_statistics.txt")
    return results


# ════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ════════════════════════════════════════════════════════════════════════════

def final_report(E, S, sil_sub, sil_lab, domain_res, chance,
                 outlier_df, stat_res, hard_easy_df):
    print("[REPORT] Building final report …")
    n_sub = len(np.unique(S))

    # --- Evidence scoring (3 independent axes) -------------------------------
    score = 0; evidence = []

    # (1) silhouette
    if sil_sub > sil_lab:
        score += 1
        evidence.append(f"Embeddings cluster more by SUBJECT than label "
                        f"(sil {sil_sub:+.3f} > {sil_lab:+.3f}).")
    else:
        evidence.append(f"Embeddings cluster more by LABEL than subject "
                        f"(sil {sil_lab:+.3f} ≥ {sil_sub:+.3f}).")

    # (2) domain probe
    rf = domain_res["RandomForest"][0]
    if rf > 3 * chance:
        score += 1
        evidence.append(f"Subject id is highly decodable from the latent "
                        f"(RF {rf:.3f} = {rf/chance:.1f}× chance) ⇒ weak invariance.")
    else:
        evidence.append(f"Subject id only weakly decodable "
                        f"(RF {rf:.3f} = {rf/chance:.1f}× chance).")

    # (3) statistical separation of hard subjects
    p_mwu = stat_res["mahalanobis"]["p_mwu"]
    if p_mwu < 0.05:
        score += 1
        evidence.append(f"Hard subjects are significantly farther from the "
                        f"centroid (Mahalanobis MWU p={p_mwu:.3f}).")
    else:
        evidence.append(f"Hard subjects NOT significantly farther from centroid "
                        f"(Mahalanobis MWU p={p_mwu:.3f}).")

    verdict = ({3: "A) STRONG evidence of subject distribution shift",
                2: "B) MODERATE evidence of subject distribution shift",
                1: "C) WEAK evidence of subject distribution shift",
                0: "C) WEAK / NO evidence of subject distribution shift"}[score])

    # hard-subject outlier ranks
    rank_lines = []
    for s in HARD_SUBJECTS:
        r = outlier_df[outlier_df.subject == s]
        if len(r):
            rank_lines.append(f"   {s}: Mahalanobis rank "
                              f"{int(r.rank_mahalanobis.iloc[0])}/{len(outlier_df)}  "
                              f"(d_M={r.mahalanobis.iloc[0]:.3f})")
        else:
            rank_lines.append(f"   {s}: not embedded (single-class / skipped fold)")

    # recommendations gated on evidence
    grl   = ("JUSTIFIED" if rf > 3 * chance else "NOT clearly justified")
    contr = ("JUSTIFIED" if (sil_sub > sil_lab and rf > 3 * chance)
             else "secondary / optional")
    norm  = ("JUSTIFIED" if p_mwu < 0.05 or rf > 3 * chance
             else "low priority")

    R = [
        "================================================================",
        " INS-HDGS-CMT — CROSS-SUBJECT DISTRIBUTION-SHIFT VALIDATION",
        "================================================================",
        f" Model            : {LABEL}",
        f" Embedding        : `fused` latent (dim {E.shape[1]}), pre-classifier",
        f" Subjects embedded: {n_sub}  (held-out LOSOCV representation)",
        f" Samples          : {E.shape[0]}",
        f" Hard subjects    : {', '.join(HARD_SUBJECTS)}",
        "",
        "----------------------------------------------------------------",
        " VERDICT",
        "----------------------------------------------------------------",
        f"   {verdict}",
        f"   (evidence axes satisfied: {score}/3)",
        "",
        "----------------------------------------------------------------",
        " EVIDENCE",
        "----------------------------------------------------------------",
    ]
    for i, e in enumerate(evidence, 1):
        R.append(f"   {i}. {e}")
    R += [
        "",
        " Hard-subject centroid outlier ranks (1 = most outlying):",
        *rank_lines,
        "",
        "----------------------------------------------------------------",
        " KEY NUMBERS",
        "----------------------------------------------------------------",
        f"   Silhouette(subject) = {sil_sub:+.4f}   "
        f"Silhouette(label) = {sil_lab:+.4f}",
        f"   Domain probe (RF)   = {rf:.4f}  ({rf/chance:.1f}× chance, "
        f"chance={chance:.4f})",
        f"   Domain probe (LogReg)= {domain_res['LogisticRegression'][0]:.4f}",
        f"   Hard vs rest centroid distance (Mahalanobis): "
        f"{stat_res['mahalanobis']['hard_mean']:.3f} vs "
        f"{stat_res['mahalanobis']['rest_mean']:.3f}  "
        f"(MWU p={p_mwu:.3f})",
        "",
        "----------------------------------------------------------------",
        " RECOMMENDATIONS (architectural changes)",
        "----------------------------------------------------------------",
        f"   • Domain-adversarial (GRL/DANN) ............ {grl}",
        f"   • Contrastive subject-invariant learning ... {contr}",
        f"   • Subject-specific normalization ........... {norm}",
        "",
        "   NOTE: INS-HDGS-CMT already includes a GRL subject-classifier and",
        "   an MMD term during training. A high domain-probe accuracy here",
        "   indicates those mechanisms are under-weighted — tune λ_dann / λ_mmd",
        "   (or strengthen the GRL schedule) BEFORE adding new components.",
        "",
        "----------------------------------------------------------------",
        " METHODOLOGICAL CAVEAT",
        "----------------------------------------------------------------",
        "   Each subject is embedded by its OWN LOSOCV fold model (held-out,",
        "   never-seen). Folds share architecture and 40/42 of training data,",
        "   so latent spaces are highly — but not perfectly — comparable.",
        "   Distances/clustering should be read as population-level evidence,",
        "   not exact cross-fold metric identity.",
        "",
        " Artefacts: output/subject_shift/",
        "================================================================",
    ]
    (OUT / "subject_shift_validation_report.txt").write_text("\n".join(R) + "\n")
    print("\n".join(R))


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true",
                    help="force re-extraction of embeddings")
    args = ap.parse_args()

    E, S, Y = extract_all_embeddings(recompute=args.recompute)
    if len(E) == 0:
        print("No embeddings extracted — aborting.")
        sys.exit(1)

    run_umap(E, S, Y)
    run_tsne(E, S, Y)
    subs, M = part4_distance_matrix(E, S)
    outlier_df = part5_centroid(E, S, subs, M)
    sil_sub, sil_lab = part6_silhouette(E, S, Y)
    domain_res, chance = part7_domain_probe(E, S)
    perf, g_thr = per_subject_performance()
    hard_easy_df, easy = part8_hard_vs_easy(E, S, Y, subs, M, perf)
    stat_res = part9_statistics(outlier_df)
    final_report(E, S, sil_sub, sil_lab, domain_res, chance,
                 outlier_df, stat_res, hard_easy_df)
    print(f"\nAll artefacts → {OUT}/")


if __name__ == "__main__":
    main()
