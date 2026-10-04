#!/usr/bin/env python
"""Round-2 statistics from the saved per-fold CSVs (no training).  Run from the repository root:

    python scripts/revision/round2_tests.py            # writes results/statistics/round2_tests.md and .csv

Blocks (each skipped with a message when its inputs are missing):
  A. Engagement index, 37 folds: decoder vs EEGNet+ET-LSTM and vs ShallowConvNet+ET-LSTM; ET-LSTM vs each
     fusion.  Paired Wilcoxon (zero differences discarded), Holm within each metric, bootstrap 95% CI of the
     mean paired difference, matched-pairs rank-biserial r, wins/ties/losses.
  B. Fold-wise label vs pooled label for EEGNet, ShallowConvNet, ET-LSTM (same statistics; Table S17 rows).
  C. Purchase label, 42 folds: decoder vs EEGNet, ET-LSTM, EEGNet+ET-LSTM.
  D. Incremental validity on the purchase label: LOSOCV logistic probes on log dwell alone, on each model's
     epoch probability alone, and on both together (mean per-fold ROC-AUC and pooled ROC-AUC).
  E. Pooled ECE (10 equal-width bins) and Brier score for EEGNet+ET-LSTM and ShallowConvNet+ET-LSTM (Table 5).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "baselines"
PROD = ROOT / "results" / "label_product"
OUT = ROOT / "results" / "statistics"
EPOCHS = ROOT / "src/data_pipeline/04_segmentation/output/engagement_product/product_epochs.csv"
METRICS = ["balanced_acc", "roc_auc", "mcc"]
LINES: list[str] = []
ROWS: list[dict] = []


def say(s: str = "") -> None:
    print(s); LINES.append(s)


def load(csv: Path) -> pd.DataFrame | None:
    if not csv.exists():
        say(f"  (missing: {csv.relative_to(ROOT)})"); return None
    return pd.read_csv(csv).drop_duplicates("test_subject", keep="last").set_index("test_subject")


def boot_ci(d: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict:
    """a minus b over the shared test subjects."""
    idx = a.index.intersection(b.index)
    d = (a.loc[idx, metric] - b.loc[idx, metric]).to_numpy(float)
    nz = d[d != 0]
    if len(nz) >= 5:
        p = float(wilcoxon(nz, zero_method="wilcox").pvalue)
        r = rankdata(np.abs(nz)); w_pos = r[nz > 0].sum(); w_neg = r[nz < 0].sum()
        rrb = float((w_pos - w_neg) / (w_pos + w_neg))
    else:
        p, rrb = float("nan"), float("nan")
    lo, hi = boot_ci(d)
    return dict(n=len(d), mean_diff=float(d.mean()), ci_lo=lo, ci_hi=hi, p_raw=p, r_rb=rrb,
                wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()))


def holm(ps: list[float]) -> list[float]:
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps); adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * ps[i]); adj[i] = min(1.0, running)
    return adj.tolist()


def family(title: str, pairs: list[tuple[str, pd.DataFrame | None, pd.DataFrame | None]], block: str) -> None:
    say(f"\n## {title}")
    pairs = [(n, a, b) for n, a, b in pairs if a is not None and b is not None]
    if not pairs:
        say("  skipped: inputs missing"); return
    for metric in METRICS:
        res = [paired(a, b, metric) for _, a, b in pairs]
        adj = holm([r["p_raw"] for r in res])
        say(f"\n{metric}: comparison | n | mean diff [95% CI] | p raw | p Holm | r_rb | W/T/L")
        for (name, _, _), r, ph in zip(pairs, res, adj):
            say(f"  {name} | {r['n']} | {r['mean_diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
                f"{r['p_raw']:.3f} | {ph:.3f} | {r['r_rb']:+.2f} | {r['wins']}/{r['ties']}/{r['losses']}")
            ROWS.append(dict(block=block, comparison=name, metric=metric, p_holm=ph, **r))


# ---------------------------------------------------------------- A. index: decoder / ET-LSTM vs new fusions
say("# Round-2 tests from saved per-fold CSVs")
# losocv_abl_full.csv is the historical 24-channel run (balanced accuracy 0.747);
# the final production run's per-fold metrics are losocv_abl_full_final.csv
# (see results/statistics/STATS_ENVIRONMENT.md).
full = load(ROOT / "results/ablation/abl_full/losocv_abl_full_final.csv")
et = load(BASE / "dl_tuned/losocv_et_lstm.csv")
eegnet_et = load(BASE / "dl_tuned/losocv_eegnet_et.csv")
shallow_et = load(BASE / "dl_tuned/losocv_shallow_et.csv")
family("A. Engagement index (37 folds): paired differences, first minus second",
       [("decoder vs ET-LSTM (check: S14 gives -0.011 ROC-AUC)", full, et),
        ("decoder vs EEGNet+ET-LSTM", full, eegnet_et),
        ("decoder vs ShallowConvNet+ET-LSTM", full, shallow_et),
        ("ET-LSTM vs EEGNet+ET-LSTM", et, eegnet_et),
        ("ET-LSTM vs ShallowConvNet+ET-LSTM", et, shallow_et)], "A")

# ---------------------------------------------------------------- B. fold-wise vs pooled label
pairs_b = []
for stem, name in [("eegnet", "EEGNet"), ("shallow", "ShallowConvNet"), ("et_lstm", "ET-LSTM")]:
    v1 = load(BASE / f"dl_tuned/losocv_{stem}.csv"); v2 = load(BASE / f"dl_tuned_v2/losocv_{stem}.csv")
    pairs_b.append((f"{name}: fold-wise minus pooled", v2, v1))
family("B. Fold-wise label minus pooled label (Table S17 rows)", pairs_b, "B")

# ---------------------------------------------------------------- C. purchase label
pfull = load(PROD / "ablation/abl_full/losocv_abl_full.csv")
pairs_c = []
for stem, name in [("eegnet", "EEGNet"), ("et_lstm", "ET-LSTM"), ("eegnet_et", "EEGNet+ET-LSTM")]:
    pairs_c.append((f"decoder vs {name}", pfull, load(PROD / f"baselines/dl_tuned/losocv_{stem}.csv")))
family("C. Purchase label (42 folds): decoder minus each standard model", pairs_c, "C")

# ---------------------------------------------------------------- D. incremental validity beyond dwell
say("\n## D. Purchase label: does an epoch model add anything beyond session-level dwell?")
if not EPOCHS.exists():
    say(f"  skipped: {EPOCHS.relative_to(ROOT)} missing")
else:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    ep = pd.read_csv(EPOCHS)
    ep["logdwell"] = np.log1p(ep["total_dwell_s"].astype(float))

    def per_epoch_probs(model: str) -> pd.Series | None:
        """Model probability per epoch, aligned to product_epochs.csv row order within each subject."""
        if model == "decoder":
            if pfull is None: return None
            parts = []
            for sid, row in pfull.iterrows():
                p = np.asarray(ast.literal_eval(row["y_prob"]), float); y = np.asarray(ast.literal_eval(row["y_true"]))
                sub = ep.index[ep["subject_id"] == sid]
                if len(sub) != len(p) or not np.array_equal(ep.loc[sub, "label"].to_numpy(), y):
                    say(f"  alignment failed for {sid} (decoder): {len(sub)} epochs vs {len(p)} probabilities"); return None
                parts.append(pd.Series(p, index=sub))
            return pd.concat(parts)
        f = PROD / f"baselines/dl_tuned/fold_probs/probs_{model}.csv"
        if not f.exists():
            say(f"  (missing: {f.relative_to(ROOT)})"); return None
        pr = pd.read_csv(f); parts = []
        for sid, g in pr.groupby("test_subject", sort=False):
            sub = ep.index[ep["subject_id"] == sid]
            if len(sub) != len(g) or not np.array_equal(ep.loc[sub, "label"].to_numpy(), g["y_true"].to_numpy()):
                say(f"  alignment failed for {sid} ({model})"); return None
            parts.append(pd.Series(g["p1"].to_numpy(float), index=sub))
        return pd.concat(parts)

    def losocv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
        fold_auc, pooled_p, pooled_y = [], [], []
        for g in pd.unique(groups):
            tr, te = groups != g, groups == g
            if len(np.unique(y[te])) < 2: continue
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            fold_auc.append(roc_auc_score(y[te], p)); pooled_p.append(p); pooled_y.append(y[te])
        return float(np.mean(fold_auc)), float(roc_auc_score(np.concatenate(pooled_y), np.concatenate(pooled_p)))

    y = ep["label"].to_numpy(int); groups = ep["subject_id"].to_numpy()
    say(f"  probe | mean per-fold ROC-AUC | pooled ROC-AUC   ({len(ep)} epochs, {ep['subject_id'].nunique()} participants)")
    m_d, p_d = losocv_auc(ep[["logdwell"]].to_numpy(), y, groups)
    say(f"  log dwell alone | {m_d:.3f} | {p_d:.3f}")
    ROWS.append(dict(block="D", comparison="log dwell alone", metric="roc_auc", mean_fold=m_d, pooled=p_d))
    for model in ["decoder", "eegnet", "et_lstm", "eegnet_et"]:
        pm = per_epoch_probs(model)
        if pm is None: continue
        z = np.log(np.clip(pm.reindex(ep.index).to_numpy(float), 1e-6, 1 - 1e-6)); z = np.log(np.exp(z) / (1 - np.exp(z)))
        m1, p1 = losocv_auc(z[:, None], y, groups)
        m2, p2 = losocv_auc(np.column_stack([ep["logdwell"].to_numpy(), z]), y, groups)
        say(f"  {model} probability alone | {m1:.3f} | {p1:.3f}")
        say(f"  log dwell + {model} probability | {m2:.3f} | {p2:.3f}   (gain over dwell alone: {m2 - m_d:+.3f} per fold)")
        ROWS.append(dict(block="D", comparison=f"{model} alone", metric="roc_auc", mean_fold=m1, pooled=p1))
        ROWS.append(dict(block="D", comparison=f"dwell + {model}", metric="roc_auc", mean_fold=m2, pooled=p2, gain=m2 - m_d))

# ---------------------------------------------------------------- E. calibration of the new fusions
say("\n## E. Pooled calibration of the two new fusions (10 equal-width bins, 347 held-out epochs)")
for stem, name in [("eegnet_et", "EEGNet+ET-LSTM"), ("shallow_et", "ShallowConvNet+ET-LSTM")]:
    f = BASE / f"dl_tuned/fold_probs/probs_{stem}.csv"
    if not f.exists():
        say(f"  (missing: {f.relative_to(ROOT)})"); continue
    pr = pd.read_csv(f); p = pr["p1"].to_numpy(float); yy = pr["y_true"].to_numpy(int)
    bins = np.clip((p * 10).astype(int), 0, 9); ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any(): ece += m.mean() * abs(p[m].mean() - yy[m].mean())
    brier = float(np.mean((p - yy) ** 2))
    say(f"  {name}: ECE {ece:.2f}, Brier {brier:.2f}, epochs {len(pr)}")
    ROWS.append(dict(block="E", comparison=name, metric="calibration", ece=ece, brier=brier, n=len(pr)))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "round2_tests.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")
pd.DataFrame(ROWS).to_csv(OUT / "round2_tests.csv", index=False)
say(f"\nwrote {OUT / 'round2_tests.md'} and .csv")
