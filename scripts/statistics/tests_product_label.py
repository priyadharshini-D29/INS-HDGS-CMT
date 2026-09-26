#!/usr/bin/env python
"""Product-selection label: tests for the ShallowConvNet and ShallowConvNet+ET-LSTM runs (no training).
Run from the repository root AFTER merge_fold_parts.py has written the merged files:

    python scripts/revision/tests_product_label.py     # writes results/statistics/tests_product_label.md and .csv

Blocks (each skipped with a message when its inputs are missing):
  0. Run check: every product-label model has 42 folds, one row per participant, no leftover part files.
  1. Table 7 rows: mean +/- SD over folds of balanced accuracy, ROC-AUC and MCC for the decoder and all
     five standard models (EEGNet, ShallowConvNet, ET-LSTM, EEGNet+ET-LSTM, ShallowConvNet+ET-LSTM).
  2. Decoder minus each standard model, paired over the 42 folds: Wilcoxon signed-rank (zero differences
     discarded), bootstrap 95% CI of the mean paired difference (10,000 resamples), matched-pairs
     rank-biserial r, wins/ties/losses.  Holm correction within each metric over the FIVE comparisons
     (the family grows from three to five, so the three earlier p_Holm values are re-reported here).
  3. ShallowConvNet minus EEGNet and ShallowConvNet+ET-LSTM minus EEGNet+ET-LSTM (do the two standard
     convolutional encoders differ on this label?), Holm within each metric over these two.
  4. Incremental validity beyond session-level dwell: LOSOCV logistic probe on log dwell alone, on each
     model's held-out epoch probability alone, and on both; mean per-fold and pooled ROC-AUC, gain over
     dwell with a bootstrap 95% CI and Wilcoxon p over folds.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "results" / "label_product"
DL = PROD / "baselines" / "dl_tuned"
OUT = ROOT / "results" / "statistics"
EPOCHS = ROOT / "src/data_pipeline/04_segmentation/output/engagement_product/product_epochs.csv"
METRICS = ["balanced_acc", "roc_auc", "mcc"]
EXPECTED = 42
STANDARD = [("eegnet", "EEGNet"), ("shallow", "ShallowConvNet"), ("et_lstm", "ET-LSTM"),
            ("eegnet_et", "EEGNet+ET-LSTM"), ("shallow_et", "ShallowConvNet+ET-LSTM")]
LINES: list[str] = []
ROWS: list[dict] = []


def say(s: str = "") -> None:
    print(s, flush=True); LINES.append(s)


def load(csv: Path) -> pd.DataFrame | None:
    if not csv.exists():
        say(f"  (missing: {csv.relative_to(ROOT)})"); return None
    df = pd.read_csv(csv)
    dup = int(df["test_subject"].duplicated().sum())
    if dup:
        say(f"  WARNING {csv.name}: {dup} duplicated participants (last kept)")
    return df.drop_duplicates("test_subject", keep="last").set_index("test_subject")


def boot_ci(d: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_d(d: np.ndarray) -> dict:
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


def paired(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict:
    idx = a.index.intersection(b.index)
    return paired_d((a.loc[idx, metric] - b.loc[idx, metric]).to_numpy(float))


def holm(ps: list[float]) -> list[float]:
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps); adj = np.empty(m); running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * ps[i]); adj[i] = min(1.0, running)
    return adj.tolist()


def family(title: str, pairs: list, block: str) -> None:
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
                f"{r['p_raw']:.4f} | {ph:.4f} | {r['r_rb']:+.2f} | {r['wins']}/{r['ties']}/{r['losses']}")
            ROWS.append(dict(block=block, comparison=name, metric=metric, p_holm=ph, **r))


# ---------------------------------------------------------------- 0. run check
say("## 0. Run check (product-selection label)")
leftover = sorted(p.name for p in DL.glob("losocv_shallow*.part*.csv"))
merged = {s: (DL / f"losocv_{s}.csv").exists() for s in ("shallow", "shallow_et")}
say(f"  part files present: {len(leftover)} | merged shallow: {merged['shallow']} | merged shallow_et: {merged['shallow_et']}")
if leftover and not all(merged.values()):
    say("  -> parts exist but are not merged yet: run merge_fold_parts.py first (see box_launch_r3_shallow.sh block 3)")
dec = load(PROD / "ablation/abl_full/losocv_abl_full.csv")
models = {"decoder": ("Audited decoder", dec)}
for stem, name in STANDARD:
    models[stem] = (name, load(DL / f"losocv_{stem}.csv"))
for stem, (name, df) in models.items():
    if df is None:
        continue
    flag = "OK" if len(df) == EXPECTED else f"CHECK: expected {EXPECTED}"
    say(f"  {name}: {len(df)} folds  [{flag}]")

# ---------------------------------------------------------------- 1. Table 7 rows with SD
say("\n## 1. Table 7 rows: mean +/- SD over folds")
say("  model | folds | BalAcc | ROC-AUC | MCC")
for stem, (name, df) in models.items():
    if df is None:
        continue
    cells = [f"{df[m].mean():.3f} +/- {df[m].std(ddof=1):.3f}" for m in METRICS]
    say(f"  {name} | {len(df)} | " + " | ".join(cells))
    ROWS.append(dict(block="1", comparison=name, metric="summary", n=len(df),
                     **{f"{m}_mean": df[m].mean() for m in METRICS}, **{f"{m}_sd": df[m].std(ddof=1) for m in METRICS}))

# ---------------------------------------------------------------- 2. decoder vs the five standard models
family("2. Decoder minus each standard model (42 folds; Holm over five comparisons per metric)",
       [(f"decoder vs {name}", dec, models[stem][1]) for stem, name in STANDARD], "2")

# ---------------------------------------------------------------- 3. the two convolutional encoders
family("3. ShallowConvNet minus EEGNet, alone and fused (Holm over two comparisons per metric)",
       [("ShallowConvNet vs EEGNet", models["shallow"][1], models["eegnet"][1]),
        ("ShallowConvNet+ET-LSTM vs EEGNet+ET-LSTM", models["shallow_et"][1], models["eegnet_et"][1])], "3")

# ---------------------------------------------------------------- 4. incremental validity beyond dwell
say("\n## 4. Incremental validity: does an epoch model add anything beyond session-level dwell?")
if not EPOCHS.exists():
    say(f"  skipped: {EPOCHS.relative_to(ROOT)} missing")
else:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    ep = pd.read_csv(EPOCHS)
    ep["logdwell"] = np.log1p(ep["total_dwell_s"].astype(float))
    y = ep["label"].to_numpy(int); groups = ep["subject_id"].to_numpy()

    def per_epoch_probs(model: str) -> pd.Series | None:
        if model == "decoder":
            if dec is None: return None
            parts = []
            for sid, row in dec.iterrows():
                p = np.asarray(ast.literal_eval(row["y_prob"]), float); yt = np.asarray(ast.literal_eval(row["y_true"]))
                sub = ep.index[ep["subject_id"] == sid]
                if len(sub) != len(p) or not np.array_equal(ep.loc[sub, "label"].to_numpy(), yt):
                    say(f"  alignment failed for {sid} (decoder)"); return None
                parts.append(pd.Series(p, index=sub))
            return pd.concat(parts)
        f = DL / f"fold_probs/probs_{model}.csv"
        if not f.exists():
            say(f"  (missing: {f.relative_to(ROOT)})"); return None
        pr = pd.read_csv(f); parts = []
        for sid, g in pr.groupby("test_subject", sort=False):
            sub = ep.index[ep["subject_id"] == sid]
            if len(sub) != len(g) or not np.array_equal(ep.loc[sub, "label"].to_numpy(), g["y_true"].to_numpy()):
                say(f"  alignment failed for {sid} ({model})"); return None
            parts.append(pd.Series(g["p1"].to_numpy(float), index=sub))
        return pd.concat(parts)

    def losocv(X: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        fold, pooled_p, pooled_y = {}, [], []
        for g in pd.unique(groups):
            tr, te = groups != g, groups == g
            if len(np.unique(y[te])) < 2: continue
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            fold[g] = roc_auc_score(y[te], p); pooled_p.append(p); pooled_y.append(y[te])
        return fold, np.concatenate(pooled_y), np.concatenate(pooled_p)

    say(f"  probe | mean per-fold ROC-AUC | pooled ROC-AUC | gain over dwell [95% CI], Wilcoxon p, W/T/L   "
        f"({len(ep)} epochs, {ep['subject_id'].nunique()} participants)")
    f_d, yy, pp = losocv(ep[["logdwell"]].to_numpy())
    m_d = float(np.mean(list(f_d.values())))
    say(f"  log dwell alone | {m_d:.3f} | {roc_auc_score(yy, pp):.3f}")
    ROWS.append(dict(block="4", comparison="log dwell alone", metric="roc_auc", mean_fold=m_d, pooled=roc_auc_score(yy, pp)))
    for model in ["decoder"] + [s for s, _ in STANDARD]:
        pm = per_epoch_probs(model)
        if pm is None: continue
        pr = np.clip(pm.reindex(ep.index).to_numpy(float), 1e-6, 1 - 1e-6); z = np.log(pr / (1 - pr))
        f1, y1, p1 = losocv(z[:, None])
        f2, y2, p2 = losocv(np.column_stack([ep["logdwell"].to_numpy(), z]))
        keys = [k for k in f_d if k in f2]
        gain = paired_d(np.array([f2[k] - f_d[k] for k in keys]))
        m1, m2 = float(np.mean(list(f1.values()))), float(np.mean(list(f2.values())))
        say(f"  {model} probability alone | {m1:.3f} | {roc_auc_score(y1, p1):.3f}")
        say(f"  log dwell + {model} | {m2:.3f} | {roc_auc_score(y2, p2):.3f} | {gain['mean_diff']:+.3f} "
            f"[{gain['ci_lo']:+.3f}, {gain['ci_hi']:+.3f}], p={gain['p_raw']:.3f}, "
            f"{gain['wins']}/{gain['ties']}/{gain['losses']}")
        ROWS.append(dict(block="4", comparison=f"{model} alone", metric="roc_auc", mean_fold=m1, pooled=roc_auc_score(y1, p1)))
        ROWS.append(dict(block="4", comparison=f"dwell + {model}", metric="roc_auc", mean_fold=m2,
                         pooled=roc_auc_score(y2, p2), gain=gain["mean_diff"], **{k: v for k, v in gain.items() if k != "mean_diff"}))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tests_product_label.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")
pd.DataFrame(ROWS).to_csv(OUT / "tests_product_label.csv", index=False)
say(f"\nwrote {OUT / 'tests_product_label.md'} and .csv")
