"""
================================================================
Soft-rule fidelity vs. the bypass gate (Reviewer 2, comment 5)
================================================================
Eq. (18):  y = softmax( alpha * (W_b z + b_b) + (1 - alpha) * R ),
with R = sum_r a_r * l_r the soft-rule evidence (Eq. 17) and alpha a
learned sigmoid gate. The reviewer asks whether the rule activations are
faithful to the prediction, given that alpha lets the classifier bypass
the rules entirely.

This script answers with three measurements on the trained fold
checkpoints (no retraining):

  1. alpha             learned bypass weight per fold / ensemble member
  2. fidelity          agreement between the rule-only decision
                       argmax(R) and the final decision argmax(logits),
                       plus the correlation between the rule margin
                       (R_HIGH - R_LOW) and the final logit margin,
                       and per-rule correlation of a_r with P(HIGH)
  3. post-hoc alpha=0  LOSOCV metrics when the decision is taken from R
                       alone at inference (alpha forced to 0 after
                       training), and alpha=1 (bypass alone), against the
                       reported gated decision.

and, optionally, summarises a LOSOCV run TRAINED with alpha fixed at 0
(`NEUMA_NS_ALPHA_MODE=rule_only python main.py --label ns_rule_only`, or
`run_component_ablation.py --variant ns_rule_only`) against the full
model, as a paired Wilcoxon test.

Checkpoints are located as
  <ckpt-dir>/<label>_fold{NN}_e{k}.pt     (ensemble members)
  <ckpt-dir>/<label>_fold{NN}.pt          (single model)
and the fold -> test-subject mapping is read from the per-fold CSV.

Outputs
-------
  results/statistics/rule_fidelity_per_fold.csv
  results/statistics/rule_fidelity_per_rule.csv
  results/statistics/rule_fidelity.md

Usage
-----
  cd src/model
  CUDA_VISIBLE_DEVICES="" python ../../scripts/analysis/rule_fidelity.py \
      --ckpt-dir output/checkpoints/ins_hdgs_cmt_ch19fix \
      --label ins_hdgs_cmt_ch19fix \
      --fold-csv output/metrics/ins_hdgs_cmt_ch19fix/losocv_ins_hdgs_cmt_ch19fix.csv \
      [--rule-only-csv output/metrics/ns_rule_only/losocv_ns_rule_only.csv]
================================================================
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "src" / "model"
for p in (str(MODEL), str(MODEL.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.dataset import NeumaGraphDataset, build_dataloaders   # noqa: E402
from evaluation.losocv import _make_model                        # noqa: E402
from models.ins_hdgs_cmt import AblationConfig                   # noqa: E402

OUT = ROOT / "results" / "statistics"


def load_state(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    return {k.replace("module.", ""): v for k, v in sd.items()}


def fold_checkpoints(ckpt_dir: Path, label: str, fold: int):
    members = sorted(ckpt_dir.glob(f"{label}_fold{fold:02d}_e*.pt"),
                     key=lambda p: int(re.search(r"_e(\d+)\.pt$", p.name).group(1)))
    if members:
        return members
    single = ckpt_dir / f"{label}_fold{fold:02d}.pt"
    return [single] if single.exists() else []


@torch.no_grad()
def forward_all(model, loader):
    keys = ["logits", "rule_evidence", "bypass_logits", "rule_act"]
    acc = {k: [] for k in keys}
    ys = []
    for b in loader:
        out = model(eeg_windows=b["eeg_windows"].float(), adj_matrices=b["adj_matrices"].float(),
                    et_seq=b["et_seq"].float(), roi_vector=b["roi_vector"].float(),
                    weighted_adjs=b["weighted_adjs"].float())
        for k in keys:
            acc[k].append(out[k].cpu().numpy())
        ys.append(b["label"].numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}, np.concatenate(ys), float(out["bypass_alpha"].item())


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _metrics(y, p_high):
    pred = (p_high >= 0.5).astype(int)
    return dict(balanced_acc=balanced_accuracy_score(y, pred),
                roc_auc=roc_auc_score(y, p_high) if len(np.unique(y)) > 1 else np.nan,
                mcc=matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--fold-csv", required=True, help="per-fold LOSOCV CSV (fold, test_subject)")
    ap.add_argument("--rule-only-csv", default=None, help="per-fold CSV of a run trained with alpha=0")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    folds = pd.read_csv(args.fold_csv)[["fold", "test_subject"]].drop_duplicates()
    if args.max_folds:
        folds = folds.head(args.max_folds)

    per_fold, per_rule_rows = [], []
    pooled = {"a": [], "p_final": [], "y": []}
    for _, row in folds.iterrows():
        fold, subj = int(row.fold), str(row.test_subject)
        cks = fold_checkpoints(ckpt_dir, args.label, fold)
        if not cks:
            print(f"[fold {fold:02d}] no checkpoint for {subj} — skipped")
            continue
        ds = NeumaGraphDataset(subject_ids=[subj], precompute_graphs=True, augment=False)
        _, loader = build_dataloaders(ds, ds, batch_size=min(32, len(ds)), num_workers=0)
        mem = []
        alphas = []
        y = None
        for ck in cks:
            model = _make_model(ds.n_eeg_ch, ds.n_classes, AblationConfig.full(), ds.n_et_ch).eval()
            missing, unexpected = model.load_state_dict(load_state(ck), strict=False)
            if missing:
                print(f"  [warn] {ck.name}: {len(missing)} missing keys (e.g. {missing[:2]})")
            o, y, alpha = forward_all(model, loader)
            mem.append(o); alphas.append(alpha)
        # ensemble average of probabilities under each decision rule
        P_final = np.mean([_softmax(o["logits"]) for o in mem], axis=0)[:, 1]
        P_rule = np.mean([_softmax(o["rule_evidence"]) for o in mem], axis=0)[:, 1]
        P_byp = np.mean([_softmax(o["bypass_logits"]) for o in mem], axis=0)[:, 1]
        A = np.mean([o["rule_act"] for o in mem], axis=0)                 # (N, n_rules)
        margin_final = np.mean([o["logits"][:, 1] - o["logits"][:, 0] for o in mem], axis=0)
        margin_rule = np.mean([o["rule_evidence"][:, 1] - o["rule_evidence"][:, 0] for o in mem], axis=0)
        agree = float(((P_rule >= 0.5) == (P_final >= 0.5)).mean())
        r_margin = pearsonr(margin_rule, margin_final)[0] if len(y) > 2 and np.std(margin_rule) > 0 else np.nan
        m_final, m_rule, m_byp = _metrics(y, P_final), _metrics(y, P_rule), _metrics(y, P_byp)
        per_fold.append(dict(fold=fold, test_subject=subj, n=len(y), n_members=len(cks),
                             alpha_mean=float(np.mean(alphas)), alpha_min=float(np.min(alphas)), alpha_max=float(np.max(alphas)),
                             fidelity_agreement=agree, r_margin_rule_vs_final=r_margin,
                             **{f"{k}_gated": v for k, v in m_final.items()},
                             **{f"{k}_rule_only_posthoc": v for k, v in m_rule.items()},
                             **{f"{k}_bypass_only_posthoc": v for k, v in m_byp.items()}))
        pooled["a"].append(A); pooled["p_final"].append(P_final); pooled["y"].append(y)
        print(f"[fold {fold:02d}] {subj:>4s} n={len(y):2d} alpha={np.mean(alphas):.3f} agree={agree:.2f} "
              f"AUC gated={m_final['roc_auc']:.3f} rule-only={m_rule['roc_auc']:.3f} bypass-only={m_byp['roc_auc']:.3f}")

    if not per_fold:
        sys.exit("no folds evaluated")
    pf = pd.DataFrame(per_fold)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pf.to_csv(out / "rule_fidelity_per_fold.csv", index=False)

    # per-rule correlation of activation with the final P(HIGH), pooled over held-out epochs
    A = np.concatenate(pooled["a"]); P = np.concatenate(pooled["p_final"]); Y = np.concatenate(pooled["y"])
    for r in range(A.shape[1]):
        rho = spearmanr(A[:, r], P).correlation if np.std(A[:, r]) > 0 else np.nan
        per_rule_rows.append(dict(rule=r + 1, mean_act=float(A[:, r].mean()),
                                  mean_act_HIGH=float(A[Y == 1, r].mean()), mean_act_LOW=float(A[Y == 0, r].mean()),
                                  spearman_act_vs_pHIGH=float(rho),
                                  dominant_frac=float((A.argmax(1) == r).mean())))
    pr = pd.DataFrame(per_rule_rows)
    pr.to_csv(out / "rule_fidelity_per_rule.csv", index=False)

    # paired tests: gated vs post-hoc rule-only
    def _paired(a, b):
        d = pf[a].values - pf[b].values
        d = d[np.isfinite(d)]
        try:
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        return float(d.mean()), float(p)

    lines = [f"# Soft-rule fidelity — {args.label} ({len(pf)} folds)", "",
             f"Learned bypass gate alpha: mean {pf.alpha_mean.mean():.3f} (range {pf.alpha_min.min():.3f}–{pf.alpha_max.max():.3f} "
             f"across folds/members; init sigmoid(0.30)=0.574).", "",
             f"Decision fidelity: argmax(R) agrees with the final decision on {pf.fidelity_agreement.mean()*100:.1f}% "
             f"of held-out epochs (fold mean; min {pf.fidelity_agreement.min()*100:.0f}%); Pearson r between the rule "
             f"margin (R_HIGH−R_LOW) and the final logit margin: mean {np.nanmean(pf.r_margin_rule_vs_final):.3f}.", "",
             "| decision rule at inference | BalAcc | ROC-AUC | MCC |", "|---|---|---|---|"]
    for tag, name in [("gated", "gated (as reported, learned α)"), ("rule_only_posthoc", "rule evidence only (α forced 0)"),
                      ("bypass_only_posthoc", "bypass only (α forced 1)")]:
        lines.append(f"| {name} | {pf[f'balanced_acc_{tag}'].mean():.3f} ± {pf[f'balanced_acc_{tag}'].std():.3f} | "
                     f"{pf[f'roc_auc_{tag}'].mean():.3f} ± {pf[f'roc_auc_{tag}'].std():.3f} | "
                     f"{pf[f'mcc_{tag}'].mean():.3f} ± {pf[f'mcc_{tag}'].std():.3f} |")
    for m in ["balanced_acc", "roc_auc", "mcc"]:
        d, p = _paired(f"{m}_gated", f"{m}_rule_only_posthoc")
        lines.append(f"- gated − rule-only ({m}): Δ = {d:+.3f}, Wilcoxon p = {p:.3f}")
    lines += ["", "## Per-rule activation vs. final P(HIGH) (pooled held-out epochs)", "",
              "| rule | mean a_r | a_r HIGH | a_r LOW | Spearman ρ(a_r, P(HIGH)) | dominant in |", "|---|---|---|---|---|---|"]
    for _, r in pr.iterrows():
        lines.append(f"| {int(r.rule)} | {r.mean_act:.3f} | {r.mean_act_HIGH:.3f} | {r.mean_act_LOW:.3f} | "
                     f"{r.spearman_act_vs_pHIGH:+.2f} | {r.dominant_frac*100:.0f}% |")

    if args.rule_only_csv and Path(args.rule_only_csv).exists():
        ro = pd.read_csv(args.rule_only_csv).set_index("test_subject")
        full = pd.read_csv(args.fold_csv).set_index("test_subject")
        common = ro.index.intersection(full.index)
        lines += ["", f"## Model TRAINED with α fixed at 0 (`{Path(args.rule_only_csv).name}`, {len(common)} paired folds)", "",
                  "| metric | full (learned α) | rule-only trained | Δ | Wilcoxon p |", "|---|---|---|---|---|"]
        for m in ["balanced_acc", "roc_auc", "mcc"]:
            d = ro.loc[common, m].values - full.loc[common, m].values
            try:
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = 1.0
            lines.append(f"| {m} | {full.loc[common, m].mean():.3f} | {ro.loc[common, m].mean():.3f} | {d.mean():+.3f} | {p:.3f} |")
    else:
        lines += ["", "Rule-only TRAINED run not supplied. Launch on the GPU server:", "",
                  "```", "NEUMA_NS_ALPHA_MODE=rule_only python main.py --fold-parallel --label ns_rule_only", "```"]
    (out / "rule_fidelity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
