"""
==========================================================================
NEUMA Phase 8 — Diagnostics & Failure-Analysis Module
==========================================================================
Comprehensive post-hoc analysis of LOSOCV results.

Analyses provided
-----------------
1.  Subject-wise failure analysis  (subject_failure_analysis.csv / .json)
2.  Pathological-fold detection with labelled failure types
3.  Calibration diagnostics        (ECE, reliability diagrams, histograms)
4.  Subject data distribution      (label balance, EEG/ET stats, graph density)
5.  Fold variance analysis         (EASY / MEDIUM / HARD tier comparison)
6.  Publication-ready plots        (output/analysis/<label>/*.png)
7.  Latent-space visualisation     (t-SNE / UMAP on extracted embeddings)
8.  Ablation-summary utility       (static method, compare multiple runs)
9.  Concise textual summary        (summary.txt + summary.json)

Inputs
------
• output/metrics/losocv_<label>.csv   (required — LOSOCV results)
• output/checkpoints/<label>/         (optional — for inference & embeddings)
• NeumaGraphDataset                   (optional — for subject data profiling)

No test-subject statistics are used in any way that could cause leakage;
all analysis is strictly post-hoc.
==========================================================================
"""
from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Non-interactive matplotlib back-end — must be set before pyplot import.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from sklearn.calibration import calibration_curve
    from sklearn.manifold import TSNE
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

import torch


# ── Thresholds ─────────────────────────────────────────────────────────────────

EASY_MCC  = 0.50   # MCC ≥ EASY_MCC  → EASY fold
HARD_MCC  = 0.30   # MCC <  HARD_MCC → HARD fold
LOW_N_TEST = 8     # test_n < threshold → LOW_SAMPLE_COUNT

# Colour palette (colourblind-friendly subset)
_C = {
    "EASY"    : "#27ae60",
    "HARD"    : "#c0392b",
    "MEDIUM"  : "#e67e22",
    "BLUE"    : "#2980b9",
    "PURPLE"  : "#8e44ad",
    "GREY"    : "#95a5a6",
    "BLACK"   : "#2c3e50",
}


# ── Expected Calibration Error ─────────────────────────────────────────────────

def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE aligned with training/metrics.py."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, max(len(y_true), 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        conf = float(y_prob[mask].mean())
        acc  = float((y_true[mask] == (y_prob[mask] >= 0.5).astype(int)).mean())
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


# ── Main class ─────────────────────────────────────────────────────────────────

class DiagnosticsRunner:
    """
    Orchestrates all diagnostic analyses for one LOSOCV experiment.

    Parameters
    ----------
    label        : experiment label (e.g. "ins_hdgs_cmt")
    results_csv  : path to LOSOCV CSV; auto-detected from label if None
    output_dir   : root for analysis outputs (default: output/analysis)
    ckpt_dir     : checkpoint directory; auto-detected from label if None
    subject_ids  : subjects to profile in distribution analysis
    ablation     : AblationConfig for embedding extraction (full if None)
    """

    def __init__(
        self,
        label        : str                  = "ins_hdgs_cmt",
        results_csv  : Optional[Path]       = None,
        output_dir   : Optional[Path]       = None,
        ckpt_dir     : Optional[Path]       = None,
        subject_ids  : Optional[List[str]]  = None,
        ablation                            = None,
    ):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from config.settings import METRICS_DIR, CKPT_DIR, SUBJECT_IDS

        self.label       = label
        self.ckpt_dir    = Path(ckpt_dir  or CKPT_DIR    / label)
        self.results_csv = Path(results_csv or METRICS_DIR / f"losocv_{label}.csv")
        self.out_dir     = Path(output_dir or "output/analysis") / label
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.subject_ids = subject_ids or SUBJECT_IDS
        self.ablation    = ablation

        self.df           : Optional[pd.DataFrame] = None
        self._failure_df  : Optional[pd.DataFrame] = None
        # Cache: fold_no → {y_true, y_prob_high, embeddings}
        self._inference   : Dict[int, dict]         = {}

    # ────────────────────────── public orchestrator ────────────────────────────

    def run_all(
        self,
        skip_embeddings : bool = False,
        skip_data_stats : bool = False,
    ) -> None:
        """Run every diagnostic stage and save all outputs."""
        w = 62
        print(f"\n{'='*w}")
        print(f"  DIAGNOSTICS — {self.label}")
        print(f"{'='*w}")

        self._load_results()
        if self.df is None or self.df.empty:
            print(f"  [ERROR] No results at {self.results_csv}")
            return

        print(f"  {len(self.df)} folds loaded.\n")

        # ── 1. Re-run inference (populates y_true / y_prob per fold) ──────────
        print("  [1/8] Running per-fold inference from checkpoints …")
        self._collect_inference()

        # ── 2. Subject-wise failure analysis ──────────────────────────────────
        print("  [2/8] Failure analysis …")
        self.generate_failure_analysis()

        # ── 3. Calibration diagnostics ─────────────────────────────────────────
        print("  [3/8] Calibration diagnostics …")
        self.calibration_diagnostics()

        # ── 4. Subject distribution ────────────────────────────────────────────
        if not skip_data_stats:
            print("  [4/8] Subject distribution analysis …")
            self.subject_distribution_analysis()
        else:
            print("  [4/8] Subject distribution  — skipped (--skip-data-stats)")

        # ── 5. Fold variance ───────────────────────────────────────────────────
        print("  [5/8] Fold variance analysis …")
        self.fold_variance_analysis()

        # ── 6. Visualisations ──────────────────────────────────────────────────
        print("  [6/8] Generating plots …")
        self.generate_visualizations()

        # ── 7. Latent-space ────────────────────────────────────────────────────
        if not skip_embeddings and self._inference:
            print("  [7/8] Latent-space visualisation …")
            self.latent_space_visualization()
        else:
            print("  [7/8] Latent-space visualisation — skipped")

        # ── 8. Textual summary ─────────────────────────────────────────────────
        print("  [8/8] Summary …")
        self.generate_summary()

        print(f"\n  All outputs → {self.out_dir}")
        print(f"{'='*w}\n")

    # ──────────────────────────── data loading ──────────────────────────────────

    def _load_results(self) -> None:
        if not self.results_csv.exists():
            print(f"  [WARN] CSV not found: {self.results_csv}")
            return
        try:
            self.df = pd.read_csv(self.results_csv)
        except Exception as e:
            print(f"  [WARN] CSV load failed: {e}")

    # ────────────────────── inference from checkpoints ──────────────────────────

    def _collect_inference(self) -> None:
        """
        Load each fold's best checkpoint and run inference on the test subject.

        Populates self._inference[fold_no] = {
            y_true      : np.ndarray,
            y_prob_high : np.ndarray,   # P(HIGH=1) after temperature scaling
            embeddings  : np.ndarray,   # (N, D) EEG embedding
        }

        Folds whose checkpoint cannot be found or loaded are silently skipped.
        """
        try:
            from config.settings import (
                DEVICE, BATCH_SIZE,
                EMBED_DIM, GAT_L1_HEAD_DIM, GAT_L1_HEADS,
                T_NHEAD, T_LAYERS, T_FF_DIM,
                ET_LSTM_HIDDEN, ET_LSTM_LAYERS,
                ROI_HIDDEN_DIM, FUSION_HEADS, CLS_HIDDEN,
                N_ROIS, N_WINDOWS, ET_INPUT_DIM,
                SNN_TIME_STEPS, SNN_HIDDEN_DIM,
                NS_N_RULES, NS_HIDDEN_DIM,
                TEMPERATURE, DROPOUT, N_CLASSES,
            )
            from data.dataset        import NeumaGraphDataset, build_dataloaders
            from models.ins_hdgs_cmt import INS_HDGS_CMT, AblationConfig
        except ImportError as e:
            print(f"    [SKIP] Imports unavailable ({e}); metrics-only mode.")
            return

        ablation = self.ablation or AblationConfig.full()

        for _, row in self.df.iterrows():
            fold_no = int(row.get("fold", 0) or 0)
            subj    = str(row.get("test_subject", ""))

            # Checkpoint search order: ensemble-0 → bare fold file
            ckpt = self._find_ckpt(fold_no)
            if ckpt is None:
                continue

            try:
                ds = NeumaGraphDataset(subject_ids=[subj], precompute_graphs=True)
            except Exception:
                continue
            if len(ds) == 0:
                continue

            try:
                model = INS_HDGS_CMT(
                    n_eeg_ch       = ds.n_eeg_ch,
                    n_et_ch        = ds.n_et_ch,
                    n_rois         = N_ROIS,
                    n_windows      = N_WINDOWS,
                    n_classes      = N_CLASSES,
                    embed_dim      = EMBED_DIM,
                    snn_time_steps = SNN_TIME_STEPS,
                    snn_hidden_dim = SNN_HIDDEN_DIM,
                    gat_head_dim   = GAT_L1_HEAD_DIM,
                    gat_heads      = GAT_L1_HEADS,
                    t_nhead        = T_NHEAD,
                    t_layers       = T_LAYERS,
                    t_ff_dim       = T_FF_DIM,
                    et_lstm_hidden = ET_LSTM_HIDDEN,
                    et_lstm_layers = ET_LSTM_LAYERS,
                    roi_hidden     = ROI_HIDDEN_DIM,
                    fusion_heads   = FUSION_HEADS,
                    ns_n_rules     = NS_N_RULES,
                    ns_hidden_dim  = NS_HIDDEN_DIM,
                    cls_hidden     = CLS_HIDDEN,
                    dropout        = DROPOUT,
                    temperature    = TEMPERATURE,
                    ablation       = ablation,
                )
                raw_ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
                state = (raw_ckpt["model_state_dict"]
                         if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt
                         else raw_ckpt)
                model.load_state_dict(state, strict=False)
                model = model.to(DEVICE).eval()
            except Exception as e:
                print(f"    [WARN] Fold {fold_no:02d} checkpoint load: {e}")
                continue

            _, loader = build_dataloaders(
                ds, ds, batch_size=min(BATCH_SIZE, len(ds))
            )

            yt_lst, yp_lst, emb_lst = [], [], []
            with torch.no_grad():
                for batch in loader:
                    bdev = {k: v.to(DEVICE) if torch.is_tensor(v) else v
                            for k, v in batch.items()}
                    try:
                        out = model(
                            eeg_windows   = bdev["eeg_windows"],
                            adj_matrices  = bdev["adj_matrices"],
                            et_seq        = bdev["et_seq"],
                            roi_vector    = bdev["roi_vector"],
                            weighted_adjs = bdev.get("weighted_adjs"),
                        )
                    except TypeError:
                        out = model(
                            eeg_windows  = bdev["eeg_windows"],
                            adj_matrices = bdev["adj_matrices"],
                            et_seq       = bdev["et_seq"],
                            roi_vector   = bdev["roi_vector"],
                        )

                    logits = out["logits"]
                    probs  = torch.softmax(logits, dim=1)[:, 1]
                    yt_lst.extend(bdev["label"].cpu().numpy().tolist())
                    yp_lst.extend(probs.cpu().numpy().tolist())

                    # Collect EEG embedding (try multiple possible keys)
                    for key in ("eeg_emb", "fused_emb", "cls_emb", "emb"):
                        e = out.get(key)
                        if e is not None:
                            emb_lst.append(e.cpu().numpy())
                            break

            self._inference[fold_no] = {
                "y_true"      : np.array(yt_lst, dtype=int),
                "y_prob_high" : np.array(yp_lst, dtype=float),
                "embeddings"  : np.vstack(emb_lst) if emb_lst else np.empty((0,)),
                "subject"     : subj,
            }

            del model, ds, loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        n = len(self._inference)
        print(f"    Inference collected for {n}/{len(self.df)} folds.")

    def _find_ckpt(self, fold_no: int) -> Optional[Path]:
        """Return the best available checkpoint for a fold (e0 preferred)."""
        for suffix in (f"_e0.pt", f"_e1.pt", f"_e2.pt", f".pt"):
            p = self.ckpt_dir / f"{self.label}_fold{fold_no:02d}{suffix}"
            if p.exists():
                return p
        return None

    # ─────────────────────── failure type classification ───────────────────────

    def _classify_failure(self, row: pd.Series) -> Tuple[str, str, int, int]:
        """
        Classify a fold into a failure type.

        Uses per-sample y_true from inference cache (most accurate) when
        available, falling back to metric-pattern heuristics.

        Returns
        -------
        (failure_type, dominant_class, n_low, n_high)
        """
        fold_no = int(row.get("fold", 0) or 0)
        inf     = self._inference.get(fold_no, {})

        # ── Class counts from inference ───────────────────────────────────────
        if "y_true" in inf and len(inf["y_true"]) > 0:
            yt      = inf["y_true"]
            n_high  = int((yt == 1).sum())
            n_low   = int((yt == 0).sum())
            n_total = len(yt)
        else:
            n_total = int(row.get("test_n", 0) or 0)
            n_high  = -1     # unknown
            n_low   = -1

        # ── Metric values ─────────────────────────────────────────────────────
        mcc      = float(row.get("mcc",           0) or 0)
        f1       = float(row.get("f1",            0) or 0)
        recall   = float(row.get("recall",        0) or 0)
        roc_auc  = float(row.get("roc_auc",     0.5) or 0.5)
        pr_auc   = float(row.get("pr_auc",        0) or 0)
        ece      = float(row.get("ece",           0) or 0)
        acc      = float(row.get("accuracy",    0.0) or 0)
        bal_acc  = float(row.get("balanced_acc",0.0) or 0)

        dominant = "HIGH" if n_high >= n_low else "LOW"
        if n_high < 0:
            dominant = "HIGH" if acc > 0.5 else "LOW"

        # ── Priority-ordered failure detection ────────────────────────────────

        # 1. Insufficient test data → unreliable metrics
        if n_total < LOW_N_TEST:
            return "LOW_SAMPLE_COUNT", dominant, max(n_low, 0), max(n_high, 0)

        # 2. Single-class test set (roc_auc forced to 0.5 by compute_metrics,
        #    balanced_acc == 0 or == 1, accuracy is extreme)
        if n_high >= 0:
            if n_high == 0:
                return "SINGLE_CLASS_LOW",  "LOW",  n_total, 0
            if n_low == 0:
                return "SINGLE_CLASS_HIGH", "HIGH", 0, n_total

        # 3. Model predicts constant class (both precision and recall → 0,
        #    mcc == 0, roc_auc == 0.5 → model saw both classes but collapsed)
        if f1 < 0.01 and recall < 0.01 and roc_auc <= 0.51 and pr_auc < 0.01:
            return "OVERCONFIDENT_COLLAPSE", dominant, max(n_low, 0), max(n_high, 0)

        # 4. Negative MCC or AUC < chance → domain shift / adversarial fold
        if mcc < -0.05 or roc_auc < 0.45:
            return "DOMAIN_SHIFT", dominant, max(n_low, 0), max(n_high, 0)

        # 5. Zero F1 but usable AUC → model separates in probability space
        #    but threshold is misaligned
        if f1 < 0.01 and roc_auc > 0.55:
            return "ZERO_F1", dominant, max(n_low, 0), max(n_high, 0)

        # 6. Strong class imbalance in test set (makes metrics unstable)
        if n_high >= 0:
            ratio = max(n_high, n_low) / max(n_total, 1)
            if ratio >= 0.80:
                return "CLASS_IMBALANCE", dominant, max(n_low, 0), max(n_high, 0)

        # 7. Overconfident (high ECE even though accuracy OK)
        if ece > 0.35 and mcc < 0.35:
            return "OVERCONFIDENT_COLLAPSE", dominant, max(n_low, 0), max(n_high, 0)

        # 8. Low performance, no obvious single cause
        if mcc < HARD_MCC and roc_auc < 0.65:
            return "DOMAIN_SHIFT", dominant, max(n_low, 0), max(n_high, 0)

        return "OK", dominant, max(n_low, 0), max(n_high, 0)

    # ─────────────────────── requirement 1–3: failure analysis ─────────────────

    def generate_failure_analysis(self) -> pd.DataFrame:
        """
        Build subject_failure_analysis.csv sorted by MCC ascending.

        Columns
        -------
        test_subject, accuracy, f1, kappa, mcc, balanced_acc, roc_auc,
        pr_auc, ece, dominant_class, test_n, low_count, high_count,
        failure_type
        """
        metric_cols = [
            "accuracy", "f1", "kappa", "mcc", "balanced_acc",
            "roc_auc", "pr_auc", "ece",
        ]
        def _safe_float(v) -> float:
            """Return float(v) or NaN — handles 0.0 correctly (0.0 != NaN)."""
            if v is None:
                return float("nan")
            try:
                f = float(v)
                return f  # includes 0.0
            except (TypeError, ValueError):
                return float("nan")

        rows = []
        for _, row in self.df.iterrows():
            ftype, dominant, n_low, n_high = self._classify_failure(row)
            entry = {
                "fold"          : int(row.get("fold",   0) or 0),
                "test_subject"  : str(row.get("test_subject", "")),
                "test_n"        : int(row.get("test_n", 0) or 0),
                "failure_type"  : ftype,
                "dominant_class": dominant,
                "low_count"     : n_low,
                "high_count"    : n_high,
                "duration_s"    : _safe_float(row.get("duration_s")),
            }
            for c in metric_cols:
                entry[c] = _safe_float(row.get(c))
            rows.append(entry)

        out = (pd.DataFrame(rows)
               .sort_values("mcc", ascending=True)
               .reset_index(drop=True))

        out.to_csv(self.out_dir / "subject_failure_analysis.csv", index=False)
        out.to_json(self.out_dir / "subject_failure_analysis.json",
                    orient="records", indent=2)

        # ── Console report ────────────────────────────────────────────────────
        print(f"    Saved subject_failure_analysis.csv ({len(out)} folds)")
        bad = out[out["failure_type"] != "OK"]
        if not bad.empty:
            print(f"    Pathological folds ({len(bad)}):")
            for _, r in bad.iterrows():
                print(f"      {r['test_subject']:6s}  "
                      f"MCC={r['mcc']:.3f}  type={r['failure_type']}")

        # ── Failure-type count ────────────────────────────────────────────────
        fc = out["failure_type"].value_counts().to_dict()
        with (self.out_dir / "failure_type_counts.json").open("w") as f:
            json.dump({k: int(v) for k, v in fc.items()}, f, indent=2)

        self._failure_df = out
        return out

    # ─────────────────────── requirement 4: calibration ───────────────────────

    def calibration_diagnostics(self) -> dict:
        """
        Save per-fold ECE, mean confidence, and calibration statistics.

        Outputs
        -------
        calibration_ece_per_fold.csv
        calibration_diagnostics.json
        """
        records = {}
        for _, row in self.df.iterrows():
            fold_no = int(row.get("fold", 0) or 0)
            subj    = str(row.get("test_subject", ""))
            inf     = self._inference.get(fold_no, {})

            stored_ece = float(row.get("ece", float("nan")) or float("nan"))

            if "y_true" in inf and len(inf["y_true"]) > 0:
                yt = inf["y_true"]
                yp = inf["y_prob_high"]
                computed_ece  = _ece(yt, yp)
                mean_conf     = float(yp.mean())
                mean_acc      = float((yt == (yp >= 0.5).astype(int)).mean())
                overconfident = bool(mean_conf > mean_acc + 0.10)
                records[subj] = {
                    "fold"           : fold_no,
                    "stored_ece"     : round(stored_ece,   4),
                    "computed_ece"   : round(computed_ece, 4),
                    "mean_conf"      : round(mean_conf,    4),
                    "mean_acc"       : round(mean_acc,     4),
                    "overconfident"  : overconfident,
                    "n_samples"      : int(len(yt)),
                }
            else:
                records[subj] = {
                    "fold"       : fold_no,
                    "stored_ece" : round(stored_ece, 4),
                    "note"       : "no inference data",
                }

        with (self.out_dir / "calibration_diagnostics.json").open("w") as f:
            json.dump(records, f, indent=2)

        cal_df = pd.DataFrame.from_dict(records, orient="index").reset_index()
        cal_df.rename(columns={"index": "test_subject"}, inplace=True)
        cal_df.to_csv(self.out_dir / "calibration_ece_per_fold.csv", index=False)

        print(f"    Saved calibration_ece_per_fold.csv ({len(records)} folds)")
        return records

    # ─────────────────── requirement 5: subject distribution ──────────────────

    def subject_distribution_analysis(self) -> dict:
        """
        Per-subject label distribution, EEG/ET mean±std, graph density.

        Outputs
        -------
        subject_distribution.csv
        subject_distribution.json
        """
        try:
            from data.dataset import NeumaGraphDataset, collate_fn
            from torch.utils.data import DataLoader
        except ImportError as e:
            print(f"    [SKIP] Dataset import failed: {e}")
            return {}

        results: dict = {}
        for sid in self.subject_ids:
            try:
                ds = NeumaGraphDataset(subject_ids=[sid], precompute_graphs=False)
            except Exception as e:
                results[sid] = {"error": str(e)[:120]}
                continue
            if len(ds) == 0:
                results[sid] = {"n_samples": 0}
                continue

            loader = DataLoader(
                ds,
                batch_size    = min(16, len(ds)),
                shuffle       = False,
                collate_fn    = collate_fn,
                drop_last     = False,
            )

            all_labels, eeg_m, eeg_s, et_m, et_s, graph_d = [], [], [], [], [], []

            for batch in loader:
                all_labels.extend(batch["label"].numpy().tolist())

                eeg = batch["eeg_windows"].numpy()   # (B, W, Ch, T)
                eeg_m.append(float(eeg.mean()))
                eeg_s.append(float(eeg.std()))

                et = batch["et_seq"].numpy()          # (B, T_et, Ch_et)
                et_m.append(float(et.mean()))
                et_s.append(float(et.std()))

                adj = batch["adj_matrices"].numpy()   # (B, W, Ch, Ch)
                if adj.size > 0:
                    graph_d.append(float((adj > 0).mean()))

            lab = np.array(all_labels)
            n_h = int((lab == 1).sum())
            n_l = int((lab == 0).sum())

            results[sid] = {
                "n_samples"    : len(all_labels),
                "n_high"       : n_h,
                "n_low"        : n_l,
                "high_ratio"   : round(n_h / max(len(all_labels), 1), 4),
                "eeg_mean"     : round(float(np.mean(eeg_m)), 6) if eeg_m else None,
                "eeg_std"      : round(float(np.mean(eeg_s)), 6) if eeg_s else None,
                "et_mean"      : round(float(np.mean(et_m)),  6) if et_m  else None,
                "et_std"       : round(float(np.mean(et_s)),  6) if et_s  else None,
                "graph_density": round(float(np.mean(graph_d)), 4) if graph_d else None,
            }
            del ds, loader

        with (self.out_dir / "subject_distribution.json").open("w") as f:
            json.dump(results, f, indent=2)

        dist_df = pd.DataFrame.from_dict(results, orient="index").reset_index()
        dist_df.rename(columns={"index": "subject_id"}, inplace=True)
        dist_df.to_csv(self.out_dir / "subject_distribution.csv", index=False)

        print(f"    Saved subject_distribution.csv ({len(results)} subjects)")
        return results

    # ─────────────────── requirement 6: fold variance ─────────────────────────

    def fold_variance_analysis(self) -> dict:
        """
        Stratify folds into EASY / MEDIUM / HARD and compare statistics.

        Outputs
        -------
        fold_variance_summary.csv
        fold_variance_analysis.json
        """
        df = self._failure_df if self._failure_df is not None else self.df
        metric_cols = [
            c for c in ["accuracy", "f1", "kappa", "mcc",
                         "balanced_acc", "roc_auc", "pr_auc", "ece"]
            if c in df.columns
        ]

        def _tier_stats(sub: pd.DataFrame) -> dict:
            stats = {"n": int(len(sub))}
            for c in metric_cols:
                vals = pd.to_numeric(sub[c], errors="coerce").dropna()
                if vals.empty:
                    continue
                stats[c] = {
                    "mean"  : round(float(vals.mean()),   4),
                    "std"   : round(float(vals.std()),    4),
                    "median": round(float(vals.median()), 4),
                    "min"   : round(float(vals.min()),    4),
                    "max"   : round(float(vals.max()),    4),
                }
            return stats

        mcc_col = pd.to_numeric(df["mcc"], errors="coerce")
        easy  = df[mcc_col >= EASY_MCC]
        medium= df[(mcc_col >= HARD_MCC) & (mcc_col < EASY_MCC)]
        hard  = df[mcc_col <  HARD_MCC]

        # Overall fold-level variance
        variance: dict = {}
        for c in metric_cols:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if vals.empty:
                continue
            variance[c] = {
                "mean" : round(float(vals.mean()), 4),
                "std"  : round(float(vals.std()),  4),
                "cv"   : round(float(vals.std() / (abs(vals.mean()) + 1e-9)), 4),
                "range": round(float(vals.max() - vals.min()), 4),
            }

        subj_col = df.get("test_subject", df.index).astype(str)
        summary = {
            "fold_variance"   : variance,
            "EASY"            : _tier_stats(easy),
            "MEDIUM"          : _tier_stats(medium),
            "HARD"            : _tier_stats(hard),
            "easy_subjects"   : subj_col[mcc_col >= EASY_MCC].tolist(),
            "hard_subjects"   : subj_col[mcc_col <  HARD_MCC].tolist(),
        }

        with (self.out_dir / "fold_variance_analysis.json").open("w") as f:
            json.dump(summary, f, indent=2)

        # Flat CSV for easy diffing
        tier_rows = []
        for tier_name, tier_df in [("EASY", easy), ("MEDIUM", medium), ("HARD", hard)]:
            r: dict = {"tier": tier_name, "n_folds": len(tier_df)}
            for c in metric_cols:
                vals = pd.to_numeric(tier_df[c], errors="coerce").dropna()
                if not vals.empty:
                    r[f"{c}_mean"] = round(float(vals.mean()), 4)
                    r[f"{c}_std"]  = round(float(vals.std()),  4)
            tier_rows.append(r)

        pd.DataFrame(tier_rows).to_csv(
            self.out_dir / "fold_variance_summary.csv", index=False
        )
        print(f"    Saved fold_variance_analysis.json  "
              f"(EASY={len(easy)}, MED={len(medium)}, HARD={len(hard)})")
        return summary

    # ──────────────────── requirement 7: visualisations ───────────────────────

    def generate_visualizations(self) -> None:
        """Orchestrate all plot generation."""
        df = self._failure_df if self._failure_df is not None else self.df
        self._plot_mcc_histogram(df)
        self._plot_subject_mcc_bar(df)
        self._plot_calibration_curves()
        self._plot_confidence_distributions()
        self._plot_easy_hard_comparison(df)
        self._plot_fold_variance_boxplot(df)
        self._plot_failure_type_pie(df)

    def _plot_mcc_histogram(self, df: pd.DataFrame) -> None:
        mcc = pd.to_numeric(df["mcc"], errors="coerce").dropna().values
        if len(mcc) == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(mcc, bins=max(8, len(mcc) // 2),
                color=_C["BLUE"], edgecolor="white", linewidth=0.8, alpha=0.85)
        ax.axvline(EASY_MCC, color=_C["EASY"], ls="--", lw=1.8,
                   label=f"EASY ≥ {EASY_MCC}")
        ax.axvline(HARD_MCC, color=_C["HARD"], ls="--", lw=1.8,
                   label=f"HARD < {HARD_MCC}")
        ax.axvline(float(np.mean(mcc)), color=_C["BLACK"], ls="-", lw=2.0,
                   label=f"Mean = {np.mean(mcc):.3f}")
        ax.set_xlabel("Matthews Correlation Coefficient", fontsize=12)
        ax.set_ylabel("Fold count", fontsize=12)
        ax.set_title(f"MCC Distribution — {self.label}", fontsize=13)
        ax.legend(framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        self._save_fig(fig, "mcc_histogram.png")

    def _plot_subject_mcc_bar(self, df: pd.DataFrame) -> None:
        df_s = df.sort_values("mcc", ascending=True).reset_index(drop=True)
        subjects = (df_s.get("test_subject", df_s.index)).astype(str).values
        mcc      = pd.to_numeric(df_s["mcc"], errors="coerce").fillna(0).values

        colors = [
            _C["EASY"]   if m >= EASY_MCC else
            _C["GREY"]   if m >= HARD_MCC else
            _C["HARD"]
            for m in mcc
        ]

        fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 0.42), 5))
        ax.bar(range(len(subjects)), mcc, color=colors,
               edgecolor="white", linewidth=0.5)
        ax.axhline(0,        color=_C["BLACK"], lw=0.8)
        ax.axhline(EASY_MCC, color=_C["EASY"], lw=1.2, ls="--", alpha=0.8)
        ax.axhline(HARD_MCC, color=_C["HARD"], lw=1.2, ls="--", alpha=0.8)
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=50, ha="right", fontsize=8)
        ax.set_ylabel("MCC", fontsize=12)
        ax.set_title(f"Subject-wise MCC — {self.label}", fontsize=13)
        ax.legend(handles=[
            Patch(facecolor=_C["EASY"], label=f"EASY (≥{EASY_MCC})"),
            Patch(facecolor=_C["GREY"], label="MEDIUM"),
            Patch(facecolor=_C["HARD"], label=f"HARD (<{HARD_MCC})"),
        ], framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        self._save_fig(fig, "subject_mcc_bar.png")

    def _plot_calibration_curves(self) -> None:
        if not self._inference or not _SKLEARN:
            return

        fig, (ax_cal, ax_ece) = plt.subplots(1, 2, figsize=(13, 5))
        fold_eces: List[float] = []

        for fold_no, inf in self._inference.items():
            yt = inf.get("y_true")
            yp = inf.get("y_prob_high")
            if yt is None or len(yt) < 4 or len(np.unique(yt)) < 2:
                continue

            fold_eces.append(_ece(yt, yp))
            try:
                fp, mp = calibration_curve(yt, yp, n_bins=6, strategy="quantile")
                ax_cal.plot(mp, fp, alpha=0.3, color=_C["BLUE"], lw=1.2)
            except Exception:
                pass

        # Pooled reliability diagram
        all_yt = np.concatenate([v["y_true"]      for v in self._inference.values()
                                  if "y_true" in v])
        all_yp = np.concatenate([v["y_prob_high"] for v in self._inference.values()
                                  if "y_prob_high" in v])
        if len(all_yt) >= 4 and len(np.unique(all_yt)) >= 2:
            try:
                fp, mp = calibration_curve(all_yt, all_yp,
                                           n_bins=10, strategy="quantile")
                pooled_ece = _ece(all_yt, all_yp)
                ax_cal.plot(mp, fp, color=_C["BLACK"], lw=2.8,
                            label=f"Pooled (ECE={pooled_ece:.3f})", zorder=5)
            except Exception:
                pass

        ax_cal.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect calibration")
        ax_cal.set_xlabel("Mean predicted confidence", fontsize=11)
        ax_cal.set_ylabel("Fraction of positives",    fontsize=11)
        ax_cal.set_title("Reliability diagram", fontsize=12)
        ax_cal.legend(framealpha=0.9)
        ax_cal.grid(alpha=0.3)

        if fold_eces:
            ax_ece.hist(fold_eces, bins=10, color=_C["BLUE"],
                        edgecolor="white", alpha=0.85)
            ax_ece.axvline(float(np.mean(fold_eces)), color=_C["BLACK"], lw=2.0,
                           label=f"Mean ECE = {np.mean(fold_eces):.3f}")
            ax_ece.set_xlabel("ECE", fontsize=11)
            ax_ece.set_ylabel("Fold count", fontsize=11)
            ax_ece.set_title("ECE distribution across folds", fontsize=12)
            ax_ece.legend(framealpha=0.9)
            ax_ece.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        self._save_fig(fig, "calibration_curves.png")

    def _plot_confidence_distributions(self) -> None:
        if not self._inference:
            return

        probs_h, probs_l = [], []
        for inf in self._inference.values():
            yt = inf.get("y_true")
            yp = inf.get("y_prob_high")
            if yt is None:
                continue
            probs_h.extend(yp[yt == 1].tolist())
            probs_l.extend(yp[yt == 0].tolist())

        if not probs_h and not probs_l:
            return

        bins = np.linspace(0, 1, 21)
        fig, ax = plt.subplots(figsize=(9, 5))
        if probs_h:
            ax.hist(probs_h, bins=bins, alpha=0.65, color=_C["EASY"],
                    label=f"True HIGH  (n={len(probs_h)})", density=True)
        if probs_l:
            ax.hist(probs_l, bins=bins, alpha=0.65, color=_C["HARD"],
                    label=f"True LOW   (n={len(probs_l)})", density=True)
        ax.axvline(0.5, color=_C["BLACK"], ls="--", lw=1.4,
                   label="Decision boundary")
        ax.set_xlabel("P(HIGH)", fontsize=12)
        ax.set_ylabel("Density",  fontsize=12)
        ax.set_title(f"Confidence distribution by true class — {self.label}",
                     fontsize=13)
        ax.legend(framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        self._save_fig(fig, "confidence_distributions.png")

    def _plot_easy_hard_comparison(self, df: pd.DataFrame) -> None:
        metrics = [c for c in
                   ["accuracy", "f1", "balanced_acc", "roc_auc", "mcc"]
                   if c in df.columns]
        if not metrics:
            return

        mcc_col = pd.to_numeric(df["mcc"], errors="coerce")
        easy = df[mcc_col >= EASY_MCC]
        hard = df[mcc_col <  HARD_MCC]
        if easy.empty or hard.empty:
            return

        x    = np.arange(len(metrics))
        w    = 0.35
        e_m  = [pd.to_numeric(easy[m], errors="coerce").mean() for m in metrics]
        h_m  = [pd.to_numeric(hard[m], errors="coerce").mean() for m in metrics]
        e_s  = [pd.to_numeric(easy[m], errors="coerce").std()  for m in metrics]
        h_s  = [pd.to_numeric(hard[m], errors="coerce").std()  for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - w/2, e_m, w, yerr=e_s, capsize=4,
               color=_C["EASY"], alpha=0.85, label=f"EASY  (n={len(easy)})")
        ax.bar(x + w/2, h_m, w, yerr=h_s, capsize=4,
               color=_C["HARD"], alpha=0.85, label=f"HARD  (n={len(hard)})")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=10)
        ax.set_ylabel("Metric value", fontsize=12)
        ax.set_title(f"EASY vs HARD fold comparison — {self.label}", fontsize=13)
        ax.legend(framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=min(0, min(min(h_m), 0)) - 0.05)
        self._save_fig(fig, "easy_hard_comparison.png")

    def _plot_fold_variance_boxplot(self, df: pd.DataFrame) -> None:
        metrics = [c for c in
                   ["accuracy", "f1", "balanced_acc", "roc_auc", "mcc",
                    "kappa", "ece"]
                   if c in df.columns]
        if not metrics:
            return

        data   = [pd.to_numeric(df[m], errors="coerce").dropna().values
                  for m in metrics]
        labels = [m.replace("_", "\n") for m in metrics]

        fig, ax = plt.subplots(figsize=(12, 6))
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color=_C["BLACK"], linewidth=2.0),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2))
        for patch in bp["boxes"]:
            patch.set_facecolor(_C["BLUE"])
            patch.set_alpha(0.65)
        ax.axhline(0, color=_C["BLACK"], lw=0.8, ls="--")
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(f"Fold metric variance — {self.label}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        self._save_fig(fig, "fold_variance_boxplot.png")

    def _plot_failure_type_pie(self, df: pd.DataFrame) -> None:
        if "failure_type" not in df.columns:
            return
        counts = df["failure_type"].value_counts()
        if counts.empty:
            return

        palette = [
            "#27ae60", "#c0392b", "#e74c3c", "#e67e22",
            "#2980b9", "#8e44ad", "#95a5a6", "#1abc9c",
        ]
        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.0f%%",
            colors=palette[:len(counts)],
            startangle=140,
            pctdistance=0.75,
        )
        for t in texts + autotexts:
            t.set_fontsize(9)
        ax.set_title(f"Failure type distribution — {self.label}", fontsize=12)
        self._save_fig(fig, "failure_type_pie.png")

    # ─────────────────── requirement 8: latent-space ──────────────────────────

    def latent_space_visualization(self) -> None:
        """
        Collect EEG embeddings from inference cache and visualise with
        t-SNE (always) and UMAP (if umap-learn is installed).

        Outputs
        -------
        tsne_by_engagement.png
        tsne_by_subject.png
        umap_by_engagement.png  (if umap available)
        umap_by_subject.png     (if umap available)
        embeddings.csv
        """
        all_emb, all_y, all_sid = [], [], []

        for fold_no, inf in self._inference.items():
            emb = inf.get("embeddings")
            yt  = inf.get("y_true")
            sid = inf.get("subject", f"fold{fold_no:02d}")
            if emb is None or emb.ndim < 2 or emb.shape[0] == 0:
                continue
            if yt is None or len(yt) != emb.shape[0]:
                continue
            all_emb.append(emb)
            all_y.extend(yt.tolist())
            all_sid.extend([sid] * emb.shape[0])

        if not all_emb:
            print("    [SKIP] No embeddings available in inference cache.")
            return

        X   = np.vstack(all_emb)
        y   = np.array(all_y,  dtype=int)
        sid = np.array(all_sid)
        print(f"    Embeddings: {X.shape}  ({len(np.unique(sid))} subjects)")

        if not _SKLEARN:
            print("    [SKIP] sklearn not available — cannot run t-SNE.")
            return

        # t-SNE
        perp = min(30, max(5, len(X) // 5))
        try:
            X_tsne = TSNE(
                n_components=2, perplexity=perp,
                random_state=42, n_iter=1000, verbose=0,
            ).fit_transform(X)
        except Exception as e:
            print(f"    [WARN] t-SNE failed: {e}")
            X_tsne = None

        # UMAP
        X_umap = None
        try:
            import umap as _umap
            X_umap = _umap.UMAP(
                n_components=2, random_state=42, verbose=False
            ).fit_transform(X)
        except Exception:
            pass

        # Build subject-integer colour array
        uniq_sids = np.unique(sid)
        sid_int   = np.array([int(np.where(uniq_sids == s)[0][0]) for s in sid])

        for algo_name, X2d in [("tsne", X_tsne), ("umap", X_umap)]:
            if X2d is None:
                continue
            self._save_embedding_plot(
                X2d, y,
                title    = f"{algo_name.upper()} — coloured by engagement",
                path     = self.out_dir / f"{algo_name}_by_engagement.png",
                cmap_cls = ["LOW", "HIGH"],
            )
            self._save_embedding_plot(
                X2d, sid_int,
                title    = f"{algo_name.upper()} — coloured by subject",
                path     = self.out_dir / f"{algo_name}_by_subject.png",
                cmap_name= "tab20",
                n_unique = len(uniq_sids),
            )
            print(f"      → {algo_name}_by_engagement.png  "
                  f"{algo_name}_by_subject.png")

        # Save embeddings CSV
        D = X.shape[1]
        emb_df = pd.DataFrame(X, columns=[f"d{i}" for i in range(D)])
        emb_df["label"]   = y
        emb_df["subject"] = sid
        if X_tsne is not None:
            emb_df["tsne_x"] = X_tsne[:, 0]
            emb_df["tsne_y"] = X_tsne[:, 1]
        if X_umap is not None:
            emb_df["umap_x"] = X_umap[:, 0]
            emb_df["umap_y"] = X_umap[:, 1]
        emb_df.to_csv(self.out_dir / "embeddings.csv", index=False)

    @staticmethod
    def _save_embedding_plot(
        X2d         : np.ndarray,
        colors      : np.ndarray,
        title       : str,
        path        : Path,
        cmap_cls    : Optional[List[str]] = None,
        cmap_name   : str                 = "tab20",
        n_unique    : Optional[int]       = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        uniq = np.unique(colors)

        if cmap_cls and len(uniq) == len(cmap_cls):
            palette = [_C["HARD"], _C["EASY"]]   # LOW=red, HIGH=green
            for i, (lbl, c) in enumerate(zip(cmap_cls, uniq)):
                mask = colors == c
                ax.scatter(X2d[mask, 0], X2d[mask, 1], s=18,
                           alpha=0.65, edgecolors="none",
                           color=palette[i % len(palette)], label=lbl)
            ax.legend(framealpha=0.9, markerscale=2.0)
        else:
            n = n_unique or len(uniq)
            cmap = plt.cm.get_cmap(cmap_name, n)       # type: ignore[attr-defined]
            sc   = ax.scatter(X2d[:, 0], X2d[:, 1], c=colors,
                              cmap=cmap, s=18, alpha=0.65, edgecolors="none",
                              vmin=0, vmax=n - 1)
            plt.colorbar(sc, ax=ax, label="Subject index", shrink=0.8)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Component 1", fontsize=10)
        ax.set_ylabel("Component 2", fontsize=10)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ──────────────────── requirement 9: ablation summary ─────────────────────

    @staticmethod
    def ablation_summary(
        ablation_csvs : Dict[str, str],
        output_dir    : Path,
        metric_keys   : Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple ablation LOSOCV results side-by-side.

        Parameters
        ----------
        ablation_csvs : {ablation_label: path_to_losocv_csv}
        output_dir    : directory to save ablation_summary.csv / .png
        metric_keys   : metrics to compare (default: accuracy, f1, mcc,
                        roc_auc, balanced_acc, ece)

        Returns
        -------
        DataFrame with one row per ablation, mean±std for each metric.
        """
        if metric_keys is None:
            metric_keys = ["accuracy", "f1", "mcc", "roc_auc",
                           "balanced_acc", "ece"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for abl_label, csv_path in ablation_csvs.items():
            p = Path(csv_path)
            if not p.exists():
                warnings.warn(f"Ablation CSV not found: {csv_path}")
                continue
            df = pd.read_csv(p)
            row: dict = {"ablation": abl_label, "n_folds": len(df)}
            for m in metric_keys:
                if m in df.columns:
                    vals = pd.to_numeric(df[m], errors="coerce").dropna()
                    row[f"{m}_mean"] = round(float(vals.mean()), 4)
                    row[f"{m}_std"]  = round(float(vals.std()),  4)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        abl_df = pd.DataFrame(rows)
        abl_df.to_csv(output_dir / "ablation_summary.csv", index=False)

        # Heatmap
        mean_cols = [f"{m}_mean" for m in metric_keys
                     if f"{m}_mean" in abl_df.columns]
        if mean_cols:
            heat = (abl_df[["ablation"] + mean_cols]
                    .set_index("ablation")
                    .rename(columns=lambda c: c.replace("_mean", "")))

            fig, ax = plt.subplots(
                figsize=(max(8, len(mean_cols) * 1.5),
                         max(4, len(abl_df) * 0.7 + 1.5))
            )
            im = ax.imshow(heat.values.astype(float),
                           cmap="YlGn", aspect="auto",
                           vmin=0, vmax=1)
            ax.set_xticks(range(len(heat.columns)))
            ax.set_xticklabels(heat.columns, rotation=35, ha="right", fontsize=10)
            ax.set_yticks(range(len(heat.index)))
            ax.set_yticklabels(heat.index, fontsize=10)
            ax.set_title("Ablation comparison (mean across folds)", fontsize=13)
            for i in range(len(heat.index)):
                for j in range(len(heat.columns)):
                    v = heat.values[i, j]
                    if not np.isnan(float(v)):
                        ax.text(j, i, f"{float(v):.3f}", ha="center", va="center",
                                fontsize=9,
                                color="white" if float(v) > 0.70 else "black")
            plt.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(output_dir / "ablation_summary.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    → ablation_summary.csv + ablation_summary.png")

        return abl_df

    # ──────────────────── requirement 12: textual summary ─────────────────────

    def generate_summary(self) -> str:
        """
        Write a concise human-readable summary as summary.txt + summary.json.

        Covers:
        • Overall performance statistics
        • Best / worst subjects by MCC
        • Failure-pattern breakdown
        • Calibration observations
        • Subject heterogeneity (MCC coefficient of variation)
        """
        df = self._failure_df if self._failure_df is not None else self.df

        available = [c for c in
                     ["accuracy", "f1", "mcc", "balanced_acc",
                      "roc_auc", "pr_auc", "ece"]
                     if c in df.columns]

        def _s(col: str) -> Tuple[float, float]:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            return (float(v.mean()), float(v.std())) if not v.empty else (0.0, 0.0)

        mcc_col  = pd.to_numeric(df["mcc"], errors="coerce")
        subj_col = df.get("test_subject", df.index).astype(str)

        best_idx  = mcc_col.nlargest(5).index
        worst_idx = mcc_col.nsmallest(5).index
        best_subs  = [(subj_col[i], float(mcc_col[i])) for i in best_idx]
        worst_subs = [(subj_col[i], float(mcc_col[i])) for i in worst_idx]

        fail_counts: dict = {}
        if "failure_type" in df.columns:
            fail_counts = df["failure_type"].value_counts().to_dict()

        easy_n = int((mcc_col >= EASY_MCC).sum())
        hard_n = int((mcc_col <  HARD_MCC).sum())
        mid_n  = int(len(df)) - easy_n - hard_n
        mcc_cv = float(mcc_col.std() / (abs(mcc_col.mean()) + 1e-9))

        ece_m, ece_s = _s("ece") if "ece" in df.columns else (float("nan"), 0.0)

        lines = [
            "=" * 62,
            f"  DIAGNOSTICS SUMMARY — {self.label}",
            f"  {len(df)} folds analysed",
            "=" * 62,
            "",
            "  ── Overall performance ──────────────────────────────────",
        ]
        for c in available:
            m, s = _s(c)
            lines.append(f"    {c:<16}  {m:.4f} ± {s:.4f}")

        lines += [
            "",
            "  ── Best subjects (MCC) ──────────────────────────────────",
        ]
        for subj, mcc in best_subs:
            lines.append(f"    {subj:<8}  MCC = {mcc:+.4f}")

        lines += [
            "",
            "  ── Worst subjects (MCC) ─────────────────────────────────",
        ]
        for subj, mcc in worst_subs:
            lines.append(f"    {subj:<8}  MCC = {mcc:+.4f}")

        lines += [
            "",
            "  ── Major failure patterns ───────────────────────────────",
        ]
        if fail_counts:
            for ft, cnt in sorted(fail_counts.items(), key=lambda x: -x[1]):
                pct = 100 * cnt / max(len(df), 1)
                lines.append(f"    {ft:<30}  {cnt:3d} folds  ({pct:.0f}%)")
        else:
            lines.append("    (run generate_failure_analysis first)")

        lines += [
            "",
            "  ── Fold difficulty distribution ─────────────────────────",
            f"    EASY   (MCC ≥ {EASY_MCC})  : {easy_n} folds "
            f"({100*easy_n//max(len(df),1)}%)",
            f"    MEDIUM ({HARD_MCC}–{EASY_MCC}) : {mid_n} folds "
            f"({100*mid_n//max(len(df),1)}%)",
            f"    HARD   (MCC < {HARD_MCC})  : {hard_n} folds "
            f"({100*hard_n//max(len(df),1)}%)",
            "",
            "  ── Calibration observations ─────────────────────────────",
            f"    Mean ECE : {ece_m:.4f} ± {ece_s:.4f}",
            f"    Interpretation: < 0.10 = well-calibrated | "
            f"> 0.25 = overconfident",
            "",
            "  ── Subject heterogeneity ────────────────────────────────",
            f"    MCC mean = {mcc_col.mean():.4f}  "
            f"std = {mcc_col.std():.4f}  CV = {mcc_cv:.2f}",
            f"    {'High' if mcc_cv > 0.5 else 'Low'} inter-subject variability "
            f"(CV {'>' if mcc_cv > 0.5 else '≤'} 0.5)",
            "",
            "=" * 62,
        ]

        text = "\n".join(lines)
        print(text)

        (self.out_dir / "summary.txt").write_text(text)
        json_dict = {
            "label"         : self.label,
            "n_folds"       : int(len(df)),
            "best_subjects" : [{"subject": s, "mcc": round(m, 4)}
                               for s, m in best_subs],
            "worst_subjects": [{"subject": s, "mcc": round(m, 4)}
                               for s, m in worst_subs],
            "failure_counts": {k: int(v) for k, v in fail_counts.items()},
            "n_easy"        : easy_n,
            "n_medium"      : mid_n,
            "n_hard"        : hard_n,
            "ece_mean"      : round(ece_m, 4) if not np.isnan(ece_m) else None,
            "ece_std"       : round(ece_s, 4) if not np.isnan(ece_s) else None,
            "mcc_cv"        : round(mcc_cv, 4),
            "metrics"       : {c: {"mean": round(_s(c)[0], 4),
                                   "std" : round(_s(c)[1], 4)}
                               for c in available},
        }
        with (self.out_dir / "summary.json").open("w") as f:
            json.dump(json_dict, f, indent=2)

        print(f"\n  → summary.txt  summary.json")
        return text

    # ────────────────────────── shared helpers ─────────────────────────────────

    def _save_fig(self, fig: plt.Figure, name: str, dpi: int = 150) -> None:
        p = self.out_dir / name
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"      → {name}")
