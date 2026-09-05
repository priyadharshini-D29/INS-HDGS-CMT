"""
================================================================
INS-HDGS-CMT — Ablation Studies
================================================================
Systematically isolates each architectural component's
contribution to engagement classification performance.

Ablation configurations:
  1. full_ins_hdgs_cmt   — complete model (reference)
  2. no_snn              — remove SNN encoder
  3. no_graph            — remove dynamic GAT
  4. no_neuro_symbolic   — remove rule layer (standard head)
  5. no_fusion_tf        — replace 4-stage fusion with cross-attn
  6. eeg_only            — no ET stream
  7. no_et               — no eye tracking branch
  8. no_roi              — no ROI gating
  9. no_contrastive      — λ_contrast = 0
 10. no_mmd              — λ_MMD = 0
 11. baseline_linear     — linear EEG encoder, no components
================================================================
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings     import ABLATION_DIR, EPOCHS, BATCH_SIZE
from models.ins_hdgs_cmt import AblationConfig
from .losocv             import run_losocv


ABLATION_CONFIGS = {
    "full_ins_hdgs_cmt"   : AblationConfig.full(),
    "no_snn"              : AblationConfig.no_snn(),
    "no_graph"            : AblationConfig.no_graph(),
    "no_neuro_symbolic"   : AblationConfig.no_neuro_symbolic(),
    "ns_rule_only"        : AblationConfig.ns_rule_only(),
    "ns_explain_only"     : AblationConfig.ns_explain_only(),
    "no_fusion_transformer": AblationConfig.no_fusion_transformer(),
    "eeg_only"            : AblationConfig.eeg_only(),
    "eeg_only_mmd"        : AblationConfig.eeg_only_mmd(),
    "no_et"               : AblationConfig.no_et(),
    "no_roi"              : AblationConfig(use_roi=False, use_roi_modulation=False),
    "no_contrastive"      : AblationConfig(use_contrastive=False, use_infonce=False),
    "no_mmd"              : AblationConfig(use_mmd=False),
    "baseline_linear"     : AblationConfig.baseline_linear(),
}


def run_ablation(
    subject_ids  = None,
    configs      = None,
    epochs       : int  = EPOCHS,
    batch_size   : int  = BATCH_SIZE,
    save_dir     : Path = None,
    verbose      : bool = True,
) -> pd.DataFrame:
    """
    Run all ablation experiments and return a summary DataFrame.

    Parameters
    ----------
    configs    : dict name → AblationConfig (None = all)
    save_dir   : output directory

    Returns
    -------
    summary_df : pivot table (metric × ablation config)
    """
    save_dir = Path(save_dir or ABLATION_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    configs = configs or ABLATION_CONFIGS

    all_results = []

    for name, ablation in configs.items():
        if verbose:
            print(f"\n{'='*56}")
            print(f"  ABLATION: {name}")
            print(f"{'='*56}")

        try:
            df = run_losocv(
                subject_ids = subject_ids,
                ablation    = ablation,
                epochs      = epochs,
                batch_size  = batch_size,
                label       = name,
                save_dir    = save_dir,
                verbose     = verbose,
            )

            if df.empty:
                continue

            key_metrics = ["accuracy", "f1", "roc_auc", "pr_auc",
                           "kappa", "mcc", "balanced_acc", "ece",
                           "precision", "recall"]
            for m in key_metrics:
                if m not in df.columns:
                    continue
                vals = df[m].dropna()
                all_results.append({
                    "ablation": name,
                    "metric"  : m,
                    "mean"    : vals.mean(),
                    "std"     : vals.std(),
                    "n_folds" : len(vals),
                })

        except Exception as exc:
            if verbose:
                print(f"  [ERROR] {name}: {exc}")
            continue

    if not all_results:
        return pd.DataFrame()

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(save_dir / "ablation_summary.csv", index=False)

    # Pivot for easy comparison
    pivot = summary_df.pivot_table(
        index="metric", columns="ablation", values="mean"
    )

    if verbose:
        print(f"\n{'='*56}")
        print("  ABLATION SUMMARY (Mean across folds)")
        print(f"{'='*56}")
        print(pivot.round(4).to_string())
        print(f"\n  Saved: {save_dir}/ablation_summary.csv")

    pivot.to_csv(save_dir / "ablation_pivot.csv")
    return summary_df
