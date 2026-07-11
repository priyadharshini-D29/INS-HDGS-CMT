"""
Standalone explainability pass — no retraining.
Reads existing checkpoints and writes GradCAM / ROI / GAT attention files.

Usage:
  cd src/model
  CUDA_VISIBLE_DEVICES=0 python run_explain_only.py --label ins_hdgs_cmt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from config.settings import (
    SUBJECT_IDS, CKPT_DIR, EXPLAINABILITY_DIR,
)
from models.ins_hdgs_cmt import INS_HDGS_CMT as RDGANet, AblationConfig


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="ins_hdgs_cmt")
    p.add_argument("--config", default="full",
                   choices=["full", "phase8c", "eeg_only", "eeg_et", "eeg_gat",
                            "no_roi", "no_graph", "no_contrastive", "no_mmd"])
    p.add_argument("--subjects", default=None,
                   help="Comma-separated subject IDs; default=all")
    return p.parse_args()


def main():
    args   = _parse()
    label  = args.label

    _CONFIG_MAP = {
        "full"          : AblationConfig.full,
        "phase8c"       : AblationConfig.phase8c,
        "eeg_only"      : AblationConfig.eeg_only,
        "eeg_et"        : AblationConfig.eeg_et,
        "eeg_gat"       : AblationConfig.eeg_gat,
        "no_roi"        : AblationConfig.no_roi,
        "no_graph"      : AblationConfig.no_graph,
        "no_contrastive": AblationConfig.no_contrastive,
        "no_mmd"        : AblationConfig.no_mmd,
    }
    ablation = _CONFIG_MAP[args.config]()

    if args.subjects:
        from data.dataset import _normalize_subject_id
        subject_ids = [_normalize_subject_id(s.strip())
                       for s in args.subjects.split(",")]
    else:
        subject_ids = SUBJECT_IDS

    ckpt_dir    = CKPT_DIR    / label
    explain_dir = EXPLAINABILITY_DIR / label
    explain_dir.mkdir(parents=True, exist_ok=True)

    print(f"Explainability pass: {label}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Output:      {explain_dir}")
    print(f"  Subjects:    {len(subject_ids)}")

    # Import after path setup
    from training.train_rdganet import _run_explainability
    _run_explainability(
        subject_ids = subject_ids,
        ablation    = ablation,
        label       = label,
        ckpt_dir    = ckpt_dir,
        explain_dir = explain_dir,
        verbose     = True,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
