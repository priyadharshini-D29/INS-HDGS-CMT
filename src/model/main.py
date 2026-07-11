"""
================================================================
INS-HDGS-CMT — Main Entry Point
Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking
Cognitive Multimodal Transformer
================================================================
Cognitive State Decoding for Neuromarketing Intelligence

Task: HIGH_ENGAGEMENT vs LOW_ENGAGEMENT (from EEG + Eye Tracking)

Usage:
  python main.py                   # full LOSOCV (all subjects)
  python main.py --ablation        # ablation study suite
  python main.py --explain         # explainability reports
  python main.py --subject S02     # single-subject mode
  python main.py --fold-parallel   # multi-GPU fold-parallel
  python main.py --epochs 50       # override epoch count
  python main.py --audit           # leakage audit and exit
  python main.py --sanity          # random-label sanity check
  python main.py --save-labels     # save engagement label reports
  python main.py --no-snn          # ablation: disable SNN
  python main.py --no-ns           # ablation: disable neuro-symbolic

Multi-GPU:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py

Architecture components:
  [1] SpikingEEGEncoder       — LIF temporal dynamics
  [2] DynamicGAT              — functional connectivity graph
  [3] ETAttentionEncoder      — visual attention modeling
  [4] NeuroFusionTransformer  — cross-modal cognitive fusion
  [5] NeuroSymbolicRuleLayer  — interpretable IF-THEN rules
  [6] ExplainabilityModule    — GradCAM + electrode saliency
================================================================
"""

import sys
import os
import warnings
import argparse
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
import io as _io
try:
    if isinstance(sys.stdout, _io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, _io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")
# data/dataset.py imports `model.inference...` as an absolute package
# path, so the *parent* of this directory (not just this directory itself)
# must be importable too -- mirrors the pattern in sensitivity_sweep.py.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

torch.manual_seed(42)
np.random.seed(42)

from config.settings import (
    SUBJECT_IDS, DEVICE, EPOCHS, BATCH_SIZE, ALL_OUTPUT_DIRS,
    PLOTS_DIR, METRICS_DIR, EXPLAINABILITY_DIR, ABLATION_DIR,
    BENCHMARK_DIR, LABELS_DIR, NUM_GPUS, USE_DP, RANDOM_SEED,
    EMBED_DIM, GAT_L1_HEAD_DIM, GAT_L1_HEADS,
    T_NHEAD, T_LAYERS, T_FF_DIM,
    ET_LSTM_HIDDEN, ET_LSTM_LAYERS,
    ROI_HIDDEN_DIM, FUSION_HEADS, CLS_HIDDEN, N_ROIS, N_WINDOWS,
    ET_INPUT_DIM, DROPOUT, TEMPERATURE,
    SNN_TIME_STEPS, SNN_HIDDEN_DIM, NS_N_RULES, NS_HIDDEN_DIM,
    LR, WEIGHT_DECAY, PATIENCE, FOCAL_ALPHA, FOCAL_GAMMA,
    LAMBDA_CLS, LAMBDA_CONTRAST, LAMBDA_ROI, LAMBDA_CONNECTIVITY, LAMBDA_MMD,
    ENGAGEMENT_CLASS_NAMES,
)
try:
    from config.settings import CURRICULUM_LEARNING, CURRICULUM_HARD_SUBJECTS
except ImportError:
    CURRICULUM_LEARNING      = False
    CURRICULUM_HARD_SUBJECTS = []
from data.dataset        import NeumaGraphDataset, build_dataloaders
from models.ins_hdgs_cmt import INS_HDGS_CMT, AblationConfig
from training.losses     import MultiTaskLoss
from training.trainer    import Trainer
from training.metrics    import compute_metrics, print_losocv_summary
from evaluation.losocv   import run_losocv
from evaluation.ablation import run_ablation, ABLATION_CONFIGS
from evaluation.leakage_audit import run_leakage_audit
from explainability.gradcam   import EEGGradCAM
from explainability.saliency  import ElectrodeSaliency
from explainability.topomap   import save_topomap
from visualization.plot_connectivity import (
    plot_connectivity_matrix, plot_dynamic_graphs, plot_electrode_graph
)
from visualization.plot_attention import plot_attention_heatmap, plot_roi_gate
from visualization.plot_results import (
    plot_losocv_results, plot_ablation_comparison,
    plot_confusion_matrix_grid, plot_roc_multiclass
)
from utils.gpu import print_gpu_summary
from sklearn.utils.class_weight import compute_class_weight


# ── Argument Parser ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="INS-HDGS-CMT: Cognitive State Decoding for Neuromarketing"
    )
    p.add_argument("--subject",       type=str,  default=None)
    p.add_argument("--epochs",        type=int,  default=EPOCHS)
    p.add_argument("--batch-size",    type=int,  default=BATCH_SIZE)
    p.add_argument("--fold-parallel", action="store_true")
    p.add_argument("--ablation",      action="store_true")
    p.add_argument("--explain",       action="store_true")
    p.add_argument("--audit",         action="store_true")
    p.add_argument("--sanity",        action="store_true")
    p.add_argument("--save-labels",   action="store_true")
    p.add_argument("--no-snn",        action="store_true")
    p.add_argument("--no-ns",         action="store_true")
    p.add_argument("--label",         type=str, default=None,
                   help="Override experiment label (default: ins_hdgs_cmt)")
    p.add_argument("--alpha-strategy", type=str, default="balanced",
                   choices=["balanced","inv_freq","effective_num","sqrt_inv_freq","none"],
                   help="Class-weight strategy for focal loss (default: balanced)")
    p.add_argument("--focal-gamma",   type=float, default=None,
                   help="Override focal loss gamma (default: FOCAL_GAMMA from settings)")
    p.add_argument("--n-ensemble",    type=int,   default=None,
                   help="Override ensemble size N_ENSEMBLE (default: from settings)")
    p.add_argument("--lambda-dann",   type=float, default=None,
                   help="Override DANN/GRL adversarial weight (default: 0.10). "
                        "Phase-1 subject-invariance sweep.")
    p.add_argument("--lambda-mmd",    type=float, default=None,
                   help="Override MMD domain-alignment weight (default: LAMBDA_MMD). "
                        "Phase-2 distribution-alignment sweep.")
    p.add_argument("--mmd-mode",      type=str, default="marginal",
                   choices=["marginal", "class_conditional"],
                   help="MMD variant: marginal (existing) or class_conditional "
                        "(align HIGH-HIGH, LOW-LOW). Phase-2.")
    p.add_argument("--norm-mode",     type=str, default="zscore",
                   choices=["zscore", "robust", "instance", "layer"],
                   help="Subject-aware feature normalization before graph "
                        "construction. Phase-3.")
    p.add_argument("--pretrained-eeg", type=str, default=None,
                   help="Path to a self-supervised EEG-branch checkpoint "
                        "(pretrain_contrastive.py). Transferred into each fold "
                        "model before fine-tuning. Phase-4.")
    return p.parse_args()


# ── Model Factory ─────────────────────────────────────────────────────────────

def _make_model(
    n_eeg_ch : int,
    n_classes: int,
    ablation : AblationConfig,
    n_et_ch  : int = ET_INPUT_DIM,
    n_rois   : int = N_ROIS,
) -> INS_HDGS_CMT:
    return INS_HDGS_CMT(
        n_eeg_ch       = n_eeg_ch,
        n_et_ch        = n_et_ch,
        n_rois         = n_rois,
        n_windows      = N_WINDOWS,
        n_classes      = n_classes,
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


# ── Phase 8A: Connectivity Visualisation ─────────────────────────────────────

def phase_8a_connectivity(dataset: NeumaGraphDataset, out_dir: Path):
    print("\n── Phase 8A: EEG Connectivity Graph Visualisation ──")
    if len(dataset) == 0:
        return

    sample = dataset[0]
    adj_w  = sample["weighted_adjs"].numpy()

    try:
        plot_connectivity_matrix(adj_w.mean(axis=0),
                                 out_path=out_dir / "connectivity_matrix.png")
        plot_dynamic_graphs(adj_w, out_path=out_dir / "dynamic_graphs.png")
        plot_electrode_graph(adj_w.mean(axis=0),
                             out_path=out_dir / "electrode_graph.png")
    except Exception as e:
        print(f"  Connectivity plot skipped: {e}")

    print(f"  Connectivity plots → {out_dir}")


# ── Phase 8B–D: LOSOCV Training ───────────────────────────────────────────────

def phase_8_losocv(ablation, args, label="ins_hdgs_cmt") -> pd.DataFrame:
    print(f"\n── Phase 8 LOSOCV — {label} ──")

    subject_ids = [args.subject] if args.subject else None

    if subject_ids is None and CURRICULUM_LEARNING:
        easy = [s for s in SUBJECT_IDS if s not in CURRICULUM_HARD_SUBJECTS]
        hard = [s for s in SUBJECT_IDS if s in CURRICULUM_HARD_SUBJECTS]
        subject_ids = easy + hard
        print(f"  [Curriculum] Easy first ({len(easy)}), hard last: {hard}")

    return run_losocv(
        subject_ids          = subject_ids,
        ablation             = ablation,
        epochs               = args.epochs,
        batch_size           = args.batch_size,
        label                = label,
        fold_parallel        = args.fold_parallel,
        verbose              = True,
        alpha_strategy       = getattr(args, "alpha_strategy", "balanced"),
        focal_gamma_override = getattr(args, "focal_gamma",    None),
        n_ensemble_override  = getattr(args, "n_ensemble",     None),
        lambda_dann_override = getattr(args, "lambda_dann",    None),
        lambda_mmd_override  = getattr(args, "lambda_mmd",     None),
        mmd_mode             = getattr(args, "mmd_mode",       "marginal"),
        norm_mode            = getattr(args, "norm_mode",      "zscore"),
        pretrained_eeg       = getattr(args, "pretrained_eeg", None),
    )


# ── Phase 8E: Explainability ──────────────────────────────────────────────────

def phase_8e_explainability(dataset, ablation, out_dir: Path):
    print("\n── Phase 8E: Explainability ──")
    if len(dataset) == 0:
        return

    model = _make_model(dataset.n_eeg_ch, dataset.n_classes, ablation,
                        dataset.n_et_ch).to(DEVICE)
    model.eval()

    sample = dataset[0]
    eeg_w  = sample["eeg_windows"].unsqueeze(0).to(DEVICE)
    adj    = sample["adj_matrices"].unsqueeze(0).to(DEVICE)
    et_seq = sample["et_seq"].unsqueeze(0).to(DEVICE)
    roi_v  = sample["roi_vector"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(eeg_w, adj, et_seq, roi_v)

    class_names = list(ENGAGEMENT_CLASS_NAMES.values())
    explanation = model.explain(out, class_names=class_names)

    print("\n  Global Neuro-Symbolic Rules:")
    for r in explanation["rules_text"]:
        print(f"    {r}")
    print(f"\n{explanation.get('sample_text', '')}")

    rules_path = out_dir / "neuro_symbolic_rules.txt"
    with open(rules_path, "w") as f:
        f.write("INS-HDGS-CMT Neuro-Symbolic Rules\n")
        f.write("=" * 50 + "\n\n")
        for r in explanation["rules_text"]:
            f.write(f"  {r}\n")
        if "sample_text" in explanation:
            f.write(f"\n{explanation['sample_text']}\n")
    print(f"  Rules → {rules_path}")

    # Build a batch dict once — both explainability classes expect this format
    batch_dict = {
        "eeg_windows" : sample["eeg_windows"].to(DEVICE),
        "adj_matrices": sample["adj_matrices"].to(DEVICE),
        "et_seq"      : sample["et_seq"].to(DEVICE),
        "roi_vector"  : sample["roi_vector"].to(DEVICE),
    }
    if "weighted_adjs" in sample:
        batch_dict["weighted_adjs"] = sample["weighted_adjs"].to(DEVICE)

    # EEGGradCAM uses .generate(batch_dict, device=...) not .compute()
    try:
        gc  = EEGGradCAM(model)
        res = gc.generate(batch_dict, device=DEVICE)
        if res.get("cam") is not None:
            np.save(out_dir / "gradcam_heatmap.npy", res["cam"])
        gc.remove_hooks()
    except Exception as e:
        print(f"  EEGGradCAM skipped: {e}")

    # ElectrodeSaliency uses .compute(batch_dict, target_class=None)
    try:
        sal = ElectrodeSaliency(model, device=DEVICE)
        arr = sal.compute(batch_dict)
        np.save(out_dir / "electrode_saliency.npy", arr)
    except Exception as e:
        print(f"  ElectrodeSaliency skipped: {e}")

    try:
        saliency = np.load(out_dir / "electrode_saliency.npy")
        save_topomap(saliency, out_path=out_dir / "electrode_topomap.png")
    except Exception:
        pass

    print(f"  Explainability → {out_dir}")


# ── Phase 9: Ablation ─────────────────────────────────────────────────────────

def phase_9_ablation(args, ablation_out: Path):
    print("\n── Phase 9: Ablation Study ──")
    abl_results = run_ablation(
        epochs     = min(args.epochs, 30),
        batch_size = args.batch_size,
        verbose    = True,
    )
    if abl_results:
        abl_df = pd.concat(abl_results.values(), ignore_index=True)
        abl_df.to_csv(ablation_out / "ablation_results.csv", index=False)
        try:
            plot_ablation_comparison(abl_df,
                                     out_path=ablation_out / "ablation_comparison.png")
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    print_gpu_summary()
    print(f"\n  INS-HDGS-CMT | Task: HIGH vs LOW ENGAGEMENT | "
          f"GPU: {DEVICE} | Subject: {args.subject or 'all'}")

    # ── Ablation config ───────────────────────────────────────────────────
    ablation = (
        AblationConfig.no_snn()              if args.no_snn and not args.no_ns
        else AblationConfig.no_neuro_symbolic() if args.no_ns and not args.no_snn
        else AblationConfig.full()
    )

    if args.audit:
        report = run_leakage_audit(
            subject_ids=[args.subject] if args.subject else None,
            verbose=True,
        )
        print(f"  Leakage risk: {'DETECTED' if report.get('has_risk') else 'NONE'}")
        return

    if args.sanity:
        print("\n── Sanity Check: Random Labels (expect ~50%%) ──")
        df = run_losocv(
            subject_ids=[args.subject] if args.subject else None,
            epochs=min(args.epochs, 10),
            batch_size=args.batch_size,
            label="sanity_random_labels",
            random_labels=True,
        )
        if not df.empty:
            print(f"  Mean accuracy: {df['accuracy'].mean():.4f}")
        return

    # ── Load compact dataset for Phase 8A / 8E (no full precompute) ───────
    vis_ds = None
    try:
        vis_sids = [args.subject] if args.subject else SUBJECT_IDS[:3]
        vis_ds   = NeumaGraphDataset(
            subject_ids   = vis_sids,
            save_labels   = args.save_labels,
            label_out_dir = LABELS_DIR,
            precompute_graphs = False,
        )
        print(f"\n  Loaded {len(vis_ds)} epochs from {vis_ds.unique_subjects}")
    except FileNotFoundError as e:
        print(f"  [WARN] Could not load dataset: {e}")

    # ── Phase 8A: Connectivity vis ────────────────────────────────────────
    if vis_ds is not None and len(vis_ds) > 0:
        try:
            pre_ds = NeumaGraphDataset(
                subject_ids=vis_ds.unique_subjects,
                precompute_graphs=True,
            )
            phase_8a_connectivity(pre_ds, PLOTS_DIR)
        except Exception as e:
            print(f"  Phase 8A skipped: {e}")

    # ── Phase 8B–D: LOSOCV ───────────────────────────────────────────────
    label = args.label or "ins_hdgs_cmt"
    if args.no_snn:
        label += "_no_snn"
    if args.no_ns:
        label += "_no_ns"

    results_df = phase_8_losocv(ablation, args, label=label)

    if not results_df.empty:
        print_losocv_summary(results_df, label)
        label_plot_dir = PLOTS_DIR / label
        label_plot_dir.mkdir(parents=True, exist_ok=True)
        for fn, path in [
            (plot_losocv_results,
             label_plot_dir / f"losocv_{label}.png"),
            (plot_roc_multiclass,
             label_plot_dir / f"roc_{label}.png"),
        ]:
            try:
                fn(results_df, save_path=path)
            except Exception:
                pass
        try:
            plot_confusion_matrix_grid(
                results_df,
                class_names=list(ENGAGEMENT_CLASS_NAMES.values()),
                save_path=label_plot_dir / f"confusion_grid_{label}.png",
            )
        except Exception:
            pass

    # ── Phase 8E: Explainability ──────────────────────────────────────────
    if args.explain and vis_ds is not None:
        try:
            if not hasattr(vis_ds, '_graph_cache') or len(vis_ds._graph_cache) == 0:
                vis_ds_pre = NeumaGraphDataset(
                    subject_ids=vis_ds.unique_subjects,
                    precompute_graphs=True,
                )
            else:
                vis_ds_pre = vis_ds
            phase_8e_explainability(vis_ds_pre, ablation, EXPLAINABILITY_DIR)
        except Exception as e:
            print(f"  Phase 8E skipped: {e}")

    # ── Phase 9: Ablation ─────────────────────────────────────────────────
    if args.ablation:
        try:
            phase_9_ablation(args, ABLATION_DIR)
        except Exception as e:
            print(f"  Phase 9 skipped: {e}")

    print(f"\n  Outputs → {Path('output').resolve()}")
    print("  INS-HDGS-CMT pipeline complete.")


if __name__ == "__main__":
    main()
