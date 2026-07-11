"""
================================================================
INS-HDGS-CMT
Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking
Cognitive Multimodal Transformer
================================================================
Full architecture for Cognitive State Decoding in Neuromarketing.

PRIMARY TASK:   HIGH_ENGAGEMENT vs LOW_ENGAGEMENT
SECONDARY TASKS (optional): Purchase Intent, Visual Attention,
                             Consumer Preference, Cognitive Saliency

Six components (all optional via AblationConfig):
─────────────────────────────────────────────────
  1. SpikingEEGEncoder        — temporal LIF neural dynamics
  2. DynamicGAT               — functional connectivity graph
  3. ETAttentionEncoder        — gaze/fixation visual attention
  4. NeuroFusionTransformer   — cross-modal cognitive fusion
  5. NeuroSymbolicRuleLayer   — interpretable decision rules
  6. ExplainabilityModule     — GradCAM + electrode importance

Forward pass:
─────────────────────────────────────────────────
  eeg_windows   (B, W, C, 5)   → DynamicGAT           → graph_emb  (B, D)
  eeg_raw       (B, C, S)      → SpikingEEGEncoder     → snn_emb    (B, D)
  et_seq        (B, T, 3)      → ETAttentionEncoder    → et_emb     (B, D)
  roi_vector    (B, N_rois)    → ROIAttention gating

  [graph_emb + snn_emb + et_emb] → NeuroFusionTransformer → fused   (B, D)
  fused                          → NeuroSymbolicRuleLayer  → logits  (B, 2)

Loss:
─────────────────────────────────────────────────
  L_total = λ1 L_focal        (engagement classification)
           + λ2 L_contrast    (EEG ↔ ET alignment)
           + λ3 L_roi         (ROI supervision)
           + λ4 L_conn        (graph sparsity)
           + λ5 L_mmd         (cross-subject domain adapt.)
           + λ6 L_rules       (rule diversity + sparsity)

Expected LOSOCV performance (honest, no leakage):
  Accuracy     : 70 – 85%
  F1           : 0.70 – 0.85
  Kappa        : 0.60 – 0.80
  ROC-AUC      : 0.80 – 0.92
================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat_encoder          import GATEncoder
from .temporal_transformer import TemporalTransformer
from .et_encoder           import ETEncoder, ETAttentionEncoder
from .roi_attention        import ROIAttention
from .fusion_attention     import CrossModalFusion
from .contrastive          import NTXentLoss, InfoNCELoss
from .mmd                  import MMDLoss
from .roi_modulation       import ROIGraphModulation
from .dynamic_gat          import DynamicGAT
from .eeg_encoder          import EEGTemporalEncoder
from .fusion_transformer   import NeuroFusionTransformer
from .spiking_encoder      import SpikingEEGEncoder
from .neuro_symbolic       import NeuroSymbolicRuleLayer


# ── Self-supervised EEG-branch transfer (Phase 4) ─────────────────────────────

def load_pretrained_eeg(model, path, verbose: bool = True) -> int:
    """Load self-supervised EEG-branch weights into a freshly built model.

    The checkpoint (written by pretrain_contrastive.py) contains only the
    EEG-branch tensors (dynamic_gat / gat / snn_encoder / snn_band_weights /
    eeg_merge).  Loaded with ``strict=False`` so the rest of the model keeps
    its random init.  Returns the number of tensors successfully transferred.
    """
    sd = torch.load(path, map_location="cpu", weights_only=True)
    incompat = model.load_state_dict(sd, strict=False)
    unexpected = set(incompat.unexpected_keys)
    n_loaded = sum(1 for k in sd if k not in unexpected)
    if verbose:
        print(f"  [pretrained-eeg] transferred {n_loaded}/{len(sd)} EEG-branch "
              f"tensors from {path}"
              + (f"  (unexpected={len(unexpected)})" if unexpected else ""),
              flush=True)
    return n_loaded


# ── Gradient Reversal Layer (DANN) ────────────────────────────────────────────

class _GRLFunction(torch.autograd.Function):
    """Reverses gradients during backward — forces encoder to be subject-invariant."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return _GRLFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha


# ── Ablation Configuration ────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """
    Controls which INS-HDGS-CMT components are active.

    Canonical flags (used by the forward pass)
    -------------------------------------------
    use_snn                : SpikingEEGEncoder — LIF temporal dynamics
    use_graph              : DynamicGAT — functional connectivity graph
    use_temporal           : TemporalTransformer on GAT window sequence
    use_roi                : ROIAttention — embedding-level gating
    use_roi_modulation     : ROIGraphModulation — modulate adj before GAT
    use_et                 : Eye-tracking branch
    use_et_attention       : ETAttentionEncoder (vs BiLSTM)
    use_fusion_transformer : NeuroFusionTransformer — 4-stage cross-modal
    use_neuro_symbolic     : NeuroSymbolicRuleLayer — interpretable rules
    use_contrastive        : EEG↔ET contrastive alignment loss
    use_infonce            : InfoNCE with hard negatives (vs NT-Xent)
    use_mmd                : MMD cross-subject domain adaptation

    Semantic aliases (new names for the same concepts)
    ---------------------------------------------------
    use_dynamic_gnn             → synced with use_graph
    use_cross_modal_transformer → synced with use_fusion_transformer
    use_roi_attention           → synced with use_roi
    use_connectivity_graphs     → synced with use_graph (graph connectivity)

    Backward-compat flags (kept for old RD-GANet code)
    ---------------------------------------------------
    use_eeg, use_gnn, use_transformer, use_dynamic_gat
    """

    # ── Canonical INS-HDGS-CMT flags ─────────────────────────────────────────
    use_snn                : bool = True
    use_graph              : bool = True
    use_temporal           : bool = True
    use_roi                : bool = True
    use_roi_modulation     : bool = True
    use_et                 : bool = True
    use_et_attention       : bool = True
    use_fusion_transformer : bool = True
    use_neuro_symbolic     : bool = True
    use_contrastive        : bool = True
    use_infonce            : bool = True
    use_mmd                : bool = True

    # ── Semantic aliases (user-requested; synced in __post_init__) ───────────
    use_dynamic_gnn             : bool = True   # alias → use_graph
    use_cross_modal_transformer : bool = True   # alias → use_fusion_transformer
    use_roi_attention           : bool = True   # alias → use_roi
    use_connectivity_graphs     : bool = True   # alias → use_graph

    # ── Legacy RD-GANet flags (full set; synced in __post_init__) ────────────
    use_eeg                : bool = True   # always True; EEG is always present
    use_gnn                : bool = True   # alias → use_graph
    use_transformer        : bool = True   # alias → use_fusion_transformer
    use_dynamic_gat        : bool = True   # alias → use_graph

    # EEG temporal encoding aliases (RD-GANet naming)
    use_eeg_temporal_enc   : bool = True   # alias → use_temporal
    use_multiscale_cnn     : bool = True   # standalone CNN front-end flag
    use_temporal_transformer: bool = True  # alias → use_temporal

    # Graph attention alias
    use_graph_attention    : bool = True   # alias → use_graph

    # Fusion / cross-attention alias
    use_cross_attention    : bool = True   # alias → use_fusion_transformer

    # Explainability module flag
    use_explainability     : bool = True   # GradCAM + electrode saliency

    def __post_init__(self):
        """
        Synchronise semantic aliases with canonical flags.

        Rules:
          • Alias overrides canonical if alias was set to False
            (so disabling via alias actually disables the component).
          • Canonical overrides alias when canonical was explicitly disabled.
        """
        # ── Graph aliases → use_graph ─────────────────────────────────────────
        if (not self.use_dynamic_gnn or not self.use_gnn
                or not self.use_dynamic_gat or not self.use_graph_attention
                or not self.use_connectivity_graphs):
            self.use_graph = False
        if not self.use_graph:
            self.use_dynamic_gnn = self.use_gnn = self.use_dynamic_gat = False
            self.use_graph_attention = self.use_connectivity_graphs = False

        # ── Temporal aliases → use_temporal ──────────────────────────────────
        if not self.use_eeg_temporal_enc or not self.use_temporal_transformer:
            self.use_temporal = False
        if not self.use_temporal:
            self.use_eeg_temporal_enc = self.use_temporal_transformer = False

        # ── Fusion aliases → use_fusion_transformer ───────────────────────────
        if (not self.use_cross_modal_transformer or not self.use_transformer
                or not self.use_cross_attention):
            self.use_fusion_transformer = False
        if not self.use_fusion_transformer:
            self.use_cross_modal_transformer = self.use_transformer = False
            self.use_cross_attention = False

        # ── ROI attention alias → use_roi ─────────────────────────────────────
        if not self.use_roi_attention:
            self.use_roi = False
        if not self.use_roi:
            self.use_roi_attention = False

    # ── Factory Methods ───────────────────────────────────────────────────────

    @classmethod
    def full(cls) -> AblationConfig:
        """Complete INS-HDGS-CMT — all components active."""
        return cls(
            use_snn                     = True,
            use_graph                   = True,
            use_temporal                = True,
            use_roi                     = True,
            use_roi_modulation          = True,
            use_et                      = True,
            use_et_attention            = True,
            use_fusion_transformer      = True,
            use_neuro_symbolic          = True,
            use_contrastive             = True,
            use_infonce                 = True,
            use_mmd                     = True,
            # semantic aliases
            use_dynamic_gnn             = True,
            use_cross_modal_transformer = True,
            use_roi_attention           = True,
            use_connectivity_graphs     = True,
            # legacy RD-GANet backward-compat
            use_eeg                     = True,
            use_gnn                     = True,
            use_transformer             = True,
            use_dynamic_gat             = True,
            use_eeg_temporal_enc        = True,
            use_multiscale_cnn          = True,
            use_temporal_transformer    = True,
            use_graph_attention         = True,
            use_cross_attention         = True,
            use_explainability          = True,
        )

    @classmethod
    def eeg_only(cls) -> AblationConfig:
        """Ablation: EEG + SNN + graph only — no ET, no fusion."""
        return cls(
            use_et=False, use_roi=False, use_contrastive=False,
            use_mmd=False, use_roi_modulation=False,
            use_et_attention=False, use_fusion_transformer=False,
        )

    @classmethod
    def no_snn(cls) -> AblationConfig:
        """Ablation: remove SNN; rely on graph + transformer only."""
        return cls(use_snn=False)

    @classmethod
    def no_graph(cls) -> AblationConfig:
        """Ablation: remove graph connectivity (all graph-related flags)."""
        return cls(
            use_graph=False, use_temporal=False, use_roi_modulation=False,
            use_dynamic_gnn=False, use_gnn=False, use_dynamic_gat=False,
            use_connectivity_graphs=False,
        )

    @classmethod
    def no_neuro_symbolic(cls) -> AblationConfig:
        """Ablation: remove rule layer — use standard classification head."""
        return cls(use_neuro_symbolic=False)

    @classmethod
    def no_et(cls) -> AblationConfig:
        """Ablation: EEG + graph only — no eye-tracking."""
        return cls(use_et=False, use_fusion_transformer=False,
                   use_contrastive=False)

    @classmethod
    def no_fusion_transformer(cls) -> AblationConfig:
        """Ablation: replace 4-stage fusion with simple cross-attention."""
        return cls(
            use_fusion_transformer=False,
            use_cross_modal_transformer=False,
            use_transformer=False,
        )

    @classmethod
    def baseline_linear(cls) -> AblationConfig:
        """Baseline: linear EEG encoder — no graph, SNN, ET, or rules."""
        return cls(
            use_snn=False, use_graph=False, use_temporal=False,
            use_roi=False, use_roi_modulation=False, use_et=False,
            use_et_attention=False, use_fusion_transformer=False,
            use_neuro_symbolic=False, use_contrastive=False,
            use_infonce=False, use_mmd=False,
            use_dynamic_gnn=False, use_cross_modal_transformer=False,
            use_roi_attention=False, use_connectivity_graphs=False,
            use_gnn=False, use_transformer=False, use_dynamic_gat=False,
        )

    # ── Legacy factory aliases (kept for old RD-GANet code) ──────────────────

    @classmethod
    def phase8c(cls) -> AblationConfig:
        """Legacy: Phase 8C full model (maps to full())."""
        return cls.full()

    @classmethod
    def eeg_et(cls) -> AblationConfig:
        """Legacy: linear EEG + ET fusion — no GAT."""
        return cls(use_graph=False, use_temporal=False, use_roi=True,
                   use_et=True, use_contrastive=True, use_mmd=True,
                   use_roi_modulation=False, use_dynamic_gnn=False)

    @classmethod
    def eeg_gat(cls) -> AblationConfig:
        """Legacy: EEG + GAT only — no ET."""
        return cls(use_et=False, use_roi=False, use_contrastive=False,
                   use_mmd=False, use_roi_modulation=False,
                   use_graph=True, use_temporal=True)

    @classmethod
    def et_only(cls) -> AblationConfig:
        """Legacy: linear EEG + ET only — no graph."""
        return cls(use_graph=False, use_temporal=False, use_roi=False,
                   use_contrastive=False, use_mmd=False,
                   use_roi_modulation=False, use_dynamic_gnn=False)

    @classmethod
    def no_roi(cls) -> AblationConfig:
        return cls(use_roi=False, use_roi_modulation=False,
                   use_roi_attention=False)

    @classmethod
    def no_contrastive(cls) -> AblationConfig:
        return cls(use_contrastive=False, use_infonce=False)

    @classmethod
    def no_mmd(cls) -> AblationConfig:
        return cls(use_mmd=False)


# ── Linear EEG Fallback ───────────────────────────────────────────────────────

class _LinearEEGEncoder(nn.Module):
    """Ablation baseline when both graph and SNN are disabled."""
    def __init__(self, n_windows, n_ch, in_feat, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_windows * n_ch * in_feat, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, adj=None):
        return self.proj(x)


# ── INS-HDGS-CMT ─────────────────────────────────────────────────────────────

class INS_HDGS_CMT(nn.Module):
    """
    Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking
    Cognitive Multimodal Transformer.

    Parameters
    ----------
    n_eeg_ch        : EEG channel count (graph nodes)
    n_et_ch         : ET feature channels (default 3)
    n_rois          : ROI vector dimension
    n_windows       : temporal windows per epoch (for dynamic graph)
    n_classes       : engagement classes (default 2: LOW/HIGH)
    embed_dim       : universal embedding dimension
    snn_time_steps  : LIF simulation steps
    snn_hidden_dim  : SNN hidden neuron count
    gat_head_dim    : GAT per-head dimension
    gat_heads       : GAT attention heads
    t_nhead         : transformer heads
    t_layers        : transformer depth
    t_ff_dim        : transformer feedforward dim
    et_lstm_hidden  : ET encoder hidden size
    et_lstm_layers  : ET encoder depth
    roi_hidden      : ROI attention hidden size
    fusion_heads    : cross-modal fusion heads
    ns_n_rules      : neuro-symbolic rule count
    ns_hidden_dim   : neuro-symbolic key space dim
    cls_hidden      : classification head hidden size
    dropout         : global dropout rate
    temperature     : contrastive loss temperature
    n_eeg_samples   : raw EEG samples (for SNN)
    feature_names   : optional electrode names for rule explanation
    ablation        : AblationConfig (default: full model)
    """

    def __init__(
        self,
        n_eeg_ch       : int              = 24,
        n_et_ch        : int              = 3,
        n_rois         : int              = 10,
        n_windows      : int              = 10,
        n_classes      : int              = 2,
        embed_dim      : int              = 128,
        snn_time_steps : int              = 10,
        snn_hidden_dim : int              = 128,
        gat_head_dim   : int              = 32,
        gat_heads      : int              = 4,
        t_nhead        : int              = 4,
        t_layers       : int              = 3,
        t_ff_dim       : int              = 512,
        et_lstm_hidden : int              = 64,
        et_lstm_layers : int              = 2,
        roi_hidden     : int              = 64,
        fusion_heads   : int              = 4,
        ns_n_rules     : int              = 8,
        ns_hidden_dim  : int              = 64,
        cls_hidden     : int              = 64,
        dropout        : float            = 0.30,
        temperature    : float            = 0.07,
        hard_neg_weight: float            = 0.50,
        n_eeg_samples  : int              = 1500,
        feature_names  : Optional[List[str]] = None,
        ablation       : Optional[AblationConfig] = None,
    ):
        super().__init__()
        self.n_windows = n_windows
        self.n_eeg_ch  = n_eeg_ch
        self.embed_dim = embed_dim
        self.ablation  = ablation or AblationConfig.full()
        cfg            = self.ablation

        # ── 1. Spiking EEG Encoder ────────────────────────────────────────────
        self.snn_encoder = (
            SpikingEEGEncoder(
                n_channels = n_eeg_ch,
                embed_dim  = embed_dim,
                hidden_dim = snn_hidden_dim,
                n_layers   = 2,
                time_steps = snn_time_steps,
                dropout    = dropout,
            )
            if cfg.use_snn else None
        )
        # Learnable band-importance weights for SNN proxy construction.
        # Init strongly favours theta (idx 1) and alpha (idx 2) — the two
        # bands with the clearest engagement signatures in the literature.
        if cfg.use_snn:
            self.snn_band_weights = nn.Parameter(
                torch.tensor([0.05, 0.40, 0.35, 0.10, 0.10])
            )

        # ── 2. Dynamic Graph Network ──────────────────────────────────────────
        self.roi_mod = (
            ROIGraphModulation(n_rois=n_rois, n_channels=n_eeg_ch,
                               dropout=dropout / 3)
            if cfg.use_roi_modulation else None
        )

        if cfg.use_graph:
            self.dynamic_gat = DynamicGAT(
                in_dim       = 5,
                embed_dim    = embed_dim,
                gat_heads    = gat_heads,
                gat_head_dim = gat_head_dim,
                n_windows    = n_windows,
                t_nhead      = t_nhead,
                t_layers     = t_layers,
                t_ff_dim     = t_ff_dim,
                dropout      = dropout,
            )
            self.gat     = None
            self.temporal = None
        else:
            self.dynamic_gat = None
            self.gat = _LinearEEGEncoder(n_windows, n_eeg_ch, 5, embed_dim)
            self.temporal = None

        # ── 3. ET Attention Encoder ───────────────────────────────────────────
        if cfg.use_et:
            if cfg.use_et_attention:
                self.et_encoder = ETAttentionEncoder(
                    input_dim  = n_et_ch,
                    hidden_dim = et_lstm_hidden,
                    embed_dim  = embed_dim,
                    n_rois     = n_rois,
                    n_heads    = t_nhead,
                    n_layers   = et_lstm_layers,
                    n_windows  = n_windows,
                    dropout    = dropout / 1.5,
                )
            else:
                self.et_encoder = ETEncoder(
                    input_dim  = n_et_ch,
                    hidden_dim = et_lstm_hidden,
                    embed_dim  = embed_dim,
                    num_layers = et_lstm_layers,
                    dropout    = dropout / 1.5,
                )
        else:
            self.et_encoder = None

        # ── ROI Attention (embedding-level gating) ────────────────────────────
        self.roi_attn = (
            ROIAttention(eeg_dim=embed_dim, roi_dim=n_rois,
                         hidden_dim=roi_hidden, dropout=dropout / 3)
            if cfg.use_roi else None
        )

        # ── 4. Cross-Modal Cognitive Transformer ──────────────────────────────
        if cfg.use_et:
            if cfg.use_fusion_transformer:
                self.fusion = NeuroFusionTransformer(
                    embed_dim    = embed_dim,
                    num_heads    = fusion_heads,
                    num_layers   = t_layers,
                    ff_dim       = t_ff_dim,
                    dropout      = dropout / 3,
                    n_modalities = 3,
                )
            else:
                self.fusion = CrossModalFusion(
                    embed_dim = embed_dim,
                    num_heads = fusion_heads,
                    dropout   = dropout / 3,
                )
        else:
            self.fusion = None

        # SNN + graph merge projection (when both active)
        self.eeg_merge = (
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            if (cfg.use_snn and cfg.use_graph) else None
        )

        # ── 5. Neuro-Symbolic Rule Layer ──────────────────────────────────────
        if cfg.use_neuro_symbolic:
            self.rule_layer = NeuroSymbolicRuleLayer(
                embed_dim     = embed_dim,
                n_rules       = ns_n_rules,
                n_classes     = n_classes,
                hidden_dim    = ns_hidden_dim,
                temperature   = 1.0,
                dropout       = dropout / 3,
                feature_names = feature_names,
            )
        else:
            self.rule_layer = None
            # Standard classification head when rule layer is off
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, cls_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(cls_hidden, n_classes),
            )

        # ── Losses ────────────────────────────────────────────────────────────
        if cfg.use_contrastive:
            self.contrast_loss = (
                InfoNCELoss(temperature, hard_neg_weight)
                if cfg.use_infonce else NTXentLoss(temperature)
            )
        else:
            self.contrast_loss = None

        self.mmd_loss       = MMDLoss() if cfg.use_mmd else None
        self.conn_reg_proj  = nn.Parameter(torch.ones(1))

        # ── 7. DANN Subject Classifier (training only) ────────────────────────
        # Parallel branch on `fused`. GRL reverses gradients so the shared
        # encoder is pushed to produce subject-invariant representations.
        # At inference this head is never called — zero pipeline change.
        self.grl = GradientReversalLayer(alpha=1.0)
        self.subject_classifier = nn.Sequential(
            nn.Linear(embed_dim, cls_hidden),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(cls_hidden, 64),   # 64 subject classes is an upper bound
        )

    # ── EEG-only encoder (self-supervised pretraining) ─────────────────────────

    def encode_eeg(
        self,
        eeg_windows  : torch.Tensor,               # (B, W, C, 5)
        adj_matrices : torch.Tensor,               # (B, W, C, C)
    ) -> torch.Tensor:
        """Return the fused EEG embedding ``eeg_emb`` (B, D).

        Replicates exactly Steps 3a/3b/merge of :meth:`forward` (LayerNorm →
        dynamic-GAT graph encoding + SNN proxy → eeg_merge) and nothing else —
        no ET, ROI gating, fusion, or classification.  Used by the contrastive
        pretraining entry-point (pretrain_contrastive.py) to learn a
        subject-invariant EEG representation that is then transferred into the
        full model.  This method is purely additive and never called by
        :meth:`forward`, so it cannot perturb existing train/eval behaviour.
        """
        cfg = self.ablation

        # Step 3a — intra-epoch LayerNorm + dynamic graph attention
        eeg_windows = torch.nn.functional.layer_norm(
            eeg_windows, eeg_windows.shape[-2:]
        )
        graph_emb = None
        if cfg.use_graph and self.dynamic_gat is not None:
            graph_emb, _, _ = self.dynamic_gat(eeg_windows, adj_matrices)
        elif not cfg.use_graph:
            graph_emb = self.gat(eeg_windows, adj_matrices)

        # Step 3b — SNN temporal encoding via band-weighted proxy
        snn_emb = None
        if cfg.use_snn and self.snn_encoder is not None:
            band_w    = torch.softmax(self.snn_band_weights, dim=0)   # (5,)
            raw_proxy = (eeg_windows * band_w).sum(dim=-1).permute(0, 2, 1)
            snn_emb   = self.snn_encoder(raw_proxy)

        # Merge SNN + graph (mirrors forward)
        if (snn_emb is not None and graph_emb is not None
                and self.eeg_merge is not None):
            eeg_emb = self.eeg_merge(torch.cat([graph_emb, snn_emb], dim=-1))
        elif snn_emb is not None:
            eeg_emb = snn_emb
        elif graph_emb is not None:
            eeg_emb = graph_emb
        else:
            raise RuntimeError("No EEG encoder active.")
        return eeg_emb

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        eeg_windows   : torch.Tensor,              # (B, W, C, 5)
        adj_matrices  : torch.Tensor,              # (B, W, C, C)
        et_seq        : torch.Tensor,              # (B, T_et, C_et)
        roi_vector    : torch.Tensor,              # (B, N_rois)
        weighted_adjs : Optional[torch.Tensor] = None,
        eeg_raw       : Optional[torch.Tensor] = None,  # (B, C, S)
    ) -> Dict[str, Any]:
        """
        Returns
        -------
        dict with keys:
          logits          (B, n_classes)
          fused           (B, D)
          eeg_emb         (B, D)
          snn_emb         (B, D) or None
          graph_emb       (B, D) or None
          et_emb          (B, D) or None
          et_roi_attn     (B, N_rois) or None
          roi_logits      (B, N_rois) or None
          gate            (B, D) or None
          rule_info       dict or None
          attn            dict
          gat_attn        dict
        """
        B, W, C, F = eeg_windows.shape
        cfg        = self.ablation

        # ── Step 1: ET Encoding ───────────────────────────────────────────────
        et_emb        = None
        et_roi_attn   = None
        et_window_seq = None
        if cfg.use_et and self.et_encoder is not None:
            if cfg.use_et_attention:
                et_emb, et_roi_attn, et_window_seq = self.et_encoder(et_seq)
            else:
                et_emb = self.et_encoder(et_seq)

        # ── Step 2: ROI Graph Modulation ──────────────────────────────────────
        if cfg.use_roi_modulation and self.roi_mod is not None:
            mod_roi      = et_roi_attn if et_roi_attn is not None else roi_vector
            adj_matrices = self.roi_mod(adj_matrices, mod_roi)

        # ── Step 3a: Dynamic Graph EEG Encoding ──────────────────────────────
        # Normalize band-power features over (C, 5) dimensions per window to
        # remove intra-epoch non-stationarity before graph attention.  Per-
        # subject z-score (dataset.py) removes inter-subject amplitude shift;
        # this LayerNorm removes intra-epoch window-to-window drift.
        eeg_windows = torch.nn.functional.layer_norm(
            eeg_windows, eeg_windows.shape[-2:]
        )
        graph_emb  = None
        gat_attn   = {}
        window_seq = None
        if cfg.use_graph and self.dynamic_gat is not None:
            graph_emb, gat_attn, window_seq = self.dynamic_gat(eeg_windows, adj_matrices)
        elif not cfg.use_graph:
            graph_emb = self.gat(eeg_windows, adj_matrices)

        # ── Step 3b: SNN Temporal EEG Encoding ───────────────────────────────
        snn_emb = None
        if cfg.use_snn and self.snn_encoder is not None:
            if eeg_raw is not None:
                snn_emb = self.snn_encoder(eeg_raw)
            elif graph_emb is not None:
                # Band-weighted proxy: learnable weights favour theta/alpha
                # (B, W, C, 5) → weighted sum over bands → (B, C, W)
                band_w    = torch.softmax(self.snn_band_weights, dim=0)   # (5,)
                raw_proxy = (eeg_windows * band_w).sum(dim=-1).permute(0, 2, 1)
                snn_emb   = self.snn_encoder(raw_proxy)

        # ── Merge SNN + Graph ─────────────────────────────────────────────────
        if (snn_emb is not None and graph_emb is not None
                and self.eeg_merge is not None):
            eeg_emb = self.eeg_merge(
                torch.cat([graph_emb, snn_emb], dim=-1)
            )
        elif snn_emb is not None:
            eeg_emb = snn_emb
        elif graph_emb is not None:
            eeg_emb = graph_emb
        else:
            raise RuntimeError("No EEG encoder active.")

        # ── Step 4: ROI Attention Gating ─────────────────────────────────────
        roi_logits = None
        gate       = None
        if cfg.use_roi and self.roi_attn is not None:
            eeg_emb, roi_logits, gate = self.roi_attn(eeg_emb, roi_vector)

        # ── Step 4: Cross-Modal Fusion ────────────────────────────────────────
        attn_weights = {}
        if cfg.use_et and self.fusion is not None and et_emb is not None:
            if cfg.use_fusion_transformer:
                g_in = graph_emb if graph_emb is not None else eeg_emb
                fused, attn_weights = self.fusion(
                    eeg_emb, g_in, et_emb,
                    graph_seq  = window_seq,
                    et_seq_emb = et_window_seq,
                )
            else:
                fused, attn_weights = self.fusion(
                    eeg_emb.unsqueeze(1), et_emb.unsqueeze(1)
                )
        else:
            fused = eeg_emb

        # ── Step 5: Classification ────────────────────────────────────────────
        rule_info = None
        if cfg.use_neuro_symbolic and self.rule_layer is not None:
            logits, rule_info = self.rule_layer(fused)
        else:
            logits = self.classifier(fused)

        # ── DataParallel-safe return ──────────────────────────────────────────
        B   = logits.size(0)
        dev = logits.device
        D   = self.embed_dim

        assert isinstance(logits, torch.Tensor), "logits must be a Tensor"

        if self.training:
            # ── DANN subject logits (training only — ignored at inference) ────
            # GRL reverses gradients: encoder learns to fool subject classifier
            # → representations become subject-invariant.
            # alpha is scheduled externally via grl.set_alpha().
            dann_logits = self.subject_classifier(self.grl(fused))

            # ── Aux loss (label-free: rule regularisation + contrastive) ─────
            loss_aux = logits.new_zeros([])

            if (cfg.use_neuro_symbolic and self.rule_layer is not None
                    and rule_info is not None):
                loss_aux = loss_aux + self.rule_layer.regularisation_loss(
                    rule_info, lambda_div=0.01, lambda_sparse=0.01
                )
            if (cfg.use_contrastive and self.contrast_loss is not None
                    and eeg_emb is not None and et_emb is not None):
                loss_aux = loss_aux + self.contrast_loss(eeg_emb, et_emb)

            # ── Safe EEG embedding (fallback to zeros if branch inactive) ────
            if eeg_emb is None:
                eeg_emb = logits.new_zeros(B, self.embed_dim)

            # ── Safe ET embedding (fallback to zeros if ET branch inactive) ──
            if et_emb is None:
                et_emb = logits.new_zeros(B, eeg_emb.size(-1))

            # ── Coerce to correct device (defensive; should be no-op) ────────
            eeg_emb  = torch.as_tensor(eeg_emb,  device=dev)
            et_emb   = torch.as_tensor(et_emb,   device=dev)
            loss_aux = torch.as_tensor(loss_aux, device=dev)

            # ── Shape / type assertions ───────────────────────────────────────
            assert torch.is_tensor(logits),   "logits must be a Tensor"
            assert torch.is_tensor(loss_aux), "loss_aux must be a Tensor"
            assert torch.is_tensor(eeg_emb),  "eeg_emb must be a Tensor"
            assert torch.is_tensor(et_emb),   "et_emb must be a Tensor"

            return {
                "logits"      : logits,      # (B, C) — engagement classification scores
                "dann_logits" : dann_logits, # (B, n_subjects) — subject classifier scores
                "loss_aux"    : loss_aux,    # scalar — pre-computed label-free auxiliary loss
                "eeg_emb"     : eeg_emb,    # (B, D) — for MMD / external contrastive
                "et_emb"      : et_emb,     # (B, D) — for MMD / external contrastive
            }

        # ── Eval / Explainability mode ────────────────────────────────────────
        # Full intermediate tensors for analysis. None values are replaced with
        # zero tensors of the correct shape, nested dicts are flattened, so this
        # dict is also gatherable by DataParallel in eval.
        # Infer ROI dimension from actual tensors (avoids sub-module attr lookup)
        if roi_logits is not None:
            n_r = roi_logits.size(-1)
        elif roi_vector is not None:
            n_r = roi_vector.size(-1)
        else:
            n_r = 1
        n_rules = (self.rule_layer.n_rules
                   if self.rule_layer is not None else 1)

        _zD  = lambda: torch.zeros(B, D,      device=dev)
        _zR  = lambda: torch.zeros(B, n_r,    device=dev)
        _zRU = lambda: torch.zeros(B, n_rules, device=dev)

        rule_act = (rule_info["activations"]
                    if rule_info is not None and "activations" in rule_info
                    else _zRU())

        return {
            # Core outputs
            "logits"     : logits,
            "fused"      : fused,
            "loss_aux"   : logits.new_zeros([]),
            # EEG branch (None → zeros)
            "eeg_emb"    : eeg_emb    if eeg_emb    is not None else _zD(),
            "snn_emb"    : snn_emb    if snn_emb    is not None else _zD(),
            "graph_emb"  : graph_emb  if graph_emb  is not None else _zD(),
            # ET branch (None → zeros)
            "et_emb"     : et_emb     if et_emb     is not None else _zD(),
            "et_roi_attn": et_roi_attn if et_roi_attn is not None else _zR(),
            # ROI / gate
            "roi_logits" : roi_logits if roi_logits  is not None else _zR(),
            "gate"       : gate       if gate        is not None else torch.zeros(B, 1, device=dev),
            # Symbolic rules — activations tensor only, no nested dict
            "rule_act"   : rule_act,
            # GAT attention weights (dict of tensors, for explainability)
            "gat_attn"   : gat_attn,
            # Presence flags (float tensors for DataParallel compatibility)
            "has_et"     : logits.new_tensor(float(et_emb     is not None)),
            "has_roi"    : logits.new_tensor(float(roi_logits  is not None)),
            "has_rules"  : logits.new_tensor(float(rule_info   is not None)),
        }

    # ── Loss Computation ──────────────────────────────────────────────────────

    def compute_loss(
        self,
        out              : Dict[str, Any],
        labels           : torch.Tensor,
        roi_vector       : torch.Tensor,
        weighted_adjs    : Optional[torch.Tensor] = None,
        source_emb       : Optional[torch.Tensor] = None,
        target_emb       : Optional[torch.Tensor] = None,
        lambda_cls       : float = 1.00,
        lambda_contrast  : float = 0.30,
        lambda_roi       : float = 0.20,
        lambda_conn      : float = 0.10,
        lambda_mmd       : float = 0.05,
        lambda_rules     : float = 0.02,
        class_weights    : Optional[torch.Tensor] = None,
        focal_alpha      : float = 0.25,
        focal_gamma      : float = 2.00,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-task loss for cognitive engagement decoding.

        L_total = λ1·L_focal  + λ2·L_contrast + λ3·L_roi
                + λ4·L_conn   + λ5·L_mmd      + λ6·L_rules
        """
        cfg    = self.ablation
        dev    = labels.device
        losses = {}

        # ── L_focal: focal loss for engagement classification ─────────────────
        # label_smoothing=0.05 prevents p→0/1 collapse and lowers ECE.
        ce      = F.cross_entropy(out["logits"], labels, weight=class_weights,
                                  reduction="none", label_smoothing=0.05)
        probs   = F.softmax(out["logits"], dim=-1)
        p_t     = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_w = focal_alpha * (1.0 - p_t) ** focal_gamma
        losses["cls"] = (focal_w * ce).mean()

        # ── DataParallel training path: compact dict from self.training branch ──
        # forward() pre-computed contrastive + rule losses into loss_aux.
        # eeg_emb / et_emb are present (with zero fallbacks) for MMD.
        if "has_et" not in out:
            # Training-mode compact path: loss_aux bundles contrastive + rule reg.
            # Must be scaled by lambda_contrast to prevent it dominating cls loss.
            aux = out.get("loss_aux", torch.tensor(0.0, device=dev))
            losses["contrast"]     = aux
            losses["roi"]          = torch.tensor(0.0, device=dev)
            losses["connectivity"] = torch.tensor(0.0, device=dev)
            losses["mmd"]          = torch.tensor(0.0, device=dev)
            losses["rules"]        = torch.tensor(0.0, device=dev)
            losses["total"] = (
                lambda_cls      * losses["cls"]
                + lambda_contrast * aux
            )
            return losses

        # ── Eval / full-dict path: use has_* flags instead of is-None checks ──
        has_et    = out["has_et"].item()    > 0.5
        has_roi   = out["has_roi"].item()   > 0.5
        has_rules = out["has_rules"].item() > 0.5

        # ── L_contrast: EEG ↔ ET embedding alignment ──────────────────────────
        # Skipped here — pre-computed in forward() and included in loss_aux.
        losses["contrast"] = out.get("loss_aux", torch.tensor(0.0, device=dev))

        # ── L_roi: ROI auxiliary supervision ─────────────────────────────────
        if cfg.use_roi and has_roi:
            roi_pred   = F.log_softmax(out["roi_logits"], dim=-1)
            et_roi     = out["et_roi_attn"]
            roi_target = (et_roi if has_et else roi_vector).detach()
            roi_target = roi_target + 1e-10
            roi_target = roi_target / roi_target.sum(dim=-1, keepdim=True)
            losses["roi"] = F.kl_div(roi_pred, roi_target, reduction="batchmean")
        else:
            losses["roi"] = torch.tensor(0.0, device=dev)

        # ── L_conn: graph sparsity regularisation ────────────────────────────
        if weighted_adjs is not None:
            eye     = torch.eye(weighted_adjs.size(-1), device=dev)
            off_d   = weighted_adjs - eye.unsqueeze(0).unsqueeze(0)
            losses["connectivity"] = off_d.pow(2).mean()
        else:
            losses["connectivity"] = torch.tensor(0.0, device=dev)

        # ── L_mmd: cross-subject domain adaptation ───────────────────────────
        if (cfg.use_mmd and self.mmd_loss is not None
                and source_emb is not None and target_emb is not None):
            losses["mmd"] = self.mmd_loss(source_emb, target_emb)
        else:
            losses["mmd"] = torch.tensor(0.0, device=dev)

        # ── L_rules: rule layer regularisation ───────────────────────────────
        # Skipped here — pre-computed in forward() and folded into loss_aux.
        losses["rules"] = torch.tensor(0.0, device=dev)

        # ── Total ──────────────────────────────────────────────────────────────
        losses["total"] = (
            lambda_cls      * losses["cls"]
            + lambda_contrast * losses["contrast"]
            + lambda_roi      * losses["roi"]
            + lambda_conn     * losses["connectivity"]
            + lambda_mmd      * losses["mmd"]
            + lambda_rules    * losses["rules"]
        )

        return losses

    # ── Explainability ────────────────────────────────────────────────────────

    def explain(
        self,
        out           : Dict[str, Any],
        class_names   : Optional[List[str]] = None,
        sample_idx    : int   = 0,
    ) -> Dict[str, Any]:
        """
        Generate symbolic explanations for a batch output.

        Returns
        -------
        dict with:
          rules_text  : list of global rule strings
          sample_text : per-sample activated rule explanation
        """
        cfg  = self.ablation
        out_ = {}

        if cfg.use_neuro_symbolic and self.rule_layer is not None:
            out_["rules_text"]  = self.rule_layer.explain_rules(
                class_names=class_names
            )
            if out.get("rule_info") is not None:
                out_["sample_text"] = self.rule_layer.explain_sample(
                    out["rule_info"],
                    sample_idx  = sample_idx,
                    class_names = class_names,
                )
        else:
            out_["rules_text"]  = ["(Neuro-symbolic layer not active)"]
            out_["sample_text"] = "(Neuro-symbolic layer not active)"

        return out_

    # ── Architecture Summary ──────────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_architecture(self) -> None:
        cfg    = self.ablation
        active = [
            ("SNN EEG Encoder",              cfg.use_snn),
            ("Dynamic GAT (graph)",          cfg.use_graph),
            ("ROI Graph Modulation",         cfg.use_roi_modulation),
            ("ET Attention Encoder",         cfg.use_et_attention),
            ("ROI Attention",                cfg.use_roi),
            ("NeuroFusion Transformer",      cfg.use_fusion_transformer),
            ("Neuro-Symbolic Rule Layer",    cfg.use_neuro_symbolic),
            ("Contrastive (InfoNCE)",        cfg.use_infonce),
            ("MMD Domain Adaptation",        cfg.use_mmd),
        ]
        print("=" * 60)
        print("INS-HDGS-CMT Architecture")
        print("Interpretable Neuro-Symbolic Hybrid Dynamic Graph")
        print("Spiking Cognitive Multimodal Transformer")
        print("=" * 60)
        print("  Task: HIGH_ENGAGEMENT vs LOW_ENGAGEMENT")
        print("─" * 60)
        for name, flag in active:
            status = "  ON " if flag else " off "
            print(f"  [{status}] {name}")
        print(f"\n  Parameters: {self.n_params:,}")
        print("=" * 60)


# ── Backward-compatibility alias ──────────────────────────────────────────────
RDGANet = INS_HDGS_CMT
