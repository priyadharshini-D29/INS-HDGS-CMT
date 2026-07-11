"""
================================================================
NEUMA Phase 8C — RD-GANet: Full Model
ROI-Guided Dynamic Graph Attention NeuroFusion Network
================================================================
Complete forward pass (Phase 8C):

  0. ETAttentionEncoder (opt):  (B, T_et, C) → emb + roi_attn
  1. ROIGraphModulation (opt):  A' = A ⊙ (1 + α · roi_i · roi_j)
  2. EEG Encoding:
       a. DynamicGAT   (opt):  (B, W, C, 5) + A' → (B, D)
       b. GATEncoder + TemporalTransformer (default)
       c. EEGTemporalEncoder (opt, needs raw EEG)
  3. ROIAttention (opt):        eeg_emb ⊙ σ(W_r r + b_r) → (B, D)
  4. Cross-Modal Fusion:
       a. NeuroFusionTransformer (opt): EEG + Graph + ET → (B, D)
       b. CrossModalFusion (default):  Attention(EEG, ET) → (B, D)
  5. Classifier:                (B, D) → (B, N_classes)

Ablation flags in AblationConfig control all components.

Loss:
  L = λ1 L_cls + λ2 L_contrast + λ3 L_roi + λ4 L_conn + λ5 L_mmd
================================================================
"""

from __future__ import annotations
from models.et_temporal_encoder import ETTemporalEncoder
from typing import Optional, Dict, Any

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

# ── AblationConfig: single canonical definition lives in ins_hdgs_cmt ─────────
# All code that previously imported AblationConfig from rd_ganet will now get
# the same class as code importing from ins_hdgs_cmt, eliminating divergence
# and the AttributeError on missing flags (use_snn, use_dynamic_gnn, etc.).
from .ins_hdgs_cmt import AblationConfig  # noqa: F401  (re-export)


# ── Fallback Linear EEG Encoder ──────────────────────────────────────────────

class _LinearEEGEncoder(nn.Module):
    """Ablation baseline: replace GAT+Transformer with a simple linear layer."""
    def __init__(self, n_windows: int, n_ch: int, in_feat: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_windows * n_ch * in_feat, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor, adj=None) -> torch.Tensor:
        return self.proj(x)


# ── RD-GANet ──────────────────────────────────────────────────────────────────

class RDGANet(nn.Module):
    """
    ROI-Guided Dynamic Graph Attention NeuroFusion Network.

    Phase 8C extends the original model with six new modules, all optional
    via AblationConfig.  Use AblationConfig.phase8c() for the full Phase 8C
    configuration and AblationConfig.full() for the original Phase 8 model.

    Parameters
    ----------
    n_eeg_ch      : EEG channel count (graph nodes)
    n_et_ch       : ET feature channels (default 3: gaze_x, gaze_y, pupil)
    n_rois        : ROI vector dimension
    n_windows     : temporal windows per epoch
    n_classes     : classification output size
    embed_dim     : universal embedding dimension
    gat_head_dim  : per-head dimension in GAT layer 1
    gat_heads     : attention heads in GAT
    t_nhead       : heads in TemporalTransformer / DynamicGAT temporal encoder
    t_layers      : TemporalTransformer / DynamicGAT temporal encoder depth
    t_ff_dim      : feedforward dimension in temporal modules
    et_lstm_hidden: hidden size for BiLSTM ETEncoder
    et_lstm_layers: layers for BiLSTM ETEncoder
    roi_hidden    : ROIAttention hidden dimension
    fusion_heads  : attention heads in fusion module
    cls_hidden    : classifier hidden size
    dropout       : global dropout rate
    temperature   : contrastive loss temperature τ
    hard_neg_weight: InfoNCE hard-negative mixing weight (Phase 8C)
    n_eeg_samples : raw EEG samples per epoch for EEGTemporalEncoder (Phase 8C)
    ablation      : AblationConfig instance (default: full Phase 8 model)
    """

    def __init__(
        self,
        n_eeg_ch       : int            = 24,
        n_et_ch        : int            = 3,
        n_rois         : int            = 10,
        n_windows      : int            = 10,
        n_classes      : int            = 6,
        embed_dim      : int            = 128,
        gat_head_dim   : int            = 32,
        gat_heads      : int            = 4,
        t_nhead        : int            = 4,
        t_layers       : int            = 2,
        t_ff_dim       : int            = 256,
        et_lstm_hidden : int            = 64,
        et_lstm_layers : int            = 2,
        roi_hidden     : int            = 64,
        fusion_heads   : int            = 4,
        cls_hidden     : int            = 64,
        dropout        : float          = 0.30,
        temperature    : float          = 0.07,
        hard_neg_weight: float          = 0.5,
        n_eeg_samples  : int            = 1500,
        ablation       : AblationConfig = None,
    ):
        super().__init__()

        self.n_windows = n_windows
        self.n_eeg_ch  = n_eeg_ch
        self.embed_dim = embed_dim
        self.ablation  = ablation or AblationConfig.full()

        cfg = self.ablation

        # ── Phase 8C: ROI Graph Modulation ────────────────────────────────────
        self.roi_mod = (
            ROIGraphModulation(
                n_rois     = n_rois,
                n_channels = n_eeg_ch,
                dropout    = dropout / 3,
            )
            if cfg.use_roi_modulation else None
        )

        # ── EEG Encoding ──────────────────────────────────────────────────────
        if cfg.use_graph:
            if cfg.use_dynamic_gat:
                # Phase 8C: DynamicGAT handles GAT + temporal aggregation in one module
                self.gat         = None
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
                self.temporal = None
            else:
                # Original: separate GATEncoder + TemporalTransformer
                self.gat = GATEncoder(
                    in_dim   = 5,
                    head_dim = gat_head_dim,
                    heads    = gat_heads,
                    out_dim  = embed_dim,
                    dropout  = dropout,
                )
                self.dynamic_gat = None
                self.temporal = (
                    TemporalTransformer(
                        d_model    = embed_dim,
                        nhead      = t_nhead,
                        num_layers = t_layers,
                        dim_ff     = t_ff_dim,
                        dropout    = dropout / 3,
                    )
                    if cfg.use_temporal else None
                )
        else:
            # Ablation baseline: flat linear encoder
            self.gat         = _LinearEEGEncoder(n_windows, n_eeg_ch, 5, embed_dim)
            self.dynamic_gat = None
            self.temporal    = None

        # ── Phase 8C: Multi-Scale EEG Temporal Encoder ───────────────────────
        # Processes raw EEG (B, C, S) → (B, D) as an additional EEG stream.
        # Requires eeg_raw to be passed to forward(); disabled when eeg_raw is None.
        self.eeg_temporal_enc = (
            EEGTemporalEncoder(
                n_channels = n_eeg_ch,
                embed_dim  = embed_dim,
                n_samples  = n_eeg_samples,
                dropout    = dropout,
            )
            if cfg.use_eeg_temporal_enc else None
        )
        # ── ET Encoding ───────────────────────────────────────────────────────
        if cfg.use_et:
            self.et_encoder = ETTemporalEncoder(
                input_dim = n_et_ch
            )

        else:
            self.et_encoder = None
        # ── ROI Attention (embedding-level gating, original Phase 8) ─────────
        self.roi_attn = (
            ROIAttention(
                eeg_dim    = embed_dim,
                roi_dim    = n_rois,
                hidden_dim = roi_hidden,
                dropout    = dropout / 3,
            )
            if cfg.use_roi else None
        )

        # ── Cross-Modal Fusion ────────────────────────────────────────────────
        if cfg.use_et:
            if cfg.use_fusion_transformer:
                # Phase 8C: 4-stage cross-modal fusion transformer
                self.fusion = NeuroFusionTransformer(
                    embed_dim    = embed_dim,
                    num_heads    = fusion_heads,
                    num_layers   = t_layers,
                    ff_dim       = t_ff_dim,
                    dropout      = dropout / 3,
                    n_modalities = 3,
                )
            else:
                # Original: single cross-attention fusion
                self.fusion = CrossModalFusion(
                    embed_dim  = embed_dim,
                    num_heads  = fusion_heads,
                    dropout    = dropout / 3,
                )
            self.eeg_only_proj = None
        else:
            self.fusion        = None
            self.eeg_only_proj = nn.Identity()

        # ── Classification Head ───────────────────────────────────────────────
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
                if cfg.use_infonce
                else NTXentLoss(temperature)
            )
        else:
            self.contrast_loss = None

        self.mmd_loss      = MMDLoss() if cfg.use_mmd else None
        self.conn_reg_proj = nn.Parameter(torch.ones(1))

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        eeg_windows   : torch.Tensor,
        adj_matrices  : torch.Tensor,
        et_seq        : torch.Tensor,
        roi_vector    : torch.Tensor,
        weighted_adjs : Optional[torch.Tensor] = None,
        eeg_raw       : Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        eeg_windows   : (B, W, C, 5)     band-power node features
        adj_matrices  : (B, W, C, C)     adjacency matrices per window
        et_seq        : (B, T_et, C_et)  raw ET sequence
        roi_vector    : (B, N_rois)      dwell-time ROI vector
        weighted_adjs : (B, W, C, C)     optional float adj for L_conn
        eeg_raw       : (B, C, S)        raw EEG epoch for EEGTemporalEncoder
                                         (required only if use_eeg_temporal_enc=True)

        Returns
        -------
        dict with keys:
          logits         : (B, N_classes)
          eeg_emb        : (B, D)   EEG embedding going into fusion
          et_emb         : (B, D) or None
          et_roi_attn    : (B, N_rois) or None   from ETAttentionEncoder
          eeg_temporal_emb: (B, D) or None        from EEGTemporalEncoder
          fused          : (B, D)   fused embedding used for classification
          roi_logits     : (B, N_rois) or None    from ROIAttention
          gate           : (B, D) or None         from ROIAttention
          attn           : dict    fusion attention weights
          gat_attn       : dict    GAT layer attention weights
        """
        B, W, C, F = eeg_windows.shape
        cfg = self.ablation

        # ── Step 1: ET Encoding ───────────────────────────────────────────────
        # Run ET first so ETAttentionEncoder's roi_attn can guide adjacency
        # modulation in the next step.
        et_emb      = None
        et_roi_attn = None

        if cfg.use_et and self.et_encoder is not None:
            # ---------------------------------------------------------
            # ET Encoder
            # ---------------------------------------------------------
            et_out = self.et_encoder(et_seq)

            # ET encoder returns dictionary
            if isinstance(et_out, dict):
                et_emb = et_out["et_emb"]
                # Temporal attention exists but is NOT ROI attention
                # Shape: (B, 600) — ROI modulation expects (B, N_ROIS=10)
                et_roi_attn = None
            else:
                # fallback safety
                et_emb = et_out
                et_roi_attn = None

        # ── Step 2: ROI Graph Modulation (Phase 8C) ───────────────────────────
        # Modulate adjacency using the ROI attention vector.
        # Prefer learned et_roi_attn when available; fall back to roi_vector.
        if cfg.use_roi_modulation and self.roi_mod is not None:
            mod_roi      = et_roi_attn if et_roi_attn is not None else roi_vector
            adj_matrices = self.roi_mod(adj_matrices, mod_roi)   # (B, W, C, C)

        # ── Step 3: EEG Encoding ─────────────────────────────────────────────
        gat_attn = {}

        if cfg.use_graph:
            if cfg.use_dynamic_gat and self.dynamic_gat is not None:
                # Phase 8C: DynamicGAT handles windowed graphs + temporal agg
                graph_emb, gat_attn, _window_seq = self.dynamic_gat(eeg_windows, adj_matrices)
            else:
                # Original: vectorised GATEncoder over all windows
                x_bw   = eeg_windows.view(B * W, C, F)
                adj_bw = adj_matrices.view(B * W, C, C)

                node_emb, gat_attn = self.gat(x_bw, adj_bw)         # (B*W, C, D)
                pooled   = node_emb.mean(dim=1)                       # (B*W, D)
                eeg_seq  = pooled.view(B, W, self.embed_dim)          # (B, W, D)

                if self.temporal is not None:
                    eeg_seq = self.temporal(eeg_seq)                  # (B, W, D)

                graph_emb = eeg_seq.mean(dim=1)                       # (B, D)
        else:
            graph_emb = self.gat(eeg_windows, adj_matrices)           # (B, D)

        # ── Step 3b: EEG Temporal Encoder (Phase 8C, optional extra stream) ──
        # Processes raw EEG time series in parallel with the graph path.
        eeg_temporal_emb = None
        if (cfg.use_eeg_temporal_enc
                and self.eeg_temporal_enc is not None
                and eeg_raw is not None):
            eeg_temporal_emb = self.eeg_temporal_enc(eeg_raw)         # (B, D)

        # For fusion and downstream use:
        #   graph_emb    — connectivity/topology embedding from GAT path
        #   eeg_emb      — temporal EEG embedding (temporal CNN if active, else graph_emb)
        eeg_emb = eeg_temporal_emb if eeg_temporal_emb is not None else graph_emb

        # ── Step 4: ROI Attention (embedding-level gating, original Phase 8) ─
        roi_logits = None
        gate       = None

        if cfg.use_roi and self.roi_attn is not None:
            eeg_emb, roi_logits, gate = self.roi_attn(eeg_emb, roi_vector)

        # ── Step 5: Cross-Modal Fusion ────────────────────────────────────────
        attn_weights = {}

        if cfg.use_et and self.fusion is not None and et_emb is not None:
            if cfg.use_fusion_transformer:
                # Phase 8C: 4-stage NeuroFusionTransformer
                # eeg_emb  = temporal CNN embedding (or graph_emb if unavailable)
                # graph_emb = GAT/DynamicGAT connectivity embedding
                # et_emb    = ET embedding
                fused, attn_weights = self.fusion(eeg_emb, graph_emb, et_emb)
            else:
                # Original: cross-attention between EEG and ET tokens
                fused, attn_weights = self.fusion(
                    eeg_emb.unsqueeze(1),    # (B, 1, D)
                    et_emb.unsqueeze(1),     # (B, 1, D)
                )                            # fused: (B, D)
        else:
            fused = eeg_emb

        # ── Step 6: Classification ────────────────────────────────────────────
        logits = self.classifier(fused)                               # (B, N_classes)

        return {
            "logits"          : logits,
            "eeg_emb"         : eeg_emb,
            "et_emb"          : et_emb,
            "et_roi_attn"     : et_roi_attn,
            "eeg_temporal_emb": eeg_temporal_emb,
            "graph_emb"       : graph_emb,
            "fused"           : fused,
            "roi_logits"      : roi_logits,
            "gate"            : gate,
            "attn"            : attn_weights,
            "gat_attn"        : gat_attn,
        }

    # ── Loss Computation ──────────────────────────────────────────────────────

    def compute_loss(
        self,
        out             : Dict[str, Any],
        labels          : torch.Tensor,
        roi_vector      : torch.Tensor,
        weighted_adjs   : Optional[torch.Tensor] = None,
        source_emb      : Optional[torch.Tensor] = None,
        target_emb      : Optional[torch.Tensor] = None,
        lambda_cls      : float = 1.00,
        lambda_contrast : float = 0.30,
        lambda_roi      : float = 0.20,
        lambda_conn     : float = 0.10,
        lambda_mmd      : float = 0.05,
        class_weights   : Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-task loss:
          L_total = λ1 L_cls + λ2 L_contrast + λ3 L_roi + λ4 L_conn + λ5 L_mmd

        class_weights: optional (N_classes,) tensor for weighted cross-entropy
          (handles class imbalance).

        Returns dict of individual and total losses.
        """
        cfg    = self.ablation
        losses = {}
        dev    = labels.device

        # ── L_cls: weighted cross-entropy classification ──────────────────────
        losses["cls"] = F.cross_entropy(out["logits"], labels, weight=class_weights)

        # ── L_contrast: EEG ↔ ET embedding alignment ─────────────────────────
        if (cfg.use_contrastive
                and self.contrast_loss is not None
                and out["et_emb"] is not None):
            losses["contrast"] = self.contrast_loss(out["eeg_emb"], out["et_emb"])
        else:
            losses["contrast"] = torch.tensor(0.0, device=dev)

        # ── L_roi: auxiliary ROI supervision ─────────────────────────────────
        # Use et_roi_attn (Phase 8C learned distribution) when available;
        # otherwise use raw dwell-time roi_vector.
        if cfg.use_roi and out["roi_logits"] is not None:
            roi_pred = F.log_softmax(out["roi_logits"], dim=-1)

            # Choose supervision signal
            if out.get("et_roi_attn") is not None:
                # Phase 8C: supervise against ETAttentionEncoder's attention dist
                roi_target = out["et_roi_attn"].detach()
            else:
                roi_target = roi_vector

            roi_target = roi_target + 1e-10
            roi_target = roi_target / roi_target.sum(dim=-1, keepdim=True)
            losses["roi"] = F.kl_div(roi_pred, roi_target, reduction="batchmean")
        else:
            losses["roi"] = torch.tensor(0.0, device=dev)

        # ── L_conn: Frobenius sparsity regularisation on connectivity ────────
        if weighted_adjs is not None:
            off_diag = weighted_adjs - torch.eye(
                weighted_adjs.size(-1), device=dev
            ).unsqueeze(0).unsqueeze(0)
            losses["connectivity"] = off_diag.pow(2).mean()
        else:
            losses["connectivity"] = torch.tensor(0.0, device=dev)

        # ── L_mmd: cross-subject domain adaptation ───────────────────────────
        if (cfg.use_mmd
                and self.mmd_loss is not None
                and source_emb is not None
                and target_emb is not None):
            losses["mmd"] = self.mmd_loss(source_emb, target_emb)
        else:
            losses["mmd"] = torch.tensor(0.0, device=dev)

        # ── Total ─────────────────────────────────────────────────────────────
        losses["total"] = (
            lambda_cls      * losses["cls"]
            + lambda_contrast * losses["contrast"]
            + lambda_roi      * losses["roi"]
            + lambda_conn     * losses["connectivity"]
            + lambda_mmd      * losses["mmd"]
        )

        return losses

        # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def n_params(self) -> int:

        total = 0

        for p in self.parameters():

            try:
                total += p.numel()

            except ValueError:
                # Skip uninitialized LazyLinear params
                continue

        return total


    def print_architecture(self) -> None:

        cfg = self.ablation

        active = [

            ("ROI Graph Modulation",   cfg.use_roi_modulation),
            ("Dynamic GAT",            cfg.use_dynamic_gat),
            ("EEG Temporal Encoder",   cfg.use_eeg_temporal_enc),
            ("ET Attention Encoder",   cfg.use_et_attention),
            ("Fusion Transformer",     cfg.use_fusion_transformer),
            ("InfoNCE loss",           cfg.use_infonce),
            ("ROI Attention",          cfg.use_roi),
            ("ET stream",              cfg.use_et),
            ("Contrastive loss",       cfg.use_contrastive),
            ("MMD loss",               cfg.use_mmd),

        ]

        print("=" * 50)
        print("RD-GANet Architecture")
        print("=" * 50)

        for name, flag in active:

            status = "ON " if flag else "off"

            print(f"  [{status}] {name}")

        print(f"\n  Parameters: {self.n_params:,}")

        print("=" * 50)