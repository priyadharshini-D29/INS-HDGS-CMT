"""
================================================================
INS-HDGS-CMT — Neuro-Symbolic Interpretable Rule Layer
================================================================
Implements a differentiable rule-learning layer that produces
human-readable IF-THEN rules for cognitive engagement decoding.

Scientific motivation
---------------------
Neural networks are black boxes. For neuromarketing applications,
practitioners need explanations like:

  IF frontal_theta ↑ AND fixation_duration ↑ AND pupil_dilation ↑
  THEN purchase_intent = HIGH  (confidence: 0.89)

This module learns such rules in a fully differentiable manner,
enabling end-to-end training while preserving interpretability.

Architecture
------------

  Rule Premise Network:
    Each rule r has a soft premise vector p_r ∈ R^D learned
    via attention over the fused embedding.

  Rule Activation:
    a_r = softmax(p_r · z / √D)     rule-to-embedding similarity
    Each rule fires proportionally to its premise match.

  Rule Conclusion:
    Each rule produces a class distribution via a small linear head.

  Rule Aggregation:
    Final logits = Σ_r a_r · logits_r

  Interpretability:
    Rule premises are decoded to feature names via the highest-
    activation dimensions. Human-readable rule strings are
    generated at inference time.

Training regularisation
-----------------------
  • Diversity loss:  push rule premises to be distinct
  • Sparsity loss:   encourage few rules to fire per input
  • Entropy loss:    push rule activations to be confident

Reference
---------
Shao et al. (2021) "Concept Bottleneck Models"
Rudin et al. (2022) "Interpretable Machine Learning"
================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Rule Layer ────────────────────────────────────────────────────────────────

class NeuroSymbolicRuleLayer(nn.Module):
    """
    Differentiable neuro-symbolic rule layer for engagement classification.

    Parameters
    ----------
    embed_dim   : input embedding dimension (fused representation)
    n_rules     : number of learnable rules
    n_classes   : classification output classes
    hidden_dim  : rule premise projection dimension
    temperature : softmax temperature for rule activation
    dropout     : dropout on rule premise queries
    feature_names: optional list of feature/electrode names for explanation
    """

    def __init__(
        self,
        embed_dim     : int,
        n_rules       : int   = 8,
        n_classes     : int   = 2,
        hidden_dim    : int   = 64,
        temperature   : float = 1.0,
        dropout       : float = 0.10,
        feature_names : Optional[List[str]] = None,
        alpha_mode    : str   = "learned",
    ):
        """
        alpha_mode controls the bypass gate of Eq. (18):
          "learned"     : alpha = sigmoid(alpha_0) is a trainable parameter
                          (default; the configuration reported in the paper).
          "rule_only"   : alpha is fixed at 0 — the prediction is the soft-rule
                          evidence R alone (Reviewer 2, comment 5). The bypass
                          classifier is still constructed so checkpoints stay
                          key-compatible, but it receives no gradient.
          "bypass_only" : alpha fixed at 1 — plain classifier, rules are
                          computed for inspection only (explanation-only).
        """
        super().__init__()
        if alpha_mode not in ("learned", "rule_only", "bypass_only"):
            raise ValueError(f"alpha_mode must be learned|rule_only|bypass_only, got {alpha_mode!r}")
        self.embed_dim    = embed_dim
        self.n_rules      = n_rules
        self.n_classes    = n_classes
        self.temperature  = temperature
        self.feature_names = feature_names
        self.alpha_mode   = alpha_mode

        # Rule premise embeddings (each rule is a query vector)
        self.rule_queries = nn.Parameter(
            torch.randn(n_rules, hidden_dim) * 0.02
        )

        # Project fused embedding to rule-query space
        self.key_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-rule classification heads (small — one linear layer each)
        self.rule_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_classes)
            for _ in range(n_rules)
        ])

        # Global bypass: direct classifier (acts as ensemble stabiliser)
        self.bypass = nn.Linear(embed_dim, n_classes)

        # Bypass mixing weight (learned)
        self.bypass_alpha = nn.Parameter(torch.tensor(0.30))

    def forward(
        self,
        z: torch.Tensor,                       # (B, embed_dim)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Parameters
        ----------
        z : (B, embed_dim)  fused cognitive embedding

        Returns
        -------
        logits      : (B, n_classes)    ensemble classification
        rule_info   : dict with keys:
            activations  (B, n_rules)   per-rule activation weights
            rule_logits  (B, n_rules, n_classes)
            keys         (B, hidden_dim)
        """
        B = z.shape[0]

        # Project embedding → rule-key space
        keys = self.key_proj(z)                            # (B, hidden_dim)

        # Rule activation: similarity between each rule query and the key
        # queries : (n_rules, hidden_dim)
        # keys    : (B, hidden_dim)
        sim  = torch.einsum("rd,bd->br", self.rule_queries, keys)   # (B, n_rules)
        sim  = sim / (self.embed_dim ** 0.5)
        acts = F.softmax(sim / self.temperature, dim=-1)             # (B, n_rules)

        # Per-rule logits
        rule_logits = torch.stack(
            [head(keys) for head in self.rule_heads], dim=1
        )                                                            # (B, n_rules, C)

        # Weighted aggregation
        agg = (acts.unsqueeze(-1) * rule_logits).sum(dim=1)         # (B, C)

        # Mix with bypass
        alpha  = self.effective_alpha()
        bypass_logits = self.bypass(z)                               # (B, C)
        logits = alpha * bypass_logits + (1.0 - alpha) * agg         # (B, C)

        rule_info = {
            "activations"  : acts,                   # (B, n_rules)
            "rule_logits"  : rule_logits,            # (B, n_rules, C)
            "keys"         : keys,                   # (B, hidden_dim)
            "rule_evidence": agg,                    # (B, C)  R of Eq. (17)
            "bypass_logits": bypass_logits,          # (B, C)  W_b z + b_b
            "bypass_alpha" : float(alpha.item()),
        }
        return logits, rule_info

    def effective_alpha(self) -> torch.Tensor:
        """Bypass weight actually used by forward(), honouring alpha_mode."""
        if self.alpha_mode == "rule_only":
            return torch.zeros((), device=self.bypass_alpha.device)
        if self.alpha_mode == "bypass_only":
            return torch.ones((), device=self.bypass_alpha.device)
        return torch.sigmoid(self.bypass_alpha)

    def regularisation_loss(
        self,
        rule_info    : Dict,
        lambda_div   : float = 0.01,
        lambda_sparse: float = 0.01,
    ) -> torch.Tensor:
        """
        Auxiliary loss for rule quality:
          L_div    : cosine dissimilarity between rule premises → diversity
          L_sparse : entropy of rule activations → sparsity
        """
        dev = self.rule_queries.device

        # Diversity: pairwise cosine similarity between premises
        q_n = F.normalize(self.rule_queries, dim=-1)               # (R, D)
        sim = q_n @ q_n.T                                          # (R, R)
        mask = ~torch.eye(self.n_rules, dtype=torch.bool, device=dev)
        l_div = sim[mask].pow(2).mean()

        # Sparsity: entropy of per-sample rule activations (want low entropy)
        acts  = rule_info.get("activations")
        if acts is not None:
            l_sparse = -(acts * (acts + 1e-10).log()).sum(dim=-1).mean()
        else:
            l_sparse = torch.tensor(0.0, device=dev)

        return lambda_div * l_div + lambda_sparse * l_sparse

    # ── Human-Readable Rule Extraction ───────────────────────────────────────

    def explain_rules(
        self,
        top_k         : int = 3,
        class_names   : Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate human-readable IF-THEN rules from learned parameters.

        Each rule premise is decoded by finding the top-k dimensions of
        the rule query vector. If feature_names were provided at init,
        dimension indices are mapped to feature names.

        Parameters
        ----------
        top_k       : number of top premise features to report per rule
        class_names : optional class label strings

        Returns
        -------
        rules : list of rule strings
        """
        class_names = class_names or [f"class_{i}" for i in range(self.n_classes)]
        rules       = []

        q = self.rule_queries.detach()          # (n_rules, hidden_dim)

        for r in range(self.n_rules):
            q_r = q[r]
            # Top positive premise dimensions
            pos_idx = q_r.topk(top_k).indices.tolist()
            neg_idx = (-q_r).topk(top_k).indices.tolist()

            pos_names = [
                (self.feature_names[i] if self.feature_names
                 and i < len(self.feature_names) else f"feature_{i}")
                for i in pos_idx
            ]
            neg_names = [
                (self.feature_names[i] if self.feature_names
                 and i < len(self.feature_names) else f"feature_{i}")
                for i in neg_idx
            ]

            # Rule conclusion from the rule head bias
            head_weight = self.rule_heads[r].weight.detach()
            conclusion  = class_names[head_weight.sum(dim=1).argmax().item()]

            premise = " AND ".join([f"{n} ↑" for n in pos_names] +
                                   [f"{n} ↓" for n in neg_names[:1]])
            rule_str = f"Rule {r+1}: IF {premise}  THEN  {conclusion}"
            rules.append(rule_str)

        return rules

    def explain_sample(
        self,
        rule_info   : Dict,
        sample_idx  : int = 0,
        class_names : Optional[List[str]] = None,
        threshold   : float = 0.10,
    ) -> str:
        """
        Generate per-sample explanation showing which rules fired.

        Parameters
        ----------
        rule_info  : output dict from forward()
        sample_idx : which sample in the batch to explain
        threshold  : minimum activation to include a rule in explanation
        """
        class_names = class_names or ["LOW_ENGAGEMENT", "HIGH_ENGAGEMENT"]
        acts        = rule_info["activations"][sample_idx]     # (n_rules,)
        r_logits    = rule_info["rule_logits"][sample_idx]     # (n_rules, C)

        active_rules = (acts >= threshold).nonzero(as_tuple=True)[0].tolist()
        lines        = [f"  Sample {sample_idx} — active rules (activation ≥ {threshold}):"]

        for r in active_rules:
            cls = class_names[r_logits[r].argmax().item()]
            lines.append(
                f"    Rule {r+1}: activation={acts[r].item():.3f}  → {cls}"
            )

        if not active_rules:
            lines.append("    (no single rule dominates — ensemble decision)")

        return "\n".join(lines)
