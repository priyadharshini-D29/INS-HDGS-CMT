"""
Extract the genuine learned neuro-symbolic rules from a trained checkpoint.

Honest scope
------------
The rule premises live in the rule layer's 64-d latent space, NOT in named
EEG/ET features. We therefore report only what is genuinely recoverable from
the trained weights:

  • each rule's learned conclusion (HIGH/LOW) and its decision margin
  • how distinct the 8 rules are (pairwise cosine similarity / diversity)
  • the learned rule-vs-bypass reliance  alpha = sigmoid(bypass_alpha)

We do NOT attach neurophysiological names to latent dimensions, since that
mapping does not exist in the current architecture.

Usage:
  python analysis/extract_neurosymbolic_rules.py <checkpoint.pt> [out.json]
"""
import sys, json, math
import torch
import torch.nn.functional as F

CKPT = sys.argv[1]
OUT  = sys.argv[2] if len(sys.argv) > 2 else "neurosymbolic_rules_extracted.json"
CLASS_NAMES = ["LOW_ENGAGEMENT", "HIGH_ENGAGEMENT"]

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck

Q = sd["rule_layer.rule_queries"]                       # (R, H)
R, H = Q.shape
alpha = torch.sigmoid(sd["rule_layer.bypass_alpha"]).item()

# Per-rule conclusion: feed each rule's own premise through its head.
# When rule r dominates, the key aligns with q_r, so head_r(q_r) approximates
# the class this rule concludes.
rules = []
for r in range(R):
    w = sd[f"rule_layer.rule_heads.{r}.weight"]         # (C, H)
    b = sd[f"rule_layer.rule_heads.{r}.bias"]           # (C,)
    logit = w @ Q[r] + b                                # (C,)
    prob  = F.softmax(logit, dim=-1)
    cls   = int(logit.argmax().item())
    margin = float((logit[1] - logit[0]).item())        # >0 -> HIGH, <0 -> LOW
    rules.append({
        "rule": r + 1,
        "concludes": CLASS_NAMES[cls],
        "p_high": round(float(prob[1].item()), 3),
        "decision_margin": round(margin, 3),
    })

# Diversity: pairwise cosine similarity between rule premises
Qn = F.normalize(Q, dim=-1)
S  = Qn @ Qn.T
off = S[~torch.eye(R, dtype=torch.bool)]
mean_sim = float(off.abs().mean().item())
max_sim  = float(off.abs().max().item())

n_high = sum(1 for r in rules if r["concludes"] == "HIGH_ENGAGEMENT")
n_low  = R - n_high

report = {
    "checkpoint": CKPT,
    "n_rules": R, "latent_dim": H,
    "rule_vs_bypass": {
        "alpha_bypass": round(alpha, 3),
        "rule_share": round(1 - alpha, 3),
        "interpretation": f"final logit = {alpha:.2f}*bypass + {1-alpha:.2f}*rules",
    },
    "rule_diversity": {
        "mean_abs_cosine": round(mean_sim, 3),
        "max_abs_cosine": round(max_sim, 3),
        "note": "0 = fully distinct rules, 1 = redundant/collapsed",
    },
    "vote_balance": {"rules_concluding_HIGH": n_high, "rules_concluding_LOW": n_low},
    "rules": rules,
}

with open(OUT, "w") as f:
    json.dump(report, f, indent=2)

# Pretty print
print(f"\nCheckpoint: {CKPT}")
print(f"{R} learned rules in a {H}-d latent space")
print(f"Reliance: alpha(bypass)={alpha:.3f}  ->  rules carry {1-alpha:.1%} of the decision")
print(f"Diversity: mean|cos|={mean_sim:.3f}  max|cos|={max_sim:.3f}  (0=distinct,1=redundant)")
print(f"Votes: {n_high} rules -> HIGH, {n_low} rules -> LOW\n")
print(f"{'Rule':>4} | {'concludes':<16} | {'P(HIGH)':>7} | {'margin':>7}")
print("-" * 46)
for r in rules:
    print(f"{r['rule']:>4} | {r['concludes']:<16} | {r['p_high']:>7.3f} | {r['decision_margin']:>+7.3f}")
print(f"\nSaved -> {OUT}")
