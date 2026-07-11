"""
Focal-loss ablation sweep for INS-HDGS-CMT.

Runs 13 full 37-fold LOSOCV configurations in parallel across 8 GPUs:
  - 1 CE baseline  (gamma=0, balanced)
  - 4 gamma × 3 alpha strategies = 12 focal configs

Each config uses N_ENSEMBLE=5 (vs production 15) to keep wall time ≤ 45 min
per config on 8 A100s.  The winning config should be re-run with N_ENSEMBLE=15
for the final paper number.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python sweep_focal_ablation.py
"""

import subprocess, os, sys, time
from pathlib import Path

PYTHON       = sys.executable
MAIN         = str(Path(__file__).parent / "main.py")
N_ENSEMBLE   = 5          # reduced for sweep speed; re-run winner at 15
EPOCHS       = 250
ENV_BASE     = {**os.environ,
                "PYTHONPATH": str(Path(__file__).parent.parent),
                "PYTHONUTF8": "1"}

GAMMAS   = [0.0, 1.0, 1.5, 2.0, 3.0]   # 0.0 = pure CE baseline
ALPHAS   = ["balanced", "effective_num", "sqrt_inv_freq"]

def label_for(gamma, alpha):
    g_str = f"g{gamma:.1f}".replace(".", "p")
    return f"focal_abl_{g_str}_{alpha}"

def log_path(label):
    return Path(__file__).parent / f"ablation_{label}.log"

configs = []
for gamma in GAMMAS:
    for alpha in ALPHAS:
        if gamma == 0.0 and alpha != "balanced":
            continue   # CE baseline: only one alpha matters (weights cancel at γ=0)
        configs.append((gamma, alpha))

print(f"Sweep: {len(configs)} configs  |  N_ENSEMBLE={N_ENSEMBLE}  |  EPOCHS={EPOCHS}")
print()

for i, (gamma, alpha) in enumerate(configs, 1):
    lbl  = label_for(gamma, alpha)
    logf = log_path(lbl)
    print(f"[{i:>2}/{len(configs)}] label={lbl}")

    cmd = [
        PYTHON, MAIN,
        "--fold-parallel",
        "--label",          lbl,
        "--epochs",         str(EPOCHS),
        "--alpha-strategy", alpha,
        "--n-ensemble",     str(N_ENSEMBLE),
    ]
    if gamma > 0:
        cmd += ["--focal-gamma", str(gamma)]
    else:
        cmd += ["--focal-gamma", "0.0"]   # explicit CE mode

    t0  = time.time()
    env = {**ENV_BASE, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
    with open(logf, "w") as fh:
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    ok = "OK" if result.returncode == 0 else f"FAILED(rc={result.returncode})"
    print(f"      {ok}  wall={elapsed/60:.1f} min  log={logf.name}")

print("\nAll configs done.")
