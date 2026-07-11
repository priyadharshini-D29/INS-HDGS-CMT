#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Convenience wrapper. Verifies the environment, then runs the
# full paper-reproduction pipeline.
#
# Steps:
#   1. Print environment/reproducibility info (Python, CUDA, GPU, seed).
#   2. Run reproducibility/reproduce_paper.sh.
#
# Usage:  bash reproducibility/run_all.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Environment ==="
python -c "import sys,platform; print('Python :', sys.version.split()[0], platform.platform())"
python -c "import torch; print('Torch  :', torch.__version__); print('CUDA   :', torch.version.cuda); print('GPU    :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" || true
echo "Seed   : ${SEED:-42}"
echo

bash reproducibility/reproduce_paper.sh
