#!/usr/bin/env bash
# Summarize Phase 1 Residual-WM RFT policy evaluation results in Japanese.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

RFT_EVAL_ROOT="${RFT_EVAL_ROOT:-${REPO_ROOT}/results/phase1/phase1_sweeps/phase1_residual_spatial_rft/rft_eval}"
WM_EVAL_ROOT="${WM_EVAL_ROOT:-${REPO_ROOT}/results/phase1/phase1_sweeps/phase1_residual_spatial/wm_eval_episode_phases_e7}"
OUT_MD="${OUT_MD:-${RFT_EVAL_ROOT}/comparison.md}"

"${REPO_ROOT}/.venv/bin/python" analysis/rft/summarize_rft_eval_ja.py \
  "${RFT_EVAL_ROOT}" \
  --wm-eval-root "${WM_EVAL_ROOT}" \
  --out-md "${OUT_MD}"

echo "summary: ${OUT_MD}"
