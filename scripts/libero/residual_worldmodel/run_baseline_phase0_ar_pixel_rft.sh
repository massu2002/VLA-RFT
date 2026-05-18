#!/usr/bin/env bash
# run_baseline_phase0_ar_pixel_rft.sh — RFT using phase0 token WM (AR-Pixel) as reward.
#
# Uses the existing token world model checkpoint
# (checkpoints/libero/WorldModel/${TASK_SUITE}/20260429_worldmodel_scratch/checkpoint-150000)
# as the reward model for GRPO training. This serves as the baseline comparison
# against v4 WM-based RFT.
# Saves to checkpoints/libero/TemporalQueryResidualWM-RFT/baseline_phase0_ar_pixel/...
#
# Usage:
#   TASK_SUITE=spatial \
#     bash scripts/libero/residual_worldmodel/run_baseline_phase0_ar_pixel_rft.sh
#
#   # Specify a particular checkpoint step:
#   TOKEN_WM_PATH=checkpoints/libero/WorldModel/spatial/20260429_worldmodel_scratch/checkpoint-150000 \
#   TASK_SUITE=spatial \
#     bash scripts/libero/residual_worldmodel/run_baseline_phase0_ar_pixel_rft.sh
#
# Key env vars:
#   TASK_SUITE       spatial | object | goal | 10 (default: spatial)
#   TOKEN_WM_PATH    Path to token WM checkpoint dir
#                    (default: checkpoints/libero/WorldModel/${TASK_SUITE}/20260429_worldmodel_scratch/checkpoint-150000)
#   RFT_EXP_NAME     RFT run label (default: baseline_phase0_ar_pixel)
#   RFT_STEPS        GRPO gradient steps (default: 400)
#   N_GPUS_PER_NODE  Number of GPUs (default: 8)
#   SMOKE            1 to run tiny smoke test
#   DRY_RUN          1 to print config without executing
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

# ── Config ──────────────────────────────────────────────────────────────────
export TASK_SUITE="${TASK_SUITE:-spatial}"
TOKEN_WM_PATH="${TOKEN_WM_PATH:-${REPO_ROOT}/checkpoints/libero/WorldModel/${TASK_SUITE}/20260429_worldmodel_scratch/checkpoint-150000}"
RFT_EXP_NAME="${RFT_EXP_NAME:-baseline_phase0_ar_pixel}"

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ ! -d "${TOKEN_WM_PATH}" ]]; then
    echo "[run_baseline_phase0_ar_pixel_rft] ERROR: TOKEN_WM_PATH not found: ${TOKEN_WM_PATH}" >&2
    echo "  Set TOKEN_WM_PATH to the token WM checkpoint dir." >&2
    echo "  Examples:" >&2
    echo "    checkpoints/libero/WorldModel/spatial/20260429_worldmodel_scratch/checkpoint-150000" >&2
    exit 2
fi

echo "[run_baseline_phase0_ar_pixel_rft]"
echo "  TASK_SUITE     : ${TASK_SUITE}"
echo "  TOKEN_WM_PATH  : ${TOKEN_WM_PATH}"
echo "  RFT_EXP_NAME   : ${RFT_EXP_NAME}"
echo "  RFT_CKPT_ROOT  : ${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM-RFT/baseline_phase0_ar_pixel"

# ── Export and launch ────────────────────────────────────────────────────────
export MODEL_GENERATION="baseline"
export WORLD_MODEL_CKPT="${TOKEN_WM_PATH}"
export EXP_NAME="${RFT_EXP_NAME}"
export RFT_CKPT_ROOT="${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM-RFT/baseline_phase0_ar_pixel"

bash "${SCRIPT_DIR}/post_train_phase1_residual_rft.sh"
