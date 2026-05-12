#!/usr/bin/env bash
# run_v4_temporal_query_rft.sh — Canonical v4 WM → RFT training launcher.
#
# Resolves checkpoint path from WM_RUN_NAME + WM_EXP_NAME + CKPT_SEED,
# then calls post_train_phase1_residual_rft.sh directly (EXP_NAME safe).
# Saves to checkpoints/libero/TemporalQueryResidualWM-RFT/${WM_RUN_NAME}/...
#
# Usage:
#   WM_RUN_NAME=v4_improved_spatial \
#   WM_EXP_NAME=v4b_q8_k2_rank1_motion \
#   TASK_SUITE=spatial \
#   WORLD_REWARD_TYPE=hybrid \
#     bash scripts/libero/residual_worldmodel/run_v4_temporal_query_rft.sh
#
# Key env vars:
#   WM_RUN_NAME          WM training run name (default: v4_improved_spatial)
#   WM_EXP_NAME          Experiment name within that run (required)
#   CKPT_SEED            Seed of the WM training (default: 42)
#   TASK_SUITE           spatial | object | goal | 10 (default: spatial)
#   WORLD_REWARD_TYPE    lpips_mae | rank_score | hybrid (default: hybrid for v4b; auto-forced to lpips_mae for v4a)
#   RANK_REWARD_ALPHA    hybrid visual weight (default: 0.2)
#   RANK_REWARD_BETA     hybrid rank_score weight (default: 0.8)
#   NORMALIZE_RANK_REWARD  1|0 (default: 1)
#   CLIP_RANK_REWARD       1|0 (default: 1)
#   RANK_REWARD_CLIP_VALUE clip magnitude (default: 5.0)
#   RFT_EXP_NAME         Override RFT run label (default: ${WM_EXP_NAME}_rft_${WORLD_REWARD_TYPE})
#   RFT_STEPS            GRPO gradient steps (default: 400)
#   N_GPUS_PER_NODE      Number of GPUs (default: 8)
#   SMOKE                1 to run tiny smoke test
#   DRY_RUN              1 to print config without executing
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

# ── Config ──────────────────────────────────────────────────────────────────
WM_RUN_NAME="${WM_RUN_NAME:-v4_improved_spatial}"
WM_EXP_NAME="${WM_EXP_NAME:?'WM_EXP_NAME is required (e.g. v4b_q8_k2_rank1_motion)'}"
CKPT_SEED="${CKPT_SEED:-42}"
export TASK_SUITE="${TASK_SUITE:-spatial}"
WORLD_REWARD_TYPE="${WORLD_REWARD_TYPE:-hybrid}"
RANK_REWARD_ALPHA="${RANK_REWARD_ALPHA:-0.2}"
RANK_REWARD_BETA="${RANK_REWARD_BETA:-0.8}"
NORMALIZE_RANK_REWARD="${NORMALIZE_RANK_REWARD:-1}"
CLIP_RANK_REWARD="${CLIP_RANK_REWARD:-1}"
RANK_REWARD_CLIP_VALUE="${RANK_REWARD_CLIP_VALUE:-5.0}"

# ── Checkpoint path resolution ───────────────────────────────────────────────
CKPT_ROOT="${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM/${WM_RUN_NAME}"
WORLD_MODEL_CKPT="${CKPT_ROOT}/${TASK_SUITE}/${WM_EXP_NAME}/s${CKPT_SEED}/final"
WORLD_MODEL_CONFIG="${WORLD_MODEL_CKPT}/v4_config.json"

if [[ ! -d "${WORLD_MODEL_CKPT}" ]]; then
    echo "[run_v4_temporal_query_rft] ERROR: checkpoint not found: ${WORLD_MODEL_CKPT}" >&2
    exit 2
fi
if [[ ! -f "${WORLD_MODEL_CONFIG}" ]]; then
    echo "[run_v4_temporal_query_rft] ERROR: v4_config.json not found: ${WORLD_MODEL_CONFIG}" >&2
    exit 2
fi

# ── Scorer check: v4a に hybrid/rank_score は使えない ────────────────────────
use_scorer=$(python3 -c "
import json
c = json.load(open('${WORLD_MODEL_CONFIG}'))
print(int(bool(c.get('use_action_future_scorer', False))))")
if [[ "${use_scorer}" == "0" && "${WORLD_REWARD_TYPE}" =~ ^(hybrid|rank_score)$ ]]; then
    echo "[run_v4_temporal_query_rft] WARN: use_action_future_scorer=false in v4_config (v4a model)" >&2
    echo "[run_v4_temporal_query_rft] WARN: WORLD_REWARD_TYPE forced from '${WORLD_REWARD_TYPE}' to 'lpips_mae'" >&2
    WORLD_REWARD_TYPE="lpips_mae"
fi

# ── EXP_NAME: human-readable（ckpt path の basename を使わない） ─────────────
RFT_EXP_NAME="${RFT_EXP_NAME:-${WM_EXP_NAME}_rft_${WORLD_REWARD_TYPE}}"

echo "[run_v4_temporal_query_rft]"
echo "  WM_RUN_NAME       : ${WM_RUN_NAME}"
echo "  WM_EXP_NAME       : ${WM_EXP_NAME}"
echo "  CKPT_SEED         : ${CKPT_SEED}"
echo "  TASK_SUITE        : ${TASK_SUITE}"
echo "  WORLD_MODEL_CKPT  : ${WORLD_MODEL_CKPT}"
echo "  WORLD_REWARD_TYPE : ${WORLD_REWARD_TYPE}"
echo "  RFT_EXP_NAME      : ${RFT_EXP_NAME}"
echo "  RANK_REWARD_ALPHA : ${RANK_REWARD_ALPHA}"
echo "  RANK_REWARD_BETA  : ${RANK_REWARD_BETA}"

# ── Export and launch ────────────────────────────────────────────────────────
export MODEL_GENERATION="v4"
export TARGET_MODE="temporal_query_residual"
export WORLD_MODEL_CKPT
export WORLD_MODEL_CONFIG
export EXP_NAME="${RFT_EXP_NAME}"
export RFT_CKPT_ROOT="${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM-RFT/${WM_RUN_NAME}"
export WORLD_REWARD_TYPE
export RANK_REWARD_ALPHA
export RANK_REWARD_BETA
export NORMALIZE_RANK_REWARD
export CLIP_RANK_REWARD
export RANK_REWARD_CLIP_VALUE

# world_reward.* を全量 hydra override で渡す（YAML default より env var を優先）
export USER_EXTRA_RFT_ARGS="\
world_reward.type=${WORLD_REWARD_TYPE} \
world_reward.rank_alpha=${RANK_REWARD_ALPHA} \
world_reward.rank_beta=${RANK_REWARD_BETA} \
world_reward.normalize_rank=${NORMALIZE_RANK_REWARD} \
world_reward.clip_rank=${CLIP_RANK_REWARD} \
world_reward.clip_value=${RANK_REWARD_CLIP_VALUE} \
${USER_EXTRA_RFT_ARGS:-}"

bash "${SCRIPT_DIR}/post_train_phase1_residual_rft.sh"
