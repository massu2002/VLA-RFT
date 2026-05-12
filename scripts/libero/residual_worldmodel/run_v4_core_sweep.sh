#!/usr/bin/env bash
# run_v4_core_sweep.sh — Backend per-condition runner for v4 core sweep.
#
# Called by scripts/libero/phase1/run_v4_core_sweep.sh for each experiment
# condition defined in a v4 sweep JSON config.
#
# Do NOT call this script directly in production; use the frontend script.
#
# Required env vars (set by frontend):
#   EXP_NAME           experiment name (unique per condition)
#   STAGE              v4a | v4b
#   HISTORY_LENGTH     K
#   NUM_DYNAMIC_QUERIES Q
#   USE_MOTION_BIAS    0|1
#   USE_ACTION_FUTURE_SCORER 0|1
#   LAMBDA_RANK        float
#   RANK_MARGIN        float
#   NEGATIVE_TYPE      metadata only; model uses batch fallback internally
#   NEGATIVE_MIX       metadata only
#   LAMBDA_IMAGE, LAMBDA_DYNAMIC, LAMBDA_STATIC, LAMBDA_QUERY, LAMBDA_SPARSE
#   TASK_SUITE, SEED, RUN_NAME
#   CKPT_ROOT          checkpoint root (train output)
#   OUT_ROOT           results root (eval + metadata output)
#   SWEEP_CONFIG       path to v4_core_sweep.json
#   MODE               train_only | eval_only | train_eval | rft_only | all
#   DRY_RUN            0|1
#   SMOKE              0|1
#   SKIP_EXISTING      0|1
#   OVERWRITE          0|1
#   EVAL_HORIZON       7
#   PHASE0_COMPATIBLE  0|1
#   MAX_STEPS, LR, PRECISION, ...  (passed through)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log() { echo "[v4-sweep-backend] $(date +%H:%M:%S) [${EXP_NAME}] $*"; }

# ---------------------------------------------------------------------------
# Required inputs
# ---------------------------------------------------------------------------
EXP_NAME="${EXP_NAME:?'EXP_NAME required'}"
STAGE="${STAGE:-v4b}"
HISTORY_LENGTH="${HISTORY_LENGTH:-2}"
NUM_DYNAMIC_QUERIES="${NUM_DYNAMIC_QUERIES:-8}"
USE_MOTION_BIAS="${USE_MOTION_BIAS:-0}"
USE_ACTION_FUTURE_SCORER="${USE_ACTION_FUTURE_SCORER:-1}"
LAMBDA_RANK="${LAMBDA_RANK:-1.0}"
RANK_MARGIN="${RANK_MARGIN:-0.1}"
RANK_TEMPERATURE="${RANK_TEMPERATURE:-0.07}"
NEGATIVE_TYPE="${NEGATIVE_TYPE:-same_task_other_window}"  # metadata; model uses batch fallback
NEGATIVE_MIX="${NEGATIVE_MIX:-}"                          # metadata only
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.15}"
INIT_FROM_CHECKPOINT="${INIT_FROM_CHECKPOINT:-}"          # warm-start checkpoint path (optional)

# Resolve @exp_name references: "@v4a_q8_k2_motion" →
# ${CKPT_ROOT}/${TASK_SUITE}/v4a_q8_k2_motion/s${CKPT_SEED}/final
if [[ "${INIT_FROM_CHECKPOINT}" == @* ]]; then
  _ref_exp="${INIT_FROM_CHECKPOINT:1}"
  INIT_FROM_CHECKPOINT="${CKPT_ROOT}/${TASK_SUITE}/${_ref_exp}/s${CKPT_SEED}/final"
  log "Resolved init_from_checkpoint @${_ref_exp} → ${INIT_FROM_CHECKPOINT}"
fi

LAMBDA_IMAGE="${LAMBDA_IMAGE:-0.1}"
LAMBDA_DYNAMIC="${LAMBDA_DYNAMIC:-1.0}"
LAMBDA_STATIC="${LAMBDA_STATIC:-0.2}"
LAMBDA_QUERY="${LAMBDA_QUERY:-0.5}"
LAMBDA_SPARSE="${LAMBDA_SPARSE:-0.01}"

TASK_SUITE="${TASK_SUITE:-spatial}"
SEED="${SEED:-42}"
# CKPT_SEED: seed used during training (determines checkpoint path s${CKPT_SEED}).
# Defaults to SEED so single-seed runs work without change.
# Set CKPT_SEED=42 when evaluating the same model with multiple eval seeds.
CKPT_SEED="${CKPT_SEED:-${SEED}}"
RUN_NAME="${RUN_NAME:?'RUN_NAME required'}"
EVAL_HORIZON="${EVAL_HORIZON:-7}"
TRAIN_HORIZON="${TRAIN_HORIZON:-${EVAL_HORIZON}}"
PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE:-1}"

# SEGMENT_LENGTH = K + H + 2
SEGMENT_LENGTH=$(( HISTORY_LENGTH + TRAIN_HORIZON + 2 ))
export SEGMENT_LENGTH

CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM/${RUN_NAME}}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/residual_worldmodel/${RUN_NAME}}"
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/libero/phase1/v4_core_sweep.json}"
MODE="${MODE:-train_eval}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OVERWRITE="${OVERWRITE:-0}"

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
CKPT_DIR="${CKPT_ROOT}/${TASK_SUITE}/${EXP_NAME}/s${CKPT_SEED}"
CKPT_FINAL="${CKPT_DIR}/final"
COND_OUT="${OUT_ROOT}/${EXP_NAME}"
mkdir -p "${COND_OUT}"

# Smoke: reduce steps/windows
if is_true "${SMOKE}"; then
  MAX_STEPS="${MAX_STEPS_SMOKE:-100}"
  NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS_SMOKE:-4}"
  NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS_SMOKE:-4}"
  SAVE_STEPS="${SAVE_STEPS:-100}"
fi

# ---------------------------------------------------------------------------
# SKIP_EXISTING / OVERWRITE logic
# ---------------------------------------------------------------------------
_need_train() {
  is_true "${OVERWRITE}" && return 0
  is_true "${SKIP_EXISTING}" || return 0
  [ ! -f "${CKPT_FINAL}/v4_config.json" ]
}
_need_eval() {
  is_true "${OVERWRITE}" && return 0
  is_true "${SKIP_EXISTING}" || return 0
  [ ! -f "${COND_OUT}/aggregate_metrics.json" ]
}

# ---------------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------------
_save_metadata() {
  python3 - <<PYEOF
import json, os, subprocess
from pathlib import Path

out = Path("${COND_OUT}")
out.mkdir(parents=True, exist_ok=True)

cfg_used = {
    "exp_name": "${EXP_NAME}",
    "stage": "${STAGE}",
    "run_name": "${RUN_NAME}",
    "task_suite": "${TASK_SUITE}",
    "seed": int("${SEED}"),
    "mode": "${MODE}",
    "segment_length": int("${SEGMENT_LENGTH}"),
    "train_horizon": int("${TRAIN_HORIZON}"),
    "eval_horizon": int("${EVAL_HORIZON}"),
    "history_length": int("${HISTORY_LENGTH}"),
    "num_dynamic_queries": int("${NUM_DYNAMIC_QUERIES}"),
    "use_motion_bias": int("${USE_MOTION_BIAS}"),
    "use_action_future_scorer": int("${USE_ACTION_FUTURE_SCORER}"),
    "lambda_rank": float("${LAMBDA_RANK}"),
    "rank_margin": float("${RANK_MARGIN}"),
    "rank_temperature": float("${RANK_TEMPERATURE}"),
    "negative_type": "${NEGATIVE_TYPE}",
    "negative_mix": "${NEGATIVE_MIX}" or None,
    "lambda_image": float("${LAMBDA_IMAGE}"),
    "lambda_dynamic": float("${LAMBDA_DYNAMIC}"),
    "lambda_static": float("${LAMBDA_STATIC}"),
    "lambda_query": float("${LAMBDA_QUERY}"),
    "lambda_sparse": float("${LAMBDA_SPARSE}"),
    "phase0_compatible": int("${PHASE0_COMPATIBLE}"),
    "ckpt_dir": "${CKPT_DIR}",
    "cond_out": "${COND_OUT}",
    "sweep_config": "${SWEEP_CONFIG}",
}
(out / "v4_sweep_config_used.json").write_text(json.dumps(cfg_used, indent=2))

# git status
try:
    gs = subprocess.check_output(["git", "status", "--short"], cwd="${REPO_ROOT}", text=True)
    (out / "git_status.txt").write_text(gs)
except Exception:
    pass

import platform, datetime
env_info = {
    "date": datetime.datetime.now().isoformat(),
    "hostname": platform.node(),
    "python": platform.python_version(),
    "repo_root": "${REPO_ROOT}",
    "exp_name": "${EXP_NAME}",
}
(out / "env_used.txt").write_text(json.dumps(env_info, indent=2))
PYEOF
}

# ---------------------------------------------------------------------------
# Determine which phases to run
# ---------------------------------------------------------------------------
RUN_TRAIN=0; RUN_EVAL=0; RUN_RFT=0
case "${MODE}" in
  train_only)  RUN_TRAIN=1 ;;
  eval_only)   RUN_EVAL=1 ;;
  train_eval)  RUN_TRAIN=1; RUN_EVAL=1 ;;
  rft_only)    RUN_RFT=1 ;;
  all)         RUN_TRAIN=1; RUN_EVAL=1; RUN_RFT=1 ;;
  *)
    echo "[v4-sweep-backend] Unknown MODE=${MODE}. Use train_only|eval_only|train_eval|rft_only|all" >&2
    exit 2
    ;;
esac

log "MODE=${MODE}  stage=${STAGE}  K=${HISTORY_LENGTH}  Q=${NUM_DYNAMIC_QUERIES}  seg_len=${SEGMENT_LENGTH}"
log "  motion_bias=${USE_MOTION_BIAS}  scorer=${USE_ACTION_FUTURE_SCORER}  lambda_rank=${LAMBDA_RANK}"
log "  ckpt=${CKPT_DIR}"
log "  out=${COND_OUT}"

_save_metadata

# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
if [ "${RUN_TRAIN}" = "1" ]; then
  if ! _need_train; then
    log "SKIP train (checkpoint exists and SKIP_EXISTING=1)"
  else
    TRAIN_CMD="TASK_SUITE=${TASK_SUITE} EXP_NAME=${EXP_NAME} OUTPUT_ROOT=${CKPT_ROOT} SEED=${SEED} \
HISTORY_LENGTH=${HISTORY_LENGTH} NUM_DYNAMIC_QUERIES=${NUM_DYNAMIC_QUERIES} \
USE_MOTION_BIAS=${USE_MOTION_BIAS} USE_ACTION_FUTURE_SCORER=${USE_ACTION_FUTURE_SCORER} \
LAMBDA_RANK=${LAMBDA_RANK} RANK_MARGIN=${RANK_MARGIN} RANK_TEMPERATURE=${RANK_TEMPERATURE} \
NEGATIVE_TYPE=${NEGATIVE_TYPE} NEGATIVE_MIX=${NEGATIVE_MIX} \
ACTION_NOISE_STD=${ACTION_NOISE_STD} INIT_FROM_CHECKPOINT=${INIT_FROM_CHECKPOINT} \
LAMBDA_IMAGE=${LAMBDA_IMAGE} LAMBDA_DYNAMIC=${LAMBDA_DYNAMIC} LAMBDA_STATIC=${LAMBDA_STATIC} \
LAMBDA_QUERY=${LAMBDA_QUERY} LAMBDA_SPARSE=${LAMBDA_SPARSE} \
SEGMENT_LENGTH=${SEGMENT_LENGTH} TRAIN_HORIZON=${TRAIN_HORIZON} \
MAX_STEPS=${MAX_STEPS:-150000} BATCH_SIZE=${BATCH_SIZE:-1} WORLD_MODEL_BATCH_SIZE=${WORLD_MODEL_BATCH_SIZE:-16} LR=${LR:-5e-5} \
bash ${SCRIPT_DIR}/train_v4_temporal_query_worldmodel.sh ${TASK_SUITE}"
    echo "${TRAIN_CMD}" > "${COND_OUT}/train_command.txt"

    if is_true "${DRY_RUN}"; then
      log "DRY_RUN: ${TRAIN_CMD}"
    else
      log "Starting train..."
      TASK_SUITE="${TASK_SUITE}" \
      EXP_NAME="${EXP_NAME}" \
      OUTPUT_ROOT="${CKPT_ROOT}" \
      SEED="${SEED}" \
      HISTORY_LENGTH="${HISTORY_LENGTH}" \
      NUM_DYNAMIC_QUERIES="${NUM_DYNAMIC_QUERIES}" \
      USE_MOTION_BIAS="${USE_MOTION_BIAS}" \
      USE_ACTION_FUTURE_SCORER="${USE_ACTION_FUTURE_SCORER}" \
      LAMBDA_RANK="${LAMBDA_RANK}" \
      RANK_MARGIN="${RANK_MARGIN}" \
      RANK_TEMPERATURE="${RANK_TEMPERATURE}" \
      NEGATIVE_TYPE="${NEGATIVE_TYPE}" \
      NEGATIVE_MIX="${NEGATIVE_MIX}" \
      ACTION_NOISE_STD="${ACTION_NOISE_STD}" \
      INIT_FROM_CHECKPOINT="${INIT_FROM_CHECKPOINT}" \
      LAMBDA_IMAGE="${LAMBDA_IMAGE}" \
      LAMBDA_DYNAMIC="${LAMBDA_DYNAMIC}" \
      LAMBDA_STATIC="${LAMBDA_STATIC}" \
      LAMBDA_QUERY="${LAMBDA_QUERY}" \
      LAMBDA_SPARSE="${LAMBDA_SPARSE}" \
      SEGMENT_LENGTH="${SEGMENT_LENGTH}" \
      TRAIN_HORIZON="${TRAIN_HORIZON}" \
        bash "${SCRIPT_DIR}/train_v4_temporal_query_worldmodel.sh" "${TASK_SUITE}"
      log "Train done → ${CKPT_DIR}"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# EVAL
# ---------------------------------------------------------------------------
if [ "${RUN_EVAL}" = "1" ]; then
  if ! _need_eval; then
    log "SKIP eval (aggregate_metrics.json exists and SKIP_EXISTING=1)"
  else
    # Require checkpoint (unless DRY_RUN)
    if ! is_true "${DRY_RUN}" && [ ! -d "${CKPT_FINAL}" ]; then
      log "SKIP eval: final checkpoint not found at ${CKPT_FINAL}"
    else
      EVAL_ARGS=(
        "MODEL_DIR=${CKPT_FINAL}"
        "TASK_SUITE=${TASK_SUITE}"
        "OUTPUT_DIR=${COND_OUT}"
        "CONDITION_NAME=${EXP_NAME}"
        "SEED=${SEED}"
        "EVAL_HORIZON=${EVAL_HORIZON}"
        "NUM_EVAL_WINDOWS=${NUM_EVAL_WINDOWS:-200}"
        "NUM_RANKING_WINDOWS=${NUM_RANKING_WINDOWS:-100}"
        "WINDOW_POSITION_MODE=${WINDOW_POSITION_MODE:-episode_phases}"
        "NEGATIVE_EVAL_TYPES=${NEGATIVE_EVAL_TYPES:-same_phase,temporal_shift,action_noise,mixed}"
        "TEMPORAL_SHIFT_MAX=${TEMPORAL_SHIFT_MAX:-3}"
        "ACTION_NOISE_STD=${ACTION_NOISE_STD:-0.15}"
        "ACTION_ABLATION=${ACTION_ABLATION:-0}"
        "SAVE_DEBUG_VISUALS=${SAVE_DEBUG_VISUALS:-0}"
      )
      EVAL_CMD="bash ${SCRIPT_DIR}/eval_v4_temporal_query_worldmodel.sh ${EVAL_ARGS[*]}"
      echo "${EVAL_CMD}" > "${COND_OUT}/eval_command.txt"
      [ "${PHASE0_COMPATIBLE}" = "1" ] && echo "  + PHASE0_COMPATIBLE=1" >> "${COND_OUT}/eval_command.txt"

      if is_true "${DRY_RUN}"; then
        log "DRY_RUN: ${EVAL_CMD}"
      else
        log "Starting eval..."
        USE_WINDOW_MANIFEST="${USE_WINDOW_MANIFEST:-0}" \
        WINDOW_MANIFEST="${WINDOW_MANIFEST:-}" \
        PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE}" \
          bash "${SCRIPT_DIR}/eval_v4_temporal_query_worldmodel.sh" "${EVAL_ARGS[@]}"

        # Copy v4_config.json alongside eval results for easy reference
        if [ -f "${CKPT_FINAL}/v4_config.json" ] && [ ! -f "${COND_OUT}/v4_config.json" ]; then
          cp "${CKPT_FINAL}/v4_config.json" "${COND_OUT}/v4_config.json"
        fi
        # Copy eval_protocol_config.json if produced
        if [ -f "${COND_OUT}/eval_protocol_config.json" ]; then
          true  # already there
        fi
        log "Eval done → ${COND_OUT}/aggregate_metrics.json"
      fi
    fi
  fi
fi

# ---------------------------------------------------------------------------
# RFT (optional, single condition)
# ---------------------------------------------------------------------------
if [ "${RUN_RFT}" = "1" ]; then
  if ! is_true "${DRY_RUN}" && [ ! -d "${CKPT_FINAL}" ]; then
    log "SKIP rft: final checkpoint not found at ${CKPT_FINAL}"
  else
    WM_REWARD_TYPE="${WORLD_REWARD_TYPE:-hybrid}"
    # v4a models have no ActionFutureScorer; force lpips_mae instead of hybrid/rank_score
    if [[ "${USE_ACTION_FUTURE_SCORER:-0}" == "0" && "${WM_REWARD_TYPE}" =~ ^(hybrid|rank_score)$ ]]; then
      log "WARN: USE_ACTION_FUTURE_SCORER=0 (v4a); forcing WORLD_REWARD_TYPE from '${WM_REWARD_TYPE}' to 'lpips_mae'"
      WM_REWARD_TYPE="lpips_mae"
    fi
    RFT_EXP="${EXP_NAME}_rft_${WM_REWARD_TYPE}"
    RFT_OUT="${COND_OUT}/rft_${WM_REWARD_TYPE}"
    RFT_CMD="WORLD_MODEL_CKPT=${CKPT_FINAL} WORLD_REWARD_TYPE=${WM_REWARD_TYPE} EXP_NAME=${RFT_EXP} OUTPUT_DIR=${RFT_OUT} bash post_train_phase1_residual_rft.sh"
    echo "${RFT_CMD}" > "${COND_OUT}/rft_command_${WM_REWARD_TYPE}.txt"

    if is_true "${DRY_RUN}"; then
      log "DRY_RUN rft: ${RFT_CMD}"
    else
      log "Starting RFT (${WM_REWARD_TYPE})..."
      TASK_SUITE="${TASK_SUITE}" \
      MODEL_GENERATION="v4" \
      TARGET_MODE="temporal_query_residual" \
      WORLD_MODEL_CKPT="${CKPT_FINAL}" \
      WORLD_MODEL_CONFIG="${CKPT_FINAL}/v4_config.json" \
      EXP_NAME="${RFT_EXP}" \
      OUTPUT_DIR="${RFT_OUT}" \
      RFT_STEPS="${RFT_STEPS:-400}" \
      SEED="${RFT_SEED:-7}" \
      WORLD_REWARD_TYPE="${WM_REWARD_TYPE}" \
      RANK_REWARD_ALPHA="${RANK_REWARD_ALPHA:-0.2}" \
      RANK_REWARD_BETA="${RANK_REWARD_BETA:-0.8}" \
      NORMALIZE_RANK_REWARD="${NORMALIZE_RANK_REWARD:-1}" \
      CLIP_RANK_REWARD="${CLIP_RANK_REWARD:-1}" \
      SMOKE="${SMOKE}" \
        bash "${SCRIPT_DIR}/post_train_phase1_residual_rft.sh"
      log "RFT done → ${RFT_OUT}"
    fi
  fi
fi

log "Condition complete."
