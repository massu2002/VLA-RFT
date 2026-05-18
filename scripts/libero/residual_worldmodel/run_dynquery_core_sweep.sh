#!/usr/bin/env bash
# run_dynquery_core_sweep.sh — Backend per-condition runner for DynQuery core sweep.
#
# Called by scripts/libero/phase1/run_dynquery_core_sweep.sh for each experiment
# condition defined in dynquery_core_sweep.json.
#
# Do NOT call this script directly; use the frontend script.
#
# Required env vars (set by frontend):
#   EXP_NAME, STAGE, HISTORY_LENGTH, NUM_DYNAMIC_QUERIES
#   USE_ACTION_CONDITIONED_MASK, PREDICTOR_MODE, USE_DYNAMIC_RESIDUAL_GATE
#   LAMBDA_MASK_DYNAMIC, LAMBDA_QUERY_DELTA_SPARSE
#   USE_MOTION_BIAS, USE_ACTION_FUTURE_SCORER
#   LAMBDA_RANK, RANK_MARGIN, NEGATIVE_TYPE, NEGATIVE_MIX
#   LAMBDA_IMAGE, LAMBDA_DYNAMIC, LAMBDA_STATIC, LAMBDA_QUERY
#   TASK_SUITE, SEED, RUN_NAME
#   CKPT_ROOT, OUT_ROOT, SWEEP_CONFIG
#   MODE, DRY_RUN, SMOKE, SKIP_EXISTING, OVERWRITE

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log() { echo "[dynquery-sweep-backend] $(date +%H:%M:%S) [${EXP_NAME}] $*"; }

# ---------------------------------------------------------------------------
# Required inputs
# ---------------------------------------------------------------------------
EXP_NAME="${EXP_NAME:?'EXP_NAME required'}"
STAGE="${STAGE:-dq_b}"
HISTORY_LENGTH="${HISTORY_LENGTH:-2}"
NUM_DYNAMIC_QUERIES="${NUM_DYNAMIC_QUERIES:-8}"

# Core feature flags
USE_ACTION_CONDITIONED_MASK="${USE_ACTION_CONDITIONED_MASK:-1}"
PREDICTOR_MODE="${PREDICTOR_MODE:-query_wise}"
USE_DYNAMIC_RESIDUAL_GATE="${USE_DYNAMIC_RESIDUAL_GATE:-1}"
LAMBDA_MASK_DYNAMIC="${LAMBDA_MASK_DYNAMIC:-0.1}"
LAMBDA_QUERY_DELTA_SPARSE="${LAMBDA_QUERY_DELTA_SPARSE:-0.001}"

USE_MOTION_BIAS="${USE_MOTION_BIAS:-1}"
USE_ACTION_FUTURE_SCORER="${USE_ACTION_FUTURE_SCORER:-1}"
LAMBDA_RANK="${LAMBDA_RANK:-1.0}"
RANK_MARGIN="${RANK_MARGIN:-0.1}"
RANK_TEMPERATURE="${RANK_TEMPERATURE:-0.07}"
NEGATIVE_TYPE="${NEGATIVE_TYPE:-mixed}"
NEGATIVE_MIX="${NEGATIVE_MIX:-}"
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.15}"
INIT_FROM_CHECKPOINT="${INIT_FROM_CHECKPOINT:-}"

# Resolve @exp_name references: "@dq_core14_no_scorer" →
# ${CKPT_ROOT}/${TASK_SUITE}/dq_core14_no_scorer/s${CKPT_SEED}/final
if [[ "${INIT_FROM_CHECKPOINT}" == @* ]]; then
  _ref_exp="${INIT_FROM_CHECKPOINT:1}"
  INIT_FROM_CHECKPOINT="${CKPT_ROOT}/${TASK_SUITE}/${_ref_exp}/s${CKPT_SEED}/final"
  log "Resolved init_from_checkpoint @${_ref_exp} → ${INIT_FROM_CHECKPOINT}"
fi

LAMBDA_IMAGE="${LAMBDA_IMAGE:-0.1}"
LAMBDA_DYNAMIC="${LAMBDA_DYNAMIC:-1.0}"
LAMBDA_STATIC="${LAMBDA_STATIC:-0.2}"
LAMBDA_QUERY="${LAMBDA_QUERY:-0.5}"

TASK_SUITE="${TASK_SUITE:-spatial}"
SEED="${SEED:-42}"
CKPT_SEED="${CKPT_SEED:-${SEED}}"
RUN_NAME="${RUN_NAME:?'RUN_NAME required'}"
EVAL_HORIZON="${EVAL_HORIZON:-8}"
TRAIN_HORIZON="${TRAIN_HORIZON:-${EVAL_HORIZON}}"
PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE:-1}"

SEGMENT_LENGTH=$(( HISTORY_LENGTH + TRAIN_HORIZON + 1 ))
export SEGMENT_LENGTH

CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/DynQueryWorldModel/core_sweep}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/DynQueryWorldModel_core_sweep}"
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/libero/phase1/dynquery_core_sweep.json}"
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

if is_true "${SMOKE}"; then
  MAX_STEPS="${MAX_STEPS_SMOKE:-100}"
  NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS_SMOKE:-4}"
  NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS_SMOKE:-4}"
  SAVE_STEPS="${SAVE_STEPS:-100}"
fi

# ---------------------------------------------------------------------------
# SKIP / OVERWRITE logic
# ---------------------------------------------------------------------------
_need_train() {
  is_true "${OVERWRITE}" && return 0
  is_true "${SKIP_EXISTING}" || return 0
  [ ! -f "${CKPT_FINAL}/dynquery_config.json" ]
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
    "use_action_conditioned_mask": int("${USE_ACTION_CONDITIONED_MASK}"),
    "predictor_mode": "${PREDICTOR_MODE}",
    "use_dynamic_residual_gate": int("${USE_DYNAMIC_RESIDUAL_GATE}"),
    "lambda_mask_dynamic": float("${LAMBDA_MASK_DYNAMIC}"),
    "lambda_query_delta_sparse": float("${LAMBDA_QUERY_DELTA_SPARSE}"),
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
    "phase0_compatible": int("${PHASE0_COMPATIBLE}"),
    "ckpt_dir": "${CKPT_DIR}",
    "cond_out": "${COND_OUT}",
    "sweep_config": "${SWEEP_CONFIG}",
}
(out / "dynquery_sweep_config_used.json").write_text(json.dumps(cfg_used, indent=2))

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
RUN_TRAIN=0; RUN_EVAL=0
case "${MODE}" in
  train_only)  RUN_TRAIN=1 ;;
  eval_only)   RUN_EVAL=1 ;;
  train_eval)  RUN_TRAIN=1; RUN_EVAL=1 ;;
  *)
    echo "[dynquery-sweep-backend] Unknown MODE=${MODE}. Use train_only|eval_only|train_eval" >&2
    exit 2
    ;;
esac

log "MODE=${MODE}  stage=${STAGE}  K=${HISTORY_LENGTH}  Q=${NUM_DYNAMIC_QUERIES}  seg_len=${SEGMENT_LENGTH}"
log "  Core1_act_cond=${USE_ACTION_CONDITIONED_MASK}  Core2_pred=${PREDICTOR_MODE}  Core3_gate=${USE_DYNAMIC_RESIDUAL_GATE}"
log "  λ_mask_dyn=${LAMBDA_MASK_DYNAMIC}  λ_q_delta=${LAMBDA_QUERY_DELTA_SPARSE}"
log "  motion_bias=${USE_MOTION_BIAS}  scorer=${USE_ACTION_FUTURE_SCORER}  λ_rank=${LAMBDA_RANK}"
log "  ckpt=${CKPT_DIR}"
log "  out=${COND_OUT}"

_save_metadata

# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
if [ "${RUN_TRAIN}" = "1" ]; then
  if ! _need_train; then
    log "SKIP train (dynquery_config.json exists and SKIP_EXISTING=1)"
  else
    if is_true "${DRY_RUN}"; then
      log "DRY_RUN: would run train_dynquery.sh with EXP_NAME=${EXP_NAME} MAX_STEPS=${MAX_STEPS:-150000}"
    else
      log "Starting train..."
      TASK_SUITE="${TASK_SUITE}" \
      EXP_NAME="${EXP_NAME}" \
      OUTPUT_ROOT="${CKPT_ROOT}" \
      SEED="${SEED}" \
      HISTORY_LENGTH="${HISTORY_LENGTH}" \
      NUM_DYNAMIC_QUERIES="${NUM_DYNAMIC_QUERIES}" \
      USE_ACTION_CONDITIONED_MASK="${USE_ACTION_CONDITIONED_MASK}" \
      PREDICTOR_MODE="${PREDICTOR_MODE}" \
      USE_DYNAMIC_RESIDUAL_GATE="${USE_DYNAMIC_RESIDUAL_GATE}" \
      LAMBDA_MASK_DYNAMIC="${LAMBDA_MASK_DYNAMIC}" \
      LAMBDA_QUERY_DELTA_SPARSE="${LAMBDA_QUERY_DELTA_SPARSE}" \
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
      SEGMENT_LENGTH="${SEGMENT_LENGTH}" \
      TRAIN_HORIZON="${TRAIN_HORIZON}" \
      MAX_STEPS="${MAX_STEPS:-150000}" \
      BATCH_SIZE="${BATCH_SIZE:-8}" \
      WORLD_MODEL_BATCH_SIZE="${WORLD_MODEL_BATCH_SIZE:-64}" \
      LR="${LR:-1e-4}" \
      SAVE_STEPS="${SAVE_STEPS:-10000}" \
      SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}" \
      LOGGING_STEPS="${LOGGING_STEPS:-20}" \
        bash "${SCRIPT_DIR}/train_dynquery.sh" "${TASK_SUITE}"
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

      if is_true "${DRY_RUN}"; then
        log "DRY_RUN: would eval at ${CKPT_FINAL}"
      else
        log "Starting eval..."
        USE_WINDOW_MANIFEST="${USE_WINDOW_MANIFEST:-0}" \
        WINDOW_MANIFEST="${WINDOW_MANIFEST:-}" \
        PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE}" \
          bash "${SCRIPT_DIR}/eval_v4_temporal_query_worldmodel.sh" "${EVAL_ARGS[@]}"

        if [ -f "${CKPT_FINAL}/dynquery_config.json" ] && [ ! -f "${COND_OUT}/dynquery_config.json" ]; then
          cp "${CKPT_FINAL}/dynquery_config.json" "${COND_OUT}/dynquery_config.json"
        fi
        log "Eval done → ${COND_OUT}/aggregate_metrics.json"
      fi
    fi
  fi
fi

log "Condition complete."
