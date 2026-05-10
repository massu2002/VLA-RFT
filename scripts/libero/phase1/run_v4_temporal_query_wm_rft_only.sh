#!/usr/bin/env bash
# run_v4_temporal_query_wm_rft_only.sh — Phase 1 v4 WM → RFT pipeline launcher.
#
# Runs RFT training using a pre-trained v4 TemporalDynamicQueryResidualWM as reward model.
# Supports three reward modes:
#   visual     — negative MSE reward (same as v1/v3)
#   rank_score — ActionFutureScorer ranking signal only (v4b required)
#   hybrid     — alpha * visual + beta * rank_score (v4c, default for v4b checkpoints)
#
# Usage:
#   WORLD_MODEL_CKPT=checkpoints/libero/TemporalQueryResidualWM/spatial/v4b/s42 \
#   WORLD_REWARD_TYPE=hybrid \
#     bash scripts/libero/phase1/run_v4_temporal_query_wm_rft_only.sh spatial
#
# Smoke test (no real checkpoint required for v4a stage):
#   SMOKE=1 WORLD_REWARD_TYPE=hybrid \
#     bash scripts/libero/phase1/run_v4_temporal_query_wm_rft_only.sh spatial
#
# Sweep over multiple v4 checkpoints:
#   SWEEP=1 CKPT_ROOT=checkpoints/libero/TemporalQueryResidualWM/phase1_sweeps/v4_sweep \
#   WORLD_REWARD_TYPE=hybrid NUM_NODES=2 NODE_INDEX=0 \
#     bash scripts/libero/phase1/run_v4_temporal_query_wm_rft_only.sh spatial
#
# Useful overrides:
#   WORLD_REWARD_TYPE=visual | rank_score | hybrid  (default: hybrid)
#   RANK_REWARD_ALPHA=0.2   RANK_REWARD_BETA=0.8    (hybrid weights)
#   NORMALIZE_RANK_REWARD=1  CLIP_RANK_REWARD=1  RANK_REWARD_CLIP_VALUE=5.0
#   RFT_STEPS=400
#   SMOKE=1 for smoke test
#   DRY_RUN=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"

source "${WM_SCRIPTS}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log() { echo "[phase1-v4-rft] $(date +%H:%M:%S) $*"; }

TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
SWEEP="${SWEEP:-0}"

# v4 RFT defaults
WORLD_REWARD_TYPE="${WORLD_REWARD_TYPE:-hybrid}"
export WORLD_REWARD_TYPE
export RANK_REWARD_ALPHA="${RANK_REWARD_ALPHA:-0.2}"
export RANK_REWARD_BETA="${RANK_REWARD_BETA:-0.8}"
export NORMALIZE_RANK_REWARD="${NORMALIZE_RANK_REWARD:-1}"
export CLIP_RANK_REWARD="${CLIP_RANK_REWARD:-1}"
export RANK_REWARD_CLIP_VALUE="${RANK_REWARD_CLIP_VALUE:-5.0}"

RFT_STEPS="${RFT_STEPS:-400}"
SEED="${SEED:-42}"
RFT_SEED="${RFT_SEED:-7}"
NUM_NODES="${NUM_NODES:-${NUM_PCS:-${SHARD_COUNT:-1}}}"
NODE_INDEX="${NODE_INDEX:-${PC_INDEX:-${SHARD_INDEX:-0}}}"
GPU_IDS="${GPU_IDS:-auto}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-auto}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

DATE_TAG=$(timestamp)
RFT_RUN_NAME="${RFT_RUN_NAME:-v4_rft_${WORLD_REWARD_TYPE}_${DATE_TAG}_${TASK_SUITE}}"
RFT_CKPT_ROOT="${RFT_CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM-RFT}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/v4_rft/${RFT_RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs/node${NODE_INDEX}_of_${NUM_NODES}}"
MANIFEST="${OUT_ROOT}/rft_manifest_node${NODE_INDEX}_of_${NUM_NODES}.tsv"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
printf '%s\n' "${OUT_ROOT}" > "${REPO_ROOT}/results/phase1/latest_rft_sweep.txt"

auto_gpu_ids() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ] && [ "${CUDA_VISIBLE_DEVICES}" != "void" ]; then
    echo "${CUDA_VISIBLE_DEVICES}"; return
  fi
  local n; n="$(detect_gpu_count)"
  local ids=(); local i
  for ((i = 0; i < n; i++)); do ids+=("${i}"); done
  local IFS=,; echo "${ids[*]}"
}

[ "${GPU_IDS}" = "auto" ] && GPU_IDS="$(auto_gpu_ids)"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS// /,}"
[ "${N_GPUS_PER_NODE}" = "auto" ] && N_GPUS_PER_NODE="${#GPU_ARRAY[@]}"

{
  echo -e "job_id\tnode_index\texp_name\twm_ckpt\treward_type\trft_ckpt_root\toutput_dir\tstatus\tlog"
} > "${MANIFEST}"

log "=== Phase 1 v4 RFT sweep ==="
log "task_suite       : ${TASK_SUITE}"
log "rft_run_name     : ${RFT_RUN_NAME}"
log "reward_type      : ${WORLD_REWARD_TYPE}"
log "alpha/beta       : ${RANK_REWARD_ALPHA} / ${RANK_REWARD_BETA}"
log "node shard       : ${NODE_INDEX}/${NUM_NODES}"
log "rft_steps        : ${RFT_STEPS}"
log "rft_ckpt_root    : ${RFT_CKPT_ROOT}"
log "out_root         : ${OUT_ROOT}"

setup_env

# ---------------------------------------------------------------------------
# Collect checkpoints to run
# ---------------------------------------------------------------------------
collect_ckpts() {
  if ! is_true "${SWEEP}"; then
    # Single-checkpoint mode
    WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:?'Set WORLD_MODEL_CKPT env var to a v4 checkpoint directory'}"
    echo "${WORLD_MODEL_CKPT}"
  else
    # Sweep mode — discover all v4 checkpoint dirs under CKPT_ROOT
    CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM}"
    find "${CKPT_ROOT}" -name "v4_config.json" -exec dirname {} \; | sort
  fi
}

job_id=0
fail_count=0

while IFS= read -r wm_ckpt; do
  [ -z "${wm_ckpt}" ] && continue
  [ ! -d "${wm_ckpt}" ] && log "SKIP (missing): ${wm_ckpt}" && continue

  current_id="${job_id}"
  job_id=$((job_id + 1))
  if [ $((current_id % NUM_NODES)) -ne "${NODE_INDEX}" ]; then
    continue
  fi

  exp_name="$(basename "${wm_ckpt}")_rft_${WORLD_REWARD_TYPE}"
  out_dir="${OUT_ROOT}/${exp_name}"
  log_file="${LOG_ROOT}/${exp_name}.log"

  log "RUN job=${current_id}: ${exp_name}"
  echo -e "${current_id}\t${NODE_INDEX}\t${exp_name}\t${wm_ckpt}\t${WORLD_REWARD_TYPE}\t${RFT_CKPT_ROOT}\t${out_dir}\trunning\t${log_file}" >> "${MANIFEST}"

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    TASK_SUITE="${TASK_SUITE}" \
    MODEL_GENERATION="v4" \
    TARGET_MODE="temporal_query_residual" \
    WORLD_MODEL_CKPT="${wm_ckpt}" \
    WORLD_MODEL_CONFIG="${wm_ckpt}/v4_config.json" \
    EXP_NAME="${exp_name}" \
    OUTPUT_DIR="${out_dir}" \
    RFT_CKPT_ROOT="${RFT_CKPT_ROOT}" \
    RFT_STEPS="${RFT_STEPS}" \
    SEED="${RFT_SEED}" \
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
    SMOKE="${SMOKE}" \
    DRY_RUN="${DRY_RUN}" \
    WORLD_REWARD_TYPE="${WORLD_REWARD_TYPE}" \
    RANK_REWARD_ALPHA="${RANK_REWARD_ALPHA}" \
    RANK_REWARD_BETA="${RANK_REWARD_BETA}" \
    NORMALIZE_RANK_REWARD="${NORMALIZE_RANK_REWARD}" \
    CLIP_RANK_REWARD="${CLIP_RANK_REWARD}" \
    RANK_REWARD_CLIP_VALUE="${RANK_REWARD_CLIP_VALUE}" \
    USER_EXTRA_RFT_ARGS="world_reward.type=${WORLD_REWARD_TYPE} world_reward.rank_alpha=${RANK_REWARD_ALPHA} world_reward.rank_beta=${RANK_REWARD_BETA} ${USER_EXTRA_RFT_ARGS:-}" \
      bash "${WM_SCRIPTS}/post_train_phase1_residual_rft.sh"
  ) > "${log_file}" 2>&1
  rc=$?
  set -e

  if [ "${rc}" -eq 0 ]; then
    log "DONE job=${current_id}: ${exp_name}"
  else
    log "FAILED job=${current_id}: ${exp_name} rc=${rc}; see ${log_file}"
    fail_count=$((fail_count + 1))
    is_true "${STOP_ON_FAIL}" && exit "${rc}"
  fi
done < <(collect_ckpts)

if [ "${fail_count}" -gt 0 ]; then
  log "WARNING: ${fail_count} RFT job(s) failed."
  exit 1
fi

log "=== v4 RFT sweep complete ==="
log "manifest  : ${MANIFEST}"
log "logs      : ${LOG_ROOT}"
log "rft ckpts : ${RFT_CKPT_ROOT}"
