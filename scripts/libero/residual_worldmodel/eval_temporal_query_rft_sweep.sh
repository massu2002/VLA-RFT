#!/usr/bin/env bash
# eval_temporal_query_rft_sweep.sh — Evaluate v4 TemporalQueryResidualWM-RFT policies.
#
# Discovers actor checkpoints under:
#   checkpoints/libero/TemporalQueryResidualWM-RFT/${RFT_RUN_NAME}/${TASK_SUITE}/<exp>/global_step_*/actor
# and writes evaluation outputs under:
#   results/phase1/${RFT_RUN_NAME}_rft/<exp>/
#
# Common usage:
#   bash scripts/libero/residual_worldmodel/eval_temporal_query_rft_sweep.sh
#
# Useful overrides:
#   RFT_RUN_NAME=v4_improved_spatial
#   TASK_SUITE=spatial
#   RFT_STEP=latest|400
#   EXPERIMENTS="phase1_v4a_q8_k2_motion_rft_lpips_mae ..."
#   NUM_TRIALS=50
#   SEED=7
#   SEED_SUBDIR=1              # write to <OUT_ROOT>/<exp>/seed<SEED>/
#   SMOKE=1
#   DRY_RUN=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log() { echo "[v4-rft-eval] $(date +%H:%M:%S) $*"; }

TASK_SUITE="${TASK_SUITE:-spatial}"
RFT_RUN_NAME="${RFT_RUN_NAME:-v4_improved_spatial}"
RFT_CKPT_ROOT="${RFT_CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM-RFT/${RFT_RUN_NAME}}"
BASELINE_RFT_CKPT_ROOT="${BASELINE_RFT_CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM-RFT/baseline_phase0_ar_pixel}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/${RFT_RUN_NAME}_rft}"
EXPERIMENTS="${EXPERIMENTS:-auto}"
RFT_STEP="${RFT_STEP:-latest}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-7}"
SEED_SUBDIR="${SEED_SUBDIR:-0}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv5090_eval}"
BASE_VLA_PATH="${BASE_VLA_PATH:-${REPO_ROOT}/checkpoints/libero/Base_VLA/${TASK_SUITE}}"
ROLLOUT_PHASE_DIR="${ROLLOUT_PHASE_DIR:-${RFT_RUN_NAME}_rft}"
NUM_NODES="${NUM_NODES:-${NUM_PCS:-${SHARD_COUNT:-1}}}"
NODE_INDEX="${NODE_INDEX:-${PC_INDEX:-${SHARD_INDEX:-0}}}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

TASK_CKPT_DIR="${RFT_CKPT_ROOT}/${TASK_SUITE}"
BASELINE_TASK_CKPT_DIR="${BASELINE_RFT_CKPT_ROOT}/${TASK_SUITE}"
if is_true "${SEED_SUBDIR}"; then
  LOG_ROOT="${OUT_ROOT}/logs/seed${SEED}/node${NODE_INDEX}_of_${NUM_NODES}"
  MANIFEST="${OUT_ROOT}/rft_eval_manifest_seed${SEED}_node${NODE_INDEX}_of_${NUM_NODES}.tsv"
else
  LOG_ROOT="${OUT_ROOT}/logs/node${NODE_INDEX}_of_${NUM_NODES}"
  MANIFEST="${OUT_ROOT}/rft_eval_manifest_node${NODE_INDEX}_of_${NUM_NODES}.tsv"
fi
if is_true "${SEED_SUBDIR}"; then
  SETTINGS_MD="${OUT_ROOT}/experiment_settings_seed${SEED}.md"
else
  SETTINGS_MD="${OUT_ROOT}/experiment_settings.md"
fi

if ! [[ "${NUM_NODES}" =~ ^[0-9]+$ ]] || [ "${NUM_NODES}" -lt 1 ]; then
  echo "NUM_NODES must be a positive integer, got ${NUM_NODES}" >&2
  exit 2
fi
if ! [[ "${NODE_INDEX}" =~ ^[0-9]+$ ]] || [ "${NODE_INDEX}" -lt 0 ] || [ "${NODE_INDEX}" -ge "${NUM_NODES}" ]; then
  echo "NODE_INDEX must be in [0, NUM_NODES-1], got NODE_INDEX=${NODE_INDEX}, NUM_NODES=${NUM_NODES}" >&2
  exit 2
fi
if [ ! -d "${TASK_CKPT_DIR}" ] && [ ! -d "${BASELINE_TASK_CKPT_DIR}" ]; then
  echo "RFT checkpoint task directory not found: ${TASK_CKPT_DIR}" >&2
  echo "Baseline RFT checkpoint task directory not found: ${BASELINE_TASK_CKPT_DIR}" >&2
  exit 2
fi
if [ ! -d "${BASE_VLA_PATH}" ]; then
  echo "BASE_VLA_PATH not found: ${BASE_VLA_PATH}" >&2
  exit 2
fi
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "VENV_PATH not found: ${VENV_PATH}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
printf '%s\n' "${OUT_ROOT}" > "${REPO_ROOT}/results/phase1/latest_rft_eval.txt"

list_experiments() {
  if [ "${EXPERIMENTS}" = "auto" ]; then
    {
      [ -d "${TASK_CKPT_DIR}" ] && find "${TASK_CKPT_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
      [ -d "${BASELINE_TASK_CKPT_DIR}" ] && find "${BASELINE_TASK_CKPT_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
    } | sort -u
  else
    # shellcheck disable=SC2086
    printf '%s\n' ${EXPERIMENTS}
  fi
}

latest_actor_for_exp_dir() {
  local exp_dir="$1"
  local latest_file="${exp_dir}/latest_checkpointed_iteration.txt"
  local latest_step=""
  if [ -f "${latest_file}" ]; then
    latest_step="$(tr -d '[:space:]' < "${latest_file}")"
    latest_step="${latest_step#global_step_}"
    if [[ "${latest_step}" =~ ^[0-9]+$ ]] && [ -d "${exp_dir}/global_step_${latest_step}/actor" ]; then
      printf '%s\n' "${exp_dir}/global_step_${latest_step}/actor"
      return 0
    fi
  fi

  local best_step="-1"
  local best_actor=""
  local actor step_name step_num
  while IFS= read -r actor; do
    step_name="$(basename "$(dirname "${actor}")")"
    step_num="${step_name#global_step_}"
    if [[ "${step_num}" =~ ^[0-9]+$ ]] && [ "${step_num}" -gt "${best_step}" ]; then
      best_step="${step_num}"
      best_actor="${actor}"
    fi
  done < <(find "${exp_dir}" -mindepth 2 -maxdepth 2 -type d -name actor | sort)
  [ -n "${best_actor}" ] && printf '%s\n' "${best_actor}"
}

actor_for_exp() {
  local exp_name="$1"
  local exp_dir actor
  for exp_dir in "${TASK_CKPT_DIR}/${exp_name}" "${BASELINE_TASK_CKPT_DIR}/${exp_name}"; do
    [ -d "${exp_dir}" ] || continue
    if [ "${RFT_STEP}" = "latest" ]; then
      actor="$(latest_actor_for_exp_dir "${exp_dir}" || true)"
      [ -n "${actor}" ] && printf '%s\n' "${actor}" && return 0
    else
      actor="${exp_dir}/global_step_${RFT_STEP}/actor"
      [ -d "${actor}" ] && printf '%s\n' "${actor}" && return 0
    fi
  done
  return 1
}

step_from_actor() {
  local actor="$1"
  basename "$(dirname "${actor}")" | sed 's/^global_step_//'
}

write_settings_markdown() {
  local tmp="${SETTINGS_MD}.tmp"
  {
    echo "# v4 Improved Spatial RFT Evaluation Settings"
    echo
    echo "- Generated: $(date -Is)"
    echo "- Task suite: \`${TASK_SUITE}\`"
    echo "- RFT checkpoint root: \`${RFT_CKPT_ROOT}\`"
    echo "- Baseline RFT checkpoint root: \`${BASELINE_RFT_CKPT_ROOT}\`"
    echo "- Output root: \`${OUT_ROOT}\`"
    echo "- RFT step: \`${RFT_STEP}\`"
    echo "- Num trials per task: \`${NUM_TRIALS}\`"
    echo "- Seed: \`${SEED}\`"
    echo "- Base VLA path: \`${BASE_VLA_PATH}\`"
    echo
    echo "| Experiment | Reward | WM condition | Actor step | Actor checkpoint | Output dir |"
    echo "|---|---:|---|---:|---|---|"
    while IFS= read -r exp_name; do
      [ -z "${exp_name}" ] && continue
      local actor reward wm_condition actor_step out_dir
      actor="$(actor_for_exp "${exp_name}" || true)"
      reward="${exp_name##*_rft_}"
      wm_condition="${exp_name#phase1_}"
      wm_condition="${wm_condition%_rft_*}"
      actor_step=""
      [ -n "${actor}" ] && actor_step="$(step_from_actor "${actor}")"
      if is_true "${SEED_SUBDIR}"; then
        out_dir="${OUT_ROOT}/${exp_name}/seed${SEED}"
      else
        out_dir="${OUT_ROOT}/${exp_name}"
      fi
      echo "| \`${exp_name}\` | \`${reward}\` | \`${wm_condition}\` | \`${actor_step:-missing}\` | \`${actor:-missing}\` | \`${out_dir}\` |"
    done < <(list_experiments)
  } > "${tmp}"
  mv "${tmp}" "${SETTINGS_MD}"
}

{
  echo -e "job_id\tnode_index\tnum_nodes\texp_name\tpolicy_ckpt\trft_step\toutput_dir\tstatus\tlog"
} > "${MANIFEST}"

write_settings_markdown

log "=== v4 TemporalQueryResidualWM-RFT policy eval ==="
log "task_suite   : ${TASK_SUITE}"
log "rft_run_name : ${RFT_RUN_NAME}"
log "rft ckpt root: ${RFT_CKPT_ROOT}"
log "baseline root: ${BASELINE_RFT_CKPT_ROOT}"
log "out root     : ${OUT_ROOT}"
log "experiments  : ${EXPERIMENTS}"
log "rft step     : ${RFT_STEP}"
log "num trials   : ${NUM_TRIALS}"
log "seed         : ${SEED}"
log "seed subdir  : ${SEED_SUBDIR}"
log "node shard   : ${NODE_INDEX}/${NUM_NODES}"
log "gpu id       : ${GPU_ID}"
log "settings md  : ${SETTINGS_MD}"

job_id=0
fail_count=0
while IFS= read -r exp_name; do
  [ -z "${exp_name}" ] && continue
  case "${exp_name}" in \#*) continue ;; esac

  current_id="${job_id}"
  job_id=$((job_id + 1))
  if [ $((current_id % NUM_NODES)) -ne "${NODE_INDEX}" ]; then
    continue
  fi

  policy_ckpt="$(actor_for_exp "${exp_name}" || true)"
  if is_true "${SEED_SUBDIR}"; then
    out_dir="${OUT_ROOT}/${exp_name}/seed${SEED}"
  else
    out_dir="${OUT_ROOT}/${exp_name}"
  fi
  log_file="${LOG_ROOT}/${exp_name}.log"

  if [ -z "${policy_ckpt}" ] || [ ! -d "${policy_ckpt}" ]; then
    log "SKIP job=${current_id} ${exp_name}: actor checkpoint not found"
    echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp_name}\t${policy_ckpt:-}\t${RFT_STEP}\t${out_dir}\tskipped_missing_actor\t${log_file}" >> "${MANIFEST}"
    continue
  fi

  if [ -f "${out_dir}/success_summary.json" ] && ! is_true "${FORCE}"; then
    log "SKIP job=${current_id} ${exp_name}: existing success_summary.json (FORCE=1 to rerun)"
    echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp_name}\t${policy_ckpt}\t${RFT_STEP}\t${out_dir}\tskipped_existing\t${log_file}" >> "${MANIFEST}"
    continue
  fi

  log "RUN job=${current_id}: ${exp_name}"
  log "  actor: ${policy_ckpt}"
  echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp_name}\t${policy_ckpt}\t${RFT_STEP}\t${out_dir}\trunning\t${log_file}" >> "${MANIFEST}"

  effective_rollout_phase_dir="${ROLLOUT_PHASE_DIR}"
  if is_true "${SEED_SUBDIR}"; then
    effective_rollout_phase_dir="${ROLLOUT_PHASE_DIR}_seed${SEED}"
  fi

  set +e
  (
    TASK_SUITE="${TASK_SUITE}" \
    POLICY_CKPT="${policy_ckpt}" \
    EXP_NAME="${exp_name}" \
    OUTPUT_DIR="${out_dir}" \
    NUM_TRIALS="${NUM_TRIALS}" \
    SEED="${SEED}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    VENV_PATH="${VENV_PATH}" \
    BASE_VLA_PATH="${BASE_VLA_PATH}" \
    ROLLOUT_PHASE_DIR="${effective_rollout_phase_dir}" \
    SMOKE="${SMOKE}" \
    DRY_RUN="${DRY_RUN}" \
      bash "${SCRIPT_DIR}/eval_phase1_residual_rft.sh"
  ) > "${log_file}" 2>&1
  rc=$?
  set -e

  if [ "${rc}" -eq 0 ]; then
    if is_true "${DRY_RUN}"; then
      log "DRY-RUN job=${current_id}: ${exp_name}"
      echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp_name}\t${policy_ckpt}\t${RFT_STEP}\t${out_dir}\tdry_run\t${log_file}" >> "${MANIFEST}"
    else
      log "DONE job=${current_id}: ${exp_name}"
      echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp_name}\t${policy_ckpt}\t${RFT_STEP}\t${out_dir}\tdone\t${log_file}" >> "${MANIFEST}"
    fi
  else
    log "FAILED job=${current_id}: ${exp_name} rc=${rc}; see ${log_file}"
    echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp_name}\t${policy_ckpt}\t${RFT_STEP}\t${out_dir}\tfailed_rc_${rc}\t${log_file}" >> "${MANIFEST}"
    fail_count=$((fail_count + 1))
    is_true "${STOP_ON_FAIL}" && exit "${rc}"
  fi
done < <(list_experiments)

write_settings_markdown

if [ "${fail_count}" -gt 0 ]; then
  log "WARNING: ${fail_count} eval job(s) failed."
  exit 1
fi

log "=== v4 RFT policy eval complete ==="
log "manifest: ${MANIFEST}"
log "outputs : ${OUT_ROOT}"
