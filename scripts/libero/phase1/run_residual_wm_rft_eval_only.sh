#!/usr/bin/env bash
# run_residual_wm_rft_eval_only.sh — Evaluate Phase 1 Residual-WM RFT policies.
#
# This mirrors the Phase 1 training/RFT sharding style:
#   PC 0:
#     NUM_NODES=3 NODE_INDEX=0 RUN_NAME=phase1_residual_spatial \
#       bash scripts/libero/phase1/run_residual_wm_rft_eval_only.sh spatial
#   PC 1:
#     NUM_NODES=3 NODE_INDEX=1 RUN_NAME=phase1_residual_spatial \
#       bash scripts/libero/phase1/run_residual_wm_rft_eval_only.sh spatial
#   PC 2:
#     NUM_NODES=3 NODE_INDEX=2 RUN_NAME=phase1_residual_spatial \
#       bash scripts/libero/phase1/run_residual_wm_rft_eval_only.sh spatial
#
# By default each experiment is evaluated at its latest global_step_*/actor.
#
# Common overrides:
#   RFT_CKPT_ROOT=checkpoints/libero/PixelResidualWM-RFT
#   RFT_STEP=400              # use a fixed step instead of latest
#   NUM_TRIALS=50
#   GPU_ID=0                  # LIBERO eval is single-GPU
#   SWEEP_PRESET=auto|core|compact|roi_weights
#   DRY_RUN=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"

source "${WM_SCRIPTS}/common.sh"

log() { echo "[phase1-rft-eval] $(date +%H:%M:%S) $*"; }
is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }

TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
RUN_NAME="${RUN_NAME:-phase1_residual_spatial}"
RFT_RUN_NAME="${RFT_RUN_NAME:-${RUN_NAME}_rft}"
SWEEP_PRESET="${SWEEP_PRESET:-auto}"
NUM_NODES="${NUM_NODES:-${NUM_PCS:-${SHARD_COUNT:-1}}}"
NODE_INDEX="${NODE_INDEX:-${PC_INDEX:-${SHARD_INDEX:-0}}}"
RFT_CKPT_ROOT="${RFT_CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/PixelResidualWM-RFT}"
RFT_STEP="${RFT_STEP:-latest}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-7}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv5090_eval}"
BASE_VLA_PATH="${BASE_VLA_PATH:-${REPO_ROOT}/checkpoints/libero/Base_VLA/${TASK_SUITE}}"
ROLLOUT_PHASE_DIR="${ROLLOUT_PHASE_DIR:-rft_phase1}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
FORCE="${FORCE:-0}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/phase1_sweeps/${RFT_RUN_NAME}/rft_eval}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs/node${NODE_INDEX}_of_${NUM_NODES}}"
MANIFEST="${OUT_ROOT}/rft_eval_manifest_node${NODE_INDEX}_of_${NUM_NODES}.tsv"

if ! [[ "${NUM_NODES}" =~ ^[0-9]+$ ]] || [ "${NUM_NODES}" -lt 1 ]; then
  echo "NUM_NODES must be a positive integer, got: ${NUM_NODES}" >&2
  exit 2
fi
if ! [[ "${NODE_INDEX}" =~ ^[0-9]+$ ]] || [ "${NODE_INDEX}" -lt 0 ] || [ "${NODE_INDEX}" -ge "${NUM_NODES}" ]; then
  echo "NODE_INDEX must be in [0, NUM_NODES-1], got NODE_INDEX=${NODE_INDEX}, NUM_NODES=${NUM_NODES}" >&2
  exit 2
fi
if [ ! -d "${RFT_CKPT_ROOT}/${TASK_SUITE}" ]; then
  echo "RFT checkpoint task directory not found: ${RFT_CKPT_ROOT}/${TASK_SUITE}" >&2
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

build_preset_jobs() {
  case "${SWEEP_PRESET}" in
    compact)
      cat <<'EOF'
phase1_pixel_baseline_rft
phase1_v1_residual_rft
phase1_v1_roi_d2_g2_s05_rft
phase1_v3_roi_d2_g2_s05_w02_rft
EOF
      ;;
    core)
      cat <<'EOF'
phase1_pixel_baseline_rft
phase1_v1_residual_rft
phase1_v1_roi_d2_g2_s05_rft
phase1_v3_residual_w02_rft
phase1_v3_roi_d2_g2_s05_w02_rft
phase1_v3_roi_d4_g2_s05_w02_rft
phase1_v3_roi_d2_g4_s05_w02_rft
phase1_v3_roi_d2_g2_s01_w02_rft
phase1_v3_roi_d2_g2_s05_w05_rft
EOF
      ;;
    roi_weights)
      cat <<'EOF'
phase1_v3_roi_d1_g2_s05_w02_rft
phase1_v3_roi_d2_g2_s05_w02_rft
phase1_v3_roi_d4_g2_s05_w02_rft
phase1_v3_roi_d2_g1_s05_w02_rft
phase1_v3_roi_d2_g4_s05_w02_rft
phase1_v3_roi_d2_g2_s01_w02_rft
phase1_v3_roi_d2_g2_s10_w02_rft
phase1_v3_roi_d2_g2_s05_w05_rft
EOF
      ;;
    auto)
      find "${RFT_CKPT_ROOT}/${TASK_SUITE}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
      ;;
    *)
      echo "Unknown SWEEP_PRESET=${SWEEP_PRESET}. Use auto, core, compact, or roi_weights." >&2
      exit 2
      ;;
  esac
}

latest_actor_for_exp() {
  local exp_dir="$1"
  local best_step="-1"
  local best_actor=""
  local actor=""
  local step_name=""
  local step_num=""

  while IFS= read -r actor; do
    step_name="$(basename "$(dirname "${actor}")")"
    step_num="${step_name#global_step_}"
    if [[ "${step_num}" =~ ^[0-9]+$ ]] && [ "${step_num}" -gt "${best_step}" ]; then
      best_step="${step_num}"
      best_actor="${actor}"
    fi
  done < <(find "${exp_dir}" -mindepth 2 -maxdepth 2 -type d -name actor | sort)

  if [ -n "${best_actor}" ]; then
    printf '%s\n' "${best_actor}"
  fi
}

actor_for_exp() {
  local exp_name="$1"
  local exp_dir="${RFT_CKPT_ROOT}/${TASK_SUITE}/${exp_name}"
  if [ ! -d "${exp_dir}" ]; then
    return 1
  fi
  if [ "${RFT_STEP}" = "latest" ]; then
    latest_actor_for_exp "${exp_dir}"
  else
    local actor="${exp_dir}/global_step_${RFT_STEP}/actor"
    if [ -d "${actor}" ]; then
      printf '%s\n' "${actor}"
    fi
  fi
}

{
  echo -e "job_id\tnode_index\tnum_nodes\texp_name\tpolicy_ckpt\trft_step\toutput_dir\tstatus\tlog"
} > "${MANIFEST}"

log "=== Phase 1 RFT policy eval sweep ==="
log "task_suite   : ${TASK_SUITE}"
log "rft_run_name : ${RFT_RUN_NAME}"
log "preset       : ${SWEEP_PRESET}"
log "node shard   : ${NODE_INDEX}/${NUM_NODES}"
log "rft ckpt root: ${RFT_CKPT_ROOT}"
log "rft step     : ${RFT_STEP}"
log "num trials   : ${NUM_TRIALS}"
log "gpu id       : ${GPU_ID}"
log "rollout phase: ${ROLLOUT_PHASE_DIR}"
log "out root     : ${OUT_ROOT}"

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
  out_dir="${OUT_ROOT}/${exp_name}"
  log_file="${LOG_ROOT}/${exp_name}.log"

  if [ -z "${policy_ckpt}" ] || [ ! -d "${policy_ckpt}" ]; then
    log "SKIP job=${current_id} ${exp_name}: actor checkpoint not found"
    echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp_name}\t${policy_ckpt:-}\t${RFT_STEP}\t${out_dir}\tskipped_missing_actor\t${log_file}" >> "${MANIFEST}"
    continue
  fi

  if [ -f "${out_dir}/success_summary.json" ] && ! is_true "${FORCE}"; then
    log "SKIP job=${current_id} ${exp_name}: existing success_summary.json (set FORCE=1 to rerun)"
    echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp_name}\t${policy_ckpt}\t${RFT_STEP}\t${out_dir}\tskipped_existing\t${log_file}" >> "${MANIFEST}"
    continue
  fi

  log "RUN job=${current_id}: ${exp_name}"
  log "  actor: ${policy_ckpt}"
  echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp_name}\t${policy_ckpt}\t${RFT_STEP}\t${out_dir}\trunning\t${log_file}" >> "${MANIFEST}"

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
    ROLLOUT_PHASE_DIR="${ROLLOUT_PHASE_DIR}" \
    SMOKE="${SMOKE}" \
    DRY_RUN="${DRY_RUN}" \
      bash "${WM_SCRIPTS}/eval_phase1_residual_rft.sh"
  ) > "${log_file}" 2>&1
  rc=$?
  set -e

  if [ "${rc}" -eq 0 ]; then
    log "DONE job=${current_id}: ${exp_name}"
  else
    log "FAILED job=${current_id}: ${exp_name} rc=${rc}; see ${log_file}"
    fail_count=$((fail_count + 1))
    if is_true "${STOP_ON_FAIL}"; then
      exit "${rc}"
    fi
  fi
done < <(build_preset_jobs)

if [ "${fail_count}" -gt 0 ]; then
  log "WARNING: ${fail_count} eval job(s) failed."
  exit 1
fi

log "=== RFT policy eval sweep complete ==="
log "manifest: ${MANIFEST}"
log "logs    : ${LOG_ROOT}"
log "outputs : ${OUT_ROOT}"
