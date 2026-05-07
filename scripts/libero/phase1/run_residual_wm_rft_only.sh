#!/usr/bin/env bash
# run_residual_wm_rft_only.sh — Phase 1 Residual WM RFT launcher.
#
# This mirrors run_residual_wm_train_only.sh sharding:
#   PC 0:
#     NUM_NODES=3 NODE_INDEX=0 RUN_NAME=phase1_residual_spatial \
#       bash scripts/libero/phase1/run_residual_wm_rft_only.sh spatial
#   PC 1:
#     NUM_NODES=3 NODE_INDEX=1 RUN_NAME=phase1_residual_spatial \
#       bash scripts/libero/phase1/run_residual_wm_rft_only.sh spatial
#   PC 2:
#     NUM_NODES=3 NODE_INDEX=2 RUN_NAME=phase1_residual_spatial \
#       bash scripts/libero/phase1/run_residual_wm_rft_only.sh spatial
#
# Each assigned RFT job is run one by one on all local visible GPUs.
#
# Common overrides:
#   RUN_NAME=phase1_residual_spatial
#   RFT_RUN_NAME=phase1_residual_spatial_rft
#   SWEEP_PRESET=core | compact | roi_weights
#   NUM_NODES=3 NODE_INDEX=0
#   RFT_STEPS=400
#   RFT_CKPT_ROOT=checkpoints/libero/PixelResidualWM-RFT
#   DRY_RUN=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"

source "${WM_SCRIPTS}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log() { echo "[phase1-rft] $(date +%H:%M:%S) $*"; }

TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
RUN_NAME="${RUN_NAME:-phase1_residual_spatial}"
RFT_RUN_NAME="${RFT_RUN_NAME:-${RUN_NAME}_rft}"
SWEEP_PRESET="${SWEEP_PRESET:-core}"
SEED="${SEED:-42}"
RFT_SEED="${RFT_SEED:-7}"
NUM_NODES="${NUM_NODES:-${NUM_PCS:-${SHARD_COUNT:-1}}}"
NODE_INDEX="${NODE_INDEX:-${PC_INDEX:-${SHARD_INDEX:-0}}}"
GPU_IDS="${GPU_IDS:-auto}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-auto}"
RFT_STEPS="${RFT_STEPS:-400}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/PixelResidualWM/phase1_sweeps/${RUN_NAME}}"
RFT_CKPT_ROOT="${RFT_CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/PixelResidualWM-RFT}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/phase1_sweeps/${RFT_RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs/node${NODE_INDEX}_of_${NUM_NODES}}"
MANIFEST="${OUT_ROOT}/rft_manifest_node${NODE_INDEX}_of_${NUM_NODES}.tsv"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
printf '%s\n' "${OUT_ROOT}" > "${REPO_ROOT}/results/phase1/latest_rft_sweep.txt"

auto_gpu_ids() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ] && [ "${CUDA_VISIBLE_DEVICES}" != "void" ]; then
    echo "${CUDA_VISIBLE_DEVICES}"
    return
  fi
  local n
  n="$(detect_gpu_count)"
  local ids=()
  local i
  for ((i = 0; i < n; i++)); do
    ids+=("${i}")
  done
  local IFS=,
  echo "${ids[*]}"
}

if [ "${GPU_IDS}" = "auto" ]; then
  GPU_IDS="$(auto_gpu_ids)"
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS// /,}"
if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then
  echo "GPU_IDS is empty" >&2
  exit 2
fi
if [ "${N_GPUS_PER_NODE}" = "auto" ]; then
  N_GPUS_PER_NODE="${#GPU_ARRAY[@]}"
fi

if ! [[ "${NUM_NODES}" =~ ^[0-9]+$ ]] || [ "${NUM_NODES}" -lt 1 ]; then
  echo "NUM_NODES must be a positive integer, got: ${NUM_NODES}" >&2
  exit 2
fi
if ! [[ "${NODE_INDEX}" =~ ^[0-9]+$ ]] || [ "${NODE_INDEX}" -lt 0 ] || [ "${NODE_INDEX}" -ge "${NUM_NODES}" ]; then
  echo "NODE_INDEX must be in [0, NUM_NODES-1], got NODE_INDEX=${NODE_INDEX}, NUM_NODES=${NUM_NODES}" >&2
  exit 2
fi

# Each row:
# exp_name|target_mode|generation
# pixel_baseline|pixel|baseline
build_jobs() {
  case "${SWEEP_PRESET}" in
    compact)
      cat <<'EOF'
v1_residual|pixel_residual|v1
v1_roi_d2_g2_s05|pixel_residual_roi_dynamic|v1
v3_roi_d2_g2_s05_w02|pixel_residual_roi_dynamic|v3
EOF
      ;;
    roi_weights)
      cat <<'EOF'
v3_roi_d1_g2_s05_w02|pixel_residual_roi_dynamic|v3
v3_roi_d2_g2_s05_w02|pixel_residual_roi_dynamic|v3
v3_roi_d4_g2_s05_w02|pixel_residual_roi_dynamic|v3
v3_roi_d2_g1_s05_w02|pixel_residual_roi_dynamic|v3
v3_roi_d2_g4_s05_w02|pixel_residual_roi_dynamic|v3
v3_roi_d2_g2_s01_w02|pixel_residual_roi_dynamic|v3
v3_roi_d2_g2_s10_w02|pixel_residual_roi_dynamic|v3
v3_roi_d2_g2_s05_w05|pixel_residual_roi_dynamic|v3
EOF
      ;;
    core)
      cat <<'EOF'
v1_residual|pixel_residual|v1
v1_roi_d2_g2_s05|pixel_residual_roi_dynamic|v1
v3_residual_w02|pixel_residual|v3
v3_roi_d2_g2_s05_w02|pixel_residual_roi_dynamic|v3
v3_roi_d4_g2_s05_w02|pixel_residual_roi_dynamic|v3
v3_roi_d2_g4_s05_w02|pixel_residual_roi_dynamic|v3
v3_roi_d2_g2_s01_w02|pixel_residual_roi_dynamic|v3
v3_roi_d2_g2_s05_w05|pixel_residual_roi_dynamic|v3
EOF
      ;;
    *)
      echo "Unknown SWEEP_PRESET=${SWEEP_PRESET}. Use core, compact, or roi_weights." >&2
      exit 2
      ;;
  esac
}

target_mode_dir() {
  local mode="$1"
  echo "${CKPT_ROOT}/${TASK_SUITE}/${mode}"
}

wm_ckpt_for() {
  local exp="$1"
  local mode="$2"
  echo "$(target_mode_dir "${mode}")/${exp}/s${SEED}"
}

{
  echo -e "job_id\tnode_index\tnum_nodes\texp_name\ttarget_mode\tgeneration\twm_ckpt\trft_exp_name\trft_ckpt_root\toutput_dir\tstatus\tlog"
} > "${MANIFEST}"

log "=== Phase 1 Residual WM RFT sweep ==="
log "task_suite       : ${TASK_SUITE}"
log "wm_run_name      : ${RUN_NAME}"
log "rft_run_name     : ${RFT_RUN_NAME}"
log "preset           : ${SWEEP_PRESET}"
log "node shard       : ${NODE_INDEX}/${NUM_NODES}"
log "local gpu ids    : ${GPU_IDS} (n_gpus_per_node=${N_GPUS_PER_NODE})"
log "wm ckpt root     : ${CKPT_ROOT}"
log "rft ckpt root    : ${RFT_CKPT_ROOT}"
log "out root         : ${OUT_ROOT}"
log "rft steps        : ${RFT_STEPS}"

setup_env

job_id=0
fail_count=0
while IFS='|' read -r exp mode gen; do
  [ -z "${exp}" ] && continue
  case "${exp}" in \#*) continue ;; esac

  current_id="${job_id}"
  job_id=$((job_id + 1))
  if [ $((current_id % NUM_NODES)) -ne "${NODE_INDEX}" ]; then
    continue
  fi

  wm_ckpt="$(wm_ckpt_for "${exp}" "${mode}")"
  wm_cfg="${wm_ckpt}/pixel_residual_config.json"
  rft_exp="${exp}_rft"
  out_dir="${OUT_ROOT}/${rft_exp}"
  log_file="${LOG_ROOT}/${rft_exp}.log"

  if [ ! -d "${wm_ckpt}" ]; then
    log "SKIP job=${current_id} ${exp}: missing checkpoint ${wm_ckpt}"
    echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp}\t${mode}\t${gen}\t${wm_ckpt}\t${rft_exp}\t${RFT_CKPT_ROOT}\t${out_dir}\tskipped_missing_ckpt\t${log_file}" >> "${MANIFEST}"
    continue
  fi
  if [ ! -f "${wm_cfg}" ]; then
    log "SKIP job=${current_id} ${exp}: missing config ${wm_cfg}"
    echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp}\t${mode}\t${gen}\t${wm_ckpt}\t${rft_exp}\t${RFT_CKPT_ROOT}\t${out_dir}\tskipped_missing_config\t${log_file}" >> "${MANIFEST}"
    continue
  fi

  log "RUN job=${current_id}: ${exp} -> ${rft_exp}"
  echo -e "${current_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp}\t${mode}\t${gen}\t${wm_ckpt}\t${rft_exp}\t${RFT_CKPT_ROOT}\t${out_dir}\trunning\t${log_file}" >> "${MANIFEST}"

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    TASK_SUITE="${TASK_SUITE}" \
    MODEL_GENERATION="${gen}" \
    TARGET_MODE="${mode}" \
    WORLD_MODEL_CKPT="${wm_ckpt}" \
    WORLD_MODEL_CONFIG="${wm_cfg}" \
    EXP_NAME="${rft_exp}" \
    OUTPUT_DIR="${out_dir}" \
    RFT_CKPT_ROOT="${RFT_CKPT_ROOT}" \
    RFT_STEPS="${RFT_STEPS}" \
    SEED="${RFT_SEED}" \
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
    SMOKE="${SMOKE}" \
    DRY_RUN="${DRY_RUN}" \
      bash "${WM_SCRIPTS}/post_train_phase1_residual_rft.sh"
  ) > "${log_file}" 2>&1
  rc=$?
  set -e

  if [ "${rc}" -eq 0 ]; then
    log "DONE job=${current_id}: ${rft_exp}"
  else
    log "FAILED job=${current_id}: ${rft_exp} rc=${rc}; see ${log_file}"
    fail_count=$((fail_count + 1))
    if is_true "${STOP_ON_FAIL}"; then
      exit "${rc}"
    fi
  fi
done < <(build_jobs)

if [ "${fail_count}" -gt 0 ]; then
  log "WARNING: ${fail_count} RFT job(s) failed."
  exit 1
fi

log "=== RFT sweep complete ==="
log "manifest : ${MANIFEST}"
log "logs     : ${LOG_ROOT}"
log "rft ckpts: ${RFT_CKPT_ROOT}"
