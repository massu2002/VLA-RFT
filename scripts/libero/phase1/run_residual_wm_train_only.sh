#!/usr/bin/env bash
# run_residual_wm_train_only.sh — Phase 1 Residual WM train launcher.
#
# Default behavior remains a thin train-only wrapper:
#   bash scripts/libero/phase1/run_residual_wm_train_only.sh spatial
#
# Sweep launcher for one node with multiple GPUs:
#   SWEEP=1 RUN_NAME=phase1_core bash scripts/libero/phase1/run_residual_wm_train_only.sh spatial
#
# In this default mode, every hyperparameter combination is trained one by one,
# and each training job uses all local visible GPUs via torchrun/DDP.
#
# Optional multi-node/manual split mode. Run the same RUN_NAME on each node and
# set NODE_INDEX to split jobs by modulo:
#   Node 0: SWEEP=1 NUM_NODES=3 NODE_INDEX=0 RUN_NAME=phase1_core ...
#   Node 1: SWEEP=1 NUM_NODES=3 NODE_INDEX=1 RUN_NAME=phase1_core ...
#   Node 2: SWEEP=1 NUM_NODES=3 NODE_INDEX=2 RUN_NAME=phase1_core ...
#
# Local GPUs are auto-detected by default. Each job uses all local visible GPUs
# via torchrun/DDP. Override GPU_IDS only when you want a specific subset,
# e.g. GPU_IDS=0,2.
#
# Useful overrides:
#   RUN_NAME=phase1_localdyn_$(date +%Y%m%d_%H%M%S)
#   MAX_STEPS=150000 WORLD_MODEL_BATCH_SIZE=16 BATCH_SIZE=1 TRAIN_HORIZON=7 SEED=42
#   SWEEP_PRESET=core | roi_weights | compact
#   NUM_NODES=1 NODE_INDEX=0
#   DRY_RUN=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"

source "${WM_SCRIPTS}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log() { echo "[phase1-train] $(date +%H:%M:%S) $*"; }

TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
SWEEP="${SWEEP:-0}"

if ! is_true "${SWEEP}"; then
  export SKIP_EVAL=1
  exec bash "${SCRIPT_DIR}/run_residual_wm_eval.sh" "${TASK_SUITE}"
fi

# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------
SEED="${SEED:-42}"
SWEEP_PRESET="${SWEEP_PRESET:-core}"
DATE_TAG=$(timestamp)
RUN_NAME="${RUN_NAME:-${DATE_TAG}_${TASK_SUITE}_${SWEEP_PRESET}}"
NUM_NODES="${NUM_NODES:-${NUM_PCS:-${SHARD_COUNT:-1}}}"
NODE_INDEX="${NODE_INDEX:-${PC_INDEX:-${SHARD_INDEX:-0}}}"
GPU_IDS="${GPU_IDS:-auto}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/PixelResidualWM/phase1_sweeps/${RUN_NAME}}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/phase1_sweeps/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
NODE_TAG="${NODE_TAG:-node${NODE_INDEX}_of_${NUM_NODES}}"
PC_TAG="${PC_TAG:-${NODE_TAG}}"
LOG_ROOT="${LOG_ROOT}/${PC_TAG}"
MANIFEST="${OUT_ROOT}/sweep_manifest_${PC_TAG}.tsv"
SUMMARY="${OUT_ROOT}/README.md"
EVAL_COMMANDS="${OUT_ROOT}/eval_commands.sh"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
printf '%s\n' "${OUT_ROOT}" > "${REPO_ROOT}/results/phase1/latest_train_sweep.txt"

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
if [ -z "${MAX_PARALLEL}" ]; then
  MAX_PARALLEL=1
fi
LOCAL_NPROC="${#GPU_ARRAY[@]}"
if ! [[ "${NUM_NODES}" =~ ^[0-9]+$ ]] || [ "${NUM_NODES}" -lt 1 ]; then
  echo "NUM_NODES must be a positive integer, got: ${NUM_NODES}" >&2
  exit 2
fi
if ! [[ "${NODE_INDEX}" =~ ^[0-9]+$ ]] || [ "${NODE_INDEX}" -lt 0 ] || [ "${NODE_INDEX}" -ge "${NUM_NODES}" ]; then
  echo "NODE_INDEX must be in [0, NUM_NODES-1], got NODE_INDEX=${NODE_INDEX}, NUM_NODES=${NUM_NODES}" >&2
  exit 2
fi

# Each row:
# exp_name|target_mode|generation|lambda_dynamic|lambda_gripper|lambda_static|lambda_write|dyn_threshold|roi_size|write_bias|spatial_action|write_mask
build_jobs() {
  case "${SWEEP_PRESET}" in
    compact)
      cat <<'EOF'
v1_residual|pixel_residual|v1|0.0|0.0|0.0|0.0|0.05|64|-2.0|0|0
v1_roi_d2_g2_s05|pixel_residual_roi_dynamic|v1|2.0|2.0|0.5|0.0|0.05|64|-2.0|0|0
v3_roi_d2_g2_s05_w02|pixel_residual_roi_dynamic|v3|2.0|2.0|0.5|0.2|0.05|64|-2.0|1|1
EOF
      ;;
    roi_weights)
      cat <<'EOF'
v3_roi_d1_g2_s05_w02|pixel_residual_roi_dynamic|v3|1.0|2.0|0.5|0.2|0.05|64|-2.0|1|1
v3_roi_d2_g2_s05_w02|pixel_residual_roi_dynamic|v3|2.0|2.0|0.5|0.2|0.05|64|-2.0|1|1
v3_roi_d4_g2_s05_w02|pixel_residual_roi_dynamic|v3|4.0|2.0|0.5|0.2|0.05|64|-2.0|1|1
v3_roi_d2_g1_s05_w02|pixel_residual_roi_dynamic|v3|2.0|1.0|0.5|0.2|0.05|64|-2.0|1|1
v3_roi_d2_g4_s05_w02|pixel_residual_roi_dynamic|v3|2.0|4.0|0.5|0.2|0.05|64|-2.0|1|1
v3_roi_d2_g2_s01_w02|pixel_residual_roi_dynamic|v3|2.0|2.0|0.1|0.2|0.05|64|-2.0|1|1
v3_roi_d2_g2_s10_w02|pixel_residual_roi_dynamic|v3|2.0|2.0|1.0|0.2|0.05|64|-2.0|1|1
v3_roi_d2_g2_s05_w05|pixel_residual_roi_dynamic|v3|2.0|2.0|0.5|0.5|0.05|64|-2.0|1|1
EOF
      ;;
    core)
      cat <<'EOF'
pixel_baseline|pixel|baseline|0.0|0.0|0.0|0.0|0.05|64|-2.0|0|0
v1_residual|pixel_residual|v1|0.0|0.0|0.0|0.0|0.05|64|-2.0|0|0
v1_roi_d2_g2_s05|pixel_residual_roi_dynamic|v1|2.0|2.0|0.5|0.0|0.05|64|-2.0|0|0
v3_residual_w02|pixel_residual|v3|0.0|0.0|0.0|0.2|0.05|64|-2.0|1|1
v3_roi_d2_g2_s05_w02|pixel_residual_roi_dynamic|v3|2.0|2.0|0.5|0.2|0.05|64|-2.0|1|1
v3_roi_d4_g2_s05_w02|pixel_residual_roi_dynamic|v3|4.0|2.0|0.5|0.2|0.05|64|-2.0|1|1
v3_roi_d2_g4_s05_w02|pixel_residual_roi_dynamic|v3|2.0|4.0|0.5|0.2|0.05|64|-2.0|1|1
v3_roi_d2_g2_s01_w02|pixel_residual_roi_dynamic|v3|2.0|2.0|0.1|0.2|0.05|64|-2.0|1|1
v3_roi_d2_g2_s05_w05|pixel_residual_roi_dynamic|v3|2.0|2.0|0.5|0.5|0.05|64|-2.0|1|1
EOF
      ;;
    *)
      echo "Unknown SWEEP_PRESET=${SWEEP_PRESET}. Use core, compact, or roi_weights." >&2
      exit 2
      ;;
  esac
}

JOBS_FILE="${OUT_ROOT}/jobs.tsv"
if [ -n "${SWEEP_JOBS_FILE:-}" ]; then
  cp "${SWEEP_JOBS_FILE}" "${JOBS_FILE}"
else
  build_jobs > "${JOBS_FILE}"
fi

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "# Run after all PCs have produced/copied checkpoints for RUN_NAME=${RUN_NAME}."
  job_id=0
  while IFS='|' read -r exp mode gen l_dyn l_grip l_static l_write dyn_thr roi write_bias spatial write_mask; do
    [ -z "${exp}" ] && continue
    case "${exp}" in \#*) continue ;; esac
    printf 'SKIP_TRAIN=1 TRAIN_CONDITIONS=%q EXP_NAME=%q CKPT_ROOT=%q OUT_ROOT=%q bash scripts/libero/phase1/run_residual_wm_eval.sh %q\n' \
      "${mode}" "${exp}" "${CKPT_ROOT}" "${OUT_ROOT}/eval/${exp}" "${TASK_SUITE}"
    job_id=$((job_id + 1))
  done < "${JOBS_FILE}"
  echo "bash scripts/libero/phase1/summarize_residual_wm_eval.sh ${OUT_ROOT}/eval"
} > "${EVAL_COMMANDS}"
chmod +x "${EVAL_COMMANDS}"

cat > "${SUMMARY}" <<EOF
# Phase 1 Residual WM Train Sweep

- task_suite: ${TASK_SUITE}
- run_name: ${RUN_NAME}
- preset: ${SWEEP_PRESET}
- num_nodes: ${NUM_NODES}
- this_node_index: ${NODE_INDEX}
- local_gpu_ids: ${GPU_IDS}
- local_nproc_per_job: ${LOCAL_NPROC}
- max_parallel_local_jobs: ${MAX_PARALLEL}
- ckpt_root: ${CKPT_ROOT}
- out_root: ${OUT_ROOT}
- seed: ${SEED}
- phase0_matching_defaults: max_steps=${MAX_STEPS:-150000}, global_batch=${WORLD_MODEL_BATCH_SIZE:-16}, per_device_batch=${BATCH_SIZE:-1}, train_horizon=${TRAIN_HORIZON:-7}, lr=${LR:-5e-5}, action_conditioning=${ACTION_CONDITIONING_MODE:-discrete_tokens}, context_anchor=${CONTEXT_ANCHOR_MODE:-spatial_tokens}

Checkpoints are saved under:
\`${CKPT_ROOT}/${TASK_SUITE}/<target_mode>/<exp_name>/s${SEED}\`

After training, evaluate selected checkpoints with:
\`SKIP_TRAIN=1 TRAIN_CONDITIONS="pixel_residual_roi_dynamic" EXP_NAME=<exp_name> CKPT_ROOT=${CKPT_ROOT} bash scripts/libero/phase1/run_residual_wm_eval.sh ${TASK_SUITE}\`

For full sweep evaluation after collecting all checkpoints, run:
\`bash ${EVAL_COMMANDS}\`
EOF

{
  echo -e "job_id\tnode_index\tnum_nodes\texp_name\ttarget_mode\tgeneration\tlambda_dynamic\tlambda_gripper\tlambda_static\tlambda_write\tdynamic_threshold\troi_size\twrite_bias\tspatial_action\twrite_mask\tgpu\tstatus\tlog"
} > "${MANIFEST}"

log "=== Phase 1 train sweep ==="
log "task_suite : ${TASK_SUITE}"
log "preset     : ${SWEEP_PRESET}"
log "run_name   : ${RUN_NAME}"
log "node shard : ${NODE_INDEX}/${NUM_NODES}"
log "local gpus : ${GPU_IDS} (nproc_per_job=${LOCAL_NPROC}, max_parallel_jobs=${MAX_PARALLEL})"
if [ "${MAX_PARALLEL}" -gt 1 ]; then
  log "WARNING: MAX_PARALLEL>1 means multiple DDP jobs may share the same GPUs."
fi
log "ckpt_root  : ${CKPT_ROOT}"
log "out_root   : ${OUT_ROOT}"
log "jobs       : ${JOBS_FILE}"

setup_env

running=0
job_idx=0
status_files=()

wait_one_if_needed() {
  while [ "${running}" -ge "${MAX_PARALLEL}" ]; do
    wait -n || true
    running=$((running - 1))
    if is_true "${STOP_ON_FAIL}" && find "${LOG_ROOT}" -name '*.status' -exec grep -Lqx 'ok' {} \; | grep -q .; then
      echo "A sweep job failed; STOP_ON_FAIL=1" >&2
      exit 1
    fi
  done
}

launch_job() {
  local row="$1"
  local global_job_id="$2"
  IFS='|' read -r exp mode gen l_dyn l_grip l_static l_write dyn_thr roi write_bias spatial write_mask <<< "${row}"
  local gpu="${GPU_IDS}"
  local log_file="${LOG_ROOT}/${exp}.log"
  local status_file="${LOG_ROOT}/${exp}.status"

  log "launch job=${global_job_id} gpus=${gpu} nproc=${LOCAL_NPROC}: ${exp} (${mode}, ${gen})"
  echo -e "${global_job_id}\t${NODE_INDEX}\t${NUM_NODES}\t${exp}\t${mode}\t${gen}\t${l_dyn}\t${l_grip}\t${l_static}\t${l_write}\t${dyn_thr}\t${roi}\t${write_bias}\t${spatial}\t${write_mask}\t${gpu}\trunning\t${log_file}" >> "${MANIFEST}"

  if is_true "${DRY_RUN}"; then
    cat <<EOF | tee "${log_file}"
CUDA_VISIBLE_DEVICES=${gpu} NPROC=${LOCAL_NPROC} TARGET_MODE=${mode} EXP_NAME=${exp} OUTPUT_ROOT=${CKPT_ROOT} \\
MAX_STEPS=${MAX_STEPS:-150000} WORLD_MODEL_BATCH_SIZE=${WORLD_MODEL_BATCH_SIZE:-16} BATCH_SIZE=${BATCH_SIZE:-1} TRAIN_HORIZON=${TRAIN_HORIZON:-7} LR=${LR:-5e-5} \\
ACTION_CONDITIONING_MODE=${ACTION_CONDITIONING_MODE:-discrete_tokens} CONTEXT_ANCHOR_MODE=${CONTEXT_ANCHOR_MODE:-spatial_tokens} \\
LAMBDA_DYNAMIC=${l_dyn} LAMBDA_GRIPPER=${l_grip} LAMBDA_STATIC=${l_static} LAMBDA_WRITE_MASK=${l_write} \\
DYNAMIC_THRESHOLD=${dyn_thr} ROI_CROP_SIZE=${roi} WRITE_MASK_BIAS_INIT=${write_bias} \\
USE_SPATIAL_ACTION_CONDITIONING=${spatial} USE_RESIDUAL_WRITE_MASK=${write_mask} \\
bash ${WM_SCRIPTS}/train_pixel_residual_worldmodel.sh ${TASK_SUITE}
EOF
    echo "dry_run" > "${status_file}"
    job_idx=$((job_idx + 1))
    return
  fi

  status_files+=("${status_file}")
  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export NPROC="${LOCAL_NPROC}"
    export TASK_SUITE="${TASK_SUITE}"
    export TARGET_MODE="${mode}"
    export EXP_NAME="${exp}"
    export OUTPUT_ROOT="${CKPT_ROOT}"
    export SEED="${SEED}"
    export DEVICE="cuda"
    export LAMBDA_DYNAMIC="${l_dyn}"
    export LAMBDA_GRIPPER="${l_grip}"
    export LAMBDA_STATIC="${l_static}"
    export LAMBDA_WRITE_MASK="${l_write}"
    export DYNAMIC_THRESHOLD="${dyn_thr}"
    export ROI_CROP_SIZE="${roi}"
    export WRITE_MASK_BIAS_INIT="${write_bias}"
    export USE_SPATIAL_ACTION_CONDITIONING="${spatial}"
    export USE_RESIDUAL_WRITE_MASK="${write_mask}"

    # Keep caller-provided training knobs: MAX_STEPS, BATCH_SIZE, GRAD_ACCUM,
    # LR, SAVE_STEPS, PRECISION, etc. are inherited automatically.
    bash "${WM_SCRIPTS}/train_pixel_residual_worldmodel.sh" "${TASK_SUITE}"
    rc=$?
    if [ "${rc}" -eq 0 ]; then
      echo "ok" > "${status_file}"
    else
      echo "fail:${rc}" > "${status_file}"
    fi
    exit "${rc}"
  ) > "${log_file}" 2>&1 &

  running=$((running + 1))
  job_idx=$((job_idx + 1))
}

while IFS= read -r row; do
  [ -z "${row}" ] && continue
  case "${row}" in \#*) continue ;; esac
  global_job_id="${row_idx:-0}"
  row_idx=$((global_job_id + 1))
  if [ $((global_job_id % NUM_NODES)) -ne "${NODE_INDEX}" ]; then
    continue
  fi
  wait_one_if_needed
  launch_job "${row}" "${global_job_id}"
done < "${JOBS_FILE}"

if ! is_true "${DRY_RUN}"; then
  while [ "${running}" -gt 0 ]; do
    wait -n || true
    running=$((running - 1))
  done

  fail_count=0
  for status_file in "${status_files[@]}"; do
    if [ ! -f "${status_file}" ] || ! grep -qx "ok" "${status_file}"; then
      fail_count=$((fail_count + 1))
      log "FAILED: ${status_file} ($(cat "${status_file}" 2>/dev/null || echo missing))"
    fi
  done
  if [ "${fail_count}" -gt 0 ]; then
    log "WARNING: ${fail_count} job(s) failed. Check ${LOG_ROOT}"
    exit 1
  fi
fi

log "=== train sweep complete ==="
log "manifest : ${MANIFEST}"
log "logs     : ${LOG_ROOT}"
log "ckpts    : ${CKPT_ROOT}"
