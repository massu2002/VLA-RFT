#!/usr/bin/env bash
# run_v4_temporal_query_ablation.sh — Ablation sweep for v4 TemporalDynamicQueryResidualWM.
#
# Sweeps over key v4 axes:
#   stage         v4a (no scorer) vs v4b (with scorer)
#   history       K = 1, 2, 4
#   queries       Q = 4, 8, 16
#   lambda_rank   0.5, 1.0, 2.0
#   lambda_query  0.25, 0.5, 1.0
#
# Usage:
#   bash scripts/libero/residual_worldmodel/run_v4_temporal_query_ablation.sh [spatial|object|goal|10]
#   SWEEP_PRESET=stage_only DRY_RUN=1 \
#     bash scripts/libero/residual_worldmodel/run_v4_temporal_query_ablation.sh spatial
#
# Useful overrides:
#   SWEEP_PRESET=stage_only | history | queries | lambda | full | quick
#   RUN_NAME=v4_ablation_$(date +%Y%m%d)
#   NUM_NODES=1 NODE_INDEX=0
#   GPU_IDS=auto
#   MAX_STEPS=50000
#   DRY_RUN=1
#   STOP_ON_FAIL=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

source "${SCRIPT_DIR}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log() { echo "[v4-ablation] $(date +%H:%M:%S) $*"; }

TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
SWEEP_PRESET="${SWEEP_PRESET:-stage_only}"
DATE_TAG=$(timestamp)
RUN_NAME="${RUN_NAME:-v4_ablation_${DATE_TAG}_${TASK_SUITE}_${SWEEP_PRESET}}"
SEED="${SEED:-42}"
NUM_NODES="${NUM_NODES:-1}"
NODE_INDEX="${NODE_INDEX:-0}"
GPU_IDS="${GPU_IDS:-auto}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM/ablation/${RUN_NAME}}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/v4_ablation/${RUN_NAME}}"
LOG_ROOT="${OUT_ROOT}/logs/node${NODE_INDEX}_of_${NUM_NODES}"
MANIFEST="${OUT_ROOT}/ablation_manifest_node${NODE_INDEX}.tsv"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

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
LOCAL_NPROC="${#GPU_ARRAY[@]}"

# Each row: exp_name|use_scorer|history_K|num_queries_Q|lambda_rank|lambda_query
build_jobs() {
  case "${SWEEP_PRESET}" in
    stage_only)
      cat <<'EOF'
v4a_K2_Q8|0|2|8|0.0|0.5
v4b_K2_Q8|1|2|8|1.0|0.5
EOF
      ;;
    history)
      cat <<'EOF'
v4b_K1_Q8|1|1|8|1.0|0.5
v4b_K2_Q8|1|2|8|1.0|0.5
v4b_K4_Q8|1|4|8|1.0|0.5
EOF
      ;;
    queries)
      cat <<'EOF'
v4b_K2_Q4|1|2|4|1.0|0.5
v4b_K2_Q8|1|2|8|1.0|0.5
v4b_K2_Q16|1|2|16|1.0|0.5
EOF
      ;;
    lambda)
      cat <<'EOF'
v4b_K2_Q8_Lr05_Lq25|1|2|8|0.5|0.25
v4b_K2_Q8_Lr10_Lq50|1|2|8|1.0|0.5
v4b_K2_Q8_Lr20_Lq50|1|2|8|2.0|0.5
v4b_K2_Q8_Lr10_Lq100|1|2|8|1.0|1.0
EOF
      ;;
    quick)
      cat <<'EOF'
v4a_K2_Q8|0|2|8|0.0|0.5
v4b_K2_Q8|1|2|8|1.0|0.5
v4b_K1_Q8|1|1|8|1.0|0.5
v4b_K2_Q4|1|2|4|1.0|0.5
EOF
      ;;
    full)
      cat <<'EOF'
v4a_K2_Q8|0|2|8|0.0|0.5
v4b_K1_Q4|1|1|4|1.0|0.5
v4b_K1_Q8|1|1|8|1.0|0.5
v4b_K2_Q4|1|2|4|1.0|0.5
v4b_K2_Q8|1|2|8|1.0|0.5
v4b_K2_Q16|1|2|16|1.0|0.5
v4b_K4_Q8|1|4|8|1.0|0.5
v4b_K2_Q8_Lr20|1|2|8|2.0|0.5
v4b_K2_Q8_Lq100|1|2|8|1.0|1.0
EOF
      ;;
    *)
      echo "Unknown SWEEP_PRESET=${SWEEP_PRESET}. Use stage_only, history, queries, lambda, quick, full." >&2
      exit 2
      ;;
  esac
}

{
  echo -e "job_id\tnode_index\texp_name\tuse_scorer\thistory_K\tnum_queries_Q\tlambda_rank\tlambda_query\tgpu\tstatus\tlog"
} > "${MANIFEST}"

log "=== v4 ablation sweep ==="
log "task_suite : ${TASK_SUITE}"
log "preset     : ${SWEEP_PRESET}"
log "run_name   : ${RUN_NAME}"
log "node shard : ${NODE_INDEX}/${NUM_NODES}"
log "local gpus : ${GPU_IDS} (nproc=${LOCAL_NPROC})"
log "ckpt_root  : ${CKPT_ROOT}"
log "out_root   : ${OUT_ROOT}"

setup_env

job_id=0
fail_count=0

while IFS='|' read -r exp use_scorer hist_k num_q lr_val lq_val; do
  [ -z "${exp}" ] && continue
  case "${exp}" in \#*) continue ;; esac

  current_id="${job_id}"
  job_id=$((job_id + 1))
  if [ $((current_id % NUM_NODES)) -ne "${NODE_INDEX}" ]; then
    continue
  fi

  save_dir="${CKPT_ROOT}/${TASK_SUITE}/${exp}/s${SEED}"
  log_file="${LOG_ROOT}/${exp}.log"

  log "RUN job=${current_id}: ${exp} (scorer=${use_scorer} K=${hist_k} Q=${num_q} λ_rank=${lr_val} λ_query=${lq_val})"
  echo -e "${current_id}\t${NODE_INDEX}\t${exp}\t${use_scorer}\t${hist_k}\t${num_q}\t${lr_val}\t${lq_val}\t${GPU_IDS}\trunning\t${log_file}" >> "${MANIFEST}"

  if is_true "${DRY_RUN}"; then
    echo "DRY_RUN: USE_ACTION_FUTURE_SCORER=${use_scorer} HISTORY_LENGTH=${hist_k} NUM_DYNAMIC_QUERIES=${num_q} LAMBDA_RANK=${lr_val} LAMBDA_QUERY=${lq_val} EXP_NAME=${exp} bash train_v4_temporal_query_worldmodel.sh ${TASK_SUITE}"
    continue
  fi

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    export NPROC="${LOCAL_NPROC}"
    export TASK_SUITE="${TASK_SUITE}"
    export EXP_NAME="${exp}"
    export OUTPUT_ROOT="${CKPT_ROOT}"
    export SEED="${SEED}"
    export USE_ACTION_FUTURE_SCORER="${use_scorer}"
    export HISTORY_LENGTH="${hist_k}"
    export NUM_DYNAMIC_QUERIES="${num_q}"
    export LAMBDA_RANK="${lr_val}"
    export LAMBDA_QUERY="${lq_val}"
    # Inherit MAX_STEPS, LR, PRECISION, etc. from caller
    bash "${SCRIPT_DIR}/train_v4_temporal_query_worldmodel.sh" "${TASK_SUITE}"
  ) > "${log_file}" 2>&1
  rc=$?
  set -e

  if [ "${rc}" -eq 0 ]; then
    log "DONE job=${current_id}: ${exp}"
  else
    log "FAILED job=${current_id}: ${exp} rc=${rc}; see ${log_file}"
    fail_count=$((fail_count + 1))
    is_true "${STOP_ON_FAIL}" && exit "${rc}"
  fi
done < <(build_jobs)

if [ "${fail_count}" -gt 0 ]; then
  log "WARNING: ${fail_count} ablation job(s) failed."
  exit 1
fi

log "=== ablation sweep complete ==="
log "manifest : ${MANIFEST}"
log "logs     : ${LOG_ROOT}"
log "ckpts    : ${CKPT_ROOT}"
