#!/usr/bin/env bash
# run_v4_core_sweep.sh — Phase 1 v4 core sweep orchestrator.
#
# Reads SWEEP_CONFIG (JSON), iterates enabled experiments, and dispatches
# each to the backend runner with full env-var injection.
#
# Usage:
#   SWEEP_CONFIG=configs/libero/phase1/v4_core_sweep.json \
#   TASK_SUITE=spatial \
#   RUN_NAME=v4_core_sweep_spatial \
#   MODE=train_eval \
#     bash scripts/libero/phase1/run_v4_core_sweep.sh
#
# Examples:
#
#   # Dry run — print all planned commands without executing
#   DRY_RUN=1 TASK_SUITE=spatial RUN_NAME=v4_core_sweep_spatial \
#     bash scripts/libero/phase1/run_v4_core_sweep.sh
#
#   # Smoke test — tiny steps/windows, fast verification
#   SMOKE=1 TASK_SUITE=spatial RUN_NAME=v4_core_sweep_smoke \
#     bash scripts/libero/phase1/run_v4_core_sweep.sh
#
#   # Distributed: 4-way split, this node handles experiments 0,4,8
#   NUM_NODES=4 NODE_INDEX=0 TASK_SUITE=spatial RUN_NAME=v4_core_sweep_spatial \
#     bash scripts/libero/phase1/run_v4_core_sweep.sh
#
#   # Eval only (after training has completed)
#   MODE=eval_only TASK_SUITE=spatial RUN_NAME=v4_core_sweep_spatial \
#     bash scripts/libero/phase1/run_v4_core_sweep.sh
#
#   # Summarize results
#   RUN_NAME=v4_core_sweep_spatial \
#     bash scripts/libero/phase1/summarize_v4_core_sweep.sh
#
# Key env-var overrides:
#   SWEEP_CONFIG      path to JSON config  (default: configs/libero/phase1/v4_core_sweep.json)
#   TASK_SUITE        spatial | object | goal | 10
#   RUN_NAME          unique name for this sweep run
#   MODE              train_only | eval_only | train_eval | rft_only | all
#   OUT_ROOT          results root  (default: results/phase1/residual_worldmodel/${RUN_NAME})
#   CKPT_ROOT         checkpoint root
#   NUM_NODES         total number of parallel nodes
#   NODE_INDEX        0-based index of this node
#   DRY_RUN           1 = print commands only, do not execute
#   SMOKE             1 = reduced steps/windows for fast verification
#   SKIP_EXISTING     1 = skip conditions whose checkpoint/eval already exists (default: 1)
#   OVERWRITE         1 = re-run even if outputs exist (overrides SKIP_EXISTING)
#   STOP_ON_FAIL      1 = abort sweep on first failure
#   EXP_FILTER        comma-separated exp_names to run (empty = all enabled)
#   GPU_IDS           comma-separated GPU indices or "auto"

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"

source "${WM_SCRIPTS}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log()     { echo "[v4-core-sweep] $(date +%H:%M:%S) $*"; }
die()     { echo "[v4-core-sweep] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/libero/phase1/v4_core_sweep.json}"
TASK_SUITE="${TASK_SUITE:-spatial}"
RUN_NAME="${RUN_NAME:-v4_core_sweep_$(timestamp)_${TASK_SUITE}}"
MODE="${MODE:-train_eval}"
SEED="${SEED:-42}"
NUM_NODES="${NUM_NODES:-${NUM_PCS:-${SHARD_COUNT:-1}}}"
NODE_INDEX="${NODE_INDEX:-${PC_INDEX:-${SHARD_INDEX:-0}}}"
GPU_IDS="${GPU_IDS:-auto}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OVERWRITE="${OVERWRITE:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
EXP_FILTER="${EXP_FILTER:-}"   # comma-separated exp_names to run; empty = all

CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM/${RUN_NAME}}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/residual_worldmodel/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs/node${NODE_INDEX}_of_${NUM_NODES}}"
MANIFEST="${OUT_ROOT}/sweep_manifest_node${NODE_INDEX}_of_${NUM_NODES}.tsv"

# Validate
[ -f "${SWEEP_CONFIG}" ] || die "SWEEP_CONFIG not found: ${SWEEP_CONFIG}"
python3 -c "import json; json.load(open('${SWEEP_CONFIG}'))" \
  || die "SWEEP_CONFIG is not valid JSON: ${SWEEP_CONFIG}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
printf '%s\n' "${OUT_ROOT}" > "${REPO_ROOT}/results/phase1/latest.txt"

# GPU detection
auto_gpu_ids() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && \
     [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ] && \
     [ "${CUDA_VISIBLE_DEVICES}" != "void" ]; then
    echo "${CUDA_VISIBLE_DEVICES}"; return
  fi
  local n; n="$(detect_gpu_count)"
  local ids=(); local i
  for ((i=0; i<n; i++)); do ids+=("${i}"); done
  local IFS=,; echo "${ids[*]}"
}
[ "${GPU_IDS}" = "auto" ] && GPU_IDS="$(auto_gpu_ids)"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS// /,}"
LOCAL_NPROC="${#GPU_ARRAY[@]}"

setup_env

# ---------------------------------------------------------------------------
# Emit experiment list from config via Python
# ---------------------------------------------------------------------------
# Output: TSV with columns:
# exp_name|stage|history_length|num_dynamic_queries|use_motion_bias|
# use_action_future_scorer|lambda_rank|rank_margin|rank_temperature|negative_type|negative_mix|
# lambda_image|lambda_dynamic|lambda_static|lambda_query|lambda_sparse
JOBS_TSV="$(python3 - <<PYEOF
import json, sys

cfg = json.load(open("${SWEEP_CONFIG}"))
common = cfg.get("common", {})
exp_filter_raw = "${EXP_FILTER}"
exp_filter = [e.strip() for e in exp_filter_raw.split(",") if e.strip()]

for exp in cfg.get("experiments", []):
    if not exp.get("enabled", True):
        continue
    name = exp["exp_name"]
    if exp_filter and name not in exp_filter:
        continue

    def get(key, default=""):
        v = exp.get(key)
        if v is None:
            v = common.get(key)
        if v is None:
            return str(default)
        return str(v)

    parts = [
        get("exp_name"),
        get("stage", "v4b"),
        get("history_length", 2),
        get("num_dynamic_queries", 8),
        get("use_motion_bias", 0),
        get("use_action_future_scorer", 1),
        get("lambda_rank", 1.0),
        get("rank_margin", 0.1),
        get("rank_temperature", 0.07),
        get("negative_type", "same_task_other_window"),
        get("negative_mix", ""),
        get("lambda_image", 0.1),
        get("lambda_dynamic", 1.0),
        get("lambda_static", 0.2),
        get("lambda_query", 0.5),
        get("lambda_sparse", 0.01),
    ]
    print("|".join(parts))
PYEOF
)"

if [ -z "${JOBS_TSV}" ]; then
  log "No enabled experiments found in ${SWEEP_CONFIG}"
  exit 0
fi

TOTAL_JOBS=$(echo "${JOBS_TSV}" | wc -l)

log "=== Phase 1 v4 core sweep ==="
log "config     : ${SWEEP_CONFIG}"
log "task_suite : ${TASK_SUITE}"
log "run_name   : ${RUN_NAME}"
log "mode       : ${MODE}"
log "node shard : ${NODE_INDEX}/${NUM_NODES}"
log "local gpus : ${GPU_IDS} (nproc=${LOCAL_NPROC})"
log "total jobs : ${TOTAL_JOBS}  (this node: ~$((TOTAL_JOBS / NUM_NODES + 1)))"
log "skip_exist : ${SKIP_EXISTING}  overwrite: ${OVERWRITE}  dry_run: ${DRY_RUN}  smoke: ${SMOKE}"
log "ckpt_root  : ${CKPT_ROOT}"
log "out_root   : ${OUT_ROOT}"

# Manifest header
{
  printf 'job_id\tnode_index\texp_name\tstage\thistory_K\tnum_queries_Q'
  printf '\tuse_motion_bias\tuse_scorer\tlambda_rank\tnegative_type\tstatus\tlog\n'
} > "${MANIFEST}"

# DRY_RUN: show plan only
if is_true "${DRY_RUN}"; then
  log ""
  log "DRY_RUN plan (this node would run):"
  log "  run knobs: MODE=${MODE} MAX_STEPS=${MAX_STEPS:-150000} BATCH_SIZE=${BATCH_SIZE:-1} WORLD_MODEL_BATCH_SIZE=${WORLD_MODEL_BATCH_SIZE:-16} LR=${LR:-5e-5}"
  log "  ckpt root: ${CKPT_ROOT}/${TASK_SUITE}/<EXP_NAME>/s${SEED}"
  log "  out root : ${OUT_ROOT}/<EXP_NAME>"
  job_id=0
  while IFS='|' read -r exp stage hist_k num_q motion scorer lr_val margin rank_temp neg_type neg_mix \
                          li ld ls lq lsp; do
    if [ $((job_id % NUM_NODES)) -eq "${NODE_INDEX}" ]; then
      log "  job=${job_id}  ${exp}  (stage=${stage} K=${hist_k} Q=${num_q} motion=${motion} scorer=${scorer} lambda_rank=${lr_val} rank_temp=${rank_temp} neg=${neg_type} neg_mix=${neg_mix:-none})"
      log "       save=${CKPT_ROOT}/${TASK_SUITE}/${exp}/s${SEED}  eval=${OUT_ROOT}/${exp}"
    fi
    job_id=$((job_id + 1))
  done <<< "${JOBS_TSV}"
  log ""
  log "  Set DRY_RUN=0 to execute."
  log "  Set MODE=train_only|eval_only|train_eval|rft_only|all"
  exit 0
fi

# ---------------------------------------------------------------------------
# Execute jobs
# ---------------------------------------------------------------------------
job_id=0
fail_count=0
success_count=0

while IFS='|' read -r exp stage hist_k num_q motion scorer lr_val margin rank_temp neg_type neg_mix \
                        li ld ls lq lsp; do
  current_id="${job_id}"
  job_id=$((job_id + 1))

  if [ $((current_id % NUM_NODES)) -ne "${NODE_INDEX}" ]; then
    continue
  fi

  log "──────────────────────────────────────────────────────────────"
  log "START job=${current_id}  ${exp}"
  log "  stage=${stage}  K=${hist_k}  Q=${num_q}  motion=${motion}  scorer=${scorer}"
  log "  lambda_rank=${lr_val}  negative_type=${neg_type}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\trunning\t%s\n' \
    "${current_id}" "${NODE_INDEX}" "${exp}" "${stage}" "${hist_k}" "${num_q}" \
    "${motion}" "${scorer}" "${lr_val}" "${neg_type}" \
    "${LOG_ROOT}/${exp}.log" >> "${MANIFEST}"

  log_file="${LOG_ROOT}/${exp}.log"

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    export NPROC="${LOCAL_NPROC}"
    export EXP_NAME="${exp}"
    export STAGE="${stage}"
    export HISTORY_LENGTH="${hist_k}"
    export NUM_DYNAMIC_QUERIES="${num_q}"
    export USE_MOTION_BIAS="${motion}"
    export USE_ACTION_FUTURE_SCORER="${scorer}"
    export LAMBDA_RANK="${lr_val}"
    export RANK_MARGIN="${margin}"
    export RANK_TEMPERATURE="${rank_temp}"
    export NEGATIVE_TYPE="${neg_type}"
    export NEGATIVE_MIX="${neg_mix}"
    export LAMBDA_IMAGE="${li}"
    export LAMBDA_DYNAMIC="${ld}"
    export LAMBDA_STATIC="${ls}"
    export LAMBDA_QUERY="${lq}"
    export LAMBDA_SPARSE="${lsp}"
    export TASK_SUITE
    export SEED
    export RUN_NAME
    export CKPT_ROOT
    export OUT_ROOT
    export SWEEP_CONFIG
    export MODE
    export DRY_RUN
    export SMOKE
    export SKIP_EXISTING
    export OVERWRITE
    export EVAL_HORIZON="${EVAL_HORIZON:-7}"
    export TRAIN_HORIZON="${TRAIN_HORIZON:-7}"
    export PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE:-1}"
    export NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-200}"
    export NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS:-100}"
    export WINDOW_POSITION_MODE="${WINDOW_POSITION_MODE:-random}"
    export ACTION_ABLATION="${ACTION_ABLATION:-0}"
    export SAVE_DEBUG_VISUALS="${SAVE_DEBUG_VISUALS:-0}"
    # Passthrough training knobs
    export MAX_STEPS="${MAX_STEPS:-150000}"
    export LR="${LR:-5e-5}"
    export PRECISION="${PRECISION:-bf16}"
    export LR_SCHEDULER="${LR_SCHEDULER:-constant}"
    export WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
    export WORLD_MODEL_BATCH_SIZE="${WORLD_MODEL_BATCH_SIZE:-16}"
    export BATCH_SIZE="${BATCH_SIZE:-1}"
    bash "${WM_SCRIPTS}/run_v4_core_sweep.sh"
  ) 2>&1 | tee "${log_file}"
  rc="${PIPESTATUS[0]}"
  set -e

  if [ "${rc}" -eq 0 ]; then
    log "DONE job=${current_id}: ${exp}"
    success_count=$((success_count + 1))
  else
    log "FAILED job=${current_id}: ${exp} (rc=${rc})"
    fail_count=$((fail_count + 1))
    is_true "${STOP_ON_FAIL}" && exit "${rc}"
  fi
done <<< "${JOBS_TSV}"

log "══════════════════════════════════════════════════════════════"
log "Sweep complete.  success=${success_count}  fail=${fail_count}"
log "manifest : ${MANIFEST}"
log "results  : ${OUT_ROOT}"
log ""
log "Next steps:"
log "  Summarize:        RUN_NAME=${RUN_NAME} bash scripts/libero/phase1/summarize_v4_core_sweep.sh"
log "  Select best:      RUN_NAME=${RUN_NAME} BEST_CRITERION=hybrid_score bash scripts/libero/residual_worldmodel/select_best_v4_for_rft.sh"
log "  RFT sweep:        RUN_NAME=${RUN_NAME} bash scripts/libero/phase1/run_v4_selected_rft_sweep.sh"

[ "${fail_count}" -gt 0 ] && exit 1
exit 0
