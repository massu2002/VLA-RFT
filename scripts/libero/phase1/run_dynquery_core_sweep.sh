#!/usr/bin/env bash
# run_dynquery_core_sweep.sh — DynQueryWorldModel core sweep orchestrator.
#
# Reads SWEEP_CONFIG (JSON), iterates enabled experiments, dispatches each
# to the backend runner with full env-var injection.
#
# Usage:
#   TASK_SUITE=spatial RUN_NAME=DynQueryWorldModel_core_sweep MODE=train_only \
#     bash scripts/libero/phase1/run_dynquery_core_sweep.sh
#
# Examples:
#   # Dry run — print all planned commands without executing
#   DRY_RUN=1 TASK_SUITE=spatial \
#     bash scripts/libero/phase1/run_dynquery_core_sweep.sh
#
#   # Smoke test — tiny steps/windows
#   SMOKE=1 TASK_SUITE=spatial \
#     bash scripts/libero/phase1/run_dynquery_core_sweep.sh
#
#   # Single condition
#   EXP_FILTER=dq_full_rank1 TASK_SUITE=spatial \
#     bash scripts/libero/phase1/run_dynquery_core_sweep.sh
#
#   # Eval only (after training)
#   MODE=eval_only TASK_SUITE=spatial \
#     bash scripts/libero/phase1/run_dynquery_core_sweep.sh
#
# Key env-var overrides:
#   SWEEP_CONFIG   path to JSON   (default: configs/libero/phase1/dynquery_core_sweep.json)
#   TASK_SUITE     spatial | object | goal | 10
#   RUN_NAME       unique name for this sweep run
#   MODE           train_only | eval_only | train_eval
#   CKPT_ROOT      checkpoint root  (default: checkpoints/libero/DynQueryWorldModel/core_sweep)
#   OUT_ROOT       results root     (default: results/phase1/DynQueryWorldModel_core_sweep)
#   EXP_FILTER     comma-separated exp_names to run (empty = all)
#   DRY_RUN, SMOKE, SKIP_EXISTING, OVERWRITE, STOP_ON_FAIL
#   NUM_NODES, NODE_INDEX   for distributed scheduling across nodes

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"

source "${WM_SCRIPTS}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log()     { echo "[dynquery-core-sweep] $(date +%H:%M:%S) $*"; }
die()     { echo "[dynquery-core-sweep] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/libero/phase1/dynquery_core_sweep.json}"
TASK_SUITE="${TASK_SUITE:-spatial}"
RUN_NAME="${RUN_NAME:-DynQueryWorldModel_core_sweep}"
MODE="${MODE:-train_only}"
SEED="${SEED:-42}"
NUM_NODES="${NUM_NODES:-1}"
NODE_INDEX="${NODE_INDEX:-0}"
GPU_IDS="${GPU_IDS:-auto}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OVERWRITE="${OVERWRITE:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
EXP_FILTER="${EXP_FILTER:-}"

CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/DynQueryWorldModel/core_sweep}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/DynQueryWorldModel_core_sweep}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs/node${NODE_INDEX}_of_${NUM_NODES}}"
MANIFEST="${OUT_ROOT}/sweep_manifest_node${NODE_INDEX}_of_${NUM_NODES}.tsv"

SHARED_WINDOW_MANIFEST="${SHARED_WINDOW_MANIFEST:-}"

[ -f "${SWEEP_CONFIG}" ] || die "SWEEP_CONFIG not found: ${SWEEP_CONFIG}"
python3 -c "import json; json.load(open('${SWEEP_CONFIG}'))" \
  || die "SWEEP_CONFIG is not valid JSON: ${SWEEP_CONFIG}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

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
# Emit experiment list from JSON via Python
# Columns (pipe-separated):
#   exp_name|stage|history_length|num_dynamic_queries|
#   use_action_conditioned_mask|predictor_mode|use_dynamic_residual_gate|
#   lambda_mask_dynamic|lambda_query_delta_sparse|
#   use_motion_bias|use_action_future_scorer|lambda_rank|rank_margin|rank_temperature|
#   negative_type|negative_mix|
#   lambda_image|lambda_dynamic|lambda_static|lambda_query|
#   max_steps|init_from_checkpoint
# ---------------------------------------------------------------------------
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
        get("stage", "dq_b"),
        get("history_length", 2),
        get("num_dynamic_queries", 8),
        get("use_action_conditioned_mask", 1),
        get("predictor_mode", "query_wise"),
        get("use_dynamic_residual_gate", 1),
        get("lambda_mask_dynamic", 0.1),
        get("lambda_query_delta_sparse", 0.001),
        get("use_motion_bias", 1),
        get("use_action_future_scorer", 1),
        get("lambda_rank", 1.0),
        get("rank_margin", 0.1),
        get("rank_temperature", 0.07),
        get("negative_type", "mixed"),
        get("negative_mix", ""),
        get("lambda_image", 0.1),
        get("lambda_dynamic", 1.0),
        get("lambda_static", 0.2),
        get("lambda_query", 0.5),
        get("max_steps", 150000),
        get("init_from_checkpoint", ""),
    ]
    print("|".join(parts))
PYEOF
)"

if [ -z "${JOBS_TSV}" ]; then
  log "No enabled experiments found in ${SWEEP_CONFIG}"
  exit 0
fi

TOTAL_JOBS=$(echo "${JOBS_TSV}" | wc -l)

log "=== DynQuery core sweep ==="
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

{
  printf 'job_id\tnode_index\texp_name\tstage\tK\tQ\tact_cond\tpred_mode\tgate\tscorer\tlambda_rank\tneg_type\tstatus\tlog\n'
} > "${MANIFEST}"

if is_true "${DRY_RUN}"; then
  log ""
  log "DRY_RUN plan (this node would run):"
  job_id=0
  while IFS='|' read -r exp stage hist_k num_q act_cond pred_mode gate \
                          lmd lqds motion scorer lr_val margin rank_temp \
                          neg_type neg_mix li ld ls lq max_steps_exp init_ckpt; do
    if [ $((job_id % NUM_NODES)) -eq "${NODE_INDEX}" ]; then
      log "  job=${job_id}  ${exp}  (stage=${stage} K=${hist_k} Q=${num_q})"
      log "       Core1=${act_cond} Core2=${pred_mode} Core3=${gate} λ_mask_dyn=${lmd} λ_q_delta=${lqds}"
      log "       motion=${motion} scorer=${scorer} λ_rank=${lr_val} neg=${neg_type} max_steps=${max_steps_exp}"
      [ -n "${init_ckpt}" ] && log "       init_ckpt=${init_ckpt}"
      log "       ckpt=${CKPT_ROOT}/${TASK_SUITE}/${exp}/s${SEED}  eval=${OUT_ROOT}/${exp}"
    fi
    job_id=$((job_id + 1))
  done <<< "${JOBS_TSV}"
  log ""
  log "  Set DRY_RUN=0 to execute."
  exit 0
fi

# ---------------------------------------------------------------------------
# Execute jobs
# ---------------------------------------------------------------------------
job_id=0
fail_count=0
success_count=0

while IFS='|' read -r exp stage hist_k num_q act_cond pred_mode gate \
                        lmd lqds motion scorer lr_val margin rank_temp \
                        neg_type neg_mix li ld ls lq max_steps_exp init_ckpt; do
  current_id="${job_id}"
  job_id=$((job_id + 1))

  if [ $((current_id % NUM_NODES)) -ne "${NODE_INDEX}" ]; then
    continue
  fi

  log "──────────────────────────────────────────────────────────────"
  log "START job=${current_id}  ${exp}"
  log "  stage=${stage}  K=${hist_k}  Q=${num_q}"
  log "  Core1(act_cond)=${act_cond}  Core2(pred)=${pred_mode}  Core3(gate)=${gate}"
  log "  λ_mask_dyn=${lmd}  λ_q_delta=${lqds}  motion=${motion}  scorer=${scorer}  λ_rank=${lr_val}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\trunning\t%s\n' \
    "${current_id}" "${NODE_INDEX}" "${exp}" "${stage}" "${hist_k}" "${num_q}" \
    "${act_cond}" "${pred_mode}" "${gate}" "${scorer}" "${lr_val}" "${neg_type}" \
    "${LOG_ROOT}/${exp}.log" >> "${MANIFEST}"

  log_file="${LOG_ROOT}/${exp}.log"

  _use_manifest=0; _manifest_path=""
  if [ -n "${SHARED_WINDOW_MANIFEST}" ] && [ -f "${SHARED_WINDOW_MANIFEST}" ]; then
    _use_manifest=1; _manifest_path="${SHARED_WINDOW_MANIFEST}"
  fi

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    export NPROC="${LOCAL_NPROC}"
    export EXP_NAME="${exp}"
    export STAGE="${stage}"
    export HISTORY_LENGTH="${hist_k}"
    export NUM_DYNAMIC_QUERIES="${num_q}"
    export USE_ACTION_CONDITIONED_MASK="${act_cond}"
    export PREDICTOR_MODE="${pred_mode}"
    export USE_DYNAMIC_RESIDUAL_GATE="${gate}"
    export LAMBDA_MASK_DYNAMIC="${lmd}"
    export LAMBDA_QUERY_DELTA_SPARSE="${lqds}"
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
    export MAX_STEPS="${max_steps_exp:-${MAX_STEPS:-150000}}"
    export INIT_FROM_CHECKPOINT="${init_ckpt:-}"
    export TASK_SUITE
    export SEED
    export CKPT_SEED="${CKPT_SEED:-${SEED}}"
    export RUN_NAME
    export CKPT_ROOT
    export OUT_ROOT
    export SWEEP_CONFIG
    export MODE
    export DRY_RUN
    export SMOKE
    export SKIP_EXISTING
    export OVERWRITE
    export EVAL_HORIZON="${EVAL_HORIZON:-8}"
    export TRAIN_HORIZON="${TRAIN_HORIZON:-8}"
    export PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE:-1}"
    export NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-200}"
    export NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS:-100}"
    export WINDOW_POSITION_MODE="${WINDOW_POSITION_MODE:-episode_phases}"
    export NEGATIVE_EVAL_TYPES="${NEGATIVE_EVAL_TYPES:-same_phase,temporal_shift,action_noise,mixed}"
    export TEMPORAL_SHIFT_MAX="${TEMPORAL_SHIFT_MAX:-3}"
    export ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.15}"
    export ACTION_ABLATION="${ACTION_ABLATION:-0}"
    export SAVE_DEBUG_VISUALS="${SAVE_DEBUG_VISUALS:-0}"
    export USE_WINDOW_MANIFEST="${_use_manifest}"
    export WINDOW_MANIFEST="${_manifest_path}"
    export LR="${LR:-1e-4}"
    export PRECISION="${PRECISION:-bf16}"
    export LR_SCHEDULER="${LR_SCHEDULER:-constant}"
    export WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
    export BATCH_SIZE="${BATCH_SIZE:-8}"
    export WORLD_MODEL_BATCH_SIZE="${WORLD_MODEL_BATCH_SIZE:-64}"
    export SAVE_STEPS="${SAVE_STEPS:-10000}"
    export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
    export LOGGING_STEPS="${LOGGING_STEPS:-20}"
    export TF32="${TF32:-1}"
    bash "${WM_SCRIPTS}/run_dynquery_core_sweep.sh"
  ) 2>&1 | tee "${log_file}"
  rc="${PIPESTATUS[0]}"
  set -e

  if [ "${rc}" -eq 0 ]; then
    log "DONE job=${current_id}: ${exp}"
    success_count=$((success_count + 1))
    if [ -z "${SHARED_WINDOW_MANIFEST}" ]; then
      _new_manifest="${OUT_ROOT}/${exp}/window_manifest.json"
      if [ -f "${_new_manifest}" ]; then
        SHARED_WINDOW_MANIFEST="${_new_manifest}"
        log "Shared manifest set from ${exp}: ${SHARED_WINDOW_MANIFEST}"
      fi
    fi
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
log "  Summarize:  RUN_NAME=${RUN_NAME} bash scripts/libero/phase1/summarize_v4_core_sweep.sh"
log "  RFT:        best checkpoint → command_train_rft.sh"

[ "${fail_count}" -gt 0 ] && exit 1
exit 0
