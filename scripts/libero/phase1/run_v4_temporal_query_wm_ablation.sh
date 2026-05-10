#!/usr/bin/env bash
# run_v4_temporal_query_wm_ablation.sh — Phase 1 v4 WM ablation sweep launcher.
#
# Runs a full train + optional eval sweep over v4 architectural variants.
#
# Usage:
#   SWEEP_PRESET=stage_only bash scripts/libero/phase1/run_v4_temporal_query_wm_ablation.sh spatial
#   SWEEP_PRESET=full NUM_NODES=2 NODE_INDEX=0 \
#     bash scripts/libero/phase1/run_v4_temporal_query_wm_ablation.sh spatial
#
# Sweep presets:
#   stage_only  — v4a vs v4b (2 jobs)
#   history     — K = 1, 2, 4 (3 jobs)
#   queries     — Q = 4, 8, 16 (3 jobs)
#   lambda      — λ_rank + λ_query grid (4 jobs)
#   quick       — essential 4-job subset
#   full        — comprehensive 9-job sweep
#
# After training, set SKIP_TRAIN=1 to run eval only.
#
# Useful overrides:
#   SWEEP_PRESET=quick
#   RUN_NAME=v4_ablation_$(date +%Y%m%d)
#   SKIP_TRAIN=0 / SKIP_EVAL=1
#   NUM_NODES=1 NODE_INDEX=0
#   MAX_STEPS=50000
#   DRY_RUN=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"

source "${WM_SCRIPTS}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log() { echo "[phase1-v4-ablation] $(date +%H:%M:%S) $*"; }

TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
SWEEP_PRESET="${SWEEP_PRESET:-stage_only}"
DATE_TAG=$(timestamp)
RUN_NAME="${RUN_NAME:-v4_ablation_${DATE_TAG}_${TASK_SUITE}_${SWEEP_PRESET}}"
SEED="${SEED:-42}"
NUM_NODES="${NUM_NODES:-${NUM_PCS:-${SHARD_COUNT:-1}}}"
NODE_INDEX="${NODE_INDEX:-${PC_INDEX:-${SHARD_INDEX:-0}}}"
GPU_IDS="${GPU_IDS:-auto}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM/ablation/${RUN_NAME}}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/v4_ablation/${RUN_NAME}}"

mkdir -p "${OUT_ROOT}"
printf '%s\n' "${OUT_ROOT}" > "${REPO_ROOT}/results/phase1/latest.txt"

log "=== Phase 1 v4 WM ablation sweep ==="
log "task_suite  : ${TASK_SUITE}"
log "preset      : ${SWEEP_PRESET}"
log "run_name    : ${RUN_NAME}"
log "node shard  : ${NODE_INDEX}/${NUM_NODES}"
log "skip_train  : ${SKIP_TRAIN}"
log "skip_eval   : ${SKIP_EVAL}"
log "ckpt_root   : ${CKPT_ROOT}"
log "out_root    : ${OUT_ROOT}"

# ---------------------------------------------------------------------------
# Train phase
# ---------------------------------------------------------------------------
if ! is_true "${SKIP_TRAIN}"; then
  SWEEP=0 \
  SWEEP_PRESET="${SWEEP_PRESET}" \
  RUN_NAME="${RUN_NAME}" \
  SEED="${SEED}" \
  NUM_NODES="${NUM_NODES}" \
  NODE_INDEX="${NODE_INDEX}" \
  GPU_IDS="${GPU_IDS}" \
  DRY_RUN="${DRY_RUN}" \
  STOP_ON_FAIL="${STOP_ON_FAIL}" \
  CKPT_ROOT="${CKPT_ROOT}" \
  OUT_ROOT="${OUT_ROOT}" \
    bash "${WM_SCRIPTS}/run_v4_temporal_query_ablation.sh" "${TASK_SUITE}"
  log "Train phase complete."
else
  log "Skipping train phase (SKIP_TRAIN=1)."
fi

# ---------------------------------------------------------------------------
# Eval phase
# ---------------------------------------------------------------------------
if ! is_true "${SKIP_EVAL}"; then
  EVAL_OUT="${OUT_ROOT}/eval"
  mkdir -p "${EVAL_OUT}"
  log "Starting eval phase → ${EVAL_OUT}"

  setup_env
  fail_count=0

  while IFS= read -r cfg_path; do
    ckpt_dir="$(dirname "${cfg_path}")"
    condition_name="$(basename "${ckpt_dir}")"
    out_dir="${EVAL_OUT}/${condition_name}"
    log_file="${OUT_ROOT}/logs/eval_${condition_name}.log"
    mkdir -p "${OUT_ROOT}/logs" "${out_dir}"

    log "EVAL: ${condition_name}"
    set +e
    bash "${WM_SCRIPTS}/eval_v4_temporal_query_worldmodel.sh" \
      "MODEL_DIR=${ckpt_dir}" \
      "TASK_SUITE=${TASK_SUITE}" \
      "OUTPUT_DIR=${out_dir}" \
      "CONDITION_NAME=${condition_name}" \
      "SEED=${SEED}" \
      > "${log_file}" 2>&1
    rc=$?
    set -e

    if [ "${rc}" -ne 0 ]; then
      log "FAILED: ${condition_name} rc=${rc}; see ${log_file}"
      fail_count=$((fail_count + 1))
      is_true "${STOP_ON_FAIL}" && exit "${rc}"
    fi
  done < <(find "${CKPT_ROOT}" -name "v4_config.json" | sort)

  [ "${fail_count}" -gt 0 ] && log "WARNING: ${fail_count} eval(s) failed."

  # Summarize
  bash "${WM_SCRIPTS}/summarize_v4_temporal_query_eval.sh" "${EVAL_OUT}" || true
  printf '%s\n' "${EVAL_OUT}" > "${REPO_ROOT}/results/phase1/latest.txt"
  log "Eval phase complete → ${EVAL_OUT}"
else
  log "Skipping eval phase (SKIP_EVAL=1)."
fi

log "=== v4 ablation sweep complete ==="
log "ckpts   : ${CKPT_ROOT}"
log "results : ${OUT_ROOT}"
