#!/usr/bin/env bash
# run_v4_temporal_query_wm_eval_only.sh — Phase 1 v4 WM evaluation launcher.
#
# Usage:
#   MODEL_DIR=checkpoints/libero/TemporalQueryResidualWM/spatial/v4b/s42 \
#     bash scripts/libero/phase1/run_v4_temporal_query_wm_eval_only.sh spatial
#
# Sweep eval over all v4 checkpoints under CKPT_ROOT:
#   SWEEP=1 CKPT_ROOT=checkpoints/libero/TemporalQueryResidualWM/phase1_sweeps/v4_sweep \
#     bash scripts/libero/phase1/run_v4_temporal_query_wm_eval_only.sh spatial
#
# Useful overrides:
#   MODEL_DIR, CKPT_ROOT, CONDITION_NAME, OUTPUT_DIR
#   NUM_EVAL_WINDOWS=200, NUM_RANKING_WINDOWS=100
#   EVAL_HORIZON=7
#   ACTION_ABLATION=1
#   SAVE_DEBUG_VISUALS=1
#   DRY_RUN_WINDOWS=5   (quick sanity check)
#   SWEEP=1 to auto-discover all v4 checkpoints under CKPT_ROOT

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"

source "${WM_SCRIPTS}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log() { echo "[phase1-v4-eval] $(date +%H:%M:%S) $*"; }

TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
SWEEP="${SWEEP:-0}"

# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------
if ! is_true "${SWEEP}"; then
  MODEL_DIR="${MODEL_DIR:?'Set MODEL_DIR env var to point to a v4 checkpoint directory'}"
  exec bash "${WM_SCRIPTS}/eval_v4_temporal_query_worldmodel.sh" \
    "MODEL_DIR=${MODEL_DIR}" \
    "TASK_SUITE=${TASK_SUITE}"
fi

# ---------------------------------------------------------------------------
# Sweep eval — discover all v4 checkpoint dirs under CKPT_ROOT
# ---------------------------------------------------------------------------
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/v4_eval/$(basename "${CKPT_ROOT}")}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
SEED="${SEED:-42}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
printf '%s\n' "${OUT_ROOT}" > "${REPO_ROOT}/results/phase1/latest.txt"

log "=== Phase 1 v4 eval sweep ==="
log "task_suite : ${TASK_SUITE}"
log "ckpt_root  : ${CKPT_ROOT}"
log "out_root   : ${OUT_ROOT}"

setup_env

fail_count=0
eval_count=0

# Discover checkpoints: dirs containing v4_config.json
while IFS= read -r cfg_path; do
  ckpt_dir="$(dirname "${cfg_path}")"
  condition_name="$(basename "${ckpt_dir}")"
  out_dir="${OUT_ROOT}/${condition_name}"
  log_file="${LOG_ROOT}/${condition_name}.log"

  log "EVAL: ${condition_name}"
  mkdir -p "${out_dir}"

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

  if [ "${rc}" -eq 0 ]; then
    log "DONE: ${condition_name}"
    eval_count=$((eval_count + 1))
  else
    log "FAILED: ${condition_name} rc=${rc}; see ${log_file}"
    fail_count=$((fail_count + 1))
    is_true "${STOP_ON_FAIL}" && exit "${rc}"
  fi
done < <(find "${CKPT_ROOT}" -name "v4_config.json" | sort)

log "Evaluated ${eval_count} checkpoint(s)."
[ "${fail_count}" -gt 0 ] && log "WARNING: ${fail_count} eval(s) failed." && exit 1

log "=== v4 eval sweep complete ==="
log "results : ${OUT_ROOT}"
echo ""
echo "  To summarize:"
echo "    bash scripts/libero/phase1/summarize_v4_temporal_query_eval.sh ${OUT_ROOT}"
