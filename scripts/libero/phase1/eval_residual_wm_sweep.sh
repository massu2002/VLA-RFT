#!/usr/bin/env bash
# eval_residual_wm_sweep.sh — Evaluate all Phase 1 sweep checkpoints.
#
# This script scans a sweep checkpoint directory such as:
#   checkpoints/libero/PixelResidualWM/phase1_sweeps/<RUN_NAME>/<TASK_SUITE>/<target_mode>/<exp_name>/s<SEED>
#
# and writes evaluation results under:
#   results/phase1/phase1_sweeps/<RUN_NAME>/wm_eval/<exp_name>
#
# Usage:
#   RUN_NAME=phase1_spatial_core_20260501 \
#     bash scripts/libero/phase1/eval_residual_wm_sweep.sh spatial
#
# Smoke test:
#   SMOKE=1 RUN_NAME=phase1_spatial_core_20260501 \
#     bash scripts/libero/phase1/eval_residual_wm_sweep.sh spatial
#
# Useful overrides:
#   TASK_SUITE=spatial | object | goal | 10
#   RUN_NAME=phase1_spatial_core_20260501
#   CKPT_ROOT=/path/to/checkpoints/root
#   EVAL_ROOT=/path/to/output/root
#   SEED=42
#   NUM_EVAL_WINDOWS=200
#   NUM_RANKING_WINDOWS=100
#   VENV_NAME=.venv5090_eval
#   DEVICE=auto
#   WINDOW_POSITION_MODE=random | episode_phases
#   NUM_EVAL_EPISODES_PER_TASK=0
#   INCLUDE_REGEX='v3_.*'
#   EXCLUDE_REGEX='pixel_baseline'
#   DRY_RUN=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

source "${WM_SCRIPTS}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log() { echo "[phase1-eval-sweep] $(date +%H:%M:%S) $*"; }

TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
RUN_NAME="${RUN_NAME:-phase1_${TASK_SUITE}_core_20260501}"
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
VENV_NAME="${VENV_NAME:-.venv5090_eval}"
DEVICE="${DEVICE:-auto}"

CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/PixelResidualWM/phase1_sweeps/${RUN_NAME}/${TASK_SUITE}}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/results/phase1/phase1_sweeps/${RUN_NAME}/wm_eval}"
INCLUDE_REGEX="${INCLUDE_REGEX:-}"
EXCLUDE_REGEX="${EXCLUDE_REGEX:-}"

NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-200}"
NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS:-100}"
DRY_RUN_WINDOWS="${DRY_RUN_WINDOWS:-0}"
SAVE_DEBUG_IMAGES="${SAVE_DEBUG_IMAGES:-0}"
WINDOW_POSITION_MODE="${WINDOW_POSITION_MODE:-random}"
NUM_EVAL_EPISODES_PER_TASK="${NUM_EVAL_EPISODES_PER_TASK:-0}"

if is_true "${SMOKE}"; then
  NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS_SMOKE:-20}"
  NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS_SMOKE:-10}"
  DRY_RUN_WINDOWS="${DRY_RUN_WINDOWS_SMOKE:-5}"
fi

if [ ! -d "${CKPT_ROOT}" ]; then
  echo "[phase1-eval-sweep] ERROR: CKPT_ROOT not found: ${CKPT_ROOT}" >&2
  exit 2
fi

mkdir -p "${EVAL_ROOT}"
printf '%s\n' "${EVAL_ROOT}" > "${REPO_ROOT}/results/phase1/latest_sweep_eval.txt"

log "=== Phase 1 sweep evaluation ==="
log "task_suite        : ${TASK_SUITE}"
log "run_name          : ${RUN_NAME}"
log "ckpt_root         : ${CKPT_ROOT}"
log "eval_root         : ${EVAL_ROOT}"
log "seed              : ${SEED}"
log "num_eval_windows  : ${NUM_EVAL_WINDOWS}"
log "num_ranking_win   : ${NUM_RANKING_WINDOWS}"
log "dry_run_windows   : ${DRY_RUN_WINDOWS}"
log "window_pos_mode   : ${WINDOW_POSITION_MODE}"
log "eval_episodes/task: ${NUM_EVAL_EPISODES_PER_TASK}"
log "venv              : ${VENV_NAME}"
log "device            : ${DEVICE}"
[ -n "${INCLUDE_REGEX}" ] && log "include_regex     : ${INCLUDE_REGEX}"
[ -n "${EXCLUDE_REGEX}" ] && log "exclude_regex     : ${EXCLUDE_REGEX}"

mapfile -t CHECKPOINTS < <(
  find "${CKPT_ROOT}" -mindepth 3 -maxdepth 3 -type d -name "s${SEED}" | sort
)

if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
  echo "[phase1-eval-sweep] ERROR: no checkpoints found under ${CKPT_ROOT} with s${SEED}" >&2
  exit 3
fi

MANIFEST="${EVAL_ROOT}/eval_manifest.tsv"
echo -e "exp_name\ttarget_mode\tmodel_dir\toutput_dir\tstatus" > "${MANIFEST}"

for ckpt in "${CHECKPOINTS[@]}"; do
  exp_name=$(basename "$(dirname "${ckpt}")")
  target_mode=$(basename "$(dirname "$(dirname "${ckpt}")")")

  if [ -n "${INCLUDE_REGEX}" ] && ! [[ "${exp_name}" =~ ${INCLUDE_REGEX} ]]; then
    log "skip ${exp_name}: does not match INCLUDE_REGEX"
    continue
  fi
  if [ -n "${EXCLUDE_REGEX}" ] && [[ "${exp_name}" =~ ${EXCLUDE_REGEX} ]]; then
    log "skip ${exp_name}: matches EXCLUDE_REGEX"
    continue
  fi

  out_dir="${EVAL_ROOT}/${exp_name}"
  log "=== EVAL: ${exp_name} (${target_mode}) ==="
  echo -e "${exp_name}\t${target_mode}\t${ckpt}\t${out_dir}\trunning" >> "${MANIFEST}"

  if is_true "${DRY_RUN}"; then
    cat <<EOF
VENV_NAME=${VENV_NAME} MODEL_DIR=${ckpt} TASK_SUITE=${TASK_SUITE} CONDITION_NAME=${exp_name} OUTPUT_DIR=${out_dir} \\
NUM_EVAL_WINDOWS=${NUM_EVAL_WINDOWS} NUM_RANKING_WINDOWS=${NUM_RANKING_WINDOWS} DRY_RUN_WINDOWS=${DRY_RUN_WINDOWS} DEVICE=${DEVICE} \\
WINDOW_POSITION_MODE=${WINDOW_POSITION_MODE} NUM_EVAL_EPISODES_PER_TASK=${NUM_EVAL_EPISODES_PER_TASK} \\
bash scripts/libero/residual_worldmodel/eval_pixel_residual_worldmodel.sh
EOF
    continue
  fi

  VENV_NAME="${VENV_NAME}" \
  MODEL_DIR="${ckpt}" \
  TASK_SUITE="${TASK_SUITE}" \
  CONDITION_NAME="${exp_name}" \
  OUTPUT_DIR="${out_dir}" \
  NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS}" \
  NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS}" \
  DRY_RUN_WINDOWS="${DRY_RUN_WINDOWS}" \
  WINDOW_POSITION_MODE="${WINDOW_POSITION_MODE}" \
  NUM_EVAL_EPISODES_PER_TASK="${NUM_EVAL_EPISODES_PER_TASK}" \
  SAVE_DEBUG_IMAGES="${SAVE_DEBUG_IMAGES}" \
  DEVICE="${DEVICE}" \
  PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE:-0}" \
  TASK_INDICES="${TASK_INDICES:-}" \
    bash "${WM_SCRIPTS}/eval_pixel_residual_worldmodel.sh" \
    && log "done ${exp_name}" \
    || log "WARNING: eval failed for ${exp_name}; check ${out_dir}/eval.log"
done

if ! is_true "${DRY_RUN}"; then
  log "=== Summarizing sweep eval ==="
  VENV_NAME="${VENV_NAME}" \
    bash "${SCRIPT_DIR}/summarize_residual_wm_eval.sh" "${EVAL_ROOT}"
fi

log "=== Phase 1 sweep evaluation complete ==="
log "eval_root : ${EVAL_ROOT}"
log "summary   : ${EVAL_ROOT}/comparison.md"
