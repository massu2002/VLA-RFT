#!/usr/bin/env bash
# summarize_v4_core_sweep.sh — Phase 1 frontend: aggregate v4 core sweep results.
#
# Usage:
#   RUN_NAME=v4_core_sweep_spatial bash scripts/libero/phase1/summarize_v4_core_sweep.sh
#   RUN_NAME=v4_core_sweep_spatial OUT_ROOT=/custom/path bash ...
#
# Outputs in ${OUT_ROOT}/summary/:
#   v4_core_sweep_summary.md
#   v4_core_sweep_summary.csv
#   v4_core_sweep_summary.json
#   v4_core_sweep_ranking.csv

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"

source "${WM_SCRIPTS}/common.sh"

RUN_NAME="${RUN_NAME:?'RUN_NAME is required. Set RUN_NAME=<sweep run name>'}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/residual_worldmodel/${RUN_NAME}}"
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/libero/phase1/v4_core_sweep.json}"
TASK_SUITE="${TASK_SUITE:-}"

[ -d "${OUT_ROOT}" ] || { echo "OUT_ROOT not found: ${OUT_ROOT}" >&2; exit 1; }

log() { echo "[summarize-v4-phase1] $(date +%H:%M:%S) $*"; }
log "Summarizing: ${OUT_ROOT}"

setup_env

export RUN_NAME OUT_ROOT SWEEP_CONFIG TASK_SUITE

bash "${WM_SCRIPTS}/summarize_v4_core_sweep.sh"

SUMMARY_DIR="${OUT_ROOT}/summary"
log ""
log "Summary files:"
ls -1 "${SUMMARY_DIR}"/v4_core_sweep_* 2>/dev/null | sed 's/^/  /'

log ""
log "Next:"
log "  Best checkpoint:  RUN_NAME=${RUN_NAME} BEST_CRITERION=hybrid_score bash scripts/libero/residual_worldmodel/select_best_v4_for_rft.sh"
log "  RFT sweep:        RUN_NAME=${RUN_NAME} bash scripts/libero/phase1/run_v4_selected_rft_sweep.sh"
