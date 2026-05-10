#!/usr/bin/env bash
# summarize_v4_temporal_query_eval.sh — Aggregate v4 Phase 1 WM results into comparison.md
#
# Usage:
#   bash scripts/libero/phase1/summarize_v4_temporal_query_eval.sh <phase1-output-dir>
#   bash scripts/libero/phase1/summarize_v4_temporal_query_eval.sh   # uses results/phase1/latest.txt
#
# Reads per-condition outputs under <phase1-dir>/<condition>/:
#   aggregate_metrics.json  (v4 metrics including pairwise_acc_score, score_gap_mean, …)
#   metrics_by_task.csv
#
# Writes:
#   <phase1-dir>/v4_summary.json
#   <phase1-dir>/v4_comparison.md
#   <phase1-dir>/v4_comparison_table.csv
#
# This script is a thin wrapper around the backend summarize script.
# It also compares v4 results against any v1/v3 aggregate_metrics.json present.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

source "${WM_SCRIPTS}/common.sh"

# ---------------------------------------------------------------------------
# Resolve output directory
# ---------------------------------------------------------------------------
if [ -n "${1:-}" ]; then
  PHASE1_DIR="$1"
elif [ -f "${REPO_ROOT}/results/phase1/latest.txt" ]; then
  PHASE1_DIR=$(cat "${REPO_ROOT}/results/phase1/latest.txt")
else
  echo "Usage: $0 <phase1-output-dir>" >&2
  exit 1
fi

[ -d "${PHASE1_DIR}" ] || { echo "Directory not found: ${PHASE1_DIR}" >&2; exit 1; }

log() { echo "[summarize-v4-phase1] $(date +%H:%M:%S) $*"; }
log "Summarizing v4 Phase 1 results: ${PHASE1_DIR}"

setup_env

# ---------------------------------------------------------------------------
# Run the v4-specific summarize backend
# ---------------------------------------------------------------------------
bash "${WM_SCRIPTS}/summarize_v4_temporal_query_eval.sh" "${PHASE1_DIR}"

# ---------------------------------------------------------------------------
# Also integrate into comparison.md if v1/v3 conditions co-exist
# ---------------------------------------------------------------------------
# (existing summarize_residual_wm_eval.sh handles combined view)
if find "${PHASE1_DIR}" -maxdepth 2 -name "pixel_residual_config.json" -quit 2>/dev/null | grep -q .; then
  log "v1/v3 conditions detected alongside v4 — also running combined summarize_residual_wm_eval.sh"
  bash "${SCRIPT_DIR}/summarize_residual_wm_eval.sh" "${PHASE1_DIR}" || true
fi

log "Done."
echo ""
echo "  v4 comparison: ${PHASE1_DIR}/v4_comparison.md"
echo "  v4 summary   : ${PHASE1_DIR}/v4_summary.json"
echo "  v4 table     : ${PHASE1_DIR}/v4_comparison_table.csv"
