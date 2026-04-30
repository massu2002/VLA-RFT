#!/usr/bin/env bash
# init_phase1_run.sh — Phase 1 実験の共有出力ディレクトリを初期化する。
#
# 複数のPCで条件を分担して実験を行う際、最初に1回だけ実行する。
# OUT_ROOT を生成して run_config.json を書き込み、パスを標準出力に表示する。
# 各PCはそのパスを OUT_ROOT として run_residual_wm_eval.sh に渡す。
#
# Usage（作業用PCで1回だけ実行）:
#   bash scripts/libero/phase1/init_phase1_run.sh [spatial|object|goal|10]
#
# 出力例:
#   [init] OUT_ROOT: /home/.../results/phase1/20260430_120000_spatial
#   [init] Wrote run_config.json
#   [init] Copy the OUT_ROOT and use it on each PC:
#   [init]   export OUT_ROOT=/home/.../results/phase1/20260430_120000_spatial
#
# 各PCでの実行例（PC1〜PC3それぞれで）:
#   export OUT_ROOT=/home/.../results/phase1/20260430_120000_spatial
#   TRAIN_CONDITIONS=pixel                      bash scripts/libero/phase1/run_residual_wm_eval.sh spatial
#   TRAIN_CONDITIONS=pixel_residual             bash scripts/libero/phase1/run_residual_wm_eval.sh spatial
#   TRAIN_CONDITIONS=pixel_residual_roi_dynamic bash scripts/libero/phase1/run_residual_wm_eval.sh spatial
#
# 全PC完了後にサマリーを生成:
#   bash scripts/libero/phase1/summarize_residual_wm_eval.sh "${OUT_ROOT}"

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

source "${WM_SCRIPTS}/common.sh"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"
ALL_CONDITIONS="${ALL_CONDITIONS:-pixel pixel_residual pixel_residual_roi_dynamic}"
NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-200}"
NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS:-100}"

DATE_TAG=$(timestamp)
RESULTS_ROOT="${REPO_ROOT}/results/phase1"
OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/${DATE_TAG}_${TASK_SUITE}}"

log() { echo "[init] $*"; }

# ---------------------------------------------------------------------------
# Create directory and write run_config.json
# ---------------------------------------------------------------------------
mkdir -p "${OUT_ROOT}"
mkdir -p "${RESULTS_ROOT}"

python3 - << PYEOF
import json
data = {
    "date":                 "${DATE_TAG}",
    "suite":                "${TASK_SUITE}",
    "seed":                 ${SEED},
    "smoke":                "${SMOKE}",
    "all_conditions":       "${ALL_CONDITIONS}".split(),
    "num_eval_windows":     ${NUM_EVAL_WINDOWS},
    "num_ranking_windows":  ${NUM_RANKING_WINDOWS},
    "initialized_by":       "init_phase1_run.sh",
}
json.dump(data, open("${OUT_ROOT}/run_config.json", "w"), indent=2)
print("  wrote: ${OUT_ROOT}/run_config.json")
PYEOF

echo "${OUT_ROOT}" > "${RESULTS_ROOT}/latest.txt"

# ---------------------------------------------------------------------------
# Print instructions for each PC
# ---------------------------------------------------------------------------
log "========================================"
log "OUT_ROOT: ${OUT_ROOT}"
log "========================================"
log ""
log "Copy the OUT_ROOT and run on each PC:"
log ""
log "  export OUT_ROOT=${OUT_ROOT}"
log ""

idx=1
for COND in ${ALL_CONDITIONS}; do
  log "  # PC${idx} — condition: ${COND}"
  log "  TRAIN_CONDITIONS=${COND} OUT_ROOT=\${OUT_ROOT} \\"
  log "    bash scripts/libero/phase1/run_residual_wm_eval.sh ${TASK_SUITE}"
  log ""
  idx=$(( idx + 1 ))
done

log "After all PCs finish:"
log "  bash scripts/libero/phase1/summarize_residual_wm_eval.sh \${OUT_ROOT}"
log ""
log "Checkpoint locations:"
for COND in ${ALL_CONDITIONS}; do
  log "  ${COND}: checkpoints/libero/PixelResidualWM/${TASK_SUITE}/${COND}/v1/s${SEED}/"
done

# Also export for callers that source this script
export OUT_ROOT
