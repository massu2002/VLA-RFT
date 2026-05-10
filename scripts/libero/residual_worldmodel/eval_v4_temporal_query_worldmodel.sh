#!/usr/bin/env bash
# eval_v4_temporal_query_worldmodel.sh — Evaluate a trained TemporalDynamicQueryResidualWM (v4).
#
# Usage:
#   bash scripts/libero/residual_worldmodel/eval_v4_temporal_query_worldmodel.sh \
#       MODEL_DIR=checkpoints/libero/TemporalQueryResidualWM/spatial/v4b/s42 \
#       TASK_SUITE=spatial
#
#   Or via env vars:
#     MODEL_DIR=... TASK_SUITE=spatial \
#       bash scripts/libero/residual_worldmodel/eval_v4_temporal_query_worldmodel.sh
#
# Key env-var overrides:
#   MODEL_DIR (required)    — directory containing v4_config.json + *.pt
#   TASK_SUITE              — spatial | object | goal | 10
#   CONDITION_NAME          — label in output (default: basename of MODEL_DIR)
#   OUTPUT_DIR              — output dir
#   NUM_EVAL_WINDOWS        — total windows (default: 200)
#   NUM_RANKING_WINDOWS     — windows for ranking eval (default: 100)
#   EVAL_HORIZON            — rollout horizon H (default: 7)
#   SEED, DEVICE
#   DRY_RUN_WINDOWS         — >0 limits windows per task for quick sanity check
#   SAVE_DEBUG_VISUALS      — 1 to save v4 debug PNGs (masks, queries, overlays)
#   ACTION_ABLATION         — 1 to run action variant diagnostics
#   WINDOW_POSITION_MODE    — random | episode_phases
#   WINDOW_MANIFEST         — path to existing window_manifest.json
#   USE_WINDOW_MANIFEST     — 1 to reuse manifest

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

# ---------------------------------------------------------------------------
# Parse key=value args (positional)
# ---------------------------------------------------------------------------
for _kv in "$@"; do
  case "${_kv}" in
    MODEL_DIR=*)       MODEL_DIR="${_kv#MODEL_DIR=}"       ;;
    TASK_SUITE=*)      TASK_SUITE="${_kv#TASK_SUITE=}"     ;;
    CONDITION_NAME=*)  CONDITION_NAME="${_kv#CONDITION_NAME=}" ;;
    OUTPUT_DIR=*)      OUTPUT_DIR="${_kv#OUTPUT_DIR=}"     ;;
    NUM_EVAL_WINDOWS=*)  NUM_EVAL_WINDOWS="${_kv#NUM_EVAL_WINDOWS=}"   ;;
    NUM_RANKING_WINDOWS=*) NUM_RANKING_WINDOWS="${_kv#NUM_RANKING_WINDOWS=}" ;;
    EVAL_HORIZON=*)    EVAL_HORIZON="${_kv#EVAL_HORIZON=}" ;;
    SEED=*)            SEED="${_kv#SEED=}"                 ;;
    DEVICE=*)          DEVICE="${_kv#DEVICE=}"             ;;
    DRY_RUN_WINDOWS=*) DRY_RUN_WINDOWS="${_kv#DRY_RUN_WINDOWS=}" ;;
    SAVE_DEBUG_IMAGES=*) SAVE_DEBUG_IMAGES="${_kv#SAVE_DEBUG_IMAGES=}" ;;
    SAVE_DEBUG_VISUALS=*) SAVE_DEBUG_VISUALS="${_kv#SAVE_DEBUG_VISUALS=}" ;;
    TASK_INDICES=*)     TASK_INDICES="${_kv#TASK_INDICES=}" ;;
    WINDOW_POSITION_MODE=*) WINDOW_POSITION_MODE="${_kv#WINDOW_POSITION_MODE=}" ;;
    NUM_EVAL_EPISODES_PER_TASK=*) NUM_EVAL_EPISODES_PER_TASK="${_kv#NUM_EVAL_EPISODES_PER_TASK=}" ;;
    ACTION_ABLATION=*) ACTION_ABLATION="${_kv#ACTION_ABLATION=}" ;;
    DEBUG_NUM_TASKS=*) DEBUG_NUM_TASKS="${_kv#DEBUG_NUM_TASKS=}" ;;
    DEBUG_WINDOWS_PER_TASK=*) DEBUG_WINDOWS_PER_TASK="${_kv#DEBUG_WINDOWS_PER_TASK=}" ;;
    WINDOW_MANIFEST=*) WINDOW_MANIFEST="${_kv#WINDOW_MANIFEST=}" ;;
    USE_WINDOW_MANIFEST=*) USE_WINDOW_MANIFEST="${_kv#USE_WINDOW_MANIFEST=}" ;;
  esac
done

# ---------------------------------------------------------------------------
# Config (with defaults)
# ---------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:?'MODEL_DIR is required. Pass MODEL_DIR=<path>'}"
TASK_SUITE="${TASK_SUITE:-spatial}"
CONDITION_NAME="${CONDITION_NAME:-$(basename "${MODEL_DIR}")}"
DATA_ROOT="${DATA_ROOT:-$(default_libero_data_root)}"

NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-200}"
NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS:-100}"
NUM_SHUFFLE_REPS="${NUM_SHUFFLE_REPS:-3}"
EVAL_HORIZON="${EVAL_HORIZON:-7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"


SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
DRY_RUN_WINDOWS="${DRY_RUN_WINDOWS:-0}"
SAVE_DEBUG_IMAGES="${SAVE_DEBUG_IMAGES:-0}"
SAVE_DEBUG_VISUALS="${SAVE_DEBUG_VISUALS:-0}"
TASK_INDICES="${TASK_INDICES:-}"
WINDOW_POSITION_MODE="${WINDOW_POSITION_MODE:-random}"
NUM_EVAL_EPISODES_PER_TASK="${NUM_EVAL_EPISODES_PER_TASK:-0}"
ACTION_ABLATION="${ACTION_ABLATION:-0}"
DEBUG_NUM_TASKS="${DEBUG_NUM_TASKS:-3}"
DEBUG_WINDOWS_PER_TASK="${DEBUG_WINDOWS_PER_TASK:-3}"
WINDOW_MANIFEST="${WINDOW_MANIFEST:-}"
USE_WINDOW_MANIFEST="${USE_WINDOW_MANIFEST:-0}"

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/results/phase1/residual_worldmodel}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${CONDITION_NAME}}"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ ! -d "${MODEL_DIR}" ]; then
  echo "[eval_v4] ERROR: MODEL_DIR not found: ${MODEL_DIR}" >&2
  exit 1
fi
if [ ! -f "${MODEL_DIR}/v4_config.json" ]; then
  echo "[eval_v4] ERROR: v4_config.json not found in ${MODEL_DIR}" >&2
  echo "[eval_v4] Ensure v4 training has completed and save_pretrained() was called." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
setup_env

LOGFILE="${OUTPUT_DIR}/eval.log"

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print_run_summary \
  "TASK_SUITE"         "${TASK_SUITE}" \
  "MODEL_DIR"          "${MODEL_DIR}" \
  "CONDITION_NAME"     "${CONDITION_NAME}" \
  "OUTPUT_DIR"         "${OUTPUT_DIR}" \
  "NUM_EVAL_WINDOWS"   "${NUM_EVAL_WINDOWS}" \
  "NUM_RANKING_WIN"    "${NUM_RANKING_WINDOWS}" \
  "EVAL_HORIZON"       "${EVAL_HORIZON}" \
  "DRY_RUN_WINDOWS"    "${DRY_RUN_WINDOWS}" \
  "WINDOW_POS_MODE"    "${WINDOW_POSITION_MODE}" \
  "EVAL_EPISODES_TASK" "${NUM_EVAL_EPISODES_PER_TASK}" \
  "TASK_INDICES"       "${TASK_INDICES:-<all>}" \
  "ACTION_ABLATION"    "${ACTION_ABLATION}" \
  "DEBUG_VISUALS"      "${SAVE_DEBUG_VISUALS}" \
  "USE_MANIFEST"       "${USE_WINDOW_MANIFEST}" \
  "WINDOW_MANIFEST"    "${WINDOW_MANIFEST:-<none>}" \
  "SEED"               "${SEED}" \
  "DEVICE"             "${DEVICE}" \
  "DATA_ROOT"          "${DATA_ROOT}" \
  "LOGFILE"            "${LOGFILE}"

# ---------------------------------------------------------------------------
# Build eval command
# ---------------------------------------------------------------------------
EVAL_ARGS=(
  -m worldmodel.residual_worldmodel.eval_v4_temporal_query_libero
  --task-suite           "${TASK_SUITE}"
  --model-dir            "${MODEL_DIR}"
  --data-root            "${DATA_ROOT}"
  --output-dir           "${OUTPUT_DIR}"
  --condition-name       "${CONDITION_NAME}"
  --num-eval-windows     "${NUM_EVAL_WINDOWS}"
  --num-ranking-windows  "${NUM_RANKING_WINDOWS}"
  --num-shuffle-reps     "${NUM_SHUFFLE_REPS}"
  --eval-horizon         "${EVAL_HORIZON}"
  --eval-batch-size      "${EVAL_BATCH_SIZE}"
  --seed                 "${SEED}"
  --device               "${DEVICE}"
  --window-position-mode "${WINDOW_POSITION_MODE}"
)

if [ "${NUM_EVAL_EPISODES_PER_TASK}" -gt 0 ] 2>/dev/null; then
  EVAL_ARGS+=(--num-eval-episodes-per-task "${NUM_EVAL_EPISODES_PER_TASK}")
fi
if [ -n "${TASK_INDICES}" ]; then
  EVAL_ARGS+=(--task-indices "${TASK_INDICES}")
fi
if [ "${DRY_RUN_WINDOWS}" -gt 0 ] 2>/dev/null; then
  EVAL_ARGS+=(--dry-run-windows "${DRY_RUN_WINDOWS}")
fi
if [ "${SAVE_DEBUG_IMAGES}" = "1" ]; then
  EVAL_ARGS+=(--save-debug-images)
fi
if [ "${ACTION_ABLATION}" = "1" ]; then
  EVAL_ARGS+=(--action-ablation)
fi
if [ "${SAVE_DEBUG_VISUALS}" = "1" ]; then
  EVAL_ARGS+=(--save-debug-visuals --debug-num-tasks "${DEBUG_NUM_TASKS}" --debug-windows-per-task "${DEBUG_WINDOWS_PER_TASK}")
fi
if [ "${USE_WINDOW_MANIFEST}" = "1" ]; then
  EVAL_ARGS+=(--use-window-manifest --window-manifest "${WINDOW_MANIFEST}")
fi

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo ""
echo "  Starting v4 evaluation → ${OUTPUT_DIR}"
echo ""

export TF_CPP_MIN_LOG_LEVEL=3

run_visualize_command "${OUTPUT_DIR}" "${LOGFILE}" "${EVAL_ARGS[@]}"

echo ""
echo "  Evaluation complete."
echo "    output : ${OUTPUT_DIR}"
echo "    log    : ${LOGFILE}"
echo ""
if [ -f "${OUTPUT_DIR}/aggregate_metrics.json" ]; then
  echo "  Key v4 metrics:"
  python3 -c "
import json, math
m = json.load(open('${OUTPUT_DIR}/aggregate_metrics.json'))
metrics = m.get('metrics', m)
def sf(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 'N/A'
    return f'{float(v):.5f}'
print(f'    full_mse              : {sf(metrics.get(\"full_mse\"))}')
print(f'    pairwise_acc_lpips    : {sf(metrics.get(\"pairwise_acc_lpips\", metrics.get(\"pairwise_acc\")))}')
print(f'    pairwise_acc_score    : {sf(metrics.get(\"pairwise_acc_score\"))}')
print(f'    score_gap_mean        : {sf(metrics.get(\"score_gap_mean\"))}')
print(f'    fuser_mask_mean       : {sf(metrics.get(\"fuser_mask_mean\"))}')
print(f'    dynamic_mask_mean     : {sf(metrics.get(\"dynamic_mask_mean\"))}')
print(f'    future_dq_norm        : {sf(metrics.get(\"future_dynamic_query_norm\"))}')
" 2>/dev/null || true
fi
