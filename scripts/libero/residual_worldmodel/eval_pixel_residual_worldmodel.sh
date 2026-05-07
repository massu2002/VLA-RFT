#!/usr/bin/env bash
# eval_pixel_residual_worldmodel.sh — Evaluate a trained PixelResidualWorldModel.
#
# Usage:
#   bash scripts/libero/residual_worldmodel/eval_pixel_residual_worldmodel.sh \
#       MODEL_DIR=checkpoints/libero/PixelResidualWM/spatial/pixel_residual/v1/s42 \
#       TASK_SUITE=spatial
#
#   Or via env vars:
#     MODEL_DIR=... TASK_SUITE=spatial \
#       bash scripts/libero/residual_worldmodel/eval_pixel_residual_worldmodel.sh
#
# Key env-var overrides:
#   MODEL_DIR (required)    — directory containing pixel_residual_config.json + *.pt
#   TASK_SUITE              — spatial | object | goal | 10
#   CONDITION_NAME          — label in output JSON (default: basename of MODEL_DIR)
#   OUTPUT_DIR              — output dir (default: results/phase1/residual_worldmodel/<condition_name>)
#   NUM_EVAL_WINDOWS        — total windows (default: 200)
#   NUM_RANKING_WINDOWS     — windows for ranking eval (default: 100)
#   EVAL_HORIZON            — rollout horizon H (default: 7)
#   SEED, DEVICE
#   DRY_RUN_WINDOWS         — >0 limits windows per task for quick sanity check
#   SAVE_DEBUG_IMAGES       — 1 to save debug PNGs
#   PHASE0_COMPATIBLE       — 1 to use Phase 0-compatible ranking/window protocol
#   TASK_INDICES            — comma-separated task ids (empty = all tasks)
#   WINDOW_POSITION_MODE    — random | episode_phases
#   NUM_EVAL_EPISODES_PER_TASK — episode count per task for episode_phases

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

# ---------------------------------------------------------------------------
# Parse key=value args (positional) then fall back to env vars
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
    PHASE0_COMPATIBLE=*) PHASE0_COMPATIBLE="${_kv#PHASE0_COMPATIBLE=}" ;;
    TASK_INDICES=*)     TASK_INDICES="${_kv#TASK_INDICES=}" ;;
    WINDOW_POSITION_MODE=*) WINDOW_POSITION_MODE="${_kv#WINDOW_POSITION_MODE=}" ;;
    NUM_EVAL_EPISODES_PER_TASK=*) NUM_EVAL_EPISODES_PER_TASK="${_kv#NUM_EVAL_EPISODES_PER_TASK=}" ;;
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
HELDOUT_RATIO="${HELDOUT_RATIO:-0.2}"
LPIPS_BATCH_SIZE="${LPIPS_BATCH_SIZE:-4}"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
DRY_RUN_WINDOWS="${DRY_RUN_WINDOWS:-0}"
SAVE_DEBUG_IMAGES="${SAVE_DEBUG_IMAGES:-0}"
PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE:-0}"
TASK_INDICES="${TASK_INDICES:-}"
WINDOW_POSITION_MODE="${WINDOW_POSITION_MODE:-random}"
NUM_EVAL_EPISODES_PER_TASK="${NUM_EVAL_EPISODES_PER_TASK:-0}"

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/results/phase1/residual_worldmodel}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${CONDITION_NAME}}"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ ! -d "${MODEL_DIR}" ]; then
  echo "[eval_pxr] ERROR: MODEL_DIR not found: ${MODEL_DIR}" >&2
  exit 1
fi
if [ ! -f "${MODEL_DIR}/pixel_residual_config.json" ]; then
  echo "[eval_pxr] ERROR: pixel_residual_config.json not found in ${MODEL_DIR}" >&2
  echo "[eval_pxr] Ensure training has completed and save_pretrained() was called." >&2
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
  "PHASE0_COMPAT"      "${PHASE0_COMPATIBLE}" \
  "WINDOW_POS_MODE"    "${WINDOW_POSITION_MODE}" \
  "EVAL_EPISODES_TASK" "${NUM_EVAL_EPISODES_PER_TASK}" \
  "TASK_INDICES"       "${TASK_INDICES:-<all>}" \
  "SEED"               "${SEED}" \
  "DEVICE"             "${DEVICE}" \
  "DATA_ROOT"          "${DATA_ROOT}" \
  "LOGFILE"            "${LOGFILE}"

# ---------------------------------------------------------------------------
# Build eval command
# ---------------------------------------------------------------------------
EVAL_ARGS=(
  -m worldmodel.residual_worldmodel.eval_pixel_residual_libero
  --task-suite          "${TASK_SUITE}"
  --model-dir           "${MODEL_DIR}"
  --data-root           "${DATA_ROOT}"
  --output-dir          "${OUTPUT_DIR}"
  --condition-name      "${CONDITION_NAME}"
  --num-eval-windows    "${NUM_EVAL_WINDOWS}"
  --num-ranking-windows "${NUM_RANKING_WINDOWS}"
  --num-shuffle-reps    "${NUM_SHUFFLE_REPS}"
  --eval-horizon        "${EVAL_HORIZON}"
  --eval-batch-size     "${EVAL_BATCH_SIZE}"
  --heldout-ratio       "${HELDOUT_RATIO}"
  --lpips-batch-size    "${LPIPS_BATCH_SIZE}"
  --seed                "${SEED}"
  --device              "${DEVICE}"
  --window-position-mode "${WINDOW_POSITION_MODE}"
)

if [ "${NUM_EVAL_EPISODES_PER_TASK}" -gt 0 ] 2>/dev/null; then
  EVAL_ARGS+=(--num-eval-episodes-per-task "${NUM_EVAL_EPISODES_PER_TASK}")
fi

if [ -n "${TASK_INDICES}" ]; then
  EVAL_ARGS+=(--task-indices "${TASK_INDICES}")
fi

if [ "${PHASE0_COMPATIBLE}" = "1" ]; then
  EVAL_ARGS+=(--phase0-compatible)
fi

if [ "${DRY_RUN_WINDOWS}" -gt 0 ] 2>/dev/null; then
  EVAL_ARGS+=(--dry-run-windows "${DRY_RUN_WINDOWS}")
fi

if [ "${SAVE_DEBUG_IMAGES}" = "1" ]; then
  EVAL_ARGS+=(--save-debug-images)
fi

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo ""
echo "  Starting evaluation → ${OUTPUT_DIR}"
echo ""

export TF_CPP_MIN_LOG_LEVEL=3

run_visualize_command "${OUTPUT_DIR}" "${LOGFILE}" "${EVAL_ARGS[@]}"

echo ""
echo "  Evaluation complete."
echo "    output : ${OUTPUT_DIR}"
echo "    log    : ${LOGFILE}"
echo ""
if [ -f "${OUTPUT_DIR}/aggregate_metrics.json" ]; then
  echo "  Key metrics:"
  python3 -c "
import json, math
m = json.load(open('${OUTPUT_DIR}/aggregate_metrics.json'))
metrics = m.get('metrics', m)
def sf(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 'N/A'
    return f'{float(v):.5f}'
print(f'    full_mse     : {sf(metrics.get(\"full_mse\"))}')
print(f'    gripper_mse  : {sf(metrics.get(\"gripper_mse\"))}')
print(f'    dynamic_mse  : {sf(metrics.get(\"dynamic_mse\"))}')
print(f'    pairwise_acc : {sf(metrics.get(\"pairwise_acc\"))}')
print(f'    lpips_gap    : {sf(metrics.get(\"lpips_gap\"))}')
" 2>/dev/null || true
fi
