#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/worldmodel/_compat:${REPO_ROOT}/train/verl:${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="false"

# ---------------------------------------------------------
# Defaults
# ---------------------------------------------------------
DATE="${DATE:-20260410}"
EXP_NAME="${EXP_NAME:-worldmodel_scratch}"
TRAINED_MODEL_DIR="${TRAINED_MODEL_DIR:-}"

DEVICE="${DEVICE:-auto}"
NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-100}"
EVAL_HORIZON="${EVAL_HORIZON:-7}"
HELDOUT_RATIO="${HELDOUT_RATIO:-0.2}"
SPLIT_MODE="${SPLIT_MODE:-fallback_all}"
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-2}"
DISPLAY_FRAMES="${DISPLAY_FRAMES:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SAVE_CASEBOOK_COUNT="${SAVE_CASEBOOK_COUNT:-5}"
CHUNK_FUTURE_LENGTH="${CHUNK_FUTURE_LENGTH:-7}"

COMPARE_BASE="${COMPARE_BASE:-0}"
RUN_ACTION_SENSITIVITY="${RUN_ACTION_SENSITIVITY:-0}"
RUN_DIAGNOSTIC_CHUNK="${RUN_DIAGNOSTIC_CHUNK:-0}"

# ---------------------------------------------------------
# Usage
# ---------------------------------------------------------
usage() {
  cat <<EOF
Usage:
  $(basename "$0") [task_suite1] [task_suite2] ...

Examples:
  $(basename "$0") object
  $(basename "$0") object spatial
  DATE=20260413 EXP_NAME=worldmodel_scratch $(basename "$0") goal
  NUM_EVAL_WINDOWS=200 RUN_ACTION_SENSITIVITY=1 $(basename "$0") object

Notes:
  - 位置引数には LIBERO の task suite を渡します
    例: object / spatial / goal / 10
  - 引数がない場合は、LIBERO_TASKS 環境変数を使います
  - それも無い場合は object を評価します
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# ---------------------------------------------------------
# Task suites to evaluate
# Priority:
#   1) positional args
#   2) LIBERO_TASKS env
#   3) default: object
# ---------------------------------------------------------
TASKS=()

if [[ "$#" -gt 0 ]]; then
  TASKS=("$@")
elif [[ -n "${LIBERO_TASKS:-}" ]]; then
  # shellcheck disable=SC2206
  TASKS=(${LIBERO_TASKS})
else
  TASKS=(object)
fi

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
is_true() {
  case "${1}" in
    1|true|TRUE|True|yes|YES|Yes|y|Y|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

OUTPUT_ROOT="${REPO_ROOT}/worldmodel/eval_reports/libero/${DATE}_${EXP_NAME}"
mkdir -p "${OUTPUT_ROOT}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "LIBERO task suites: ${TASKS[*]}"
echo "DATE=${DATE}"
echo "EXP_NAME=${EXP_NAME}"
echo "TRAINED_MODEL_DIR=${TRAINED_MODEL_DIR:-<auto>}"
echo "DEVICE=${DEVICE}"
echo "NUM_EVAL_WINDOWS=${NUM_EVAL_WINDOWS}"
echo "EVAL_HORIZON=${EVAL_HORIZON}"
echo "HELDOUT_RATIO=${HELDOUT_RATIO}"
echo "SPLIT_MODE=${SPLIT_MODE}"
echo "DECODE_CHUNK_SIZE=${DECODE_CHUNK_SIZE}"
echo "DISPLAY_FRAMES=${DISPLAY_FRAMES}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "SAVE_CASEBOOK_COUNT=${SAVE_CASEBOOK_COUNT}"
echo "CHUNK_FUTURE_LENGTH=${CHUNK_FUTURE_LENGTH}"
echo "COMPARE_BASE=${COMPARE_BASE}"
echo "RUN_ACTION_SENSITIVITY=${RUN_ACTION_SENSITIVITY}"
echo "RUN_DIAGNOSTIC_CHUNK=${RUN_DIAGNOSTIC_CHUNK}"
echo "Output root: ${OUTPUT_ROOT}"

for task_suite in "${TASKS[@]}"; do
  suite_out_dir="${OUTPUT_ROOT}/${task_suite}"
  mkdir -p "${suite_out_dir}"

  echo "===== Evaluating LIBERO suite: ${task_suite} ====="

  extra_args=()

  if [[ -n "${TRAINED_MODEL_DIR}" ]]; then
    extra_args+=(--trained-model-dir "${TRAINED_MODEL_DIR}")
  fi

  if is_true "${COMPARE_BASE}"; then
    extra_args+=(--compare-base)
  fi

  if is_true "${RUN_ACTION_SENSITIVITY}"; then
    extra_args+=(--run-action-sensitivity)
  fi

  if is_true "${RUN_DIAGNOSTIC_CHUNK}"; then
    extra_args+=(--run-diagnostic-chunk)
  fi

  python -m worldmodel.libero.visualize \
    --task-suite "${task_suite}" \
    --data-root "${REPO_ROOT}/data/modified_libero_rlds" \
    --base-model-root "${REPO_ROOT}/checkpoints/libero/WorldModel" \
    --trained-model-root "${REPO_ROOT}/checkpoints/libero/WorldModel" \
    --trained-exp-name "${EXP_NAME}" \
    --date "${DATE}" \
    --visual-tokenizer "${REPO_ROOT}/checkpoints/libero/WorldModel/Tokenizer" \
    --output-dir "${suite_out_dir}" \
    --device "${DEVICE}" \
    --seed 42 \
    --num-eval-windows "${NUM_EVAL_WINDOWS}" \
    --eval-horizon "${EVAL_HORIZON}" \
    --heldout-ratio "${HELDOUT_RATIO}" \
    --split-mode "${SPLIT_MODE}" \
    --decode-chunk-size "${DECODE_CHUNK_SIZE}" \
    --display-frames "${DISPLAY_FRAMES}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --save-casebook-count "${SAVE_CASEBOOK_COUNT}" \
    --chunk-future-length "${CHUNK_FUTURE_LENGTH}" \
    "${extra_args[@]}"

  echo "===== Finished LIBERO suite: ${task_suite} ====="
done

echo "All evaluations completed."
echo "Results saved under: ${OUTPUT_ROOT}"