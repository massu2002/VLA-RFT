#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/worldmodel/_compat:${REPO_ROOT}/train/verl:${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM="false"

DATE="${DATE:-20260409}"
EXP_NAME="${EXP_NAME:-worldmodel_scratch}"
TRAINED_MODEL_DIR="${TRAINED_MODEL_DIR:-}"
DISPLAY_FRAMES="${DISPLAY_FRAMES:-12}"
CHUNK_FUTURE_LENGTH="${CHUNK_FUTURE_LENGTH:-8}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
DEVICE="${DEVICE:-auto}"

if [ -n "${LIBERO_TASKS:-}" ]; then
  read -r -a TASKS <<< "${LIBERO_TASKS}"
else
  TASKS=(spatial)
fi

OUTPUT_ROOT="${REPO_ROOT}/worldmodel/visualizations/libero/${DATE}_${EXP_NAME}"
mkdir -p "${OUTPUT_ROOT}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "LIBERO tasks: ${TASKS[*]}"
echo "DATE=${DATE}"
echo "EXP_NAME=${EXP_NAME}"
echo "TRAINED_MODEL_DIR=${TRAINED_MODEL_DIR:-<auto>}"
echo "Output root: ${OUTPUT_ROOT}"

for task_suite in "${TASKS[@]}"; do
  suite_out_dir="${OUTPUT_ROOT}/${task_suite}"
  mkdir -p "${suite_out_dir}"

  echo "===== Visualizing LIBERO suite: ${task_suite} ====="

  extra_args=()
  if [ -n "${TRAINED_MODEL_DIR}" ]; then
    extra_args+=(--trained-model-dir "${TRAINED_MODEL_DIR}")
  fi

  for task_index in $(seq 0 9); do
    python -m worldmodel.libero.visualize \
      --task-suite "${task_suite}" \
      --task-index "${task_index}" \
      --data-root "${REPO_ROOT}/data/modified_libero_rlds" \
      --base-model-root "${REPO_ROOT}/checkpoints/libero/WorldModel" \
      --trained-model-root "${REPO_ROOT}/checkpoints/libero/WorldModel" \
      --trained-exp-name "${EXP_NAME}" \
      --date "${DATE}" \
      --visual-tokenizer "${REPO_ROOT}/checkpoints/libero/WorldModel/Tokenizer" \
      --output-dir "${suite_out_dir}" \
      --episode-index "${EPISODE_INDEX}" \
      --chunk-future-length "${CHUNK_FUTURE_LENGTH}" \
      --display-frames "${DISPLAY_FRAMES}" \
      --device "${DEVICE}" \
      "${extra_args[@]}"
  done

  echo "===== Finished LIBERO suite: ${task_suite} ====="
done