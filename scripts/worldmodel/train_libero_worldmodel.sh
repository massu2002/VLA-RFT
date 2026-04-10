#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/worldmodel/_compat:${REPO_ROOT}/train/verl:${PYTHONPATH:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM="false"

DATE="${DATE:-$(date +%Y%m%d)}"
EXP_NAME="${EXP_NAME:-worldmodel_scratch}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
WORLD_MODEL_BATCH_SIZE="${WORLD_MODEL_BATCH_SIZE:-16}"
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-1}"

if [ -n "${LIBERO_TASKS:-}" ]; then
  read -r -a TASKS <<< "${LIBERO_TASKS}"
else
  TASKS=(spatial object goal 10)
fi

mkdir -p "${REPO_ROOT}/logs/libero/WorldModel/${DATE}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "LIBERO tasks: ${TASKS[*]}"

TOTAL_PER_STEP=$((NPROC_PER_NODE * BATCH_SIZE_PER_DEVICE))
if (( WORLD_MODEL_BATCH_SIZE % TOTAL_PER_STEP != 0 )); then
  echo "WORLD_MODEL_BATCH_SIZE=${WORLD_MODEL_BATCH_SIZE} must be divisible by NPROC_PER_NODE*BATCH_SIZE_PER_DEVICE=${TOTAL_PER_STEP}" >&2
  exit 1
fi
GRAD_ACCUM=$((WORLD_MODEL_BATCH_SIZE / TOTAL_PER_STEP))

echo "World-model global batch size: ${WORLD_MODEL_BATCH_SIZE} (per-device=${BATCH_SIZE_PER_DEVICE}, grad-accum=${GRAD_ACCUM}, nproc=${NPROC_PER_NODE})"

for task_name in "${TASKS[@]}"; do
  model_template_dir="${REPO_ROOT}/checkpoints/libero/WorldModel/${task_name}"
  visual_tokenizer_dir="${REPO_ROOT}/checkpoints/libero/WorldModel/Tokenizer"
  output_dir="${REPO_ROOT}/checkpoints/libero/WorldModel/${task_name}/${DATE}_${EXP_NAME}"
  task_log="${REPO_ROOT}/logs/libero/WorldModel/${DATE}/${task_name}_output.log"

  mkdir -p "${output_dir}"

  if [ ! -d "${model_template_dir}" ]; then
    echo "World-model template directory not found: ${model_template_dir}" >&2
    exit 1
  fi
  if [ ! -d "${visual_tokenizer_dir}" ]; then
    echo "Visual tokenizer directory not found: ${visual_tokenizer_dir}" >&2
    exit 1
  fi

  extra_args=()
  if [ "${RESUME_PRETRAINED_WEIGHTS:-1}" = "1" ]; then
    extra_args+=(--load-pretrained-weights)
  fi

  echo "===== Starting LIBERO world-model training: ${task_name} ====="
  echo "Model template: ${model_template_dir}"
  echo "Tokenizer: ${visual_tokenizer_dir}"
  echo "Output: ${output_dir}"

  (
    cd "${REPO_ROOT}"
    torchrun --standalone --nnodes=1 \
      --nproc_per_node="${NPROC_PER_NODE}" \
      -m worldmodel.libero.train \
      --task-suite "${task_name}" \
      --data-root "${REPO_ROOT}/data/modified_libero_rlds" \
      --model-template "${model_template_dir}" \
      --visual-tokenizer "${visual_tokenizer_dir}" \
      --output-dir "${output_dir}" \
      --max-steps 150000 \
      --segment-length 8 \
      --context-length 1 \
      --tokenizer-micro-batch-size 4 \
      --batch-size-per-device 1 \
      --grad-accum "${GRAD_ACCUM}" \
      --learning-rate 5e-5 \
      --warmup-ratio 0.0 \
      --weight-decay 0.0 \
      --adam-beta1 0.9 \
      --adam-beta2 0.999 \
      --adam-epsilon 1e-8 \
      --max-grad-norm 1.0 \
      --optim adamw_torch \
      --save-steps 5000 \
      --logging-steps 10 \
      --save-total-limit 3 \
      "${extra_args[@]}"
  ) 2>&1 | tee "${task_log}"

  echo "===== Finished LIBERO world-model training: ${task_name} ====="
done
