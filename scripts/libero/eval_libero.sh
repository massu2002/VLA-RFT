#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

source "${REPO_ROOT}/.venv5090_eval/bin/activate"
export PYTHONPATH="${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
export HF_ENDPOINT="https://hf-mirror.com"

# ==============================
# ユーザ設定
# ==============================
# POST_EXP_NAME="BASE"
POST_EXP_NAME="RFT_400"

CUDA_VISIBLE_DEVICES="0"
LIBERO_TASKS=(goal 10)
# LIBERO_TASKS=(spatial object)

# タスクごとに別のモデルディレクトリを使う場合はここを編集してください。
declare -A BASE_MODEL_DIRS=(
  [spatial]="${REPO_ROOT}/checkpoints/libero/Base/spatial"
  [object]="${REPO_ROOT}/checkpoints/libero/Base/object"
  [goal]="${REPO_ROOT}/checkpoints/libero/Base/goal"
  [10]="${REPO_ROOT}/checkpoints/libero/Base/10"
)

declare -A ACTOR_MODEL_DIRS=(
  [spatial]="${REPO_ROOT}/checkpoints/libero/RFT/spatial/20260404_vla_adapter_w_fm_head/global_step_400/actor"
  [object]="${REPO_ROOT}/checkpoints/libero/RFT/object/20260404_vla_adapter_w_fm_head/global_step_400/actor"
  [goal]="${REPO_ROOT}/checkpoints/libero/RFT/goal/20260404_vla_adapter_w_fm_head/global_step_400/actor"
  [10]="${REPO_ROOT}/checkpoints/libero/RFT/10/20260404_vla_adapter_w_fm_head/global_step_400/actor"
)

current_time=$(date "+%Y-%m-%d_%H-%M-%S")
echo "Current Time: ${current_time}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "LIBERO tasks: ${LIBERO_TASKS[*]}"

resolve_model_dir() {
  local array_name="$1"
  local task_name="$2"
  local -n model_dirs="${array_name}"

  if [ -n "${model_dirs[$task_name]:-}" ]; then
    printf '%s\n' "${model_dirs[$task_name]}"
    return
  fi

  echo "Please set ${array_name} in scripts/libero/eval_libero.sh" >&2
  exit 1
}

for task_name in "${LIBERO_TASKS[@]}"; do
  LIBERO_TASK_NAME="${task_name}"
  PRETRAINED_CHECKPOINT=$(resolve_model_dir BASE_MODEL_DIRS "${LIBERO_TASK_NAME}")
  ACTOR_PATH=""
  if [[ "${POST_EXP_NAME}" == *RFT* ]]; then
    ACTOR_PATH=$(resolve_model_dir ACTOR_MODEL_DIRS "${LIBERO_TASK_NAME}")
  fi
  LOG_DIR="${REPO_ROOT}/logs/libero/eval/${POST_EXP_NAME}/${LIBERO_TASK_NAME}"
  TASK_LOG="${LOG_DIR}/eval_output.log"

  mkdir -p "${LOG_DIR}"

  if [ ! -d "${PRETRAINED_CHECKPOINT}" ]; then
    echo "Base model directory not found for task ${LIBERO_TASK_NAME}: ${PRETRAINED_CHECKPOINT}" >&2
    exit 1
  fi
  if [ -n "${ACTOR_PATH}" ] && [ ! -d "${ACTOR_PATH}" ]; then
    echo "Actor checkpoint directory not found for task ${LIBERO_TASK_NAME}: ${ACTOR_PATH}" >&2
    exit 1
  fi

  echo "===== Starting LIBERO eval: ${LIBERO_TASK_NAME} ====="
  echo "Base model directory: ${PRETRAINED_CHECKPOINT}"
  if [ -n "${ACTOR_PATH}" ]; then
    echo "Actor checkpoint directory: ${ACTOR_PATH}"
  fi
  echo "Log directory: ${LOG_DIR}"

  ACTOR_ARGS=()
  if [ -n "${ACTOR_PATH}" ]; then
    ACTOR_ARGS=(--actor_path "${ACTOR_PATH}")
  fi

  (
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python train/verl/vla-adapter/openvla-oft/experiments/robot/libero/run_libero_eval.py \
      --use_l1_regression False \
      --use_diffusion False \
      --use_flow_matching True \
      --use_proprio True \
      --use_film False \
      --num_images_in_input 1 \
      --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
      "${ACTOR_ARGS[@]}" \
      --task_suite_name "libero_${LIBERO_TASK_NAME}" \
      --save_version v1 \
      --use_minivla True \
      --run_id_note "${POST_EXP_NAME}" \
      --local_log_dir "${LOG_DIR}"
  ) 2>&1 | tee "${TASK_LOG}"

  echo "===== Finished LIBERO eval: ${LIBERO_TASK_NAME} ====="
done
