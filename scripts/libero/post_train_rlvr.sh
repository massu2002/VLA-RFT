#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

source "${REPO_ROOT}/.venv/bin/activate"

# =========================
# 固定設定値はここで管理
# =========================
export DATE=$(date +%Y%m%d)
export POST_EXP_NAME="vla_adapter_w_fm_head"
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH="${REPO_ROOT}/train/verl:${PYTHONPATH:-}"
export N_GPUS_PER_NODE=8

WORLD_MODEL_VERSION="20260416_worldmodel_scratch"
WORLD_MODEL_PATH_TEMPLATE="checkpoints/libero/WorldModel/{task}/${WORLD_MODEL_VERSION}"

# =========================
# タスク名は引数で受け取る
# 例:
#   bash scripts/libero/post_train_rlvr.sh spatial
#   bash scripts/libero/post_train_rlvr.sh spatial object goal 10
# =========================
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <task1> [task2 ...]"
  echo "Example: $0 spatial"
  echo "Example: $0 spatial object goal 10"
  exit 1
fi

TASKS=("$@")

mkdir -p "${REPO_ROOT}/logs/libero/RFT/${DATE}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
echo "WORLD_MODEL_PATH_TEMPLATE=${WORLD_MODEL_PATH_TEMPLATE}"
echo "LIBERO tasks: ${TASKS[*]}"
python -c "import verl; print(verl.__file__)"

for task_name in "${TASKS[@]}"; do
  export LIBERO_TASK_NAME="${task_name}"
  export TENSORBOARD_DIR="${REPO_ROOT}/logs/libero/RFT/${DATE}/${LIBERO_TASK_NAME}_${POST_EXP_NAME}"
  task_log="${REPO_ROOT}/logs/libero/RFT/${DATE}/${LIBERO_TASK_NAME}_output.log"

  export WORLD_MODEL_PATH="${WORLD_MODEL_PATH_TEMPLATE//\{task\}/${LIBERO_TASK_NAME}}"

  echo "===== Starting LIBERO task: ${LIBERO_TASK_NAME} ====="
  echo "Log file: ${task_log}"
  echo "WORLD_MODEL_PATH=${WORLD_MODEL_PATH}"
  echo "N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"

  (
    cd "${REPO_ROOT}"
    bash train/verl/examples/grpo_trainer/run_vla_rft.sh
  ) 2>&1 | tee "${task_log}"

  echo "===== Finished LIBERO task: ${LIBERO_TASK_NAME} ====="
done