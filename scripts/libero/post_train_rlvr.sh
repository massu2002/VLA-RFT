#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

source "${REPO_ROOT}/.venv/bin/activate"

export DATE="${DATE:-$(date +%Y%m%d)}"
export POST_EXP_NAME="${POST_EXP_NAME:-vla_adapter_w_fm_head}"
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH="${REPO_ROOT}/train/verl:${PYTHONPATH:-}"

if [ -n "${LIBERO_TASKS:-}" ]; then
  read -r -a TASKS <<< "${LIBERO_TASKS}"
else
  TASKS=(spatial object goal 10)
fi

mkdir -p "${REPO_ROOT}/logs/libero/RFT/${DATE}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "LIBERO tasks: ${TASKS[*]}"
python -c "import verl; print(verl.__file__)"

for task_name in "${TASKS[@]}"; do
  export LIBERO_TASK_NAME="${task_name}"
  export TENSORBOARD_DIR="${REPO_ROOT}/logs/libero/RFT/${DATE}/${LIBERO_TASK_NAME}_${POST_EXP_NAME}"
  task_log="${REPO_ROOT}/logs/libero/RFT/${DATE}/${LIBERO_TASK_NAME}_output.log"

  echo "===== Starting LIBERO task: ${LIBERO_TASK_NAME} ====="
  echo "Log file: ${task_log}"

  (
    cd "${REPO_ROOT}"
    bash train/verl/examples/grpo_trainer/run_vla_rft.sh
  ) 2>&1 | tee "${task_log}"

  echo "===== Finished LIBERO task: ${LIBERO_TASK_NAME} ====="
done
