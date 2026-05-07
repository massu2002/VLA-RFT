#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/worldmodel/_compat:${REPO_ROOT}/train/verl:${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM="false"

DATE="${DATE:-$(date +%Y%m%d)}"
EXP_NAME="${EXP_NAME:-base_worldmodel_viz}"
DISPLAY_FRAMES="${DISPLAY_FRAMES:-12}"
CHUNK_FUTURE_LENGTH="${CHUNK_FUTURE_LENGTH:-8}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
DEVICE="${DEVICE:-auto}"

# Evaluation protocol extensions (safe defaults: disabled)
ENABLE_ROI_METRICS="${ENABLE_ROI_METRICS:-0}"
ENABLE_RANK_LOGGING="${ENABLE_RANK_LOGGING:-0}"
ENABLE_FAILURE_TEMPLATE_EXPORT="${ENABLE_FAILURE_TEMPLATE_EXPORT:-0}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-configs/libero/eval_protocol_v1.yaml}"

if [ -n "${LIBERO_TASKS:-}" ]; then
  read -r -a TASKS <<< "${LIBERO_TASKS}"
else
  TASKS=(spatial object goal 10)
fi

OUTPUT_ROOT="${REPO_ROOT}/worldmodel/visualizations/libero_base/${DATE}_${EXP_NAME}"
mkdir -p "${OUTPUT_ROOT}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "LIBERO tasks: ${TASKS[*]}"
echo "Output root: ${OUTPUT_ROOT}"

is_true() {
  case "${1}" in
    1|true|TRUE|True|yes|YES|Yes|y|Y|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

for task_suite in "${TASKS[@]}"; do
  suite_out_dir="${OUTPUT_ROOT}/${task_suite}"
  mkdir -p "${suite_out_dir}"

  echo "===== Visualizing base LIBERO suite: ${task_suite} ====="

  extra_args=()
  if is_true "${ENABLE_ROI_METRICS}"; then
    extra_args+=(--enable-roi-metrics)
  fi
  if is_true "${ENABLE_ROI_METRICS}"; then
    extra_args+=(--debug-roi)
  fi

  for task_index in $(seq 0 9); do
    python -m worldmodel.libero.visualize_base \
      --task-suite "${task_suite}" \
      --task-index "${task_index}" \
      --data-root "${DATA_ROOT:-${LIBERO_DATA_ROOT:-${LOCALDATA_ROOT:-/localdata}/modified_libero_rlds}}" \
      --base-model-root "${REPO_ROOT}/checkpoints/libero/WorldModel" \
      --visual-tokenizer "${REPO_ROOT}/checkpoints/libero/WorldModel/Tokenizer" \
      --output-dir "${suite_out_dir}" \
      --episode-index "${EPISODE_INDEX}" \
      --chunk-future-length "${CHUNK_FUTURE_LENGTH}" \
      --display-frames "${DISPLAY_FRAMES}" \
      --device "${DEVICE}" \
      "${extra_args[@]}"
  done

  echo "===== Finished base LIBERO suite: ${task_suite} ====="

  # Failure taxonomy template export (Task D)
  if is_true "${ENABLE_FAILURE_TEMPLATE_EXPORT}"; then
    python -m worldmodel.libero.export_failure_taxonomy_template \
      --sweep-dir "${suite_out_dir}" \
      --output    "${suite_out_dir}/taxonomy_template__${task_suite}.csv" || true
  fi
done
