#!/usr/bin/env bash
# Evaluate Phase0 AR-Pixel WM on a Phase1 window_manifest.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

TASK_SUITE="${TASK_SUITE:-spatial}"
PHASE0_AR_PIXEL_CKPT="${PHASE0_AR_PIXEL_CKPT:?'PHASE0_AR_PIXEL_CKPT is required'}"
PHASE0_AR_PIXEL_CONFIG="${PHASE0_AR_PIXEL_CONFIG:-}"
PHASE0_TOKENIZER_CKPT="${PHASE0_TOKENIZER_CKPT:?'PHASE0_TOKENIZER_CKPT is required'}"
WINDOW_MANIFEST="${WINDOW_MANIFEST:?'WINDOW_MANIFEST is required'}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/phase1/residual_worldmodel/phase0_ar_pixel_direct_eval}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-$(default_libero_data_root)}"
EVAL_HORIZON="${EVAL_HORIZON:-7}"
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-2}"
DEVICE="${DEVICE:-auto}"
VENV_NAME="${VENV_NAME:-.venv5090_eval}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUTPUT_DIR}"
setup_env
export PYTHONPATH="${REPO_ROOT}/worldmodel/_compat:${REPO_ROOT}/train/verl:${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"

CMD=(
  "${REPO_ROOT}/${VENV_NAME}/bin/python"
  analysis/evaluate_phase0_ar_pixel_on_manifest.py
  --phase0-ar-pixel-ckpt "${PHASE0_AR_PIXEL_CKPT}"
  --tokenizer-ckpt "${PHASE0_TOKENIZER_CKPT}"
  --window-manifest "${WINDOW_MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --task-suite "${TASK_SUITE}"
  --data-root "${LIBERO_DATA_ROOT}"
  --eval-horizon "${EVAL_HORIZON}"
  --decode-chunk-size "${DECODE_CHUNK_SIZE}"
  --device "${DEVICE}"
)

if [ -n "${PHASE0_AR_PIXEL_CONFIG}" ]; then
  CMD+=(--phase0-ar-pixel-config "${PHASE0_AR_PIXEL_CONFIG}")
fi
if [ "${SMOKE}" = "1" ]; then
  CMD+=(--smoke)
fi

printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/eval_command.txt"
printf '\n' >> "${OUTPUT_DIR}/eval_command.txt"

if [ "${DRY_RUN}" = "1" ]; then
  cat "${OUTPUT_DIR}/eval_command.txt"
  exit 0
fi

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"
echo "phase0_ar_pixel_direct_eval: ${OUTPUT_DIR}"
