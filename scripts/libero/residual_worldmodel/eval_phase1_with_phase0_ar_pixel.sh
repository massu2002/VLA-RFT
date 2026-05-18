#!/usr/bin/env bash
# Evaluate/convert Phase0 AR-Pixel WM into Phase1-compatible result schema.
#
# Fast path:
#   PHASE0_RESULTS=results/phase0/20260430_031736_spatial_all \
#     bash scripts/libero/residual_worldmodel/eval_phase1_with_phase0_ar_pixel.sh
#
# Native eval path uses the existing Phase0 AR-Pixel evaluator, then converts:
#   RUN_PHASE0_EVAL=1 PHASE0_AR_PIXEL_CKPT=... PHASE0_TOKENIZER_CKPT=... ...

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

TASK_SUITE="${TASK_SUITE:-spatial}"
PHASE0_RESULTS="${PHASE0_RESULTS:-}"
PHASE0_AR_PIXEL_CKPT="${PHASE0_AR_PIXEL_CKPT:-${PHASE0_AR_PIXEL_CONFIG:-}}"
PHASE0_TOKENIZER_CKPT="${PHASE0_TOKENIZER_CKPT:-${REPO_ROOT}/checkpoints/libero/WorldModel/Tokenizer}"
DATA_ROOT="${DATA_ROOT:-${LIBERO_DATA_ROOT:-${LOCALDATA_ROOT:-/localdata}/modified_libero_rlds}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/phase1/residual_worldmodel/phase0_ar_pixel_converted}"
RUN_PHASE0_EVAL="${RUN_PHASE0_EVAL:-0}"
NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-50}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
TASK_INDICES="${TASK_INDICES:-}"
VENV_NAME="${VENV_NAME:-.venv5090_eval}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${RUN_PHASE0_EVAL}" == "1" ]]; then
  if [[ -z "${PHASE0_AR_PIXEL_CKPT}" || ! -d "${PHASE0_AR_PIXEL_CKPT}" ]]; then
    echo "PHASE0_AR_PIXEL_CKPT must point to a Phase0 AR-Pixel model dir when RUN_PHASE0_EVAL=1" >&2
    exit 2
  fi
  if [[ ! -d "${PHASE0_TOKENIZER_CKPT}" ]]; then
    echo "PHASE0_TOKENIZER_CKPT not found: ${PHASE0_TOKENIZER_CKPT}" >&2
    exit 2
  fi
  PHASE0_RESULTS="${OUTPUT_DIR}/native_phase0_eval"
  mkdir -p "${PHASE0_RESULTS}/worldmodel"
  source "${REPO_ROOT}/${VENV_NAME}/bin/activate"
  export PYTHONPATH="${REPO_ROOT}/worldmodel/_compat:${REPO_ROOT}/train/verl:${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export TOKENIZERS_PARALLELISM=false
  export TF_CPP_MIN_LOG_LEVEL=3
  python -m worldmodel.libero.visualize \
    --task-suite "${TASK_SUITE}" \
    ${TASK_INDICES:+--task-indices "${TASK_INDICES}"} \
    --data-root "${DATA_ROOT}" \
    --base-model-root "${REPO_ROOT}/checkpoints/libero/WorldModel" \
    --trained-model-dir "${PHASE0_AR_PIXEL_CKPT}" \
    --visual-tokenizer "${PHASE0_TOKENIZER_CKPT}" \
    --output-dir "${PHASE0_RESULTS}/worldmodel" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --num-eval-windows "${NUM_EVAL_WINDOWS}" \
    --eval-horizon 7 \
    --heldout-ratio 0.2 \
    --split-mode fallback_all \
    --display-frames 8 \
    --eval-batch-size 4 \
    --save-casebook-count 3 \
    --chunk-future-length 7 \
    --compare-base \
    --run-action-sensitivity \
    --enable-roi-metrics \
    --enable-rank-logging \
    2>&1 | tee "${OUTPUT_DIR}/phase0_ar_pixel_native_eval.log"
fi

if [[ -z "${PHASE0_RESULTS}" ]]; then
  PHASE0_RESULTS="${REPO_ROOT}/results/phase0/20260430_031736_spatial_all"
fi
if [[ ! -d "${PHASE0_RESULTS}" ]]; then
  echo "PHASE0_RESULTS not found: ${PHASE0_RESULTS}" >&2
  exit 2
fi

"${REPO_ROOT}/.venv/bin/python" analysis/worldmodel/convert_to_phase1_eval.py \
  --phase0-results "${PHASE0_RESULTS}" \
  --out-dir "${OUTPUT_DIR}" \
  --task-suite "${TASK_SUITE}" \
  --checkpoint-path "${PHASE0_AR_PIXEL_CKPT}" \
  --tokenizer-path "${PHASE0_TOKENIZER_CKPT}" \
  --phase0-compatible

echo "phase0_ar_pixel_converted: ${OUTPUT_DIR}"
