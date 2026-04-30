#!/usr/bin/env bash
# train_pixel_residual_worldmodel.sh — Train PixelResidualWorldModel on LIBERO data.
#
# Usage:
#   bash scripts/libero/residual_worldmodel/train_pixel_residual_worldmodel.sh [spatial|object|goal|10]
#   TARGET_MODE=pixel_residual_roi_dynamic \
#     bash scripts/libero/residual_worldmodel/train_pixel_residual_worldmodel.sh spatial
#   DRY_RUN=1 bash scripts/libero/residual_worldmodel/train_pixel_residual_worldmodel.sh spatial
#
# Key env-var overrides (all have defaults):
#   TASK_SUITE, TARGET_MODE, EXP_NAME, SEED, DEVICE
#   LR, BATCH_SIZE, GRAD_ACCUM, SEGMENT_LENGTH, MAX_STEPS, PRECISION
#   LAMBDA_RESIDUAL, LAMBDA_IMAGE, LAMBDA_DYNAMIC, LAMBDA_GRIPPER, LAMBDA_STATIC
#   DYNAMIC_THRESHOLD, ROI_CROP_SIZE
#   DATA_ROOT, OUTPUT_ROOT
#
# Outputs under:
#   checkpoints/libero/PixelResidualWM/<suite>/<target_mode>/<exp_name>/

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

# ---------------------------------------------------------------------------
# User config — all overridable via env vars
# ---------------------------------------------------------------------------
TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
# target_mode: pixel | pixel_residual | pixel_residual_roi_dynamic
TARGET_MODE="${TARGET_MODE:-pixel_residual}"
EXP_NAME="${EXP_NAME:-v1}"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-8}"   # T+1; action_horizon = SEGMENT_LENGTH - 2
ACTION_HORIZON=$(( SEGMENT_LENGTH - 2 ))

PRECISION="${PRECISION:-auto}"
# num_workers=0 avoids TensorFlow/fork issues in DDP.
# For single-GPU, set NUM_WORKERS=2 to overlap data loading with GPU compute.
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"

# Loss λ weights
LAMBDA_RESIDUAL="${LAMBDA_RESIDUAL:-1.0}"
LAMBDA_IMAGE="${LAMBDA_IMAGE:-0.25}"
LAMBDA_DYNAMIC="${LAMBDA_DYNAMIC:-2.0}"
LAMBDA_GRIPPER="${LAMBDA_GRIPPER:-2.0}"
LAMBDA_STATIC="${LAMBDA_STATIC:-0.5}"
DYNAMIC_THRESHOLD="${DYNAMIC_THRESHOLD:-0.05}"
DYNAMIC_DILATE_KERNEL="${DYNAMIC_DILATE_KERNEL:-7}"
ROI_CROP_SIZE="${ROI_CROP_SIZE:-64}"

# Architecture
ENCODER_CHANNELS="${ENCODER_CHANNELS:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
N_HEADS="${N_HEADS:-4}"
N_PRED_LAYERS="${N_PRED_LAYERS:-4}"
FFN_DIM="${FFN_DIM:-1024}"
DROPOUT="${DROPOUT:-0.0}"
ACTION_EMB_DIM="${ACTION_EMB_DIM:-128}"

# Paths
DATA_ROOT="${DATA_ROOT:-$(default_libero_data_root)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/libero/PixelResidualWM}"

# Default MAX_STEPS by task suite (overridable)
_default_max_steps() {
  case "${TASK_SUITE}" in
    spatial|object) echo "50000" ;;
    goal)           echo "40000" ;;
    10|long)        echo "80000" ;;
    *)              echo "50000" ;;
  esac
}
MAX_STEPS="${MAX_STEPS:-$(_default_max_steps)}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
setup_env

# Allow explicit override via NPROC env var; otherwise auto-detect.
NPROC="${NPROC:-$(detect_gpu_count)}"
GLOBAL_BATCH_SIZE=$(( BATCH_SIZE * NPROC * GRAD_ACCUM ))

if (( NPROC > 1 )); then
  echo "[train_pxr] GPUs detected: ${NPROC} → using torchrun --nproc_per_node=${NPROC}"
else
  echo "[train_pxr] GPUs detected: ${NPROC} → using single-process python"
fi

# num_workers: keep 0 for DDP (each rank does its own TF loading).
# Override to 2 for single-GPU to overlap data loading and GPU compute.
if [ "${NUM_WORKERS}" = "0" ] && [ "${NPROC}" = "1" ]; then
  NUM_WORKERS=2
fi

SAVE_DIR="${OUTPUT_ROOT}/${TASK_SUITE}/${TARGET_MODE}/${EXP_NAME}/s${SEED}"
mkdir -p "${SAVE_DIR}"

LOG_DIR="${REPO_ROOT}/logs/libero/PixelResidualWM/${TASK_SUITE}/${TARGET_MODE}/${EXP_NAME}/s${SEED}"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/train_$(timestamp).log"

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print_run_summary \
  "TASK_SUITE"     "${TASK_SUITE}" \
  "TARGET_MODE"    "${TARGET_MODE}" \
  "EXP_NAME"       "${EXP_NAME}" \
  "LR"             "${LR}" \
  "BATCH(eff.)"    "per-dev=${BATCH_SIZE}  gpus=${NPROC}  accum=${GRAD_ACCUM}  eff=${GLOBAL_BATCH_SIZE}" \
  "SEGMENT_LEN"    "${SEGMENT_LENGTH}  (action_horizon=${ACTION_HORIZON})" \
  "MAX_STEPS"      "${MAX_STEPS}" \
  "PRECISION"      "${PRECISION}" \
  "λ residual"     "${LAMBDA_RESIDUAL}" \
  "λ image"        "${LAMBDA_IMAGE}" \
  "λ dynamic"      "${LAMBDA_DYNAMIC}" \
  "λ gripper"      "${LAMBDA_GRIPPER}" \
  "λ static"       "${LAMBDA_STATIC}" \
  "dyn_threshold"  "${DYNAMIC_THRESHOLD}" \
  "roi_crop_size"  "${ROI_CROP_SIZE}" \
  "SEED"           "${SEED}" \
  "DATA_ROOT"      "${DATA_ROOT}" \
  "SAVE_DIR"       "${SAVE_DIR}" \
  "LOGFILE"        "${LOGFILE}"

# ---------------------------------------------------------------------------
# Config dump
# ---------------------------------------------------------------------------
dump_run_config "${SAVE_DIR}" \
  "task_suite=${TASK_SUITE}" \
  "target_mode=${TARGET_MODE}" \
  "exp_name=${EXP_NAME}" \
  "lr=${LR}" \
  "batch_size=${BATCH_SIZE}" \
  "grad_accum=${GRAD_ACCUM}" \
  "global_batch_size=${GLOBAL_BATCH_SIZE}" \
  "segment_length=${SEGMENT_LENGTH}" \
  "action_horizon=${ACTION_HORIZON}" \
  "max_steps=${MAX_STEPS}" \
  "precision=${PRECISION}" \
  "lambda_residual=${LAMBDA_RESIDUAL}" \
  "lambda_image=${LAMBDA_IMAGE}" \
  "lambda_dynamic=${LAMBDA_DYNAMIC}" \
  "lambda_gripper=${LAMBDA_GRIPPER}" \
  "lambda_static=${LAMBDA_STATIC}" \
  "dynamic_threshold=${DYNAMIC_THRESHOLD}" \
  "roi_crop_size=${ROI_CROP_SIZE}" \
  "seed=${SEED}" \
  "data_root=${DATA_ROOT}" \
  "save_dir=${SAVE_DIR}"

copy_script_snapshot "${SAVE_DIR}" "${BASH_SOURCE[0]}"

# ---------------------------------------------------------------------------
# Build training command
# ---------------------------------------------------------------------------
TRAIN_ARGS=(
  -m worldmodel.residual_worldmodel.train_pixel_residual_libero
  --task-suite        "${TASK_SUITE}"
  --data-root         "${DATA_ROOT}"
  --output-dir        "${SAVE_DIR}"
  --target-mode       "${TARGET_MODE}"
  --max-steps         "${MAX_STEPS}"
  --segment-length    "${SEGMENT_LENGTH}"
  --batch-size-per-device "${BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --learning-rate     "${LR}"
  --warmup-ratio      "${WARMUP_RATIO}"
  --weight-decay      "${WEIGHT_DECAY}"
  --max-grad-norm     "${MAX_GRAD_NORM}"
  --lr-scheduler-type "${LR_SCHEDULER}"
  --grad-accum        "${GRAD_ACCUM}"
  --num-workers       "${NUM_WORKERS}"
  --precision         "${PRECISION}"
  --save-steps        "${SAVE_STEPS}"
  --save-total-limit  "${SAVE_TOTAL_LIMIT}"
  --logging-steps     10
  --seed              "${SEED}"
  # Architecture
  --encoder-channels  "${ENCODER_CHANNELS}"
  --hidden-dim        "${HIDDEN_DIM}"
  --n-heads           "${N_HEADS}"
  --n-pred-layers     "${N_PRED_LAYERS}"
  --ffn-dim           "${FFN_DIM}"
  --dropout           "${DROPOUT}"
  --action-emb-dim    "${ACTION_EMB_DIM}"
  # Loss
  --lambda-residual   "${LAMBDA_RESIDUAL}"
  --lambda-image      "${LAMBDA_IMAGE}"
  --lambda-dynamic    "${LAMBDA_DYNAMIC}"
  --lambda-gripper    "${LAMBDA_GRIPPER}"
  --lambda-static     "${LAMBDA_STATIC}"
  --dynamic-threshold "${DYNAMIC_THRESHOLD}"
  --dynamic-dilate-kernel "${DYNAMIC_DILATE_KERNEL}"
  --roi-crop-size     "${ROI_CROP_SIZE}"
)

# TF32 opt-in for Ampere+
if [ "${TF32:-0}" = "1" ]; then
  TRAIN_ARGS+=(--tf32)
fi

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
echo ""
echo "  Starting training → ${SAVE_DIR}"
echo "  Log → ${LOGFILE}"
echo ""

export TF_CPP_MIN_LOG_LEVEL=3

run_train_command "${NPROC}" "${SAVE_DIR}" "${LOGFILE}" "${TRAIN_ARGS[@]}"

echo ""
echo "  Training complete."
echo "    checkpoint : ${SAVE_DIR}"
echo "    log        : ${LOGFILE}"
echo ""
echo "  To evaluate:"
echo "    bash scripts/libero/residual_worldmodel/eval_pixel_residual_worldmodel.sh \\"
echo "      MODEL_DIR=${SAVE_DIR} TASK_SUITE=${TASK_SUITE}"
