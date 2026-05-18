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
#   LR, WORLD_MODEL_BATCH_SIZE, BATCH_SIZE, GRAD_ACCUM, TRAIN_HORIZON,
#   SEGMENT_LENGTH, MAX_STEPS, PRECISION
#   LAMBDA_RESIDUAL, LAMBDA_IMAGE, LAMBDA_DYNAMIC, LAMBDA_GRIPPER, LAMBDA_STATIC
#   DYNAMIC_THRESHOLD, ROI_CROP_SIZE
#   PIXEL_OUTPUT_ACTIVATION, RESIDUAL_OUTPUT_ACTIVATION, RESIDUAL_OUTPUT_SCALE
#   ACTION_CONDITIONING_MODE, CONTEXT_ANCHOR_MODE
#   USE_SPATIAL_ACTION_CONDITIONING, USE_RESIDUAL_WRITE_MASK, WRITE_MASK_TEMPERATURE
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

MATCH_PHASE0_HPARAMS="${MATCH_PHASE0_HPARAMS:-1}"

# Phase0 AR-Pixel WM defaults:
#   max_steps=150000, global_batch=16, per-device batch=1, lr=5e-5,
#   warmup=0, scheduler=constant, precision=bf16, future horizon=8.
# PixelResidualWorldModel internally uses pixels[:,1] as current and
# pixels[:,2:] as target future, so horizon=8 requires SEGMENT_LENGTH=10.
case "${MATCH_PHASE0_HPARAMS}" in
  0|false|FALSE|False|no|NO)
    _DEFAULT_LR="1e-4"
    _DEFAULT_WORLD_BATCH=""
    _DEFAULT_BATCH_SIZE="4"
    _DEFAULT_TRAIN_HORIZON="6"
    _DEFAULT_PRECISION="auto"
    _DEFAULT_WARMUP_RATIO="0.02"
    _DEFAULT_LR_SCHEDULER="cosine"
    _DEFAULT_ACTION_CONDITIONING="continuous_mlp"
    _DEFAULT_CONTEXT_ANCHOR="mean_pool"
    ;;
  *)
    _DEFAULT_LR="5e-5"
    _DEFAULT_WORLD_BATCH="16"
    _DEFAULT_BATCH_SIZE="1"
    _DEFAULT_TRAIN_HORIZON="8"
    _DEFAULT_PRECISION="bf16"
    _DEFAULT_WARMUP_RATIO="0.0"
    _DEFAULT_LR_SCHEDULER="constant"
    _DEFAULT_ACTION_CONDITIONING="discrete_tokens"
    _DEFAULT_CONTEXT_ANCHOR="spatial_tokens"
    ;;
esac
LR="${LR:-${_DEFAULT_LR}}"
WORLD_MODEL_BATCH_SIZE="${WORLD_MODEL_BATCH_SIZE:-${_DEFAULT_WORLD_BATCH}}"
BATCH_SIZE="${BATCH_SIZE:-${_DEFAULT_BATCH_SIZE}}"
TRAIN_HORIZON="${TRAIN_HORIZON:-${_DEFAULT_TRAIN_HORIZON}}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-$(( TRAIN_HORIZON + 2 ))}"   # model effective horizon = SEGMENT_LENGTH - 2
ACTION_HORIZON=$(( SEGMENT_LENGTH - 2 ))

PRECISION="${PRECISION:-${_DEFAULT_PRECISION}}"
# num_workers=0 avoids TensorFlow/fork issues in DDP.
# For single-GPU, set NUM_WORKERS=2 to overlap data loading with GPU compute.
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
WARMUP_RATIO="${WARMUP_RATIO:-${_DEFAULT_WARMUP_RATIO}}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LR_SCHEDULER="${LR_SCHEDULER:-${_DEFAULT_LR_SCHEDULER}}"

# Loss λ weights
LAMBDA_RESIDUAL="${LAMBDA_RESIDUAL:-1.0}"
LAMBDA_IMAGE="${LAMBDA_IMAGE:-0.25}"
LAMBDA_DYNAMIC="${LAMBDA_DYNAMIC:-2.0}"
LAMBDA_GRIPPER="${LAMBDA_GRIPPER:-2.0}"
LAMBDA_STATIC="${LAMBDA_STATIC:-0.5}"
LAMBDA_WRITE_MASK="${LAMBDA_WRITE_MASK:-0.2}"
DYNAMIC_THRESHOLD="${DYNAMIC_THRESHOLD:-0.05}"
DYNAMIC_DILATE_KERNEL="${DYNAMIC_DILATE_KERNEL:-7}"
ROI_CROP_SIZE="${ROI_CROP_SIZE:-64}"

# Output activation. New Phase 1 training defaults avoid hard clamp saturation
# and make residual models start from a stable current-frame anchor.
PIXEL_OUTPUT_ACTIVATION="${PIXEL_OUTPUT_ACTIVATION:-sigmoid}"
RESIDUAL_OUTPUT_ACTIVATION="${RESIDUAL_OUTPUT_ACTIVATION:-tanh}"
RESIDUAL_OUTPUT_SCALE="${RESIDUAL_OUTPUT_SCALE:-1.0}"
USE_SPATIAL_ACTION_CONDITIONING="${USE_SPATIAL_ACTION_CONDITIONING:-1}"
USE_RESIDUAL_WRITE_MASK="${USE_RESIDUAL_WRITE_MASK:-1}"
WRITE_MASK_TEMPERATURE="${WRITE_MASK_TEMPERATURE:-1.0}"
WRITE_MASK_BIAS_INIT="${WRITE_MASK_BIAS_INIT:--2.0}"

# Architecture
ENCODER_CHANNELS="${ENCODER_CHANNELS:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
N_HEADS="${N_HEADS:-4}"
N_PRED_LAYERS="${N_PRED_LAYERS:-4}"
N_SPATIAL_ACTION_LAYERS="${N_SPATIAL_ACTION_LAYERS:-2}"
FFN_DIM="${FFN_DIM:-1024}"
DROPOUT="${DROPOUT:-0.0}"
ACTION_EMB_DIM="${ACTION_EMB_DIM:-128}"
ACTION_BINS="${ACTION_BINS:-256}"
ACTION_CONDITIONING_MODE="${ACTION_CONDITIONING_MODE:-${_DEFAULT_ACTION_CONDITIONING}}"
CONTEXT_ANCHOR_MODE="${CONTEXT_ANCHOR_MODE:-${_DEFAULT_CONTEXT_ANCHOR}}"

# Paths
DATA_ROOT="${DATA_ROOT:-$(default_libero_data_root)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/libero/PixelResidualWM}"

# Default MAX_STEPS by task suite (overridable)
_default_max_steps() {
  case "${TASK_SUITE}" in
    spatial|object|goal|10|long) echo "150000" ;;
    *)                           echo "150000" ;;
  esac
}
MAX_STEPS="${MAX_STEPS:-$(_default_max_steps)}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
setup_env

# Allow explicit override via NPROC env var; otherwise auto-detect.
NPROC="${NPROC:-$(detect_gpu_count)}"

if [ -z "${GRAD_ACCUM:-}" ]; then
  if [ -n "${WORLD_MODEL_BATCH_SIZE}" ]; then
    TOTAL_PER_STEP=$(( BATCH_SIZE * NPROC ))
    if (( WORLD_MODEL_BATCH_SIZE % TOTAL_PER_STEP != 0 )); then
      echo "[train_pxr] WORLD_MODEL_BATCH_SIZE=${WORLD_MODEL_BATCH_SIZE} must be divisible by BATCH_SIZE*NPROC=${TOTAL_PER_STEP}." >&2
      echo "[train_pxr] Override BATCH_SIZE, WORLD_MODEL_BATCH_SIZE, or GRAD_ACCUM explicitly." >&2
      exit 1
    fi
    GRAD_ACCUM=$(( WORLD_MODEL_BATCH_SIZE / TOTAL_PER_STEP ))
  else
    GRAD_ACCUM=4
  fi
fi
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
  "PHASE0_MATCH"   "${MATCH_PHASE0_HPARAMS}" \
  "BATCH(eff.)"    "per-dev=${BATCH_SIZE}  gpus=${NPROC}  accum=${GRAD_ACCUM}  eff=${GLOBAL_BATCH_SIZE}" \
  "TRAIN_HORIZON"  "${TRAIN_HORIZON}" \
  "SEGMENT_LEN"    "${SEGMENT_LENGTH}  (effective_action_horizon=${ACTION_HORIZON})" \
  "MAX_STEPS"      "${MAX_STEPS}" \
  "LR_SCHEDULER"   "${LR_SCHEDULER}" \
  "WARMUP_RATIO"   "${WARMUP_RATIO}" \
  "PRECISION"      "${PRECISION}" \
  "λ residual"     "${LAMBDA_RESIDUAL}" \
  "λ image"        "${LAMBDA_IMAGE}" \
  "λ dynamic"      "${LAMBDA_DYNAMIC}" \
  "λ gripper"      "${LAMBDA_GRIPPER}" \
  "λ static"       "${LAMBDA_STATIC}" \
  "λ write_mask"   "${LAMBDA_WRITE_MASK}" \
  "dyn_threshold"  "${DYNAMIC_THRESHOLD}" \
  "roi_crop_size"  "${ROI_CROP_SIZE}" \
  "pixel_activation" "${PIXEL_OUTPUT_ACTIVATION}" \
  "residual_activation" "${RESIDUAL_OUTPUT_ACTIVATION}" \
  "residual_scale" "${RESIDUAL_OUTPUT_SCALE}" \
  "action_cond"    "${ACTION_CONDITIONING_MODE}" \
  "context_anchor" "${CONTEXT_ANCHOR_MODE}" \
  "spatial_action" "${USE_SPATIAL_ACTION_CONDITIONING}" \
  "write_mask"     "${USE_RESIDUAL_WRITE_MASK}" \
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
  "match_phase0_hparams=${MATCH_PHASE0_HPARAMS}" \
  "batch_size=${BATCH_SIZE}" \
  "grad_accum=${GRAD_ACCUM}" \
  "world_model_batch_size=${WORLD_MODEL_BATCH_SIZE}" \
  "global_batch_size=${GLOBAL_BATCH_SIZE}" \
  "train_horizon=${TRAIN_HORIZON}" \
  "segment_length=${SEGMENT_LENGTH}" \
  "action_horizon=${ACTION_HORIZON}" \
  "max_steps=${MAX_STEPS}" \
  "precision=${PRECISION}" \
  "lambda_residual=${LAMBDA_RESIDUAL}" \
  "lambda_image=${LAMBDA_IMAGE}" \
  "lambda_dynamic=${LAMBDA_DYNAMIC}" \
  "lambda_gripper=${LAMBDA_GRIPPER}" \
  "lambda_static=${LAMBDA_STATIC}" \
  "lambda_write_mask=${LAMBDA_WRITE_MASK}" \
  "dynamic_threshold=${DYNAMIC_THRESHOLD}" \
  "roi_crop_size=${ROI_CROP_SIZE}" \
  "pixel_output_activation=${PIXEL_OUTPUT_ACTIVATION}" \
  "residual_output_activation=${RESIDUAL_OUTPUT_ACTIVATION}" \
  "residual_output_scale=${RESIDUAL_OUTPUT_SCALE}" \
  "action_bins=${ACTION_BINS}" \
  "action_conditioning_mode=${ACTION_CONDITIONING_MODE}" \
  "context_anchor_mode=${CONTEXT_ANCHOR_MODE}" \
  "use_spatial_action_conditioning=${USE_SPATIAL_ACTION_CONDITIONING}" \
  "use_residual_write_mask=${USE_RESIDUAL_WRITE_MASK}" \
  "write_mask_temperature=${WRITE_MASK_TEMPERATURE}" \
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
  --action-bins       "${ACTION_BINS}"
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
  --n-spatial-action-layers "${N_SPATIAL_ACTION_LAYERS}"
  --ffn-dim           "${FFN_DIM}"
  --dropout           "${DROPOUT}"
  --action-emb-dim    "${ACTION_EMB_DIM}"
  --action-conditioning-mode "${ACTION_CONDITIONING_MODE}"
  --context-anchor-mode "${CONTEXT_ANCHOR_MODE}"
  --pixel-output-activation "${PIXEL_OUTPUT_ACTIVATION}"
  --residual-output-activation "${RESIDUAL_OUTPUT_ACTIVATION}"
  --residual-output-scale "${RESIDUAL_OUTPUT_SCALE}"
  --write-mask-temperature "${WRITE_MASK_TEMPERATURE}"
  --write-mask-bias-init "${WRITE_MASK_BIAS_INIT}"
  # Loss
  --lambda-residual   "${LAMBDA_RESIDUAL}"
  --lambda-image      "${LAMBDA_IMAGE}"
  --lambda-dynamic    "${LAMBDA_DYNAMIC}"
  --lambda-gripper    "${LAMBDA_GRIPPER}"
  --lambda-static     "${LAMBDA_STATIC}"
  --lambda-write-mask "${LAMBDA_WRITE_MASK}"
  --dynamic-threshold "${DYNAMIC_THRESHOLD}"
  --dynamic-dilate-kernel "${DYNAMIC_DILATE_KERNEL}"
  --roi-crop-size     "${ROI_CROP_SIZE}"
)

case "${USE_SPATIAL_ACTION_CONDITIONING}" in
  0|false|FALSE|False|no|NO) TRAIN_ARGS+=(--no-spatial-action-conditioning) ;;
  *)                         TRAIN_ARGS+=(--use-spatial-action-conditioning) ;;
esac

case "${USE_RESIDUAL_WRITE_MASK}" in
  0|false|FALSE|False|no|NO) TRAIN_ARGS+=(--no-residual-write-mask) ;;
  *)                         TRAIN_ARGS+=(--use-residual-write-mask) ;;
esac

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
