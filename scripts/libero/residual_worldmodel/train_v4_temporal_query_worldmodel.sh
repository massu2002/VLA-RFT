#!/usr/bin/env bash
# train_v4_temporal_query_worldmodel.sh — Train TemporalDynamicQueryResidualWM (v4) on LIBERO data.
#
# Usage:
#   bash scripts/libero/residual_worldmodel/train_v4_temporal_query_worldmodel.sh [spatial|object|goal|10]
#   HISTORY_LENGTH=2 NUM_DYNAMIC_QUERIES=8 \
#     bash scripts/libero/residual_worldmodel/train_v4_temporal_query_worldmodel.sh spatial
#   DRY_RUN=1 bash scripts/libero/residual_worldmodel/train_v4_temporal_query_worldmodel.sh spatial
#
# Key env-var overrides (all have defaults):
#   TASK_SUITE, EXP_NAME, SEED, DEVICE
#   LR, WORLD_MODEL_BATCH_SIZE, BATCH_SIZE, GRAD_ACCUM, TRAIN_HORIZON
#   SEGMENT_LENGTH, MAX_STEPS, PRECISION
#   HISTORY_LENGTH, NUM_DYNAMIC_QUERIES
#   LAMBDA_IMAGE, LAMBDA_DYNAMIC, LAMBDA_STATIC, LAMBDA_QUERY, LAMBDA_RANK, LAMBDA_SPARSE
#   RANK_MARGIN, DYNAMIC_THRESHOLD
#   N_CONTEXT_LAYERS, N_FUSER_LAYERS, N_SCORER_LAYERS
#   USE_ACTION_FUTURE_SCORER (1=v4b, 0=v4a)
#   USE_MOTION_BIAS
#   DATA_ROOT, OUTPUT_ROOT
#
# Stage variants:
#   v4a: USE_ACTION_FUTURE_SCORER=0  LAMBDA_RANK=0.0
#   v4b: USE_ACTION_FUTURE_SCORER=1  LAMBDA_RANK=1.0  (default)
#
# Outputs under:
#   checkpoints/libero/TemporalQueryResidualWM/<suite>/<exp_name>/s<seed>/

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

# ---------------------------------------------------------------------------
# User config — all overridable via env vars
# ---------------------------------------------------------------------------
TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
EXP_NAME="${EXP_NAME:-v4b}"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

# ---------------------------------------------------------------------------
# Training hyperparameters — match Phase 0 AR-Pixel defaults
# ---------------------------------------------------------------------------
LR="${LR:-5e-5}"
WORLD_MODEL_BATCH_SIZE="${WORLD_MODEL_BATCH_SIZE:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_HORIZON="${TRAIN_HORIZON:-7}"
PRECISION="${PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"

# ---------------------------------------------------------------------------
# V4 architecture
# ---------------------------------------------------------------------------
HISTORY_LENGTH="${HISTORY_LENGTH:-2}"           # K
NUM_DYNAMIC_QUERIES="${NUM_DYNAMIC_QUERIES:-8}" # Q
ENCODER_CHANNELS="${ENCODER_CHANNELS:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
N_HEADS="${N_HEADS:-4}"
N_CONTEXT_LAYERS="${N_CONTEXT_LAYERS:-2}"
N_FUSER_LAYERS="${N_FUSER_LAYERS:-2}"
N_SCORER_LAYERS="${N_SCORER_LAYERS:-2}"
FFN_DIM="${FFN_DIM:-1024}"
DROPOUT="${DROPOUT:-0.0}"
ACTION_EMB_DIM="${ACTION_EMB_DIM:-128}"
ACTION_BINS="${ACTION_BINS:-256}"
ACTION_CONDITIONING_MODE="${ACTION_CONDITIONING_MODE:-discrete_tokens}"
USE_ACTION_FUTURE_SCORER="${USE_ACTION_FUTURE_SCORER:-1}"  # 1=v4b, 0=v4a
USE_MOTION_BIAS="${USE_MOTION_BIAS:-0}"
RESIDUAL_OUTPUT_ACTIVATION="${RESIDUAL_OUTPUT_ACTIVATION:-tanh}"
RESIDUAL_OUTPUT_SCALE="${RESIDUAL_OUTPUT_SCALE:-1.0}"
PIXEL_OUTPUT_ACTIVATION="${PIXEL_OUTPUT_ACTIVATION:-sigmoid}"

# ---------------------------------------------------------------------------
# Loss λ weights
# ---------------------------------------------------------------------------
LAMBDA_IMAGE="${LAMBDA_IMAGE:-0.1}"
LAMBDA_DYNAMIC="${LAMBDA_DYNAMIC:-1.0}"
LAMBDA_STATIC="${LAMBDA_STATIC:-0.2}"
LAMBDA_QUERY="${LAMBDA_QUERY:-0.5}"
LAMBDA_RANK="${LAMBDA_RANK:-1.0}"
LAMBDA_SPARSE="${LAMBDA_SPARSE:-0.01}"
RANK_MARGIN="${RANK_MARGIN:-0.1}"
DYNAMIC_THRESHOLD="${DYNAMIC_THRESHOLD:-0.05}"
DYNAMIC_DILATE_KERNEL="${DYNAMIC_DILATE_KERNEL:-7}"
ROI_CROP_SIZE="${ROI_CROP_SIZE:-64}"

# ---------------------------------------------------------------------------
# Segment length: v4 requires K + H + 2 frames per window
# ---------------------------------------------------------------------------
# SEGMENT_LENGTH = HISTORY_LENGTH + TRAIN_HORIZON + 2
SEGMENT_LENGTH="${SEGMENT_LENGTH:-$(( HISTORY_LENGTH + TRAIN_HORIZON + 2 ))}"
ACTION_HORIZON=$(( SEGMENT_LENGTH - HISTORY_LENGTH - 2 ))

# ---------------------------------------------------------------------------
# Max steps
# ---------------------------------------------------------------------------
_default_max_steps() {
  case "${TASK_SUITE}" in
    spatial|object|goal|10|long) echo "150000" ;;
    *)                           echo "150000" ;;
  esac
}
MAX_STEPS="${MAX_STEPS:-$(_default_max_steps)}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$(default_libero_data_root)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/libero/TemporalQueryResidualWM}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
setup_env

NPROC="${NPROC:-$(detect_gpu_count)}"

if [ -z "${GRAD_ACCUM:-}" ]; then
  if [ -n "${WORLD_MODEL_BATCH_SIZE}" ]; then
    TOTAL_PER_STEP=$(( BATCH_SIZE * NPROC ))
    if (( WORLD_MODEL_BATCH_SIZE % TOTAL_PER_STEP != 0 )); then
      echo "[train_v4] WORLD_MODEL_BATCH_SIZE=${WORLD_MODEL_BATCH_SIZE} must be divisible by BATCH_SIZE*NPROC=${TOTAL_PER_STEP}." >&2
      exit 1
    fi
    GRAD_ACCUM=$(( WORLD_MODEL_BATCH_SIZE / TOTAL_PER_STEP ))
  else
    GRAD_ACCUM=4
  fi
fi
GLOBAL_BATCH_SIZE=$(( BATCH_SIZE * NPROC * GRAD_ACCUM ))

if [ "${NUM_WORKERS}" = "0" ] && [ "${NPROC}" = "1" ]; then
  NUM_WORKERS=2
fi

# Stage label
if [ "${USE_ACTION_FUTURE_SCORER}" = "0" ] || [ "${USE_ACTION_FUTURE_SCORER}" = "false" ]; then
  STAGE="v4a"
else
  STAGE="v4b"
fi

SAVE_DIR="${OUTPUT_ROOT}/${TASK_SUITE}/${EXP_NAME}/s${SEED}"
mkdir -p "${SAVE_DIR}"

LOG_DIR="${REPO_ROOT}/logs/libero/TemporalQueryResidualWM/${TASK_SUITE}/${EXP_NAME}/s${SEED}"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/train_$(timestamp).log"

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print_run_summary \
  "TASK_SUITE"        "${TASK_SUITE}" \
  "EXP_NAME"          "${EXP_NAME}" \
  "STAGE"             "${STAGE}" \
  "HISTORY_LENGTH"    "${HISTORY_LENGTH} (K)" \
  "NUM_DYNAMIC_Q"     "${NUM_DYNAMIC_QUERIES} (Q)" \
  "SEGMENT_LEN"       "${SEGMENT_LENGTH}  (K=${HISTORY_LENGTH}  H=${ACTION_HORIZON})" \
  "LR"                "${LR}" \
  "BATCH(eff.)"       "per-dev=${BATCH_SIZE}  gpus=${NPROC}  accum=${GRAD_ACCUM}  eff=${GLOBAL_BATCH_SIZE}" \
  "TRAIN_HORIZON"     "${TRAIN_HORIZON}" \
  "MAX_STEPS"         "${MAX_STEPS}" \
  "SAVE_STEPS"        "${SAVE_STEPS}" \
  "SAVE_TOTAL_LIMIT"  "${SAVE_TOTAL_LIMIT}" \
  "LOGGING_STEPS"     "${LOGGING_STEPS}" \
  "LR_SCHEDULER"      "${LR_SCHEDULER}" \
  "PRECISION"         "${PRECISION}" \
  "λ image"           "${LAMBDA_IMAGE}" \
  "λ dynamic"         "${LAMBDA_DYNAMIC}" \
  "λ static"          "${LAMBDA_STATIC}" \
  "λ query"           "${LAMBDA_QUERY}" \
  "λ rank"            "${LAMBDA_RANK}" \
  "λ sparse"          "${LAMBDA_SPARSE}" \
  "rank_margin"       "${RANK_MARGIN}" \
  "dyn_threshold"     "${DYNAMIC_THRESHOLD}" \
  "action_cond"       "${ACTION_CONDITIONING_MODE}" \
  "use_scorer"        "${USE_ACTION_FUTURE_SCORER}" \
  "use_motion_bias"   "${USE_MOTION_BIAS}" \
  "SEED"              "${SEED}" \
  "DATA_ROOT"         "${DATA_ROOT}" \
  "SAVE_DIR"          "${SAVE_DIR}" \
  "LOGFILE"           "${LOGFILE}"

# ---------------------------------------------------------------------------
# Config dump
# ---------------------------------------------------------------------------
dump_run_config "${SAVE_DIR}" \
  "task_suite=${TASK_SUITE}" \
  "exp_name=${EXP_NAME}" \
  "stage=${STAGE}" \
  "history_length=${HISTORY_LENGTH}" \
  "num_dynamic_queries=${NUM_DYNAMIC_QUERIES}" \
  "segment_length=${SEGMENT_LENGTH}" \
  "action_horizon=${ACTION_HORIZON}" \
  "lr=${LR}" \
  "batch_size=${BATCH_SIZE}" \
  "grad_accum=${GRAD_ACCUM}" \
  "global_batch_size=${GLOBAL_BATCH_SIZE}" \
  "train_horizon=${TRAIN_HORIZON}" \
  "max_steps=${MAX_STEPS}" \
  "precision=${PRECISION}" \
  "lambda_image=${LAMBDA_IMAGE}" \
  "lambda_dynamic=${LAMBDA_DYNAMIC}" \
  "lambda_static=${LAMBDA_STATIC}" \
  "lambda_query=${LAMBDA_QUERY}" \
  "lambda_rank=${LAMBDA_RANK}" \
  "lambda_sparse=${LAMBDA_SPARSE}" \
  "rank_margin=${RANK_MARGIN}" \
  "dynamic_threshold=${DYNAMIC_THRESHOLD}" \
  "use_action_future_scorer=${USE_ACTION_FUTURE_SCORER}" \
  "use_motion_bias=${USE_MOTION_BIAS}" \
  "action_conditioning_mode=${ACTION_CONDITIONING_MODE}" \
  "seed=${SEED}" \
  "data_root=${DATA_ROOT}" \
  "save_dir=${SAVE_DIR}"

copy_script_snapshot "${SAVE_DIR}" "${BASH_SOURCE[0]}"

# ---------------------------------------------------------------------------
# Build training command
# ---------------------------------------------------------------------------
TRAIN_ARGS=(
  -m worldmodel.residual_worldmodel.train_v4_temporal_query_libero
  --task-suite               "${TASK_SUITE}"
  --data-root                "${DATA_ROOT}"
  --output-dir               "${SAVE_DIR}"
  --history-length           "${HISTORY_LENGTH}"
  --num-dynamic-queries      "${NUM_DYNAMIC_QUERIES}"
  --encoder-channels         "${ENCODER_CHANNELS}"
  --hidden-dim               "${HIDDEN_DIM}"
  --n-heads                  "${N_HEADS}"
  --n-context-layers         "${N_CONTEXT_LAYERS}"
  --n-fuser-layers           "${N_FUSER_LAYERS}"
  --n-scorer-layers          "${N_SCORER_LAYERS}"
  --ffn-dim                  "${FFN_DIM}"
  --dropout                  "${DROPOUT}"
  --action-bins              "${ACTION_BINS}"
  --action-emb-dim           "${ACTION_EMB_DIM}"
  --action-conditioning-mode "${ACTION_CONDITIONING_MODE}"
  --lambda-image             "${LAMBDA_IMAGE}"
  --lambda-dynamic           "${LAMBDA_DYNAMIC}"
  --lambda-static            "${LAMBDA_STATIC}"
  --lambda-query             "${LAMBDA_QUERY}"
  --lambda-rank              "${LAMBDA_RANK}"
  --lambda-sparse            "${LAMBDA_SPARSE}"
  --rank-margin              "${RANK_MARGIN}"
  --dynamic-threshold        "${DYNAMIC_THRESHOLD}"
  --dynamic-dilate-kernel    "${DYNAMIC_DILATE_KERNEL}"
  --roi-crop-size            "${ROI_CROP_SIZE}"
  --residual-output-activation "${RESIDUAL_OUTPUT_ACTIVATION}"
  --residual-output-scale    "${RESIDUAL_OUTPUT_SCALE}"
  --pixel-output-activation  "${PIXEL_OUTPUT_ACTIVATION}"
  --max-steps                "${MAX_STEPS}"
  --segment-length           "${SEGMENT_LENGTH}"
  --batch-size-per-device    "${BATCH_SIZE}"
  --global-batch-size        "${GLOBAL_BATCH_SIZE}"
  --learning-rate            "${LR}"
  --warmup-ratio             "${WARMUP_RATIO}"
  --weight-decay             "${WEIGHT_DECAY}"
  --max-grad-norm            "${MAX_GRAD_NORM}"
  --lr-scheduler-type        "${LR_SCHEDULER}"
  --grad-accum               "${GRAD_ACCUM}"
  --num-workers              "${NUM_WORKERS}"
  --precision                "${PRECISION}"
  --save-steps               "${SAVE_STEPS}"
  --save-total-limit         "${SAVE_TOTAL_LIMIT}"
  --logging-steps            "${LOGGING_STEPS}"
  --seed                     "${SEED}"
)

case "${USE_ACTION_FUTURE_SCORER}" in
  0|false|FALSE|False|no|NO) TRAIN_ARGS+=(--no-action-future-scorer) ;;
  *)                         TRAIN_ARGS+=(--use-action-future-scorer) ;;
esac

case "${USE_MOTION_BIAS}" in
  1|true|TRUE|True|yes|YES) TRAIN_ARGS+=(--use-motion-bias) ;;
  *)                        TRAIN_ARGS+=(--no-motion-bias) ;;
esac

if [ "${TF32:-0}" = "1" ]; then
  TRAIN_ARGS+=(--tf32)
fi

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
echo ""
echo "  Starting v4 training → ${SAVE_DIR}"
echo "  Stage: ${STAGE}  (use_action_future_scorer=${USE_ACTION_FUTURE_SCORER})"
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
echo "    bash scripts/libero/residual_worldmodel/eval_v4_temporal_query_worldmodel.sh \\"
echo "      MODEL_DIR=${SAVE_DIR} TASK_SUITE=${TASK_SUITE}"
