#!/usr/bin/env bash
# train_dynquery.sh — Train DynQueryWorldModel on LIBERO data.
#
# Usage:
#   bash scripts/libero/residual_worldmodel/train_dynquery.sh [spatial|object|goal|10]
#
#   # With overrides:
#   EXP_NAME=dq_full_rank1 NUM_DYNAMIC_QUERIES=8 \
#     bash scripts/libero/residual_worldmodel/train_dynquery.sh spatial
#
#   DRY_RUN=1 bash scripts/libero/residual_worldmodel/train_dynquery.sh spatial
#
# Key env-var overrides (all have defaults):
#   TASK_SUITE, EXP_NAME, SEED
#   LR, WORLD_MODEL_BATCH_SIZE, BATCH_SIZE, GRAD_ACCUM, TRAIN_HORIZON
#   SEGMENT_LENGTH, MAX_STEPS, PRECISION
#   HISTORY_LENGTH, NUM_DYNAMIC_QUERIES
#   -- Core 1-4 feature flags --
#   USE_ACTION_CONDITIONED_MASK   1=on (Core 1)     default: 1
#   PREDICTOR_MODE                query_wise|linear_expand (Core 2)  default: query_wise
#   USE_DYNAMIC_RESIDUAL_GATE     1=on (Core 3)     default: 1
#   LAMBDA_MASK_DYNAMIC           Core 4a loss weight  default: 0.1
#   LAMBDA_QUERY_DELTA_SPARSE     Core 4b loss weight  default: 0.001
#   -- Standard loss weights --
#   LAMBDA_IMAGE, LAMBDA_DYNAMIC, LAMBDA_STATIC, LAMBDA_QUERY
#   LAMBDA_RANK, RANK_MARGIN
#   -- Scorer / negatives --
#   USE_ACTION_FUTURE_SCORER      1=stage_b (with scorer), 0=stage_a
#   USE_MOTION_BIAS
#   NEGATIVE_TYPE, NEGATIVE_MIX   (read as env vars by DynQueryCollator)
#   -- Paths --
#   DATA_ROOT, OUTPUT_ROOT
#
# Outputs under:
#   checkpoints/libero/DynQueryWorldModel/core_sweep/<suite>/<exp_name>/s<seed>/

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

# ---------------------------------------------------------------------------
# Config — all overridable via env vars
# ---------------------------------------------------------------------------
TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
EXP_NAME="${EXP_NAME:-dq_full_rank1}"

SEED="${SEED:-42}"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
LR="${LR:-1e-4}"
WORLD_MODEL_BATCH_SIZE="${WORLD_MODEL_BATCH_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_HORIZON="${TRAIN_HORIZON:-8}"
PRECISION="${PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"

# ---------------------------------------------------------------------------
# DynQuery architecture
# ---------------------------------------------------------------------------
HISTORY_LENGTH="${HISTORY_LENGTH:-2}"            # K
NUM_DYNAMIC_QUERIES="${NUM_DYNAMIC_QUERIES:-8}"  # Q
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

# Core feature flags (default: all cores on)
USE_ACTION_CONDITIONED_MASK="${USE_ACTION_CONDITIONED_MASK:-1}"   # Core 1
PREDICTOR_MODE="${PREDICTOR_MODE:-query_wise}"                    # Core 2
USE_DYNAMIC_RESIDUAL_GATE="${USE_DYNAMIC_RESIDUAL_GATE:-1}"       # Core 3

# Scorer (stage_b = 1, stage_a = 0)
USE_ACTION_FUTURE_SCORER="${USE_ACTION_FUTURE_SCORER:-1}"
USE_MOTION_BIAS="${USE_MOTION_BIAS:-1}"

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
LAMBDA_MASK_DYNAMIC="${LAMBDA_MASK_DYNAMIC:-0.1}"          # Core 4a
LAMBDA_QUERY_DELTA_SPARSE="${LAMBDA_QUERY_DELTA_SPARSE:-0.001}"  # Core 4b
RANK_MARGIN="${RANK_MARGIN:-0.1}"
DYNAMIC_THRESHOLD="${DYNAMIC_THRESHOLD:-0.05}"
DYNAMIC_DILATE_KERNEL="${DYNAMIC_DILATE_KERNEL:-7}"
ROI_CROP_SIZE="${ROI_CROP_SIZE:-64}"

# ---------------------------------------------------------------------------
# Negatives / warm-start
# (NEGATIVE_TYPE and NEGATIVE_MIX are passed as env vars to DynQueryCollator)
# ---------------------------------------------------------------------------
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.15}"
INIT_FROM_CHECKPOINT="${INIT_FROM_CHECKPOINT:-}"

# ---------------------------------------------------------------------------
# Segment length: K + current + H frames per window
# ---------------------------------------------------------------------------
SEGMENT_LENGTH="${SEGMENT_LENGTH:-$(( HISTORY_LENGTH + TRAIN_HORIZON + 1 ))}"
ACTION_HORIZON=$(( SEGMENT_LENGTH - HISTORY_LENGTH - 1 ))

# ---------------------------------------------------------------------------
# Max steps
# ---------------------------------------------------------------------------
MAX_STEPS="${MAX_STEPS:-150000}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$(default_libero_data_root)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/libero/DynQueryWorldModel/core_sweep}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
setup_env

NPROC="${NPROC:-$(detect_gpu_count)}"

if [ -z "${GRAD_ACCUM:-}" ]; then
  if [ -n "${WORLD_MODEL_BATCH_SIZE}" ]; then
    TOTAL_PER_STEP=$(( BATCH_SIZE * NPROC ))
    if (( WORLD_MODEL_BATCH_SIZE % TOTAL_PER_STEP != 0 )); then
      echo "[train_dynquery] WORLD_MODEL_BATCH_SIZE=${WORLD_MODEL_BATCH_SIZE} must be divisible by BATCH_SIZE*NPROC=${TOTAL_PER_STEP}." >&2
      exit 1
    fi
    GRAD_ACCUM=$(( WORLD_MODEL_BATCH_SIZE / TOTAL_PER_STEP ))
  else
    GRAD_ACCUM=1
  fi
fi
GLOBAL_BATCH_SIZE=$(( BATCH_SIZE * NPROC * GRAD_ACCUM ))

if [ "${NUM_WORKERS}" = "0" ] && [ "${NPROC}" = "1" ]; then
  NUM_WORKERS=2
fi

# Stage label
if [ "${USE_ACTION_FUTURE_SCORER}" = "0" ] || [ "${USE_ACTION_FUTURE_SCORER}" = "false" ]; then
  STAGE="dq_a"
else
  STAGE="dq_b"
fi

SAVE_DIR="${OUTPUT_ROOT}/${TASK_SUITE}/${EXP_NAME}/s${SEED}"
mkdir -p "${SAVE_DIR}"

LOG_DIR="${REPO_ROOT}/logs/libero/DynQueryWorldModel/${TASK_SUITE}/${EXP_NAME}/s${SEED}"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/train_$(timestamp).log"

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print_run_summary \
  "TASK_SUITE"            "${TASK_SUITE}" \
  "EXP_NAME"              "${EXP_NAME}" \
  "STAGE"                 "${STAGE}" \
  "HISTORY_LENGTH"        "${HISTORY_LENGTH} (K)" \
  "NUM_DYNAMIC_Q"         "${NUM_DYNAMIC_QUERIES} (Q)" \
  "SEGMENT_LEN"           "${SEGMENT_LENGTH}  (K=${HISTORY_LENGTH}  H=${ACTION_HORIZON})" \
  "LR"                    "${LR}" \
  "BATCH(eff.)"           "per-dev=${BATCH_SIZE}  gpus=${NPROC}  accum=${GRAD_ACCUM}  eff=${GLOBAL_BATCH_SIZE}" \
  "MAX_STEPS"             "${MAX_STEPS}" \
  "PRECISION"             "${PRECISION}" \
  "Core1 act_cond_mask"   "${USE_ACTION_CONDITIONED_MASK}" \
  "Core2 predictor_mode"  "${PREDICTOR_MODE}" \
  "Core3 dyn_res_gate"    "${USE_DYNAMIC_RESIDUAL_GATE}" \
  "Core4 λ_mask_dyn"      "${LAMBDA_MASK_DYNAMIC}  λ_q_delta=${LAMBDA_QUERY_DELTA_SPARSE}" \
  "motion_bias"           "${USE_MOTION_BIAS}" \
  "use_scorer"            "${USE_ACTION_FUTURE_SCORER}  λ_rank=${LAMBDA_RANK}" \
  "SEED"                  "${SEED}" \
  "DATA_ROOT"             "${DATA_ROOT}" \
  "SAVE_DIR"              "${SAVE_DIR}" \
  "LOGFILE"               "${LOGFILE}"

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
  "max_steps=${MAX_STEPS}" \
  "precision=${PRECISION}" \
  "use_action_conditioned_mask=${USE_ACTION_CONDITIONED_MASK}" \
  "predictor_mode=${PREDICTOR_MODE}" \
  "use_dynamic_residual_gate=${USE_DYNAMIC_RESIDUAL_GATE}" \
  "lambda_mask_dynamic=${LAMBDA_MASK_DYNAMIC}" \
  "lambda_query_delta_sparse=${LAMBDA_QUERY_DELTA_SPARSE}" \
  "lambda_image=${LAMBDA_IMAGE}" \
  "lambda_dynamic=${LAMBDA_DYNAMIC}" \
  "lambda_static=${LAMBDA_STATIC}" \
  "lambda_query=${LAMBDA_QUERY}" \
  "lambda_rank=${LAMBDA_RANK}" \
  "rank_margin=${RANK_MARGIN}" \
  "dynamic_threshold=${DYNAMIC_THRESHOLD}" \
  "use_action_future_scorer=${USE_ACTION_FUTURE_SCORER}" \
  "use_motion_bias=${USE_MOTION_BIAS}" \
  "action_conditioning_mode=${ACTION_CONDITIONING_MODE}" \
  "action_noise_std=${ACTION_NOISE_STD}" \
  "init_from_checkpoint=${INIT_FROM_CHECKPOINT}" \
  "seed=${SEED}" \
  "data_root=${DATA_ROOT}" \
  "save_dir=${SAVE_DIR}"

copy_script_snapshot "${SAVE_DIR}" "${BASH_SOURCE[0]}"

# ---------------------------------------------------------------------------
# Build training command
# ---------------------------------------------------------------------------
TRAIN_ARGS=(
  -m worldmodel.dynquery.train
  --task-suite               "${TASK_SUITE}"
  --data-root                "${DATA_ROOT}"
  --output-dir               "${SAVE_DIR}"
  --history-length           "${HISTORY_LENGTH}"
  --action-horizon           "${ACTION_HORIZON}"
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
  # Core 2: predictor mode
  --predictor-mode           "${PREDICTOR_MODE}"
  # Loss weights
  --lambda-image             "${LAMBDA_IMAGE}"
  --lambda-dynamic           "${LAMBDA_DYNAMIC}"
  --lambda-static            "${LAMBDA_STATIC}"
  --lambda-query             "${LAMBDA_QUERY}"
  --lambda-rank              "${LAMBDA_RANK}"
  --lambda-mask-dynamic      "${LAMBDA_MASK_DYNAMIC}"
  --lambda-query-delta-sparse "${LAMBDA_QUERY_DELTA_SPARSE}"
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

# Core 1: action-conditioned mask
case "${USE_ACTION_CONDITIONED_MASK}" in
  0|false|FALSE|False|no|NO) TRAIN_ARGS+=(--no-action-conditioned-mask) ;;
  *)                         TRAIN_ARGS+=(--use-action-conditioned-mask) ;;
esac

# Core 3: dynamic residual gate
case "${USE_DYNAMIC_RESIDUAL_GATE}" in
  0|false|FALSE|False|no|NO) TRAIN_ARGS+=(--no-dynamic-residual-gate) ;;
  *)                         TRAIN_ARGS+=(--use-dynamic-residual-gate) ;;
esac

# Scorer (stage_b)
case "${USE_ACTION_FUTURE_SCORER}" in
  0|false|FALSE|False|no|NO) TRAIN_ARGS+=(--no-action-future-scorer) ;;
  *)                         TRAIN_ARGS+=(--use-action-future-scorer) ;;
esac

# Motion bias
case "${USE_MOTION_BIAS}" in
  1|true|TRUE|True|yes|YES) TRAIN_ARGS+=(--use-motion-bias) ;;
  *)                        TRAIN_ARGS+=(--no-motion-bias) ;;
esac

TRAIN_ARGS+=(--action-noise-std "${ACTION_NOISE_STD}")

if [ -n "${INIT_FROM_CHECKPOINT}" ]; then
  TRAIN_ARGS+=(--init-from-checkpoint "${INIT_FROM_CHECKPOINT}")
fi

if [ "${TF32:-0}" = "1" ]; then
  TRAIN_ARGS+=(--tf32)
fi

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
echo ""
echo "  Starting DynQuery training → ${SAVE_DIR}"
echo "  Stage: ${STAGE}  (use_action_future_scorer=${USE_ACTION_FUTURE_SCORER})"
echo "  Log → ${LOGFILE}"
echo ""

export TF_CPP_MIN_LOG_LEVEL=3

run_train_command "${NPROC}" "${SAVE_DIR}" "${LOGFILE}" "${TRAIN_ARGS[@]}"

echo ""
echo "  Training complete."
echo "    checkpoint : ${SAVE_DIR}/final"
echo "    log        : ${LOGFILE}"
echo ""
echo "  To evaluate:"
echo "    bash scripts/libero/residual_worldmodel/eval_v4_temporal_query_worldmodel.sh \\"
echo "      MODEL_DIR=${SAVE_DIR}/final TASK_SUITE=${TASK_SUITE}"
