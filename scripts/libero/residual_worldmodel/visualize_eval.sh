#!/usr/bin/env bash
# visualize_eval.sh — 学習済み checkpoint の評価・可視化
#
# 使い方:
#   1. USER CONFIG の RUN_DIR / CHECKPOINT_PATH を書き換える
#   2. bash scripts/libero/residual_worldmodel/visualize_eval.sh
#
# 評価したいモデルのタイプを MODEL_TYPE で切り替える:
#   MODEL_TYPE=focused  → ActionConditionedFocusedResidualWM (DINO)
#   MODEL_TYPE=residual → LatentResidualWorldModel (FSQ)

set -euo pipefail

######################################################################
########## USER CONFIG — ここだけ書き換えて使う ##########
######################################################################

# ---- ターゲット run ---------------------------------------------
RUN_DIR="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}/checkpoints/libero/FocusedWM/spatial/baseline_v1/s42/20260424_full_vits14_lr1e-4_bs4_hor7"

# ---- Checkpoint 選択 -------------------------------------------
# best_recon | best_dino_feature | best_rank | final | latest | auto
USE_CHECKPOINT="best_recon"
# 直接指定する場合は CHECKPOINT_PATH に model.pt のパスを入れる
# 空のままにすると USE_CHECKPOINT で自動選択
CHECKPOINT_PATH=""

# ---- タスク / モデルタイプ --------------------------------------
TASK_NAME="spatial"         # spatial | object | goal | long
MODEL_TYPE="focused"        # focused | residual

# ---- 評価設定 ---------------------------------------------------
SPLIT="heldout"             # heldout | all (※ residual model は別フラグ体系)
NUM_SAMPLES=200             # 評価するウィンドウ/サンプル数
SAVE_VIS=true               # true=可視化 PNG を保存
SAVE_REPORT=true            # true=metrics.json / summary.csv を保存

# focused model 固有
NUM_RANK_EVAL_BATCHES=64    # action ranking 評価バッチ数
SAVE_VIZ_STEP_EXAMPLES=10  # casebook 用のサンプル数

# residual model 固有 (MODEL_TYPE=residual の場合のみ使用)
VISUAL_TOKENIZER="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}/checkpoints/libero/WorldModel/Tokenizer"
SEGMENT_LENGTH=8
DISPLAY_FRAMES=6
NUM_FULL_EPISODES=1
MOTION_THRESHOLD=0.05
FG_THRESHOLD=0.05
ROBOT_ROI_FRAC=0.35
ROI_MIN_PIXELS=50

# ---- 出力先 -----------------------------------------------------
OUTPUT_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}/evals/libero"
# 空にすると RUN_DIR/eval_YYYYMMDD/ に保存
OUTPUT_DIR=""

# ---- その他 -----------------------------------------------------
SEED=42
DEVICE="auto"
DATASET_PATH="${DATASET_PATH:-}"

######################################################################
########## END USER CONFIG ##########
######################################################################

# ---------------------------------------------------------------------------
# Parse KEY=VALUE overrides from command-line arguments
# (allows train_single.sh etc. to call this script with overrides)
# ---------------------------------------------------------------------------
for _arg in "$@"; do
  case "${_arg}" in
    RUN_DIR=*)               RUN_DIR="${_arg#*=}" ;;
    CHECKPOINT_PATH=*)       CHECKPOINT_PATH="${_arg#*=}" ;;
    USE_CHECKPOINT=*)        USE_CHECKPOINT="${_arg#*=}" ;;
    TASK_NAME=*)             TASK_NAME="${_arg#*=}" ;;
    MODEL_TYPE=*)            MODEL_TYPE="${_arg#*=}" ;;
    SPLIT=*)                 SPLIT="${_arg#*=}" ;;
    NUM_SAMPLES=*)           NUM_SAMPLES="${_arg#*=}" ;;
    OUTPUT_DIR=*)            OUTPUT_DIR="${_arg#*=}" ;;
    SAVE_VIS=*)              SAVE_VIS="${_arg#*=}" ;;
    SAVE_REPORT=*)           SAVE_REPORT="${_arg#*=}" ;;
    SEED=*)                  SEED="${_arg#*=}" ;;
    NUM_RANK_EVAL_BATCHES=*) NUM_RANK_EVAL_BATCHES="${_arg#*=}" ;;
    DATASET_PATH=*)          DATASET_PATH="${_arg#*=}" ;;
    *)                       echo "[WARN] Unknown argument: ${_arg}" >&2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Common setup
# ---------------------------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"
setup_env
DATASET_PATH="$(resolve_libero_data_root "${DATASET_PATH:-}")"
validate_task_name "${TASK_NAME}"
TASK_SUITE="$(normalize_libero_task_name "${TASK_NAME}")"

# ---------------------------------------------------------------------------
# Validate RUN_DIR
# ---------------------------------------------------------------------------
if [ ! -d "${RUN_DIR}" ]; then
  echo "[ERROR] RUN_DIR not found: ${RUN_DIR}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Resolve checkpoint
# ---------------------------------------------------------------------------
if [ -z "${CHECKPOINT_PATH}" ]; then
  CHECKPOINT_PATH="$(find_checkpoint "${RUN_DIR}" "${USE_CHECKPOINT}")"
  if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "[ERROR] No checkpoint found in ${RUN_DIR} (type=${USE_CHECKPOINT})." >&2
    exit 1
  fi
fi

# model.pt path
if [ -f "${CHECKPOINT_PATH}/model.pt" ]; then
  MODEL_PT="${CHECKPOINT_PATH}/model.pt"
elif [ -f "${CHECKPOINT_PATH}" ]; then
  MODEL_PT="${CHECKPOINT_PATH}"
else
  echo "[ERROR] model.pt not found in ${CHECKPOINT_PATH}" >&2
  exit 1
fi

# Config path
CONFIG_JSON=""
for cfg_path in \
  "${CHECKPOINT_PATH}/config.json" \
  "${RUN_DIR}/checkpoint-final-step*/config.json" \
  "${RUN_DIR}/config_dump.json"; do
  found_cfg=$(ls ${cfg_path} 2>/dev/null | head -1 || true)
  if [ -n "${found_cfg}" ] && [ -f "${found_cfg}" ]; then
    CONFIG_JSON="${found_cfg}"
    break
  fi
done

# ---------------------------------------------------------------------------
# Output dir
# ---------------------------------------------------------------------------
RUN_LABEL=$(basename "${RUN_DIR}")
if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_TYPE}/${TASK_NAME}/${RUN_LABEL}/eval_$(short_date)"
fi
ensure_dir "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print_run_summary \
  "MODEL_TYPE"       "${MODEL_TYPE}" \
  "TASK_NAME"        "${TASK_NAME}" \
  "TASK_SUITE"       "${TASK_SUITE}" \
  "RUN_DIR"          "${RUN_DIR}" \
  "CHECKPOINT"       "${CHECKPOINT_PATH}" \
  "MODEL_PT"         "${MODEL_PT}" \
  "CONFIG_JSON"      "${CONFIG_JSON:-none}" \
  "DATASET_PATH"     "${DATASET_PATH}" \
  "SPLIT"            "${SPLIT}" \
  "NUM_SAMPLES"      "${NUM_SAMPLES}" \
  "SAVE_VIS"         "${SAVE_VIS}" \
  "SAVE_REPORT"      "${SAVE_REPORT}" \
  "OUTPUT_DIR"       "${OUTPUT_DIR}"

copy_script_snapshot "${OUTPUT_DIR}" "${BASH_SOURCE[0]}"
dump_run_config "${OUTPUT_DIR}" \
  "run_dir=${RUN_DIR}" \
  "checkpoint_path=${CHECKPOINT_PATH}" \
  "model_pt=${MODEL_PT}" \
  "task_name=${TASK_NAME}" \
  "task_suite=${TASK_SUITE}" \
  "model_type=${MODEL_TYPE}" \
  "dataset_path=${DATASET_PATH}" \
  "split=${SPLIT}" \
  "num_samples=${NUM_SAMPLES}"

LOGFILE="${OUTPUT_DIR}/eval.log"

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
if [ "${MODEL_TYPE}" = "focused" ]; then
  # ---- ActionConditionedFocusedResidualWM (DINO) ----------------------
  EVAL_ARGS=(
    -m worldmodel.residual_worldmodel.focused_visualize
    --mode full_eval
    --task-suite "${TASK_SUITE}"
    --data-root "${DATASET_PATH}"
    --model-pt "${MODEL_PT}"
    --output-dir "${OUTPUT_DIR}"
    --max-batches "${NUM_SAMPLES}"
    --seed "${SEED}"
  )

  if [ -n "${CONFIG_JSON}" ]; then
    EVAL_ARGS+=(--config-json "${CONFIG_JSON}")
  fi

  if [ "${SAVE_VIS}" = "true" ]; then
    EVAL_ARGS+=(--save-vis)
  fi

  if [ "${SAVE_REPORT}" = "true" ]; then
    EVAL_ARGS+=(--save-report)
  fi

  run_visualize_command "${OUTPUT_DIR}" "${LOGFILE}" "${EVAL_ARGS[@]}"

elif [ "${MODEL_TYPE}" = "residual" ]; then
  # ---- LatentResidualWorldModel (FSQ) ---------------------------------
  if [ ! -d "${VISUAL_TOKENIZER}" ]; then
    echo "[ERROR] VISUAL_TOKENIZER not found: ${VISUAL_TOKENIZER}" >&2
    exit 1
  fi

  SAVE_FRAMES_FLAG=""
  # if needed: SAVE_FRAMES_FLAG="--save-full-episode-frames"

  EVAL_ARGS=(
    -m worldmodel.residual_worldmodel.visualize
    --task-suite "${TASK_SUITE}"
    --data-root "${DATASET_PATH}"
    --model-dir "${RUN_DIR}"
    --visual-tokenizer "${VISUAL_TOKENIZER}"
    --output-dir "${OUTPUT_DIR}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --num-eval-windows "${NUM_SAMPLES}"
    --segment-length "${SEGMENT_LENGTH}"
    --split-mode "${SPLIT}"
    --heldout-ratio 0.2
    --eval-batch-size 1
    --display-frames "${DISPLAY_FRAMES}"
    --save-casebook-count 5
    --num-full-episodes-per-task "${NUM_FULL_EPISODES}"
    --full-episode-index 0
    --full-episode-split-mode fallback_all
    --full-episode-display-cols "${DISPLAY_FRAMES}"
    --motion-threshold "${MOTION_THRESHOLD}"
    --fg-threshold "${FG_THRESHOLD}"
    --robot-roi-frac "${ROBOT_ROI_FRAC}"
    --roi-min-pixels "${ROI_MIN_PIXELS}"
    ${SAVE_FRAMES_FLAG}
  )

  run_visualize_command "${OUTPUT_DIR}" "${LOGFILE}" "${EVAL_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "  Evaluation done."
echo "  Results: ${OUTPUT_DIR}"
echo "  Log    : ${LOGFILE}"

# Print key metrics if metrics.json exists
METRICS_JSON="${OUTPUT_DIR}/metrics.json"
if [ -f "${METRICS_JSON}" ]; then
  echo ""
  echo "  Key metrics:"
  python -c "
import json, sys
with open('${METRICS_JSON}') as f:
    m = json.load(f)
keys = ['future_image_smooth_l1', 'dino_feature_mse', 'dino_cosine_similarity',
        'focus_mean', 'iou_vs_change', 'pairwise_acc']
for k in keys:
    if k in m:
        print(f'    {k:<35} {m[k]:.5f}')
" 2>/dev/null || true
fi
