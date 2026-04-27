#!/usr/bin/env bash
# train_single.sh — 単発実験用スクリプト
#
# 使い方:
#   1. 下の USER CONFIG ブロックを書き換える
#   2. bash scripts/libero/residual_worldmodel/train_single.sh
#
# 実行後の保存先:
#   ${OUTPUT_ROOT}/${RUN_NAME}/
#     ├── cmd.sh              ← 再実行可能なコマンド
#     ├── config_dump.json    ← 設定の完全ダンプ
#     ├── script_snapshot.sh  ← このスクリプト自身のコピー
#     ├── train.log           ← 学習ログ
#     ├── train_metrics.jsonl ← ステップ毎 metrics
#     ├── train_metrics.csv   ← 同 CSV
#     └── checkpoint-*/       ← checkpoints

set -euo pipefail

######################################################################
########## USER CONFIG — ここだけ書き換えて使う ##########
######################################################################

# ---- 実験識別 ---------------------------------------------------
EXP_NAME="baseline_v1"          # 実験名 (run_name に使われる)
TASK_NAME="spatial"             # spatial | object | goal | long
TASK_SCHEDULE_KEY=""            # 空=TASK_NAMEから自動; 例: spatial_debug

# ---- パス -------------------------------------------------------
# 環境変数で上書き可能:
#   LIBERO_DATA_ROOT=/localdata/modified_libero_rlds
#   LOCALDATA_ROOT=/localdata
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}/checkpoints/libero/FocusedWM"

# ---- DINO backbone ----------------------------------------------
DINO_BACKBONE="dinov2_vits14"   # dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14
DINO_INPUT_SIZE=224             # 224/14=16 → 16×16=256 patches
DINO_FROZEN=true                # true=frozen(推奨) | false=全部学習
DINO_FINETUNE_N_LAYERS=0        # DINO_FROZEN=false の時のみ有効
DINO_WEIGHTS_PATH=""            # ローカルファイルパス (絶対 or REPO_ROOT からの相対); 空なら torch.hub でダウンロード
                               # 例: "checkpoints/dino/dinov2_vits14_pretrain.pth"

# ---- モデル構成 -------------------------------------------------
MODEL_VARIANT="full"            # full | no_focus | no_dino_loss | pixel_focus | image_rank
#   full         : すべての損失あり (推奨)
#   no_focus     : FocusHead なし
#   no_dino_loss : DINO 特徴損失なし (pixel focus のみ)
#   pixel_focus  : focus supervision を pixel diff で行う
#   image_rank   : ランキングスコアを image only にする

IMAGE_HEIGHT=256                # 再構成ターゲット解像度 (256/patch_hw=16 → 2^4 ✓)
IMAGE_WIDTH=256
HIDDEN_DIM=256
ACTION_EMB_DIM=128
N_ACTION_ENC_LAYERS=2
N_PRED_LAYERS=4
N_HEADS=4
FFN_DIM=1024
DROPOUT=0.0

# ---- 学習設定 ---------------------------------------------------
LR="1e-4"                       # 学習率
BATCH_SIZE=4                    # per-device batch size
GRAD_ACCUM=1                    # gradient accumulation steps
#   effective batch = BATCH_SIZE * N_GPUS * GRAD_ACCUM
ACTION_HORIZON=7                # = segment_length - 1
OVERRIDE_MAX_STEPS=""           # 空=task別デフォルト; 例: 12000
PRECISION="bf16"                # bf16 | fp16 | fp32
WARMUP_RATIO=0.02
WEIGHT_DECAY=0.0
MAX_GRAD_NORM=1.0

# ---- 損失 -------------------------------------------------------
RECON_LOSS_WEIGHT=1.0
USE_LPIPS=false                 # true=LPIPS損失あり (pip install lpips が必要)
LPIPS_LOSS_WEIGHT=0.1

USE_DINO_FEATURE_LOSS=true
DINO_FEATURE_LOSS_WEIGHT=0.1

FOCUS_SUPERVISION_TYPE="dino_diff"  # dino_diff | pixel_diff
FOCUS_SUPERVISION_WEIGHT=0.1
CHANGE_TARGET_THRESHOLD=0.0         # 0=soft / >0=binarize

USE_FOCUS_SPARSITY=true
FOCUS_SPARSITY_MODE="l1"            # l1 | entropy
FOCUS_SPARSITY_WEIGHT=0.01

# ---- 候補ランキング ----------------------------------------------
RANKING_SCORE_TYPE="dino_only"      # dino_only | image_only | combined
RANKING_IMAGE_WEIGHT=0.3            # "combined" のみ使用
NEGATIVE_MODE="all"                 # all | noise | shuffle | roll
NOISE_STD=0.05
NUM_CANDIDATES=4

# ---- 評価 / 可視化 スケジュール ---------------------------------
OVERRIDE_EVAL_EVERY=""          # 空=task別デフォルト
OVERRIDE_SAVE_VIZ_EVERY=""      # 空=task別デフォルト
NUM_RANK_EVAL_BATCHES=32        # action ranking 評価バッチ数

# ---- Checkpoint -------------------------------------------------
SAVE_STEPS=5000
SAVE_TOTAL_LIMIT=3

# ---- その他 -----------------------------------------------------
SEED=42
NUM_WORKERS=2
DEBUG_MODE="fast_debug"             # fast_debug | normal | full_report
RESUME_PATH=""                  # 空=最初から / "path/to/model.pt"=再開

# ---- 学習後の可視化 ---------------------------------------------
RUN_VIS_AFTER_TRAIN=true       # true にすると学習後に visualize_eval.sh を自動実行
VIS_SPLIT="heldout"             # 可視化時のデータ split

######################################################################
########## END USER CONFIG ##########
######################################################################

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
TASK_SCHEDULE_KEY="$(resolve_task_schedule_key "${TASK_NAME}" "${TASK_SCHEDULE_KEY:-}")"
MAX_STEPS="$(resolve_task_schedule_value get_max_steps_for_task "${TASK_SCHEDULE_KEY}" "${OVERRIDE_MAX_STEPS:-}")"
EVAL_EVERY="$(resolve_task_schedule_value get_eval_every_for_task "${TASK_SCHEDULE_KEY}" "${OVERRIDE_EVAL_EVERY:-}")"
SAVE_VIZ_EVERY="$(resolve_task_schedule_value get_vis_every_for_task "${TASK_SCHEDULE_KEY}" "${OVERRIDE_SAVE_VIZ_EVERY:-}")"

NPROC=$(detect_gpu_count)
SEGMENT_LENGTH=$(( ACTION_HORIZON + 1 ))
GLOBAL_BATCH_SIZE=$(( BATCH_SIZE * NPROC * GRAD_ACCUM ))

# ---------------------------------------------------------------------------
# Run name & save dir
# ---------------------------------------------------------------------------
RUN_NAME="$(make_run_name \
  "${MODEL_VARIANT}" \
  "${DINO_BACKBONE}" \
  "${LR}" \
  "${BATCH_SIZE}" \
  "${ACTION_HORIZON}")"

SAVE_DIR="$(make_save_dir "${OUTPUT_ROOT}/${TASK_NAME}/${EXP_NAME}/s${SEED}" "${RUN_NAME}")"
LOG_DIR="${REPO_ROOT}/logs/libero/FocusedWM/${TASK_NAME}/${EXP_NAME}/s${SEED}"
ensure_dir "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${RUN_NAME}.log"

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print_run_summary \
  "EXP_NAME"         "${EXP_NAME}" \
  "TASK_NAME"        "${TASK_NAME}" \
  "TASK_SUITE"       "${TASK_SUITE}" \
  "TASK_SCHED_KEY"   "${TASK_SCHEDULE_KEY}" \
  "MODEL_VARIANT"    "${MODEL_VARIANT}" \
  "DINO_BACKBONE"    "${DINO_BACKBONE}  (frozen=${DINO_FROZEN})" \
  "LR"               "${LR}" \
  "BATCH(eff.)"      "per-dev=${BATCH_SIZE}  gpus=${NPROC}  accum=${GRAD_ACCUM}  eff=${GLOBAL_BATCH_SIZE}" \
  "ACTION_HORIZON"   "${ACTION_HORIZON}  (segment_len=${SEGMENT_LENGTH})" \
  "MAX_STEPS"        "${MAX_STEPS}" \
  "EVAL_EVERY"       "${EVAL_EVERY}" \
  "VIS_EVERY"        "${SAVE_VIZ_EVERY}" \
  "LOSSES"           "recon=${RECON_LOSS_WEIGHT}  dino_feat=${DINO_FEATURE_LOSS_WEIGHT}  focus_sup=${FOCUS_SUPERVISION_WEIGHT}" \
  "FOCUS_TYPE"       "${FOCUS_SUPERVISION_TYPE}  sparsity=${FOCUS_SPARSITY_WEIGHT}" \
  "RANKING"          "${RANKING_SCORE_TYPE}  negs=${NEGATIVE_MODE}" \
  "SEED"             "${SEED}" \
  "DEBUG_MODE"       "${DEBUG_MODE}" \
  "DATASET_PATH"     "${DATASET_PATH}" \
  "SAVE_DIR"         "${SAVE_DIR}" \
  "LOGFILE"          "${LOGFILE}"

# ---------------------------------------------------------------------------
# Config dump (before training)
# ---------------------------------------------------------------------------
dump_run_config "${SAVE_DIR}" \
  "exp_name=${EXP_NAME}" \
  "task_name=${TASK_NAME}" \
  "task_suite=${TASK_SUITE}" \
  "task_schedule_key=${TASK_SCHEDULE_KEY}" \
  "model_variant=${MODEL_VARIANT}" \
  "dino_backbone=${DINO_BACKBONE}" \
  "dino_frozen=${DINO_FROZEN}" \
  "dino_input_size=${DINO_INPUT_SIZE}" \
  "lr=${LR}" \
  "batch_size=${BATCH_SIZE}" \
  "grad_accum=${GRAD_ACCUM}" \
  "global_batch_size=${GLOBAL_BATCH_SIZE}" \
  "action_horizon=${ACTION_HORIZON}" \
  "max_steps=${MAX_STEPS}" \
  "eval_every=${EVAL_EVERY}" \
  "vis_every=${SAVE_VIZ_EVERY}" \
  "precision=${PRECISION}" \
  "recon_loss_weight=${RECON_LOSS_WEIGHT}" \
  "dino_feature_loss_weight=${DINO_FEATURE_LOSS_WEIGHT}" \
  "focus_supervision_type=${FOCUS_SUPERVISION_TYPE}" \
  "focus_supervision_weight=${FOCUS_SUPERVISION_WEIGHT}" \
  "focus_sparsity_weight=${FOCUS_SPARSITY_WEIGHT}" \
  "ranking_score_type=${RANKING_SCORE_TYPE}" \
  "negative_mode=${NEGATIVE_MODE}" \
  "dataset_path=${DATASET_PATH}" \
  "seed=${SEED}" \
  "run_name=${RUN_NAME}" \
  "save_dir=${SAVE_DIR}"

copy_script_snapshot "${SAVE_DIR}" "${BASH_SOURCE[0]}"

# ---------------------------------------------------------------------------
# Build training command
# ---------------------------------------------------------------------------
_build_train_args() {
  local args=(
    -m worldmodel.residual_worldmodel.train_focused_libero
    --task-suite "${TASK_SUITE}"
    --data-root "${DATASET_PATH}"
    --output-dir "${SAVE_DIR}"
    --max-steps "${MAX_STEPS}"
    --segment-length "${SEGMENT_LENGTH}"
    --batch-size-per-device "${BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --precision "${PRECISION}"
    --learning-rate "${LR}"
    --warmup-ratio "${WARMUP_RATIO}"
    --weight-decay "${WEIGHT_DECAY}"
    --max-grad-norm "${MAX_GRAD_NORM}"
    --save-steps "${SAVE_STEPS}"
    --logging-steps 10
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
    --eval-every "${EVAL_EVERY}"
    --save-viz-every "${SAVE_VIZ_EVERY}"
    --num-rank-eval-batches "${NUM_RANK_EVAL_BATCHES}"
    --debug-mode "${DEBUG_MODE}"
    --seed "${SEED}"
    # DINO
    --dino-model-name "${DINO_BACKBONE}"
    --dino-input-size "${DINO_INPUT_SIZE}"
    --dino-finetune-last-n-layers "${DINO_FINETUNE_N_LAYERS}"
    # Architecture
    --model-variant "${MODEL_VARIANT}"
    --image-height "${IMAGE_HEIGHT}"
    --image-width "${IMAGE_WIDTH}"
    --hidden-dim "${HIDDEN_DIM}"
    --action-emb-dim "${ACTION_EMB_DIM}"
    --n-action-enc-layers "${N_ACTION_ENC_LAYERS}"
    --n-pred-layers "${N_PRED_LAYERS}"
    --n-heads "${N_HEADS}"
    --ffn-dim "${FFN_DIM}"
    --dropout "${DROPOUT}"
    # Loss weights
    --recon-loss-weight "${RECON_LOSS_WEIGHT}"
    --dino-feature-loss-weight "${DINO_FEATURE_LOSS_WEIGHT}"
    --focus-supervision-weight "${FOCUS_SUPERVISION_WEIGHT}"
    --focus-sparsity-weight "${FOCUS_SPARSITY_WEIGHT}"
    --focus-sparsity-mode "${FOCUS_SPARSITY_MODE}"
    --change-target-threshold "${CHANGE_TARGET_THRESHOLD}"
    # Ranking / negatives
    --ranking-score-type "${RANKING_SCORE_TYPE}"
    --ranking-image-weight "${RANKING_IMAGE_WEIGHT}"
    --negative-mode "${NEGATIVE_MODE}"
    --noise-std "${NOISE_STD}"
    --num-action-candidates "${NUM_CANDIDATES}"
    # Ranking eval
    --run-rank-eval
  )

  # DINO frozen flag
  if [ "${DINO_FROZEN}" = "true" ]; then
    args+=(--dino-frozen)
  else
    args+=(--dino-no-frozen)
  fi

  # DINO local weights — resolve absolute path, warn and fall back to torch.hub if missing
  if [ -n "${DINO_WEIGHTS_PATH}" ]; then
    local _dino_abs=""
    if [ -f "${DINO_WEIGHTS_PATH}" ]; then
      _dino_abs="${DINO_WEIGHTS_PATH}"
    elif [ -f "${REPO_ROOT}/${DINO_WEIGHTS_PATH}" ]; then
      _dino_abs="${REPO_ROOT}/${DINO_WEIGHTS_PATH}"
    fi
    if [ -n "${_dino_abs}" ]; then
      args+=(--dino-weights-path "${_dino_abs}")
    else
      echo "[WARN] DINO_WEIGHTS_PATH='${DINO_WEIGHTS_PATH}' not found — falling back to torch.hub" >&2
    fi
  fi

  # LPIPS
  if [ "${USE_LPIPS}" = "true" ]; then
    args+=(--use-lpips-loss --lpips-loss-weight "${LPIPS_LOSS_WEIGHT}")
  fi

  # DINO feature loss
  if [ "${USE_DINO_FEATURE_LOSS}" = "true" ]; then
    args+=(--use-dino-feature-loss)
  else
    args+=(--no-dino-feature-loss)
  fi

  # Focus supervision type
  case "${FOCUS_SUPERVISION_TYPE}" in
    dino_diff)
      args+=(--use-dino-focus-supervision)
      ;;
    pixel_diff)
      args+=(--no-dino-focus-supervision --use-pixel-focus-supervision)
      ;;
    mixed)
      args+=(--use-dino-focus-supervision --use-pixel-focus-supervision)
      ;;
  esac

  # Focus sparsity
  if [ "${USE_FOCUS_SPARSITY}" = "true" ]; then
    args+=(--use-focus-sparsity)
  else
    args+=(--no-focus-sparsity)
  fi

  # Model variant overrides
  case "${MODEL_VARIANT}" in
    no_focus)
      args+=(--no-focus-head --no-dino-focus-supervision --no-focus-sparsity)
      ;;
    no_dino_loss)
      args+=(--no-dino-feature-loss --no-dino-focus-supervision)
      ;;
  esac

  # Resume
  if [ -n "${RESUME_PATH}" ]; then
    args+=(--resume-from "${RESUME_PATH}")
  fi

  echo "${args[@]}"
}

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
echo ""
echo "  Starting training → ${SAVE_DIR}"
echo ""

read -ra TRAIN_ARGS <<< "$(_build_train_args)"

run_train_command "${NPROC}" "${SAVE_DIR}" "${LOGFILE}" "${TRAIN_ARGS[@]}"

echo ""
echo "  Training done. Outputs:"
echo "    checkpoint : ${SAVE_DIR}"
echo "    log        : ${LOGFILE}"
echo ""

# ---------------------------------------------------------------------------
# Optional: post-training visualization
# ---------------------------------------------------------------------------
if [ "${RUN_VIS_AFTER_TRAIN}" = "true" ]; then
  echo "  Running post-training evaluation …"
  CKPT="$(find_checkpoint "${SAVE_DIR}" auto)"
  if [ -n "${CKPT}" ]; then
    bash "${SCRIPT_DIR}/visualize_eval.sh" \
      RUN_DIR="${SAVE_DIR}" \
      CHECKPOINT_PATH="${CKPT}" \
      TASK_NAME="${TASK_NAME}" \
      SPLIT="${VIS_SPLIT}" \
      OUTPUT_DIR="${SAVE_DIR}/full_eval"
  else
    echo "  [WARN] No checkpoint found; skipping post-training visualization."
  fi
fi
