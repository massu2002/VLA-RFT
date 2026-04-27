#!/usr/bin/env bash
# train_ablation.sh — アブレーション比較用スクリプト
#
# 使い方:
#   1. ABLATION_DESC と ABLATION_CONDITIONS を書き換える
#   2. 比較軸に合わせて BASE_* 設定を調整する
#   3. bash scripts/libero/residual_worldmodel/train_ablation.sh
#
# 各条件の run_name に ablation ラベルが入るので
# collect_results.sh で後から比較しやすい。
#
# dry-run:
#   DRY_RUN=true bash scripts/libero/residual_worldmodel/train_ablation.sh

set -euo pipefail

######################################################################
########## USER CONFIG — ここだけ書き換えて使う ##########
######################################################################

# ---- このアブレーションの説明 (スクリプト冒頭に書いておく) --------
ABLATION_DESC="
比較軸:
  1. FocusHead あり vs なし
  2. DINO 特徴損失 あり vs なし
  3. Focus supervision target: dino_diff vs pixel_diff vs mixed
  4. Ranking score: dino_only vs image_only vs combined
  5. Negative 生成: noise vs shuffle vs all
"
# ---- 共通固定設定 -----------------------------------------------
ABLATION_GROUP="focus_and_dino_ablation"  # グループ名 (ディレクトリに使用)
TASK_NAME="spatial"                      # 空の TASK_NAME_LIST を使うときの既定 task
TASK_NAME_LIST=()                        # 例: ("spatial" "long")
TASK_SCHEDULE_KEY_LIST=()                # 例: ("spatial" "long_debug")
DINO_BACKBONE="dinov2_vits14"
DINO_FROZEN=true
DINO_INPUT_SIZE=224
IMAGE_HEIGHT=256                # 再構成ターゲット解像度 (256/patch_hw=16 → 2^4 ✓)
IMAGE_WIDTH=256

LR="1e-4"
BATCH_SIZE=4
GRAD_ACCUM=1
ACTION_HORIZON=7
OVERRIDE_MAX_STEPS=""                    # 全 task 共通 override
OVERRIDE_MAX_STEPS_LIST=()               # task ごとの override
PRECISION="bf16"
SEED=42

# 共通損失設定 (条件で上書きしない項目)
RECON_LOSS_WEIGHT=1.0
FOCUS_SUPERVISION_WEIGHT=0.1
FOCUS_SPARSITY_WEIGHT=0.01
DINO_FEATURE_LOSS_WEIGHT=0.1
NUM_CANDIDATES=4
NOISE_STD=0.05
OVERRIDE_EVAL_EVERY=""                   # 全 task 共通 override
OVERRIDE_EVAL_EVERY_LIST=()              # task ごとの override
OVERRIDE_SAVE_VIZ_EVERY=""               # 全 task 共通 override
OVERRIDE_SAVE_VIZ_EVERY_LIST=()          # task ごとの override
NUM_RANK_EVAL_BATCHES=32
SAVE_STEPS=5000
NUM_WORKERS=2
DEBUG_MODE="normal"

# ---- 学習後の可視化 ---------------------------------------------
RUN_VIS_AFTER_TRAIN=true       # true にすると各条件の学習後に visualize_eval.sh を自動実行
VIS_SPLIT="heldout"             # 可視化時のデータ split

# ---- アブレーション条件 -----------------------------------------
# フォーマット: "ラベル|model_variant|focus_on|dino_feat_loss|focus_target|rank_score|neg_mode"
#
# 各フィールドの意味:
#   label        : run_name に使う短い識別子
#   model_variant: full | no_focus | no_dino_loss | pixel_focus | image_rank
#   focus_on     : true | false (FocusHead を使うか)
#   dino_feat    : true | false (DINO feature 損失を使うか)
#   focus_target : dino_diff | pixel_diff | mixed
#   rank_score   : dino_only | image_only | combined
#   neg_mode     : all | noise | shuffle | roll

ABLATION_CONDITIONS=(
  # ---- Axis 1: Focus あり vs なし --------------------------------
  "full|full|true|true|dino_diff|dino_only|all"
  "nofocus|no_focus|false|true|dino_diff|dino_only|all"

  # ---- Axis 2: DINO 特徴損失 あり vs なし -----------------------
  "nodino|no_dino_loss|true|false|dino_diff|dino_only|all"
  # "full" は Axis 1 の結果を再利用

  # ---- Axis 3: Focus supervision target -------------------------
  "pixfocus|pixel_focus|true|true|pixel_diff|dino_only|all"
  "mixfocus|full|true|true|mixed|dino_only|all"

  # ---- Axis 4: Ranking score type --------------------------------
  "imgrank|image_rank|true|true|dino_diff|image_only|all"
  "combrank|full|true|true|dino_diff|combined|all"

  # ---- Axis 5: Negative mode ------------------------------------
  "negnoise|full|true|true|dino_diff|dino_only|noise"
  "negshuffle|full|true|true|dino_diff|dino_only|shuffle"
)

# ---- 実行制御 ---------------------------------------------------
DRY_RUN="${DRY_RUN:-false}"     # 外部から DRY_RUN=true で上書き可
START_IDX="${START_IDX:-0}"     # 外部から START_IDX=3 で途中再開可

# ---- パス -------------------------------------------------------
# 環境変数で上書き可能:
#   LIBERO_DATA_ROOT=/localdata/modified_libero_rlds
#   LOCALDATA_ROOT=/localdata
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}/checkpoints/libero/FocusedWM/ablation/${ABLATION_GROUP}"

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

NPROC=$(detect_gpu_count)
SEGMENT_LENGTH=$(( ACTION_HORIZON + 1 ))
GLOBAL_BATCH_SIZE=$(( BATCH_SIZE * NPROC * GRAD_ACCUM ))
LOG_DIR="${REPO_ROOT}/logs/libero/FocusedWM/ablation/${ABLATION_GROUP}"
ensure_dir "${LOG_DIR}"
ensure_dir "${OUTPUT_ROOT}"

TASKS_TO_RUN=()
if (( ${#TASK_NAME_LIST[@]} > 0 )); then
  TASKS_TO_RUN=("${TASK_NAME_LIST[@]}")
else
  TASKS_TO_RUN=("${TASK_NAME}")
fi

if (( ${#TASK_SCHEDULE_KEY_LIST[@]} > 0 )) && (( ${#TASK_SCHEDULE_KEY_LIST[@]} != ${#TASKS_TO_RUN[@]} )); then
  echo "[ERROR] TASK_SCHEDULE_KEY_LIST length must match TASK_NAME_LIST length." >&2
  exit 1
fi
if (( ${#OVERRIDE_MAX_STEPS_LIST[@]} > 0 )) && (( ${#OVERRIDE_MAX_STEPS_LIST[@]} != ${#TASKS_TO_RUN[@]} )); then
  echo "[ERROR] OVERRIDE_MAX_STEPS_LIST length must match TASK_NAME_LIST length." >&2
  exit 1
fi
if (( ${#OVERRIDE_EVAL_EVERY_LIST[@]} > 0 )) && (( ${#OVERRIDE_EVAL_EVERY_LIST[@]} != ${#TASKS_TO_RUN[@]} )); then
  echo "[ERROR] OVERRIDE_EVAL_EVERY_LIST length must match TASK_NAME_LIST length." >&2
  exit 1
fi
if (( ${#OVERRIDE_SAVE_VIZ_EVERY_LIST[@]} > 0 )) && (( ${#OVERRIDE_SAVE_VIZ_EVERY_LIST[@]} != ${#TASKS_TO_RUN[@]} )); then
  echo "[ERROR] OVERRIDE_SAVE_VIZ_EVERY_LIST length must match TASK_NAME_LIST length." >&2
  exit 1
fi

TASK_DISPLAY_NAMES=()
TASK_SUITE_NAMES=()
TASK_SCHEDULE_KEYS=()
TASK_MAX_STEPS=()
TASK_EVAL_EVERY=()
TASK_SAVE_VIZ_EVERY=()

for idx in "${!TASKS_TO_RUN[@]}"; do
  task_name="${TASKS_TO_RUN[$idx]}"
  validate_task_name "${task_name}"

  schedule_key_override=""
  if (( ${#TASK_SCHEDULE_KEY_LIST[@]} > 0 )); then
    schedule_key_override="${TASK_SCHEDULE_KEY_LIST[$idx]}"
  fi
  task_schedule_key="$(resolve_task_schedule_key "${task_name}" "${schedule_key_override}")"

  max_steps_override="${OVERRIDE_MAX_STEPS:-}"
  eval_every_override="${OVERRIDE_EVAL_EVERY:-}"
  vis_every_override="${OVERRIDE_SAVE_VIZ_EVERY:-}"
  if (( ${#OVERRIDE_MAX_STEPS_LIST[@]} > 0 )); then
    max_steps_override="${OVERRIDE_MAX_STEPS_LIST[$idx]}"
  fi
  if (( ${#OVERRIDE_EVAL_EVERY_LIST[@]} > 0 )); then
    eval_every_override="${OVERRIDE_EVAL_EVERY_LIST[$idx]}"
  fi
  if (( ${#OVERRIDE_SAVE_VIZ_EVERY_LIST[@]} > 0 )); then
    vis_every_override="${OVERRIDE_SAVE_VIZ_EVERY_LIST[$idx]}"
  fi

  TASK_DISPLAY_NAMES+=("${task_name}")
  TASK_SUITE_NAMES+=("$(normalize_libero_task_name "${task_name}")")
  TASK_SCHEDULE_KEYS+=("${task_schedule_key}")
  TASK_MAX_STEPS+=("$(resolve_task_schedule_value get_max_steps_for_task "${task_schedule_key}" "${max_steps_override}")")
  TASK_EVAL_EVERY+=("$(resolve_task_schedule_value get_eval_every_for_task "${task_schedule_key}" "${eval_every_override}")")
  TASK_SAVE_VIZ_EVERY+=("$(resolve_task_schedule_value get_vis_every_for_task "${task_schedule_key}" "${vis_every_override}")")
done

# ---------------------------------------------------------------------------
# Ablation header
# ---------------------------------------------------------------------------
print_header "Ablation: ${ABLATION_GROUP}"
echo "${ABLATION_DESC}"
TOTAL_CONDITIONS=$(( ${#TASK_DISPLAY_NAMES[@]} * ${#ABLATION_CONDITIONS[@]} ))
printf "  %-25s %s\n" "Tasks"            "${TASK_DISPLAY_NAMES[*]}"
printf "  %-25s %s\n" "DINO backbone"    "${DINO_BACKBONE}"
printf "  %-25s %s\n" "LR / Batch(eff.)" "${LR} / ${GLOBAL_BATCH_SIZE}"
printf "  %-25s %d\n" "N conditions"     "${TOTAL_CONDITIONS}"
printf "  %-25s %s\n" "Dataset root"     "${DATASET_PATH}"
printf "  %-25s %s\n" "Output root"      "${OUTPUT_ROOT}"
printf '%*s\n' 68 '' | tr ' ' '-'
printf "  %-10s  %-10s  %-10s  %-10s\n" "TASK" "MAX" "EVAL" "VIS"
for task_idx in "${!TASK_DISPLAY_NAMES[@]}"; do
  printf "  %-10s  %-10s  %-10s  %-10s\n" \
    "${TASK_DISPLAY_NAMES[$task_idx]}" \
    "${TASK_MAX_STEPS[$task_idx]}" \
    "${TASK_EVAL_EVERY[$task_idx]}" \
    "${TASK_SAVE_VIZ_EVERY[$task_idx]}"
done
printf '%*s\n' 68 '' | tr ' ' '-'
echo ""
echo "  Conditions:"
condition_counter=0
for task_idx in "${!TASK_DISPLAY_NAMES[@]}"; do
  for i in "${!ABLATION_CONDITIONS[@]}"; do
    IFS='|' read -r label variant focus_on dino_feat focus_target rank_score neg_mode \
      <<< "${ABLATION_CONDITIONS[$i]}"
    printf "    [%d] task=%-8s label=%-12s variant=%-14s focus=%-5s dino_feat=%-5s tgt=%-9s rank=%-10s neg=%s\n" \
      "${condition_counter}" "${TASK_DISPLAY_NAMES[$task_idx]}" "${label}" "${variant}" "${focus_on}" "${dino_feat}" \
      "${focus_target}" "${rank_score}" "${neg_mode}"
    condition_counter=$(( condition_counter + 1 ))
  done
done
echo ""

# Save ablation manifest
{
  echo "ablation_group=${ABLATION_GROUP}"
  echo "tasks=${TASK_DISPLAY_NAMES[*]}"
  echo "task_schedules:"
  for task_idx in "${!TASK_DISPLAY_NAMES[@]}"; do
    echo "  ${TASK_DISPLAY_NAMES[$task_idx]}|suite=${TASK_SUITE_NAMES[$task_idx]}|schedule=${TASK_SCHEDULE_KEYS[$task_idx]}|max=${TASK_MAX_STEPS[$task_idx]}|eval=${TASK_EVAL_EVERY[$task_idx]}|vis=${TASK_SAVE_VIZ_EVERY[$task_idx]}"
  done
  echo "dataset_path=${DATASET_PATH}"
  echo "started=$(timestamp)"
  echo ""
  echo "# ${ABLATION_DESC}"
  echo ""
  echo "# conditions:"
  for c in "${ABLATION_CONDITIONS[@]}"; do echo "  ${c}"; done
} > "${OUTPUT_ROOT}/ablation_manifest.txt"
copy_script_snapshot "${OUTPUT_ROOT}" "${BASH_SOURCE[0]}"

# ---------------------------------------------------------------------------
# Ablation loop
# ---------------------------------------------------------------------------
FAILED_CONDITIONS=()
ALL_ABLATION_RUNS=()

for task_idx in "${!TASK_DISPLAY_NAMES[@]}"; do
  for cond_idx in "${!ABLATION_CONDITIONS[@]}"; do
    ALL_ABLATION_RUNS+=("${task_idx}|${cond_idx}")
  done
done

for i in "${!ALL_ABLATION_RUNS[@]}"; do
  if (( i < START_IDX )); then continue; fi

  IFS='|' read -r task_idx cond_idx <<< "${ALL_ABLATION_RUNS[$i]}"
  IFS='|' read -r label variant focus_on dino_feat focus_target rank_score neg_mode \
    <<< "${ABLATION_CONDITIONS[$cond_idx]}"

  TASK_LABEL="${TASK_DISPLAY_NAMES[$task_idx]}"
  TASK_SUITE="${TASK_SUITE_NAMES[$task_idx]}"
  TASK_SCHEDULE_KEY="${TASK_SCHEDULE_KEYS[$task_idx]}"
  MAX_STEPS="${TASK_MAX_STEPS[$task_idx]}"
  EVAL_EVERY="${TASK_EVAL_EVERY[$task_idx]}"
  SAVE_VIZ_EVERY="${TASK_SAVE_VIZ_EVERY[$task_idx]}"

  RUN_NAME="$(make_run_name \
    "${variant}" \
    "${DINO_BACKBONE}" \
    "${LR}" \
    "${BATCH_SIZE}" \
    "${ACTION_HORIZON}" \
    "${label}")"

  SAVE_DIR="${OUTPUT_ROOT}/${TASK_LABEL}/s${SEED}/${RUN_NAME}"
  LOGFILE="${LOG_DIR}/${TASK_LABEL}/s${SEED}/${RUN_NAME}.log"

  echo "  [$((i+1))/${#ALL_ABLATION_RUNS[@]}] task=${TASK_LABEL}  ${label}: variant=${variant}  focus=${focus_on}  dino_feat=${dino_feat}  target=${focus_target}  rank=${rank_score}  neg=${neg_mode}"
  echo "    schedule: key=${TASK_SCHEDULE_KEY}  max=${MAX_STEPS}  eval=${EVAL_EVERY}  vis=${SAVE_VIZ_EVERY}"
  echo "    → ${SAVE_DIR}"

  if [ "${DRY_RUN}" = "true" ]; then
    echo "    [DRY_RUN] Skipping."
    continue
  fi

  ensure_dir "${SAVE_DIR}"
  dump_run_config "${SAVE_DIR}" \
    "ablation_group=${ABLATION_GROUP}" \
    "ablation_label=${label}" \
    "condition_idx=${i}" \
    "task_name=${TASK_LABEL}" \
    "task_suite=${TASK_SUITE}" \
    "task_schedule_key=${TASK_SCHEDULE_KEY}" \
    "model_variant=${variant}" \
    "focus_on=${focus_on}" \
    "dino_feature_loss=${dino_feat}" \
    "focus_target=${focus_target}" \
    "ranking_score_type=${rank_score}" \
    "negative_mode=${neg_mode}" \
    "dataset_path=${DATASET_PATH}" \
    "dino_backbone=${DINO_BACKBONE}" \
    "lr=${LR}" \
    "batch_size=${BATCH_SIZE}" \
    "action_horizon=${ACTION_HORIZON}" \
    "max_steps=${MAX_STEPS}" \
    "eval_every=${EVAL_EVERY}" \
    "vis_every=${SAVE_VIZ_EVERY}" \
    "seed=${SEED}" \
    "run_name=${RUN_NAME}"

  # Build command
  TRAIN_ARGS=(
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
    --warmup-ratio 0.02
    --weight-decay 0.0
    --max-grad-norm 1.0
    --save-steps "${SAVE_STEPS}"
    --logging-steps 10
    --save-total-limit 2
    --eval-every "${EVAL_EVERY}"
    --save-viz-every "${SAVE_VIZ_EVERY}"
    --num-rank-eval-batches "${NUM_RANK_EVAL_BATCHES}"
    --debug-mode "${DEBUG_MODE}"
    --seed "${SEED}"
    --dino-model-name "${DINO_BACKBONE}"
    --dino-input-size "${DINO_INPUT_SIZE}"
    --image-height "${IMAGE_HEIGHT}"
    --image-width "${IMAGE_WIDTH}"
    --model-variant "${variant}"
    --recon-loss-weight "${RECON_LOSS_WEIGHT}"
    --dino-feature-loss-weight "${DINO_FEATURE_LOSS_WEIGHT}"
    --focus-supervision-weight "${FOCUS_SUPERVISION_WEIGHT}"
    --focus-sparsity-mode "l1"
    --focus-sparsity-weight "${FOCUS_SPARSITY_WEIGHT}"
    --ranking-score-type "${rank_score}"
    --negative-mode "${neg_mode}"
    --noise-std "${NOISE_STD}"
    --num-action-candidates "${NUM_CANDIDATES}"
    --run-rank-eval
  )

  # DINO frozen
  if [ "${DINO_FROZEN}" = "true" ]; then TRAIN_ARGS+=(--dino-frozen)
  else TRAIN_ARGS+=(--dino-no-frozen); fi

  # Focus head
  if [ "${focus_on}" = "true" ]; then TRAIN_ARGS+=(--use-focus-head)
  else TRAIN_ARGS+=(--no-focus-head --no-dino-focus-supervision --no-focus-sparsity); fi

  # DINO feature loss
  if [ "${dino_feat}" = "true" ]; then TRAIN_ARGS+=(--use-dino-feature-loss)
  else TRAIN_ARGS+=(--no-dino-feature-loss); fi

  # Focus supervision target
  if [ "${focus_on}" = "true" ]; then
    case "${focus_target}" in
      dino_diff)  TRAIN_ARGS+=(--use-dino-focus-supervision) ;;
      pixel_diff) TRAIN_ARGS+=(--no-dino-focus-supervision --use-pixel-focus-supervision) ;;
      mixed)      TRAIN_ARGS+=(--use-dino-focus-supervision --use-pixel-focus-supervision) ;;
    esac
    TRAIN_ARGS+=(--use-focus-sparsity)
  fi

  save_cmd "${SAVE_DIR}" python "${TRAIN_ARGS[@]}"

  set +e
  run_train_command "${NPROC}" "${SAVE_DIR}" "${LOGFILE}" "${TRAIN_ARGS[@]}"
  EXIT_CODE=$?
  set -e

  if (( EXIT_CODE != 0 )); then
    echo "    [FAIL] ${TASK_LABEL}/${label}"
    FAILED_CONDITIONS+=("${i}: ${TASK_LABEL}/${label}")
    echo "FAILED" > "${SAVE_DIR}/FAILED"
  else
    echo "    [OK] ${TASK_LABEL}/${label}"
    echo "DONE" > "${SAVE_DIR}/DONE"
    if [ "${RUN_VIS_AFTER_TRAIN}" = "true" ]; then
      CKPT="$(find_checkpoint "${SAVE_DIR}" auto)"
      if [ -n "${CKPT}" ]; then
        bash "${SCRIPT_DIR}/visualize_eval.sh" \
          RUN_DIR="${SAVE_DIR}" \
          CHECKPOINT_PATH="${CKPT}" \
          TASK_NAME="${TASK_LABEL}" \
          SPLIT="${VIS_SPLIT}" \
          OUTPUT_DIR="${SAVE_DIR}/full_eval"
      else
        echo "    [WARN] No checkpoint found; skipping post-training visualization."
      fi
    fi
  fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
print_header "Ablation complete: ${ABLATION_GROUP}"
printf "  Conditions: %d\n" "${#ALL_ABLATION_RUNS[@]}"
if (( ${#FAILED_CONDITIONS[@]} > 0 )); then
  printf "  Failed:\n"
  for f in "${FAILED_CONDITIONS[@]}"; do printf "    %s\n" "${f}"; done
fi
printf "  Results: %s\n" "${OUTPUT_ROOT}"
printf "  Collect: bash %s/collect_results.sh\n" "${SCRIPT_DIR}"
echo ""
echo "  Collect results:"
echo "    bash ${SCRIPT_DIR}/collect_results.sh --output-root ${OUTPUT_ROOT}"
