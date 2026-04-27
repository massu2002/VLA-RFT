#!/usr/bin/env bash
# train_sweep.sh — FocusedWM sweep runner (mode-aware)
#
# 使い方:
#   1. USER CONFIG を編集
#   2. bash scripts/libero/residual_worldmodel/train_sweep.sh
#      (USER CONFIG の TASK_NAME_LIST を使用)
#
# task を CLI で上書き:
#   bash scripts/libero/residual_worldmodel/train_sweep.sh spatial goal
#
# 確認モード:
#   LIST_ONLY=true bash scripts/libero/residual_worldmodel/train_sweep.sh goal long
#   DRY_RUN=true   bash scripts/libero/residual_worldmodel/train_sweep.sh object
#   MAX_RUNS=5     bash scripts/libero/residual_worldmodel/train_sweep.sh spatial

set -euo pipefail

######################################################################
########## USER CONFIG — ここだけ書き換えて使う ##########
######################################################################

# ---- Sweep identity -------------------------------------------------
SWEEP_NAME="baseline"
SWEEP_MODE="baseline_core"   # baseline_core | focus_ablation | ranking_ablation | model_ablation | robustness

# ---- Paths ----------------------------------------------------------
# 環境変数で上書き可能:
#   LIBERO_DATA_ROOT=/localdata/modified_libero_rlds
#   LOCALDATA_ROOT=/localdata
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}/checkpoints/libero/FocusedWM/sweep/${SWEEP_NAME}"

# ---- Shared fixed defaults (modeで必要に応じて上書き) --------------
TASK_NAME=""
TASK_NAME_LIST=("spatial" "object" "goal" "long")
TASK_SCHEDULE_KEY_LIST=()
OVERRIDE_MAX_STEPS=""
OVERRIDE_MAX_STEPS_LIST=()
OVERRIDE_EVAL_EVERY=""
OVERRIDE_EVAL_EVERY_LIST=()
OVERRIDE_SAVE_VIZ_EVERY=""
OVERRIDE_SAVE_VIZ_EVERY_LIST=()

LR_LIST=("1e-4")
BATCH_SIZE_LIST=(4)
ACTION_HORIZON_LIST=(7)
MODEL_VARIANT_LIST=("full")
DINO_BACKBONE_LIST=("dinov2_vits14")
DINO_FROZEN_LIST=(true)
DINO_INPUT_SIZE_LIST=(224)
IMAGE_HEIGHT_LIST=(256)
PRECISION_LIST=("bf16")

RECON_LOSS_WEIGHT_LIST=("1.0")
DINO_FEATURE_LOSS_WEIGHT_LIST=("0.1")
FOCUS_SUPERVISION_TYPE_LIST=("dino_diff")
FOCUS_SUPERVISION_WEIGHT_LIST=("0.1")
USE_FOCUS_SPARSITY_LIST=(true)
FOCUS_SPARSITY_MODE_LIST=("l1")
FOCUS_SPARSITY_WEIGHT_LIST=("0.01")

RANKING_SCORE_TYPE_LIST=("dino_only") # alias許可: recon_only->image_only, mixed->combined
NEGATIVE_MODE_LIST=("all")
NUM_CANDIDATES_LIST=(4)
SEED_LIST=(42)

# ---- Mode presets ---------------------------------------------------
# 使わない軸は length=1 の配列に固定して条件爆発を防ぐ。
case "${SWEEP_MODE}" in
  baseline_core)
    TASK_NAME_LIST=("spatial" "object" "goal" "long")
    LR_LIST=("1e-4" "3e-4" "1e-3")
    BATCH_SIZE_LIST=(2 4 8)
    ACTION_HORIZON_LIST=(4 7 10)
    DINO_FEATURE_LOSS_WEIGHT_LIST=("0.05" "0.1" "0.5" "1.0")
    RANKING_SCORE_TYPE_LIST=("dino_only" "mixed")
    SEED_LIST=(42)
    ;;
  focus_ablation)
    TASK_NAME_LIST=("spatial" "object" "goal" "long")
    FOCUS_SUPERVISION_TYPE_LIST=("dino_diff" "pixel_diff" "mixed")
    FOCUS_SUPERVISION_WEIGHT_LIST=("0.03" "0.1" "0.3")
    USE_FOCUS_SPARSITY_LIST=(true false)
    FOCUS_SPARSITY_MODE_LIST=("l1" "entropy")
    FOCUS_SPARSITY_WEIGHT_LIST=("0.001" "0.01" "0.05")
    ;;
  ranking_ablation)
    TASK_NAME_LIST=("spatial" "object" "goal" "long")
    NEGATIVE_MODE_LIST=("noise" "shuffle" "all")
    NUM_CANDIDATES_LIST=(2 4 8)
    RANKING_SCORE_TYPE_LIST=("dino_only" "recon_only" "mixed")
    ;;
  model_ablation)
    TASK_NAME_LIST=("spatial" "object" "goal" "long")
    MODEL_VARIANT_LIST=("baseline" "full")
    DINO_FROZEN_LIST=(true false)
    RECON_LOSS_WEIGHT_LIST=("0.5" "1.0")
    ;;
  robustness)
    TASK_NAME_LIST=("spatial" "object" "goal" "long")
    SEED_LIST=(42 43 44)
    ;;
  *)
    echo "[ERROR] Unsupported SWEEP_MODE='${SWEEP_MODE}'" >&2
    exit 1
    ;;
esac

# ---- Other fixed settings ------------------------------------------
IMAGE_WIDTH=256
GRAD_ACCUM=1
WARMUP_RATIO=0.02
WEIGHT_DECAY=0.0
MAX_GRAD_NORM=1.0

FOCUS_SUPERVISION_WEIGHT_DEFAULT="0.1"
FOCUS_SPARSITY_WEIGHT_DEFAULT="0.01"
NUM_CANDIDATES_DEFAULT=4
NEGATIVE_MODE_DEFAULT="all"

NUM_RANK_EVAL_BATCHES=32
SAVE_STEPS=5000
SAVE_TOTAL_LIMIT=3
NUM_WORKERS=2
DEBUG_MODE="normal"
NOISE_STD=0.05

RUN_VIS_AFTER_TRAIN=true
VIS_SPLIT="heldout"

# ---- Execution control ---------------------------------------------
MAX_RUNS="${MAX_RUNS:-0}"     # 0=unlimited
DRY_RUN="${DRY_RUN:-false}"
LIST_ONLY="${LIST_ONLY:-false}"
START_IDX="${START_IDX:-0}"

######################################################################
########## END USER CONFIG ##########
######################################################################

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"
setup_env
DATASET_PATH="$(resolve_libero_data_root "${DATASET_PATH:-}")"

# ---------------------------------------------------------------------------
# CLI task override (TASK_NAME_LIST only)
# ---------------------------------------------------------------------------
CLI_TASK_NAME_LIST=()
while (( $# > 0 )); do
  case "${1}" in
    --tasks)
      shift
      continue
      ;;
    -h|--help)
      cat <<'USAGE'
Usage:
  bash scripts/libero/residual_worldmodel/train_sweep.sh [task...]

Examples:
  bash scripts/libero/residual_worldmodel/train_sweep.sh
  bash scripts/libero/residual_worldmodel/train_sweep.sh spatial goal
  LIST_ONLY=true bash scripts/libero/residual_worldmodel/train_sweep.sh spatial long
  DRY_RUN=true bash scripts/libero/residual_worldmodel/train_sweep.sh object
  MAX_RUNS=5 bash scripts/libero/residual_worldmodel/train_sweep.sh spatial object
USAGE
      exit 0
      ;;
    --*)
      echo "[ERROR] Unknown option: ${1}" >&2
      echo "[ERROR] Pass tasks as positional args. Example: .../train_sweep.sh spatial goal" >&2
      exit 1
      ;;
    *)
      CLI_TASK_NAME_LIST+=("${1}")
      shift
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

normalize_model_variant() {
  local in="${1:-full}"
  case "${in}" in
    full|no_focus|no_dino_loss|pixel_focus|image_rank) echo "${in}" ;;
    baseline)
      echo "no_focus"
      ;;
    *)
      echo "[ERROR] Unsupported model variant: ${in}" >&2
      return 1
      ;;
  esac
}

normalize_ranking_score_type() {
  local in="${1:-dino_only}"
  case "${in}" in
    dino_only|image_only|combined) echo "${in}" ;;
    recon_only) echo "image_only" ;;
    mixed) echo "combined" ;;
    *)
      echo "[ERROR] Unsupported ranking score type: ${in}" >&2
      return 1
      ;;
  esac
}

short_bool() {
  if [ "${1}" = "true" ]; then echo "t"; else echo "f"; fi
}

warn_mode_aliases() {
  local warned=false
  for mv in "${MODEL_VARIANT_LIST[@]}"; do
    if [ "${mv}" = "baseline" ]; then
      echo "[WARN] model_variant='baseline' is mapped to CLI variant 'no_focus'."
      warned=true
      break
    fi
  done
  for rt in "${RANKING_SCORE_TYPE_LIST[@]}"; do
    case "${rt}" in
      recon_only)
        echo "[WARN] ranking_score_type='recon_only' is mapped to CLI 'image_only'."
        warned=true
        ;;
      mixed)
        echo "[WARN] ranking_score_type='mixed' is mapped to CLI 'combined'."
        warned=true
        ;;
    esac
  done
  if [ "${warned}" = "true" ]; then
    echo "[WARN] alias mapping is recorded in LIST_ONLY, run_name, and config_dump."
  fi
}

# ---------------------------------------------------------------------------
# Resolve task schedule per task
# ---------------------------------------------------------------------------
NPROC=$(detect_gpu_count)
LOG_DIR="${REPO_ROOT}/logs/libero/FocusedWM/sweep/${SWEEP_NAME}"
ensure_dir "${LOG_DIR}"
ensure_dir "${OUTPUT_ROOT}"

TASKS_TO_RUN=()
TASK_SOURCE="USER CONFIG"
CLI_TASK_COUNT=${#CLI_TASK_NAME_LIST[@]}
if (( ${#CLI_TASK_NAME_LIST[@]} > 0 )); then
  TASKS_TO_RUN=("${CLI_TASK_NAME_LIST[@]}")
  TASK_SOURCE="CLI args"
elif (( ${#TASK_NAME_LIST[@]} > 0 )); then
  TASKS_TO_RUN=("${TASK_NAME_LIST[@]}")
else
  TASKS_TO_RUN=("${TASK_NAME}")
  TASK_SOURCE="TASK_NAME fallback"
fi

if (( ${#TASKS_TO_RUN[@]} == 0 )) || { (( ${#TASKS_TO_RUN[@]} == 1 )) && [ -z "${TASKS_TO_RUN[0]}" ]; }; then
  echo "[ERROR] No tasks resolved. Provide CLI tasks or set TASK_NAME_LIST/TASK_NAME in USER CONFIG." >&2
  exit 1
fi

check_task_list_length() {
  local var_name="${1:?'var name required'}"
  local count="${2:?'count required'}"
  local expected="${3:?'expected required'}"
  if (( count > 0 && count != expected )); then
    echo "[ERROR] ${var_name} length mismatch." >&2
    echo "        task_source=${TASK_SOURCE}" >&2
    echo "        cli_task_count=${CLI_TASK_COUNT}" >&2
    echo "        resolved_task_count=${expected}" >&2
    echo "        ${var_name}_count=${count}" >&2
    exit 1
  fi
}

check_task_list_length "TASK_SCHEDULE_KEY_LIST" "${#TASK_SCHEDULE_KEY_LIST[@]}" "${#TASKS_TO_RUN[@]}"
check_task_list_length "OVERRIDE_MAX_STEPS_LIST" "${#OVERRIDE_MAX_STEPS_LIST[@]}" "${#TASKS_TO_RUN[@]}"
check_task_list_length "OVERRIDE_EVAL_EVERY_LIST" "${#OVERRIDE_EVAL_EVERY_LIST[@]}" "${#TASKS_TO_RUN[@]}"
check_task_list_length "OVERRIDE_SAVE_VIZ_EVERY_LIST" "${#OVERRIDE_SAVE_VIZ_EVERY_LIST[@]}" "${#TASKS_TO_RUN[@]}"

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

warn_mode_aliases

# ---------------------------------------------------------------------------
# Build conditions (mode controls effective lists)
# ---------------------------------------------------------------------------
ALL_CONDITIONS=()
for task_idx in "${!TASK_DISPLAY_NAMES[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for bs in "${BATCH_SIZE_LIST[@]}"; do
      for hor in "${ACTION_HORIZON_LIST[@]}"; do
        for mv in "${MODEL_VARIANT_LIST[@]}"; do
          for dino_bb in "${DINO_BACKBONE_LIST[@]}"; do
            for dino_in in "${DINO_INPUT_SIZE_LIST[@]}"; do
              for img_h in "${IMAGE_HEIGHT_LIST[@]}"; do
                for prec in "${PRECISION_LIST[@]}"; do
                  for dino_frozen in "${DINO_FROZEN_LIST[@]}"; do
                    for recon_w in "${RECON_LOSS_WEIGHT_LIST[@]}"; do
                      for dino_w in "${DINO_FEATURE_LOSS_WEIGHT_LIST[@]}"; do
                        for focus_type in "${FOCUS_SUPERVISION_TYPE_LIST[@]}"; do
                          for focus_w in "${FOCUS_SUPERVISION_WEIGHT_LIST[@]}"; do
                            for use_sp in "${USE_FOCUS_SPARSITY_LIST[@]}"; do
                              for sp_mode in "${FOCUS_SPARSITY_MODE_LIST[@]}"; do
                                for sp_w in "${FOCUS_SPARSITY_WEIGHT_LIST[@]}"; do
                                  for rank in "${RANKING_SCORE_TYPE_LIST[@]}"; do
                                    for neg in "${NEGATIVE_MODE_LIST[@]}"; do
                                      for ncan in "${NUM_CANDIDATES_LIST[@]}"; do
                                        for seed in "${SEED_LIST[@]}"; do
                                          ALL_CONDITIONS+=("${task_idx}|${lr}|${bs}|${hor}|${mv}|${dino_bb}|${dino_in}|${img_h}|${prec}|${dino_frozen}|${recon_w}|${dino_w}|${focus_type}|${focus_w}|${use_sp}|${sp_mode}|${sp_w}|${rank}|${neg}|${ncan}|${seed}")
                                        done
                                      done
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

TOTAL=${#ALL_CONDITIONS[@]}

if (( TOTAL > 300 )); then
  echo "[STRONG WARN] Sweep conditions=${TOTAL}. Consider narrowing lists or using START_IDX/MAX_RUNS."
elif (( TOTAL > 100 )); then
  echo "[WARN] Sweep conditions=${TOTAL}. Consider LIST_ONLY/DRY_RUN before full launch."
fi

# ---------------------------------------------------------------------------
# LIST_ONLY
# ---------------------------------------------------------------------------
if [ "${LIST_ONLY}" = "true" ]; then
  print_header "Sweep: ${SWEEP_NAME}  mode=${SWEEP_MODE}  (${TOTAL} conditions)"
  printf "  %-25s %s\n" "Task source" "${TASK_SOURCE}"
  printf "  %-25s %s\n" "Tasks" "${TASK_DISPLAY_NAMES[*]}"
  printf "  %-4s %-8s %-8s %-8s %-5s %-5s %-10s %-7s %-6s %-5s %-8s %-8s %-3s %-7s %-7s %-6s %-6s\n" \
    "IDX" "task" "max" "lr" "batch" "hor" "model" "dino_w" "f_type" "f_w" "sp_on" "sp_mode" "sp_w" "rank" "neg" "cand" "seed"
  printf "  %s\n" "$(printf '%0.s-' {1..175})"

  for i in "${!ALL_CONDITIONS[@]}"; do
    IFS='|' read -r task_idx lr bs hor mv dino_bb dino_in img_h prec dino_frozen recon_w dino_w focus_type focus_w use_sp sp_mode sp_w rank neg ncan seed <<< "${ALL_CONDITIONS[$i]}"
    printf "  %-4d %-8s %-8s %-8s %-5s %-5s %-10s %-7s %-6s %-5s %-8s %-8s %-3s %-7s %-7s %-6s %-6s\n" \
      "${i}" \
      "${TASK_DISPLAY_NAMES[$task_idx]}" \
      "${TASK_MAX_STEPS[$task_idx]}" \
      "${lr}" "${bs}" "${hor}" "${mv}" "${dino_w}" "${focus_type}" "${focus_w}" \
      "${use_sp}" "${sp_mode}" "${sp_w}" "${rank}" "${neg}" "${ncan}" "${seed}"
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Header + manifest
# ---------------------------------------------------------------------------
print_header "Sweep: ${SWEEP_NAME}"
printf "  %-25s %s\n" "SWEEP_MODE" "${SWEEP_MODE}"
printf "  %-25s %s\n" "Task source" "${TASK_SOURCE}"
printf "  %-25s %s\n" "Tasks" "${TASK_DISPLAY_NAMES[*]}"
printf "  %-25s %s\n" "NPROC" "${NPROC}"
printf "  %-25s %s\n" "Total conditions" "${TOTAL}"
printf "  %-25s %s\n" "DRY_RUN" "${DRY_RUN}"
printf "  %-25s %s\n" "START_IDX" "${START_IDX}"
printf "  %-25s %s\n" "MAX_RUNS" "${MAX_RUNS} (0=unlimited)"
printf "  %-25s %s\n" "Dataset root" "${DATASET_PATH}"
printf "  %-25s %s\n" "Output root" "${OUTPUT_ROOT}"
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

{
  echo "sweep_name=${SWEEP_NAME}"
  echo "sweep_mode=${SWEEP_MODE}"
  echo "task_source=${TASK_SOURCE}"
  echo "tasks=${TASK_DISPLAY_NAMES[*]}"
  echo "total_conditions=${TOTAL}"
  echo "dataset_path=${DATASET_PATH}"
  echo "started=$(timestamp)"
  echo "task_schedules:"
  for task_idx in "${!TASK_DISPLAY_NAMES[@]}"; do
    echo "  ${TASK_DISPLAY_NAMES[$task_idx]}|suite=${TASK_SUITE_NAMES[$task_idx]}|schedule=${TASK_SCHEDULE_KEYS[$task_idx]}|max=${TASK_MAX_STEPS[$task_idx]}|eval=${TASK_EVAL_EVERY[$task_idx]}|vis=${TASK_SAVE_VIZ_EVERY[$task_idx]}"
  done
  echo "list_sizes:"
  echo "  tasks=${#TASK_DISPLAY_NAMES[@]} lr=${#LR_LIST[@]} bs=${#BATCH_SIZE_LIST[@]} hor=${#ACTION_HORIZON_LIST[@]} model=${#MODEL_VARIANT_LIST[@]}"
  echo "  dino_backbone=${#DINO_BACKBONE_LIST[@]} dino_input=${#DINO_INPUT_SIZE_LIST[@]} image_h=${#IMAGE_HEIGHT_LIST[@]} precision=${#PRECISION_LIST[@]}"
  echo "  dino_frozen=${#DINO_FROZEN_LIST[@]} recon_w=${#RECON_LOSS_WEIGHT_LIST[@]} dino_w=${#DINO_FEATURE_LOSS_WEIGHT_LIST[@]}"
  echo "  focus_type=${#FOCUS_SUPERVISION_TYPE_LIST[@]} focus_w=${#FOCUS_SUPERVISION_WEIGHT_LIST[@]} use_sp=${#USE_FOCUS_SPARSITY_LIST[@]} sp_mode=${#FOCUS_SPARSITY_MODE_LIST[@]} sp_w=${#FOCUS_SPARSITY_WEIGHT_LIST[@]}"
  echo "  rank=${#RANKING_SCORE_TYPE_LIST[@]} neg=${#NEGATIVE_MODE_LIST[@]} candidates=${#NUM_CANDIDATES_LIST[@]} seed=${#SEED_LIST[@]}"
  echo "conditions:"
  for c in "${ALL_CONDITIONS[@]}"; do echo "  ${c}"; done
} > "${OUTPUT_ROOT}/sweep_manifest.txt"
copy_script_snapshot "${OUTPUT_ROOT}" "${BASH_SOURCE[0]}"

# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
FAILED_CONDITIONS=()
RUN_COUNT=0

for i in "${!ALL_CONDITIONS[@]}"; do
  if (( i < START_IDX )); then continue; fi
  if (( MAX_RUNS > 0 && RUN_COUNT >= MAX_RUNS )); then
    echo ""
    echo "  MAX_RUNS=${MAX_RUNS} reached. Stopping."
    break
  fi

  IFS='|' read -r task_idx lr bs hor mv dino_bb dino_in img_h prec dino_frozen recon_w dino_w focus_type focus_w use_sp sp_mode sp_w rank neg ncan seed <<< "${ALL_CONDITIONS[$i]}"

  model_cli="$(normalize_model_variant "${mv}")"
  rank_cli="$(normalize_ranking_score_type "${rank}")"

  TASK_LABEL="${TASK_DISPLAY_NAMES[$task_idx]}"
  TASK_SUITE="${TASK_SUITE_NAMES[$task_idx]}"
  TASK_SCHEDULE_KEY="${TASK_SCHEDULE_KEYS[$task_idx]}"
  MAX_STEPS="${TASK_MAX_STEPS[$task_idx]}"
  EVAL_EVERY="${TASK_EVAL_EVERY[$task_idx]}"
  SAVE_VIZ_EVERY="${TASK_SAVE_VIZ_EVERY[$task_idx]}"

  SEGMENT_LENGTH=$(( hor + 1 ))
  GLOBAL_BATCH_SIZE=$(( bs * NPROC * GRAD_ACCUM ))
  IMAGE_WIDTH_CUR="${img_h}"  # square reconstruction by default

  RUN_NAME="$(make_run_name \
    "${model_cli}" \
    "${dino_bb}" \
    "${lr}" \
    "${bs}" \
    "${hor}" \
    "mode${SWEEP_MODE}" "dw${dino_w}" "ft${focus_type}" "fw${focus_w}" \
    "sp$(short_bool "${use_sp}")${sp_mode}${sp_w}" "rk${rank}" "neg${neg}" "nc${ncan}" \
    "df$(short_bool "${dino_frozen}")" "rw${recon_w}" "s${seed}")"

  SAVE_DIR="${OUTPUT_ROOT}/${TASK_LABEL}/s${seed}/${RUN_NAME}"
  LOGFILE="${LOG_DIR}/${TASK_LABEL}/s${seed}/${RUN_NAME}.log"

  echo ""
  printf "  [%d/%d] task=%s max=%s lr=%s batch=%s hor=%s model=%s dino_w=%s rank=%s seed=%s\n" \
    "$((i+1))" "${TOTAL}" "${TASK_LABEL}" "${MAX_STEPS}" "${lr}" "${bs}" "${hor}" "${mv}" "${dino_w}" "${rank}" "${seed}"
  printf "         focus=%s/%s sparsity=%s(%s,%s) neg=%s candidates=%s frozen=%s recon_w=%s\n" \
    "${focus_type}" "${focus_w}" "${use_sp}" "${sp_mode}" "${sp_w}" "${neg}" "${ncan}" "${dino_frozen}" "${recon_w}"
  printf "         Save dir: %s\n" "${SAVE_DIR}"

  if [ "${DRY_RUN}" = "true" ]; then
    echo "         [DRY_RUN] Skipping execution."
    RUN_COUNT=$(( RUN_COUNT + 1 ))
    continue
  fi

  ensure_dir "${SAVE_DIR}"
  dump_run_config "${SAVE_DIR}" \
    "sweep_name=${SWEEP_NAME}" \
    "sweep_mode=${SWEEP_MODE}" \
    "condition_idx=${i}" \
    "task_name=${TASK_LABEL}" \
    "task_suite=${TASK_SUITE}" \
    "task_schedule_key=${TASK_SCHEDULE_KEY}" \
    "max_steps=${MAX_STEPS}" \
    "lr=${lr}" \
    "batch_size=${bs}" \
    "grad_accum=${GRAD_ACCUM}" \
    "global_batch_size=${GLOBAL_BATCH_SIZE}" \
    "action_horizon=${hor}" \
    "precision=${prec}" \
    "model_variant=${mv}" \
    "model_variant_cli=${model_cli}" \
    "dino_backbone=${dino_bb}" \
    "dino_input_size=${dino_in}" \
    "dino_frozen=${dino_frozen}" \
    "image_height=${img_h}" \
    "image_width=${IMAGE_WIDTH_CUR}" \
    "recon_loss_weight=${recon_w}" \
    "dino_feature_loss_weight=${dino_w}" \
    "focus_supervision_type=${focus_type}" \
    "focus_supervision_weight=${focus_w}" \
    "use_focus_sparsity=${use_sp}" \
    "focus_sparsity_mode=${sp_mode}" \
    "focus_sparsity_weight=${sp_w}" \
    "ranking_score_type=${rank}" \
    "ranking_score_type_cli=${rank_cli}" \
    "negative_mode=${neg}" \
    "num_candidates=${ncan}" \
    "seed=${seed}" \
    "dataset_path=${DATASET_PATH}" \
    "run_name=${RUN_NAME}" \
    "save_dir=${SAVE_DIR}"

  TRAIN_ARGS=(
    -m worldmodel.residual_worldmodel.train_focused_libero
    --task-suite "${TASK_SUITE}"
    --data-root "${DATASET_PATH}"
    --output-dir "${SAVE_DIR}"
    --max-steps "${MAX_STEPS}"
    --segment-length "${SEGMENT_LENGTH}"
    --batch-size-per-device "${bs}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --precision "${prec}"
    --learning-rate "${lr}"
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
    --seed "${seed}"
    --dino-model-name "${dino_bb}"
    --dino-input-size "${dino_in}"
    --image-height "${img_h}"
    --image-width "${IMAGE_WIDTH_CUR}"
    --model-variant "${model_cli}"
    --recon-loss-weight "${recon_w}"
    --dino-feature-loss-weight "${dino_w}"
    --focus-supervision-weight "${focus_w}"
    --focus-sparsity-mode "${sp_mode}"
    --focus-sparsity-weight "${sp_w}"
    --ranking-score-type "${rank_cli}"
    --negative-mode "${neg}"
    --noise-std "${NOISE_STD}"
    --num-action-candidates "${ncan}"
    --run-rank-eval
  )

  if [ "${dino_frozen}" = "true" ]; then
    TRAIN_ARGS+=(--dino-frozen)
  else
    TRAIN_ARGS+=(--dino-no-frozen)
  fi

  if [ "${use_sp}" = "true" ]; then
    TRAIN_ARGS+=(--use-focus-sparsity)
  else
    TRAIN_ARGS+=(--no-focus-sparsity)
  fi

  case "${focus_type}" in
    dino_diff)
      TRAIN_ARGS+=(--use-dino-focus-supervision)
      ;;
    pixel_diff)
      TRAIN_ARGS+=(--no-dino-focus-supervision --use-pixel-focus-supervision)
      ;;
    mixed)
      TRAIN_ARGS+=(--use-dino-focus-supervision --use-pixel-focus-supervision)
      ;;
    *)
      echo "[ERROR] Unsupported focus supervision type: ${focus_type}" >&2
      exit 1
      ;;
  esac

  if [ "${model_cli}" = "no_dino_loss" ]; then
    TRAIN_ARGS+=(--no-dino-feature-loss)
  else
    TRAIN_ARGS+=(--use-dino-feature-loss)
  fi

  save_cmd "${SAVE_DIR}" python "${TRAIN_ARGS[@]}"

  set +e
  run_train_command "${NPROC}" "${SAVE_DIR}" "${LOGFILE}" "${TRAIN_ARGS[@]}"
  EXIT_CODE=$?
  set -e

  if (( EXIT_CODE != 0 )); then
    echo "  [FAIL] condition ${i}: ${ALL_CONDITIONS[$i]}"
    FAILED_CONDITIONS+=("${i}: ${ALL_CONDITIONS[$i]}")
    echo "FAILED" > "${SAVE_DIR}/FAILED"
  else
    echo "  [OK]   condition ${i} done."
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
        echo "  [WARN] No checkpoint found; skipping post-training visualization."
      fi
    fi
  fi

  RUN_COUNT=$(( RUN_COUNT + 1 ))
done

echo ""
print_header "Sweep complete: ${SWEEP_NAME}"
printf "  Ran: %d / %d conditions\n" "${RUN_COUNT}" "${TOTAL}"
if (( ${#FAILED_CONDITIONS[@]} > 0 )); then
  printf "  Failed (%d):\n" "${#FAILED_CONDITIONS[@]}"
  for f in "${FAILED_CONDITIONS[@]}"; do printf "    %s\n" "${f}"; done
fi
printf "  Results: %s\n" "${OUTPUT_ROOT}"
printf "  Collect: bash %s/collect_results.sh\n" "${SCRIPT_DIR}"
