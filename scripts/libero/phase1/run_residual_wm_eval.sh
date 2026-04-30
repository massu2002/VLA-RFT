#!/usr/bin/env bash
# run_residual_wm_eval.sh — Phase 1 Pixel-Residual World Model experiment.
#
# Trains and evaluates 3 conditions sequentially on LIBERO:
#   1. pixel                     — predict full future image (baseline)
#   2. pixel_residual            — predict residual (future - current)
#   3. pixel_residual_roi_dynamic — residual + dynamic mask + gripper ROI losses
#
# Usage:
#   bash scripts/libero/phase1/run_residual_wm_eval.sh [spatial|object|goal|10]
#   SMOKE=1  bash scripts/libero/phase1/run_residual_wm_eval.sh
#   SKIP_TRAIN=1 bash scripts/libero/phase1/run_residual_wm_eval.sh spatial
#
# Key env-var overrides:
#   TASK_SUITE            — spatial | object | goal | 10  (default: spatial)
#   SEED                  — random seed (default: 42)
#   DEVICE                — cuda device string or "auto"
#   CUDA_VISIBLE_DEVICES  — which GPU(s) to use, e.g. "0" or "0,1,2,3"
#                           torchrun uses all visible GPUs automatically
#   NPROC                 — override GPU count (default: auto-detected via torch.cuda.device_count())
#   SMOKE                 — 1 = tiny run (5 steps, 10 eval windows, 5 dry-run windows)
#   SKIP_TRAIN            — 1 = skip training, evaluate pre-existing checkpoints
#   SKIP_EVAL             — 1 = train only, skip evaluation
#   TRAIN_CONDITIONS      — space-separated subset, e.g. "pixel pixel_residual"
#   MAX_STEPS             — override max training steps for all conditions
#   BATCH_SIZE            — per-device batch size
#   GRAD_ACCUM            — gradient accumulation steps
#   NUM_EVAL_WINDOWS      — total eval windows per condition (default: 200)
#   NUM_RANKING_WINDOWS   — ranking eval windows (default: 100)
#   CKPT_ROOT             — checkpoint root (default: checkpoints/libero/PixelResidualWM)
#   OUT_ROOT              — output root (default: results/phase1/TIMESTAMP_suite)
#
# Outputs under results/phase1/<TIMESTAMP>_<suite>/:
#   <condition>/aggregate_metrics.json
#   <condition>/metrics_by_task.csv
#   <condition>/ranking_by_window.csv
#   <condition>/ranking_by_task.csv
#   run_config.json
#
# Checkpoints under checkpoints/libero/PixelResidualWM/<suite>/<condition>/v1/s<seed>/

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PHASE1_DIR="${SCRIPT_DIR}"
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

source "${WM_SCRIPTS}/common.sh"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log() { echo "[phase1] $(date +%H:%M:%S) $*"; }
fail() { echo "[phase1 ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_SUITE="${1:-${TASK_SUITE:-spatial}}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
SMOKE="${SMOKE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
TRAIN_CONDITIONS="${TRAIN_CONDITIONS:-pixel pixel_residual pixel_residual_roi_dynamic}"
NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-200}"
NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS:-100}"

DATE_TAG=$(timestamp)
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/PixelResidualWM}"

if [ -z "${OUT_ROOT:-}" ]; then
  OUT_ROOT="${REPO_ROOT}/results/phase1/${DATE_TAG}_${TASK_SUITE}"
fi

# Smoke overrides
if [ "${SMOKE}" = "1" ]; then
  export MAX_STEPS="${MAX_STEPS:-5}"
  export BATCH_SIZE="${BATCH_SIZE:-2}"
  export GRAD_ACCUM="${GRAD_ACCUM:-1}"
  NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-10}"
  export SAVE_STEPS="${SAVE_STEPS:-5}"
  DRY_RUN_WINDOWS="${DRY_RUN_WINDOWS:-5}"
  log "SMOKE=1: tiny run (5 steps / 10 eval windows)"
else
  DRY_RUN_WINDOWS="${DRY_RUN_WINDOWS:-0}"
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup_env
mkdir -p "${OUT_ROOT}"
mkdir -p "${REPO_ROOT}/results/phase1"
echo "${OUT_ROOT}" > "${REPO_ROOT}/results/phase1/latest.txt"

# run_config.json は init_phase1_run.sh が作成済みの場合は上書きしない。
# 存在しない場合のみ書き込む（複数PCから同じ OUT_ROOT を使う場合の競合を防ぐ）。
if [ ! -f "${OUT_ROOT}/run_config.json" ]; then
  python3 -c "
import json
data = {
  'date': '${DATE_TAG}',
  'suite': '${TASK_SUITE}',
  'seed': ${SEED},
  'smoke': '${SMOKE}',
  'conditions': '${TRAIN_CONDITIONS}'.split(),
  'skip_train': '${SKIP_TRAIN}',
  'skip_eval': '${SKIP_EVAL}',
  'num_eval_windows': ${NUM_EVAL_WINDOWS},
  'num_ranking_windows': ${NUM_RANKING_WINDOWS},
}
json.dump(data, open('${OUT_ROOT}/run_config.json', 'w'), indent=2)
"
fi

log "=== Phase 1: Pixel-Residual World Model ==="
log "task_suite  : ${TASK_SUITE}"
log "conditions  : ${TRAIN_CONDITIONS}"
log "ckpt_root   : ${CKPT_ROOT}"
log "output      : ${OUT_ROOT}"
log "skip_train  : ${SKIP_TRAIN} / skip_eval: ${SKIP_EVAL}"

# Export common settings for child scripts
export TASK_SUITE SEED DEVICE
export DATA_ROOT="${DATA_ROOT:-$(default_libero_data_root)}"
[ -n "${MAX_STEPS:-}"  ] && export MAX_STEPS
[ -n "${BATCH_SIZE:-}" ] && export BATCH_SIZE
[ -n "${GRAD_ACCUM:-}" ] && export GRAD_ACCUM
[ -n "${SAVE_STEPS:-}" ] && export SAVE_STEPS
[ -n "${NPROC:-}"      ] && export NPROC

# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------
if ! is_true "${SKIP_TRAIN}"; then
  log "--- Training phase ---"
  for COND in ${TRAIN_CONDITIONS}; do
    COND_CKPT="${CKPT_ROOT}/${TASK_SUITE}/${COND}/v1/s${SEED}"
    log "=== TRAIN: ${COND} → ${COND_CKPT} ==="

    TARGET_MODE="${COND}" \
    OUTPUT_ROOT="${CKPT_ROOT}" \
    EXP_NAME="v1" \
    bash "${WM_SCRIPTS}/train_pixel_residual_worldmodel.sh" "${TASK_SUITE}" \
      && log "  TRAIN ${COND} done" \
      || log "  WARNING: TRAIN ${COND} failed — continuing"
  done
  log "--- Training phase complete ---"
else
  log "SKIP_TRAIN=1 — using pre-existing checkpoints"
fi

# ---------------------------------------------------------------------------
# EVALUATION LOOP
# ---------------------------------------------------------------------------
if ! is_true "${SKIP_EVAL}"; then
  log "--- Evaluation phase ---"
  for COND in ${TRAIN_CONDITIONS}; do
    COND_CKPT="${CKPT_ROOT}/${TASK_SUITE}/${COND}/v1/s${SEED}"
    COND_OUT="${OUT_ROOT}/${COND}"

    if [ ! -d "${COND_CKPT}" ]; then
      log "  WARNING: checkpoint not found at ${COND_CKPT} — skipping ${COND}"
      continue
    fi
    if [ ! -f "${COND_CKPT}/pixel_residual_config.json" ]; then
      log "  WARNING: pixel_residual_config.json missing in ${COND_CKPT} — skipping ${COND}"
      continue
    fi

    log "=== EVAL: ${COND} → ${COND_OUT} ==="

    MODEL_DIR="${COND_CKPT}" \
    CONDITION_NAME="${COND}" \
    OUTPUT_DIR="${COND_OUT}" \
    NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS}" \
    NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS}" \
    DRY_RUN_WINDOWS="${DRY_RUN_WINDOWS}" \
    bash "${WM_SCRIPTS}/eval_pixel_residual_worldmodel.sh" \
      && log "  EVAL ${COND} done" \
      || log "  WARNING: EVAL ${COND} failed (check ${COND_OUT}/eval.log)"
  done
  log "--- Evaluation phase complete ---"
else
  log "SKIP_EVAL=1 — skipping evaluation"
fi

# ---------------------------------------------------------------------------
# Summary pointer
# ---------------------------------------------------------------------------
log "=== Phase 1 run complete ==="
log "Output: ${OUT_ROOT}"
log ""
log "Next step — run the summarizer:"
log "  bash scripts/libero/phase1/summarize_residual_wm_eval.sh ${OUT_ROOT}"
