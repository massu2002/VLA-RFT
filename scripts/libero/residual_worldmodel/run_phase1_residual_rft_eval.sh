#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

TASK_SUITE="${TASK_SUITE:-spatial}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/phase1/residual_worldmodel/rft}"
RFT_STEPS="${RFT_STEPS:-400}"
NUM_TRIALS="${NUM_TRIALS:-10}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

declare -a CONDITIONS=(
  "baseline|baseline|pixel|${BASELINE_PIXEL_CKPT:-checkpoints/libero/WorldModel/${TASK_SUITE}}|${BASELINE_PIXEL_CONFIG:-}"
  "v1_pixel_residual|v1|pixel_residual|${V1_PIXEL_RESIDUAL_CKPT:-}|${V1_PIXEL_RESIDUAL_CONFIG:-}"
  "v1_pixel_residual_roi_dynamic|v1|pixel_residual_roi_dynamic|${V1_PIXEL_RESIDUAL_ROI_DYNAMIC_CKPT:-}|${V1_PIXEL_RESIDUAL_ROI_DYNAMIC_CONFIG:-}"
  "v3_pixel_residual|v3|pixel_residual|${V3_PIXEL_RESIDUAL_CKPT:-}|${V3_PIXEL_RESIDUAL_CONFIG:-}"
  "v3_pixel_residual_roi_dynamic|v3|pixel_residual_roi_dynamic|${V3_PIXEL_RESIDUAL_ROI_DYNAMIC_CKPT:-}|${V3_PIXEL_RESIDUAL_ROI_DYNAMIC_CONFIG:-}"
)

mkdir -p "${OUTPUT_ROOT}"
LOG="${OUTPUT_ROOT}/run_phase1_residual_rft_eval.log"
: > "${LOG}"

for item in "${CONDITIONS[@]}"; do
  IFS='|' read -r EXP GEN MODE CKPT CFG <<< "${item}"
  if [[ -z "${CKPT}" || ! -d "${CKPT}" ]]; then
    echo "[SKIP] ${EXP}: checkpoint missing (${CKPT:-unset})" | tee -a "${LOG}"
    continue
  fi
  if [[ -z "${CFG}" ]]; then
    CFG="${CKPT}/pixel_residual_config.json"
  fi
  if [[ ! -f "${CFG}" ]]; then
    echo "[SKIP] ${EXP}: config missing (${CFG})" | tee -a "${LOG}"
    continue
  fi

  echo "[RUN] ${EXP}" | tee -a "${LOG}"
  TASK_SUITE="${TASK_SUITE}" \
  WORLD_MODEL_CKPT="${CKPT}" \
  WORLD_MODEL_CONFIG="${CFG}" \
  TARGET_MODE="${MODE}" \
  MODEL_GENERATION="${GEN}" \
  EXP_NAME="${EXP}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/${EXP}" \
  RFT_STEPS="${RFT_STEPS}" \
  SMOKE="${SMOKE}" \
  DRY_RUN="${DRY_RUN}" \
    bash "${SCRIPT_DIR}/post_train_phase1_residual_rft.sh" | tee -a "${LOG}"

  if [[ "${DRY_RUN}" == "1" || "${RUN_EVAL}" != "1" ]]; then
    continue
  fi
  POLICY_CKPT=$(cat "${OUTPUT_ROOT}/${EXP}/rft_checkpoint_path.txt")
  TASK_SUITE="${TASK_SUITE}" \
  POLICY_CKPT="${POLICY_CKPT}" \
  EXP_NAME="${EXP}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/${EXP}/eval" \
  NUM_TRIALS="${NUM_TRIALS}" \
  SMOKE="${SMOKE}" \
    bash "${SCRIPT_DIR}/eval_phase1_residual_rft.sh" | tee -a "${LOG}"
done

bash "${SCRIPT_DIR}/summarize_phase1_residual_rft.sh"
