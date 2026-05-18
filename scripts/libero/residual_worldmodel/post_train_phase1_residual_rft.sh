#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

TASK_SUITE="${TASK_SUITE:-spatial}"
WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-}"
WORLD_MODEL_CONFIG="${WORLD_MODEL_CONFIG:-}"
TARGET_MODE="${TARGET_MODE:-pixel_residual_roi_dynamic}"
MODEL_GENERATION="${MODEL_GENERATION:-v3}"
EXP_NAME="${EXP_NAME:-${MODEL_GENERATION}_${TARGET_MODE}_rft}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/phase1/residual_worldmodel/rft/${EXP_NAME}}"
RFT_STEPS="${RFT_STEPS:-400}"
SEED="${SEED:-7}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
BASE_VLA_PATH="${BASE_VLA_PATH:-${REPO_ROOT}/checkpoints/libero/Base_VLA/${TASK_SUITE}}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
DATA_ROOT="${DATA_ROOT:-${LIBERO_DATA_ROOT:-${LOCALDATA_ROOT:-/localdata}/modified_libero_rlds}}"
RFT_CKPT_ROOT="${RFT_CKPT_ROOT:-${REPO_ROOT}/checkpoints/libero/VLA-RFT}"

if [[ -z "${WORLD_MODEL_CKPT}" ]]; then
  echo "WORLD_MODEL_CKPT is required" >&2
  exit 2
fi
if [[ "${MODEL_GENERATION}" != "baseline" && -z "${WORLD_MODEL_CONFIG}" ]]; then
  WORLD_MODEL_CONFIG="${WORLD_MODEL_CKPT}/pixel_residual_config.json"
fi
if [[ ! -d "${WORLD_MODEL_CKPT}" ]]; then
  echo "WORLD_MODEL_CKPT not found: ${WORLD_MODEL_CKPT}" >&2
  exit 2
fi
if [[ "${MODEL_GENERATION}" != "baseline" && ! -f "${WORLD_MODEL_CONFIG}" ]]; then
  echo "WORLD_MODEL_CONFIG not found: ${WORLD_MODEL_CONFIG}" >&2
  exit 2
fi

if [[ "${SMOKE}" == "1" ]]; then
  RFT_STEPS="${RFT_STEPS_SMOKE:-2}"
  N_GPUS_PER_NODE="${N_GPUS_PER_NODE_SMOKE:-1}"
fi

mkdir -p "${OUTPUT_DIR}/smoke"

"${VENV_PATH}/bin/python" - <<PY
import json, pathlib, subprocess, os
out = pathlib.Path("${OUTPUT_DIR}")
cfg = {}
if "${MODEL_GENERATION}" != "baseline":
    cfg = json.load(open("${WORLD_MODEL_CONFIG}", encoding="utf-8"))
    if cfg.get("target_mode") != "${TARGET_MODE}":
        raise SystemExit(f"target_mode mismatch: config={cfg.get('target_mode')} requested=${TARGET_MODE}")
    if "${MODEL_GENERATION}" == "v3" and not cfg.get("use_residual_write_mask", False):
        print("[WARN] MODEL_GENERATION=v3 but config.use_residual_write_mask is false")
info = {
    "task_suite": "${TASK_SUITE}",
    "world_model_ckpt": "${WORLD_MODEL_CKPT}",
    "world_model_config": "" if "${MODEL_GENERATION}" == "baseline" else "${WORLD_MODEL_CONFIG}",
    "target_mode": "${TARGET_MODE}",
    "model_generation": "${MODEL_GENERATION}",
    "rft_steps": int("${RFT_STEPS}"),
    "seed": int("${SEED}"),
    "base_vla_path": "${BASE_VLA_PATH}",
    "data_root": "${DATA_ROOT}",
    "rft_ckpt_root": "${RFT_CKPT_ROOT}",
    "rft_local_dir": "${RFT_CKPT_ROOT}/${TASK_SUITE}/phase1_${EXP_NAME}",
}
(out / "world_model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
(out / "rft_config_used.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
PY

git status --short > "${OUTPUT_DIR}/git_status.txt" || true
{
  echo "date=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "pwd=${REPO_ROOT}"
  echo "VENV_PATH=${VENV_PATH}"
  "${VENV_PATH}/bin/python" -V
} > "${OUTPUT_DIR}/env_info.txt"

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ "${MODEL_GENERATION}" != "baseline" ]]; then
    "${VENV_PATH}/bin/python" -m worldmodel.dynquery.rft \
      --checkpoint "${WORLD_MODEL_CKPT}" \
      --target-mode "${TARGET_MODE}" \
      --model-generation "${MODEL_GENERATION}" \
      --output-dir "${OUTPUT_DIR}/smoke" \
      --device "${SMOKE_DEVICE:-cpu}" \
      --horizon 8 | tee "${OUTPUT_DIR}/smoke/smoke.log"
  else
    printf '{"model_generation":"baseline","target_mode":"pixel","note":"AR-Pixel baseline uses existing token world model path"}\n' \
      > "${OUTPUT_DIR}/smoke/debug_stats.json"
  fi
fi

export LIBERO_TASK_NAME="${TASK_SUITE}"
export BASE_VLA_PATH
export WORLD_MODEL_PATH="${WORLD_MODEL_CKPT}"
export RFT_STEPS
export N_GPUS_PER_NODE
export DATE="phase1"
export POST_EXP_NAME="${EXP_NAME}"
RFT_LOCAL_DIR="${RFT_CKPT_ROOT}/${TASK_SUITE}/${DATE}_${POST_EXP_NAME}"

# Derive world_reward hydra overrides from env vars so that any caller
# (e.g. run_v4_core_sweep.sh) that sets WORLD_REWARD_TYPE gets the correct
# value forwarded to compute_rft_reward() — not silently overridden by the
# YAML default. USER_EXTRA_RFT_ARGS is placed last so callers can still
# override individual keys.
_WR_TYPE="${WORLD_REWARD_TYPE:-lpips_mae}"
_WR_ALPHA="${RANK_REWARD_ALPHA:-0.2}"
_WR_BETA="${RANK_REWARD_BETA:-0.8}"
_WR_NORM="${NORMALIZE_RANK_REWARD:-1}"
_WR_CLIP="${CLIP_RANK_REWARD:-1}"
_WR_CLIPV="${RANK_REWARD_CLIP_VALUE:-5.0}"
_WR_OVERRIDES="world_reward.type=${_WR_TYPE} world_reward.rank_alpha=${_WR_ALPHA} world_reward.rank_beta=${_WR_BETA} world_reward.normalize_rank=${_WR_NORM} world_reward.clip_rank=${_WR_CLIP} world_reward.clip_value=${_WR_CLIPV}"

if [[ "${MODEL_GENERATION}" != "baseline" ]]; then
  export EXTRA_RFT_ARGS="processor.phase1_residual.enabled=True processor.phase1_residual.checkpoint=${WORLD_MODEL_CKPT} processor.phase1_residual.config=${WORLD_MODEL_CONFIG} processor.phase1_residual.target_mode=${TARGET_MODE} processor.phase1_residual.model_generation=${MODEL_GENERATION} world_model_rollout.phase1_residual.enabled=True data.video.dataset_path=${DATA_ROOT} data.video.dataset_name=libero_${TASK_SUITE}_no_noops trainer.default_local_dir=${RFT_LOCAL_DIR} ${_WR_OVERRIDES} ${USER_EXTRA_RFT_ARGS:-}"
else
  export EXTRA_RFT_ARGS="data.video.dataset_path=${DATA_ROOT} data.video.dataset_name=libero_${TASK_SUITE}_no_noops trainer.default_local_dir=${RFT_LOCAL_DIR} ${USER_EXTRA_RFT_ARGS:-}"
fi

printf 'bash train/verl/examples/grpo_trainer/run_vla_rft.sh\nEXTRA_RFT_ARGS=%s\n' "${EXTRA_RFT_ARGS}" > "${OUTPUT_DIR}/train_command.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${OUTPUT_DIR}/train_command.txt"
  exit 0
fi

source "${VENV_PATH}/bin/activate"
bash train/verl/examples/grpo_trainer/run_vla_rft.sh 2>&1 | tee "${OUTPUT_DIR}/train.log"

CKPT_ROOT="${RFT_LOCAL_DIR}"
ACTOR_CKPT=$(find "${CKPT_ROOT}" -path "*/actor" -type d | sort -V | tail -n 1 || true)
if [[ -z "${ACTOR_CKPT}" ]]; then
  echo "No actor checkpoint found under ${CKPT_ROOT}" >&2
  exit 3
fi
printf '%s\n' "${ACTOR_CKPT}" > "${OUTPUT_DIR}/rft_checkpoint_path.txt"
echo "RFT checkpoint: ${ACTOR_CKPT}"
