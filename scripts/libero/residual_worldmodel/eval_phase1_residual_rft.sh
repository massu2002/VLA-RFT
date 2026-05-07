#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

TASK_SUITE="${TASK_SUITE:-spatial}"
POLICY_CKPT="${POLICY_CKPT:-}"
EXP_NAME="${EXP_NAME:-phase1_residual_rft}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/phase1/residual_worldmodel/rft/${EXP_NAME}/eval}"
NUM_TRIALS="${NUM_TRIALS:-10}"
SEED="${SEED:-7}"
EVAL_SETTING="${EVAL_SETTING:-standard}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv5090_eval}"
BASE_VLA_PATH="${BASE_VLA_PATH:-${REPO_ROOT}/checkpoints/libero/Base/${TASK_SUITE}}"

if [[ -z "${POLICY_CKPT}" ]]; then
  if [[ -f "${REPO_ROOT}/results/phase1/residual_worldmodel/rft/${EXP_NAME}/rft_checkpoint_path.txt" ]]; then
    POLICY_CKPT=$(cat "${REPO_ROOT}/results/phase1/residual_worldmodel/rft/${EXP_NAME}/rft_checkpoint_path.txt")
  else
    echo "POLICY_CKPT is required" >&2
    exit 2
  fi
fi
if [[ ! -d "${POLICY_CKPT}" ]]; then
  echo "POLICY_CKPT not found: ${POLICY_CKPT}" >&2
  exit 2
fi
if [[ ! -d "${BASE_VLA_PATH}" ]]; then
  echo "BASE_VLA_PATH not found: ${BASE_VLA_PATH}" >&2
  exit 2
fi
if [[ "${SMOKE}" == "1" ]]; then
  NUM_TRIALS="${NUM_TRIALS_SMOKE:-1}"
fi

mkdir -p "${OUTPUT_DIR}"

"${REPO_ROOT}/.venv/bin/python" - <<PY
import json, pathlib
out = pathlib.Path("${OUTPUT_DIR}")
cfg = {
    "task_suite": "${TASK_SUITE}",
    "policy_ckpt": "${POLICY_CKPT}",
    "exp_name": "${EXP_NAME}",
    "num_trials": int("${NUM_TRIALS}"),
    "seed": int("${SEED}"),
    "eval_setting": "${EVAL_SETTING}",
    "base_vla_path": "${BASE_VLA_PATH}",
}
(out / "eval_config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY

CMD=(
  python train/verl/vla-adapter/openvla-oft/experiments/robot/libero/run_libero_eval.py
  --use_l1_regression False
  --use_diffusion False
  --use_flow_matching True
  --use_proprio True
  --use_film False
  --num_images_in_input 1
  --pretrained_checkpoint "${BASE_VLA_PATH}"
  --actor_path "${POLICY_CKPT}"
  --actor_model_version "${EXP_NAME}"
  --task_suite_name "libero_${TASK_SUITE}"
  --save_version v1
  --use_minivla True
  --run_id_note "RFT_${EXP_NAME}"
  --local_log_dir "${OUTPUT_DIR}"
  --num_trials_per_task "${NUM_TRIALS}"
  --seed "${SEED}"
)

printf '%q ' "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" > "${OUTPUT_DIR}/eval_command.txt"
printf '\n' >> "${OUTPUT_DIR}/eval_command.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${OUTPUT_DIR}/eval_command.txt"
  exit 0
fi

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"

"${REPO_ROOT}/.venv/bin/python" - <<PY
import csv, json, pathlib
out = pathlib.Path("${OUTPUT_DIR}")
overall = sorted(out.glob("**/overall_results.json"))
tasks = sorted(out.glob("**/task_results.json"))
if overall:
    data = json.load(open(overall[-1], encoding="utf-8"))
    (out / "success_summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
if tasks:
    data = json.load(open(tasks[-1], encoding="utf-8"))
    rows = data.get("tasks", [])
    with open(out / "success_by_task.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["task_id", "task_description", "num_trials", "num_successes", "num_failures", "success_rate"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
PY
