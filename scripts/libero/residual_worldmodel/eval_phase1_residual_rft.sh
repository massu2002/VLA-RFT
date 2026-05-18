#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

TASK_SUITE="${TASK_SUITE:-spatial}"
POLICY_CKPT="${POLICY_CKPT:-}"
EXP_NAME="${EXP_NAME:-phase1_residual_rft}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/phase1/residual_worldmodel/rft/${EXP_NAME}/eval}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-7}"
EVAL_SETTING="${EVAL_SETTING:-standard}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv5090_eval}"
BASE_VLA_PATH="${BASE_VLA_PATH:-${REPO_ROOT}/checkpoints/libero/Base_VLA/${TASK_SUITE}}"
ROLLOUT_PHASE_DIR="${ROLLOUT_PHASE_DIR:-rft_phase1}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${REPO_ROOT}/rollouts/libero/${ROLLOUT_PHASE_DIR}/${EXP_NAME}/${TASK_SUITE}}"
LIBERO_BENCHMARK_ROOT="${LIBERO_BENCHMARK_ROOT:-${REPO_ROOT}/third_party/LIBERO/libero/libero}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero_phase1_eval}"
LIBERO_CONFIG_FILE="${LIBERO_CONFIG_PATH}/config.yaml"

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
if [[ ! -d "${LIBERO_BENCHMARK_ROOT}" ]]; then
  echo "LIBERO_BENCHMARK_ROOT not found: ${LIBERO_BENCHMARK_ROOT}" >&2
  exit 2
fi
for required_libero_dir in assets bddl_files init_files; do
  if [[ ! -d "${LIBERO_BENCHMARK_ROOT}/${required_libero_dir}" ]]; then
    echo "LIBERO ${required_libero_dir} directory not found: ${LIBERO_BENCHMARK_ROOT}/${required_libero_dir}" >&2
    exit 2
  fi
done
if [[ "${SMOKE}" == "1" ]]; then
  NUM_TRIALS="${NUM_TRIALS_SMOKE:-1}"
fi
if [[ -x "${VENV_PATH}/bin/python" ]]; then
  EVAL_PYTHON="${VENV_PATH}/bin/python"
elif [[ -x "${VENV_PATH}/bin/python3" ]]; then
  EVAL_PYTHON="${VENV_PATH}/bin/python3"
else
  echo "Python executable not found under VENV_PATH: ${VENV_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LIBERO_CONFIG_PATH}"

"${REPO_ROOT}/.venv/bin/python" - <<PY
from pathlib import Path
root = Path("${LIBERO_BENCHMARK_ROOT}").resolve()
config_path = Path("${LIBERO_CONFIG_FILE}")
config_path.write_text(
    "\\n".join([
        f"assets: {root / 'assets'}",
        f"bddl_files: {root / 'bddl_files'}",
        f"benchmark_root: {root}",
        f"datasets: {root.parent / 'datasets'}",
        f"init_states: {root / 'init_files'}",
        "",
    ]),
    encoding="utf-8",
)
PY

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
    "rollout_phase_dir": "${ROLLOUT_PHASE_DIR}",
    "rollout_root": "${ROLLOUT_ROOT}",
    "libero_config_path": "${LIBERO_CONFIG_PATH}",
    "libero_config_file": "${LIBERO_CONFIG_FILE}",
    "libero_benchmark_root": "${LIBERO_BENCHMARK_ROOT}",
}
(out / "eval_config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY

CMD=(
  "${EVAL_PYTHON}" train/verl/vla-adapter/openvla-oft/experiments/robot/libero/run_libero_eval.py
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
  --run_id_note "${ROLLOUT_PHASE_DIR}"
  --local_log_dir "${OUTPUT_DIR}"
  --num_trials_per_task "${NUM_TRIALS}"
  --seed "${SEED}"
)

printf '%q ' "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}" "${CMD[@]}" > "${OUTPUT_DIR}/eval_command.txt"
printf '\n' >> "${OUTPUT_DIR}/eval_command.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${OUTPUT_DIR}/eval_command.txt"
  exit 0
fi

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"
eval_rc=${PIPESTATUS[0]}
set -e

"${REPO_ROOT}/.venv/bin/python" - <<PY
import csv, json, pathlib
out = pathlib.Path("${OUTPUT_DIR}")
rollout = pathlib.Path("${ROLLOUT_ROOT}")
overall = sorted(out.glob("**/overall_results.json")) or sorted(rollout.glob("overall_results.json"))
tasks = sorted(out.glob("**/task_results.json")) or sorted(rollout.glob("task_results.json"))
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

(out / "rollout_root.txt").write_text(str(rollout.resolve()) + "\n", encoding="utf-8")
PY

if [[ "${eval_rc}" -ne 0 ]]; then
  if [[ -f "${OUTPUT_DIR}/success_summary.json" ]]; then
    echo "WARNING: eval command exited rc=${eval_rc}, but success_summary.json was recovered; continuing." >&2
  else
    exit "${eval_rc}"
  fi
fi
