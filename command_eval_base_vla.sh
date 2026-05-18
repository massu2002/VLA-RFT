#!/usr/bin/env bash
# Evaluate Base_VLA checkpoints on LIBERO task suites.
#
# Results are written to:
#   results/base_vla/<task_suite>/
#
# Examples:
#   bash command_base_eval.sh
#   TASK_SUITES=spatial SMOKE=1 bash command_base_eval.sh
#   TASK_SUITES="spatial object goal 10" NUM_TRIALS=50 bash command_base_eval.sh
#   SEEDS="7 8 9" bash command_base_eval.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

TASK_SUITES="${TASK_SUITES:-spatial object goal 10}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/results/base_vla}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEEDS="${SEEDS:-${SEED:-7 8 9}}"
SEED_SUBDIR="${SEED_SUBDIR:-1}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/.venv5090_eval}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
LIBERO_BENCHMARK_ROOT="${LIBERO_BENCHMARK_ROOT:-${SCRIPT_DIR}/third_party/LIBERO/libero/libero}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${SCRIPT_DIR}/.libero_base_eval}"
LIBERO_CONFIG_FILE="${LIBERO_CONFIG_PATH}/config.yaml"

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

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "VENV_PATH not found: ${VENV_PATH}" >&2
  exit 2
fi
if [[ ! -d "${LIBERO_BENCHMARK_ROOT}" ]]; then
  echo "LIBERO_BENCHMARK_ROOT not found: ${LIBERO_BENCHMARK_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}" "${LIBERO_CONFIG_PATH}"
settings_file="${OUT_ROOT}/multi_seed_experiment_settings.md"
{
  echo "# Base VLA LIBERO Multi-Seed Evaluation"
  echo
  echo "- Generated: $(date -Is)"
  echo "- Task suites: \`${TASK_SUITES}\`"
  echo "- Seeds: \`${SEEDS}\`"
  echo "- Output root: \`${OUT_ROOT}\`"
  echo "- Num trials per task: \`${NUM_TRIALS}\`"
  echo
  echo "| Seed | Manifest |"
  echo "|---:|---|"
} > "${settings_file}"

"${SCRIPT_DIR}/.venv/bin/python" - <<PY
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

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${SCRIPT_DIR}/third_party/LIBERO:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

collect_results() {
  local task_suite="$1"
  local out_dir="$2"
  local rollout_phase_dir="$3"
  TASK_SUITE_FOR_COLLECT="${task_suite}" OUT_DIR_FOR_COLLECT="${out_dir}" SCRIPT_DIR_FOR_COLLECT="${SCRIPT_DIR}" ROLLOUT_PHASE_DIR_FOR_COLLECT="${rollout_phase_dir}" \
    "${SCRIPT_DIR}/.venv/bin/python" - <<'PY'
import csv
import json
import os
import shutil
from pathlib import Path

task_suite = os.environ["TASK_SUITE_FOR_COLLECT"]
out = Path(os.environ["OUT_DIR_FOR_COLLECT"])
script_dir = Path(os.environ["SCRIPT_DIR_FOR_COLLECT"])
rollout_phase_dir = os.environ["ROLLOUT_PHASE_DIR_FOR_COLLECT"]
rollout = script_dir / "rollouts" / "libero" / rollout_phase_dir / "base_vla" / task_suite

overall = sorted(out.glob("**/overall_results.json")) or sorted(rollout.glob("overall_results.json"))
tasks = sorted(out.glob("**/task_results.json")) or sorted(rollout.glob("task_results.json"))

if overall:
    data = json.load(open(overall[-1], encoding="utf-8"))
    (out / "success_summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(overall[-1], out / "overall_results.json")

if tasks:
    data = json.load(open(tasks[-1], encoding="utf-8"))
    (out / "task_results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    rows = data.get("tasks", [])
    with open(out / "success_by_task.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["task_id", "task_description", "num_trials", "num_successes", "num_failures", "success_rate"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

for task_json in sorted(rollout.glob("task_*__*/task_results.json")):
    dest_dir = out / task_json.parent.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task_json, dest_dir / "task_results.json")

(out / "rollout_root.txt").write_text(str(rollout.resolve()) + "\n", encoding="utf-8")
PY
}

for seed in ${SEEDS}; do
  manifest="${OUT_ROOT}/base_eval_manifest_seed${seed}.tsv"
  printf 'task_suite\tbase_vla_path\tseed\toutput_dir\tstatus\tlog\n' > "${manifest}"
  echo "| ${seed} | \`base_eval_manifest_seed${seed}.tsv\` |" >> "${settings_file}"

  for task_suite in ${TASK_SUITES}; do
    base_vla_path="${BASE_VLA_ROOT:-${SCRIPT_DIR}/checkpoints/libero/Base_VLA}/${task_suite}"
    if [[ "${SEED_SUBDIR}" == "1" ]]; then
      out_dir="${OUT_ROOT}/${task_suite}/seed${seed}"
      rollout_phase_dir="base_vla_seed${seed}"
    else
      out_dir="${OUT_ROOT}/${task_suite}"
      rollout_phase_dir="base_vla"
    fi
    log_file="${out_dir}/eval.log"

    if [[ ! -d "${base_vla_path}" ]]; then
      echo "Base VLA checkpoint not found for ${task_suite}: ${base_vla_path}" >&2
      exit 2
    fi

    mkdir -p "${out_dir}"

    if [[ "${DRY_RUN}" != "1" ]]; then
      collect_results "${task_suite}" "${out_dir}" "${rollout_phase_dir}" || true
    fi

    if [[ -f "${out_dir}/success_summary.json" && "${FORCE}" != "1" ]]; then
      echo "[base-eval] skip existing seed=${seed} ${task_suite}: ${out_dir}/success_summary.json (FORCE=1 to rerun)"
      echo -e "${task_suite}\t${base_vla_path}\t${seed}\t${out_dir}\tskipped_existing\t${log_file}" >> "${manifest}"
      continue
    fi

    "${SCRIPT_DIR}/.venv/bin/python" - <<PY
import json
from pathlib import Path
out = Path("${out_dir}")
cfg = {
    "task_suite": "${task_suite}",
    "base_vla_path": "${base_vla_path}",
    "output_dir": "${out_dir}",
    "num_trials": int("${NUM_TRIALS}"),
    "seed": int("${seed}"),
    "seed_subdir": "${SEED_SUBDIR}" == "1",
    "rollout_phase_dir": "${rollout_phase_dir}",
    "libero_config_path": "${LIBERO_CONFIG_PATH}",
    "libero_config_file": "${LIBERO_CONFIG_FILE}",
}
(out / "eval_config_used.json").write_text(json.dumps(cfg, indent=2) + "\\n", encoding="utf-8")
PY

    cmd=(
    "${EVAL_PYTHON}" train/verl/vla-adapter/openvla-oft/experiments/robot/libero/run_libero_eval.py
    --use_l1_regression False
    --use_diffusion False
    --use_flow_matching True
    --use_proprio True
    --use_film False
    --num_images_in_input 1
    --pretrained_checkpoint "${base_vla_path}"
    --actor_model_version "base_vla"
    --task_suite_name "libero_${task_suite}"
    --save_version v1
    --use_minivla True
    --run_id_note "${rollout_phase_dir}"
    --local_log_dir "${out_dir}"
    --num_trials_per_task "${NUM_TRIALS}"
    --seed "${seed}"
    )

    printf '%q ' "CUDA_VISIBLE_DEVICES=${GPU_ID}" "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}" "${cmd[@]}" > "${out_dir}/eval_command.txt"
    printf '\n' >> "${out_dir}/eval_command.txt"

    echo "[base-eval] seed=${seed} task_suite=${task_suite}"
    echo "[base-eval] base_vla=${base_vla_path}"
    echo "[base-eval] out=${out_dir}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      cat "${out_dir}/eval_command.txt"
      echo -e "${task_suite}\t${base_vla_path}\t${seed}\t${out_dir}\tdry_run\t${log_file}" >> "${manifest}"
      continue
    fi

    echo -e "${task_suite}\t${base_vla_path}\t${seed}\t${out_dir}\trunning\t${log_file}" >> "${manifest}"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" 2>&1 | tee "${log_file}"
    eval_rc=${PIPESTATUS[0]}
    set -e

    collect_results "${task_suite}" "${out_dir}" "${rollout_phase_dir}"

    if [[ "${eval_rc}" -ne 0 ]]; then
      if [[ -f "${out_dir}/success_summary.json" ]] && grep -q "EGL_NOT_INITIALIZED" "${log_file}"; then
        echo "[base-eval] WARNING: ignored robosuite EGL shutdown error after results were written for seed=${seed} ${task_suite}" >&2
        echo -e "${task_suite}\t${base_vla_path}\t${seed}\t${out_dir}\tdone_ignored_egl_rc_${eval_rc}\t${log_file}" >> "${manifest}"
      else
        echo -e "${task_suite}\t${base_vla_path}\t${seed}\t${out_dir}\tfailed_rc_${eval_rc}\t${log_file}" >> "${manifest}"
        exit "${eval_rc}"
      fi
    else
      echo -e "${task_suite}\t${base_vla_path}\t${seed}\t${out_dir}\tdone\t${log_file}" >> "${manifest}"
    fi
  done
done
