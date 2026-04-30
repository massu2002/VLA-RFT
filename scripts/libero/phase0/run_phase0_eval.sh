#!/usr/bin/env bash
# run_phase0_eval.sh — Phase 0 "Why didn't WorldModel help RFT?" diagnostics
#
# Evaluates 3 conditions on a fixed 4-task subset (default: LIBERO-Spatial):
#   1. baseline WorldModel   — ROI metrics, ranking JSONL, failure taxonomy
#   2. Base VLA (no RFT)     — policy success rate via simulation
#   3. VLA-RFT               — policy success rate via simulation
#
# Usage:
#   bash scripts/libero/phase0/run_phase0_eval.sh [spatial|object|goal]
#   SMOKE=1 bash scripts/libero/phase0/run_phase0_eval.sh       # 1-task smoke test
#   SKIP_VLA=1 bash scripts/libero/phase0/run_phase0_eval.sh    # WorldModel only
#
# Outputs under:
#   results/phase0/YYYYMMDD_HHMMSS_<suite>/

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# ---------------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------------
TASK_SUITE="${1:-spatial}"
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"              # set to 1 for 1-task smoke test
SKIP_VLA="${SKIP_VLA:-0}"        # set to 1 to skip VLA sim evaluation
SKIP_WM="${SKIP_WM:-0}"         # set to 1 to skip WorldModel eval (reuse existing WM outputs)
DEVICE="${DEVICE:-auto}"
# TASK_SCOPE: "4task" (default, backward-compat 4-task subset) or "all" (all tasks, per-task WM eval)
TASK_SCOPE="${TASK_SCOPE:-4task}"

RFT_EXP_NAME="20260429_vla_adapter_w_fm_head"
RFT_STEP="400"

DATE_TAG=$(date +%Y%m%d_%H%M%S)
# Append scope suffix to output dir only for non-default scopes
# Allow OUT_ROOT override via env var (e.g. to resume into an existing dir)
if [ -z "${OUT_ROOT:-}" ]; then
  if [ "${TASK_SCOPE}" = "4task" ]; then
    OUT_ROOT="${REPO_ROOT}/results/phase0/${DATE_TAG}_${TASK_SUITE}"
  else
    OUT_ROOT="${REPO_ROOT}/results/phase0/${DATE_TAG}_${TASK_SUITE}_${TASK_SCOPE}"
  fi
fi

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
is_true() {
  case "${1}" in
    1|true|TRUE|True|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

log() { echo "[phase0] $(date +%H:%M:%S) $*" ; }

fail() { echo "[phase0 ERROR] $*" >&2 ; exit 1 ; }

# ---------------------------------------------------------------------------
# Resolve task subset from task_subset_v1.json
# ---------------------------------------------------------------------------
SUBSET_JSON="${REPO_ROOT}/configs/libero/task_subset_v1.json"
if [ ! -f "${SUBSET_JSON}" ]; then
  fail "task_subset_v1.json not found at ${SUBSET_JSON}"
fi

if is_true "${SMOKE}"; then
  TASK_INDICES_JSON=$(python3 -c "
import json
d = json.load(open('${SUBSET_JSON}'))
idxs = d['small_run']['suites'].get('${TASK_SUITE}', {}).get('task_indices', [0])
print(' '.join(str(i) for i in idxs))
")
  log "SMOKE TEST (TASK_SCOPE=${TASK_SCOPE}): task_indices = ${TASK_INDICES_JSON}"
  WM_NUM_WINDOWS=20
  WM_NUM_WINDOWS_PER_TASK="${WM_NUM_WINDOWS_PER_TASK:-5}"
  VLA_NUM_TRIALS=3
elif [ "${TASK_SCOPE}" = "all" ]; then
  TASK_INDICES_JSON=$(python3 -c "
import json
d = json.load(open('${SUBSET_JSON}'))
idxs = d['phase0_all']['suites'].get('${TASK_SUITE}', {}).get('task_indices', list(range(10)))
print(' '.join(str(i) for i in idxs))
")
  log "ALL-task run (TASK_SCOPE=all): task_indices = ${TASK_INDICES_JSON}"
  WM_NUM_WINDOWS="${WM_NUM_WINDOWS:-50}"          # used only in 4task fallback
  WM_NUM_WINDOWS_PER_TASK="${WM_NUM_WINDOWS_PER_TASK:-5}"
  VLA_NUM_TRIALS="${VLA_NUM_TRIALS:-10}"
else
  TASK_INDICES_JSON=$(python3 -c "
import json
d = json.load(open('${SUBSET_JSON}'))
idxs = d['suites'].get('${TASK_SUITE}', {}).get('task_indices', [0, 2, 5, 8])
print(' '.join(str(i) for i in idxs))
")
  log "4-task run (TASK_SCOPE=4task): task_indices = ${TASK_INDICES_JSON}"
  WM_NUM_WINDOWS="${WM_NUM_WINDOWS:-50}"
  WM_NUM_WINDOWS_PER_TASK="${WM_NUM_WINDOWS_PER_TASK:-50}"
  VLA_NUM_TRIALS="${VLA_NUM_TRIALS:-10}"
fi
read -ra TASK_INDICES <<< "${TASK_INDICES_JSON}"

# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
BASE_VLA_DIR="${REPO_ROOT}/checkpoints/libero/Base_VLA/${TASK_SUITE}"
RFT_ACTOR_DIR="${REPO_ROOT}/checkpoints/libero/VLA-RFT/${TASK_SUITE}/${RFT_EXP_NAME}/global_step_${RFT_STEP}/actor"
WM_BASE_ROOT="${REPO_ROOT}/checkpoints/libero/WorldModel"
WM_TOKENIZER="${REPO_ROOT}/checkpoints/libero/WorldModel/Tokenizer"
DATA_ROOT="${REPO_ROOT}/data/modified_libero_rlds"

# Resolve trained WorldModel directory: prefer experiment subdirs over suite root
_resolve_wm_trained_dir() {
  local suite="$1"
  local base="${WM_BASE_ROOT}/${suite}"
  # Find experiment subdirs (depth=1) that have model.safetensors (not checkpoint subdirs)
  local found
  found=$(find "${base}" -mindepth 2 -maxdepth 2 -name "model.safetensors" \
    ! -path "*/checkpoint-*" 2>/dev/null \
    | sort -rV | head -1)
  if [ -n "${found}" ]; then
    dirname "${found}"
  else
    # Fallback: suite root itself (base model)
    echo "${base}"
  fi
}
WM_TRAINED_DIR=$(_resolve_wm_trained_dir "${TASK_SUITE}")

# Validate paths
[ -d "${BASE_VLA_DIR}" ] || fail "Base VLA dir not found: ${BASE_VLA_DIR}"
[ -d "${WM_BASE_ROOT}/${TASK_SUITE}" ] || fail "WorldModel dir not found: ${WM_BASE_ROOT}/${TASK_SUITE}"
[ -f "${WM_TRAINED_DIR}/model.safetensors" ] || fail "WorldModel trained model.safetensors not found in ${WM_TRAINED_DIR}"
[ -d "${WM_TOKENIZER}" ] || fail "WorldModel Tokenizer not found: ${WM_TOKENIZER}"
[ -d "${DATA_ROOT}" ] || fail "Data root not found: ${DATA_ROOT}"

log "Base VLA      : ${BASE_VLA_DIR}"
log "RFT actor     : ${RFT_ACTOR_DIR}"
log "WorldModel    : ${WM_BASE_ROOT}/${TASK_SUITE}"
log "WM trained dir: ${WM_TRAINED_DIR}"
log "Output root   : ${OUT_ROOT}"

mkdir -p "${OUT_ROOT}"
python3 -c "import json; json.dump({'date':'${DATE_TAG}','suite':'${TASK_SUITE}','task_scope':'${TASK_SCOPE}','task_indices':[$(IFS=,; echo "${TASK_INDICES[*]}")],'seed':${SEED},'smoke':${SMOKE}}, open('${OUT_ROOT}/run_config.json','w'))"

# ---------------------------------------------------------------------------
# CONDITION 1: baseline WorldModel eval (ROI + ranking + taxonomy)
# ---------------------------------------------------------------------------
log "=== CONDITION: baseline WorldModel (TASK_SCOPE=${TASK_SCOPE}) ==="
WM_OUT="${OUT_ROOT}/worldmodel"
mkdir -p "${WM_OUT}"

if is_true "${SKIP_WM}"; then
  log "SKIP_WM=1 — skipping WorldModel eval (reusing existing outputs in ${WM_OUT})"
else

# Use the .venv5090_eval environment which the worldmodel scripts require
VENV_WM="${REPO_ROOT}/.venv5090_eval"
[ -f "${VENV_WM}/bin/activate" ] || VENV_WM="${REPO_ROOT}/.venv"

# Helper: export taxonomy from a ranking_eval dir and merge into WM_OUT/taxonomy.csv
_export_and_merge_taxonomy() {
  local _jsonl_dir="$1"
  local _out_dir="$2"
  local _label="$3"      # used as taxonomy CSV filename prefix
  if ls "${_jsonl_dir}"/*.jsonl 2>/dev/null | head -1 | grep -q jsonl; then
    for _jf in "${_jsonl_dir}"/*.jsonl; do
      python -m worldmodel.libero.export_failure_taxonomy_template \
        --jsonl  "${_jf}" \
        --task   "${TASK_SUITE}" \
        --suite  "${TASK_SUITE}" \
        --output "${_out_dir}/${_label}_taxonomy.csv" \
        2>/dev/null || true
    done
  fi
}

if [ "${TASK_SCOPE}" = "all" ]; then
  # ---- Per-task WM eval (Plan A) -------------------------------------------
  # Run visualize.py once per task index with --task-indices <idx>
  # Each task writes its own eval_report + ranking_eval/ + taxonomy CSV
  for task_idx in "${TASK_INDICES[@]}"; do
    TASK_WM_OUT="${WM_OUT}/task${task_idx}"
    mkdir -p "${TASK_WM_OUT}"
    log "  WM eval: task_idx=${task_idx} (${WM_NUM_WINDOWS_PER_TASK} windows)"
    (
      source "${VENV_WM}/bin/activate"
      export PYTHONPATH="${REPO_ROOT}/worldmodel/_compat:${REPO_ROOT}/train/verl:${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
      export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
      export TOKENIZERS_PARALLELISM="false"
      export TF_CPP_MIN_LOG_LEVEL=3

      cd "${REPO_ROOT}"
      python -m worldmodel.libero.visualize \
        --task-suite      "${TASK_SUITE}" \
        --task-indices    "${task_idx}" \
        --data-root       "${DATA_ROOT}" \
        --base-model-root "${WM_BASE_ROOT}" \
        --trained-model-dir "${WM_TRAINED_DIR}" \
        --visual-tokenizer  "${WM_TOKENIZER}" \
        --output-dir      "${TASK_WM_OUT}" \
        --device          "${DEVICE}" \
        --seed            "${SEED}" \
        --num-eval-windows "${WM_NUM_WINDOWS_PER_TASK}" \
        --eval-horizon    7 \
        --heldout-ratio   0.2 \
        --split-mode      fallback_all \
        --display-frames  8 \
        --eval-batch-size 4 \
        --save-casebook-count 3 \
        --chunk-future-length 7 \
        --compare-base \
        --run-action-sensitivity \
        --enable-roi-metrics \
        --enable-rank-logging \
        2>&1 | tee "${TASK_WM_OUT}/eval.log"

      _export_and_merge_taxonomy \
        "${TASK_WM_OUT}/ranking_eval" \
        "${TASK_WM_OUT}" \
        "task${task_idx}"
    ) && log "  WM task${task_idx} done" \
      || log "  WARNING: WM task${task_idx} failed (check ${TASK_WM_OUT}/eval.log)"
  done

  # Merge all per-task taxonomy CSVs into WM_OUT/taxonomy.csv
  (
    source "${VENV_WM}/bin/activate"
    python3 -c "
import csv, glob, os
rows = []
for p in sorted(glob.glob('${WM_OUT}/task*/*_taxonomy.csv')):
    try:
        with open(p) as f:
            reader = csv.DictReader(f)
            for r in reader:
                r['_task_dir'] = os.path.basename(os.path.dirname(p))
                rows.append(r)
    except Exception:
        pass
if rows:
    keys = list(rows[0].keys())
    with open('${WM_OUT}/taxonomy.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print(f'Merged {len(rows)} taxonomy rows from {len(set(r[\"_task_dir\"] for r in rows))} tasks')
" 2>/dev/null || true
  )
  log "WorldModel per-task eval done"

else
  # ---- Original single-call WM eval (4task / smoke) ------------------------
  (
    source "${VENV_WM}/bin/activate"
    export PYTHONPATH="${REPO_ROOT}/worldmodel/_compat:${REPO_ROOT}/train/verl:${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export TOKENIZERS_PARALLELISM="false"
    export TF_CPP_MIN_LOG_LEVEL=3

    cd "${REPO_ROOT}"
    python -m worldmodel.libero.visualize \
      --task-suite "${TASK_SUITE}" \
      --data-root "${DATA_ROOT}" \
      --base-model-root "${WM_BASE_ROOT}" \
      --trained-model-dir "${WM_TRAINED_DIR}" \
      --visual-tokenizer "${WM_TOKENIZER}" \
      --output-dir "${WM_OUT}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --num-eval-windows "${WM_NUM_WINDOWS}" \
      --eval-horizon 7 \
      --heldout-ratio 0.2 \
      --split-mode fallback_all \
      --display-frames 8 \
      --eval-batch-size 4 \
      --save-casebook-count 5 \
      --chunk-future-length 7 \
      --compare-base \
      --run-action-sensitivity \
      --enable-roi-metrics \
      --enable-rank-logging \
      2>&1 | tee "${WM_OUT}/eval.log"

    _WM_JSONL_DIR="${WM_OUT}/ranking_eval"
    if ls "${_WM_JSONL_DIR}"/*.jsonl 2>/dev/null | head -1 | grep -q jsonl; then
      for _jf in "${_WM_JSONL_DIR}"/*.jsonl; do
        _tax_name="$(basename "${_jf}" .jsonl)_taxonomy.csv"
        python -m worldmodel.libero.export_failure_taxonomy_template \
          --jsonl   "${_jf}" \
          --task    "${TASK_SUITE}" \
          --suite   "${TASK_SUITE}" \
          --output  "${WM_OUT}/${_tax_name}" \
          2>/dev/null || true
      done
      python3 -c "
import csv, glob
rows = []
for p in sorted(glob.glob('${WM_OUT}/*_taxonomy.csv')):
    with open(p) as f:
        reader = csv.DictReader(f)
        rows.extend(list(reader))
if rows:
    with open('${WM_OUT}/taxonomy.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print(f'Merged {len(rows)} taxonomy rows')
" 2>/dev/null || true
    fi
  ) && log "WorldModel eval done" || log "WARNING: WorldModel eval failed (check ${WM_OUT}/eval.log)"
fi

fi  # end SKIP_WM

# Copy key metrics to output root
for f in metrics.json per_task_summary.csv; do
  src=$(find "${WM_OUT}" -name "${f}" 2>/dev/null -print -quit)
  [ -n "${src}" ] && cp "${src}" "${OUT_ROOT}/worldmodel_${f}" || true
done
[ -f "${WM_OUT}/taxonomy.csv" ] && cp "${WM_OUT}/taxonomy.csv" "${OUT_ROOT}/worldmodel_taxonomy.csv" || true

# ---------------------------------------------------------------------------
# CONDITION 2: Base VLA evaluation (simulation)
# ---------------------------------------------------------------------------
if is_true "${SKIP_VLA}"; then
  log "SKIP_VLA=1 — skipping VLA simulation evals"
else
  log "=== CONDITION: Base VLA ==="
  BASE_VLA_OUT="${OUT_ROOT}/base_vla"
  mkdir -p "${BASE_VLA_OUT}"

  VENV_VLA="${REPO_ROOT}/.venv5090_eval"
  [ -f "${VENV_VLA}/bin/activate" ] || VENV_VLA="${REPO_ROOT}/.venv"

  for task_idx in "${TASK_INDICES[@]}"; do
    task_id_1based=$((task_idx + 1))
    log "  Base VLA: task_idx=${task_idx} (id_1based=${task_id_1based})"
    (
      source "${VENV_VLA}/bin/activate"
      export PYTHONPATH="${REPO_ROOT}/third_party/LIBERO:${REPO_ROOT}/train/verl:${PYTHONPATH:-}"
      export MUJOCO_GL="${MUJOCO_GL:-egl}"
      export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
      export TF_CPP_MIN_LOG_LEVEL=3
      cd "${REPO_ROOT}/train/verl/vla-adapter/openvla-oft"

      python experiments/robot/libero/run_libero_eval.py \
        --use_l1_regression False \
        --use_diffusion False \
        --use_flow_matching True \
        --use_proprio True \
        --use_film False \
        --num_images_in_input 1 \
        --pretrained_checkpoint "${BASE_VLA_DIR}" \
        --actor_model_version "BASE" \
        --task_suite_name "libero_${TASK_SUITE}" \
        --run_single_task True \
        --single_task_id "${task_id_1based}" \
        --num_trials_per_task "${VLA_NUM_TRIALS}" \
        --save_version v1 \
        --use_minivla True \
        --run_id_note "BASE_phase0" \
        --local_log_dir "${BASE_VLA_OUT}/task${task_id_1based}" \
        --seed "${SEED}" \
        2>&1 | tee "${BASE_VLA_OUT}/task${task_id_1based}.log"
    ) || log "  WARNING: Base VLA task ${task_idx} failed"
  done
  log "Base VLA eval done"

  # ---------------------------------------------------------------------------
  # CONDITION 3: VLA-RFT evaluation (simulation)
  # ---------------------------------------------------------------------------
  log "=== CONDITION: VLA-RFT ==="
  RFT_OUT="${OUT_ROOT}/vla_rft"
  mkdir -p "${RFT_OUT}"

  if [ ! -d "${RFT_ACTOR_DIR}" ]; then
    log "WARNING: RFT actor dir not found: ${RFT_ACTOR_DIR}. Skipping VLA-RFT."
  else
    for task_idx in "${TASK_INDICES[@]}"; do
      task_id_1based=$((task_idx + 1))
      log "  VLA-RFT: task_idx=${task_idx} (id_1based=${task_id_1based})"
      (
        source "${VENV_VLA}/bin/activate"
        export PYTHONPATH="${REPO_ROOT}/third_party/LIBERO:${REPO_ROOT}/train/verl:${PYTHONPATH:-}"
        export MUJOCO_GL="${MUJOCO_GL:-egl}"
        export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
        export TF_CPP_MIN_LOG_LEVEL=3
        cd "${REPO_ROOT}/train/verl/vla-adapter/openvla-oft"

        python experiments/robot/libero/run_libero_eval.py \
          --use_l1_regression False \
          --use_diffusion False \
          --use_flow_matching True \
          --use_proprio True \
          --use_film False \
          --num_images_in_input 1 \
          --pretrained_checkpoint "${BASE_VLA_DIR}" \
          --actor_path "${RFT_ACTOR_DIR}" \
          --actor_model_version "${RFT_EXP_NAME}" \
          --task_suite_name "libero_${TASK_SUITE}" \
          --run_single_task True \
          --single_task_id "${task_id_1based}" \
          --num_trials_per_task "${VLA_NUM_TRIALS}" \
          --save_version v1 \
          --use_minivla True \
          --run_id_note "RFT_phase0" \
          --local_log_dir "${RFT_OUT}/task${task_id_1based}" \
          --seed "${SEED}" \
          2>&1 | tee "${RFT_OUT}/task${task_id_1based}.log"
      ) || log "  WARNING: VLA-RFT task ${task_idx} failed"
    done
    log "VLA-RFT eval done"
  fi
fi

# ---------------------------------------------------------------------------
# Final summary pointer
# ---------------------------------------------------------------------------
log "=== Phase 0 eval complete ==="
log "Output: ${OUT_ROOT}"
log "Run summarizer:"
log "  bash scripts/libero/phase0/summarize_phase0_eval.sh ${OUT_ROOT}"

echo "${OUT_ROOT}" > "${REPO_ROOT}/results/phase0/latest.txt"
