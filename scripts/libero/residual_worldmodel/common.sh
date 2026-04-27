#!/usr/bin/env bash
# common.sh — shared utilities for residual worldmodel scripts.
#
# MUST be sourced, not executed directly:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# What belongs here:
#   environment setup, utility functions, run_name generation,
#   save_dir creation, config dump, command builders.
#
# What does NOT belong here:
#   experiment conditions (LR, BATCH_SIZE, MODEL_VARIANT, etc.)
#   — those stay in each calling script's USER CONFIG block.

# ---------------------------------------------------------------------------
# Guard: prevent double-sourcing
# ---------------------------------------------------------------------------
if [ -n "${_RESIDUAL_WM_COMMON_LOADED:-}" ]; then return 0; fi
_RESIDUAL_WM_COMMON_LOADED=1

# ---------------------------------------------------------------------------
# Repo root (resolved once; callers may override before sourcing)
# ---------------------------------------------------------------------------
if [ -z "${REPO_ROOT:-}" ]; then
  _COMMON_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  export REPO_ROOT=$(cd "${_COMMON_DIR}/../../.." && pwd)
fi

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
_VENV_NAME="${VENV_NAME:-".venv"}"
_DEFAULT_LIBERO_DATASET_SUBDIR="modified_libero_rlds"

setup_env() {
  # Activate venv and set PYTHONPATH.
  local venv_path="${REPO_ROOT}/${_VENV_NAME}"
  if [ ! -f "${venv_path}/bin/activate" ]; then
    echo "[common] ERROR: venv not found at ${venv_path}" >&2
    echo "[common] Set VENV_NAME to override (default: .venv)" >&2
    exit 1
  fi
  source "${venv_path}/bin/activate"
  export PYTHONPATH="${REPO_ROOT}/worldmodel/_compat:${REPO_ROOT}/train/verl:${PYTHONPATH:-}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export TOKENIZERS_PARALLELISM="false"
}

# ---------------------------------------------------------------------------
# Dataset path helpers
# ---------------------------------------------------------------------------

default_libero_data_root() {
  if [ -n "${LIBERO_DATA_ROOT:-}" ]; then
    echo "${LIBERO_DATA_ROOT}"
  elif [ -n "${LOCALDATA_ROOT:-}" ]; then
    echo "${LOCALDATA_ROOT%/}/${_DEFAULT_LIBERO_DATASET_SUBDIR}"
  else
    echo "${REPO_ROOT}/data/${_DEFAULT_LIBERO_DATASET_SUBDIR}"
  fi
}

resolve_libero_data_root() {
  local configured_path="${1:-}"
  if [ -n "${configured_path}" ]; then
    echo "${configured_path}"
  else
    default_libero_data_root
  fi
}

# ---------------------------------------------------------------------------
# Task schedule helpers
# ---------------------------------------------------------------------------

validate_task_name() {
  local task_name="${1:-}"
  case "${task_name}" in
    spatial|object|goal|long|10) return 0 ;;
    *)
      echo "[common] ERROR: unsupported TASK_NAME='${task_name}'" >&2
      echo "[common] Allowed task names: spatial | object | goal | long" >&2
      echo "[common] (legacy alias '10' is also accepted and mapped to 'long')" >&2
      return 1
      ;;
  esac
}

normalize_libero_task_name() {
  local task_name="${1:-}"
  validate_task_name "${task_name}" || return 1
  case "${task_name}" in
    long|10) echo "10" ;;
    *) echo "${task_name}" ;;
  esac
}

default_task_schedule_key() {
  local task_name="${1:-}"
  validate_task_name "${task_name}" || return 1
  case "${task_name}" in
    10) echo "long" ;;
    *) echo "${task_name}" ;;
  esac
}

validate_task_schedule_key() {
  local task_key="${1:-}"
  case "${task_key}" in
    spatial|object|goal|long|10|spatial_debug|object_debug|goal_debug|long_debug|10_debug)
      return 0
      ;;
    *)
      echo "[common] ERROR: unsupported task schedule key='${task_key}'" >&2
      echo "[common] Allowed schedule keys:" >&2
      echo "[common]   spatial | object | goal | long" >&2
      echo "[common]   spatial_debug | object_debug | goal_debug | long_debug" >&2
      return 1
      ;;
  esac
}

resolve_task_schedule_key() {
  local task_name="${1:?'resolve_task_schedule_key: task_name required'}"
  local override_key="${2:-}"
  local task_key=""

  validate_task_name "${task_name}" || return 1

  if [ -n "${override_key}" ]; then
    task_key="${override_key}"
  else
    task_key="$(default_task_schedule_key "${task_name}")"
  fi

  validate_task_schedule_key "${task_key}" || return 1
  case "${task_key}" in
    10) echo "long" ;;
    10_debug) echo "long_debug" ;;
    *) echo "${task_key}" ;;
  esac
}

get_max_steps_for_task() {
  local task_key="${1:?'get_max_steps_for_task: task key required'}"
  validate_task_schedule_key "${task_key}" || return 1
  case "${task_key}" in
    spatial|object) echo "50000" ;;
    goal)           echo "40000" ;;
    long|10)        echo "80000" ;;
    spatial_debug|object_debug) echo "3000" ;;
    goal_debug)     echo "2000" ;;
    long_debug|10_debug) echo "5000" ;;
  esac
}

get_eval_every_for_task() {
  local task_key="${1:?'get_eval_every_for_task: task key required'}"
  validate_task_schedule_key "${task_key}" || return 1
  case "${task_key}" in
    spatial|object|goal) echo "1000" ;;
    long|10)             echo "2000" ;;
    spatial_debug|object_debug|goal_debug) echo "100" ;;
    long_debug|10_debug) echo "200" ;;
  esac
}

get_vis_every_for_task() {
  local task_key="${1:?'get_vis_every_for_task: task key required'}"
  validate_task_schedule_key "${task_key}" || return 1
  case "${task_key}" in
    spatial|object|goal) echo "2000" ;;
    long|10)             echo "5000" ;;
    spatial_debug|object_debug|goal_debug) echo "200" ;;
    long_debug|10_debug) echo "500" ;;
  esac
}

resolve_task_schedule_value() {
  local resolver="${1:?'resolve_task_schedule_value: resolver required'}"
  local task_key="${2:?'resolve_task_schedule_value: task key required'}"
  local override_value="${3:-}"
  if [ -n "${override_value}" ]; then
    echo "${override_value}"
  else
    "${resolver}" "${task_key}"
  fi
}

print_task_schedule_table() {
  local keys=("spatial" "object" "goal" "long" "spatial_debug" "object_debug" "goal_debug" "long_debug")
  print_header "Task schedule defaults"
  printf "  %-16s %-12s %-12s %-12s\n" "TASK_KEY" "MAX_STEPS" "EVAL_EVERY" "VIS_EVERY"
  printf "  %s\n" "$(printf '%0.s-' {1..58})"
  local key=""
  for key in "${keys[@]}"; do
    printf "  %-16s %-12s %-12s %-12s\n" \
      "${key}" \
      "$(get_max_steps_for_task "${key}")" \
      "$(get_eval_every_for_task "${key}")" \
      "$(get_vis_every_for_task "${key}")"
  done
  printf '%*s\n' 68 '' | tr ' ' '-'
}

# ---------------------------------------------------------------------------
# Timestamp / GPU
# ---------------------------------------------------------------------------

timestamp() {
  # Returns YYYYMMDD_HHMMSS
  date +%Y%m%d_%H%M%S
}

short_date() {
  # Returns YYYYMMDD
  date +%Y%m%d
}

detect_gpu_count() {
  # Returns number of CUDA GPUs (at least 1)
  "${REPO_ROOT}/${_VENV_NAME}/bin/python" \
    -c 'import torch; print(max(torch.cuda.device_count(), 1))' 2>/dev/null || echo "1"
}

# ---------------------------------------------------------------------------
# Run name generation
# ---------------------------------------------------------------------------

make_run_name() {
  # Usage: make_run_name VARIANT DINO LR BS HOR [EXTRA...]
  # Returns a compact but human-readable run identifier.
  # Task and seed are NOT included — both are expressed as directory levels.
  #
  # Args:
  #   $1  MODEL_VARIANT  e.g. "full", "no_focus"
  #   $2  DINO_BACKBONE  e.g. "dinov2_vits14"
  #   $3  LR             e.g. "1e-4"
  #   $4  BATCH_SIZE     e.g. "4"
  #   $5  ACTION_HORIZON e.g. "7"
  #   $6+ EXTRA tags     e.g. "noLPIPS"
  local variant="${1:-full}"
  local dino="${2:-dinov2_vits14}"
  local lr="${3:-1e-4}"
  local bs="${4:-4}"
  local hor="${5:-7}"
  shift 5 2>/dev/null || true
  local _extra_raw="$*"
  local extra="${_extra_raw:+_${_extra_raw// /_}}"

  # Shorten backbone: dinov2_vits14 → vits14
  local dino_short="${dino##*_}"   # strip prefix up to last _
  dino_short="${dino_short:-${dino}}"

  # Shorten variant: no_focus → nofocus
  local v_short="${variant//_/}"

  echo "$(short_date)_${v_short}_${dino_short}_lr${lr}_bs${bs}_hor${hor}${extra}"
}

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

ensure_dir() {
  mkdir -p "$1"
}

make_save_dir() {
  # Usage: make_save_dir OUTPUT_ROOT RUN_NAME
  local root="${1:?'make_save_dir: OUTPUT_ROOT required'}"
  local name="${2:?'make_save_dir: RUN_NAME required'}"
  local dir="${root}/${name}"
  mkdir -p "${dir}"
  echo "${dir}"
}

# ---------------------------------------------------------------------------
# Config / snapshot persistence
# ---------------------------------------------------------------------------

dump_run_config() {
  # Saves a JSON of the current environment to $1/config_dump.json
  # Usage: dump_run_config SAVE_DIR [key=val ...]
  local save_dir="${1:?'dump_run_config: SAVE_DIR required'}"
  shift
  ensure_dir "${save_dir}"

  local json="{\n"
  json+="  \"timestamp\": \"$(timestamp)\",\n"
  json+="  \"repo_root\": \"${REPO_ROOT}\",\n"
  json+="  \"git_commit\": \"$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')\",\n"
  json+="  \"git_branch\": \"$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')\",\n"

  # Extra key=val pairs from caller
  for kv in "$@"; do
    local key="${kv%%=*}"
    local val="${kv#*=}"
    # Escape double quotes in value
    val="${val//\"/\\\"}"
    json+="  \"${key}\": \"${val}\",\n"
  done

  json+="  \"_end\": true\n}"
  printf "%b" "${json}" > "${save_dir}/config_dump.json"
}

copy_script_snapshot() {
  # Copies the calling script to SAVE_DIR/script_snapshot.sh
  # Usage: copy_script_snapshot SAVE_DIR SCRIPT_PATH
  local save_dir="${1:?'copy_script_snapshot: SAVE_DIR required'}"
  local script="${2:-${BASH_SOURCE[1]:-$0}}"
  ensure_dir "${save_dir}"
  if [ -f "${script}" ]; then
    cp "${script}" "${save_dir}/script_snapshot.sh"
  fi
}

save_cmd() {
  # Saves the command array to SAVE_DIR/cmd.sh for reproducibility.
  # Usage: save_cmd SAVE_DIR [cmd args...]
  local save_dir="${1:?'save_cmd: SAVE_DIR required'}"
  shift
  ensure_dir "${save_dir}"
  {
    echo "#!/usr/bin/env bash"
    echo "# Auto-generated command snapshot — $(timestamp)"
    echo "cd ${REPO_ROOT}"
    echo "source ${REPO_ROOT}/${_VENV_NAME}/bin/activate"
    echo "export PYTHONPATH=\"${REPO_ROOT}/worldmodel/_compat:${REPO_ROOT}/train/verl:\${PYTHONPATH:-}\""
    printf '%q ' "$@"
    echo
  } > "${save_dir}/cmd.sh"
  chmod +x "${save_dir}/cmd.sh"
}

# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

print_header() {
  local title="${1:-Run}"
  local w=68
  printf '%*s\n' "${w}" '' | tr ' ' '='
  printf "  %s\n" "${title}"
  printf '%*s\n' "${w}" '' | tr ' ' '='
}

print_run_summary() {
  # Usage: print_run_summary KEY VAL [KEY VAL ...]
  print_header "Run configuration"
  while [ $# -ge 2 ]; do
    printf "  %-28s %s\n" "$1" "$2"
    shift 2
  done
  printf '%*s\n' 68 '' | tr ' ' '-'
}

# ---------------------------------------------------------------------------
# Command runners
# ---------------------------------------------------------------------------

run_train_command() {
  # Usage: run_train_command NPROC SAVE_DIR LOGFILE [python args...]
  local nproc="${1:?'run_train_command: NPROC required'}"
  local save_dir="${2:?'run_train_command: SAVE_DIR required'}"
  local logfile="${3:?'run_train_command: LOGFILE required'}"
  shift 3

  ensure_dir "${save_dir}"
  ensure_dir "$(dirname "${logfile}")"

  save_cmd "${save_dir}" python "$@"

  if (( nproc == 1 )); then
    "${REPO_ROOT}/${_VENV_NAME}/bin/python" "$@" 2>&1 | tee "${logfile}"
  else
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="${nproc}" \
      "$@" 2>&1 | tee "${logfile}"
  fi
}

run_visualize_command() {
  # Usage: run_visualize_command SAVE_DIR LOGFILE [python args...]
  local save_dir="${1:?'run_visualize_command: SAVE_DIR required'}"
  local logfile="${2:?'run_visualize_command: LOGFILE required'}"
  shift 2

  ensure_dir "${save_dir}"
  ensure_dir "$(dirname "${logfile}")"

  save_cmd "${save_dir}" python "$@"
  "${REPO_ROOT}/${_VENV_NAME}/bin/python" "$@" 2>&1 | tee "${logfile}"
}

# ---------------------------------------------------------------------------
# Checkpoint finder
# ---------------------------------------------------------------------------

find_checkpoint() {
  # Returns path to checkpoint directory inside RUN_DIR.
  # Usage: find_checkpoint RUN_DIR PREFERENCE
  # PREFERENCE: best_recon | best_dino_feature | best_rank | final | latest | auto
  local run_dir="${1:?'find_checkpoint: RUN_DIR required'}"
  local pref="${2:-auto}"

  if [ "${pref}" = "auto" ]; then
    # Priority: best_recon > best_rank > final > latest
    for tag in best_recon best_rank best_dino_feature final latest; do
      local found
      found=$(find "${run_dir}" -maxdepth 1 -type d -name "checkpoint-${tag}-*" 2>/dev/null | sort -r | head -1)
      if [ -n "${found}" ]; then
        echo "${found}"
        return 0
      fi
    done
  else
    local found
    found=$(find "${run_dir}" -maxdepth 1 -type d -name "checkpoint-${pref}-*" 2>/dev/null | sort -r | head -1)
    if [ -n "${found}" ]; then
      echo "${found}"
      return 0
    fi
  fi

  # Fallback: any checkpoint dir
  local any
  any=$(find "${run_dir}" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -r | head -1)
  if [ -n "${any}" ]; then
    echo "${any}"
    return 0
  fi

  echo ""
  return 1
}
