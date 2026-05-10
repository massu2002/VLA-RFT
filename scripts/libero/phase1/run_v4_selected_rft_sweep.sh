#!/usr/bin/env bash
# run_v4_selected_rft_sweep.sh — Run RFT sweep over the best v4 checkpoint(s).
#
# Reads selected_v4_for_rft.json (written by select_best_v4_for_rft.sh),
# then launches RFT for each reward type in WORLD_REWARD_TYPE_LIST.
#
# Usage:
#   RUN_NAME=v4_core_sweep_spatial \
#     bash scripts/libero/phase1/run_v4_selected_rft_sweep.sh
#
#   # Override reward types
#   WORLD_REWARD_TYPE_LIST="visual hybrid rank_score" \
#   RUN_NAME=v4_core_sweep_spatial \
#     bash scripts/libero/phase1/run_v4_selected_rft_sweep.sh
#
#   # Smoke test (tiny steps/trials)
#   SMOKE=1 RUN_NAME=v4_core_sweep_spatial \
#     bash scripts/libero/phase1/run_v4_selected_rft_sweep.sh
#
#   # Dry run — print plan without executing
#   DRY_RUN=1 RUN_NAME=v4_core_sweep_spatial \
#     bash scripts/libero/phase1/run_v4_selected_rft_sweep.sh
#
# Key env vars:
#   RUN_NAME                 sweep run name (used to locate SELECTED_JSON)
#   SELECTED_JSON            path to selected_v4_for_rft.json
#                            default: results/phase1/residual_worldmodel/${RUN_NAME}/selected_for_rft/selected_v4_for_rft.json
#   WORLD_REWARD_TYPE_LIST   space-separated reward types to sweep (default: "visual hybrid rank_score")
#   RFT_STEPS                number of RFT gradient steps (default: 400)
#   NUM_TRIALS               robot eval trials per task (default: 50)
#   TASK_SUITE               spatial | object | goal | 10 (default: spatial)
#   SEED                     RFT random seed (default: 7)
#   SMOKE                    0|1 — tiny steps/trials for fast verification
#   DRY_RUN                  0|1 — print commands without executing
#   SKIP_EXISTING            0|1 — skip if RFT output dir already exists (default: 1)
#   OVERWRITE                0|1 — force re-run even if outputs exist
#   GPU_IDS                  comma-separated GPU indices or "auto"
#   OUT_ROOT                 results root for this sweep run

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"

source "${WM_SCRIPTS}/common.sh"

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES) return 0 ;; *) return 1 ;; esac; }
log()     { echo "[v4-rft-sweep] $(date +%H:%M:%S) $*"; }
die()     { echo "[v4-rft-sweep] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:?'RUN_NAME is required'}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/residual_worldmodel/${RUN_NAME}}"
SELECTED_JSON="${SELECTED_JSON:-${OUT_ROOT}/selected_for_rft/selected_v4_for_rft.json}"

WORLD_REWARD_TYPE_LIST="${WORLD_REWARD_TYPE_LIST:-visual hybrid rank_score}"
RFT_STEPS="${RFT_STEPS:-400}"
NUM_TRIALS="${NUM_TRIALS:-50}"
TASK_SUITE="${TASK_SUITE:-spatial}"
SEED="${SEED:-7}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OVERWRITE="${OVERWRITE:-0}"
GPU_IDS="${GPU_IDS:-auto}"

# Smoke overrides
if is_true "${SMOKE}"; then
  RFT_STEPS="${RFT_STEPS_SMOKE:-20}"
  NUM_TRIALS="${NUM_TRIALS_SMOKE:-4}"
fi

# Rank reward hyperparameters (passed through)
RANK_REWARD_ALPHA="${RANK_REWARD_ALPHA:-0.2}"
RANK_REWARD_BETA="${RANK_REWARD_BETA:-0.8}"
NORMALIZE_RANK_REWARD="${NORMALIZE_RANK_REWARD:-1}"
CLIP_RANK_REWARD="${CLIP_RANK_REWARD:-1}"
RANK_REWARD_CLIP_VALUE="${RANK_REWARD_CLIP_VALUE:-5.0}"

[ -f "${SELECTED_JSON}" ] || die "SELECTED_JSON not found: ${SELECTED_JSON}
  Run first: RUN_NAME=${RUN_NAME} bash scripts/libero/residual_worldmodel/select_best_v4_for_rft.sh"

setup_env

print_header "run_v4_selected_rft_sweep: ${RUN_NAME}"
printf "  %-28s %s\n" "SELECTED_JSON"      "${SELECTED_JSON}"
printf "  %-28s %s\n" "REWARD_TYPES"       "${WORLD_REWARD_TYPE_LIST}"
printf "  %-28s %s\n" "RFT_STEPS"          "${RFT_STEPS}"
printf "  %-28s %s\n" "NUM_TRIALS"         "${NUM_TRIALS}"
printf "  %-28s %s\n" "TASK_SUITE"         "${TASK_SUITE}"
printf "  %-28s %s\n" "SEED"               "${SEED}"
printf "  %-28s %s\n" "SMOKE"              "${SMOKE}"
printf "  %-28s %s\n" "DRY_RUN"            "${DRY_RUN}"
printf "  %-28s %s\n" "SKIP_EXISTING"      "${SKIP_EXISTING}"
printf "  %-28s %s\n" "RANK_REWARD_ALPHA"  "${RANK_REWARD_ALPHA}"
printf "  %-28s %s\n" "RANK_REWARD_BETA"   "${RANK_REWARD_BETA}"
printf '%*s\n' 68 '' | tr ' ' '-'

# ---------------------------------------------------------------------------
# Parse selected_v4_for_rft.json  →  emit rows: exp_name|ckpt_dir|stage
# ---------------------------------------------------------------------------
SELECTED_ROWS="$(python3 - <<PYEOF
import json, sys
data = json.load(open("${SELECTED_JSON}"))
for entry in data.get("selected", []):
    exp  = entry.get("exp_name", "")
    ckpt = entry.get("ckpt_dir", "")
    stage = entry.get("stage", "v4b")
    print(f"{exp}|{ckpt}|{stage}")
PYEOF
)"

if [ -z "${SELECTED_ROWS}" ]; then
  die "No selected conditions found in ${SELECTED_JSON}"
fi

NUM_SELECTED=$(echo "${SELECTED_ROWS}" | wc -l)
NUM_REWARD_TYPES=$(echo "${WORLD_REWARD_TYPE_LIST}" | wc -w)
TOTAL_JOBS=$(( NUM_SELECTED * NUM_REWARD_TYPES ))

log "Selected conditions : ${NUM_SELECTED}"
log "Reward types        : ${WORLD_REWARD_TYPE_LIST}"
log "Total RFT jobs      : ${TOTAL_JOBS}"
log ""

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
job_id=0
fail_count=0
success_count=0
skip_count=0

while IFS='|' read -r base_exp ckpt_dir stage; do
  for reward_type in ${WORLD_REWARD_TYPE_LIST}; do
    job_id=$(( job_id + 1 ))

    # Derived names
    rft_exp="${base_exp}_rft_${reward_type}"
    rft_out="${OUT_ROOT}/selected_for_rft/rft_runs/${rft_exp}"

    log "────────────────────────────────────────────────────────────"
    log "Job ${job_id}/${TOTAL_JOBS}: ${rft_exp}"
    log "  base_exp    : ${base_exp}"
    log "  stage       : ${stage}"
    log "  ckpt_dir    : ${ckpt_dir}"
    log "  reward_type : ${reward_type}"
    log "  rft_out     : ${rft_out}"

    # SKIP_EXISTING check
    if ! is_true "${OVERWRITE}" && is_true "${SKIP_EXISTING}"; then
      if [ -f "${rft_out}/rft_complete.txt" ] || [ -f "${rft_out}/training_complete.txt" ]; then
        log "  SKIP (output exists)"
        skip_count=$(( skip_count + 1 ))
        continue
      fi
    fi

    if ! is_true "${DRY_RUN}" && [ ! -d "${ckpt_dir}" ]; then
      log "  SKIP: ckpt_dir does not exist: ${ckpt_dir}"
      skip_count=$(( skip_count + 1 ))
      continue
    fi

    if is_true "${DRY_RUN}"; then
      log "  [DRY_RUN] would call run_v4_temporal_query_wm_rft_only.sh"
      log "    WORLD_MODEL_CKPT=${ckpt_dir}"
      log "    WORLD_REWARD_TYPE=${reward_type}"
      log "    EXP_NAME=${rft_exp}  RFT_STEPS=${RFT_STEPS}  SEED=${SEED}"
      continue
    fi

    mkdir -p "${rft_out}"

    set +e
    (
      export WORLD_MODEL_CKPT="${ckpt_dir}"
      export WORLD_MODEL_CONFIG="${ckpt_dir}/v4_config.json"
      export MODEL_GENERATION="v4"
      export TARGET_MODE="temporal_query_residual"
      export WORLD_REWARD_TYPE="${reward_type}"
      export EXP_NAME="${rft_exp}"
      export OUTPUT_DIR="${rft_out}"
      export TASK_SUITE="${TASK_SUITE}"
      export RFT_STEPS="${RFT_STEPS}"
      export NUM_TRIALS="${NUM_TRIALS}"
      export SEED="${SEED}"
      export SMOKE="${SMOKE}"
      export RANK_REWARD_ALPHA="${RANK_REWARD_ALPHA}"
      export RANK_REWARD_BETA="${RANK_REWARD_BETA}"
      export NORMALIZE_RANK_REWARD="${NORMALIZE_RANK_REWARD}"
      export CLIP_RANK_REWARD="${CLIP_RANK_REWARD}"
      export RANK_REWARD_CLIP_VALUE="${RANK_REWARD_CLIP_VALUE}"
      export GPU_IDS="${GPU_IDS}"
      export DRY_RUN="0"
      bash "${SCRIPT_DIR}/run_v4_temporal_query_wm_rft_only.sh"
    ) 2>&1 | tee "${rft_out}/rft_run.log"
    rc="${PIPESTATUS[0]}"
    set -e

    if [ "${rc}" -eq 0 ]; then
      log "  DONE: ${rft_exp}"
      success_count=$(( success_count + 1 ))
      date -u +"%Y-%m-%dT%H:%M:%SZ  rc=0" > "${rft_out}/rft_complete.txt"
    else
      log "  FAILED: ${rft_exp} (rc=${rc})"
      fail_count=$(( fail_count + 1 ))
    fi
  done
done <<< "${SELECTED_ROWS}"

log "══════════════════════════════════════════════════════════════"
if is_true "${DRY_RUN}"; then
  log "DRY_RUN: total planned jobs = ${job_id}"
  log "  Set DRY_RUN=0 to execute."
else
  log "RFT sweep complete."
  log "  success=${success_count}  fail=${fail_count}  skip=${skip_count}"
  log "  results: ${OUT_ROOT}/selected_for_rft/rft_runs/"
  if [ "${fail_count}" -gt 0 ]; then
    log ""
    log "  Re-run failures only:"
    log "    SKIP_EXISTING=1 OVERWRITE=0 RUN_NAME=${RUN_NAME} \\"
    log "      bash scripts/libero/phase1/run_v4_selected_rft_sweep.sh"
  fi
fi

[ "${fail_count}" -gt 0 ] && exit 1
exit 0
