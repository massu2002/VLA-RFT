#!/usr/bin/env bash
# Evaluate saved TemporalQueryResidualWM-RFT policies on the target LIBERO task suite.
#
# Results are written to:
#   results/phase1/v4_improved_spatial_rft/
#
# Examples:
#   bash command_eval_rft.sh
#   SMOKE=1 bash command_eval_rft.sh
#   RFT_STEP=400 NUM_TRIALS=50 CUDA_VISIBLE_DEVICES=0 bash command_eval_rft.sh
#   SEEDS="7 8 9" bash command_eval_rft.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

export RFT_RUN_NAME="${RFT_RUN_NAME:-v4_improved_spatial}"
export TASK_SUITE="${TASK_SUITE:-spatial}"
export RFT_CKPT_ROOT="${RFT_CKPT_ROOT:-${SCRIPT_DIR}/checkpoints/libero/TemporalQueryResidualWM-RFT/${RFT_RUN_NAME}}"
export BASELINE_RFT_CKPT_ROOT="${BASELINE_RFT_CKPT_ROOT:-${SCRIPT_DIR}/checkpoints/libero/TemporalQueryResidualWM-RFT/baseline_phase0_ar_pixel}"
export OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/results/phase1/v4_improved_spatial_rft}"
export RFT_STEP="${RFT_STEP:-latest}"
export NUM_TRIALS="${NUM_TRIALS:-50}"
export SEEDS="${SEEDS:-${SEED:-7 8 9}}"
export GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/.venv5090_eval}"
export BASE_VLA_PATH="${BASE_VLA_PATH:-${SCRIPT_DIR}/checkpoints/libero/Base_VLA/${TASK_SUITE}}"
export SEED_SUBDIR="${SEED_SUBDIR:-1}"

# Default experiments include v4 RFT policies and the phase0 AR-pixel baseline RFT policy.
# Override with EXPERIMENTS="exp_a exp_b" to run a subset, or EXPERIMENTS=auto to discover all.
export EXPERIMENTS="${EXPERIMENTS:-phase1_baseline_phase0_ar_pixel phase1_v4a_q8_k2_motion_rft_lpips_mae phase1_v4b_q8_k2_rank1_mixedneg_rft_hybrid phase1_v4b_q8_k2_rank1_mixedneg_rft_rank_score phase1_v4b_q8_k2_rank1_motion_rft_hybrid phase1_v4b_q8_k2_rank1_motion_rft_rank_score phase1_v4b_q8_k2_rank2_motion_rft_hybrid phase1_v4b_q8_k2_rank2_motion_rft_rank_score}"

settings_file="${OUT_ROOT}/multi_seed_experiment_settings.md"
mkdir -p "${OUT_ROOT}"
{
  echo "# RFT LIBERO Multi-Seed Evaluation"
  echo
  echo "- Generated: $(date -Is)"
  echo "- Task suite: \`${TASK_SUITE}\`"
  echo "- Seeds: \`${SEEDS}\`"
  echo "- Output root: \`${OUT_ROOT}\`"
  echo "- RFT checkpoint root: \`${RFT_CKPT_ROOT}\`"
  echo "- Baseline RFT checkpoint root: \`${BASELINE_RFT_CKPT_ROOT}\`"
  echo "- Experiments: \`${EXPERIMENTS}\`"
  echo
  echo "| Seed | Settings file | Manifest |"
  echo "|---:|---|---|"
} > "${settings_file}"

for seed in ${SEEDS}; do
  echo "============================================================"
  echo "RFT LIBERO eval: seed=${seed}"
  echo "  OUT_ROOT=${OUT_ROOT}"
  echo "  EXPERIMENTS=${EXPERIMENTS}"
  echo "============================================================"
  SEED="${seed}" bash scripts/libero/residual_worldmodel/eval_temporal_query_rft_sweep.sh
  echo "| ${seed} | \`experiment_settings_seed${seed}.md\` | \`rft_eval_manifest_seed${seed}_node${NODE_INDEX:-0}_of_${NUM_NODES:-1}.tsv\` |" >> "${settings_file}"
done
