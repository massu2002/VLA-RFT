#!/usr/bin/env bash
# run_residual_wm_eval_only.sh — Phase 1 Pixel-Residual WM evaluation only.
#
# Usage:
#   bash scripts/libero/phase1/run_residual_wm_eval_only.sh [spatial|object|goal|10]
#
# Examples:
#   OUT_ROOT=/home/masuoka/VLA-RFT/results/phase1/20260501_050441_spatial \
#   TRAIN_CONDITIONS=pixel_residual_roi_dynamic \
#     bash scripts/libero/phase1/run_residual_wm_eval_only.sh spatial
#
# This is a thin wrapper around run_residual_wm_eval.sh with SKIP_TRAIN=1.
# Evaluation defaults to the RTX 5090-compatible virtualenv. Override VENV_NAME
# if you need a different environment.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export SKIP_TRAIN=1
export VENV_NAME="${VENV_NAME:-.venv5090_eval}"
exec bash "${SCRIPT_DIR}/run_residual_wm_eval.sh" "$@"
