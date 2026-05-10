#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/results/phase1/residual_worldmodel}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/priority_a_summary}"

"${REPO_ROOT}/.venv/bin/python" analysis/summarize_phase1_priority_a.py "${EVAL_ROOT}" --out-dir "${OUT_DIR}"

echo "summary: ${OUT_DIR}/summary.md"
