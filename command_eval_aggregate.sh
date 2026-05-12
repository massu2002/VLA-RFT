#!/usr/bin/env bash
# ==============================================================
# command_eval_aggregate.sh — 3シードの結果を集計 (mean±std)
# ==============================================================
# 前提:
#   各 PC で command_eval_single_seed.sh を実行済みで、
#   以下のディレクトリがローカルに揃っていること:
#     ${OUT_ROOT}/seed42/
#     ${OUT_ROOT}/seed43/
#     ${OUT_ROOT}/seed44/
#
# 使い方:
#   bash command_eval_aggregate.sh
#
# 別ディレクトリを集計する場合:
#   OUT_ROOT=results/phase1/v4_improved_spatial bash command_eval_aggregate.sh
#
# 特定シードのみで集計する場合:
#   EVAL_SEEDS="42 43" bash command_eval_aggregate.sh
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

# ==============================================================
# 設定変数
# ==============================================================
export RUN_NAME="${RUN_NAME:-v4_improved_spatial}"
export SWEEP_CONFIG="${SWEEP_CONFIG:-configs/libero/phase1/v4_core_sweep.json}"
export EXP_FILTER="${EXP_FILTER:-v4a_q8_k2_motion,v4b_q8_k2_rank1_motion,v4b_q8_k2_rank2_motion,v4b_q8_k2_rank1_mixedneg}"
# OUT_ROOT は常に RUN_NAME から計算する（シェル環境変数の引き継ぎを防ぐため）
# ベースディレクトリを変えたい場合: RESULTS_ROOT=/other/base bash ...
export OUT_ROOT="${RESULTS_ROOT:-results/phase1}/${RUN_NAME}"
export EVAL_SEEDS="${EVAL_SEEDS:-42 43 44}"

# ==============================================================
# 設定確認ログ
# ==============================================================
echo "============================================================"
echo "Phase1 v4 — マルチシード集計 (mean ± std)"
echo ""
echo "  RUN_NAME    = ${RUN_NAME}"
echo "  OUT_ROOT    = ${OUT_ROOT}"
echo "  EVAL_SEEDS  = ${EVAL_SEEDS}"
echo "  EXP_FILTER  = ${EXP_FILTER:-（未設定: 全条件）}"
echo "  SWEEP_CONFIG= ${SWEEP_CONFIG}"
echo "============================================================"

# ==============================================================
# シードディレクトリの存在確認
# ==============================================================
_missing=0
for _s in ${EVAL_SEEDS}; do
  _d="${OUT_ROOT}/seed${_s}"
  if [ ! -d "${_d}" ]; then
    echo "[WARN] seed${_s} ディレクトリが見つかりません: ${_d}" >&2
    _missing=$((_missing + 1))
  else
    _n=$(find "${_d}" -name "aggregate_metrics.json" 2>/dev/null | wc -l)
    echo "  seed${_s}: ${_n} 個の aggregate_metrics.json を検出"
  fi
done

if [ "${_missing}" -gt 0 ]; then
  echo ""
  echo "[ERROR] ${_missing} シードのディレクトリが不足しています。" >&2
  echo "  各 PC から seed{N}/ を ${OUT_ROOT}/ にコピーしてから再実行してください。" >&2
  echo ""
  echo "  例 (rsync で PC2 から取得):"
  echo "    rsync -av user@pc2:~/VLA-RFT/${OUT_ROOT}/seed43/ ${OUT_ROOT}/seed43/"
  exit 1
fi

# ==============================================================
# Step 3: マルチシード集計 — mean ± std
# ==============================================================
echo ""
echo ">>> [Step 3] マルチシード集計: mean ± std を計算します..."

BASE_OUT_ROOT="${OUT_ROOT}" \
EVAL_SEEDS="${EVAL_SEEDS}" \
EXP_FILTER="${EXP_FILTER}" \
SWEEP_CONFIG="${SWEEP_CONFIG}" \
  bash scripts/libero/phase1/aggregate_v4_multiseed_eval.sh

# ==============================================================
# Step 4: 結果ファイルの確認
# ==============================================================
echo ""
echo ">>> [Step 4] 結果ファイルを確認します..."

for _s in ${EVAL_SEEDS}; do
  _d="${OUT_ROOT}/seed${_s}"
  echo ""
  echo "  Seed ${_s}:"
  ls -1 "${_d}/summary/" 2>/dev/null | sed 's/^/    /' \
    || echo "    (summary なし)"
done

echo ""
echo "  multiseed_summary:"
ls -1 "${OUT_ROOT}/multiseed_summary/" 2>/dev/null | sed 's/^/    /' \
  || echo "    (生成されていません)"

# ==============================================================
# 完了メッセージ
# ==============================================================
echo ""
echo "============================================================"
echo "マルチシード集計完了"
echo ""
echo "  multiseed (mean±std): ${OUT_ROOT}/multiseed_summary/multiseed_summary.md"
echo "  multiseed CSV       : ${OUT_ROOT}/multiseed_summary/multiseed_summary.csv"
echo ""
echo "RFT を行う場合:"
echo "  RUN_NAME=${RUN_NAME} BEST_CRITERION=hybrid_score \\"
echo "    bash scripts/libero/residual_worldmodel/select_best_v4_for_rft.sh"
echo "============================================================"
