#!/usr/bin/env bash
# ==============================================================
# command_train_wm.sh — DynQueryWorldModel core sweep 学習コマンド
# ==============================================================
# 目的:
#   DynQuery core sweep として，dynquery_core_sweep.json に定義された
#   5つの代表的条件を学習する。
#
#   [dq_a系: ActionFutureScorer なし]
#     dq_baseline         : Core1-4 すべて無効、motion bias なし、scorer なし
#                           ← 性能の下限（baseline）
#     dq_core14_no_scorer : Core1-4 すべて有効、motion bias あり、scorer なし
#                           ← 再構成側の改善効果を単独で検証
#
#   [dq_b系: ActionFutureScorer あり]
#     dq_full_rank1       : Core1-4 + scorer + mixed negatives, λ_rank=1.0
#                           ← フルモデル（メイン比較条件）
#     dq_full_q16         : dq_full_rank1 と同一だが Q=16 クエリ
#                           ← クエリ数スケーリングの効果検証
#     dq_full_2stage      : dq_core14_no_scorer (50k) → dq_full_rank1 (100k) の2段階学習
#                           ← ウォームスタートの効果検証
#
# 比較軸のまとめ:
#   baseline vs core14_no_scorer : Core 1-4 の有無（再構成側改善）
#   core14 vs full_rank1         : ActionFutureScorer の有無（ランキング信号）
#   full_rank1 vs full_q16       : クエリ数 Q=8 vs Q=16
#   full_rank1 vs full_2stage    : 直接学習 vs 2段階学習
#
# 結果保存先:
#   チェックポイント: checkpoints/libero/DynQueryWorldModel/core_sweep/${TASK_SUITE}/<EXP_NAME>/s42/final/
#   WM 評価結果    : results/phase1/DynQueryWorldModel_core_sweep/<EXP_NAME>/
#
# ==============================================================
#
# 実行例:
#   # 実行予定確認（5条件が表示されるか確認）:
#   DRY_RUN=1 bash command_train_wm.sh
#
#   # 通常学習（全5条件 × 150000 steps, MODE=train_only）:
#   bash command_train_wm.sh
#
#   # 特定条件のみ実行:
#   EXP_FILTER=dq_full_rank1 bash command_train_wm.sh
#   EXP_FILTER=dq_full_rank1,dq_full_q16 bash command_train_wm.sh
#
#   # 既存結果を上書きして再実行:
#   OVERWRITE=1 EXP_FILTER=dq_full_rank1 bash command_train_wm.sh
#
#   # 学習後すぐに eval まで実行（重い）:
#   MODE=train_eval bash command_train_wm.sh
#
#   # GPU 指定:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash command_train_wm.sh
#
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

# ==============================================================
# 共通設定
# ==============================================================

# 対象タスクスイート（spatial / object / goal / 10）
export TASK_SUITE="${TASK_SUITE:-spatial}"

# sweep 設定 JSON（実験条件を定義）
export SWEEP_CONFIG="${SWEEP_CONFIG:-configs/libero/phase1/dynquery_core_sweep.json}"

# チェックポイントの保存先ルート
export CKPT_ROOT="${CKPT_ROOT:-checkpoints/libero/DynQueryWorldModel/core_sweep}"

# 評価結果・manifest・summary の保存先
export OUT_ROOT="${OUT_ROOT:-results/phase1/DynQueryWorldModel_core_sweep}"

# 実行対象の実験を絞る（カンマ区切りで完全一致、空 = 全5条件実行）
# 2段階学習は dq_core14_no_scorer が先に完了している必要があるため，
# dq_full_2stage 単独実行時は先に dq_core14_no_scorer を実行すること。
export EXP_FILTER="${EXP_FILTER:-}"

# 学習モード（train_only / eval_only / train_eval）
export MODE="${MODE:-train_only}"

# 既存チェックポイントや評価結果があればスキップ
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

# 既存結果を上書きして再実行
export OVERWRITE="${OVERWRITE:-0}"

# DRY_RUN: 実行予定を表示するだけで実際には動かさない
export DRY_RUN="${DRY_RUN:-0}"

# LIBERO データのルートディレクトリ
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/localdata/modified_libero_rlds}"

# Phase0 互換フォーマットで保存
export PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE:-1}"

# ==============================================================
# H200 x8 高速化設定
# ==============================================================

# per-device バッチサイズ（H200 80GB では 8 が安定）
export BATCH_SIZE="${BATCH_SIZE:-8}"

# グローバル有効バッチサイズ
# GRAD_ACCUM = WORLD_MODEL_BATCH_SIZE / (BATCH_SIZE × GPU数)
# 8GPU × per-dev=8 → 64 / (8×8) = 1 (勾配累積なし、最速)
export WORLD_MODEL_BATCH_SIZE="${WORLD_MODEL_BATCH_SIZE:-64}"

# TF32: H200 Tensor Core を最大活用
export TF32="${TF32:-1}"

# 学習率: グローバルバッチ 64 に対して sqrt スケーリング済み
export LR="${LR:-1e-4}"

# MAX_STEPS は sweep JSON の per-experiment 設定を優先するが，
# ここで上書きしたい場合は設定する（空 = JSON の値を使用）
export MAX_STEPS="${MAX_STEPS:-}"

# checkpoint / logging 頻度
export SAVE_STEPS="${SAVE_STEPS:-10000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export LOGGING_STEPS="${LOGGING_STEPS:-20}"

# MODE=train_eval 時の軽量 eval 設定
export NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-50}"
export NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS:-50}"

# ==============================================================
# 設定確認ログ
# ==============================================================
echo "============================================================"
echo "DynQueryWorldModel core sweep — 学習コマンド"
echo ""
echo "  TASK_SUITE        = ${TASK_SUITE}"
echo "  SWEEP_CONFIG      = ${SWEEP_CONFIG}"
echo "  CKPT_ROOT         = ${CKPT_ROOT}"
echo "  OUT_ROOT          = ${OUT_ROOT}"
echo "  EXP_FILTER        = ${EXP_FILTER:-（未設定: 全5条件実行）}"
echo "  MODE              = ${MODE}"
echo "  SKIP_EXISTING     = ${SKIP_EXISTING}"
echo "  OVERWRITE         = ${OVERWRITE}"
echo "  LIBERO_DATA_ROOT  = ${LIBERO_DATA_ROOT}"
echo ""
echo "  [H200 x8 高速化設定]"
echo "  BATCH_SIZE        = ${BATCH_SIZE}  (per device)"
echo "  WORLD_MODEL_BATCH = ${WORLD_MODEL_BATCH_SIZE}"
echo "  TF32              = ${TF32}"
echo "  LR                = ${LR}"
echo "  SAVE_STEPS        = ${SAVE_STEPS}"
echo "  SAVE_TOTAL_LIMIT  = ${SAVE_TOTAL_LIMIT}"
echo "  LOGGING_STEPS     = ${LOGGING_STEPS}"
echo "  MAX_STEPS         = ${MAX_STEPS:-（JSON の per-experiment 設定を使用）}"
echo ""
echo "  [5 sweep 条件]"
echo "  1. dq_baseline         : Core1-4 無効 / scorer 無  ← 性能下限"
echo "  2. dq_core14_no_scorer : Core1-4 有効 / scorer 無  ← 再構成改善"
echo "  3. dq_full_rank1       : Core1-4 有効 / scorer 有  ← フルモデル"
echo "  4. dq_full_q16         : Q=16 クエリ / scorer 有  ← クエリスケール"
echo "  5. dq_full_2stage      : 2段階学習 (50k+100k)     ← ウォームスタート"
echo ""
echo "  結果保存先:"
echo "    チェックポイント: ${CKPT_ROOT}/${TASK_SUITE}/<EXP_NAME>/s42/final/"
echo "    WM 評価結果    : ${OUT_ROOT}/<EXP_NAME>/"
echo "============================================================"

if [ ! -f "${SWEEP_CONFIG}" ]; then
  echo "[ERROR] SWEEP_CONFIG が見つかりません: ${SWEEP_CONFIG}" >&2
  exit 1
fi

if [ "${DRY_RUN:-0}" != "1" ] && [ ! -d "${LIBERO_DATA_ROOT}" ]; then
  echo "[WARN] LIBERO_DATA_ROOT が見つかりません: ${LIBERO_DATA_ROOT}"
  echo "       データがマウントされているか確認してください。"
fi

# ==============================================================
# Step 0: DRY_RUN — 実行予定の確認
# ==============================================================
echo ""
echo ">>> [Step 0] DRY_RUN: 実行対象条件を確認します..."

DRY_RUN=1 \
MODE="${MODE}" \
TASK_SUITE="${TASK_SUITE}" \
SWEEP_CONFIG="${SWEEP_CONFIG}" \
CKPT_ROOT="${CKPT_ROOT}" \
OUT_ROOT="${OUT_ROOT}" \
EXP_FILTER="${EXP_FILTER}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
OVERWRITE="${OVERWRITE}" \
MAX_STEPS="${MAX_STEPS:-}" \
BATCH_SIZE="${BATCH_SIZE}" \
WORLD_MODEL_BATCH_SIZE="${WORLD_MODEL_BATCH_SIZE}" \
LR="${LR}" \
TF32="${TF32}" \
SAVE_STEPS="${SAVE_STEPS}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
LOGGING_STEPS="${LOGGING_STEPS}" \
NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS}" \
NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS}" \
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE}" \
  bash scripts/libero/phase1/run_dynquery_core_sweep.sh

echo ""
echo "    上記5条件が表示されていれば設定は正しいです。"
echo "    DRY_RUN=0 bash command_train_wm.sh で実際の学習に進んでください。"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ""
  echo ">>> DRY_RUN=1 のため、学習・評価は実行せずここで終了します。"
  exit 0
fi

# ==============================================================
# Step 1: DynQuery core sweep 学習実行
# ==============================================================
echo ""
echo ">>> [Step 1] DynQuery core sweep: ${MODE} を開始します..."
echo "    対象条件: ${EXP_FILTER:-全5条件}"
echo "    MAX_STEPS=${MAX_STEPS:-（JSON の per-experiment 設定）}"

DRY_RUN="${DRY_RUN:-0}" \
MODE="${MODE}" \
TASK_SUITE="${TASK_SUITE}" \
SWEEP_CONFIG="${SWEEP_CONFIG}" \
CKPT_ROOT="${CKPT_ROOT}" \
OUT_ROOT="${OUT_ROOT}" \
EXP_FILTER="${EXP_FILTER}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
OVERWRITE="${OVERWRITE}" \
MAX_STEPS="${MAX_STEPS:-}" \
BATCH_SIZE="${BATCH_SIZE}" \
WORLD_MODEL_BATCH_SIZE="${WORLD_MODEL_BATCH_SIZE}" \
LR="${LR}" \
TF32="${TF32}" \
SAVE_STEPS="${SAVE_STEPS}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
LOGGING_STEPS="${LOGGING_STEPS}" \
NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS}" \
NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS}" \
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE}" \
  bash scripts/libero/phase1/run_dynquery_core_sweep.sh

# ==============================================================
# 完了メッセージ
# ==============================================================
echo ""
echo "============================================================"
echo "DynQuery core sweep 学習完了"
echo ""
echo "  チェックポイント: ${CKPT_ROOT}/${TASK_SUITE}/"
echo "  WM 評価結果    : ${OUT_ROOT}/"
echo ""
echo "次のステップ:"
echo "  評価:     MODE=eval_only bash command_train_wm.sh"
echo "  RFT学習:  WM_RUN_NAME=DynQueryWorldModel_core_sweep bash command_train_rft.sh"
echo "============================================================"
