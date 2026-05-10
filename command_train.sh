#!/usr/bin/env bash
# ==============================================================
# command_train.sh — Phase 1 / v4 core sweep 学習用コマンド
# ==============================================================
# 目的:
#   v4 core sweep として、v4_core_sweep.json に定義された全10条件を学習する。
#     v4a 系（ranking head なし）:
#       - v4a_q8_k2_nomotion   : baseline（motion bias なし）
#       - v4a_q8_k2_motion     : motion bias あり
#       - v4a_q16_k2_motion    : Q=16 クエリ
#       - v4a_q8_k3_motion     : K=3 履歴フレーム
#     v4b 系（ActionFutureScorer あり）:
#       - v4b_q8_k2_rank05_motion  : lambda_rank=0.5
#       - v4b_q8_k2_rank1_motion   : lambda_rank=1.0（標準）
#       - v4b_q8_k2_rank2_motion   : lambda_rank=2.0
#       - v4b_q16_k2_rank1_motion  : Q=16
#       - v4b_q8_k3_rank1_motion   : K=3
#       - v4b_q8_k2_rank1_mixedneg : mixed negative（same_task_other_window 50%）
#
# RFT（強化学習ファインチューニング）は実行しない（RUN_RFT=0 固定）。
# RFT を行う場合は command_train.sh の Step 5 以降を参照すること。
#
# 結果保存先:
#   チェックポイント:
#     checkpoints/libero/TemporalQueryResidualWM/v4_core_sweep_spatial/${TASK_SUITE}/<EXP_NAME>/s42/
#   WM 評価結果:
#     results/phase1/v4_core_sweep_spatial/<EXP_NAME>/aggregate_metrics.json
#
# EXP_FILTER="" で全10条件を実行（デフォルト）。
# 特定条件だけ再実行する場合: EXP_FILTER="v4a_q8_k2_motion" bash command_train.sh
#
# DRY_RUN=1 で実行予定のコマンドだけ表示し、実際には動かさない。
#
# --------------------------------------------------------------
# 実行例:
#   # 実行予定確認（全10条件が表示されるか確認）:
#   #   DRY_RUN=1 bash command_train.sh
#   #
#   # 高速設定の本番学習（全10条件 × 50000 steps, evalは別途 command_eval.sh）:
#   #   bash command_train.sh
#   #
#   # フル設定（全10条件 × 150000 steps + train後evalも同時実行）:
#   #   PILOT_STEPS=150000 MODE=train_eval DISABLE_ZERO_ACTION_DIAGNOSTIC=0 bash command_train.sh
#   #
#   # GPU 指定して実行:
#   #   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash command_train.sh
#   #
#   # 特定条件だけ再実行:
#   #   EXP_FILTER="v4b_q8_k2_rank1_motion" bash command_train.sh
#   #
#   # 既存結果を上書きして再実行:
#   #   OVERWRITE=1 bash command_train.sh
#
# --------------------------------------------------------------
# 注意:
#   このファイルを直接実行できるようにするには:
#     chmod +x command_train.sh command_eval.sh
# ==============================================================

set -euo pipefail
# cd "$(dirname "$0")" は実行元ディレクトリを保持するためコメントアウト可能
cd "$(dirname "$0")"

# ==============================================================
# 共通環境変数
# ==============================================================

# 対象タスクスイート（spatial / object / goal / 10）
export TASK_SUITE="${TASK_SUITE:-spatial}"

# 結果保存ディレクトリ名
export RUN_NAME="${RUN_NAME:-v4_improved_spatial}"

# 評価結果・manifest・summary の保存先。
# 旧既定値は results/phase1/residual_worldmodel/${RUN_NAME} だったが、
# v4 core sweep では phase1 直下にまとめる。
# 例: OUT_ROOT=results/phase1/v4_core_sweep_spatial bash command_train.sh
export OUT_ROOT="${OUT_ROOT:-results/phase1/${RUN_NAME}}"

# sweep 設定 JSON（実験条件を定義）
export SWEEP_CONFIG="${SWEEP_CONFIG:-configs/libero/phase1/v4_core_sweep.json}"

# 実行対象の実験を最優先4条件に絞る（カンマ区切りで完全一致）
# 選定根拠（v4_core_sweep_spatial の結果より）:
#   v4a_q8_k2_motion       : 再構成ベースライン（比較基準）
#   v4b_q8_k2_rank1_motion : pairwise_acc_score 最良(0.46)・score_gap 最大(7.37)
#   v4b_q8_k2_rank2_motion : pairwise_acc_lpips 最良(0.66)・高λ_rank
#   v4b_q8_k2_rank1_mixedneg: negative_mix バグ修正後の初評価
# 全条件実行する場合: EXP_FILTER="" bash command_train.sh
export EXP_FILTER="${EXP_FILTER:-v4a_q8_k2_motion,v4b_q8_k2_rank1_motion,v4b_q8_k2_rank2_motion,v4b_q8_k2_rank1_mixedneg}"

# RFT は実行しない（RUN_RFT=0 を明示）
# train_eval / train_only / eval_only モードでは RFT は走らないが、
# 後から run_v4_selected_rft_sweep.sh を別途実行した場合に備えて明示する
export RUN_RFT="${RUN_RFT:-0}"

# 学習コマンドの既定は train_only。
# train_eval は各条件の学習直後に eval まで走るため重い。
# 評価・集計は command_eval.sh で明示的に行う。
export MODE="${MODE:-train_only}"

# 既存チェックポイントや評価結果があればスキップ（1=有効）
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

# 既存結果を上書きして再実行（1=上書き、0=スキップ）
export OVERWRITE="${OVERWRITE:-0}"

# 学習ステップ数
# 短縮実行する場合: PILOT_STEPS=50000 bash command_train.sh
export PILOT_STEPS="${PILOT_STEPS:-150000}"

# LIBERO データのルートディレクトリ
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/localdata/modified_libero_rlds}"

# Phase0 互換フォーマットで保存する（評価スクリプトとの互換性のため 1 のまま）
export PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE:-1}"

# MAX_STEPS は PILOT_STEPS を引き継ぐ（run_v4_core_sweep.sh に渡される）
export MAX_STEPS="${PILOT_STEPS}"

# ==============================================================
# H200 x8 高速化設定
# ==============================================================

# per-device バッチサイズ
# H200 80GB では 8 が安定動作の目安。メモリに余裕があれば 16 も可。
# batch=1 → 8 にすることで:
#   1. GPU 利用率が大幅改善（Tensor Core の実効スループット向上）
#   2. V4Collator が複数サンプルを持てるようになり
#      same_task_other_window negative が正しく機能する
#      （batch=1 では常に batch-roll fallback になっていた）
export BATCH_SIZE="${BATCH_SIZE:-8}"

# グローバル有効バッチサイズ
# 計算式: GRAD_ACCUM = WORLD_MODEL_BATCH_SIZE / (BATCH_SIZE × GPU数)
# 8GPU × per-device=8 の場合: 64 / (8 × 8) = 1 → 勾配累積なし（最速）
# 旧: WORLD_MODEL_BATCH_SIZE=16, batch=1, 8GPU → GRAD_ACCUM=2
export WORLD_MODEL_BATCH_SIZE="${WORLD_MODEL_BATCH_SIZE:-64}"

# TF32: H200 の Tensor Core を bf16 matmul で最大活用
# torch.backends.cuda.matmul.allow_tf32 = True を有効化する
# 精度への影響は軽微（float32 の下位 13 bit を丸め）、速度向上は約 1.5-2x
export TF32="${TF32:-1}"

# 学習率: グローバルバッチ 16 → 64（4x 増加）に対して sqrt スケーリング
# 5e-5 × sqrt(4) ≈ 1e-4
# 線形スケーリングなら 2e-4 だが transformer では sqrt が安定しやすい
export LR="${LR:-1e-4}"

# checkpoint / logging 頻度。I/O を少し減らして学習を軽くする。
export SAVE_STEPS="${SAVE_STEPS:-10000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export LOGGING_STEPS="${LOGGING_STEPS:-20}"

# v4b の zero-action baseline は診断用で loss には使わない。
# step ごとに追加 forward が走るため、通常 sweep では無効化して高速化する。
export DISABLE_ZERO_ACTION_DIAGNOSTIC="${DISABLE_ZERO_ACTION_DIAGNOSTIC:-1}"

# MODE=train_eval で train 後 eval まで同時実行する場合の軽量 eval 設定。
# 本格評価は command_eval.sh で行う。
export NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS:-50}"
export NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS:-50}"

# 使用GPU を指定する場合は以下を有効化（コメントアウト中は自動検出: 全 8 GPU を使用）
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ==============================================================
# 設定確認ログ
# ==============================================================
echo "============================================================"
echo "Phase1 v4 core sweep — 学習コマンド"
echo ""
echo "  TASK_SUITE        = ${TASK_SUITE}"
echo "  RUN_NAME          = ${RUN_NAME}"
echo "  OUT_ROOT          = ${OUT_ROOT}"
echo "  SWEEP_CONFIG      = ${SWEEP_CONFIG}"
echo "  EXP_FILTER        = ${EXP_FILTER:-（未設定: 全10条件実行）}"
echo "  RUN_RFT           = ${RUN_RFT}  ← RFT は実行しない"
echo "  MODE              = ${MODE}"
echo "  MAX_STEPS         = ${MAX_STEPS}"
echo "  SKIP_EXISTING     = ${SKIP_EXISTING}"
echo "  OVERWRITE         = ${OVERWRITE}"
echo "  LIBERO_DATA_ROOT  = ${LIBERO_DATA_ROOT}"
echo ""
echo "  [H200 x8 高速化設定]"
echo "  BATCH_SIZE        = ${BATCH_SIZE}  (per device)"
echo "  WORLD_MODEL_BATCH = ${WORLD_MODEL_BATCH_SIZE}  → GRAD_ACCUM = $((WORLD_MODEL_BATCH_SIZE / (BATCH_SIZE * 8)))"
echo "  TF32              = ${TF32}  (H200 Tensor Core 最適化)"
echo "  LR                = ${LR}  (グローバルバッチ 4x 増に対して sqrt スケーリング)"
echo "  SAVE_STEPS        = ${SAVE_STEPS}"
echo "  SAVE_TOTAL_LIMIT  = ${SAVE_TOTAL_LIMIT}"
echo "  LOGGING_STEPS     = ${LOGGING_STEPS}"
echo "  DISABLE_ZERO_ACTION_DIAGNOSTIC = ${DISABLE_ZERO_ACTION_DIAGNOSTIC}"
echo "  NUM_EVAL_WINDOWS  = ${NUM_EVAL_WINDOWS}  (MODE=train_eval時のみ)"
echo "  NUM_RANKING_WINDOWS = ${NUM_RANKING_WINDOWS}  (MODE=train_eval時のみ)"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-（未設定: 8GPU 自動検出）}"
echo ""
echo "  結果保存先:"
echo "    チェックポイント: checkpoints/libero/TemporalQueryResidualWM/${RUN_NAME}/${TASK_SUITE}/"
echo "    WM評価結果      : ${OUT_ROOT}/<EXP_NAME>/"
echo "============================================================"

# SWEEP_CONFIG の存在確認
if [ ! -f "${SWEEP_CONFIG}" ]; then
  echo "[ERROR] SWEEP_CONFIG が見つかりません: ${SWEEP_CONFIG}" >&2
  exit 1
fi

# LIBERO データの存在確認（DRY_RUN 時はスキップ）
if [ "${DRY_RUN:-0}" != "1" ] && [ ! -d "${LIBERO_DATA_ROOT}" ]; then
  echo "[WARN] LIBERO_DATA_ROOT が見つかりません: ${LIBERO_DATA_ROOT}"
  echo "       データがマウントされているか確認してください。"
fi

# ==============================================================
# Step 0: DRY_RUN — 実行予定10条件の確認
# --------------------------------------------------------------
# 実際の学習は行わない。
# v4 core sweep 対象の全10条件が表示されるか確認するためのコマンド。
# まずこのコマンドだけを単独で実行して、EXP_FILTER が効いているか確認すること。
#
#   使い方: DRY_RUN=1 bash command_train.sh
#
# 表示されるべき10条件:
#   v4a_q8_k2_nomotion / v4a_q8_k2_motion / v4a_q16_k2_motion / v4a_q8_k3_motion
#   v4b_q8_k2_rank05_motion / v4b_q8_k2_rank1_motion / v4b_q8_k2_rank2_motion
#   v4b_q16_k2_rank1_motion / v4b_q8_k3_rank1_motion / v4b_q8_k2_rank1_mixedneg
# ==============================================================
echo ""
echo ">>> [Step 0] DRY_RUN: 実行対象条件を確認します（全10条件 or EXP_FILTER で絞った条件）..."
echo "    ※ DRY_RUN=0 の場合はこのステップはスキップされます"

DRY_RUN=1 \
MODE="${MODE}" \
TASK_SUITE="${TASK_SUITE}" \
RUN_NAME="${RUN_NAME}" \
OUT_ROOT="${OUT_ROOT}" \
SWEEP_CONFIG="${SWEEP_CONFIG}" \
EXP_FILTER="${EXP_FILTER}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
OVERWRITE="${OVERWRITE}" \
MAX_STEPS="${MAX_STEPS}" \
SAVE_STEPS="${SAVE_STEPS}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
LOGGING_STEPS="${LOGGING_STEPS}" \
DISABLE_ZERO_ACTION_DIAGNOSTIC="${DISABLE_ZERO_ACTION_DIAGNOSTIC}" \
NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS}" \
NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS}" \
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE}" \
  bash scripts/libero/phase1/run_v4_core_sweep.sh

echo ""
echo "    上記条件が表示されていれば設定は正しいです。"
echo "    DRY_RUN=0 bash command_train.sh で実際の学習に進んでください。"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ""
  echo ">>> DRY_RUN=1 のため、学習・評価は実行せずここで終了します。"
  exit 0
fi

# ==============================================================
# Step 1: 学習実行（既定は train_only）
# --------------------------------------------------------------
# v4_core_sweep.json の全10条件を学習する（既定 MAX_STEPS=50000）。
# EXP_FILTER が設定されている場合はその条件だけ実行する。
#
# RFT は RUN_RFT=0 のため実行しない。
# RFT を後から行いたい場合は以下を参照:
#   BEST_CRITERION=hybrid_score bash scripts/libero/residual_worldmodel/select_best_v4_for_rft.sh
#   bash scripts/libero/phase1/run_v4_selected_rft_sweep.sh
#
# 結果保存先:
#   チェックポイント:
#     checkpoints/libero/TemporalQueryResidualWM/${RUN_NAME}/${TASK_SUITE}/<EXP_NAME>/s42/
#   WM 評価結果:
#     ${OUT_ROOT}/<EXP_NAME>/aggregate_metrics.json
#
# 学習完了後は command_eval.sh で評価・集計を行うこと。
# ==============================================================
echo ""
echo ">>> [Step 1] v4 core sweep: ${MODE} を開始します..."
echo "    対象条件: ${EXP_FILTER:-全10条件（v4_core_sweep.json のすべての enabled 実験）}"
echo "    MAX_STEPS=${MAX_STEPS} / RUN_RFT=${RUN_RFT}"

DRY_RUN="${DRY_RUN:-0}" \
MODE="${MODE}" \
TASK_SUITE="${TASK_SUITE}" \
RUN_NAME="${RUN_NAME}" \
OUT_ROOT="${OUT_ROOT}" \
SWEEP_CONFIG="${SWEEP_CONFIG}" \
EXP_FILTER="${EXP_FILTER}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
OVERWRITE="${OVERWRITE}" \
MAX_STEPS="${MAX_STEPS}" \
SAVE_STEPS="${SAVE_STEPS}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
LOGGING_STEPS="${LOGGING_STEPS}" \
DISABLE_ZERO_ACTION_DIAGNOSTIC="${DISABLE_ZERO_ACTION_DIAGNOSTIC}" \
NUM_EVAL_WINDOWS="${NUM_EVAL_WINDOWS}" \
NUM_RANKING_WINDOWS="${NUM_RANKING_WINDOWS}" \
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE}" \
  bash scripts/libero/phase1/run_v4_core_sweep.sh

# --------------------------------------------------------------
# 参考: 学習だけを行いたい場合（評価は別途 command_eval.sh で実行）
# --------------------------------------------------------------
# DRY_RUN="${DRY_RUN:-0}" \
# MODE=train_only \
# TASK_SUITE="${TASK_SUITE}" \
# RUN_NAME="${RUN_NAME}" \
# OUT_ROOT="${OUT_ROOT}" \
# SWEEP_CONFIG="${SWEEP_CONFIG}" \
# EXP_FILTER="${EXP_FILTER}" \
# SKIP_EXISTING="${SKIP_EXISTING}" \
# OVERWRITE="${OVERWRITE}" \
# MAX_STEPS="${MAX_STEPS}" \
# LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
# PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE}" \
#   bash scripts/libero/phase1/run_v4_core_sweep.sh

# ==============================================================
# 完了メッセージ
# ==============================================================
echo ""
echo "============================================================"
echo "v4 core sweep 学習完了"
echo ""
echo "  チェックポイント: checkpoints/libero/TemporalQueryResidualWM/${RUN_NAME}/"
echo "  WM 評価結果     : ${OUT_ROOT}/"
echo ""
echo "次のステップ:"
echo "  評価・集計: OUT_ROOT=${OUT_ROOT} RUN_NAME=${RUN_NAME} bash command_eval.sh"
echo "============================================================"
