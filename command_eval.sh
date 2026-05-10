#!/usr/bin/env bash
# ==============================================================
# command_eval.sh — Phase 1 / v4 core sweep 評価・集計用コマンド
# ==============================================================
# 目的:
#   command_train.sh で学習済みの全10条件を eval_only で評価し、
#   summary を生成する。
#     v4a 系: v4a_q8_k2_nomotion / v4a_q8_k2_motion / v4a_q16_k2_motion / v4a_q8_k3_motion
#     v4b 系: v4b_q8_k2_rank05_motion / v4b_q8_k2_rank1_motion / v4b_q8_k2_rank2_motion
#             v4b_q16_k2_rank1_motion / v4b_q8_k3_rank1_motion / v4b_q8_k2_rank1_mixedneg
#
# RFT 評価はまだ行わない（RUN_RFT=0 固定）。
# RFT 評価が必要になったら Step 4 以降（command_train.sh 参照）を使うこと。
#
# 結果保存先:
#   各条件の評価結果:
#     results/phase1/v4_core_sweep_spatial/<EXP_NAME>/aggregate_metrics.json
#   summary:
#     results/phase1/v4_core_sweep_spatial/summary/v4_core_sweep_summary.md
#     results/phase1/v4_core_sweep_spatial/summary/v4_core_sweep_summary.csv
#     results/phase1/v4_core_sweep_spatial/summary/v4_core_sweep_summary.json
#
# --------------------------------------------------------------
# 実行例:
#   # 評価・集計（全10条件）:
#   #   bash command_eval.sh
#   #
#   # 特定条件だけ再評価:
#   #   EXP_FILTER="v4b_q8_k2_rank1_motion" bash command_eval.sh
#   #
#   # 既存評価を上書きして再実行:
#   #   OVERWRITE=1 bash command_eval.sh
#   #
#   # GPU 指定して実行:
#   #   CUDA_VISIBLE_DEVICES=0 bash command_eval.sh
#
# --------------------------------------------------------------
# 注意:
#   このファイルを直接実行できるようにするには:
#     chmod +x command_train.sh command_eval.sh
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

# ==============================================================
# 共通環境変数（command_train.sh と同じ設定を使う）
# ==============================================================

# 対象タスクスイート（command_train.sh と同じ値を使うこと）
export TASK_SUITE="${TASK_SUITE:-spatial}"

# 結果保存ディレクトリ名（command_train.sh と一致させること）
export RUN_NAME="${RUN_NAME:-v4_improved_spatial}"

# sweep 設定 JSON
export SWEEP_CONFIG="${SWEEP_CONFIG:-configs/libero/phase1/v4_core_sweep.json}"

# 評価対象の実験（command_train.sh と同じ4条件）
export EXP_FILTER="${EXP_FILTER:-v4a_q8_k2_motion,v4b_q8_k2_rank1_motion,v4b_q8_k2_rank2_motion,v4b_q8_k2_rank1_mixedneg}"

# RFT は実行しない（eval_only + summarize のみ）
export RUN_RFT="${RUN_RFT:-0}"

# 既存評価結果があればスキップ（1=有効）
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

# 既存結果を上書きして再評価（1=上書き）
export OVERWRITE="${OVERWRITE:-0}"

# LIBERO データのルートディレクトリ
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/localdata/modified_libero_rlds}"

# Phase0 互換フォーマットで保存する
export PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE:-1}"

# 評価結果のルート（command_train.sh と一致させる）
# 例: OUT_ROOT=results/phase1/v4_core_sweep_spatial bash command_eval.sh
export OUT_ROOT="${OUT_ROOT:-results/phase1/${RUN_NAME}}"

# 使用GPU を指定する場合は以下を有効化（コメントアウト中は自動検出）
# export CUDA_VISIBLE_DEVICES=0

# ==============================================================
# 設定確認ログ
# ==============================================================
echo "============================================================"
echo "Phase1 v4 core sweep — 評価・集計コマンド"
echo ""
echo "  TASK_SUITE        = ${TASK_SUITE}"
echo "  RUN_NAME          = ${RUN_NAME}"
echo "  SWEEP_CONFIG      = ${SWEEP_CONFIG}"
echo "  EXP_FILTER        = ${EXP_FILTER:-（未設定: 全10条件評価）}"
echo "  RUN_RFT           = ${RUN_RFT}  ← RFT 評価は実行しない"
echo "  SKIP_EXISTING     = ${SKIP_EXISTING}"
echo "  OVERWRITE         = ${OVERWRITE}"
echo "  LIBERO_DATA_ROOT  = ${LIBERO_DATA_ROOT}"
echo "  OUT_ROOT          = ${OUT_ROOT}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-（未設定: 自動検出）}"
echo ""
echo "  結果保存先:"
echo "    各条件の評価結果: ${OUT_ROOT}/<EXP_NAME>/aggregate_metrics.json"
echo "    summary         : ${OUT_ROOT}/summary/"
echo "============================================================"

# SWEEP_CONFIG の存在確認
if [ ! -f "${SWEEP_CONFIG}" ]; then
  echo "[ERROR] SWEEP_CONFIG が見つかりません: ${SWEEP_CONFIG}" >&2
  exit 1
fi

# 学習結果ディレクトリの存在確認
if [ ! -d "${OUT_ROOT}" ]; then
  echo "[WARN] OUT_ROOT が見つかりません: ${OUT_ROOT}"
  echo "       先に command_train.sh で学習を実行してください。"
fi

# ==============================================================
# Step 1: eval_only — 学習済みチェックポイントを評価する
# --------------------------------------------------------------
# 対象は EXP_FILTER で指定した条件（未設定時は全10条件）。
# 学習は行わず、既存チェックポイントに対して評価だけを実行する。
#
# 評価結果の保存先（各条件ごと）:
#   ${OUT_ROOT}/<EXP_NAME>/aggregate_metrics.json
#     主な指標: pairwise_acc_score, score_gap_mean, pairwise_acc_lpips,
#               full_mse, copy_current_mse, fuser_mask_entropy, dynamic_mask_entropy など
#   ${OUT_ROOT}/<EXP_NAME>/metrics_by_task.csv
#   ${OUT_ROOT}/<EXP_NAME>/window_manifest.json
#
# 再評価が必要な場合: OVERWRITE=1 bash command_eval.sh
# ==============================================================
echo ""
echo ">>> [Step 1] eval_only: 学習済みチェックポイントを評価します..."
echo "    対象条件: ${EXP_FILTER:-全10条件（v4_core_sweep.json のすべての enabled 実験）}"
echo "    評価結果保存先: ${OUT_ROOT}/<EXP_NAME>/aggregate_metrics.json"

MODE=eval_only \
TASK_SUITE="${TASK_SUITE}" \
RUN_NAME="${RUN_NAME}" \
SWEEP_CONFIG="${SWEEP_CONFIG}" \
EXP_FILTER="${EXP_FILTER}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
OVERWRITE="${OVERWRITE}" \
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE}" \
OUT_ROOT="${OUT_ROOT}" \
  bash scripts/libero/phase1/run_v4_core_sweep.sh

# ==============================================================
# Step 2: summary 生成 — 3条件の評価結果を集計する
# --------------------------------------------------------------
# 全条件の aggregate_metrics.json を読み込み、比較表を生成する。
#
# summary 保存先:
#   ${OUT_ROOT}/summary/v4_core_sweep_summary.md    ← メインレポート（日本語）
#   ${OUT_ROOT}/summary/v4_core_sweep_summary.csv   ← 全指標のテーブル
#   ${OUT_ROOT}/summary/v4_core_sweep_summary.json  ← 機械可読フォーマット
#   ${OUT_ROOT}/summary/v4_core_sweep_ranking.csv   ← hybrid_score ランキング
#
# まず確認すべきファイル:
#   ${OUT_ROOT}/summary/v4_core_sweep_summary.md
#   ${OUT_ROOT}/summary/v4_core_sweep_summary.csv
#   ${OUT_ROOT}/summary/v4_core_sweep_summary.json
# ==============================================================
echo ""
echo ">>> [Step 2] summary 生成: v4 core sweep 全10条件の評価結果を集計します..."
echo "    summary 保存先: ${OUT_ROOT}/summary/"

RUN_NAME="${RUN_NAME}" \
OUT_ROOT="${OUT_ROOT}" \
SWEEP_CONFIG="${SWEEP_CONFIG}" \
  bash scripts/libero/phase1/summarize_v4_core_sweep.sh

# ==============================================================
# Step 3: 結果ファイルの確認
# --------------------------------------------------------------
# 評価結果と summary が正しく生成されているか確認する。
# ==============================================================
echo ""
echo ">>> [Step 3] 結果ファイルを確認します..."

echo ""
echo "各条件の結果:"
ls -lh "${OUT_ROOT}/" 2>/dev/null || \
  echo "  [INFO] 結果ディレクトリがまだありません: ${OUT_ROOT}/"

echo ""
echo "summary:"
ls -lh "${OUT_ROOT}/summary/" 2>/dev/null || \
  echo "  [INFO] summary ディレクトリがまだありません"

# tree コマンドが使える場合はディレクトリ構造を表示
if command -v tree >/dev/null 2>&1; then
  echo ""
  echo "ディレクトリ構造:"
  tree -L 3 "${OUT_ROOT}/" 2>/dev/null || true
fi

# ==============================================================
# Step 4: 重要指標の確認ポイント（コメント）
# --------------------------------------------------------------
# summary で特に注目すべき指標と解釈の目安。
#
# [ActionFutureScorer の学習確認]
#   score_gap_mean          : pos_score - neg_score の平均。正で大きいほど良い。
#                             step 5k 時点で 0 以下なら scorer が学習できていない。
#   pairwise_acc_score      : ランキング精度（0.5=ランダム、1.0=完全）。0.6以上を目標。
#
# [future 予測品質]
#   reverse_windows_score   : 逆順 window でスコアが下がるか（sanity check）。
#   full_mse_over_copy_current_mse : 1.0 以上は copy-current collapse の兆候。
#
# [negative の質]
#   neg_task_match_rate     : 0.3 未満の場合は cross-task negative が多い（easy すぎ）。
#
# [mask の質]
#   dynamic_mask_entropy    : 低いほど mask が集中している（good）。
#   dynamic_mask_overlap    : 低いほど各クエリが異なる場所に注目している（good）。
#   fuser_mask_entropy      : TokenFuser の attention 集中度。
#   fuser_mask_overlap      : TokenFuser の多様性。
#
# [loss の収束]
#   loss_rank               : ranking loss。下がり続けていれば scorer が学習中。
#   loss_entropy            : mask entropy loss。0 に向かって減少していれば良い。
#   loss_diversity          : mask diversity loss。0 に向かって減少していれば良い。
#
# [v4a vs v4b の比較]
#   pairwise_acc_lpips      : LPIPS ベースのランキング精度。v4b が v4a より高いなら
#                             ranking head が有効に機能している。
# ==============================================================

# ==============================================================
# 完了メッセージ
# ==============================================================
echo ""
echo "============================================================"
echo "v4 core sweep 評価・集計完了"
echo ""
echo "  WM 評価サマリー: ${OUT_ROOT}/summary/v4_core_sweep_summary.md"
echo "  WM ランキング  : ${OUT_ROOT}/summary/v4_core_sweep_ranking.csv"
echo ""
echo "RFT を行う場合（現状は RUN_RFT=${RUN_RFT} のため未実行）:"
echo "  best checkpoint 選択:"
echo "    RUN_NAME=${RUN_NAME} BEST_CRITERION=hybrid_score \\"
echo "      bash scripts/libero/residual_worldmodel/select_best_v4_for_rft.sh"
echo "  RFT 学習:"
echo "    RUN_NAME=${RUN_NAME} bash scripts/libero/phase1/run_v4_selected_rft_sweep.sh"
echo "============================================================"
