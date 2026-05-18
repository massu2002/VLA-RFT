#!/usr/bin/env bash
# ==============================================================
# command_train_rft.sh — Phase 1 v4 WM → RFT 第2段階学習
# ==============================================================
#
# v4_improved_spatial の全 WM チェックポイントを使用した実験設計（計 8 グループ）:
#
#   [baseline]
#   baseline          Phase0 AR-Pixel token WM ベースライン
#                     報酬: lpips_mae
#
#   [v4a: scorer なし → lpips_mae のみ]
#   v4a_recon         v4a_q8_k2_motion (K=2, Q=8, motion bias)
#                     報酬: lpips_mae = -(LPIPS+MAE)
#                     ← baseline との比較: pixel 再構成 WM の精度向上の効果
#
#   [v4b λ_rank=1.0 × motion bias: hybrid と rank_score の両方]
#   v4b_r1_hybrid     v4b_q8_k2_rank1_motion, 報酬: hybrid
#   v4b_r1_rank       v4b_q8_k2_rank1_motion, 報酬: rank_score
#                     ← v4a_recon との比較: rank_score 信号の効果
#
#   [v4b λ_rank=2.0 × motion bias: lambda_rank の強度比較]
#   v4b_r2_hybrid     v4b_q8_k2_rank2_motion, 報酬: hybrid
#   v4b_r2_rank       v4b_q8_k2_rank2_motion, 報酬: rank_score
#                     ← v4b_r1_* との比較: WM 学習時の lambda_rank 強度の影響
#
#   [v4b λ_rank=1.0 × mixed negatives: negative 設計の比較]
#   v4b_mixneg_hybrid v4b_q8_k2_rank1_mixedneg, 報酬: hybrid
#   v4b_mixneg_rank   v4b_q8_k2_rank1_mixedneg, 報酬: rank_score
#                     ← v4b_r1_* との比較: WM 学習時の negative 設計の影響
#
# 比較軸のまとめ:
#   baseline vs v4a_recon       : WM アーキテクチャ（token vs pixel residual）の効果
#   v4a_recon vs v4b_r1_hybrid  : rank_score 信号の有無
#   v4b_r1 vs v4b_r2            : WM 学習時の lambda_rank 強度
#   v4b_r1 vs v4b_mixneg        : WM 学習時の negative 設計
#   *_hybrid vs *_rank           : RFT 報酬の hybrid vs rank_score 単体
#
# 使用例:
#   DRY_RUN=1 bash command_train_rft.sh                         # 実行予定の確認
#   bash command_train_rft.sh                                   # 全グループ実行
#   ROLE_FILTER=v4b_r1_hybrid bash command_train_rft.sh         # 特定グループのみ
#   ROLE_FILTER=v4b_r1_hybrid,v4b_r1_rank bash command_train_rft.sh
#   SMOKE=1 bash command_train_rft.sh                           # 動作確認（2 steps）
#   SKIP_EXISTING=0 ROLE_FILTER=v4b_r1_hybrid bash command_train_rft.sh  # 強制再実行
#
#   # 保存先ルートを指定
#   OUTPUT_ROOT=results/phase1/v4_improved_spatial_rft bash command_train_rft.sh
#
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

# ==============================================================
# 共通設定
# ==============================================================

# WM 学習 run name（checkpoints/libero/TemporalQueryResidualWM/${WM_RUN_NAME}/ を参照）
export WM_RUN_NAME="${WM_RUN_NAME:-v4_improved_spatial}"

# 実験テーブルの選択。
#   v4_improved_spatial: 既存の improved spatial 実験群
#   v4_next_exps       : results/phase1/v4_next_exps で学習した WM 実験群
RFT_EXPERIMENT_SET="${RFT_EXPERIMENT_SET:-}"
if [ -z "${RFT_EXPERIMENT_SET}" ]; then
    if [ "${WM_RUN_NAME}" = "v4_next_exps" ]; then
        RFT_EXPERIMENT_SET="v4_next_exps"
    else
        RFT_EXPERIMENT_SET="v4_improved_spatial"
    fi
fi

# WM チェックポイントのシード（s${CKPT_SEED}/final を使用）
export CKPT_SEED="${CKPT_SEED:-42}"

# 対象タスクスイート
export TASK_SUITE="${TASK_SUITE:-spatial}"

# GRPO 学習ステップ数
export RFT_STEPS="${RFT_STEPS:-400}"

# 使用 GPU 数
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# VLA ポリシー初期チェックポイント
export BASE_VLA_PATH="${BASE_VLA_PATH:-checkpoints/libero/Base_VLA/${TASK_SUITE}}"

# LIBERO データのルートディレクトリ
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/localdata/modified_libero_rlds}"

# 既存 RFT 結果があればスキップ（0 にすると強制再実行）
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Smoke test（2 steps, 1 GPU で動作確認）
export SMOKE="${SMOKE:-0}"

# DRY_RUN: 実行予定を表示するだけで実際には動かさない
export DRY_RUN="${DRY_RUN:-0}"

# hybrid 報酬パラメータ
export RANK_REWARD_ALPHA="${RANK_REWARD_ALPHA:-0.2}"   # -(LPIPS+MAE) 成分の重み
export RANK_REWARD_BETA="${RANK_REWARD_BETA:-0.8}"     # rank_score 成分の重み
export NORMALIZE_RANK_REWARD="${NORMALIZE_RANK_REWARD:-1}"
export CLIP_RANK_REWARD="${CLIP_RANK_REWARD:-1}"
export RANK_REWARD_CLIP_VALUE="${RANK_REWARD_CLIP_VALUE:-5.0}"

# phase0 baseline token WM のパス
# WM 評価 protocol と同じ checkpoint-150000 を使用する。
TOKEN_WM_PATH="${TOKEN_WM_PATH:-checkpoints/libero/WorldModel/${TASK_SUITE}/20260429_worldmodel_scratch/checkpoint-150000}"

# RFT 結果の保存先ルート（空のときはスクリプトデフォルトを使用）
#   設定時: ${OUTPUT_ROOT}/<role>/ に各グループの結果を保存
#   未設定: results/phase1/residual_worldmodel/rft/<exp_rft_reward>/ （既存の規則）
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

# Smoke 時の上書き
if [ "${SMOKE}" = "1" ]; then
    RFT_STEPS="${RFT_STEPS_SMOKE:-2}"
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE_SMOKE:-1}"
fi

# ==============================================================
# 実験テーブル: "ROLE|EXP_NAME|REWARD_TYPE|DESCRIPTION"
#
#   EXP_NAME = "__baseline__" のときは phase0 AR-Pixel token WM を使用
#   それ以外は WM_RUN_NAME/${TASK_SUITE}/${EXP_NAME}/s${CKPT_SEED}/final を参照
# ==============================================================
case "${RFT_EXPERIMENT_SET}" in
    v4_improved_spatial)
        _EXP_TABLE=(
            # ── baseline ──────────────────────────────────────────────────
            "baseline|__baseline__|lpips_mae|Phase0 AR-Pixel token WM ベースライン"

            # ── v4a: scorer なし → lpips_mae のみ ─────────────────────────
            "v4a_recon|v4a_q8_k2_motion|lpips_mae|v4a K=2 Q=8 motion bias, 報酬 -(LPIPS+MAE)"

            # ── v4b λ_rank=1.0 × motion bias ──────────────────────────────
            "v4b_r1_hybrid|v4b_q8_k2_rank1_motion|hybrid|v4b λ_rank=1.0: 0.2×-(LPIPS+MAE)+0.8×rank_score"
            "v4b_r1_rank|v4b_q8_k2_rank1_motion|rank_score|v4b λ_rank=1.0: rank_score のみ"

            # ── v4b λ_rank=2.0 × motion bias ──────────────────────────────
            "v4b_r2_hybrid|v4b_q8_k2_rank2_motion|hybrid|v4b λ_rank=2.0: 0.2×-(LPIPS+MAE)+0.8×rank_score"
            "v4b_r2_rank|v4b_q8_k2_rank2_motion|rank_score|v4b λ_rank=2.0: rank_score のみ"

            # ── v4b λ_rank=1.0 × mixed negatives ──────────────────────────
            "v4b_mixneg_hybrid|v4b_q8_k2_rank1_mixedneg|hybrid|v4b mixed-neg: 0.2×-(LPIPS+MAE)+0.8×rank_score"
            "v4b_mixneg_rank|v4b_q8_k2_rank1_mixedneg|rank_score|v4b mixed-neg: rank_score のみ"
        )
        ;;
    v4_next_exps)
        _EXP_TABLE=(
            # ── q16 mixed negatives: scorer あり → 3 報酬 ─────────────────
            "next_q16_lpips|v4b_q16_k2_rank1_mixedneg|lpips_mae|v4_next q16 mixed-neg: 報酬 -(LPIPS+MAE)"
            "next_q16_rank|v4b_q16_k2_rank1_mixedneg|rank_score|v4_next q16 mixed-neg: rank_score のみ"
            "next_q16_hybrid|v4b_q16_k2_rank1_mixedneg|hybrid|v4_next q16 mixed-neg: 0.2×-(LPIPS+MAE)+0.8×rank_score"

            # ── 2-stage: scorer あり → 3 報酬 ─────────────────────────────
            "next_2stage_lpips|v4b_q8_k2_rank1_2stage|lpips_mae|v4_next 2-stage: 報酬 -(LPIPS+MAE)"
            "next_2stage_rank|v4b_q8_k2_rank1_2stage|rank_score|v4_next 2-stage: rank_score のみ"
            "next_2stage_hybrid|v4b_q8_k2_rank1_2stage|hybrid|v4_next 2-stage: 0.2×-(LPIPS+MAE)+0.8×rank_score"

            # ── action-noise negatives: scorer あり → 3 報酬 ──────────────
            "next_actnoise_lpips|v4b_q8_k2_rank1_actnoiseneg|lpips_mae|v4_next action-noise negatives: 報酬 -(LPIPS+MAE)"
            "next_actnoise_rank|v4b_q8_k2_rank1_actnoiseneg|rank_score|v4_next action-noise negatives: rank_score のみ"
            "next_actnoise_hybrid|v4b_q8_k2_rank1_actnoiseneg|hybrid|v4_next action-noise negatives: 0.2×-(LPIPS+MAE)+0.8×rank_score"

        )
        ;;
    *)
        echo "Unknown RFT_EXPERIMENT_SET=${RFT_EXPERIMENT_SET}" >&2
        echo "Expected: v4_improved_spatial or v4_next_exps" >&2
        exit 2
        ;;
esac

# ROLE_FILTER: カンマ区切りで実行するロールを指定（空 = 全ロール）
ROLE_FILTER="${ROLE_FILTER:-}"

# ==============================================================
# ヘルパー関数
# ==============================================================

_filter_role() {
    local _role="$1"
    [ -z "${ROLE_FILTER}" ] && return 0
    echo "${ROLE_FILTER}" | tr ',' '\n' | grep -qx "${_role}"
}

# RFT 結果ディレクトリ（rft_checkpoint_path.txt の親）
# post_train_phase1_residual_rft.sh の OUTPUT_DIR に対応
# OUTPUT_ROOT 指定時: ${OUTPUT_ROOT}/${_role}/
# 未指定時:           results/phase1/residual_worldmodel/rft/<exp_rft_reward>/
_rft_output_dir() {
    local _role="$1" _exp="$2" _reward="$3"
    local _repo_root
    _repo_root="$(cd "$(dirname "$0")" && pwd)"
    if [ -n "${OUTPUT_ROOT}" ]; then
        # OUTPUT_ROOT が相対パスの場合はリポジトリルート基準に展開
        case "${OUTPUT_ROOT}" in
            /*) echo "${OUTPUT_ROOT}/${_role}" ;;
            *)  echo "${_repo_root}/${OUTPUT_ROOT}/${_role}" ;;
        esac
    elif [ "${_exp}" = "__baseline__" ]; then
        echo "${_repo_root}/results/phase1/residual_worldmodel/rft/baseline_phase0_ar_pixel"
    else
        echo "${_repo_root}/results/phase1/residual_worldmodel/rft/${_exp}_rft_${_reward}"
    fi
}

# WM チェックポイントのパス
_wm_ckpt_path() {
    local _exp="$1"
    if [ "${_exp}" = "__baseline__" ]; then
        echo "${TOKEN_WM_PATH}"
    else
        echo "checkpoints/libero/TemporalQueryResidualWM/${WM_RUN_NAME}/${TASK_SUITE}/${_exp}/s${CKPT_SEED}/final"
    fi
}

# ==============================================================
# 設定確認ログ
# ==============================================================
echo "============================================================"
echo "Phase1 v4 WM → RFT 第2段階学習"
echo ""
echo "  WM_RUN_NAME       = ${WM_RUN_NAME}"
echo "  RFT_EXPERIMENT_SET= ${RFT_EXPERIMENT_SET}"
echo "  CKPT_SEED         = ${CKPT_SEED}  (s${CKPT_SEED}/final)"
echo "  TASK_SUITE        = ${TASK_SUITE}"
echo "  ROLE_FILTER       = ${ROLE_FILTER:-（全グループ）}"
echo ""
echo "  [学習設定]"
echo "  RFT_STEPS         = ${RFT_STEPS}"
echo "  N_GPUS_PER_NODE   = ${N_GPUS_PER_NODE}"
echo "  BASE_VLA_PATH     = ${BASE_VLA_PATH}"
echo "  LIBERO_DATA_ROOT  = ${LIBERO_DATA_ROOT}"
echo "  SMOKE / DRY_RUN   = ${SMOKE} / ${DRY_RUN}"
echo "  SKIP_EXISTING     = ${SKIP_EXISTING}"
echo ""
echo "  [hybrid 報酬パラメータ]"
echo "  alpha(LPIPS+MAE) / beta(rank_score) = ${RANK_REWARD_ALPHA} / ${RANK_REWARD_BETA}"
echo "  normalize_rank / clip(±${RANK_REWARD_CLIP_VALUE}) = ${NORMALIZE_RANK_REWARD} / ${CLIP_RANK_REWARD}"
echo "============================================================"
echo ""

# ==============================================================
# 実験グループの表示 & フィルタ適用
# ==============================================================
echo "実験グループ一覧:"
echo ""
printf "  %-14s %-38s %-12s %-10s  %s\n" "ROLE" "WM_EXP_NAME" "REWARD" "WM ckpt" "RFT 済"
printf "  %-14s %-38s %-12s %-10s  %s\n" "----" "-----------" "------" "-------" "------"

_plan_roles=()
_plan_exps=()
_plan_rewards=()
_plan_descs=()

for _row in "${_EXP_TABLE[@]}"; do
    IFS='|' read -r _role _exp _reward _desc <<< "${_row}"
    _filter_role "${_role}" || continue

    _wm_ck="$(_wm_ckpt_path "${_exp}")"
    _out_dir="$(_rft_output_dir "${_role}" "${_exp}" "${_reward}")"
    _wm_ok="";   [ -d "${_wm_ck}" ]                          && _wm_ok="OK"   || _wm_ok="NOT FOUND"
    _rft_done=""; [ -f "${_out_dir}/rft_checkpoint_path.txt" ] && _rft_done="done" || _rft_done=""

    printf "  %-14s %-38s %-12s %-10s  %s\n" "${_role}" "${_exp}" "${_reward}" "[${_wm_ok}]" "${_rft_done}"
    printf "    └ %s\n" "${_desc}"

    _plan_roles+=("${_role}")
    _plan_exps+=("${_exp}")
    _plan_rewards+=("${_reward}")
    _plan_descs+=("${_desc}")
done

_total="${#_plan_roles[@]}"
echo ""
echo "  合計: ${_total} グループ"
echo ""

if [ "${DRY_RUN}" = "1" ]; then
    echo ">>> DRY_RUN=1: 実際の学習は実行しません。"
    echo "    実行する場合: DRY_RUN=0 bash command_train_rft.sh"
    exit 0
fi

if [ "${_total}" -eq 0 ]; then
    echo ">>> 実行対象グループが 0 件です。ROLE_FILTER を確認してください。"
    exit 0
fi

# ==============================================================
# 実行
# ==============================================================
echo ">>> RFT 学習を開始します..."
echo ""

_success=0; _fail=0; _skip=0

for _i in $(seq 0 $((_total - 1))); do
    _role="${_plan_roles[${_i}]}"
    _exp="${_plan_exps[${_i}]}"
    _reward="${_plan_rewards[${_i}]}"
    _desc="${_plan_descs[${_i}]}"
    _out_dir="$(_rft_output_dir "${_role}" "${_exp}" "${_reward}")"

    echo "------------------------------------------------------------"
    echo "  [$((_i + 1))/${_total}] ROLE=${_role}  exp=${_exp}  reward=${_reward}"
    echo "  ${_desc}"
    echo "  → 出力先: ${_out_dir}"

    if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${_out_dir}/rft_checkpoint_path.txt" ]; then
        echo "  SKIP（完了済: ${_out_dir}/rft_checkpoint_path.txt）"
        _skip=$((_skip + 1))
        echo ""
        continue
    fi

    set +e
    if [ "${_exp}" = "__baseline__" ]; then
        # Phase0 AR-Pixel token WM ベースライン
        TASK_SUITE="${TASK_SUITE}" \
        TOKEN_WM_PATH="${TOKEN_WM_PATH}" \
        RFT_EXP_NAME="baseline_phase0_ar_pixel" \
        OUTPUT_DIR="${_out_dir}" \
        RFT_STEPS="${RFT_STEPS}" \
        N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
        BASE_VLA_PATH="${BASE_VLA_PATH}" \
        LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
        SMOKE="${SMOKE}" \
        DRY_RUN="0" \
            bash scripts/libero/residual_worldmodel/run_baseline_phase0_ar_pixel_rft.sh
    else
        # v4a / v4b WM ベース RFT
        # run_v4_temporal_query_rft.sh 内で v4a は lpips_mae に自動フォールバック
        WM_RUN_NAME="${WM_RUN_NAME}" \
        WM_EXP_NAME="${_exp}" \
        CKPT_SEED="${CKPT_SEED}" \
        TASK_SUITE="${TASK_SUITE}" \
        WORLD_REWARD_TYPE="${_reward}" \
        RANK_REWARD_ALPHA="${RANK_REWARD_ALPHA}" \
        RANK_REWARD_BETA="${RANK_REWARD_BETA}" \
        NORMALIZE_RANK_REWARD="${NORMALIZE_RANK_REWARD}" \
        CLIP_RANK_REWARD="${CLIP_RANK_REWARD}" \
        RANK_REWARD_CLIP_VALUE="${RANK_REWARD_CLIP_VALUE}" \
        OUTPUT_DIR="${_out_dir}" \
        RFT_STEPS="${RFT_STEPS}" \
        N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
        BASE_VLA_PATH="${BASE_VLA_PATH}" \
        LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
        SMOKE="${SMOKE}" \
        DRY_RUN="0" \
            bash scripts/libero/residual_worldmodel/run_v4_temporal_query_rft.sh
    fi
    _rc=$?
    set -e

    if [ "${_rc}" -eq 0 ]; then
        echo "  [DONE] ${_role}"
        _success=$((_success + 1))
    else
        echo "  [FAIL] ${_role}  (rc=${_rc})" >&2
        _fail=$((_fail + 1))
    fi
    echo ""
done

# ==============================================================
# 完了メッセージ
# ==============================================================
echo "============================================================"
echo "RFT 第2段階学習 完了"
echo ""
echo "  成功: ${_success}  失敗: ${_fail}  スキップ: ${_skip}"
echo ""
echo "  RFT チェックポイント保存先:"
echo "    checkpoints/libero/TemporalQueryResidualWM-RFT/${WM_RUN_NAME}/${TASK_SUITE}/"
echo "    checkpoints/libero/TemporalQueryResidualWM-RFT/baseline_phase0_ar_pixel/${TASK_SUITE}/"
echo ""
echo "  RFT 結果ディレクトリ（rft_checkpoint_path.txt を含む）:"
if [ -n "${OUTPUT_ROOT}" ]; then
    echo "    ${OUTPUT_ROOT}/{baseline,v4a_recon,v4b_hybrid,...}/"
else
    echo "    results/phase1/residual_worldmodel/rft/<exp_rft_reward>/"
fi
echo ""

if [ "${_fail}" -gt 0 ]; then
    echo "  [WARNING] ${_fail} グループが失敗しました。"
    echo "  再実行例: SKIP_EXISTING=1 ROLE_FILTER=\"<ロール名>\" bash command_train_rft.sh"
    echo ""
fi

echo "次のステップ: ロボット評価"
echo "  bash command_eval_single_seed.sh   # シングルシード評価"
echo "  bash command_eval_aggregate.sh     # マルチシード集約評価"
echo "============================================================"

[ "${_fail}" -gt 0 ] && exit 1
exit 0
