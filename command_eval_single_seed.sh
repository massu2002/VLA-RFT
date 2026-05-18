#!/usr/bin/env bash
# ==============================================================
# command_eval_single_seed.sh — 1シード分の評価（1PC用）
# ==============================================================
# 使い方:
#   SEED=42 bash command_eval_single_seed.sh   # PC1
#   SEED=43 bash command_eval_single_seed.sh   # PC2
#   SEED=44 bash command_eval_single_seed.sh   # PC3
#
# SEED  : 評価ウィンドウのランダム選択シード（各PCで異なる値を指定）
# CKPT_SEED : チェックポイントのシード（s{CKPT_SEED}/final）
#             省略時は SEED と同じ値。
#             同一の学習済みモデルを複数シードで評価する場合は
#             CKPT_SEED=42 を固定して SEED だけ変えること。
#
# 例:
#   # 同一チェックポイント (s42) を3シードで評価する場合:
#   CKPT_SEED=42 SEED=42 bash command_eval_single_seed.sh   # PC1
#   CKPT_SEED=42 SEED=43 bash command_eval_single_seed.sh   # PC2
#   CKPT_SEED=42 SEED=44 bash command_eval_single_seed.sh   # PC3
#
# 完了後、3台の OUT_ROOT/seed{N}/ を1台に集めて
# command_eval_aggregate.sh で mean±std を計算する。
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

# ==============================================================
# 必須: シード指定
# ==============================================================
SEED="${SEED:?'SEED が未設定です。SEED=42 bash command_eval_single_seed.sh のように指定してください'}"

# ==============================================================
# 設定変数
# ==============================================================
export TASK_SUITE="${TASK_SUITE:-spatial}"
export RUN_NAME="${RUN_NAME:-v4_improved_spatial}"
export SWEEP_CONFIG="${SWEEP_CONFIG:-configs/libero/phase1/v4_core_sweep.json}"
export EXP_FILTER="${EXP_FILTER:-v4a_q8_k2_motion,v4b_q8_k2_rank1_motion,v4b_q8_k2_rank2_motion,v4b_q8_k2_rank1_mixedneg}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export OVERWRITE="${OVERWRITE:-0}"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/localdata/modified_libero_rlds}"
export PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE:-1}"

# OUT_ROOT は常に RUN_NAME から計算する（シェル環境変数の引き継ぎを防ぐため）
# ベースディレクトリを変えたい場合: RESULTS_ROOT=/other/base SEED=42 bash ...
export OUT_ROOT="${RESULTS_ROOT:-results/phase1}/${RUN_NAME}"

# チェックポイントシード: 省略時は評価シードと同じ
# 同一モデルを複数シードで評価する場合は CKPT_SEED=42 などと固定する
export CKPT_SEED="${CKPT_SEED:-${SEED}}"

export BASELINE_WM_CKPT="${BASELINE_WM_CKPT:-checkpoints/libero/WorldModel/${TASK_SUITE}/20260429_worldmodel_scratch/checkpoint-150000}"
export BASELINE_TOKENIZER_CKPT="${BASELINE_TOKENIZER_CKPT:-checkpoints/libero/WorldModel/Tokenizer}"
export RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-1}"

# ==============================================================
# 設定確認ログ
# ==============================================================
echo "============================================================"
echo "Phase1 v4 — 単シード評価  (SEED=${SEED})"
echo ""
echo "  TASK_SUITE        = ${TASK_SUITE}"
echo "  RUN_NAME          = ${RUN_NAME}"
echo "  SWEEP_CONFIG      = ${SWEEP_CONFIG}"
echo "  EXP_FILTER        = ${EXP_FILTER:-（未設定: 全条件）}"
echo "  SEED              = ${SEED}  (評価ウィンドウのランダムシード)"
echo "  CKPT_SEED         = ${CKPT_SEED}  (チェックポイント s${CKPT_SEED}/final を使用)"
echo "  SKIP_EXISTING     = ${SKIP_EXISTING}"
echo "  OVERWRITE         = ${OVERWRITE}"
echo "  LIBERO_DATA_ROOT  = ${LIBERO_DATA_ROOT}"
echo "  OUT_ROOT          = ${OUT_ROOT}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-（未設定: 自動検出）}"
echo ""
echo "  ベースライン比較:"
echo "    RUN_BASELINE_EVAL = ${RUN_BASELINE_EVAL}"
echo "    BASELINE_WM_CKPT  = ${BASELINE_WM_CKPT}"
echo "  結果保存先:"
echo "    ${OUT_ROOT}/seed${SEED}/<EXP_NAME>/aggregate_metrics.json"
echo "    ${OUT_ROOT}/seed${SEED}/summary/"
echo "============================================================"

if [ ! -f "${SWEEP_CONFIG}" ]; then
  echo "[ERROR] SWEEP_CONFIG が見つかりません: ${SWEEP_CONFIG}" >&2
  exit 1
fi

_SEED_OUT="${OUT_ROOT}/seed${SEED}"
mkdir -p "${_SEED_OUT}"

# ==============================================================
# Step 1: eval_only — 学習済みチェックポイントを評価
# ==============================================================
echo ""
echo ">>> [Seed ${SEED} / Step 1] v4 eval..."

MODE=eval_only \
SEED="${SEED}" \
CKPT_SEED="${CKPT_SEED}" \
TASK_SUITE="${TASK_SUITE}" \
RUN_NAME="${RUN_NAME}" \
SWEEP_CONFIG="${SWEEP_CONFIG}" \
EXP_FILTER="${EXP_FILTER}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
OVERWRITE="${OVERWRITE}" \
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
PHASE0_COMPATIBLE="${PHASE0_COMPATIBLE}" \
OUT_ROOT="${_SEED_OUT}" \
  bash scripts/libero/phase1/run_v4_core_sweep.sh

# ==============================================================
# Step 1.5: ベースライン評価 — Phase0 AR-Pixel WM
# ==============================================================
if [ "${RUN_BASELINE_EVAL}" = "1" ]; then
  echo ""
  echo ">>> [Seed ${SEED} / Step 1.5] ベースライン評価..."

  _BL_MANIFEST=""
  if [ -n "${EXP_FILTER}" ]; then
    for _exp in $(echo "${EXP_FILTER}" | tr ',' '\n'); do
      _cand="${_SEED_OUT}/${_exp}/window_manifest.json"
      if [ -f "${_cand}" ]; then
        _BL_MANIFEST="${_cand}"
        break
      fi
    done
  fi
  if [ -z "${_BL_MANIFEST}" ]; then
    _BL_MANIFEST=$(find "${_SEED_OUT}" -maxdepth 2 -name "window_manifest.json" \
      ! -path "*/summary/*" ! -path "*/baseline*" ! -path "*/logs/*" 2>/dev/null \
      | sort | head -1)
  fi

  _BL_OUT="${_SEED_OUT}/baseline_phase0_ar_pixel"

  if [ "${SKIP_EXISTING}" = "1" ] && [ "${OVERWRITE}" = "0" ] \
     && [ -f "${_BL_OUT}/aggregate_metrics.json" ]; then
    echo "    SKIP (既存: ${_BL_OUT}/aggregate_metrics.json)"
  elif [ ! -d "${BASELINE_WM_CKPT}" ]; then
    echo "    SKIP (BASELINE_WM_CKPT not found: ${BASELINE_WM_CKPT})" >&2
  elif [ ! -d "${BASELINE_TOKENIZER_CKPT}" ]; then
    echo "    SKIP (BASELINE_TOKENIZER_CKPT not found: ${BASELINE_TOKENIZER_CKPT})" >&2
  elif [ -z "${_BL_MANIFEST}" ]; then
    echo "    SKIP (window_manifest.json が見つかりません。Step 1 を先に完了してください)" >&2
  else
    echo "    manifest: ${_BL_MANIFEST}"
    echo "    output  : ${_BL_OUT}"
    mkdir -p "${_BL_OUT}"
    TASK_SUITE="${TASK_SUITE}" \
    SEED="${SEED}" \
    PHASE0_AR_PIXEL_CKPT="${BASELINE_WM_CKPT}" \
    PHASE0_TOKENIZER_CKPT="${BASELINE_TOKENIZER_CKPT}" \
    WINDOW_MANIFEST="${_BL_MANIFEST}" \
    OUTPUT_DIR="${_BL_OUT}" \
    EVAL_HORIZON="${EVAL_HORIZON:-8}" \
    LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
      bash scripts/libero/residual_worldmodel/eval_phase0_ar_pixel_on_phase1_manifest.sh
    echo "    完了 → ${_BL_OUT}/aggregate_metrics.json"
  fi
fi

# ==============================================================
# Step 2: per-seed summary 生成
# ==============================================================
echo ""
echo ">>> [Seed ${SEED} / Step 2] per-seed summary..."

RUN_NAME="${RUN_NAME}" \
OUT_ROOT="${_SEED_OUT}" \
SWEEP_CONFIG="${SWEEP_CONFIG}" \
  bash scripts/libero/phase1/summarize_v4_core_sweep.sh

# ==============================================================
# 完了メッセージ
# ==============================================================
echo ""
echo "============================================================"
echo "Seed ${SEED} 評価完了"
echo ""
echo "  per-seed summary: ${_SEED_OUT}/summary/v4_core_sweep_summary.md"
echo ""
echo "次のステップ:"
echo "  他の PC の seed{N}/ ディレクトリを ${OUT_ROOT}/ に集めてから"
echo "  bash command_eval_aggregate.sh  を実行してください。"
echo "============================================================"
