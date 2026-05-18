#!/usr/bin/env bash
# ==============================================================
# eval_ar_pixel_wm_standalone.sh — AR-Pixel WM 単独評価スクリプト
# ==============================================================
# Phase1 DynQuery の事前実行を必要とせず、RLDS データから直接
# ウィンドウマニフェストを生成して AR-Pixel WM を評価します。
#
# 使い方:
#   bash scripts/eval_ar_pixel_wm_standalone.sh
#
# 主な環境変数:
#   TASK_SUITE          タスクスイート (default: spatial)
#   CKPT                WMチェックポイントパス（省略時: 自動検出）
#   BASE_MODEL_CONFIG   モデル設定ディレクトリ (default: checkpoints/libero/WorldModel/spatial)
#   TOKENIZER_CKPT      Tokenizer ckpt (default: checkpoints/libero/WorldModel/Tokenizer)
#   OUTPUT_DIR          結果保存先 (default: results/baseline_ar_pixel_wm/spatial)
#   DATA_ROOT           RLDS データルート (default: data/modified_libero_rlds)
#   NUM_WINDOWS         評価ウィンドウ数 (default: 200)
#   EVAL_HORIZON        予測ホライズン (default: 8)
#   SEEDS               評価シード (default: 42, カンマ区切りで複数指定可: "42,43,44")
#                       複数指定時は seed_N/ サブディレクトリに結果を保存し平均集計
#   REGENERATE_MANIFEST 1 にするとマニフェストを強制再生成 (default: 0)
#   OVERWRITE           1 にすると既存の評価結果を無視して再評価 (default: 0)
#   GENERATE_REPORT     1 にすると評価後に日本語レポートを生成 (default: 1)
#   DECODE_CHUNK_SIZE   デコードチャンクサイズ (default: 8)
#   DEVICE              auto / cuda / cpu (default: auto)
#   SMOKE               1 にするとスモークテスト（2ウィンドウのみ）
#   NUM_GPUS            並列GPU数 (未設定時: nvidia-smi で自動検出)
#                       2以上の場合はウィンドウをGPU間で分割して並列評価し
#                       完了後に自動マージします (CUDA_VISIBLE_DEVICES=0,1,...)
#   VENV_NAME           仮想環境名 (default: .venv5090_eval)
#                       RTX 5090: .venv5090_eval (PyTorch 2.8+cu128)
#                       標準環境: .venv
# ==============================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# TASK_SUITES: カンマ区切りで複数スイートを一括実行 (例: "spatial,object,goal,10")
# 複数指定時は各スイートを順に再実行します
TASK_SUITES="${TASK_SUITES:-}"
TASK_SUITE="${TASK_SUITE:-spatial}"

if [ -n "${TASK_SUITES}" ]; then
  IFS=',' read -r -a _SUITES <<< "${TASK_SUITES}"
  if [ "${#_SUITES[@]}" -gt 1 ]; then
    for _S in "${_SUITES[@]}"; do
      _S="${_S// /}"
      echo ""
      echo "╔══════════════════════════════════════════════════════════╗"
      echo "  TASK_SUITE = ${_S}"
      echo "╚══════════════════════════════════════════════════════════╝"
      TASK_SUITE="${_S}" TASK_SUITES="" CKPT="" OUTPUT_DIR="" bash "$0"
    done
    exit 0
  else
    TASK_SUITE="${_SUITES[0]// /}"
  fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/baseline_ar_pixel_wm/${TASK_SUITE}}"
DATA_ROOT="${DATA_ROOT:-data/modified_libero_rlds}"
NUM_WINDOWS="${NUM_WINDOWS:-200}"
EVAL_HORIZON="${EVAL_HORIZON:-8}"
SEEDS="${SEEDS:-42,43,44}"    # カンマ区切りで複数シード指定可: "42,43,44"
SEED="${SEED:-42}"            # 後方互換: SEEDS未指定時のデフォルト
REGENERATE_MANIFEST="${REGENERATE_MANIFEST:-0}"
OVERWRITE="${OVERWRITE:-0}"           # 1 にすると既存の評価結果を無視して再評価
GENERATE_REPORT="${GENERATE_REPORT:-1}"
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-8}"
DEVICE="${DEVICE:-auto}"
SMOKE="${SMOKE:-0}"
VENV_NAME="${VENV_NAME:-.venv5090_eval}"
PYTHON="${REPO_ROOT}/${VENV_NAME}/bin/python3.10"
NUM_SHUFFLE_REPS="${NUM_SHUFFLE_REPS:-1}"   # 負例サンプリング回数 (default 1: 速度優先)

# ── GPU数の自動検出 ────────────────────────────────────────────────────────
if [ -z "${NUM_GPUS:-}" ]; then
  if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NUM_GPUS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
  else
    NUM_GPUS=1
  fi
fi

# ── チェックポイント自動検出 ───────────────────────────────────────────────
_WM_BASE="checkpoints/libero/WorldModel/${TASK_SUITE}"
BASE_MODEL_CONFIG="${BASE_MODEL_CONFIG:-${_WM_BASE}}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-checkpoints/libero/WorldModel/Tokenizer}"

if [ -z "${CKPT:-}" ]; then
  # checkpoint-150000 などを含む最新のサブディレクトリを検索
  CKPT=$(find "${_WM_BASE}" -name "model.safetensors" \
    ! -path "${_WM_BASE}/model.safetensors" \
    2>/dev/null \
    | xargs -I{} dirname {} \
    | sort -r \
    | head -1 || true)
  if [ -z "${CKPT}" ]; then
    echo "[ERROR] チェックポイントが見つかりません: ${_WM_BASE}" >&2
    echo "        CKPT=... を明示的に指定してください" >&2
    exit 1
  fi
fi

# ── 設定確認 ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  AR-Pixel World Model — 単独評価"
echo ""
echo "  TASK_SUITE          = ${TASK_SUITE}"
echo "  CKPT                = ${CKPT}"
echo "  BASE_MODEL_CONFIG   = ${BASE_MODEL_CONFIG}"
echo "  TOKENIZER_CKPT      = ${TOKENIZER_CKPT}"
echo "  OUTPUT_DIR          = ${OUTPUT_DIR}"
echo "  DATA_ROOT           = ${DATA_ROOT}"
echo "  NUM_WINDOWS         = ${NUM_WINDOWS}"
echo "  EVAL_HORIZON        = ${EVAL_HORIZON}"
echo "  SEEDS               = ${SEEDS}"
echo "  DEVICE              = ${DEVICE}"
echo "  REGENERATE_MANIFEST = ${REGENERATE_MANIFEST}"
echo "  GENERATE_REPORT     = ${GENERATE_REPORT}"
echo "  SMOKE               = ${SMOKE}"
echo "  NUM_SHUFFLE_REPS    = ${NUM_SHUFFLE_REPS}"
echo "  NUM_GPUS            = ${NUM_GPUS}"
echo "  VENV_NAME           = ${VENV_NAME}"
echo "  PYTHON              = ${PYTHON}"
echo "  CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES:-（未設定）}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="worldmodel/_compat:train/verl:third_party/LIBERO:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="false"

SMOKE_FLAG=""
[ "${SMOKE}" = "1" ] && SMOKE_FLAG="--smoke"

# シードリストを配列に変換
IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
_NUM_SEEDS="${#SEED_ARRAY[@]}"

# ── シードごとのループ（Step 1 + Step 2）─────────────────────────────────
for _SEED in "${SEED_ARRAY[@]}"; do
  _SEED="${_SEED// /}"   # 空白除去

  # シードが1つの場合は OUTPUT_DIR 直下、複数の場合はサブディレクトリ
  if [ "${_NUM_SEEDS}" -eq 1 ]; then
    _SEED_OUT="${OUTPUT_DIR}"
  else
    _SEED_OUT="${OUTPUT_DIR}/seed_${_SEED}"
  fi
  mkdir -p "${_SEED_OUT}"

  _MANIFEST="${_SEED_OUT}/window_manifest.json"

  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  シード: ${_SEED}  出力先: ${_SEED_OUT}"
  echo "════════════════════════════════════════════════════════════"

  # ── Step 1: ウィンドウマニフェスト生成 ──────────────────────────────────
  echo ""
  if [ "${REGENERATE_MANIFEST}" = "1" ] || [ ! -f "${_MANIFEST}" ]; then
    echo ">>> Step 1 [seed=${_SEED}]: ウィンドウマニフェスト生成 ..."
    "${PYTHON}" analysis/worldmodel/generate_manifest.py \
      --task-suite "${TASK_SUITE}" \
      --data-root "${DATA_ROOT}" \
      --output "${_MANIFEST}" \
      --eval-horizon "${EVAL_HORIZON}" \
      --num-windows "${NUM_WINDOWS}" \
      --seed "${_SEED}"
    echo "    マニフェスト保存 → ${_MANIFEST}"
  else
    echo ">>> Step 1 [seed=${_SEED}]: 既存マニフェストを使用: ${_MANIFEST}"
  fi

  # 既に評価済みならスキップ（OVERWRITE=1 で強制再評価）
  if [ -f "${_SEED_OUT}/aggregate_metrics.json" ] && [ "${REGENERATE_MANIFEST}" != "1" ] && [ "${OVERWRITE}" != "1" ]; then
    echo ">>> Step 2 [seed=${_SEED}]: 評価済みのためスキップ (aggregate_metrics.json 存在)"
    continue
  fi

  # ── Step 2: AR-Pixel WM 評価 ─────────────────────────────────────────────
  echo ""
  echo ">>> Step 2 [seed=${_SEED}]: AR-Pixel WM 評価中 (NUM_GPUS=${NUM_GPUS}) ..."

  _EVAL_COMMON_ARGS=(
    --phase0-ar-pixel-ckpt "${CKPT}"
    --phase0-ar-pixel-config "${BASE_MODEL_CONFIG}"
    --tokenizer-ckpt "${TOKENIZER_CKPT}"
    --window-manifest "${_MANIFEST}"
    --output-dir "${_SEED_OUT}"
    --task-suite "${TASK_SUITE}"
    --data-root "${DATA_ROOT}"
    --eval-horizon "${EVAL_HORIZON}"
    --decode-chunk-size "${DECODE_CHUNK_SIZE}"
    --device "${DEVICE}"
    --seed "${_SEED}"
    --num-shuffle-reps "${NUM_SHUFFLE_REPS}"
  )

  if [ "${NUM_GPUS}" -le 1 ]; then
    "${PYTHON}" analysis/worldmodel/evaluate_ar_pixel_on_manifest.py \
      "${_EVAL_COMMON_ARGS[@]}" \
      --num-shards 1 --shard-index 0 \
      ${SMOKE_FLAG} \
      2>&1 | tee "${_SEED_OUT}/eval.log"
  else
    echo "    GPU 0..$((NUM_GPUS - 1)) でシャードを並列実行します ..."
    _PIDS=()
    for _I in $(seq 0 $((NUM_GPUS - 1))); do
      _GPU_LOG="${_SEED_OUT}/eval_gpu${_I}.log"
      (
        export CUDA_VISIBLE_DEVICES="${_I}"
        "${PYTHON}" analysis/worldmodel/evaluate_ar_pixel_on_manifest.py \
          "${_EVAL_COMMON_ARGS[@]}" \
          --num-shards "${NUM_GPUS}" --shard-index "${_I}" \
          ${SMOKE_FLAG:+${SMOKE_FLAG}}
      ) > "${_GPU_LOG}" 2>&1 &
      _PIDS+=($!)
      echo "    GPU${_I} PID=${_PIDS[-1]}  ログ → ${_GPU_LOG}"
    done

    _FAIL=0
    for _I in "${!_PIDS[@]}"; do
      wait "${_PIDS[_I]}" || { echo "[ERROR] GPU${_I} が失敗しました (PID=${_PIDS[_I]})" >&2; _FAIL=1; }
    done
    [ "${_FAIL}" -eq 1 ] && exit 1

    echo ""
    echo ">>> Step 2b [seed=${_SEED}]: シャードのマージ ..."
    "${PYTHON}" analysis/worldmodel/merge_eval_shards.py \
      --output-dir "${_SEED_OUT}" \
      --num-shards "${NUM_GPUS}" \
      2>&1 | tee "${_SEED_OUT}/merge.log"
  fi

  echo "    評価完了 → ${_SEED_OUT}/aggregate_metrics.json"
done

# ── Step 3: マルチシード平均集計 ──────────────────────────────────────────
_FINAL_METRICS="${OUTPUT_DIR}/aggregate_metrics.json"
if [ "${_NUM_SEEDS}" -gt 1 ]; then
  echo ""
  echo ">>> Step 3: ${_NUM_SEEDS}シードの平均集計 ..."
  "${PYTHON}" analysis/worldmodel/average_seed_metrics.py \
    --output-dir "${OUTPUT_DIR}" \
    --seeds "${SEEDS}"
  _FINAL_METRICS="${OUTPUT_DIR}/aggregate_metrics_multiseed.json"
  echo "    平均メトリクス保存 → ${_FINAL_METRICS}"
fi

# ── Step 4: 日本語レポート生成 ────────────────────────────────────────────
if [ "${GENERATE_REPORT}" = "1" ]; then
  echo ""
  echo ">>> Step 4: 日本語評価レポート生成 ..."

  if [ ! -f "${_FINAL_METRICS}" ]; then
    echo "    [WARN] ${_FINAL_METRICS} が見つかりません。レポート生成をスキップ。" >&2
  else
    "${PYTHON}" analysis/worldmodel/report_ja.py \
      --metrics-file "${_FINAL_METRICS}" \
      --output "${OUTPUT_DIR}/wm_result.md" \
      --ckpt "${CKPT}" \
      --task-suite "${TASK_SUITE}"
    echo "    レポート保存 → ${OUTPUT_DIR}/wm_result.md"
  fi
fi

# ── 完了 ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  評価完了！"
echo ""
echo "  シード            : ${SEEDS}"
if [ "${_NUM_SEEDS}" -gt 1 ]; then
  echo "  平均メトリクス    : ${OUTPUT_DIR}/aggregate_metrics_multiseed.json"
  for _SEED in "${SEED_ARRAY[@]}"; do
    _SEED="${_SEED// /}"
    echo "  seed ${_SEED} 結果   : ${OUTPUT_DIR}/seed_${_SEED}/aggregate_metrics.json"
  done
else
  echo "  集計メトリクス    : ${OUTPUT_DIR}/aggregate_metrics.json"
  echo "  評価ログ          : ${OUTPUT_DIR}/eval.log"
fi
if [ "${GENERATE_REPORT}" = "1" ]; then
  echo "  日本語レポート    : ${OUTPUT_DIR}/wm_result.md"
fi
echo "============================================================"
