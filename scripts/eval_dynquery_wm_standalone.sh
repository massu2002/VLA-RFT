#!/usr/bin/env bash
# ==============================================================
# eval_dynquery_wm_standalone.sh — DynQuery WM 単独評価スクリプト
# ==============================================================
# Phase1 DynQueryWorldModel を RLDS データから直接評価します。
# ウィンドウマニフェストを生成し、evaluate_dynquery_on_manifest.py を
# 実行して aggregate_metrics.json を保存します。
# AR-Pixel WM 評価スクリプト (eval_ar_pixel_wm_standalone.sh) と
# 同じ出力フォーマット・手順に準拠しています。
#
# 使い方:
#   # core_sweep の dq_baseline を評価する場合:
#   SWEEP_NAME=core_sweep EXP_NAME=dq_baseline bash scripts/eval_dynquery_wm_standalone.sh
#
#   # 任意のモデルディレクトリを直接指定する場合:
#   MODEL_DIR=checkpoints/libero/DynQueryWorldModel/core_sweep/spatial/dq_baseline/s42/final \
#   OUTPUT_DIR=results/phase1/my_run \
#   bash scripts/eval_dynquery_wm_standalone.sh
#
# 主な環境変数:
#   SWEEP_NAME          スイープ名 (default: core_sweep)
#                       checkpoints/libero/DynQueryWorldModel/{SWEEP_NAME}/... を参照
#   EXP_NAME            実験名 (default: dq_baseline)
#   TASK_SUITE          タスクスイート (default: spatial)
#   MODEL_DIR           チェックポイントディレクトリ (省略時: 自動構築)
#                       {CKPT_ROOT}/{SWEEP_NAME}/{TASK_SUITE}/{EXP_NAME}/s{SEED}/final
#   CKPT_ROOT           チェックポイントルート (default: checkpoints/libero/DynQueryWorldModel)
#   OUTPUT_DIR          結果保存先 (省略時: results/phase1/DynQueryWorldModel_{SWEEP_NAME}/{EXP_NAME})
#   DATA_ROOT           RLDS データルート (default: data/modified_libero_rlds)
#   NUM_WINDOWS         評価ウィンドウ数 (default: 200)
#   EVAL_HORIZON        予測ホライズン (default: 8)
#   SEEDS               評価シード (default: 42,43,44)
#                       複数指定時は seed_N/ サブディレクトリに結果を保存し平均集計
#   REGENERATE_MANIFEST 1 にするとマニフェストを強制再生成 (default: 0)
#   OVERWRITE           1 にすると既存の評価結果を無視して再評価 (default: 0)
#   GENERATE_REPORT     1 にすると評価後に日本語レポートを生成 (default: 1)
#   DEVICE              auto / cuda / cpu (default: auto)
#   SMOKE               1 にするとスモークテスト（2ウィンドウのみ）
#   NUM_GPUS            並列GPU数 (未設定時: nvidia-smi で自動検出)
#                       2以上の場合はウィンドウをGPU間で分割して並列評価し
#                       完了後に自動マージします (CUDA_VISIBLE_DEVICES=0,1,...)
#   MANIFEST_ROOT       シードごとのマニフェスト参照元ディレクトリ
#                       (default: results/baseline_ar_pixel_wm/{TASK_SUITE})
#                       seed_N/window_manifest.json が存在すればそれを優先使用し、
#                       生成をスキップします。AR-Pixel WM と同じウィンドウで
#                       公平比較するために使います。
#                       REGENERATE_MANIFEST=1 で無視して再生成します。
#   CONDITION_NAME      evaluate_dynquery_on_manifest.py に渡す --condition-name
#                       (省略時: EXP_NAME を使用)
#   VENV_NAME           仮想環境名 (default: .venv5090_eval)
#                       RTX 5090: .venv5090_eval (PyTorch 2.8+cu128)
#                       標準環境: .venv
# ==============================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

SWEEP_NAME="${SWEEP_NAME:-core_sweep}"
EXP_NAME="${EXP_NAME:-dq_baseline}"

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
      TASK_SUITE="${_S}" TASK_SUITES="" OUTPUT_DIR="" MANIFEST_ROOT="" bash "$0"
    done
    exit 0
  else
    TASK_SUITE="${_SUITES[0]// /}"
  fi
fi
CKPT_ROOT="${CKPT_ROOT:-checkpoints/libero/DynQueryWorldModel}"
DATA_ROOT="${DATA_ROOT:-data/modified_libero_rlds}"
NUM_WINDOWS="${NUM_WINDOWS:-200}"
EVAL_HORIZON="${EVAL_HORIZON:-8}"
SEEDS="${SEEDS:-42,43,44}"
# CKPT_SEED: モデルチェックポイントに使うシード。
# 省略時は SEEDS の先頭シードを使う（評価シードが複数でもモデルは共通）。
# 明示指定例: CKPT_SEED=42
CKPT_SEED="${CKPT_SEED:-}"
REGENERATE_MANIFEST="${REGENERATE_MANIFEST:-0}"
OVERWRITE="${OVERWRITE:-0}"
GENERATE_REPORT="${GENERATE_REPORT:-1}"
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-8}"
DEVICE="${DEVICE:-auto}"
SMOKE="${SMOKE:-0}"
VENV_NAME="${VENV_NAME:-.venv5090_eval}"
PYTHON="${REPO_ROOT}/${VENV_NAME}/bin/python3.10"
CONDITION_NAME="${CONDITION_NAME:-${EXP_NAME}}"
NUM_SHUFFLE_REPS="${NUM_SHUFFLE_REPS:-1}"
# DECODE_CHUNK_SIZE は AR-Pixel 専用のため DynQuery では不使用

# デフォルト出力先（タスクスイートを含む）
OUTPUT_DIR="${OUTPUT_DIR:-results/phase1/DynQueryWorldModel_${SWEEP_NAME}/${TASK_SUITE}/${EXP_NAME}}"

# AR-Pixel WM と同一マニフェストで評価するためのルートディレクトリ
# seed_N/window_manifest.json が存在すればそれを優先参照する
MANIFEST_ROOT="${MANIFEST_ROOT:-results/baseline_ar_pixel_wm/${TASK_SUITE}}"

# CKPT_SEED が未指定なら SEEDS の先頭を使う
if [ -z "${CKPT_SEED}" ]; then
  CKPT_SEED="${SEEDS%%,*}"
fi

# ── GPU数の自動検出 ────────────────────────────────────────────────────────
if [ -z "${NUM_GPUS:-}" ]; then
  if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NUM_GPUS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
  else
    NUM_GPUS=1
  fi
fi

# ── 設定確認 ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  DynQuery World Model — 単独評価"
echo ""
echo "  SWEEP_NAME          = ${SWEEP_NAME}"
echo "  EXP_NAME            = ${EXP_NAME}"
echo "  TASK_SUITE          = ${TASK_SUITE}"
echo "  CKPT_ROOT           = ${CKPT_ROOT}"
echo "  OUTPUT_DIR          = ${OUTPUT_DIR}"
echo "  DATA_ROOT           = ${DATA_ROOT}"
echo "  NUM_WINDOWS         = ${NUM_WINDOWS}"
echo "  EVAL_HORIZON        = ${EVAL_HORIZON}"
echo "  SEEDS               = ${SEEDS}"
echo "  CKPT_SEED           = ${CKPT_SEED}  (モデル共通シード)"
echo "  MANIFEST_ROOT       = ${MANIFEST_ROOT}"
echo "  DEVICE              = ${DEVICE}"
echo "  REGENERATE_MANIFEST = ${REGENERATE_MANIFEST}"
echo "  OVERWRITE           = ${OVERWRITE}"
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

  # ── モデルディレクトリの決定 ─────────────────────────────────────────────
  if [ -n "${MODEL_DIR:-}" ]; then
    # 明示的に指定された場合はそのまま使う（シード問わず同一ディレクトリ）
    _MODEL_DIR="${MODEL_DIR}"
  else
    # CKPT_ROOT/{SWEEP_NAME}/{TASK_SUITE}/{EXP_NAME}/s{CKPT_SEED}/final を使う
    # 評価シード (_SEED) とは独立して CKPT_SEED のモデルを共通使用
    _CKPT_SEED_DIR="${CKPT_ROOT}/${SWEEP_NAME}/${TASK_SUITE}/${EXP_NAME}/s${CKPT_SEED}"
    # final/ があればそこを使い、なければ checkpoint-XXX/ の最新を探す
    if [ -d "${_CKPT_SEED_DIR}/final" ]; then
      _MODEL_DIR="${_CKPT_SEED_DIR}/final"
    else
      _MODEL_DIR=$(find "${_CKPT_SEED_DIR}" -maxdepth 1 -name "checkpoint-*" \
        -type d 2>/dev/null | sort -r | head -1 || true)
      if [ -z "${_MODEL_DIR}" ]; then
        echo "[ERROR] チェックポイントが見つかりません: ${_CKPT_SEED_DIR}" >&2
        echo "        CKPT_SEED=${CKPT_SEED} のモデルが存在しません。" >&2
        echo "        MODEL_DIR=... を明示指定するか、CKPT_SEED を変更してください" >&2
        exit 1
      fi
    fi
  fi

  # シードが1つの場合は OUTPUT_DIR 直下、複数の場合はサブディレクトリ
  if [ "${_NUM_SEEDS}" -eq 1 ]; then
    _SEED_OUT="${OUTPUT_DIR}"
  else
    _SEED_OUT="${OUTPUT_DIR}/seed_${_SEED}"
  fi
  mkdir -p "${_SEED_OUT}"

  # ── マニフェストパスの決定 ───────────────────────────────────────────────
  # 優先順位:
  #   1. MANIFEST_ROOT/seed_{SEED}/window_manifest.json  (AR-Pixel 基準、複数シード時)
  #   2. MANIFEST_ROOT/window_manifest.json              (AR-Pixel 基準、単一シード時)
  #   3. _SEED_OUT/window_manifest.json                  (DynQuery 出力先に生成)
  _MANIFEST_SRC_SEED="${MANIFEST_ROOT}/seed_${_SEED}/window_manifest.json"
  _MANIFEST_SRC_FLAT="${MANIFEST_ROOT}/window_manifest.json"
  _MANIFEST="${_SEED_OUT}/window_manifest.json"

  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  シード: ${_SEED}  モデル: ${_MODEL_DIR}"
  echo "  出力先: ${_SEED_OUT}"
  echo "════════════════════════════════════════════════════════════"

  # ── Step 1: ウィンドウマニフェスト取得 ──────────────────────────────────
  echo ""
  if [ "${REGENERATE_MANIFEST}" = "1" ]; then
    echo ">>> Step 1 [seed=${_SEED}]: マニフェスト強制再生成 (REGENERATE_MANIFEST=1) ..."
    "${PYTHON}" analysis/worldmodel/generate_manifest.py \
      --task-suite "${TASK_SUITE}" \
      --data-root "${DATA_ROOT}" \
      --output "${_MANIFEST}" \
      --eval-horizon "${EVAL_HORIZON}" \
      --num-windows "${NUM_WINDOWS}" \
      --seed "${_SEED}"
    echo "    マニフェスト保存 → ${_MANIFEST}"
  elif [ -f "${_MANIFEST_SRC_SEED}" ]; then
    # AR-Pixel WM の seed_N/ マニフェストを参照（コピーして再現性を保持）
    cp "${_MANIFEST_SRC_SEED}" "${_MANIFEST}"
    echo ">>> Step 1 [seed=${_SEED}]: AR-Pixel 基準マニフェストを使用"
    echo "    参照元 → ${_MANIFEST_SRC_SEED}"
    echo "    コピー先 → ${_MANIFEST}"
  elif [ -f "${_MANIFEST_SRC_FLAT}" ]; then
    cp "${_MANIFEST_SRC_FLAT}" "${_MANIFEST}"
    echo ">>> Step 1 [seed=${_SEED}]: AR-Pixel 基準マニフェスト（flat）を使用"
    echo "    参照元 → ${_MANIFEST_SRC_FLAT}"
    echo "    コピー先 → ${_MANIFEST}"
  elif [ -f "${_MANIFEST}" ]; then
    echo ">>> Step 1 [seed=${_SEED}]: 既存マニフェストを使用: ${_MANIFEST}"
  else
    echo ">>> Step 1 [seed=${_SEED}]: マニフェストを新規生成 ..."
    "${PYTHON}" analysis/worldmodel/generate_manifest.py \
      --task-suite "${TASK_SUITE}" \
      --data-root "${DATA_ROOT}" \
      --output "${_MANIFEST}" \
      --eval-horizon "${EVAL_HORIZON}" \
      --num-windows "${NUM_WINDOWS}" \
      --seed "${_SEED}"
    echo "    マニフェスト保存 → ${_MANIFEST}"
  fi

  # 既に評価済みならスキップ（OVERWRITE=1 で強制再評価）
  if [ -f "${_SEED_OUT}/aggregate_metrics.json" ] && [ "${REGENERATE_MANIFEST}" != "1" ] && [ "${OVERWRITE}" != "1" ]; then
    echo ">>> Step 2 [seed=${_SEED}]: 評価済みのためスキップ (aggregate_metrics.json 存在)"
    continue
  fi

  # ── Step 2: DynQuery WM 評価 ─────────────────────────────────────────────
  echo ""
  echo ">>> Step 2 [seed=${_SEED}]: DynQuery WM 評価中 (NUM_GPUS=${NUM_GPUS}) ..."

  _EVAL_COMMON_ARGS=(
    --model-dir "${_MODEL_DIR}"
    --window-manifest "${_MANIFEST}"
    --output-dir "${_SEED_OUT}"
    --task-suite "${TASK_SUITE}"
    --data-root "${DATA_ROOT}"
    --eval-horizon "${EVAL_HORIZON}"
    --device "${DEVICE}"
    --seed "${_SEED}"
    --num-shuffle-reps "${NUM_SHUFFLE_REPS}"
    --condition-name "${CONDITION_NAME}"
  )

  if [ "${NUM_GPUS}" -le 1 ]; then
    "${PYTHON}" analysis/worldmodel/evaluate_dynquery_on_manifest.py \
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
        "${PYTHON}" analysis/worldmodel/evaluate_dynquery_on_manifest.py \
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
    # MODEL_DIR は最後のシードのものを使用（代表値）
    _REPORT_MODEL_DIR="${_MODEL_DIR:-${CKPT_ROOT}/${SWEEP_NAME}/${TASK_SUITE}/${EXP_NAME}}"
    "${PYTHON}" analysis/worldmodel/report_ja.py \
      --metrics-file "${_FINAL_METRICS}" \
      --output "${OUTPUT_DIR}/wm_result.md" \
      --ckpt "${_REPORT_MODEL_DIR}" \
      --task-suite "${TASK_SUITE}"
    echo "    レポート保存 → ${OUTPUT_DIR}/wm_result.md"
  fi
fi

# ── 完了 ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  評価完了！"
echo ""
echo "  SWEEP_NAME        : ${SWEEP_NAME}"
echo "  EXP_NAME          : ${EXP_NAME}"
echo "  TASK_SUITE        : ${TASK_SUITE}"
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
