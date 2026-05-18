#!/usr/bin/env bash
# ==============================================================
# command_train_rft_baseline.sh — baseline AR-pixel WM を使った RFT 事後学習
# ==============================================================
# 目的:
#   train_libero_worldmodel.sh で学習した各タスクの最新 WorldModel を使い、
#   VLA の RFT 事後学習を 4 タスク分実行する。
#
# タスク → WM チェックポイント対応（環境変数で上書き可能）:
#   spatial : checkpoints/libero/WorldModel/spatial/20260514_baseline_ar_pixel_wm
#   object  : checkpoints/libero/WorldModel/object/20260515_baseline_ar_pixel_wm
#   goal    : checkpoints/libero/WorldModel/goal/20260515_baseline_ar_pixel_wm
#   10      : checkpoints/libero/WorldModel/10/20260429_worldmodel_scratch
#             ※ 20260518_worldmodel_scratch は学習中のため旧バージョンを使用
#             ※ 完成後は: WM_VERSION_10=20260518_worldmodel_scratch bash command_train_rft_baseline.sh
#
# 出力:
#   VLA RFT 重み : checkpoints/libero/VLA-RFT/{task}/{DATE}_{POST_EXP_NAME}/
#   ログ         : logs/libero/RFT/{DATE}/{task}_rft_output.log
#
# 実行例:
#   # 全4タスク:
#   bash command_train_rft_baseline.sh
#
#   # タスクを絞る:
#   TASK_FILTER=spatial bash command_train_rft_baseline.sh
#   TASK_FILTER=spatial,object bash command_train_rft_baseline.sh
#
#   # タスク 10 の WM が完成したら新バージョンで再実行:
#   TASK_FILTER=10 WM_VERSION_10=20260518_worldmodel_scratch bash command_train_rft_baseline.sh
#
#   # RFT ステップ数を変える:
#   RFT_STEPS=200 bash command_train_rft_baseline.sh
#
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

# ==============================================================
# タスク → WM バージョン対応（環境変数で上書き可能）
# ==============================================================
WM_VERSION_SPATIAL="${WM_VERSION_SPATIAL:-20260514_baseline_ar_pixel_wm}"
WM_VERSION_OBJECT="${WM_VERSION_OBJECT:-20260515_baseline_ar_pixel_wm}"
WM_VERSION_GOAL="${WM_VERSION_GOAL:-20260515_baseline_ar_pixel_wm}"
WM_VERSION_10="${WM_VERSION_10:-20260429_worldmodel_scratch}"

# ==============================================================
# 共通設定
# ==============================================================
export DATE="${DATE:-$(date +%Y%m%d)}"
export POST_EXP_NAME="${POST_EXP_NAME:-baseline_ar_pixel_wm}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
export RFT_STEPS="${RFT_STEPS:-400}"
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH="$(pwd)/train/verl:${PYTHONPATH:-}"

WM_ROOT="${WM_ROOT:-checkpoints/libero/WorldModel}"
BASE_VLA_ROOT="${BASE_VLA_ROOT:-checkpoints/libero/Base_VLA}"

TASK_FILTER="${TASK_FILTER:-}"
TASKS=(spatial object goal 10)

source .venv/bin/activate

# ==============================================================
# 設定確認ログ
# ==============================================================
echo "============================================================"
echo "RFT 事後学習 — baseline AR-pixel WorldModel"
echo ""
echo "  POST_EXP_NAME   = ${POST_EXP_NAME}"
echo "  RFT_STEPS       = ${RFT_STEPS}"
echo "  N_GPUS_PER_NODE = ${N_GPUS_PER_NODE}"
echo "  DATE            = ${DATE}"
echo "  TASK_FILTER     = ${TASK_FILTER:-（未設定: 全4タスク）}"
echo ""
echo "  [WM チェックポイント]"
printf "  %-8s : %s\n" "spatial" "${WM_ROOT}/spatial/${WM_VERSION_SPATIAL}"
printf "  %-8s : %s\n" "object"  "${WM_ROOT}/object/${WM_VERSION_OBJECT}"
printf "  %-8s : %s\n" "goal"    "${WM_ROOT}/goal/${WM_VERSION_GOAL}"
printf "  %-8s : %s\n" "10"      "${WM_ROOT}/10/${WM_VERSION_10}"
echo ""
echo "  [出力先]"
echo "  checkpoints/libero/VLA-RFT/{task}/${DATE}_${POST_EXP_NAME}/"
echo "  logs/libero/RFT/${DATE}/{task}_rft_output.log"
echo "============================================================"
echo ""

mkdir -p "logs/libero/RFT/${DATE}"

# ==============================================================
# タスクごとに RFT を実行
# ==============================================================
_success=0; _fail=0; _skip=0

for task in "${TASKS[@]}"; do

  # フィルタリング
  if [ -n "${TASK_FILTER}" ]; then
    match=0
    IFS=',' read -ra _filters <<< "${TASK_FILTER}"
    for f in "${_filters[@]}"; do [ "${f}" = "${task}" ] && match=1; done
    [ "${match}" -eq 0 ] && continue
  fi

  # タスクに対応する WM バージョンを選択
  case "${task}" in
    spatial) ver="${WM_VERSION_SPATIAL}" ;;
    object)  ver="${WM_VERSION_OBJECT}"  ;;
    goal)    ver="${WM_VERSION_GOAL}"    ;;
    10)      ver="${WM_VERSION_10}"      ;;
  esac

  wm_path="${WM_ROOT}/${task}/${ver}"
  base_vla_path="${BASE_VLA_ROOT}/${task}"
  rft_out="checkpoints/libero/VLA-RFT/${task}/${DATE}_${POST_EXP_NAME}"
  task_log="logs/libero/RFT/${DATE}/${task}_rft_output.log"

  echo "------------------------------------------------------------"
  echo "  task=${task}"
  echo "  WORLD_MODEL_PATH = ${wm_path}"
  echo "  BASE_VLA_PATH    = ${base_vla_path}"
  echo "  Output           = ${rft_out}"
  echo "  Log              = ${task_log}"

  # model.safetensors の存在確認
  if [ ! -f "${wm_path}/model.safetensors" ]; then
    echo "  [WARN] model.safetensors が見つかりません — スキップします"
    echo "         ${wm_path}/model.safetensors"
    _skip=$((_skip + 1))
    echo ""
    continue
  fi

  echo ""
  echo "===== Starting RFT: task=${task} ====="

  export LIBERO_TASK_NAME="${task}"
  export WORLD_MODEL_PATH="${wm_path}"
  export BASE_VLA_PATH="${base_vla_path}"
  export TENSORBOARD_DIR="logs/libero/RFT/${DATE}/${task}_${POST_EXP_NAME}"

  set +e
  (
    cd "$(pwd)"
    bash train/verl/examples/grpo_trainer/run_vla_rft.sh
  ) 2>&1 | tee "${task_log}"
  _rc="${PIPESTATUS[0]}"
  set -e

  if [ "${_rc}" -eq 0 ]; then
    echo "===== Finished RFT: task=${task} ====="
    _success=$((_success + 1))
  else
    echo "===== FAILED RFT: task=${task} (rc=${_rc}) =====" >&2
    _fail=$((_fail + 1))
  fi
  echo ""
done

# ==============================================================
# 完了メッセージ
# ==============================================================
echo "============================================================"
echo "RFT 事後学習 完了"
echo ""
echo "  成功: ${_success}  失敗: ${_fail}  スキップ: ${_skip}"
echo ""
echo "  RFT 重み保存先: checkpoints/libero/VLA-RFT/"
echo ""
if [ "${_fail}" -gt 0 ]; then
  echo "  [WARNING] ${_fail} タスクが失敗しました。"
  echo "  再実行例: TASK_FILTER=<task> bash command_train_rft_baseline.sh"
fi
echo "============================================================"

[ "${_fail}" -gt 0 ] && exit 1
exit 0
