#!/usr/bin/env bash
# select_best_for_rft.sh — sweep 結果から RFT 用に最良の run を軸ごとに選択・整理する
#
# 使い方:
#   bash scripts/libero/residual_worldmodel/select_best_for_rft.sh [KEY=VALUE ...]
#
# KEY=VALUE 引数一覧:
#   SWEEP_NAME=baseline          sweep サブディレクトリ名 (デフォルト: baseline)
#   SEARCH_AXIS=action_horizon   グループ化する config キー名
#   AXIS_PREFIX=hor              軸ディレクトリのプレフィックス (空=自動)
#   RFT_METRIC=pairwise_acc      選択指標
#   HIGHER_IS_BETTER=true        true=大きい方が良い / false=小さい方が良い
#   GROUP_BY_TASK=true           true=タスク × 軸値 / false=軸値のみ
#   TASK_FILTER=spatial,goal     対象タスクをカンマ区切りで絞り込む (空=全タスク)
#   SEARCH_ROOT=<path>           探索ルートを直接指定 (指定時 SWEEP_NAME 無効)
#   DEST_ROOT=<path>             出力先を直接指定 (指定時 SWEEP_NAME 無効)
#   COPY_MODE=false              true=ディレクトリコピー / false=シンボリックリンク
#   DRY_RUN=false                true=実行せず表示のみ
#   LIST_ONLY=false              true=一覧表示のみ (デプロイしない)
#
# 例:
#   # action_horizon 軸で baseline sweep を選択
#   bash select_best_for_rft.sh
#
#   # ablation sweep を lr 軸で選択 (一覧のみ)
#   bash select_best_for_rft.sh SWEEP_NAME=ablation SEARCH_AXIS=lr AXIS_PREFIX=lr LIST_ONLY=true
#
#   # dino_feature_loss_weight 軸でコピーモード
#   bash select_best_for_rft.sh SWEEP_NAME=baseline SEARCH_AXIS=dino_feature_loss_weight AXIS_PREFIX=dw COPY_MODE=true
#
#   # タスクを絞り込んで dry-run
#   bash select_best_for_rft.sh TASK_FILTER=spatial,goal DRY_RUN=true
#
# 出力構造:
#   DEST_ROOT/
#     {axis_prefix}{value}/     例: hor4/
#       {task_name}/            例: spatial/
#         {run_name}            (symlink or copy)
#     rft_selection_{axis}.json

set -euo pipefail

######################################################################
########## USER CONFIG — デフォルト値 (KEY=VALUE 引数で上書き可) ##########
######################################################################

SWEEP_NAME="baseline_v3"
SEARCH_AXIS="action_horizon"
AXIS_PREFIX=""           # 空にすると SEARCH_AXIS から自動推定
RFT_METRIC="pairwise_acc"
HIGHER_IS_BETTER="true"
GROUP_BY_TASK="true"
TASK_FILTER=""           # カンマ区切り例: "spatial,goal"
SEARCH_ROOT=""           # 空にすると SWEEP_NAME から自動生成
DEST_ROOT=""             # 空にすると SWEEP_NAME から自動生成
COPY_MODE="${COPY_MODE:-false}"
DRY_RUN="${DRY_RUN:-false}"
LIST_ONLY="${LIST_ONLY:-false}"

######################################################################
########## END USER CONFIG ##########
######################################################################

# ---------------------------------------------------------------------------
# Parse KEY=VALUE arguments
# ---------------------------------------------------------------------------
for _arg in "$@"; do
  case "${_arg}" in
    SWEEP_NAME=*)        SWEEP_NAME="${_arg#*=}" ;;
    SEARCH_AXIS=*)       SEARCH_AXIS="${_arg#*=}" ;;
    AXIS_PREFIX=*)       AXIS_PREFIX="${_arg#*=}" ;;
    RFT_METRIC=*)        RFT_METRIC="${_arg#*=}" ;;
    HIGHER_IS_BETTER=*)  HIGHER_IS_BETTER="${_arg#*=}" ;;
    GROUP_BY_TASK=*)     GROUP_BY_TASK="${_arg#*=}" ;;
    TASK_FILTER=*)       TASK_FILTER="${_arg#*=}" ;;
    SEARCH_ROOT=*)       SEARCH_ROOT="${_arg#*=}" ;;
    DEST_ROOT=*)         DEST_ROOT="${_arg#*=}" ;;
    COPY_MODE=*)         COPY_MODE="${_arg#*=}" ;;
    DRY_RUN=*)           DRY_RUN="${_arg#*=}" ;;
    LIST_ONLY=*)         LIST_ONLY="${_arg#*=}" ;;
    --help|-h)
      sed -n '2,/^set -euo/{ /^set -euo/d; s/^# \{0,1\}//; p }' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) echo "[WARN] Unknown argument: ${_arg}" >&2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Common setup
# ---------------------------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"
setup_env

# ---------------------------------------------------------------------------
# Resolve paths and AXIS_PREFIX
# ---------------------------------------------------------------------------
if [ -z "${SEARCH_ROOT}" ]; then
  SEARCH_ROOT="${REPO_ROOT}/checkpoints/libero/FocusedWM/sweep/${SWEEP_NAME}"
fi
if [ -z "${DEST_ROOT}" ]; then
  DEST_ROOT="${REPO_ROOT}/checkpoints/libero/FocusedWM/${SWEEP_NAME}"
fi

# Auto-derive AXIS_PREFIX from SEARCH_AXIS if not specified
if [ -z "${AXIS_PREFIX}" ]; then
  case "${SEARCH_AXIS}" in
    action_horizon)              AXIS_PREFIX="hor" ;;
    lr|learning_rate)            AXIS_PREFIX="lr" ;;
    dino_feature_loss_weight)    AXIS_PREFIX="dw" ;;
    focus_sparsity_weight)       AXIS_PREFIX="sw" ;;
    focus_supervision_weight)    AXIS_PREFIX="fw" ;;
    ranking_score_type)          AXIS_PREFIX="rk" ;;
    negative_mode)               AXIS_PREFIX="neg" ;;
    seed)                        AXIS_PREFIX="s" ;;
    batch_size)                  AXIS_PREFIX="bs" ;;
    *)                           AXIS_PREFIX="" ;;
  esac
fi

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ ! -d "${SEARCH_ROOT}" ]; then
  echo "[ERROR] SEARCH_ROOT not found: ${SEARCH_ROOT}" >&2
  echo "  SWEEP_NAME=${SWEEP_NAME}" >&2
  echo "  Try: bash $(basename "${BASH_SOURCE[0]}") SWEEP_NAME=<name>" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
print_header "select_best_for_rft: ${SWEEP_NAME}"
printf "  %-28s %s\n" "SEARCH_ROOT"      "${SEARCH_ROOT}"
printf "  %-28s %s\n" "SEARCH_AXIS"      "${SEARCH_AXIS}"
printf "  %-28s %s\n" "AXIS_PREFIX"      "${AXIS_PREFIX:-<none>}"
printf "  %-28s %s\n" "RFT_METRIC"       "${RFT_METRIC}"
printf "  %-28s %s\n" "HIGHER_IS_BETTER" "${HIGHER_IS_BETTER}"
printf "  %-28s %s\n" "GROUP_BY_TASK"    "${GROUP_BY_TASK}"
printf "  %-28s %s\n" "TASK_FILTER"      "${TASK_FILTER:-<all>}"
printf "  %-28s %s\n" "DEST_ROOT"        "${DEST_ROOT}"
printf "  %-28s %s\n" "COPY_MODE"        "${COPY_MODE}"
printf "  %-28s %s\n" "DRY_RUN"          "${DRY_RUN}"
printf "  %-28s %s\n" "LIST_ONLY"        "${LIST_ONLY}"
printf '%*s\n' 68 '' | tr ' ' '-'
echo ""

# ---------------------------------------------------------------------------
# Python: scan, score, select, deploy
# ---------------------------------------------------------------------------
python - <<PYEOF
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Optional

SEARCH_ROOT      = Path("${SEARCH_ROOT}")
SEARCH_AXIS      = "${SEARCH_AXIS}"
AXIS_PREFIX      = "${AXIS_PREFIX}"
RFT_METRIC       = "${RFT_METRIC}"
HIGHER_IS_BETTER = "${HIGHER_IS_BETTER}" == "true"
GROUP_BY_TASK    = "${GROUP_BY_TASK}" == "true"
TASK_FILTER_RAW  = "${TASK_FILTER}"
TASK_FILTER      = [t.strip() for t in TASK_FILTER_RAW.split(",") if t.strip()] if TASK_FILTER_RAW.strip() else []
DEST_ROOT        = Path("${DEST_ROOT}")
COPY_MODE        = "${COPY_MODE}" == "true"
DRY_RUN          = "${DRY_RUN}" == "true"
LIST_ONLY        = "${LIST_ONLY}" == "true"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_json(path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def read_jsonl_last(path) -> dict:
    try:
        last = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except Exception:
                        pass
        return last or {}
    except Exception:
        return {}


def get_best_rank_metric(run_dir: Path, metric: str) -> Optional[float]:
    """Read metric from the highest-step checkpoint-best_rank-*/meta.json."""
    candidates = sorted(run_dir.glob("checkpoint-best_rank-*"))
    for ckpt in reversed(candidates):
        meta = read_json(ckpt / "meta.json")
        if metric in meta:
            return float(meta[metric])
    return None


def get_metric_fallback(run_dir: Path, metric: str) -> Optional[float]:
    """Fallback: search all checkpoint metas then jsonl last line."""
    for tag in ("best_rank", "best_recon", "best_dino_feature", "final", "latest"):
        for ckpt in sorted(run_dir.glob(f"checkpoint-{tag}-*"), reverse=True):
            meta = read_json(ckpt / "meta.json")
            if metric in meta:
                return float(meta[metric])
    last = read_jsonl_last(run_dir / "train_metrics.jsonl")
    if metric in last:
        return float(last[metric])
    for alt in (f"rank/{metric}", f"feat/{metric}", f"focus/{metric}"):
        if alt in last:
            return float(last[alt])
    return None


def extract_metric(run_dir: Path, metric: str) -> Optional[float]:
    v = get_best_rank_metric(run_dir, metric)
    if v is not None:
        return v
    return get_metric_fallback(run_dir, metric)


def format_axis_dir(prefix: str, value: str) -> str:
    return f"{prefix}{value}" if prefix else value


# ---------------------------------------------------------------------------
# Scan all run directories
# ---------------------------------------------------------------------------
print(f"  Scanning {SEARCH_ROOT} ...")
runs = []
for cfg_path in sorted(SEARCH_ROOT.rglob("config_dump.json")):
    run_dir = cfg_path.parent
    cfg = read_json(cfg_path)

    axis_value = cfg.get(SEARCH_AXIS)
    if axis_value is None:
        continue

    task_name = cfg.get("task_name", "")
    if TASK_FILTER and task_name not in TASK_FILTER:
        continue

    score = extract_metric(run_dir, RFT_METRIC)
    status = ("done"   if (run_dir / "DONE").exists()   else
              "failed" if (run_dir / "FAILED").exists() else "running")

    runs.append({
        "run_dir":    run_dir,
        "task":       task_name,
        "axis_value": str(axis_value),
        "axis_dir":   format_axis_dir(AXIS_PREFIX, str(axis_value)),
        "score":      score,
        "status":     status,
        "run_name":   run_dir.name,
        "cfg":        cfg,
    })

if not runs:
    print(f"  [ERROR] No runs found with axis '{SEARCH_AXIS}' in {SEARCH_ROOT}", file=sys.stderr)
    sys.exit(1)

print(f"  Found {len(runs)} run(s) with axis '{SEARCH_AXIS}'")
n_done   = sum(1 for r in runs if r["status"] == "done")
n_failed = sum(1 for r in runs if r["status"] == "failed")
n_other  = len(runs) - n_done - n_failed
print(f"  Status — done: {n_done}  failed: {n_failed}  other: {n_other}")

# ---------------------------------------------------------------------------
# Group and select best
# ---------------------------------------------------------------------------
groups: dict = {}
for r in runs:
    key = (r["task"], r["axis_dir"]) if GROUP_BY_TASK else (r["axis_dir"],)
    groups.setdefault(key, []).append(r)

selected = {}
for key, group in sorted(groups.items()):
    scored = [r for r in group if r["score"] is not None]
    if not scored:
        print(f"  [WARN] No metric '{RFT_METRIC}' found for group {key} — skipping")
        continue
    best = max(scored, key=lambda r: r["score"]) if HIGHER_IS_BETTER else min(scored, key=lambda r: r["score"])
    selected[key] = best

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
print()
print(f"  {'GROUP':<35} {'SCORE':>10}  {'STATUS':<8}  RUN_DIR")
print(f"  {'-'*35} {'-'*10}  {'-'*8}  {'-'*50}")
for key, r in sorted(selected.items()):
    group_label = "/".join(key)
    score_str   = f"{r['score']:.5f}" if r["score"] is not None else "N/A"
    short_run   = r["run_name"][-55:] if len(r["run_name"]) > 55 else r["run_name"]
    print(f"  {group_label:<35} {score_str:>10}  {r['status']:<8}  .../{short_run}")

print()
print(f"  Metric: {RFT_METRIC}  (higher_is_better={HIGHER_IS_BETTER})")
print(f"  Selected {len(selected)} run(s) from {len(groups)} group(s)")

if LIST_ONLY:
    print()
    print("  [LIST_ONLY] Done. No files created.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Deploy (symlink or copy)
# ---------------------------------------------------------------------------
print()
print(f"  Deploying to {DEST_ROOT} ...")
print(f"  Mode: {'COPY' if COPY_MODE else 'SYMLINK'}  DRY_RUN={DRY_RUN}")
print()

report = []

for key, r in sorted(selected.items()):
    if GROUP_BY_TASK:
        task_name, axis_dir = key
        dest_parent = DEST_ROOT / axis_dir / task_name
    else:
        (axis_dir,) = key
        dest_parent = DEST_ROOT / axis_dir

    dest_link = dest_parent / r["run_name"]

    print(f"  [{'/'.join(key)}]")
    print(f"    src  : {r['run_dir']}")
    print(f"    dest : {dest_link}")
    print(f"    score: {RFT_METRIC} = {r['score']:.5f}")

    if not DRY_RUN:
        dest_parent.mkdir(parents=True, exist_ok=True)

        if dest_link.exists() or dest_link.is_symlink():
            print(f"    [SKIP] dest already exists — remove manually to overwrite.")
        else:
            if COPY_MODE:
                shutil.copytree(str(r["run_dir"]), str(dest_link), symlinks=True)
                print(f"    [OK] copied")
            else:
                try:
                    rel_src = os.path.relpath(str(r["run_dir"]), str(dest_parent))
                except ValueError:
                    rel_src = str(r["run_dir"])
                dest_link.symlink_to(rel_src)
                print(f"    [OK] symlink → {rel_src}")

        report.append({
            "group":      list(key),
            "axis":       SEARCH_AXIS,
            "axis_value": r["axis_value"],
            "task":       r["task"],
            "metric":     RFT_METRIC,
            "score":      r["score"],
            "status":     r["status"],
            "run_name":   r["run_name"],
            "run_dir":    str(r["run_dir"]),
            "dest":       str(dest_link),
            "config":     r["cfg"],
        })
    else:
        print(f"    [DRY_RUN] would deploy")
    print()

if not DRY_RUN and report:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = DEST_ROOT / f"rft_selection_{SEARCH_AXIS}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Selection report saved: {report_path}")

if DRY_RUN:
    print("  [DRY_RUN] Done. Nothing was written.")
else:
    print(f"  Done. {len(selected)} run(s) deployed to {DEST_ROOT}")
PYEOF
