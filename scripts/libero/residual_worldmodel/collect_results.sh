#!/usr/bin/env bash
# collect_results.sh — 複数 run の結果を一つの CSV にまとめる
#
# 使い方:
#   1. SEARCH_ROOT に run ディレクトリの親を指定する
#   2. bash scripts/libero/residual_worldmodel/collect_results.sh
#
# 出力:
#   OUTPUT_CSV に全 run の metrics + config を 1 行/run で集約
#
# 集計対象:
#   - train_metrics.jsonl の最終ステップ値
#   - checkpoint-best_*/meta.json の best metrics
#   - config_dump.json の run 設定
#   - DONE / FAILED ファイルによるステータス

set -euo pipefail

######################################################################
########## USER CONFIG — ここだけ書き換えて使う ##########
######################################################################

# ---- 対象ディレクトリ -------------------------------------------
# sweep / ablation 結果の親ディレクトリを指定
# (サブディレクトリを再帰的に探索する)
SEARCH_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}/checkpoints/libero/FocusedWM"

# ---- 評価結果も集める場合 ---------------------------------------
# eval_reports/libero/ 配下の metrics.json も読む
EVALS_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}/evals/libero/FocusedWM"
INCLUDE_EVAL_METRICS=true

# ---- 収集する訓練 metrics (train_metrics.jsonl の最終エントリより) --
TRAIN_METRICS_TO_COLLECT=(
  "loss"
  "recon"
  "dino_feature"
  "focus_supervision"
  "focus_sparsity"
)

# ---- 収集する eval metrics (eval_reports/metrics.json より) ------
EVAL_METRICS_TO_COLLECT=(
  "future_image_smooth_l1"
  "future_image_l1"
  "dino_feature_mse"
  "dino_cosine_similarity"
  "focus_mean"
  "focus_entropy"
  "iou_vs_change"
  "dice_vs_change"
)

# ---- 収集する ranking metrics (checkpoint-best_rank/meta.json より) -
RANK_METRICS_TO_COLLECT=(
  "pairwise_acc"
  "top1_acc"
  "mean_margin"
  "pos_score_mean"
  "neg_score_mean"
)

# ---- 出力 -------------------------------------------------------
OUTPUT_CSV="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}/results/libero_focused_wm_summary_$(date +%Y%m%d_%H%M%S).csv"

######################################################################
########## END USER CONFIG ##########
######################################################################

# ---------------------------------------------------------------------------
# Common setup
# ---------------------------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"
setup_env

ensure_dir "$(dirname "${OUTPUT_CSV}")"

# ---------------------------------------------------------------------------
# Python script to do the actual collection
# ---------------------------------------------------------------------------
print_header "Collecting results from ${SEARCH_ROOT}"

python - <<PYEOF
import json
import csv
import os
import sys
from pathlib import Path
from typing import Optional

SEARCH_ROOT      = "${SEARCH_ROOT}"
EVALS_ROOT       = "${EVALS_ROOT}"
INCLUDE_EVAL     = "${INCLUDE_EVAL_METRICS}" == "true"
OUTPUT_CSV       = "${OUTPUT_CSV}"

TRAIN_METRICS    = "${TRAIN_METRICS_TO_COLLECT[*]}".split()
EVAL_METRICS     = "${EVAL_METRICS_TO_COLLECT[*]}".split()
RANK_METRICS     = "${RANK_METRICS_TO_COLLECT[*]}".split()

# Config keys to extract from config_dump.json
CONFIG_KEYS = [
    "exp_name", "task", "model_variant", "dino_backbone",
    "dino_frozen", "lr", "batch_size", "grad_accum",
    "global_batch_size", "action_horizon", "max_steps",
    "precision", "recon_loss_weight", "dino_feature_loss_weight",
    "focus_supervision_type", "focus_supervision_weight",
    "focus_sparsity_weight", "ranking_score_type", "negative_mode",
    "seed", "run_name",
    # sweep / ablation extras
    "sweep_name", "ablation_group", "ablation_label",
    "condition_idx", "focus_on", "dino_feature_loss",
    "focus_target",
]


def read_json(path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def read_jsonl_last(path) -> dict:
    """Return last non-empty line parsed as JSON."""
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


def find_run_dirs(root: str) -> list:
    """Find all leaf-level run directories containing config_dump.json."""
    runs = []
    for p in Path(root).rglob("config_dump.json"):
        runs.append(p.parent)
    return sorted(runs)


def collect_run(run_dir: Path) -> dict:
    row = {"run_dir": str(run_dir)}

    # Status
    if (run_dir / "DONE").exists():
        row["status"] = "done"
    elif (run_dir / "FAILED").exists():
        row["status"] = "failed"
    else:
        row["status"] = "unknown"

    # Config
    cfg = read_json(run_dir / "config_dump.json")
    for k in CONFIG_KEYS:
        row[f"cfg_{k}"] = cfg.get(k, "")

    # Train metrics (last step)
    last_step = read_jsonl_last(run_dir / "train_metrics.jsonl")
    row["train_step"] = last_step.get("step", "")
    for m in TRAIN_METRICS:
        row[f"train_{m}"] = last_step.get(m, "")

    # Best checkpoint metrics
    for ckpt_type in ("best_recon", "best_dino_feature", "best_rank"):
        ckpt_dirs = sorted(run_dir.glob(f"checkpoint-{ckpt_type}-*"))
        if ckpt_dirs:
            meta = read_json(ckpt_dirs[-1] / "meta.json")
            row[f"best_{ckpt_type}_step"] = meta.get("step", "")
            row[f"best_{ckpt_type}_recon"]       = meta.get("recon", "")
            row[f"best_{ckpt_type}_dino_feature"] = meta.get("dino_feature", "")
            row[f"best_{ckpt_type}_pairwise_acc"] = meta.get("pairwise_acc", "")
        else:
            for k in ("step", "recon", "dino_feature", "pairwise_acc"):
                row[f"best_{ckpt_type}_{k}"] = ""

    # Eval metrics (from evals_root if available)
    if INCLUDE_EVAL:
        task = cfg.get("task", "")
        run_name = cfg.get("run_name", run_dir.name)
        eval_candidates = list(
            Path(EVALS_ROOT).rglob(f"*{run_name}*/metrics.json")
        ) if Path(EVALS_ROOT).exists() else []
        if eval_candidates:
            eval_met = read_json(eval_candidates[-1])
            for m in EVAL_METRICS:
                row[f"eval_{m}"] = eval_met.get(m, "")
        else:
            for m in EVAL_METRICS:
                row[f"eval_{m}"] = ""

    return row


# ---- Main ----
run_dirs = find_run_dirs(SEARCH_ROOT)
print(f"  Found {len(run_dirs)} run directories in {SEARCH_ROOT}")

if not run_dirs:
    print("  No runs found. Check SEARCH_ROOT.", file=sys.stderr)
    sys.exit(0)

rows = [collect_run(d) for d in run_dirs]

# Determine all columns
all_keys = []
seen = set()
for r in rows:
    for k in r.keys():
        if k not in seen:
            all_keys.append(k)
            seen.add(k)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in all_keys})

print(f"  Written: {OUTPUT_CSV}")
print(f"  Rows   : {len(rows)}")
PYEOF

echo ""
echo "  Summary CSV: ${OUTPUT_CSV}"
echo ""
echo "  Quick sort by recon (best first):"
echo "    python -c \""
echo "      import pandas as pd"
echo "      df = pd.read_csv('${OUTPUT_CSV}')"
echo "      print(df[['cfg_run_name','train_recon','best_best_recon_recon','best_best_rank_pairwise_acc']].sort_values('train_recon').to_string())"
echo "    \""
