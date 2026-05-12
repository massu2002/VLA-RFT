#!/usr/bin/env bash
# aggregate_v4_multiseed_eval.sh — 複数シードの評価結果を集計して mean±std を計算する
#
# Required env:
#   BASE_OUT_ROOT  — per-seed 結果の親ディレクトリ（例: results/phase1/v4_improved_spatial）
#   EVAL_SEEDS     — スペース区切りのシード値（例: "42 43 44"）
#
# Optional:
#   EXP_FILTER     — 集計対象の実験名（カンマ区切り、空=自動検出）
#   SWEEP_CONFIG   — v4_core_sweep.json へのパス

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"

source "${WM_SCRIPTS}/common.sh"

BASE_OUT_ROOT="${BASE_OUT_ROOT:?'BASE_OUT_ROOT required'}"
EVAL_SEEDS="${EVAL_SEEDS:?'EVAL_SEEDS required'}"
EXP_FILTER="${EXP_FILTER:-}"
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/libero/phase1/v4_core_sweep.json}"

SUMMARY_DIR="${BASE_OUT_ROOT}/multiseed_summary"
mkdir -p "${SUMMARY_DIR}"

log() { echo "[multiseed-agg] $(date +%H:%M:%S) $*"; }
log "BASE_OUT_ROOT = ${BASE_OUT_ROOT}"
log "EVAL_SEEDS    = ${EVAL_SEEDS}"
log "SUMMARY_DIR   = ${SUMMARY_DIR}"

setup_env
export TF_CPP_MIN_LOG_LEVEL=3
cd "${REPO_ROOT}"

export BASE_OUT_ROOT EVAL_SEEDS EXP_FILTER SWEEP_CONFIG SUMMARY_DIR

python3 - <<'PYEOF'
import csv, json, math, os, sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

BASE_OUT_ROOT = Path(os.environ["BASE_OUT_ROOT"])
EVAL_SEEDS    = os.environ["EVAL_SEEDS"].split()
EXP_FILTER    = [e.strip() for e in os.environ.get("EXP_FILTER","").split(",") if e.strip()]
SWEEP_CONFIG  = Path(os.environ.get("SWEEP_CONFIG",""))
SUMMARY_DIR   = Path(os.environ["SUMMARY_DIR"])

# ── 指標リスト ────────────────────────────────────────────────────────────────
METRIC_KEYS = [
    # Reconstruction quality (horizon-averaged, RFT-aligned)
    "horizon_avg_lpips",
    "horizon_avg_mae",
    "horizon_avg_mse",
    "rft_reward_proxy",
    "copy_current_horizon_avg_mse",
    "horizon_mse_over_copy",
    # Ranking accuracy
    "pairwise_acc_rft",
    "pairwise_acc_score",
    # Ranking gaps
    "rft_reward_gap_mean",
    "rft_reward_gap_min",
    "score_gap_mean",
    "score_gap_min",
    # Model internal structure
    "fuser_mask_entropy",
    "fuser_mask_overlap",
    "dynamic_mask_entropy",
    "dynamic_mask_overlap",
    "future_dynamic_query_norm",
    # Operational
    "skipped_history_windows",
]

# ── sweep config から実験メタを読む ───────────────────────────────────────────
exp_meta = {}
if SWEEP_CONFIG.exists():
    try:
        raw = json.loads(SWEEP_CONFIG.read_text())
        common = raw.get("common", {})
        for ex in raw.get("experiments", []):
            name = ex.get("exp_name","")
            exp_meta[name] = {**common, **ex}
    except Exception as e:
        print(f"[warn] sweep config: {e}")

# ── 各シード × 条件の metrics を収集 ─────────────────────────────────────────
# structure:
#   data[cond][seed] = {metric_key: float, ...}
#   phase_data[cond][phase][seed] = {metric_key: str, ...}
#   task_data[cond][task][seed] = {metric_key: str, ...}
#   task_phase_data[cond][task][phase][seed] = {metric_key: str, ...}
data: dict[str, dict[str, dict]] = OrderedDict()
phase_data: dict[str, dict[str, dict[str, dict]]] = OrderedDict()
task_data: dict[str, dict[str, dict[str, dict]]] = OrderedDict()
task_phase_data: dict[str, dict[str, dict[str, dict[str, dict]]]] = OrderedDict()

for seed in EVAL_SEEDS:
    seed_dir = BASE_OUT_ROOT / f"seed{seed}"
    if not seed_dir.exists():
        print(f"[warn] seed dir not found: {seed_dir}")
        continue
    for cond_dir in sorted(seed_dir.iterdir()):
        if not cond_dir.is_dir() or cond_dir.name in ("summary","logs"):
            continue
        agg_path = cond_dir / "aggregate_metrics.json"
        if not agg_path.exists():
            continue
        cond = cond_dir.name
        if EXP_FILTER and cond not in EXP_FILTER and cond != "baseline_phase0_ar_pixel":
            continue
        try:
            raw = json.loads(agg_path.read_text())
            m = raw.get("metrics", raw)
        except Exception as e:
            print(f"[warn] {agg_path}: {e}")
            continue
        # horizon_mse_over_copy is already stored in aggregate_metrics.json by the eval script.
        if cond not in data:
            data[cond] = {}
        data[cond][seed] = m

        phase_path = cond_dir / "metrics_by_phase.csv"
        if phase_path.exists():
            try:
                with phase_path.open(newline="") as f:
                    for row in csv.DictReader(f):
                        phase = row.get("window_phase", "")
                        if not phase:
                            continue
                        phase_data.setdefault(cond, OrderedDict()).setdefault(phase, {})[seed] = row
            except Exception as e:
                print(f"[warn] {phase_path}: {e}")

        task_path = cond_dir / "metrics_by_task.csv"
        if task_path.exists():
            try:
                with task_path.open(newline="") as f:
                    for row in csv.DictReader(f):
                        task = row.get("task_name", "")
                        if not task:
                            continue
                        task_data.setdefault(cond, OrderedDict()).setdefault(task, {})[seed] = row
            except Exception as e:
                print(f"[warn] {task_path}: {e}")

        task_phase_path = cond_dir / "metrics_by_task_by_phase.csv"
        if task_phase_path.exists():
            try:
                with task_phase_path.open(newline="") as f:
                    for row in csv.DictReader(f):
                        task = row.get("task_name", "")
                        phase = row.get("window_phase", "")
                        if not task or not phase:
                            continue
                        task_phase_data.setdefault(cond, OrderedDict()).setdefault(task, OrderedDict()).setdefault(phase, {})[seed] = row
            except Exception as e:
                print(f"[warn] {task_phase_path}: {e}")

if not data:
    print("[ERROR] 集計対象の aggregate_metrics.json が見つかりません。")
    print(f"  BASE_OUT_ROOT = {BASE_OUT_ROOT}")
    print(f"  EVAL_SEEDS    = {EVAL_SEEDS}")
    sys.exit(1)

print(f"Found {len(data)} conditions across {len(EVAL_SEEDS)} seeds.")

# ── mean ± std 計算 ───────────────────────────────────────────────────────────
def _fv(v):
    try:
        if v is None or v == "":
            return None
        out = float(v)
        return None if math.isnan(out) else out
    except Exception:
        return None

def _stats(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if vals:
        return {"mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=0)),
                "n":    len(vals),
                "values": vals}
    return {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}

def _fmt(mean, std, nd=4):
    if mean is None or (isinstance(mean,float) and math.isnan(mean)):
        return "N/A"
    if std is None or (isinstance(std,float) and math.isnan(std)):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"

NEG_PREFIXES = [
    "pairwise_acc_rft",
    "pairwise_acc_score",
    "rft_reward_gap_mean",
    "rft_reward_gap_min",
    "score_gap_mean",
    "score_gap_min",
]
NEG_TYPE_ORDER = ["same_phase", "temporal_shift", "action_noise", "mixed"]
PHASE_ORDER = ["early", "middle", "later"]

def _negative_type_from_key(key):
    for prefix in NEG_PREFIXES:
        stem = f"{prefix}_"
        if key.startswith(stem):
            return key[len(stem):]
    return None

def _ordered(items, preferred):
    seen = set()
    out = []
    for x in preferred:
        if x in items and x not in seen:
            out.append(x)
            seen.add(x)
    for x in sorted(items):
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

# aggregate_metrics.json の負例タイプ別キーも総合集計に含める。
neg_types_found = set()
dynamic_metric_keys = set()
for seed_dict in data.values():
    for metrics in seed_dict.values():
        for key in metrics.keys():
            neg_type = _negative_type_from_key(key)
            if neg_type:
                neg_types_found.add(neg_type)
                dynamic_metric_keys.add(key)

NEGATIVE_TYPES = _ordered(neg_types_found, NEG_TYPE_ORDER)
for key in _ordered(dynamic_metric_keys, []):
    if key not in METRIC_KEYS:
        METRIC_KEYS.append(key)

agg_stats = OrderedDict()
for cond, seed_dict in data.items():
    stats = {}
    for k in METRIC_KEYS:
        vals = [_fv(seed_dict[s].get(k)) for s in EVAL_SEEDS if s in seed_dict]
        stats[k] = _stats(vals)
    agg_stats[cond] = stats

phase_metric_keys = set()
for phase_dict in phase_data.values():
    for seed_rows in phase_dict.values():
        for row in seed_rows.values():
            phase_metric_keys.update(k for k in row.keys() if k != "window_phase")

PHASE_METRIC_KEYS = _ordered(phase_metric_keys, [
    "horizon_avg_lpips",
    "horizon_avg_mae",
    "horizon_avg_mse",
    "rft_reward_proxy",
    "copy_current_horizon_avg_mse",
    "horizon_mse_over_copy",
    "pairwise_acc_rft",
    "pairwise_acc_score",
    "score_gap",
    "rft_reward_gap",
    "fuser_mask_entropy",
    "fuser_mask_overlap",
    "dynamic_mask_entropy",
    "dynamic_mask_overlap",
    "future_dynamic_query_norm",
    "num_windows",
])

phase_stats = OrderedDict()
for cond, phase_dict in phase_data.items():
    phase_stats[cond] = OrderedDict()
    for phase in _ordered(phase_dict.keys(), PHASE_ORDER):
        seed_rows = phase_dict[phase]
        metric_stats = {}
        for key in PHASE_METRIC_KEYS:
            vals = [_fv(seed_rows[s].get(key)) for s in EVAL_SEEDS if s in seed_rows]
            metric_stats[key] = _stats(vals)
        phase_stats[cond][phase] = metric_stats

NEG_METRIC_BASES = [
    "pairwise_acc_rft",
    "pairwise_acc_score",
    "rft_reward_gap_mean",
    "rft_reward_gap_min",
    "score_gap_mean",
    "score_gap_min",
]

negative_stats = OrderedDict()
for cond, stats in agg_stats.items():
    negative_stats[cond] = OrderedDict()
    for neg_type in NEGATIVE_TYPES:
        per_neg = {}
        for base in NEG_METRIC_BASES:
            key = f"{base}_{neg_type}"
            if key in stats:
                per_neg[base] = stats[key]
        if per_neg:
            negative_stats[cond][neg_type] = per_neg

task_metric_keys = set()
for task_dict in task_data.values():
    for seed_rows in task_dict.values():
        for row in seed_rows.values():
            task_metric_keys.update(k for k in row.keys() if k != "task_name")
TASK_METRIC_KEYS = _ordered(task_metric_keys, PHASE_METRIC_KEYS + ["num_phases"])

task_stats = OrderedDict()
for cond, task_dict in task_data.items():
    task_stats[cond] = OrderedDict()
    for task, seed_rows in task_dict.items():
        metric_stats = {}
        for key in TASK_METRIC_KEYS:
            vals = [_fv(seed_rows[s].get(key)) for s in EVAL_SEEDS if s in seed_rows]
            metric_stats[key] = _stats(vals)
        task_stats[cond][task] = metric_stats

task_phase_metric_keys = set()
for task_dict in task_phase_data.values():
    for phase_dict in task_dict.values():
        for seed_rows in phase_dict.values():
            for row in seed_rows.values():
                task_phase_metric_keys.update(k for k in row.keys() if k not in ("task_name", "window_phase"))
TASK_PHASE_METRIC_KEYS = _ordered(task_phase_metric_keys, PHASE_METRIC_KEYS)

task_phase_stats = OrderedDict()
for cond, task_dict in task_phase_data.items():
    task_phase_stats[cond] = OrderedDict()
    for task, phase_dict in task_dict.items():
        task_phase_stats[cond][task] = OrderedDict()
        for phase in _ordered(phase_dict.keys(), PHASE_ORDER):
            seed_rows = phase_dict[phase]
            metric_stats = {}
            for key in TASK_PHASE_METRIC_KEYS:
                vals = [_fv(seed_rows[s].get(key)) for s in EVAL_SEEDS if s in seed_rows]
                metric_stats[key] = _stats(vals)
            task_phase_stats[cond][task][phase] = metric_stats

# ── JSON 出力 ─────────────────────────────────────────────────────────────────
json_out = {}
for cond, stats in agg_stats.items():
    json_out[cond] = {k: {kk: vv for kk,vv in v.items() if kk != "values"}
                      for k, v in stats.items()}
    meta = exp_meta.get(cond, {})
    json_out[cond]["_meta"] = {
        "stage":       meta.get("stage","?"),
        "history_k":   meta.get("history_length","?"),
        "num_q":       meta.get("num_dynamic_queries","?"),
        "motion_bias": meta.get("use_motion_bias","?"),
        "scorer":      meta.get("use_action_future_scorer","?"),
        "lambda_rank": meta.get("lambda_rank","?"),
        "neg_type":    meta.get("negative_type","?"),
        "seeds":       [s for s in EVAL_SEEDS if s in data[cond]],
    }
    if cond in phase_stats:
        json_out[cond]["_phase_metrics"] = {
            phase: {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                    for k, v in metric_stats.items()}
            for phase, metric_stats in phase_stats[cond].items()
        }
    if cond in negative_stats:
        json_out[cond]["_negative_metrics"] = {
            neg_type: {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                       for k, v in metric_stats.items()}
            for neg_type, metric_stats in negative_stats[cond].items()
        }
    if cond in task_stats:
        json_out[cond]["_task_metrics"] = {
            task: {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                   for k, v in metric_stats.items()}
            for task, metric_stats in task_stats[cond].items()
        }
    if cond in task_phase_stats:
        json_out[cond]["_task_phase_metrics"] = {
            task: {
                phase: {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                        for k, v in metric_stats.items()}
                for phase, metric_stats in phase_dict.items()
            }
            for task, phase_dict in task_phase_stats[cond].items()
        }
(SUMMARY_DIR / "multiseed_summary.json").write_text(
    json.dumps(json_out, indent=2, allow_nan=True))

# ── CSV 出力 ──────────────────────────────────────────────────────────────────
csv_cols = ["exp_name","stage","K","Q","motion","scorer","lambda_rank","neg_type","n_seeds"]
for k in METRIC_KEYS:
    csv_cols += [f"{k}_mean", f"{k}_std"]

csv_rows = []
for cond, stats in agg_stats.items():
    meta = exp_meta.get(cond, {})
    n_seeds = len([s for s in EVAL_SEEDS if s in data[cond]])
    row = {
        "exp_name":   cond,
        "stage":      meta.get("stage","?"),
        "K":          str(meta.get("history_length","?")),
        "Q":          str(meta.get("num_dynamic_queries","?")),
        "motion":     str(meta.get("use_motion_bias","?")),
        "scorer":     str(meta.get("use_action_future_scorer","?")),
        "lambda_rank":str(meta.get("lambda_rank","?")),
        "neg_type":   meta.get("negative_type","?"),
        "n_seeds":    str(n_seeds),
    }
    for k in METRIC_KEYS:
        st = stats[k]
        row[f"{k}_mean"] = f"{st['mean']:.6f}" if not math.isnan(st['mean']) else "N/A"
        row[f"{k}_std"]  = f"{st['std']:.6f}"  if not math.isnan(st['std'])  else "N/A"
    csv_rows.append(row)

with (SUMMARY_DIR / "multiseed_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_cols)
    w.writeheader()
    w.writerows(csv_rows)

phase_csv_cols = ["exp_name","stage","K","Q","motion","scorer","lambda_rank","neg_type","window_phase","n_seeds"]
for k in PHASE_METRIC_KEYS:
    phase_csv_cols += [f"{k}_mean", f"{k}_std"]

phase_csv_rows = []
for cond, phase_dict in phase_stats.items():
    meta = exp_meta.get(cond, {})
    for phase, metric_stats in phase_dict.items():
        n_seeds = max((st["n"] for st in metric_stats.values()), default=0)
        row = {
            "exp_name":   cond,
            "stage":      meta.get("stage","?"),
            "K":          str(meta.get("history_length","?")),
            "Q":          str(meta.get("num_dynamic_queries","?")),
            "motion":     str(meta.get("use_motion_bias","?")),
            "scorer":     str(meta.get("use_action_future_scorer","?")),
            "lambda_rank":str(meta.get("lambda_rank","?")),
            "neg_type":   meta.get("negative_type","?"),
            "window_phase": phase,
            "n_seeds":    str(n_seeds),
        }
        for k in PHASE_METRIC_KEYS:
            st = metric_stats[k]
            row[f"{k}_mean"] = f"{st['mean']:.6f}" if not math.isnan(st["mean"]) else "N/A"
            row[f"{k}_std"]  = f"{st['std']:.6f}"  if not math.isnan(st["std"])  else "N/A"
        phase_csv_rows.append(row)

with (SUMMARY_DIR / "multiseed_phase_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=phase_csv_cols)
    w.writeheader()
    w.writerows(phase_csv_rows)

negative_csv_cols = ["exp_name","stage","K","Q","motion","scorer","lambda_rank","config_neg_type","negative_type","n_seeds"]
for k in NEG_METRIC_BASES:
    negative_csv_cols += [f"{k}_mean", f"{k}_std"]

negative_csv_rows = []
for cond, neg_dict in negative_stats.items():
    meta = exp_meta.get(cond, {})
    for neg_type, metric_stats in neg_dict.items():
        n_seeds = max((st["n"] for st in metric_stats.values()), default=0)
        row = {
            "exp_name":   cond,
            "stage":      meta.get("stage","?"),
            "K":          str(meta.get("history_length","?")),
            "Q":          str(meta.get("num_dynamic_queries","?")),
            "motion":     str(meta.get("use_motion_bias","?")),
            "scorer":     str(meta.get("use_action_future_scorer","?")),
            "lambda_rank":str(meta.get("lambda_rank","?")),
            "config_neg_type": meta.get("negative_type","?"),
            "negative_type": neg_type,
            "n_seeds": str(n_seeds),
        }
        for k in NEG_METRIC_BASES:
            st = metric_stats.get(k, {"mean": float("nan"), "std": float("nan")})
            row[f"{k}_mean"] = f"{st['mean']:.6f}" if not math.isnan(st["mean"]) else "N/A"
            row[f"{k}_std"]  = f"{st['std']:.6f}"  if not math.isnan(st["std"])  else "N/A"
        negative_csv_rows.append(row)

with (SUMMARY_DIR / "multiseed_negative_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=negative_csv_cols)
    w.writeheader()
    w.writerows(negative_csv_rows)

task_csv_cols = ["exp_name","stage","K","Q","motion","scorer","lambda_rank","neg_type","task_name","n_seeds"]
for k in TASK_METRIC_KEYS:
    task_csv_cols += [f"{k}_mean", f"{k}_std"]

task_csv_rows = []
for cond, task_dict in task_stats.items():
    meta = exp_meta.get(cond, {})
    for task, metric_stats in task_dict.items():
        n_seeds = max((st["n"] for st in metric_stats.values()), default=0)
        row = {
            "exp_name":   cond,
            "stage":      meta.get("stage","?"),
            "K":          str(meta.get("history_length","?")),
            "Q":          str(meta.get("num_dynamic_queries","?")),
            "motion":     str(meta.get("use_motion_bias","?")),
            "scorer":     str(meta.get("use_action_future_scorer","?")),
            "lambda_rank":str(meta.get("lambda_rank","?")),
            "neg_type":   meta.get("negative_type","?"),
            "task_name":  task,
            "n_seeds":    str(n_seeds),
        }
        for k in TASK_METRIC_KEYS:
            st = metric_stats[k]
            row[f"{k}_mean"] = f"{st['mean']:.6f}" if not math.isnan(st["mean"]) else "N/A"
            row[f"{k}_std"]  = f"{st['std']:.6f}"  if not math.isnan(st["std"])  else "N/A"
        task_csv_rows.append(row)

with (SUMMARY_DIR / "multiseed_task_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=task_csv_cols)
    w.writeheader()
    w.writerows(task_csv_rows)

task_phase_csv_cols = ["exp_name","stage","K","Q","motion","scorer","lambda_rank","neg_type","task_name","window_phase","n_seeds"]
for k in TASK_PHASE_METRIC_KEYS:
    task_phase_csv_cols += [f"{k}_mean", f"{k}_std"]

task_phase_csv_rows = []
for cond, task_dict in task_phase_stats.items():
    meta = exp_meta.get(cond, {})
    for task, phase_dict in task_dict.items():
        for phase, metric_stats in phase_dict.items():
            n_seeds = max((st["n"] for st in metric_stats.values()), default=0)
            row = {
                "exp_name":   cond,
                "stage":      meta.get("stage","?"),
                "K":          str(meta.get("history_length","?")),
                "Q":          str(meta.get("num_dynamic_queries","?")),
                "motion":     str(meta.get("use_motion_bias","?")),
                "scorer":     str(meta.get("use_action_future_scorer","?")),
                "lambda_rank":str(meta.get("lambda_rank","?")),
                "neg_type":   meta.get("negative_type","?"),
                "task_name":  task,
                "window_phase": phase,
                "n_seeds":    str(n_seeds),
            }
            for k in TASK_PHASE_METRIC_KEYS:
                st = metric_stats[k]
                row[f"{k}_mean"] = f"{st['mean']:.6f}" if not math.isnan(st["mean"]) else "N/A"
                row[f"{k}_std"]  = f"{st['std']:.6f}"  if not math.isnan(st["std"])  else "N/A"
            task_phase_csv_rows.append(row)

with (SUMMARY_DIR / "multiseed_task_phase_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=task_phase_csv_cols)
    w.writeheader()
    w.writerows(task_phase_csv_rows)

# ── Markdown レポート ─────────────────────────────────────────────────────────
seeds_str = ", ".join(EVAL_SEEDS)
cond_names = list(agg_stats.keys())

lines = []
lines.append(f"# Phase 1 v4 Sweep - Multi-Seed 評価結果 (mean ± std)")
lines.append(f"")
lines.append(f"**Seeds**: {seeds_str}  |  **Conditions**: {len(cond_names)}")
lines.append(f"**Base**: `{BASE_OUT_ROOT}`")
lines.append(f"")

# --- 重要指標テーブル ---
KEY_METRICS = [
    # Reconstruction quality
    ("horizon_avg_lpips",        "horizon_lpips",    "lower"),
    ("horizon_avg_mae",          "horizon_mae",      "lower"),
    ("horizon_avg_mse",          "horizon_mse",      "lower"),
    ("rft_reward_proxy",         "rft_proxy",        "higher"),
    ("horizon_mse_over_copy",    "mse/copy",         "lower"),
    # Ranking accuracy
    ("pairwise_acc_rft",         "pairwise_acc_rft", "higher"),
    ("pairwise_acc_score",       "pairwise_acc_scr", "higher"),
    # Ranking gaps
    ("rft_reward_gap_mean",      "rft_gap_mean",     "higher"),
    ("score_gap_mean",           "score_gap_mean",   "higher"),
    # Model internals
    ("fuser_mask_entropy",       "fuser_entropy",    "neutral"),
    ("fuser_mask_overlap",       "fuser_overlap",    "neutral"),
    ("dynamic_mask_entropy",     "dyn_entropy",      "neutral"),
    ("future_dynamic_query_norm","dq_norm",          "neutral"),
]

def best_mean(key, better, conds, stats_dict):
    vals = [(c, stats_dict[c][key]["mean"]) for c in conds
            if not math.isnan(stats_dict[c][key]["mean"])]
    if not vals: return None
    return min(vals, key=lambda x: x[1])[0] if better=="lower" else max(vals, key=lambda x: x[1])[0]

lines.append("## 主要指標 (mean ± std)")
lines.append("")
lines.append("各条件の `aggregate_metrics.json` をシード方向に集計した総合表です。太字は、その指標で最も良い平均値を示します。再構成誤差系は低いほど良く、ranking accuracy と gap 系は高いほど良い値です。")
lines.append("")
header = "| Metric |" + "".join(f" {c} |" for c in cond_names)
sep    = "|---|" + "---|" * len(cond_names)
lines.append(header)
lines.append(sep)

for key, label, better in KEY_METRICS:
    best_cond = best_mean(key, better, cond_names, agg_stats)
    cells = []
    for c in cond_names:
        st = agg_stats[c][key]
        txt = _fmt(st["mean"], st["std"])
        if c == best_cond and txt != "N/A":
            txt = f"**{txt}**"
        cells.append(txt)
    lines.append(f"| `{label}` |" + "".join(f" {v} |" for v in cells))

lines.append("")

# --- negative type breakdown ---
if NEGATIVE_TYPES:
    lines.append("## 負例タイプ別の評価")
    lines.append("")
    lines.append("同じ評価windowに対して、負例の作り方ごとに ranking 指標を分けて集計した表です。`same_phase` は同じ episode phase 内の別行動、`temporal_shift` は時刻をずらした行動、`action_noise` は正例action列へノイズを加えた行動、`mixed` は複数タイプを混ぜた負例を表します。")
    lines.append("")
    lines.append("| condition | negative_type | pairwise_acc_rft | pairwise_acc_score | rft_gap_mean | score_gap_mean |")
    lines.append("|---|---|---|---|---|---|")
    for c in cond_names:
        for neg_type in NEGATIVE_TYPES:
            metric_stats = negative_stats.get(c, {}).get(neg_type, {})
            if not metric_stats:
                continue
            rft_acc = metric_stats.get("pairwise_acc_rft", {"mean": float("nan"), "std": float("nan")})
            score_acc = metric_stats.get("pairwise_acc_score", {"mean": float("nan"), "std": float("nan")})
            rft_gap = metric_stats.get("rft_reward_gap_mean", {"mean": float("nan"), "std": float("nan")})
            score_gap = metric_stats.get("score_gap_mean", {"mean": float("nan"), "std": float("nan")})
            lines.append(
                f"| `{c}` | `{neg_type}` | {_fmt(rft_acc['mean'], rft_acc['std'])} "
                f"| {_fmt(score_acc['mean'], score_acc['std'])} "
                f"| {_fmt(rft_gap['mean'], rft_gap['std'])} "
                f"| {_fmt(score_gap['mean'], score_gap['std'])} |"
            )
    lines.append("")
    lines.append("完全な負例タイプ別の数値は `multiseed_negative_summary.csv` に保存しています。")
    lines.append("")

# --- phase breakdown ---
if phase_stats:
    lines.append("## Episode Phase 別の評価")
    lines.append("")
    lines.append("評価windowを episode 内の `early` / `middle` / `later` に分け、各phase内でランダム取得したwindowの指標をシード方向に集計した表です。時間帯によって再構成やrankingの難しさが変わるかを見るための集計です。")
    lines.append("")
    lines.append("| condition | phase | horizon_mse | mse/copy | pairwise_acc_rft | pairwise_acc_score | rft_gap | score_gap |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in cond_names:
        for phase in _ordered(phase_stats.get(c, {}).keys(), PHASE_ORDER):
            metric_stats = phase_stats[c][phase]
            horizon_mse = metric_stats.get("horizon_avg_mse", {"mean": float("nan"), "std": float("nan")})
            mse_copy = metric_stats.get("horizon_mse_over_copy", {"mean": float("nan"), "std": float("nan")})
            rft_acc = metric_stats.get("pairwise_acc_rft", {"mean": float("nan"), "std": float("nan")})
            score_acc = metric_stats.get("pairwise_acc_score", {"mean": float("nan"), "std": float("nan")})
            rft_gap = metric_stats.get("rft_reward_gap", {"mean": float("nan"), "std": float("nan")})
            score_gap = metric_stats.get("score_gap", {"mean": float("nan"), "std": float("nan")})
            lines.append(
                f"| `{c}` | `{phase}` | {_fmt(horizon_mse['mean'], horizon_mse['std'])} "
                f"| {_fmt(mse_copy['mean'], mse_copy['std'])} "
                f"| {_fmt(rft_acc['mean'], rft_acc['std'])} "
                f"| {_fmt(score_acc['mean'], score_acc['std'])} "
                f"| {_fmt(rft_gap['mean'], rft_gap['std'])} "
                f"| {_fmt(score_gap['mean'], score_gap['std'])} |"
            )
    lines.append("")
    lines.append("完全なphase別の数値は `multiseed_phase_summary.csv` に保存しています。")
    lines.append("")

lines.append("## 詳細CSV")
lines.append("")
lines.append("Markdownには比較しやすい代表表だけを載せています。列数が多い詳細集計は、以下のCSVに保存しています。各CSVはシード方向に mean/std を計算しており、phase別・負例タイプ別の動的指標も列として保持します。")
lines.append("")
lines.append("| ファイル | 内容 |")
lines.append("|---|---|")
lines.append("| `multiseed_summary.csv` | conditionごとの総合集計です。`aggregate_metrics.json` 内の基本指標と負例タイプ別の動的指標を含みます。 |")
lines.append("| `multiseed_negative_summary.csv` | conditionとnegative typeごとのranking指標を集計した表です。 |")
lines.append("| `multiseed_phase_summary.csv` | conditionとepisode phaseごとの指標を集計した表です。 |")
lines.append("| `multiseed_task_summary.csv` | conditionとtaskごとの指標を集計した表です。 |")
lines.append("| `multiseed_task_phase_summary.csv` | condition、task、episode phaseごとの指標を集計した表です。 |")
lines.append("")

# --- per-seed values table for key metrics ---
lines.append("## シード別の値（主要指標）")
lines.append("")
lines.append("主要指標について、各シードの値とその mean ± std を確認するための表です。平均値だけでは見えないシード間のばらつきを確認できます。")
lines.append("")
for key, label, _ in KEY_METRICS[:8]:  # top 8 metrics
    lines.append(f"### {label}")
    lines.append("")
    h2 = "| condition |" + "".join(f" seed{s} |" for s in EVAL_SEEDS) + " mean ± std |"
    s2 = "|---|" + "---|" * len(EVAL_SEEDS) + "---|"
    lines.append(h2)
    lines.append(s2)
    for c in cond_names:
        per_seed = []
        for s in EVAL_SEEDS:
            if s in data[c]:
                v = _fv(data[c][s].get(key))
                per_seed.append(f"{v:.4f}" if v is not None and not math.isnan(v) else "N/A")
            else:
                per_seed.append("—")
        st = agg_stats[c][key]
        summary = _fmt(st["mean"], st["std"])
        lines.append(f"| `{c}` |" + "".join(f" {v} |" for v in per_seed) + f" {summary} |")
    lines.append("")

# --- condition metadata ---
lines.append("## 実験設定")
lines.append("")
lines.append("各conditionがどの設定で実行されたかを確認するための表です。`config_neg_type` に相当する値は sweep config 上の負例設定で、実際の負例タイプ別の評価は上の表とCSVに分けて保存しています。")
lines.append("")
lines.append("| exp_name | stage | K | Q | motion | scorer | λ_rank | neg_type |")
lines.append("|---|---|---|---|---|---|---|---|")
for c in cond_names:
    meta = exp_meta.get(c, {})
    lines.append(
        f"| `{c}` | {meta.get('stage','?')} | {meta.get('history_length','?')} "
        f"| {meta.get('num_dynamic_queries','?')} | {meta.get('use_motion_bias','?')} "
        f"| {meta.get('use_action_future_scorer','?')} | {meta.get('lambda_rank','?')} "
        f"| {meta.get('negative_type','?')} |"
    )
lines.append("")
lines.append("---")
lines.append(f"*Generated by `aggregate_v4_multiseed_eval.sh` — seeds: {seeds_str}*")

(SUMMARY_DIR / "multiseed_summary.md").write_text("\n".join(lines), encoding="utf-8")

# ── コンソール出力 ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Multi-seed 集計完了  conditions={len(cond_names)}  seeds={seeds_str}")
print(f"{'='*70}")
print(f"{'Metric':<35}", end="")
for c in cond_names:
    short = c[:20]
    print(f"  {short:<22}", end="")
print()
print("-" * (35 + 24 * len(cond_names)))

for key, label, better in KEY_METRICS:
    best_c = best_mean(key, better, cond_names, agg_stats)
    print(f"  {label:<33}", end="")
    for c in cond_names:
        st = agg_stats[c][key]
        txt = _fmt(st["mean"], st["std"], nd=4)
        marker = " *" if c == best_c and txt != "N/A" else "  "
        print(f"{marker}{txt:<22}", end="")
    print()

print(f"\n  * = best per metric")
print(f"\nFiles:")
print(f"  {SUMMARY_DIR}/multiseed_summary.md")
print(f"  {SUMMARY_DIR}/multiseed_summary.csv")
print(f"  {SUMMARY_DIR}/multiseed_phase_summary.csv")
print(f"  {SUMMARY_DIR}/multiseed_negative_summary.csv")
print(f"  {SUMMARY_DIR}/multiseed_task_summary.csv")
print(f"  {SUMMARY_DIR}/multiseed_task_phase_summary.csv")
print(f"  {SUMMARY_DIR}/multiseed_summary.json")
PYEOF

log "集計完了 → ${SUMMARY_DIR}/"
