#!/usr/bin/env bash
# summarize_v4_temporal_query_eval.sh — Aggregate v4 Phase 1 results into comparison.md
#
# Usage:
#   bash scripts/libero/residual_worldmodel/summarize_v4_temporal_query_eval.sh <phase1-output-dir>
#   bash scripts/libero/residual_worldmodel/summarize_v4_temporal_query_eval.sh  # uses results/phase1/latest.txt
#
# Reads per-condition outputs under <phase1-dir>/<condition>/:
#   aggregate_metrics.json (v4 metrics), metrics_by_task.csv, ranking_by_window.csv
#
# v4-specific metrics covered beyond v1/v3:
#   pairwise_acc_score, pairwise_acc_lpips, score_gap_mean, score_gap_min,
#   reverse_windows_score, fuser_mask_mean, dynamic_mask_mean,
#   future_dynamic_query_norm, skipped_history_windows
#
# Writes:
#   <phase1-dir>/v4_summary.json
#   <phase1-dir>/v4_comparison.md
#   <phase1-dir>/v4_comparison_table.csv

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PHASE1_SCRIPTS="${SCRIPT_DIR}/../phase1"
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

# ---------------------------------------------------------------------------
# Resolve output directory
# ---------------------------------------------------------------------------
if [ -n "${1:-}" ]; then
  PHASE1_DIR="$1"
elif [ -f "${REPO_ROOT}/results/phase1/latest.txt" ]; then
  PHASE1_DIR=$(cat "${REPO_ROOT}/results/phase1/latest.txt")
else
  echo "Usage: $0 <phase1-output-dir>" >&2
  exit 1
fi

[ -d "${PHASE1_DIR}" ] || { echo "Directory not found: ${PHASE1_DIR}" >&2; exit 1; }

log() { echo "[summarize-v4] $(date +%H:%M:%S) $*"; }
log "Summarizing v4 results: ${PHASE1_DIR}"

setup_env
export TF_CPP_MIN_LOG_LEVEL=3
cd "${REPO_ROOT}"

export PHASE1_DIR REPO_ROOT
python3 - << 'PYEOF'
import csv
import json
import math
import os
from pathlib import Path
from collections import OrderedDict

PHASE1 = Path(os.environ["PHASE1_DIR"])

def sf(v, nd=5):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)

def pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    try:
        return f"{float(v)*100:.1f}%"
    except (TypeError, ValueError):
        return str(v)

def _load_csv_by_task(d: Path):
    p = d / "metrics_by_task.csv"
    if not p.exists():
        return {}
    with open(p) as f:
        return {row.get("task_name", ""): row for row in csv.DictReader(f)}

# ---- Scan conditions (all subdirs with aggregate_metrics.json) ---------------
CONDITIONS: OrderedDict = OrderedDict()
for d in sorted(PHASE1.iterdir()):
    p = d / "aggregate_metrics.json"
    if not (d.is_dir() and p.exists()):
        continue
    try:
        raw = json.loads(p.read_text())
        metrics = raw.get("metrics", raw)
        CONDITIONS[d.name] = {
            "condition": d.name,
            "metrics":   metrics,
            "by_task":   _load_csv_by_task(d),
        }
    except Exception as e:
        print(f"[warn] could not read {p}: {e}")

if not CONDITIONS:
    print("[warn] No conditions found under", PHASE1)
    raise SystemExit(0)

# ---- v4 Metric definitions ---------------------------------------------------
# (base metrics shared with v1/v3 + v4-specific metrics)
SCALAR_METRICS = [
    # --- shared ---
    ("full_mse",              "Full MSE",                   "lower"),
    ("full_lpips",            "Full LPIPS",                 "lower"),
    ("gripper_mse",           "Gripper MSE",                "lower"),
    ("dynamic_mse",           "Dynamic MSE",                "lower"),
    ("static_consistency_mse","Static Consistency MSE",     "lower"),
    ("residual_abs_mean",     "|Residual| Mean",            "lower"),
    # --- LPIPS-based ranking (works for v4a and v4b) ---
    ("pairwise_acc_lpips",    "Pairwise Acc (LPIPS)",       "higher"),
    ("lpips_gap",             "LPIPS Gap (shuffled−GT)",    "higher"),
    ("correct_lpips",         "Correct LPIPS",              "lower"),
    # --- Score-based ranking (v4b: ActionFutureScorer) ---
    ("pairwise_acc_score",    "Pairwise Acc (Score)",       "higher"),
    ("score_gap_mean",        "Score Gap Mean",             "higher"),
    ("score_gap_min",         "Score Gap Min",              "higher"),
    ("reverse_windows_score", "Reverse Windows (Score)",    "lower"),
    # --- v4 debug ---
    ("fuser_mask_mean",          "Fuser Mask Mean",         "neutral"),
    ("dynamic_mask_mean",        "Dynamic Mask Mean",       "neutral"),
    ("future_dynamic_query_norm","Future DQ Norm",          "neutral"),
    ("skipped_history_windows",  "Skipped History Win",     "lower"),
]

def _best_val(key, better):
    if better == "neutral":
        return None
    vals = {}
    for name, c in CONDITIONS.items():
        v = c["metrics"].get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            try:
                vals[name] = float(v)
            except (TypeError, ValueError):
                pass
    if not vals:
        return None
    return min(vals.values()) if better == "lower" else max(vals.values())

def _is_best(val, key, better):
    if better == "neutral":
        return False
    best = _best_val(key, better)
    if best is None or val is None:
        return False
    try:
        return abs(float(val) - best) < 1e-9
    except (TypeError, ValueError):
        return False

# ---- v4_summary.json ---------------------------------------------------------
cond_names = list(CONDITIONS.keys())
summary = {
    "phase1_dir": str(PHASE1),
    "conditions": {
        name: {
            "metrics":     {k: c["metrics"].get(k) for k, *_ in SCALAR_METRICS},
            "num_windows": c["metrics"].get("num_windows"),
        }
        for name, c in CONDITIONS.items()
    },
}
(PHASE1 / "v4_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"Wrote: {PHASE1 / 'v4_summary.json'}")

# ---- v4_comparison_table.csv -------------------------------------------------
csv_rows = []
for key, label, better in SCALAR_METRICS:
    row = {"metric": label, "better": better}
    for name in cond_names:
        v = CONDITIONS[name]["metrics"].get(key)
        s = sf(v)
        row[name] = s
        row[f"{name}_best"] = "✓" if _is_best(v, key, better) else ""
    csv_rows.append(row)

fieldnames = (["metric", "better"] + cond_names + [f"{c}_best" for c in cond_names])
with open(PHASE1 / "v4_comparison_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(csv_rows)
print(f"Wrote: {PHASE1 / 'v4_comparison_table.csv'}")

# ---- v4_comparison.md --------------------------------------------------------
win_str = ", ".join(
    f"`{n}`={CONDITIONS[n]['metrics'].get('num_windows','?')}"
    for n in cond_names
)
hdr  = "| Metric | Better | " + " | ".join(cond_names) + " |"
div  = "| --- | --- | " + " | ".join(["---"] * len(cond_names)) + " |"
rows_md = []
for key, label, better in SCALAR_METRICS:
    cells = [f"`{label}`", f"`{better}`"]
    for name in cond_names:
        v = CONDITIONS[name]["metrics"].get(key)
        s = sf(v)
        if _is_best(v, key, better):
            s = f"**{s}**"
        cells.append(s)
    rows_md.append("| " + " | ".join(cells) + " |")
metric_table = "\n".join([hdr, div] + rows_md)

# Analysis
analysis = []
for key, label, better in [
    ("pairwise_acc_score", "Pairwise Acc (Score)", "higher"),
    ("pairwise_acc_lpips", "Pairwise Acc (LPIPS)", "higher"),
    ("score_gap_mean",     "Score Gap Mean",       "higher"),
    ("dynamic_mse",        "Dynamic MSE",          "lower"),
    ("full_mse",           "Full MSE",             "lower"),
]:
    if better == "neutral":
        continue
    best = _best_val(key, better)
    if best is None:
        continue
    best_conds = [n for n in cond_names if _is_best(CONDITIONS[n]["metrics"].get(key), key, better)]
    analysis.append(f"- **{label}**: best = `{sf(best)}` → `{'`, `'.join(best_conds)}`")

# pairwise_acc_score vs pairwise_acc_lpips correlation
for name in cond_names:
    pa_s = CONDITIONS[name]["metrics"].get("pairwise_acc_score")
    pa_l = CONDITIONS[name]["metrics"].get("pairwise_acc_lpips", CONDITIONS[name]["metrics"].get("pairwise_acc"))
    if pa_s is not None and pa_l is not None:
        try:
            delta = float(pa_s) - float(pa_l)
            analysis.append(f"  - `{name}`: score_acc={pct(pa_s)}  lpips_acc={pct(pa_l)}  Δ={delta:+.4f}")
        except (TypeError, ValueError):
            pass

analysis_str = "\n".join(analysis) if analysis else "_No v4 conditions found or metrics missing._"

# Per-task table
def _per_task_table():
    all_tasks = set()
    for c in CONDITIONS.values():
        all_tasks.update(c["by_task"].keys())
    all_tasks = sorted(all_tasks)
    if not all_tasks:
        return ""
    hdr_cells = ["Task"]
    for n in cond_names:
        hdr_cells += [f"{n[:14]}/pairwise_score", f"{n[:14]}/score_gap"]
    lines = ["| " + " | ".join(hdr_cells) + " |",
             "| " + " | ".join(["---"] * len(hdr_cells)) + " |"]
    for task in all_tasks:
        row = [task[:42]]
        for name, c in CONDITIONS.items():
            t = c["by_task"].get(task, {})
            row.append(sf(t.get("pairwise_acc_score", t.get("pairwise_acc"))))
            row.append(sf(t.get("score_gap_mean")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

per_task_str = _per_task_table()
per_task_section = (
    f"## 3. Per-Task Breakdown\n\n{per_task_str}\n\n---\n"
    if per_task_str else ""
)

# Go/No-Go
def _go_nogo():
    pa_best = _best_val("pairwise_acc_score", "higher")
    pa_lpips_best = _best_val("pairwise_acc_lpips", "higher")
    sg_best = _best_val("score_gap_mean", "higher")
    rows = [
        f"| Best `pairwise_acc_score` | {sf(pa_best)} | {'✓ proceed if > 0.60 (score-based ranking active)' if pa_best and float(pa_best) > 0.60 else '✗ below 0.60 — try v4b (scorer) or more training'} |",
        f"| Best `pairwise_acc_lpips` | {sf(pa_lpips_best)} | {'✓' if pa_lpips_best and float(pa_lpips_best) > 0.60 else '✗ below 0.60'} |",
        f"| Best `score_gap_mean`     | {sf(sg_best)} | {'✓ positive gap = consistent ranking signal' if sg_best is not None and float(sg_best) > 0 else '✗ gap ≤ 0; scorer may not have converged'} |",
    ]
    best_cond = [n for n in cond_names if _is_best(CONDITIONS[n]["metrics"].get("pairwise_acc_score"), "pairwise_acc_score", "higher")]
    proceed = pa_best is not None and float(pa_best) > 0.60
    rows.append(
        f"| Proceed to v4c RFT? | **{'YES' if proceed else 'NO/CONDITIONAL'}** | "
        f"{'Use `' + (best_cond[0] if best_cond else '?') + '` with WORLD_REWARD_TYPE=hybrid' if proceed else 'pairwise_acc_score ≤ 0.60; use WORLD_REWARD_TYPE=visual fallback or retrain'} |"
    )
    return "\n".join(["| Question | Value | Interpretation |",
                       "| --- | --- | --- |"] + rows)

md = f"""# Phase 1: v4 Temporal Dynamic Query Residual WM — Results

**Eval dir**: `{PHASE1}`
**Conditions**: {', '.join(f'`{n}`' for n in cond_names)}
**Eval windows**: {win_str}

---

## 1. Aggregate Metrics Comparison

> **Bold** values = best per row (neutral metrics not highlighted).

{metric_table}

---

## 2. Analysis

{analysis_str}

---

{per_task_section}## 4. Go / No-Go for v4c RFT

{_go_nogo()}

---

## 5. v4 Metric Reference

| Metric | Meaning |
| --- | --- |
| `full_mse` | Mean squared error across all pixels (↓) |
| `pairwise_acc_lpips` | Fraction: GT action → lower LPIPS than shuffled (↑ > 0.60 = proceed) |
| `pairwise_acc_score` | Fraction: ActionFutureScorer scores GT action higher than shuffled (↑, v4b only) |
| `score_gap_mean` | Mean (score_pos − score_neg) across windows (↑, positive = consistent) |
| `score_gap_min` | Minimum score gap — monitors worst-case ranking failure (↑) |
| `reverse_windows_score` | Windows where ranking is reversed (score_neg > score_pos) (↓) |
| `fuser_mask_mean` | Mean TokenFuser attention weight across all queries and tokens |
| `dynamic_mask_mean` | Mean DynamicQueryExtractor soft mask value |
| `future_dynamic_query_norm` | Mean L2 norm of predicted future dynamic queries |
| `skipped_history_windows` | Windows skipped due to insufficient history frames (↓) |

---

*Generated by `scripts/libero/residual_worldmodel/summarize_v4_temporal_query_eval.sh`*
"""

(PHASE1 / "v4_comparison.md").write_text(md)
print(f"Wrote: {PHASE1 / 'v4_comparison.md'}")

# Console output
print("\n=== v4 Phase 1 Summary ===")
for key, label, _ in [
    ("full_mse",             "Full MSE         "),
    ("pairwise_acc_lpips",   "PairwiseAcc LPIPS "),
    ("pairwise_acc_score",   "PairwiseAcc Score "),
    ("score_gap_mean",       "Score Gap Mean    "),
    ("fuser_mask_mean",      "Fuser Mask Mean   "),
]:
    vals = "  ".join(f"{n}={sf(c['metrics'].get(key))}" for n, c in CONDITIONS.items())
    print(f"  {label}: {vals}")
PYEOF

log "Summary complete."
echo ""
ls -la "${PHASE1_DIR}"/v4_*.md "${PHASE1_DIR}"/v4_*.json "${PHASE1_DIR}"/v4_*.csv 2>/dev/null || true
