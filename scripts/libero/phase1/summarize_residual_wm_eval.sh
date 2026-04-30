#!/usr/bin/env bash
# summarize_residual_wm_eval.sh — Aggregate Phase 1 results into comparison.md
#
# Usage:
#   bash scripts/libero/phase1/summarize_residual_wm_eval.sh <phase1-output-dir>
#   bash scripts/libero/phase1/summarize_residual_wm_eval.sh   # uses results/phase1/latest.txt
#
# Reads per-condition outputs under <phase1-dir>/<condition>/:
#   aggregate_metrics.json, metrics_by_task.csv, ranking_by_window.csv
#
# Writes:
#   <phase1-dir>/summary.json
#   <phase1-dir>/comparison.md
#   <phase1-dir>/comparison_table.csv

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WM_SCRIPTS="${SCRIPT_DIR}/../residual_worldmodel"
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

source "${WM_SCRIPTS}/common.sh"

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

log() { echo "[summarize] $(date +%H:%M:%S) $*"; }
log "Summarizing: ${PHASE1_DIR}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
setup_env
export TF_CPP_MIN_LOG_LEVEL=3
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Python: generate summary, comparison_table.csv, comparison.md
# ---------------------------------------------------------------------------
export PHASE1_DIR REPO_ROOT
python3 - << 'PYEOF'
import csv
import json
import math
import os
from pathlib import Path
from collections import OrderedDict

PHASE1 = Path(os.environ["PHASE1_DIR"])

# ---- Helpers ----------------------------------------------------------------
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

def _load_csv_by_window(d: Path):
    p = d / "ranking_by_window.csv"
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))

# ---- Scan conditions (all subdirs with aggregate_metrics.json) --------------
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
            "by_window": _load_csv_by_window(d),
        }
    except Exception as e:
        print(f"[warn] could not read {p}: {e}")

if not CONDITIONS:
    print("[warn] No conditions found under", PHASE1)
    print("       Expected subdirectories with aggregate_metrics.json.")
    raise SystemExit(0)

# ---- Metric definitions -----------------------------------------------------
SCALAR_METRICS = [
    ("full_mse",              "Full MSE",               "lower"),
    ("full_lpips",            "Full LPIPS",             "lower"),
    ("gripper_mse",           "Gripper MSE",            "lower"),
    ("gripper_lpips",         "Gripper LPIPS",          "lower"),
    ("goal_mse",              "Goal MSE",               "lower"),
    ("goal_lpips",            "Goal LPIPS",             "lower"),
    ("dynamic_mse",           "Dynamic MSE",            "lower"),
    ("dynamic_lpips",         "Dynamic LPIPS",          "lower"),
    ("static_consistency_mse","Static Consistency MSE", "lower"),
    ("pairwise_acc",          "Pairwise Acc",           "higher"),
    ("lpips_gap",             "LPIPS Gap (GT−shuffled)","lower"),
    ("correct_lpips",         "Correct LPIPS",          "lower"),
    ("shuffled_lpips",        "Shuffled LPIPS",         "lower"),
]

def _best_val(key, better):
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
    best = _best_val(key, better)
    if best is None or val is None:
        return False
    try:
        return abs(float(val) - best) < 1e-9
    except (TypeError, ValueError):
        return False

# ---- summary.json -----------------------------------------------------------
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
(PHASE1 / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"Wrote: {PHASE1 / 'summary.json'}")

# ---- comparison_table.csv ---------------------------------------------------
cond_names = list(CONDITIONS.keys())
csv_rows = []
for key, label, better in SCALAR_METRICS:
    row = {"metric": label, "better": better}
    for name in cond_names:
        v = CONDITIONS[name]["metrics"].get(key)
        s = sf(v)
        row[name] = s
        row[f"{name}_best"] = "✓" if _is_best(v, key, better) else ""
    csv_rows.append(row)

fieldnames = (["metric", "better"]
              + cond_names
              + [f"{c}_best" for c in cond_names])
with open(PHASE1 / "comparison_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(csv_rows)
print(f"Wrote: {PHASE1 / 'comparison_table.csv'}")

# ---- comparison.md ----------------------------------------------------------
_run_cfg = {}
if (PHASE1 / "run_config.json").exists():
    _run_cfg = json.loads((PHASE1 / "run_config.json").read_text())
suite = _run_cfg.get("suite", PHASE1.name.split("_")[-1] if "_" in PHASE1.name else "?")
win_str = ", ".join(
    f"`{n}`={CONDITIONS[n]['metrics'].get('num_windows','?')}"
    for n in cond_names
)

# Aggregate metrics table
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

# Analysis: best condition per key metric
analysis = []
for key, label, better in [
    ("pairwise_acc", "Pairwise Acc",  "higher"),
    ("gripper_mse",  "Gripper MSE",   "lower"),
    ("dynamic_mse",  "Dynamic MSE",   "lower"),
    ("lpips_gap",    "LPIPS Gap",     "lower"),
    ("full_mse",     "Full MSE",      "lower"),
]:
    best = _best_val(key, better)
    if best is None:
        continue
    best_conds = [n for n in cond_names
                  if _is_best(CONDITIONS[n]["metrics"].get(key), key, better)]
    analysis.append(
        f"- **{label}**: best = `{sf(best)}` → `{'`, `'.join(best_conds)}`"
    )

# pairwise_acc improvement vs "pixel" baseline
pa_pixel = None
if "pixel" in CONDITIONS:
    pa_pixel = CONDITIONS["pixel"]["metrics"].get("pairwise_acc")
if pa_pixel is not None:
    analysis.append("\n**Pairwise Acc vs `pixel` baseline:**")
    for name in cond_names:
        if name == "pixel":
            continue
        pa = CONDITIONS[name]["metrics"].get("pairwise_acc")
        if pa is not None:
            try:
                delta = float(pa) - float(pa_pixel)
                analysis.append(
                    f"  - `{name}`: {pct(pa_pixel)} → {pct(pa)} "
                    f"(Δ = {delta:+.4f})"
                )
            except (TypeError, ValueError):
                pass

analysis_str = "\n".join(analysis) if analysis else "_No conditions found or metrics missing._"

# Per-task table
def _per_task_table():
    all_tasks = set()
    for c in CONDITIONS.values():
        all_tasks.update(c["by_task"].keys())
    all_tasks = sorted(all_tasks)
    if not all_tasks:
        return ""

    hdr_cells  = ["Task"]
    for n in cond_names:
        hdr_cells += [f"{n[:14]}/grip_mse", f"{n[:14]}/pairwise"]
    lines = ["| " + " | ".join(hdr_cells) + " |",
             "| " + " | ".join(["---"] * len(hdr_cells)) + " |"]
    for task in all_tasks:
        row = [task[:42]]
        for name, c in CONDITIONS.items():
            t = c["by_task"].get(task, {})
            row.append(sf(t.get("gripper_mse")))
            row.append(sf(t.get("pairwise_acc")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

per_task_str = _per_task_table()
per_task_section = (
    f"## 3. Per-Task Breakdown\n\n{per_task_str}\n\n---\n"
    if per_task_str else ""
)

# Go/No-Go
def _go_nogo():
    rows = []
    pa_best = _best_val("pairwise_acc", "higher")
    g_best  = _best_val("gripper_mse",  "lower")
    d_best  = _best_val("dynamic_mse",  "lower")
    rows.append(f"| Best `pairwise_acc` | {sf(pa_best)} | {'✓ proceed if > 0.60' if pa_best and float(pa_best) > 0.60 else '✗ below 0.60 threshold'} |")
    rows.append(f"| Best `gripper_mse`  | {sf(g_best)}  | {'✓ improvement over baseline' if g_best is not None else 'N/A'} |")
    rows.append(f"| Best `dynamic_mse`  | {sf(d_best)}  | {'✓ dynamic region tracked' if d_best is not None else 'N/A'} |")

    best_cond_pa = [n for n in cond_names if _is_best(CONDITIONS[n]["metrics"].get("pairwise_acc"), "pairwise_acc", "higher")]
    proceed = pa_best is not None and float(pa_best) > 0.60
    rows.append(
        f"| Proceed to Phase 2 RFT? | **{'YES' if proceed else 'NO/CONDITIONAL'}** | "
        f"{'Use `' + (best_cond_pa[0] if best_cond_pa else '?') + '` as reward signal' if proceed else 'pairwise_acc ≤ 0.60; further tuning needed'} |"
    )
    return "\n".join(["| Question | Value | Interpretation |",
                       "| --- | --- | --- |"] + rows)

md = f"""# Phase 1: Pixel-Residual World Model — Results

**Suite**: `{suite}`
**Date**: {PHASE1.name}
**Conditions**: {', '.join(f'`{n}`' for n in cond_names)}
**Eval windows**: {win_str}

---

## 1. Aggregate Metrics Comparison

> **Bold** values = best per row.

{metric_table}

---

## 2. Analysis

{analysis_str}

---

{per_task_section}## 4. Go / No-Go for Phase 2 (RFT Training)

{_go_nogo()}

---

## 5. Metric Reference

| Metric | Meaning |
| --- | --- |
| `full_mse` | Mean squared error across all pixels (↓ better) |
| `gripper_mse` | MSE in motion center-of-mass crop (↓ better) |
| `dynamic_mse` | MSE within pixel-diff dynamic mask (↓ better) |
| `static_consistency_mse` | Residual leakage outside dynamic region (↓ better) |
| `pairwise_acc` | Fraction of windows: GT action gives lower LPIPS than shuffled (↑ better; >0.60 = proceed) |
| `lpips_gap` | GT LPIPS − shuffled LPIPS (↓ / more negative = clearer ranking signal) |

---

*Generated by `scripts/libero/phase1/summarize_residual_wm_eval.sh`*
"""

(PHASE1 / "comparison.md").write_text(md)
print(f"Wrote: {PHASE1 / 'comparison.md'}")

# ---- Console output ---------------------------------------------------------
print("\n=== Phase 1 Summary ===")
for key, label, _ in [
    ("full_mse",    "Full MSE    ", None),
    ("gripper_mse", "Gripper MSE ", None),
    ("dynamic_mse", "Dynamic MSE ", None),
    ("pairwise_acc","Pairwise Acc", None),
    ("lpips_gap",   "LPIPS Gap   ", None),
]:
    vals = "  ".join(
        f"{n}={sf(c['metrics'].get(key))}"
        for n, c in CONDITIONS.items()
    )
    print(f"  {label}: {vals}")
PYEOF

log "Summary complete."
echo ""
ls -la "${PHASE1_DIR}"/*.md "${PHASE1_DIR}"/*.json "${PHASE1_DIR}"/*.csv 2>/dev/null || true
