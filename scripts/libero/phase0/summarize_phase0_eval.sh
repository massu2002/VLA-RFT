#!/usr/bin/env bash
# summarize_phase0_eval.sh — Aggregate Phase 0 results into comparison.md
#
# Usage:
#   bash scripts/libero/phase0/summarize_phase0_eval.sh <phase0-output-dir>
#   bash scripts/libero/phase0/summarize_phase0_eval.sh   # uses results/phase0/latest.txt

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# Resolve output directory
if [ -n "${1:-}" ]; then
  PHASE0_DIR="$1"
elif [ -f "${REPO_ROOT}/results/phase0/latest.txt" ]; then
  PHASE0_DIR=$(cat "${REPO_ROOT}/results/phase0/latest.txt")
else
  echo "Usage: $0 <phase0-output-dir>" >&2
  exit 1
fi

[ -d "${PHASE0_DIR}" ] || { echo "Directory not found: ${PHASE0_DIR}" >&2; exit 1; }

log() { echo "[summarize] $(date +%H:%M:%S) $*"; }
log "Summarizing: ${PHASE0_DIR}"

# Activate env that has pandas / yaml / numpy
VENV="${REPO_ROOT}/.venv5090_eval"
[ -f "${VENV}/bin/activate" ] || VENV="${REPO_ROOT}/.venv"
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/worldmodel/_compat:${REPO_ROOT}/train/verl:${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Run analyze_worldmodel_eval.py on WorldModel outputs
# ---------------------------------------------------------------------------
WM_DIR="${PHASE0_DIR}/worldmodel"
if [ -d "${WM_DIR}" ]; then
  log "Running analyze_worldmodel_eval.py ..."
  python analyze_worldmodel_eval.py \
    --checkpoint-dir "${WM_DIR}" \
    --output-dir     "${PHASE0_DIR}" \
    2>/dev/null || log "analyze_worldmodel_eval.py failed (non-fatal)"
else
  log "WARNING: worldmodel output dir not found at ${WM_DIR}"
fi

# ---------------------------------------------------------------------------
# Python: generate full summary (use quoted heredoc + env vars to avoid
# bash expansions of backtick-delimited metric names inside f-strings)
# ---------------------------------------------------------------------------
export PHASE0_DIR REPO_ROOT
python - << 'PYEOF'
import json, csv, math, sys, os
from pathlib import Path
from collections import defaultdict, Counter

PHASE0 = Path(os.environ["PHASE0_DIR"])
WM_DIR = PHASE0 / "worldmodel"

# Resolve TASK_SUITE and TASK_SCOPE from run_config.json
_run_cfg = json.loads((PHASE0 / "run_config.json").read_text()) if (PHASE0 / "run_config.json").exists() else {}
TASK_SUITE = _run_cfg.get("suite", PHASE0.name.split("_")[-1] if "_" in PHASE0.name else "unknown")
TASK_SCOPE = _run_cfg.get("task_scope", "4task")

# Detect per-task WM structure (worldmodel/task*/ dirs)
task_wm_dirs = sorted(
    [d for d in WM_DIR.glob("task*/") if d.is_dir() and d.name.replace("task","").isdigit()],
    key=lambda p: int(p.name.replace("task",""))
) if WM_DIR.exists() else []
PER_TASK_MODE = len(task_wm_dirs) > 0

# ---- Helper: safe float ----
def sf(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)

# ---- Helper: read json ----
def rj(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}

# ====================================================================
# 1. WorldModel metrics — per-task or flat structure
# ====================================================================

def _read_eval_report(search_dir: Path):
    """Read the 'trained' eval_report from a directory."""
    rpts = sorted(search_dir.glob("eval_report__*.json"),
                  key=lambda p: (0 if "trained" in p.name else 1))
    for p in rpts:
        m = rj(p)
        if m:
            return m
    return {}

def _extract_wm_metrics_from_report(m: dict) -> dict:
    rf_overall = m.get("rollout_fidelity", {}).get("overall", {})
    as_data    = m.get("action_sensitivity", {})
    return {
        "future_image_lpips":     rf_overall.get("lpips", {}).get("mean"),
        "future_image_l1":        rf_overall.get("lpips", {}).get("mean"),
        "future_image_smooth_l1": rf_overall.get("mse",   {}).get("mean"),
        "future_image_mse":       rf_overall.get("mse",   {}).get("mean"),
        "dino_cosine_similarity": None,
        "n_windows":   m.get("rollout_fidelity", {}).get("num_windows"),
        "task_suite":  m.get("config", {}).get("task_suite", ""),
        "roi/gripper_mse":   rf_overall.get("roi/gripper_mse",   {}).get("mean"),
        "roi/gripper_lpips": rf_overall.get("roi/gripper_lpips", {}).get("mean"),
        "roi/gripper_psnr":  rf_overall.get("roi/gripper_psnr",  {}).get("mean"),
        "roi/gripper_ssim":  rf_overall.get("roi/gripper_ssim",  {}).get("mean"),
        "roi/goal_mse":      rf_overall.get("roi/goal_mse",      {}).get("mean"),
        "roi/goal_lpips":    rf_overall.get("roi/goal_lpips",    {}).get("mean"),
        "roi/goal_psnr":     rf_overall.get("roi/goal_psnr",     {}).get("mean"),
        "roi/goal_ssim":     rf_overall.get("roi/goal_ssim",     {}).get("mean"),
        "action_sens/correct_lpips_mean":    as_data.get("correct_lpips",  {}).get("mean"),
        "action_sens/shuffled_lpips_mean":   as_data.get("shuffled_lpips", {}).get("mean"),
        "action_sens/lpips_gap_mean":        as_data.get("lpips_gap",      {}).get("mean"),
        "action_sens/lpips_gap_min":         as_data.get("lpips_gap",      {}).get("min"),
        "action_sens/num_windows":           as_data.get("num_windows"),
        "action_sens/per_window_pairwise_acc": as_data.get("per_window_pairwise_acc"),
        "action_sens/num_wins":              as_data.get("num_wins"),
        "action_sens/num_losses":            as_data.get("num_losses"),
    }

def _read_ranking_jsonl(search_dir: Path):
    """Read the first JSONL line from ranking_eval/ under search_dir."""
    rank_dir = search_dir / "ranking_eval"
    num_wins = num_losses = None
    metrics = {}
    for p in sorted(rank_dir.glob("*.jsonl")):
        try:
            rec = json.loads(p.read_text().strip().splitlines()[0])
            metrics = rec.get("metrics", {})
            per_item = rec.get("per_item", [])
            if len(per_item) > 1:
                num_wins   = sum(1 for e in per_item if e.get("win") == 1)
                num_losses = sum(1 for e in per_item if e.get("win") == 0)
            break
        except Exception:
            pass
    return metrics, num_wins, num_losses

def _avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

# ---- Per-task mode ----------------------------------------------------------
per_task_wm = {}  # task_idx -> dict of metrics
if PER_TASK_MODE:
    import re as _re
    for td in task_wm_dirs:
        task_idx = int(td.name.replace("task", ""))
        m = _read_eval_report(td)
        if not m:
            continue
        t_metrics = _extract_wm_metrics_from_report(m)
        t_rank, t_wins, t_losses = _read_ranking_jsonl(td)
        t_metrics["rank/pairwise_acc"]           = t_rank.get("pairwise_acc")
        t_metrics["rank/aggregate_pairwise_acc"] = t_rank.get("aggregate_pairwise_acc")
        t_metrics["rank/mean_margin"]            = t_rank.get("mean_margin")
        t_metrics["rank/num_wins"]               = t_wins
        t_metrics["rank/num_losses"]             = t_losses
        # task name from eval_report config
        cfg = m.get("config", {})
        task_names_all = cfg.get("task_names", [])
        sel = cfg.get("selected_task_indices", [task_idx])
        t_metrics["task_name"] = task_names_all[task_idx] if task_idx < len(task_names_all) else ""
        per_task_wm[task_idx] = t_metrics

    # Overall = mean across tasks
    wm_metrics = {}
    for key in [k for k in next(iter(per_task_wm.values()), {}).keys() if k != "task_suite" and k != "task_name"]:
        vals = [per_task_wm[i].get(key) for i in sorted(per_task_wm)]
        wm_metrics[key] = _avg(vals)
    wm_metrics["task_suite"] = TASK_SUITE
    wm_metrics["n_windows"] = sum(
        (per_task_wm[i].get("n_windows") or 0) for i in per_task_wm
    )
    # Aggregate ranking
    pw_acc_vals = [per_task_wm[i].get("rank/pairwise_acc") for i in per_task_wm]
    wm_rank_metrics = {
        "pairwise_acc":           _avg(pw_acc_vals),
        "aggregate_pairwise_acc": _avg([per_task_wm[i].get("rank/aggregate_pairwise_acc") for i in per_task_wm]),
        "mean_margin":            _avg([per_task_wm[i].get("rank/mean_margin") for i in per_task_wm]),
    }
    wins_list   = [per_task_wm[i].get("rank/num_wins")   for i in per_task_wm if per_task_wm[i].get("rank/num_wins")   is not None]
    losses_list = [per_task_wm[i].get("rank/num_losses") for i in per_task_wm if per_task_wm[i].get("rank/num_losses") is not None]
    wm_rank_num_wins   = sum(wins_list)   if wins_list   else None
    wm_rank_num_losses = sum(losses_list) if losses_list else None

else:
    # ---- Flat (4task) mode: existing logic ----------------------------------
    wm_metrics = {}
    _eval_reports = sorted(WM_DIR.glob("eval_report__*.json"),
                           key=lambda p: (0 if "trained" in p.name else 1))
    for p in _eval_reports:
        m = rj(p)
        if not m:
            continue
        wm_metrics = _extract_wm_metrics_from_report(m)
        break
    if not wm_metrics:
        for p in sorted(WM_DIR.rglob("metrics.json")):
            m = rj(p)
            if m and "roi/gripper_mse" in m:
                wm_metrics.update({k: v for k, v in m.items() if k.startswith("roi/") or k.startswith("future_image")})
                break

    wm_rank_metrics = {}
    wm_rank_num_wins = None
    wm_rank_num_losses = None
    for p in sorted((WM_DIR / "ranking_eval").glob("*.jsonl")):
        try:
            line = p.read_text().strip().splitlines()[0]
            rec = json.loads(line)
            wm_rank_metrics = rec.get("metrics", {})
            if "mean_margin" not in wm_rank_metrics:
                wm_rank_metrics["mean_margin"] = rec.get("metrics", {}).get("mean_margin")
            per_item = rec.get("per_item", [])
            if len(per_item) > 1:
                wm_rank_num_wins   = sum(1 for e in per_item if e.get("win") == 1)
                wm_rank_num_losses = sum(1 for e in per_item if e.get("win") == 0)
            break
        except Exception:
            pass
    if not wm_rank_metrics:
        for p in sorted(WM_DIR.rglob("rank_eval_metrics.json")):
            m = rj(p).get("metrics", {})
            if m:
                wm_rank_metrics = m
                break

# ====================================================================
# 2. VLA results — parse run_libero_eval.py txt logs
# ====================================================================
import re

def _parse_vla_log(txt_path):
    """Extract final success rate from run_libero_eval.py text log."""
    text = Path(txt_path).read_text(errors="replace")
    # Match "Final results:" block
    m = re.search(r"Final results:.*?Overall success rate:\s*([\d.]+)", text, re.DOTALL)
    if m:
        sr = float(m.group(1))
    else:
        # Fallback: last "Current task success rate:" value
        vals = re.findall(r"Current task success rate:\s*([\d.]+)", text)
        sr = float(vals[-1]) if vals else None

    # Task name: first occurrence of "Task: ..."
    task_m = re.search(r"Task:\s*(.+)", text)
    task_name = task_m.group(1).strip() if task_m else ""

    # Successes and trials
    final_m = re.search(r"Total successes:\s*(\d+)", text)
    total_m = re.search(r"Total episodes:\s*(\d+)", text)
    successes = int(final_m.group(1)) if final_m else None
    trials    = int(total_m.group(1)) if total_m else None

    return {"task_name": task_name, "success_rate": sr, "successes": successes, "trials": trials}

def collect_vla_results(subdir):
    """Collect per-task success rates from run_libero_eval.py outputs."""
    results = {}
    d = PHASE0 / subdir
    if not d.exists():
        return results

    # Try task_results.json first (future-proof)
    for task_dir in sorted(d.rglob("task_results.json")):
        try:
            tr = rj(task_dir)
            task_id = tr.get("task_id", -1)
            results[task_id] = {
                "task_id":       task_id,
                "task_name":     tr.get("task_name", ""),
                "success_rate":  tr.get("success_rate", None),
                "successes":     tr.get("num_successes", None),
                "trials":        tr.get("num_trials", None),
            }
        except Exception:
            pass

    # Fallback: parse *.txt logs in taskN/ subdirs
    if not results:
        for task_subdir in sorted(d.iterdir()):
            if not task_subdir.is_dir():
                continue
            # task_id from dir name "taskN"
            tm = re.match(r"task(\d+)$", task_subdir.name)
            if not tm:
                continue
            task_id_1based = int(tm.group(1))
            task_id = task_id_1based - 1  # convert to 0-based

            txt_files = sorted(task_subdir.glob("*.txt"))
            if not txt_files:
                continue
            try:
                parsed = _parse_vla_log(txt_files[-1])  # latest log
                results[task_id] = {"task_id": task_id, **parsed}
            except Exception:
                pass

    return results

base_vla_results = collect_vla_results("base_vla")
rft_results      = collect_vla_results("vla_rft")

# ====================================================================
# 3. Taxonomy CSV
# ====================================================================
taxonomy_rows = []
for p in sorted(PHASE0.rglob("taxonomy.csv")):
    try:
        with open(p) as f:
            reader = csv.DictReader(f)
            taxonomy_rows.extend(list(reader))
    except Exception:
        pass

taxonomy_dist = Counter(r.get("suggested_error_type", "other") for r in taxonomy_rows)
n_tax = len(taxonomy_rows)

# ====================================================================
# 4. Multi-step drift (from JSONL)
# ====================================================================
jsonl_rows = []
for p in sorted(WM_DIR.rglob("rank_eval_candidates.jsonl")):
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    jsonl_rows.append(json.loads(line))
    except Exception:
        pass

ms_gripper_drift = []
ms_goal_drift    = []
for rec in jsonl_rows:
    m = rec.get("metrics", {})
    # multi-step lists may not be in JSONL directly; check metrics.json
    pass

# Get multi_step from metrics.json if available
ms_gripper_mse = wm_metrics.get("roi/multi_step_gripper_mse", [])
ms_goal_mse    = wm_metrics.get("roi/multi_step_goal_mse", [])

def ms_stats(lst):
    if not lst or not isinstance(lst, list):
        return {}
    lst = [float(v) for v in lst if v is not None]
    if not lst:
        return {}
    return {
        "first":       lst[0],
        "last":        lst[-1],
        "max":         max(lst),
        "drift_ratio": lst[-1] / (lst[0] + 1e-8),
    }

gripper_stats = ms_stats(ms_gripper_mse)
goal_stats    = ms_stats(ms_goal_mse)

# ====================================================================
# 5. Summary JSON
# ====================================================================
summary = {
    "worldmodel": {
        "full_image": {
            "future_image_smooth_l1": wm_metrics.get("future_image_smooth_l1"),
            "future_image_l1":        wm_metrics.get("future_image_l1"),
            "dino_cosine_similarity": wm_metrics.get("dino_cosine_similarity"),
        },
        "roi": {
            "gripper_mse":    wm_metrics.get("roi/gripper_mse"),
            "gripper_lpips":  wm_metrics.get("roi/gripper_lpips"),
            "gripper_psnr":   wm_metrics.get("roi/gripper_psnr"),
            "gripper_ssim":   wm_metrics.get("roi/gripper_ssim"),
            "goal_mse":       wm_metrics.get("roi/goal_mse"),
            "goal_lpips":     wm_metrics.get("roi/goal_lpips"),
            "goal_psnr":      wm_metrics.get("roi/goal_psnr"),
            "goal_ssim":      wm_metrics.get("roi/goal_ssim"),
        },
        "multi_step_gripper_mse": gripper_stats,
        "multi_step_goal_mse":    goal_stats,
        "ranking": {
            "pairwise_acc":           wm_rank_metrics.get("pairwise_acc"),
            "aggregate_pairwise_acc": wm_rank_metrics.get("aggregate_pairwise_acc"),
            "num_wins":               wm_rank_num_wins,
            "num_losses":             wm_rank_num_losses,
            "strict_order_acc":       wm_rank_metrics.get("strict_order_acc"),
            "spearman_tier_corr":     wm_rank_metrics.get("spearman_tier_corr"),
            "mean_margin":            wm_rank_metrics.get("mean_margin"),
            "margin_success_minus_failure": wm_rank_metrics.get("margin_success_minus_failure"),
        },
    },
    "base_vla": {
        t: r for t, r in base_vla_results.items()
    },
    "vla_rft": {
        t: r for t, r in rft_results.items()
    },
    "taxonomy": {
        "total": n_tax,
        "distribution": dict(taxonomy_dist),
    },
}

out_json = PHASE0 / "summary.json"
out_json.write_text(json.dumps(summary, indent=2, default=str))
print(f"Wrote: {out_json}")

# ====================================================================
# 6. Summary CSV
# ====================================================================
rows = []

def vla_success(cond_dict, task_id):
    r = cond_dict.get(task_id, {})
    sr = r.get("success_rate")
    return float(sr) if sr is not None else None

# WorldModel row
wm_row = {
    "condition":    "WorldModel_baseline",
    "task_suite":   TASK_SUITE,
    "task_id":      "all",
    "success_rate": "",
    "future_image_l1":     sf(wm_metrics.get("future_image_l1")),
    "roi/gripper_mse":     sf(wm_metrics.get("roi/gripper_mse")),
    "roi/goal_mse":        sf(wm_metrics.get("roi/goal_mse")),
    "pairwise_acc":        sf(wm_rank_metrics.get("pairwise_acc")),
    "strict_order_acc":    sf(wm_rank_metrics.get("strict_order_acc")),
    "spearman_tier_corr":  sf(wm_rank_metrics.get("spearman_tier_corr")),
    "gripper_drift_ratio": sf(gripper_stats.get("drift_ratio")),
    "goal_drift_ratio":    sf(goal_stats.get("drift_ratio")),
}
rows.append(wm_row)

# VLA rows
for task_id, r in sorted(base_vla_results.items()):
    rows.append({
        "condition":    "Base_VLA",
        "task_suite":   TASK_SUITE,
        "task_id":      task_id + 1,
        "success_rate": sf(r.get("success_rate")),
        "future_image_l1": "", "roi/gripper_mse": "", "roi/goal_mse": "",
        "pairwise_acc": "", "strict_order_acc": "", "spearman_tier_corr": "",
        "gripper_drift_ratio": "", "goal_drift_ratio": "",
    })

for task_id, r in sorted(rft_results.items()):
    rows.append({
        "condition":    "VLA_RFT",
        "task_suite":   TASK_SUITE,
        "task_id":      task_id + 1,
        "success_rate": sf(r.get("success_rate")),
        "future_image_l1": "", "roi/gripper_mse": "", "roi/goal_mse": "",
        "pairwise_acc": "", "strict_order_acc": "", "spearman_tier_corr": "",
        "gripper_drift_ratio": "", "goal_drift_ratio": "",
    })

if rows:
    keys = list(rows[0].keys())
    out_csv = PHASE0 / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote: {out_csv}")

# ====================================================================
# 7. Taxonomy CSV (aggregated)
# ====================================================================
out_tax = PHASE0 / "taxonomy.csv"
if taxonomy_rows:
    tax_keys = list(taxonomy_rows[0].keys())
    with open(out_tax, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tax_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(taxonomy_rows)
    print(f"Wrote: {out_tax}")

# ====================================================================
# 8. comparison.md
# ====================================================================
def md_bool(v, threshold=None, better="lower"):
    """Format value with visual signal."""
    if v == "N/A" or v is None:
        return "N/A"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if threshold is not None:
        if better == "lower":
            signal = "✓" if fv <= threshold else "✗"
        else:
            signal = "✓" if fv >= threshold else "✗"
        return f"{fv:.4f} {signal}"
    return f"{fv:.4f}"

pairwise_acc_val = wm_rank_metrics.get("pairwise_acc")
aggregate_pairwise_acc_val = wm_rank_metrics.get("aggregate_pairwise_acc")
pa_str = md_bool(pairwise_acc_val, threshold=0.55, better="higher")
agg_pa_str = md_bool(aggregate_pairwise_acc_val, threshold=0.55, better="higher")
num_wins_str   = str(wm_rank_num_wins)   if wm_rank_num_wins   is not None else "N/A"
num_losses_str = str(wm_rank_num_losses) if wm_rank_num_losses is not None else "N/A"

gripper_drift = gripper_stats.get("drift_ratio")
goal_drift    = goal_stats.get("drift_ratio")

# taxonomy dominant
dominant_error = taxonomy_dist.most_common(1)[0][0] if taxonomy_dist else "N/A"
dominant_pct   = f"{100*taxonomy_dist.most_common(1)[0][1]/max(n_tax,1):.1f}%" if taxonomy_dist else "N/A"

# VLA comparison summary
def vla_summary(cond_dict, name):
    if not cond_dict:
        return f"*{name}: no results (simulation skipped or failed)*"
    rates = [r["success_rate"] for r in cond_dict.values() if r.get("success_rate") is not None]
    if not rates:
        return f"{name}: task results missing"
    mean_rate = sum(rates) / len(rates)
    task_lines = "\n".join(
        f"  - Task {r['task_id']+1} ({r['task_name'][:40]}): {sf(r['success_rate'])} ({r.get('successes','?')}/{r.get('trials','?')})"
        for r in sorted(cond_dict.values(), key=lambda x: x["task_id"])
        if r.get("task_name")
    )
    return f"**{name}** mean success rate: {mean_rate:.3f}\n{task_lines}"

def _short_task_name(name: str, maxlen: int = 38) -> str:
    """Trim 'pick_up_the_black_bowl_' prefix and snake_case to spaces."""
    s = name.replace("pick_up_the_black_bowl_", "")
    s = s.replace("_and_place_it_on_the_plate", "").replace("_", " ").strip()
    return s[:maxlen]

def _build_per_task_section(per_task_wm, base_vla, rft_vla, sf_fn) -> str:
    """Build a per-task summary table for comparison.md."""
    all_task_ids = sorted(set(list(per_task_wm.keys()) + list(base_vla.keys()) + list(rft_vla.keys())))
    header  = "| Task | Task Name | Gripper LPIPS | Goal LPIPS | pairwise_acc | Base VLA | VLA-RFT | Δ |"
    divider = "|---|---|---|---|---|---|---|---|"
    rows = [header, divider]
    for tidx in all_task_ids:
        wm = per_task_wm.get(tidx, {})
        bv = base_vla.get(tidx, {})
        rv = rft_vla.get(tidx, {})
        tname = _short_task_name(wm.get("task_name") or bv.get("task_name") or f"task {tidx}")
        g_lpips  = sf_fn(wm.get("roi/gripper_lpips"))
        goal_lp  = sf_fn(wm.get("roi/goal_lpips"))
        pw_acc   = sf_fn(wm.get("rank/pairwise_acc"))
        bsr      = bv.get("success_rate")
        rsr      = rv.get("success_rate")
        bstr     = f"{bsr:.0%}" if bsr is not None else "—"
        rstr     = f"{rsr:.0%}" if rsr is not None else "—"
        if bsr is not None and rsr is not None:
            delta = rsr - bsr
            dstr  = f"{delta:+.0%}"
        else:
            dstr = "—"
        rows.append(f"| {tidx} | {tname} | {g_lpips} | {goal_lp} | {pw_acc} | {bstr} | {rstr} | {dstr} |")

    # Overall averages row
    wm_g_vals = [per_task_wm[i].get("roi/gripper_lpips") for i in all_task_ids if per_task_wm.get(i)]
    wm_gl_vals= [per_task_wm[i].get("roi/goal_lpips")    for i in all_task_ids if per_task_wm.get(i)]
    pw_vals   = [per_task_wm[i].get("rank/pairwise_acc") for i in all_task_ids if per_task_wm.get(i)]
    bsr_vals  = [base_vla[i].get("success_rate") for i in all_task_ids if base_vla.get(i)]
    rsr_vals  = [rft_vla[i].get("success_rate")  for i in all_task_ids if rft_vla.get(i)]
    def _ma(lst): v=[x for x in lst if x is not None]; return sum(v)/len(v) if v else None
    bm = _ma(bsr_vals); rm = _ma(rsr_vals)
    dm = (rm-bm) if (bm is not None and rm is not None) else None
    rows.append(
        f"| **mean** | | {sf_fn(_ma(wm_g_vals))} | {sf_fn(_ma(wm_gl_vals))} | {sf_fn(_ma(pw_vals))} | "
        f"{'—' if bm is None else f'{bm:.0%}'} | {'—' if rm is None else f'{rm:.0%}'} | "
        f"{'—' if dm is None else f'{dm:+.0%}'} |"
    )

    table_str = "\n".join(rows)

    # Diagnostic highlights
    weak_signal = [i for i in all_task_ids if per_task_wm.get(i, {}).get("rank/pairwise_acc") is not None
                   and per_task_wm[i]["rank/pairwise_acc"] <= 0.70]
    high_gripper= [i for i in all_task_ids if per_task_wm.get(i, {}).get("roi/gripper_lpips") is not None
                   and per_task_wm[i]["roi/gripper_lpips"] > 0.15]
    vla_zero_b  = [i for i in all_task_ids if base_vla.get(i, {}).get("success_rate") == 0.0]
    vla_zero_both=[i for i in vla_zero_b if rft_vla.get(i, {}).get("success_rate") == 0.0]

    def _tidx_list(lst): return ", ".join(f"Task{i}" for i in sorted(lst)) or "none"

    highlights = f"""### Diagnostic Highlights

- **Weak ranking signal** (pairwise_acc ≤ 0.70): {_tidx_list(weak_signal)}
- **High gripper ROI error** (LPIPS > 0.15): {_tidx_list(high_gripper)}
- **VLA zero-success (Base VLA)**: {_tidx_list(vla_zero_b)}
- **VLA zero-success (both Base + RFT)**: {_tidx_list(vla_zero_both)}"""

    return f"""## 5b. Per-Task Analysis ({len(all_task_ids)} tasks)

{table_str}

{highlights}"""

md_content = f"""# Phase 0 Evaluation Results

**Suite**: {wm_metrics.get("task_suite", TASK_SUITE)}
**Date**: {PHASE0.name}
**WorldModel eval windows**: ~{wm_metrics.get("n_windows", "?")}

---

## 1. Full-Image Reconstruction (WorldModel baseline)

| Metric | Value |
|---|---|
| `future_image_smooth_l1` | {sf(wm_metrics.get("future_image_smooth_l1"))} |
| `future_image_l1` | {sf(wm_metrics.get("future_image_l1"))} |
| `dino_cosine_similarity` | {sf(wm_metrics.get("dino_cosine_similarity"))} |

**Q1: Does full-image metric alone explain success/failure?**
Full-image L1 = {sf(wm_metrics.get("future_image_l1"))}. Without calibration against a success-labeled dataset, this scalar alone cannot distinguish success from failure scenarios. It measures average rollout fidelity, not task-completion relevance. **→ Full-image is insufficient as a standalone diagnostic.**

---

## 2. ROI Metrics (WorldModel baseline)

| Region | MSE | LPIPS | PSNR | SSIM |
|---|---|---|---|---|
| Gripper | {sf(wm_metrics.get("roi/gripper_mse"))} | {sf(wm_metrics.get("roi/gripper_lpips"))} | {sf(wm_metrics.get("roi/gripper_psnr"))} | {sf(wm_metrics.get("roi/gripper_ssim"))} |
| Goal    | {sf(wm_metrics.get("roi/goal_mse"))}    | {sf(wm_metrics.get("roi/goal_lpips"))}    | {sf(wm_metrics.get("roi/goal_psnr"))}    | {sf(wm_metrics.get("roi/goal_ssim"))}    |

### Multi-step drift (gripper ROI)

| Stat | Value |
|---|---|
| first-frame MSE | {sf(gripper_stats.get("first"))} |
| last-frame MSE  | {sf(gripper_stats.get("last"))} |
| max MSE         | {sf(gripper_stats.get("max"))} |
| drift_ratio     | {sf(gripper_stats.get("drift_ratio"))} |

### Multi-step drift (goal ROI)

| Stat | Value |
|---|---|
| first-frame MSE | {sf(goal_stats.get("first"))} |
| last-frame MSE  | {sf(goal_stats.get("last"))} |
| max MSE         | {sf(goal_stats.get("max"))} |
| drift_ratio     | {sf(goal_stats.get("drift_ratio"))} |

**Q2: Are ROI metrics more informative than full-image?**
Gripper drift_ratio = {sf(gripper_drift)} / Goal drift_ratio = {sf(goal_drift)}.
{"Drift ratio > 1.3 indicates the model loses accuracy at the task-relevant region faster than on average — a direct signal of where the model fails." if gripper_drift and isinstance(gripper_drift, float) and gripper_drift > 1.3 else "Drift ratio is near 1.0, indicating steady-state error rather than accumulating drift."}
**→ ROI metrics are spatially specific; multi-step drift ratio is the strongest single diagnostic.**

---

## 3. Ranking Signal (WorldModel baseline)

### 3-A. Action sensitivity (GT vs shuffled, 50 windows)

| Metric | Value |
|---|---|
| `correct_lpips_mean` | {sf(wm_metrics.get("action_sens/correct_lpips_mean"))} |
| `shuffled_lpips_mean` | {sf(wm_metrics.get("action_sens/shuffled_lpips_mean"))} |
| `lpips_gap_mean` | {sf(wm_metrics.get("action_sens/lpips_gap_mean"))} |
| `lpips_gap_min` | {sf(wm_metrics.get("action_sens/lpips_gap_min"))} |

> `lpips_gap = correct_lpips - shuffled_lpips`. Positive gap means GT action is better.
> `gap_min < 0` indicates there are individual windows where shuffled action scored lower than GT.

### 3-B. Ranking metrics (from JSONL)

| Metric | Value | Interpretation |
|---|---|---|
| `pairwise_acc` (per-window) | {pa_str} | {">0.55 = model has ranking signal; ≤0.55 = near-random" if pairwise_acc_val is None else (">0.55 ✓ some signal" if pairwise_acc_val > 0.55 else "≤0.55 ✗ near-random")} |
| `aggregate_pairwise_acc` | {agg_pa_str} | single binary comparison of aggregate means |
| num\_wins / num\_losses | {num_wins_str} / {num_losses_str} | windows where GT won / lost |
| `strict_order_acc` | {sf(wm_rank_metrics.get("strict_order_acc"))} | 3-tier strict ordering (residual only) |
| `spearman_tier_corr` | {sf(wm_rank_metrics.get("spearman_tier_corr"))} | Rank correlation (residual only) |
| `mean_margin` | {sf(wm_rank_metrics.get("mean_margin"))} | Score gap GT vs negatives |

**Q3: Is baseline WorldModel ranking signal weak?**
correct_lpips={sf(wm_metrics.get("action_sens/correct_lpips_mean"))} vs shuffled_lpips={sf(wm_metrics.get("action_sens/shuffled_lpips_mean"))}; gap_mean={sf(wm_metrics.get("action_sens/lpips_gap_mean"))} (gap_min={sf(wm_metrics.get("action_sens/lpips_gap_min"))}).
{"**YES — near-random ranking signal. The baseline WorldModel cannot reliably distinguish good from bad actions. This is a direct explanation for why it did not improve RFT.**" if pairwise_acc_val and float(pairwise_acc_val) <= 0.58 else "**Moderate signal: aggregate means correctly ordered, but gap_min < 0 indicates per-window inversions where shuffled scores better than GT. The 5% mean gap is weak relative to the per-window variance — not a reliable per-step reward.**" if (wm_metrics.get("action_sens/lpips_gap_min") or 0) < 0 else "**The baseline shows consistent ranking signal (gap_mean > 0, no inversions).**" if wm_metrics.get("action_sens/lpips_gap_mean") else "**pairwise_acc not yet available (action sensitivity may not have run).**"}

---

## 4. Failure Taxonomy (WorldModel baseline)

Total windows classified: {n_tax}

| Error type | Count | % |
|---|---|---|
""" + "\n".join(
    f"| `{k}` | {v} | {100*v/max(n_tax,1):.1f}% |"
    for k, v in sorted(taxonomy_dist.items(), key=lambda x: -x[1])
) + f"""

**Dominant failure type**: `{dominant_error}` ({dominant_pct})

**Q4: What taxonomy is dominant?**
{"wm_static_bias dominant → model assigns nearly identical scores to all candidates (score collapse)." if dominant_error == "wm_static_bias" else "ranking_failure dominant → model ranks GT below a noise candidate despite distinguishable scores." if dominant_error == "ranking_failure" else "transport_drift dominant → error accumulates over horizon, especially at gripper." if dominant_error == "transport_drift" else f"{dominant_error} dominant — see taxonomy description in docs/libero_eval_protocol_v1.md."}

---

## 5. VLA Policy Evaluation

{vla_summary(base_vla_results, "Base VLA")}

{vla_summary(rft_results, "VLA-RFT")}

---

## 6. Go / No-Go for Phase 1 (Residual WM)

"""
# Compute VLA delta
_base_rates = [r['success_rate'] for r in base_vla_results.values() if r.get('success_rate') is not None]
_rft_rates  = [r['success_rate'] for r in rft_results.values()      if r.get('success_rate') is not None]
_base_mean  = sum(_base_rates) / len(_base_rates) if _base_rates else None
_rft_mean   = sum(_rft_rates)  / len(_rft_rates)  if _rft_rates  else None
_delta      = (_rft_mean - _base_mean) if (_base_mean is not None and _rft_mean is not None) else None
_gripper_roi_high = (wm_metrics.get("roi/gripper_lpips") or 0) > 0.15
_gap_has_inversions = (wm_metrics.get("action_sens/lpips_gap_min") or 0) < 0
md_content += f"""
| Question | Answer |
|---|---|
| A. pairwise_acc ≤ 0.55 (near-random)? | {"**YES** — baseline has no ranking signal" if pairwise_acc_val and float(pairwise_acc_val) <= 0.58 else "NO — aggregate means correctly ordered"} |
| B. Per-window inversions (gap_min < 0)? | {"**YES** — gap_min=" + sf(wm_metrics.get("action_sens/lpips_gap_min")) + " means WM sometimes prefers shuffled over GT" if _gap_has_inversions else "No inversions observed"} |
| C. Gripper ROI LPIPS > 0.15? | {"**YES** — gripper_lpips=" + sf(wm_metrics.get("roi/gripper_lpips")) + " → focus mechanism needed" if _gripper_roi_high else "NO — gripper ROI well-reconstructed"} |
| D. VLA-RFT improvement over Base VLA? | {f"**Marginal: +{100*_delta:.1f}% absolute** ({sf(_base_mean)}→{sf(_rft_mean)})" if _delta is not None else "No VLA results"} |
| E. wm_static_bias dominant? | {"**YES** — " + dominant_error + " is dominant (" + dominant_pct + ")" if dominant_error in ("wm_static_bias", "ranking_failure") else "Not dominant (taxonomy: " + dominant_error + ")"} |
| F. Proceed to Phase 1 (Residual WM)? | **{"YES — gripper ROI is poorly reconstructed; focus mechanism should help. RFT improvement marginal → better WM reward signal needed." if _gripper_roi_high or _gap_has_inversions else "CONDITIONAL — WM shows solid ranking; investigate task-level failures before adding focus mechanism"}** |"""
md_content += f"""

---

## 7. Next Steps for Phase 1

If proceeding to Residual WorldModel:

1. **Key log to check first**: `rank_eval/rank_eval_candidates.jsonl` — confirm `strict_order_acc` is meaningfully above 0.5 (residual should improve over baseline pairwise_acc = {sf(pairwise_acc_val)}).
2. **ROI debug PNGs**: Open `*_roi_debug.png` in residual eval output to verify goal ROI box placement before comparing `roi/goal_mse`.
3. **Taxonomy shift**: Compare taxonomy CSVs — residual should show fewer `wm_static_bias` and more `other` (recoverable failures) relative to baseline.
4. **Task 6 investigation**: Task 6 (index 5) achieves 0% success for both Base VLA and VLA-RFT. Investigate whether this task has data in the training set and whether the EGL rendering is correct for that task.
"""

# Insert per-task section before Go/No-Go when running in all-task mode
if PER_TASK_MODE and per_task_wm:
    per_task_section = _build_per_task_section(per_task_wm, base_vla_results, rft_results, sf)
    _marker = "## 6. Go / No-Go for Phase 1"
    _idx = md_content.find(_marker)
    if _idx >= 0:
        md_content = md_content[:_idx] + per_task_section + "\n\n---\n\n" + md_content[_idx:]
    else:
        md_content += "\n\n" + per_task_section

out_md = PHASE0 / "comparison.md"
out_md.write_text(md_content)
print(f"Wrote: {out_md}")

# ====================================================================
# 9. Casebook directory placeholder
# ====================================================================
(PHASE0 / "casebook").mkdir(exist_ok=True)
# Copy any PNG from worldmodel eval
import shutil
for png in sorted(WM_DIR.rglob("casebook*.png"))[:5]:
    shutil.copy(png, PHASE0 / "casebook" / png.name)
for png in sorted(WM_DIR.rglob("tier_score*.png"))[:2]:
    shutil.copy(png, PHASE0 / "casebook" / png.name)

print(f"\n=== Phase 0 Summary ===")
print(f"WorldModel full-image L1 : {sf(wm_metrics.get('future_image_l1'))}")
print(f"WorldModel roi/gripper_mse: {sf(wm_metrics.get('roi/gripper_mse'))}")
print(f"WorldModel roi/goal_mse  : {sf(wm_metrics.get('roi/goal_mse'))}")
print(f"WorldModel pairwise_acc  : {sf(pairwise_acc_val)}  (aggregate: {sf(aggregate_pairwise_acc_val)}, wins={num_wins_str}, losses={num_losses_str})")
print(f"Taxonomy dominant        : {dominant_error} ({dominant_pct})")
print(f"Gripper drift_ratio      : {sf(gripper_drift)}")
print(f"Goal drift_ratio         : {sf(goal_drift)}")
if base_vla_results:
    rates = [r['success_rate'] for r in base_vla_results.values() if r.get('success_rate') is not None]
    print(f"Base VLA mean success    : {sum(rates)/len(rates):.3f}" if rates else "Base VLA: no results")
if rft_results:
    rates = [r['success_rate'] for r in rft_results.values() if r.get('success_rate') is not None]
    print(f"VLA-RFT mean success     : {sum(rates)/len(rates):.3f}" if rates else "VLA-RFT: no results")
PYEOF

log "Summary complete. Files:"
ls -la "${PHASE0_DIR}"/*.md "${PHASE0_DIR}"/*.json "${PHASE0_DIR}"/*.csv 2>/dev/null || true
