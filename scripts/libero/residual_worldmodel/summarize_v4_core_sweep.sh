#!/usr/bin/env bash
# summarize_v4_core_sweep.sh — Backend: aggregate v4 core sweep results.
#
# Reads per-condition outputs under ${OUT_ROOT}/<exp_name>/ and produces:
#   ${OUT_ROOT}/summary/v4_core_sweep_summary.md
#   ${OUT_ROOT}/summary/v4_core_sweep_summary.csv
#   ${OUT_ROOT}/summary/v4_core_sweep_summary.json
#   ${OUT_ROOT}/summary/v4_core_sweep_ranking.csv
#
# Called by scripts/libero/phase1/summarize_v4_core_sweep.sh.
#
# Required env:
#   OUT_ROOT    — results/phase1/residual_worldmodel/${RUN_NAME}
#   RUN_NAME    — sweep run name
#
# Optional:
#   SWEEP_CONFIG — path to JSON config (for original experiment metadata)
#   TASK_SUITE

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

RUN_NAME="${RUN_NAME:?'RUN_NAME required'}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/residual_worldmodel/${RUN_NAME}}"
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/libero/phase1/v4_core_sweep.json}"
TASK_SUITE="${TASK_SUITE:-}"
SUMMARY_DIR="${OUT_ROOT}/summary"

[ -d "${OUT_ROOT}" ] || { echo "[summarize-v4] OUT_ROOT not found: ${OUT_ROOT}" >&2; exit 1; }
mkdir -p "${SUMMARY_DIR}"

log() { echo "[summarize-v4-sweep] $(date +%H:%M:%S) $*"; }
log "Summarizing: ${OUT_ROOT}"
log "Summary → : ${SUMMARY_DIR}"

setup_env
export TF_CPP_MIN_LOG_LEVEL=3
cd "${REPO_ROOT}"

export OUT_ROOT SUMMARY_DIR SWEEP_CONFIG RUN_NAME

python3 - <<'PYEOF'
import csv, json, math, os, sys
from pathlib import Path
from collections import OrderedDict

OUT_ROOT   = Path(os.environ["OUT_ROOT"])
SUM_DIR    = Path(os.environ["SUMMARY_DIR"])
CFG_PATH   = Path(os.environ.get("SWEEP_CONFIG", ""))
RUN_NAME   = os.environ.get("RUN_NAME", OUT_ROOT.name)

# ── Helpers ──────────────────────────────────────────────────────────────────
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

def _fv(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None

def _latest_trainer_metrics(ckpt_dir):
    root = Path(ckpt_dir) if ckpt_dir else None
    if not root or not root.exists():
        return {}
    states = sorted(
        root.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[-1]) if p.parent.name.split("-")[-1].isdigit() else -1,
    )
    if not states:
        direct = root / "trainer_state.json"
        states = [direct] if direct.exists() else []
    if not states:
        return {}
    try:
        state = json.loads(states[-1].read_text())
    except Exception:
        return {}
    for entry in reversed(state.get("log_history", [])):
        if any(k in entry for k in (
            "loss_rank", "loss_entropy", "loss_diversity", "neg_task_match_rate",
            "neg_num_same_task", "neg_num_temporal_perm", "neg_num_zero_random",
        )):
            out = {k: entry.get(k) for k in (
                "loss_rank", "loss_entropy", "loss_diversity", "neg_task_match_rate",
                "neg_num_same_task", "neg_num_temporal_perm", "neg_num_zero_random",
                "neg_num_batch_roll_fallback",
            ) if k in entry}
            counts = {
                "same_task_other_window": entry.get("neg_num_same_task"),
                "temporal_permutation": entry.get("neg_num_temporal_perm"),
                "zero_random": entry.get("neg_num_zero_random"),
                "batch_roll_fallback": entry.get("neg_num_batch_roll_fallback"),
            }
            counts = {k: v for k, v in counts.items() if v is not None}
            if counts:
                out["negative_type_counts"] = counts
            return out
    return {}

# ── Load sweep config (for experiment metadata) ──────────────────────────────
exp_cfg_by_name = {}
if CFG_PATH.exists():
    try:
        raw_cfg = json.loads(CFG_PATH.read_text())
        common = raw_cfg.get("common", {})
        for ex in raw_cfg.get("experiments", []):
            name = ex.get("exp_name", "")
            merged = {**common, **ex}
            exp_cfg_by_name[name] = merged
    except Exception as e:
        print(f"[warn] could not load sweep config: {e}")

# ── Scan condition dirs ───────────────────────────────────────────────────────
CONDITIONS = OrderedDict()
for d in sorted(OUT_ROOT.iterdir()):
    if not d.is_dir() or d.name in ("summary", "logs"):
        continue
    agg_path = d / "aggregate_metrics.json"
    cfg_path  = d / "v4_sweep_config_used.json"
    if not agg_path.exists() and not cfg_path.exists():
        continue

    metrics = {}
    if agg_path.exists():
        try:
            raw = json.loads(agg_path.read_text())
            metrics = raw.get("metrics", raw)
        except Exception as e:
            print(f"[warn] {agg_path}: {e}")

    exp_cfg = {}
    if cfg_path.exists():
        try:
            exp_cfg = json.loads(cfg_path.read_text())
        except Exception:
            pass
    # Also merge from sweep config if available
    if d.name in exp_cfg_by_name:
        for k, v in exp_cfg_by_name[d.name].items():
            exp_cfg.setdefault(k, v)

    CONDITIONS[d.name] = {
        "exp_name":    d.name,
        "stage":       exp_cfg.get("stage", "v4b"),
        "history_k":   exp_cfg.get("history_length", "?"),
        "num_q":       exp_cfg.get("num_dynamic_queries", "?"),
        "motion_bias": exp_cfg.get("use_motion_bias", "?"),
        "scorer":      exp_cfg.get("use_action_future_scorer", "?"),
        "lambda_rank": exp_cfg.get("lambda_rank", "?"),
        "rank_margin": exp_cfg.get("rank_margin", "?"),
        "neg_type":    exp_cfg.get("negative_type", "?"),
        "neg_mix":     exp_cfg.get("negative_mix") or "",
        "lambda_img":  exp_cfg.get("lambda_image", "?"),
        "lambda_dyn":  exp_cfg.get("lambda_dynamic", "?"),
        "lambda_stat": exp_cfg.get("lambda_static", "?"),
        "lambda_qry":  exp_cfg.get("lambda_query", "?"),
        "lambda_sp":   exp_cfg.get("lambda_sparse", "?"),
        "metrics":     metrics,
        "ckpt_dir":    exp_cfg.get("ckpt_dir", ""),
        "cond_out":    str(d),
        "agg_path":    str(agg_path) if agg_path.exists() else "",
    }

    train_metrics = _latest_trainer_metrics(CONDITIONS[d.name]["ckpt_dir"])
    for k, v in train_metrics.items():
        CONDITIONS[d.name]["metrics"].setdefault(k, v)

if not CONDITIONS:
    print(f"[warn] No conditions found under {OUT_ROOT}")
    sys.exit(0)

print(f"Found {len(CONDITIONS)} condition(s).")

# ── Metric key definitions ────────────────────────────────────────────────────
METRIC_COLS = [
    ("full_mse",                    "full_mse",                 "lower"),
    ("full_lpips",                  "full_lpips",               "lower"),
    ("gripper_mse",                 "gripper_mse",              "lower"),
    ("gripper_lpips",               "gripper_lpips",            "lower"),
    ("dynamic_mse",                 "dynamic_mse",              "lower"),
    ("dynamic_lpips",               "dynamic_lpips",            "lower"),
    ("pairwise_acc_lpips",          "pairwise_acc_lpips",       "higher"),
    ("lpips_gap_mean",              "lpips_gap",                "higher"),
    ("lpips_gap_min",               "lpips_gap_min",            "higher"),
    ("reverse_windows_lpips",       "reverse_windows_lpips",    "lower"),
    ("pairwise_acc_score",          "pairwise_acc_score",       "higher"),
    ("score_gap_mean",              "score_gap_mean",           "higher"),
    ("score_gap_min",               "score_gap_min",            "higher"),
    ("reverse_windows_score",       "reverse_windows_score",    "lower"),
    ("action_correct_best_rate",    "action_correct_best_rate", "higher"),
    ("pred_correct_vs_shuffle_mse", "pred_correct_vs_shuffle_mse", "lower"),
    ("fuser_mask_mean",             "fuser_mask_mean",          "neutral"),
    ("fuser_mask_max",              "fuser_mask_max",           "neutral"),
    ("fuser_mask_overlap",          "fuser_mask_overlap",       "neutral"),
    ("dynamic_mask_mean",           "dynamic_mask_mean",        "neutral"),
    ("dynamic_mask_entropy",        "dynamic_mask_entropy",     "neutral"),
    ("dynamic_mask_overlap",        "dynamic_mask_overlap",     "neutral"),
    ("copy_current_full_mse",       "copy_current_mse",         "lower"),
    ("full_mse_over_copy_current_mse", "full_mse_over_copy_current_mse", "lower"),
    ("future_dynamic_query_norm",   "future_dynamic_query_norm","neutral"),
    ("neg_task_match_rate",         "neg_task_match_rate",      "higher"),
    ("loss_rank",                   "loss_rank",                "lower"),
    ("loss_entropy",                "loss_entropy",             "lower"),
    ("loss_diversity",              "loss_diversity",           "lower"),
    ("skipped_history_windows",     "skipped_history_windows",  "lower"),
]

def _m(cond, key):
    return _fv(cond["metrics"].get(key))

def _best_val(col_key, better, conds):
    if better == "neutral":
        return None
    vals = [v for c in conds.values() if (v := _m(c, col_key)) is not None]
    if not vals:
        return None
    return min(vals) if better == "lower" else max(vals)

def _is_best(val, col_key, better, conds):
    if better == "neutral" or val is None:
        return False
    best = _best_val(col_key, better, conds)
    return best is not None and abs(val - best) < 1e-9

# ── Compute full_mse/copy_current ratio ──────────────────────────────────────
for c in CONDITIONS.values():
    fmse = _m(c, "full_mse")
    ccmse = _m(c, "copy_current_mse")
    if fmse is not None and ccmse is not None and ccmse > 1e-9:
        c["metrics"]["full_mse_over_copy_current_mse"] = fmse / ccmse
    else:
        c["metrics"]["full_mse_over_copy_current_mse"] = None

# ── CSV ───────────────────────────────────────────────────────────────────────
CSV_STATIC_COLS = [
    "exp_name", "stage", "history_length", "num_dynamic_queries",
    "use_motion_bias", "use_action_future_scorer", "lambda_rank", "rank_margin",
    "negative_type", "negative_mix",
]
CSV_METRIC_KEYS = [
    "full_mse", "full_lpips", "gripper_mse", "gripper_lpips",
    "dynamic_mse", "dynamic_lpips",
    "pairwise_acc_lpips", "lpips_gap", "lpips_gap_min", "reverse_windows_lpips",
    "pairwise_acc_score", "score_gap_mean", "score_gap_min", "reverse_windows_score",
    "action_correct_best_rate", "pred_correct_vs_shuffle_mse",
    "fuser_mask_mean", "fuser_mask_max", "dynamic_mask_mean", "dynamic_mask_entropy",
    "fuser_mask_overlap", "dynamic_mask_overlap",
    "copy_current_mse", "full_mse_over_copy_current_mse",
    "neg_task_match_rate", "negative_type_counts",
    "loss_rank", "loss_entropy", "loss_diversity",
    "future_dynamic_query_norm", "skipped_history_windows",
]

rows = []
for c in CONDITIONS.values():
    row = {
        "exp_name":              c["exp_name"],
        "stage":                 c["stage"],
        "history_length":        str(c["history_k"]),
        "num_dynamic_queries":   str(c["num_q"]),
        "use_motion_bias":       str(c["motion_bias"]),
        "use_action_future_scorer": str(c["scorer"]),
        "lambda_rank":           str(c["lambda_rank"]),
        "rank_margin":           str(c["rank_margin"]),
        "negative_type":         c["neg_type"],
        "negative_mix":          c["neg_mix"],
        "checkpoint_path":       c["ckpt_dir"],
        "aggregate_path":        c["agg_path"],
    }
    for k in CSV_METRIC_KEYS:
        raw_v = c["metrics"].get(k)
        row[k] = json.dumps(raw_v, sort_keys=True) if isinstance(raw_v, dict) else sf(_m(c, k))
    rows.append(row)

fieldnames = CSV_STATIC_COLS + CSV_METRIC_KEYS + ["checkpoint_path", "aggregate_path"]
csv_path = SUM_DIR / "v4_core_sweep_summary.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"Wrote: {csv_path}")

# ── JSON summary ──────────────────────────────────────────────────────────────
summary = {
    "run_name":   RUN_NAME,
    "out_root":   str(OUT_ROOT),
    "n_conditions": len(CONDITIONS),
    "conditions": {
        name: {
            "stage":   c["stage"],
            "history_length": c["history_k"],
            "num_dynamic_queries": c["num_q"],
            "use_motion_bias": c["motion_bias"],
            "use_action_future_scorer": c["scorer"],
            "lambda_rank": c["lambda_rank"],
            "rank_margin": c["rank_margin"],
            "negative_mix": c["neg_mix"],
            "negative_type": c["neg_type"],
            "metrics": {
                **{k: _m(c, mk) for k, mk, _ in METRIC_COLS},
                "negative_type_counts": c["metrics"].get("negative_type_counts"),
            },
        }
        for name, c in CONDITIONS.items()
    }
}
json_path = SUM_DIR / "v4_core_sweep_summary.json"
json_path.write_text(json.dumps(summary, indent=2, default=str))
print(f"Wrote: {json_path}")

# ── Ranking CSV (sorted by pairwise_acc_score desc, fallback lpips asc) ───────
def _rank_key(c):
    pa_s = _m(c, "pairwise_acc_score")
    pa_l = _m(c, "pairwise_acc_lpips")
    sg   = _m(c, "score_gap_mean")
    fmse = _m(c, "full_mse")
    return (
        -(pa_s if pa_s is not None else -999),
        -(pa_l if pa_l is not None else -999),
        -(sg   if sg   is not None else -999),
        fmse if fmse is not None else 999,
    )

ranked = sorted(CONDITIONS.values(), key=_rank_key)
rank_path = SUM_DIR / "v4_core_sweep_ranking.csv"
with open(rank_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["rank"] + fieldnames, extrasaction="ignore")
    w.writeheader()
    for i, c in enumerate(ranked, 1):
        row2 = next(r for r in rows if r["exp_name"] == c["exp_name"])
        w.writerow({"rank": i, **row2})
print(f"Wrote: {rank_path}")

# ── Warnings ─────────────────────────────────────────────────────────────────
warnings = []
for name, c in CONDITIONS.items():
    pa_s  = _m(c, "pairwise_acc_score")
    pa_l  = _m(c, "pairwise_acc_lpips")
    fmse  = _m(c, "full_mse")
    ccmse = _m(c, "copy_current_mse")
    fm_m  = _m(c, "fuser_mask_mean")
    dm_m  = _m(c, "dynamic_mask_mean")
    stage = c["stage"]

    if stage == "v4b" and pa_s is None:
        warnings.append(f"- **{name}**: `pairwise_acc_score` is N/A (scorer may not have run)")
    if pa_s is not None and pa_s <= 0.5:
        warnings.append(f"- **{name}**: `pairwise_acc_score`={pa_s:.3f} ≤ 0.5 (scorer not improving)")
    if pa_l is not None and pa_l <= 0.5:
        warnings.append(f"- **{name}**: `pairwise_acc_lpips`={pa_l:.3f} ≤ 0.5 (worse than random)")
    if fmse is not None and ccmse is not None and ccmse > 1e-9:
        ratio = fmse / ccmse
        if ratio > 1.05:
            warnings.append(f"- **{name}**: `full_mse/copy_current_mse`={ratio:.3f} > 1 (copy-current collapse risk)")
    if fm_m is not None and fm_m < 0.01:
        warnings.append(f"- **{name}**: `fuser_mask_mean`={fm_m:.4f} < 0.01 (mask collapse?)")
    if dm_m is not None and dm_m < 0.01:
        warnings.append(f"- **{name}**: `dynamic_mask_mean`={dm_m:.4f} < 0.01 (dynamic mask collapse?)")

# ── Markdown ─────────────────────────────────────────────────────────────────
cond_names = list(CONDITIONS.keys())

def _table(conds_subset, title=""):
    if not conds_subset:
        return f"*No {title} conditions.*\n"
    names = list(conds_subset.keys())
    hdr = "| Metric | " + " | ".join(names) + " |"
    div = "| --- | " + " | ".join(["---"] * len(names)) + " |"
    rows_md = []
    for col_label, col_key, better in METRIC_COLS:
        cells = [f"`{col_label}`"]
        for nm in names:
            c = conds_subset[nm]
            v = _m(c, col_key)
            s = sf(v)
            if _is_best(v, col_key, better, conds_subset):
                s = f"**{s}**"
            cells.append(s)
        rows_md.append("| " + " | ".join(cells) + " |")
    return "\n".join([hdr, div] + rows_md)

v4a_conds = OrderedDict((k, v) for k, v in CONDITIONS.items() if v["stage"] == "v4a")
v4b_conds = OrderedDict((k, v) for k, v in CONDITIONS.items() if v["stage"] == "v4b")

# Ranking tables
def _top_n_by(key, better, n=5):
    reverse = (better == "higher")
    scored = [(name, _m(c, key)) for name, c in CONDITIONS.items() if _m(c, key) is not None]
    scored.sort(key=lambda x: x[1], reverse=reverse)
    return scored[:n]

# Best candidates
best_pa_score = _top_n_by("pairwise_acc_score", "higher", 5)
best_pa_lpips = _top_n_by("pairwise_acc_lpips", "higher", 5)
best_sg       = _top_n_by("score_gap_mean",     "higher", 5)

def _fmt_top(lst):
    if not lst:
        return "*No data.*"
    return "\n".join(f"{i+1}. `{name}` = `{sf(v)}`" for i, (name, v) in enumerate(lst))

# RFT candidates (v4b with pairwise_acc_score > 0.5)
rft_candidates = [
    name for name, c in CONDITIONS.items()
    if c["stage"] == "v4b"
    and (_m(c, "pairwise_acc_score") or 0) > 0.5
]
rft_cands_str = "\n".join(f"- `{n}`" for n in rft_candidates) or "- *None qualify yet (pairwise_acc_score ≤ 0.5)*"

warnings_str = "\n".join(warnings) if warnings else "_No warnings._"

md = f"""# Phase 1 v4 Core Sweep — Results

**Run**: `{RUN_NAME}`
**Out root**: `{OUT_ROOT}`
**Conditions**: {len(CONDITIONS)} ({len(v4a_conds)} v4a, {len(v4b_conds)} v4b)

---

## 1. Experiment Conditions

| exp_name | stage | K | Q | motion | scorer | λ_rank | neg_type |
| --- | --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(
    f"| `{c['exp_name']}` | {c['stage']} | {c['history_k']} | {c['num_q']} | {c['motion_bias']} | {c['scorer']} | {c['lambda_rank']} | {c['neg_type']} |"
    for c in CONDITIONS.values()
) + """

---

## 2. v4a Comparison (no ranking head)

### Full metrics

""" + _table(v4a_conds, "v4a") + """

---

## 3. v4b Comparison (with ActionFutureScorer)

### Full metrics

""" + _table(v4b_conds, "v4b") + f"""

---

## 4. LPIPS-Based Ranking — Top 5

> Higher `pairwise_acc_lpips` = GT action gives cleaner reconstruction than shuffled.

{_fmt_top(best_pa_lpips)}

---

## 5. Score-Based Ranking — Top 5

> Higher `pairwise_acc_score` = ActionFutureScorer distinguishes GT vs shuffled actions (v4b only).

{_fmt_top(best_pa_score)}

**score_gap_mean top 5:**

{_fmt_top(best_sg)}

---

## 6. RFT Candidates (pairwise_acc_score > 0.5, v4b)

{rft_cands_str}

To run RFT sweep:
```
RUN_NAME={RUN_NAME} \\
WORLD_REWARD_TYPE_LIST="visual hybrid rank_score" \\
  bash scripts/libero/phase1/run_v4_selected_rft_sweep.sh
```

---

## 7. Warnings

{warnings_str}

---

## 8. Files

| File | Description |
| --- | --- |
| `v4_core_sweep_summary.csv` | All conditions × all metrics |
| `v4_core_sweep_ranking.csv` | Conditions ranked by pairwise_acc_score |
| `v4_core_sweep_summary.json` | Machine-readable summary |
| `v4_core_sweep_summary.md` | This file |

---

*Generated by `summarize_v4_core_sweep.sh`*
"""

md_path = SUM_DIR / "v4_core_sweep_summary.md"
md_path.write_text(md)
print(f"Wrote: {md_path}")

# Console
print("\n=== v4 Core Sweep Summary ===")
print(f"  {'exp_name':<38} {'stage':<4} {'pa_score':>10}  {'score_gap':>10}  {'pa_lpips':>10}  {'full_mse':>10}")
print(f"  {'-'*38} {'-'*4} {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
for name, c in CONDITIONS.items():
    print(f"  {name:<38} {c['stage']:<4} "
          f"{sf(_m(c,'pairwise_acc_score')):>10}  "
          f"{sf(_m(c,'score_gap_mean')):>10}  "
          f"{sf(_m(c,'pairwise_acc_lpips')):>10}  "
          f"{sf(_m(c,'full_mse')):>10}")

if warnings:
    print(f"\n  ⚠  {len(warnings)} warning(s) — see {md_path}")
PYEOF

log "Summary complete."
ls -la "${SUMMARY_DIR}"/ 2>/dev/null || true
