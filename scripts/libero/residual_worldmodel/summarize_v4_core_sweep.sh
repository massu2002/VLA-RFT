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
        if v is None:
            return None
        out = float(v)
        return None if math.isnan(out) else out
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

def _read_csv_rows(path):
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open(newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"[warn] could not read {p}: {e}")
        return []

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
        "phase_rows":  _read_csv_rows(d / "metrics_by_phase.csv"),
        "task_phase_rows": _read_csv_rows(d / "metrics_by_task_by_phase.csv"),
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

NEGATIVE_TYPES = []
for c in CONDITIONS.values():
    for k in c["metrics"]:
        if k.startswith("pairwise_acc_rft_"):
            neg = k[len("pairwise_acc_rft_"):]
            if neg and neg not in NEGATIVE_TYPES:
                NEGATIVE_TYPES.append(neg)
NEG_ORDER = ["same_phase", "temporal_shift", "action_noise", "mixed"]
NEGATIVE_TYPES.sort(key=lambda x: NEG_ORDER.index(x) if x in NEG_ORDER else 99)
if not NEGATIVE_TYPES:
    NEGATIVE_TYPES = ["mixed"]

PHASES = []
for c in CONDITIONS.values():
    for r in c.get("phase_rows", []):
        phase = r.get("window_phase", "")
        if phase and phase not in PHASES:
            PHASES.append(phase)
PHASE_ORDER = ["early", "middle", "later"]
PHASES.sort(key=lambda x: PHASE_ORDER.index(x) if x in PHASE_ORDER else 99)

# ── Metric key definitions ────────────────────────────────────────────────────
METRIC_COLS = [
    # Reconstruction quality (horizon-averaged, RFT-aligned)
    ("horizon_avg_lpips",          "horizon_avg_lpips",        "lower"),
    ("horizon_avg_mae",            "horizon_avg_mae",          "lower"),
    ("horizon_avg_mse",            "horizon_avg_mse",          "lower"),
    ("rft_reward_proxy",           "rft_reward_proxy",         "higher"),
    ("copy_current_horizon_avg_mse","copy_current_horizon_avg_mse", "lower"),
    ("horizon_mse_over_copy",      "horizon_mse_over_copy",        "lower"),
    # Ranking accuracy
    ("pairwise_acc_rft",           "pairwise_acc_rft",         "higher"),
    ("pairwise_acc_score",         "pairwise_acc_score",       "higher"),
    # Ranking gaps
    ("rft_reward_gap_mean",        "rft_reward_gap_mean",      "higher"),
    ("rft_reward_gap_min",         "rft_reward_gap_min",       "higher"),
    ("score_gap_mean",             "score_gap_mean",           "higher"),
    ("score_gap_min",              "score_gap_min",            "higher"),
    # Model internal structure
    ("fuser_mask_entropy",         "fuser_mask_entropy",       "neutral"),
    ("fuser_mask_overlap",         "fuser_mask_overlap",       "neutral"),
    ("dynamic_mask_entropy",       "dynamic_mask_entropy",     "neutral"),
    ("dynamic_mask_overlap",       "dynamic_mask_overlap",     "neutral"),
    ("future_dynamic_query_norm",  "future_dynamic_query_norm","neutral"),
    # Operational
    ("skipped_history_windows",    "skipped_history_windows",  "lower"),
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

# horizon_mse_over_copy is already stored in aggregate_metrics.json by the eval script.

# ── CSV ───────────────────────────────────────────────────────────────────────
CSV_STATIC_COLS = [
    "exp_name", "stage", "history_length", "num_dynamic_queries",
    "use_motion_bias", "use_action_future_scorer", "lambda_rank", "rank_margin",
    "negative_type", "negative_mix", "window_position_mode", "negative_eval_types",
    "default_negative_type",
]
CSV_METRIC_KEYS = [
    # Reconstruction quality
    "horizon_avg_lpips", "horizon_avg_mae", "horizon_avg_mse",
    "rft_reward_proxy", "copy_current_horizon_avg_mse", "horizon_mse_over_copy",
    # Ranking
    "pairwise_acc_rft", "pairwise_acc_score",
    "rft_reward_gap_mean", "rft_reward_gap_min",
    "score_gap_mean", "score_gap_min",
    # Model internals
    "fuser_mask_entropy", "fuser_mask_overlap",
    "dynamic_mask_entropy", "dynamic_mask_overlap",
    "future_dynamic_query_norm",
    # Operational
    "skipped_history_windows",
]
NEGATIVE_METRIC_KEYS = []
for neg in NEGATIVE_TYPES:
    NEGATIVE_METRIC_KEYS.extend([
        f"pairwise_acc_rft_{neg}",
        f"pairwise_acc_score_{neg}",
        f"rft_reward_gap_mean_{neg}",
        f"rft_reward_gap_min_{neg}",
        f"score_gap_mean_{neg}",
        f"score_gap_min_{neg}",
    ])
CSV_METRIC_KEYS = CSV_METRIC_KEYS + [k for k in NEGATIVE_METRIC_KEYS if k not in CSV_METRIC_KEYS]

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
        "window_position_mode":  c["metrics"].get("window_position_mode", ""),
        "negative_eval_types":   ",".join(c["metrics"].get("negative_eval_types", []) or []),
        "default_negative_type": c["metrics"].get("default_negative_type", ""),
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

# ── Phase-level CSV ──────────────────────────────────────────────────────────
phase_metric_keys = [
    "horizon_avg_lpips", "horizon_avg_mae", "horizon_avg_mse",
    "horizon_mse_over_copy", "pairwise_acc_rft", "pairwise_acc_score",
    "rft_reward_gap", "score_gap",
]
for neg in NEGATIVE_TYPES:
    phase_metric_keys.extend([
        f"pairwise_acc_rft_{neg}",
        f"pairwise_acc_score_{neg}",
        f"rft_reward_gap_mean_{neg}",
        f"score_gap_mean_{neg}",
    ])
phase_metric_keys = list(dict.fromkeys(phase_metric_keys))

phase_rows_out = []
for c in CONDITIONS.values():
    for pr in c.get("phase_rows", []):
        out = {
            "exp_name": c["exp_name"],
            "stage": c["stage"],
            "window_phase": pr.get("window_phase", ""),
        }
        for k in phase_metric_keys:
            out[k] = sf(_fv(pr.get(k)))
        phase_rows_out.append(out)

phase_path = SUM_DIR / "v4_core_sweep_phase_summary.csv"
if phase_rows_out:
    with open(phase_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exp_name", "stage", "window_phase"] + phase_metric_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(phase_rows_out)
    print(f"Wrote: {phase_path}")

# ── Negative-type ranking CSV ────────────────────────────────────────────────
neg_rank_rows = []
for c in CONDITIONS.values():
    for neg in NEGATIVE_TYPES:
        neg_rank_rows.append({
            "exp_name": c["exp_name"],
            "stage": c["stage"],
            "negative_type": neg,
            "pairwise_acc_rft": sf(_m(c, f"pairwise_acc_rft_{neg}")),
            "pairwise_acc_score": sf(_m(c, f"pairwise_acc_score_{neg}")),
            "rft_reward_gap_mean": sf(_m(c, f"rft_reward_gap_mean_{neg}")),
            "score_gap_mean": sf(_m(c, f"score_gap_mean_{neg}")),
            "horizon_mse_over_copy": sf(_m(c, "horizon_mse_over_copy")),
        })

def _neg_rank_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")

neg_rank_rows.sort(key=lambda r: (
    r["negative_type"],
    -(_neg_rank_float(r, "pairwise_acc_score") if not math.isnan(_neg_rank_float(r, "pairwise_acc_score")) else -999),
    -(_neg_rank_float(r, "pairwise_acc_rft") if not math.isnan(_neg_rank_float(r, "pairwise_acc_rft")) else -999),
))
neg_rank_path = SUM_DIR / "v4_core_sweep_negative_ranking.csv"
with open(neg_rank_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "exp_name", "stage", "negative_type", "pairwise_acc_rft", "pairwise_acc_score",
        "rft_reward_gap_mean", "score_gap_mean", "horizon_mse_over_copy",
    ])
    w.writeheader()
    w.writerows(neg_rank_rows)
print(f"Wrote: {neg_rank_path}")

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
            "window_position_mode": c["metrics"].get("window_position_mode"),
            "negative_eval_types": c["metrics"].get("negative_eval_types"),
            "default_negative_type": c["metrics"].get("default_negative_type"),
            "metrics": {
                **{k: _m(c, mk) for k, mk, _ in METRIC_COLS},
                **{k: _m(c, k) for k in NEGATIVE_METRIC_KEYS},
                "negative_type_counts": c["metrics"].get("negative_type_counts"),
            },
            "phase_metrics": c.get("phase_rows", []),
        }
        for name, c in CONDITIONS.items()
    }
}
json_path = SUM_DIR / "v4_core_sweep_summary.json"
json_path.write_text(json.dumps(summary, indent=2, default=str))
print(f"Wrote: {json_path}")

# ── Ranking CSV (sorted by pairwise_acc_score desc, fallback lpips asc) ───────
def _rank_key(c):
    # _m returns float(nan) for NaN values; treat nan same as None (use sentinel).
    def _v(key):
        v = _m(c, key)
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
    default_neg = c["metrics"].get("default_negative_type", "mixed")
    pa_s   = _v(f"pairwise_acc_score_{default_neg}") or _v("pairwise_acc_score")
    pa_rft = _v(f"pairwise_acc_rft_{default_neg}") or _v("pairwise_acc_rft")
    sg     = _v(f"score_gap_mean_{default_neg}") or _v("score_gap_mean")
    mse    = _v("horizon_avg_mse")
    return (
        -(pa_s   if pa_s   is not None else -999),
        -(pa_rft if pa_rft is not None else -999),
        -(sg     if sg     is not None else -999),
        mse if mse is not None else 999,
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
    default_neg = c["metrics"].get("default_negative_type", "mixed")
    pa_s   = _m(c, f"pairwise_acc_score_{default_neg}") or _m(c, "pairwise_acc_score")
    pa_rft = _m(c, f"pairwise_acc_rft_{default_neg}") or _m(c, "pairwise_acc_rft")
    ratio  = _m(c, "horizon_mse_over_copy")
    stage  = c["stage"]

    if stage == "v4b" and pa_s is None:
        warnings.append(f"- **{name}**: `pairwise_acc_score` is N/A (scorer may not have run)")
    if pa_s is not None and pa_s <= 0.5:
        warnings.append(f"- **{name}**: `pairwise_acc_score`={pa_s:.3f} ≤ 0.5 (scorer not improving)")
    if pa_rft is None:
        warnings.append(f"- **{name}**: `pairwise_acc_rft` is N/A (run eval with updated script)")
    elif pa_rft <= 0.5:
        warnings.append(f"- **{name}**: `pairwise_acc_rft`={pa_rft:.3f} ≤ 0.5 (RFT reward not distinguishing actions)")
    if ratio is not None and ratio > 1.05:
        warnings.append(f"- **{name}**: `horizon_mse_over_copy`={ratio:.3f} > 1 (model worse than copy-current baseline)")
    for neg in NEGATIVE_TYPES:
        pa_rft_neg = _m(c, f"pairwise_acc_rft_{neg}")
        pa_score_neg = _m(c, f"pairwise_acc_score_{neg}")
        if pa_rft_neg is not None and pa_rft_neg <= 0.5:
            warnings.append(f"- **{name}**: `pairwise_acc_rft_{neg}`={pa_rft_neg:.3f} ≤ 0.5")
        if stage == "v4b" and pa_score_neg is not None and pa_score_neg <= 0.5:
            warnings.append(f"- **{name}**: `pairwise_acc_score_{neg}`={pa_score_neg:.3f} ≤ 0.5")

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
best_pa_score  = _top_n_by("pairwise_acc_score",  "higher", 5)
best_sg        = _top_n_by("score_gap_mean",      "higher", 5)
best_pa_rft    = _top_n_by("pairwise_acc_rft",    "higher", 5)
best_rft_gap   = _top_n_by("rft_reward_gap_mean", "higher", 5)

def _fmt_top(lst):
    if not lst:
        return "*No data.*"
    return "\n".join(f"{i+1}. `{name}` = `{sf(v)}`" for i, (name, v) in enumerate(lst))

def _negative_table(conds_subset):
    if not conds_subset:
        return "*No conditions.*"
    hdr = "| exp_name | neg | pa_rft | pa_score | rft_gap | score_gap |"
    div = "| --- | --- | ---: | ---: | ---: | ---: |"
    rows = []
    for name, c in conds_subset.items():
        for neg in NEGATIVE_TYPES:
            rows.append(
                f"| `{name}` | `{neg}` | {sf(_m(c, f'pairwise_acc_rft_{neg}'))} | "
                f"{sf(_m(c, f'pairwise_acc_score_{neg}'))} | "
                f"{sf(_m(c, f'rft_reward_gap_mean_{neg}'))} | "
                f"{sf(_m(c, f'score_gap_mean_{neg}'))} |"
            )
    return "\n".join([hdr, div] + rows)

def _phase_table(conds_subset):
    if not conds_subset:
        return "*No phase metrics.*"
    hdr = "| exp_name | phase | pa_rft | pa_score | rft_gap | mse/copy | n |"
    div = "| --- | --- | ---: | ---: | ---: | ---: | ---: |"
    rows = []
    for name, c in conds_subset.items():
        for pr in c.get("phase_rows", []):
            phase = pr.get("window_phase", "")
            rows.append(
                f"| `{name}` | `{phase}` | {sf(_fv(pr.get('pairwise_acc_rft')))} | "
                f"{sf(_fv(pr.get('pairwise_acc_score')))} | "
                f"{sf(_fv(pr.get('rft_reward_gap')))} | "
                f"{sf(_fv(pr.get('horizon_mse_over_copy')))} | "
                f"{pr.get('num_windows', '')} |"
            )
    return "\n".join([hdr, div] + rows) if rows else "*No phase metrics.*"

# RFT candidates (mixed/default pairwise_acc_rft > 0.5, or v4b score > 0.5)
rft_candidates = [
    name for name, c in CONDITIONS.items()
    if (_m(c, f"pairwise_acc_rft_{c['metrics'].get('default_negative_type', 'mixed')}") or _m(c, "pairwise_acc_rft") or 0) > 0.5
    or (
        c["stage"] == "v4b"
        and (_m(c, f"pairwise_acc_score_{c['metrics'].get('default_negative_type', 'mixed')}") or _m(c, "pairwise_acc_score") or 0) > 0.5
    )
]
rft_cands_str = "\n".join(f"- `{n}`" for n in rft_candidates) or "- *None qualify yet (pairwise_acc_rft ≤ 0.5)*"

warnings_str = "\n".join(warnings) if warnings else "_No warnings._"

md = f"""# Phase 1 v4 Core Sweep — Results

**Run**: `{RUN_NAME}`
**Out root**: `{OUT_ROOT}`
**Conditions**: {len(CONDITIONS)} ({len(v4a_conds)} v4a, {len(v4b_conds)} v4b)
**Window phases**: `{", ".join(PHASES) if PHASES else "N/A"}`
**Negative eval types**: `{", ".join(NEGATIVE_TYPES)}`

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

## 4. Score-Based Ranking — Top 5

> Higher `pairwise_acc_score` = ActionFutureScorer distinguishes GT vs negative actions (v4b only). The legacy column follows `default_negative_type` (usually `mixed`).

{_fmt_top(best_pa_score)}

**score_gap_mean top 5:**

{_fmt_top(best_sg)}

---

## 5. Negative-Type Breakdown

> `same_phase`: same task and same early/middle/later phase; `temporal_shift`: shifted GT action sequence; `action_noise`: perturbed GT actions; `mixed`: random mixture.

### v4a

{_negative_table(v4a_conds)}

### v4b

{_negative_table(v4b_conds)}

---

## 6. Phase Breakdown

> Each phase row is aggregated from windows sampled from that episode region. Task-level CSV metrics are phase-balanced means.

### v4a

{_phase_table(v4a_conds)}

### v4b

{_phase_table(v4b_conds)}

---

## 7. RFT Reward Ranking — Top 5

> `pairwise_acc_rft`: correct action yields higher RFT reward proxy (= -(horizon_avg_lpips + horizon_avg_mae)) than shuffled.
> Directly matches `phase1_residual_reward` used in VLA-RFT training.

{_fmt_top(best_pa_rft)}

**rft_reward_gap_mean top 5:**

{_fmt_top(best_rft_gap)}

---

## 8. RFT Candidates (default negative pairwise_acc_rft > 0.5)

{rft_cands_str}

To run RFT sweep:
```
RUN_NAME={RUN_NAME} \\
WORLD_REWARD_TYPE_LIST="visual hybrid rank_score" \\
  bash scripts/libero/phase1/run_v4_selected_rft_sweep.sh
```

---

## 9. Warnings

{warnings_str}

---

## 10. Files

| File | Description |
| --- | --- |
| `v4_core_sweep_summary.csv` | All conditions × all metrics |
| `v4_core_sweep_ranking.csv` | Conditions ranked by pairwise_acc_score |
| `v4_core_sweep_phase_summary.csv` | Conditions × early/middle/later metrics |
| `v4_core_sweep_negative_ranking.csv` | Conditions ranked within each negative type |
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
print(f"  {'exp_name':<38} {'stage':<4} {'pa_score':>10}  {'pa_rft':>10}  {'rft_gap':>10}  {'mse/copy':>10}")
print(f"  {'-'*38} {'-'*4} {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
for name, c in CONDITIONS.items():
    print(f"  {name:<38} {c['stage']:<4} "
          f"{sf(_m(c,'pairwise_acc_score')):>10}  "
          f"{sf(_m(c,'pairwise_acc_rft')):>10}  "
          f"{sf(_m(c,'rft_reward_gap_mean')):>10}  "
          f"{sf(_m(c,'horizon_mse_over_copy')):>10}")

if warnings:
    print(f"\n  ⚠  {len(warnings)} warning(s) — see {md_path}")
PYEOF

log "Summary complete."
ls -la "${SUMMARY_DIR}"/ 2>/dev/null || true
