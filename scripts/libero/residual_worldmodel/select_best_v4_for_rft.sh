#!/usr/bin/env bash
# select_best_v4_for_rft.sh — Select best v4 checkpoint(s) for RFT.
#
# Scans ${OUT_ROOT}/<exp_name>/aggregate_metrics.json for each condition,
# ranks them by BEST_CRITERION, deploys (symlink/copy) the best into
# ${SELECTED_DIR}/ and writes selected_v4_for_rft.json.
#
# Usage:
#   RUN_NAME=v4_core_sweep_spatial \
#   BEST_CRITERION=hybrid_score \
#     bash scripts/libero/residual_worldmodel/select_best_v4_for_rft.sh
#
# BEST_CRITERION choices:
#   pairwise_acc_score   — ActionFutureScorer ranking accuracy (v4b primary)
#   score_gap_mean       — mean (score_pos − score_neg) gap
#   pairwise_acc_lpips   — LPIPS-based ranking accuracy
#   middle_pairwise_score — average of pairwise_acc_score and pairwise_acc_lpips
#   hybrid_score         — 0.4*pa_score + 0.2*pa_lpips + 0.2*norm_gap - 0.1*rev_ratio - 0.1*copy_penalty
#
# Key env vars:
#   RUN_NAME, OUT_ROOT, SELECTED_DIR
#   BEST_CRITERION   (default: hybrid_score)
#   COPY_MODE        0=symlink (default), 1=full copy
#   DRY_RUN          0|1
#   LIST_ONLY        0|1 — only print table, do not deploy
#   TOP_N            select top-N conditions (default: 1)
#   STAGE_FILTER     v4a | v4b | all (default: all)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

RUN_NAME="${RUN_NAME:?'RUN_NAME required'}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/phase1/residual_worldmodel/${RUN_NAME}}"
SELECTED_DIR="${SELECTED_DIR:-${OUT_ROOT}/selected_for_rft}"
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/libero/phase1/v4_core_sweep.json}"
BEST_CRITERION="${BEST_CRITERION:-hybrid_score}"
COPY_MODE="${COPY_MODE:-0}"
DRY_RUN="${DRY_RUN:-0}"
LIST_ONLY="${LIST_ONLY:-0}"
TOP_N="${TOP_N:-1}"
STAGE_FILTER="${STAGE_FILTER:-all}"   # all | v4a | v4b

[ -d "${OUT_ROOT}" ] || { echo "[select-best-v4] OUT_ROOT not found: ${OUT_ROOT}" >&2; exit 1; }

setup_env

print_header "select_best_v4_for_rft: ${RUN_NAME}"
printf "  %-28s %s\n" "OUT_ROOT"       "${OUT_ROOT}"
printf "  %-28s %s\n" "SELECTED_DIR"   "${SELECTED_DIR}"
printf "  %-28s %s\n" "BEST_CRITERION" "${BEST_CRITERION}"
printf "  %-28s %s\n" "TOP_N"          "${TOP_N}"
printf "  %-28s %s\n" "STAGE_FILTER"   "${STAGE_FILTER}"
printf "  %-28s %s\n" "COPY_MODE"      "${COPY_MODE}"
printf "  %-28s %s\n" "DRY_RUN"        "${DRY_RUN}"
printf "  %-28s %s\n" "LIST_ONLY"      "${LIST_ONLY}"
printf '%*s\n' 68 '' | tr ' ' '-'

export OUT_ROOT SELECTED_DIR SWEEP_CONFIG BEST_CRITERION COPY_MODE DRY_RUN LIST_ONLY TOP_N STAGE_FILTER RUN_NAME

python3 - <<'PYEOF'
import json, math, os, sys, shutil
from pathlib import Path
from typing import Optional

OUT_ROOT      = Path(os.environ["OUT_ROOT"])
SELECTED_DIR  = Path(os.environ["SELECTED_DIR"])
CFG_PATH      = Path(os.environ.get("SWEEP_CONFIG", ""))
CRITERION     = os.environ.get("BEST_CRITERION", "hybrid_score")
COPY_MODE     = os.environ.get("COPY_MODE", "0") not in ("0", "false")
DRY_RUN       = os.environ.get("DRY_RUN", "0") not in ("0", "false")
LIST_ONLY     = os.environ.get("LIST_ONLY", "0") not in ("0", "false")
TOP_N         = int(os.environ.get("TOP_N", "1"))
STAGE_FILTER  = os.environ.get("STAGE_FILTER", "all")
RUN_NAME      = os.environ.get("RUN_NAME", OUT_ROOT.name)

def fv(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None

def sf(v, nd=5):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)

# ── Load experiment metadata from sweep config ────────────────────────────────
exp_cfg_by_name = {}
if CFG_PATH.exists():
    try:
        raw = json.loads(CFG_PATH.read_text())
        common = raw.get("common", {})
        for ex in raw.get("experiments", []):
            name = ex.get("exp_name", "")
            exp_cfg_by_name[name] = {**common, **ex}
    except Exception as e:
        print(f"[warn] sweep config load failed: {e}")

# ── Scan condition dirs ───────────────────────────────────────────────────────
conditions = []
for d in sorted(OUT_ROOT.iterdir()):
    if not d.is_dir() or d.name in ("summary", "logs", "selected_for_rft"):
        continue
    agg_path = d / "aggregate_metrics.json"
    cfg_path = d / "v4_sweep_config_used.json"
    if not agg_path.exists() and not cfg_path.exists():
        continue

    metrics = {}
    if agg_path.exists():
        try:
            raw = json.loads(agg_path.read_text())
            metrics = raw.get("metrics", raw)
        except Exception:
            pass

    exp_cfg = {}
    if cfg_path.exists():
        try:
            exp_cfg = json.loads(cfg_path.read_text())
        except Exception:
            pass
    if d.name in exp_cfg_by_name:
        for k, v in exp_cfg_by_name[d.name].items():
            exp_cfg.setdefault(k, v)

    stage = exp_cfg.get("stage", "v4b")
    if STAGE_FILTER != "all" and stage != STAGE_FILTER:
        continue

    ckpt_dir = exp_cfg.get("ckpt_dir", "")
    conditions.append({
        "exp_name":  d.name,
        "stage":     stage,
        "metrics":   metrics,
        "exp_cfg":   exp_cfg,
        "ckpt_dir":  ckpt_dir,
        "cond_out":  str(d),
    })

if not conditions:
    print(f"[warn] No v4 conditions found under {OUT_ROOT}")
    sys.exit(0)

print(f"Found {len(conditions)} condition(s).")

# ── Scoring functions ─────────────────────────────────────────────────────────
def _get(c, key):
    return fv(c["metrics"].get(key))

def _norm_across(conds, key, higher_is_better=True):
    """Return a dict {exp_name: normalized [0,1]} value."""
    vals = {c["exp_name"]: _get(c, key) for c in conds}
    valid = {k: v for k, v in vals.items() if v is not None}
    if not valid:
        return {c["exp_name"]: None for c in conds}
    lo, hi = min(valid.values()), max(valid.values())
    span = hi - lo
    result = {}
    for name in [c["exp_name"] for c in conds]:
        v = vals.get(name)
        if v is None:
            result[name] = None
        elif span < 1e-9:
            result[name] = 1.0
        else:
            n = (v - lo) / span
            result[name] = n if higher_is_better else (1.0 - n)
    return result

def compute_score(c, conds, criterion):
    pa_s  = _get(c, "pairwise_acc_score")
    pa_l  = _get(c, "pairwise_acc_lpips")
    sg    = _get(c, "score_gap_mean")
    rev_s = _get(c, "reverse_windows_score")
    fmse  = _get(c, "full_mse")
    ccmse = _get(c, "copy_current_mse")

    if criterion == "pairwise_acc_score":
        return pa_s if pa_s is not None else (pa_l if pa_l is not None else -999.0)

    if criterion == "score_gap_mean":
        return sg if sg is not None else -999.0

    if criterion == "pairwise_acc_lpips":
        return pa_l if pa_l is not None else -999.0

    if criterion == "middle_pairwise_score":
        vals = [v for v in [pa_s, pa_l] if v is not None]
        return sum(vals) / len(vals) if vals else -999.0

    if criterion == "hybrid_score":
        # Normalised components
        norm_pa_s  = _norm_across(conds, "pairwise_acc_score",  True)
        norm_pa_l  = _norm_across(conds, "pairwise_acc_lpips",  True)
        norm_sg    = _norm_across(conds, "score_gap_mean",      True)
        norm_rev_s = _norm_across(conds, "reverse_windows_score", False)   # lower is better
        norm_copy  = _norm_across(conds, "copy_current_mse",    True)      # lower is better → higher norm = safer

        name = c["exp_name"]
        def _safe(d):
            v = d.get(name)
            return v if v is not None else 0.5  # neutral fallback

        score = (0.4 * _safe(norm_pa_s)
               + 0.2 * _safe(norm_pa_l)
               + 0.2 * _safe(norm_sg)
               - 0.1 * (1.0 - _safe(norm_rev_s))   # penalise high reverse rate
               - 0.1 * (1.0 - _safe(norm_copy)))    # penalise copy-current collapse
        return score

    # fallback: pairwise_acc_score
    return pa_s if pa_s is not None else -999.0

# ── Rank ─────────────────────────────────────────────────────────────────────
for c in conditions:
    c["score"] = compute_score(c, conditions, CRITERION)

conditions.sort(key=lambda c: c["score"], reverse=True)

# ── Print table ───────────────────────────────────────────────────────────────
print()
print(f"  Criterion: {CRITERION}")
print()
print(f"  {'#':<3}  {'exp_name':<38}  {'stage':<4}  {'score':>10}  {'pa_score':>10}  {'pa_lpips':>10}  {'sg_mean':>10}")
print(f"  {'-'*3}  {'-'*38}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
for i, c in enumerate(conditions, 1):
    marker = "★" if i <= TOP_N else " "
    print(f"  {marker}{i:<2}  {c['exp_name']:<38}  {c['stage']:<4}  "
          f"{sf(c['score']):>10}  "
          f"{sf(_get(c,'pairwise_acc_score')):>10}  "
          f"{sf(_get(c,'pairwise_acc_lpips')):>10}  "
          f"{sf(_get(c,'score_gap_mean')):>10}")

selected = conditions[:TOP_N]
print()
print(f"  Selected top-{TOP_N}:")
for c in selected:
    print(f"    {c['exp_name']}  ckpt={c['ckpt_dir']}")

if LIST_ONLY:
    print()
    print("  [LIST_ONLY] Done.")
    sys.exit(0)

# ── Deploy ────────────────────────────────────────────────────────────────────
SELECTED_DIR.mkdir(parents=True, exist_ok=True)

report = []
for c in selected:
    name     = c["exp_name"]
    ckpt_src = Path(c["ckpt_dir"])
    dest     = SELECTED_DIR / name

    # Per-criterion text file
    criterion_file = SELECTED_DIR / f"best_by_{CRITERION}.txt"
    if not DRY_RUN:
        criterion_file.write_text(
            f"exp_name: {name}\n"
            f"criterion: {CRITERION}\n"
            f"score: {c['score']}\n"
            f"ckpt_dir: {c['ckpt_dir']}\n"
            f"pairwise_acc_score: {sf(_get(c,'pairwise_acc_score'))}\n"
            f"score_gap_mean: {sf(_get(c,'score_gap_mean'))}\n"
            f"pairwise_acc_lpips: {sf(_get(c,'pairwise_acc_lpips'))}\n"
        )

    print(f"  [{name}]")
    print(f"    ckpt : {ckpt_src}")
    print(f"    dest : {dest}")
    print(f"    score: {CRITERION} = {sf(c['score'])}")

    if DRY_RUN:
        print(f"    [DRY_RUN]")
    elif not ckpt_src.exists():
        print(f"    [SKIP] ckpt_dir does not exist: {ckpt_src}")
    elif dest.exists() or dest.is_symlink():
        print(f"    [SKIP] dest already exists")
    else:
        if COPY_MODE:
            shutil.copytree(str(ckpt_src), str(dest), symlinks=True)
            print(f"    [OK] copied")
        else:
            try:
                rel = os.path.relpath(str(ckpt_src), str(SELECTED_DIR))
            except ValueError:
                rel = str(ckpt_src)
            dest.symlink_to(rel)
            print(f"    [OK] symlink → {rel}")

    report.append({
        "exp_name":          name,
        "criterion":         CRITERION,
        "score":             c["score"],
        "stage":             c["stage"],
        "ckpt_dir":          str(ckpt_src),
        "config_path":       str(ckpt_src / "v4_config.json"),
        "cond_out":          c["cond_out"],
        "metrics": {
            "pairwise_acc_score": _get(c, "pairwise_acc_score"),
            "pairwise_acc_lpips": _get(c, "pairwise_acc_lpips"),
            "score_gap_mean":     _get(c, "score_gap_mean"),
            "score_gap_min":      _get(c, "score_gap_min"),
            "reverse_windows_score": _get(c, "reverse_windows_score"),
            "full_mse":           _get(c, "full_mse"),
            "gripper_mse":        _get(c, "gripper_mse"),
            "copy_current_mse":   _get(c, "copy_current_mse"),
            "fuser_mask_mean":    _get(c, "fuser_mask_mean"),
            "dynamic_mask_mean":  _get(c, "dynamic_mask_mean"),
        },
    })

if not DRY_RUN:
    # Write per-criterion files for other common criteria too
    for crit in ["pairwise_acc_score", "score_gap_mean", "pairwise_acc_lpips",
                 "middle_pairwise_score", "hybrid_score"]:
        if crit == CRITERION:
            continue
        crit_scores = [(c["exp_name"], compute_score(c, conditions, crit)) for c in conditions]
        crit_scores.sort(key=lambda x: x[1], reverse=True)
        best_name, best_val = crit_scores[0]
        p = SELECTED_DIR / f"best_by_{crit}.txt"
        p.write_text(f"exp_name: {best_name}\ncriterion: {crit}\nscore: {best_val}\n")

    sel_path = SELECTED_DIR / "selected_v4_for_rft.json"
    sel_path.write_text(json.dumps({
        "run_name":  RUN_NAME,
        "criterion": CRITERION,
        "selected":  report,
    }, indent=2, default=str))
    print(f"\n  Selection report: {sel_path}")

if DRY_RUN:
    print("  [DRY_RUN] Nothing written.")
else:
    print(f"\n  Deployed {len(selected)} checkpoint(s) to {SELECTED_DIR}")
PYEOF
