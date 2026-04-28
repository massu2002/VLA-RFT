#!/usr/bin/env python3
"""Export a manual-labelling CSV template for failure taxonomy analysis.

Reads ranking JSONL + metrics.json from a worldmodel evaluation run and
generates a CSV template pre-filled with heuristic suggested_error_type labels.

Error type taxonomy
-------------------
  grasp_miss            Gripper ROI error >> full-image error; model cannot predict contact.
  touch_no_lift         Gripper ROI degrades over horizon despite low full-image error.
  transport_drift       Full-image error increases steadily over multi-step horizon.
  place_misalignment    High goal ROI error; object delivered to wrong location.
  false_progress        Ranking score high (model is confident) but reconstruction poor.
  ranking_failure       GT action ranks below a negative candidate (pairwise_acc = 0).
  wm_static_bias        Score variance across candidates is very low; model ignores action.
  wm_arm_only_bias      Gripper ROI much lower than goal ROI (focus only on arm, not scene).
  other                 Does not match any of the above patterns.

Usage
-----
  python -m worldmodel.libero.export_failure_taxonomy_template \\
      --jsonl  rank_eval/rank_eval_candidates.jsonl \\
      --metrics eval_reports/libero/spatial/metrics.json \\
      --output  taxonomy_template.csv

  # With multiple JSONL / metrics sources:
  python -m worldmodel.libero.export_failure_taxonomy_template \\
      --sweep-dir checkpoints/libero/FocusedWM/sweep/baseline_v3/ \\
      --output    taxonomy_template.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Error type heuristic thresholds (tune as needed for your experiment)
# ---------------------------------------------------------------------------

_T_GRASP_MISS       = 1.5   # gripper_mse / full_mse ratio
_T_STATIC_BIAS      = 0.02  # max score − min score across K candidates
_T_ARM_ONLY         = 1.5   # goal_mse / gripper_mse ratio
_T_PLACE_MISALIGN   = 1.4   # goal_mse / full_mse ratio
_T_TRANSPORT_DRIFT  = 1.3   # last-frame error / first-frame error ratio
_T_CONFIDENT_POOR   = 0.5   # top score - bottom score (large gap = confident)


def _read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not path.exists():
        return records
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Heuristic error type classifier
# ---------------------------------------------------------------------------

def _suggest_error_type(
    item_scores: List[float],
    metrics: Dict,
    step_metrics: Optional[Dict] = None,
) -> str:
    """Return one error-type label for a single ranked item.

    Priority order: ranking_failure → wm_static_bias → specific ROI pattern → other.
    """
    K = len(item_scores)
    if K == 0:
        return "other"

    pos_score = item_scores[0]
    neg_max   = max(item_scores[1:]) if K > 1 else pos_score - 1.0
    score_range = max(item_scores) - min(item_scores) if K > 1 else 0.0

    # 1. Static bias (highest priority): model assigns nearly identical scores to ALL candidates.
    #    This is more informative than "ranking_failure" when scores are degenerate.
    if score_range < _T_STATIC_BIAS:
        return "wm_static_bias"

    # 2. Ranking failure: GT not ranked first (and scores are distinguishable)
    if neg_max >= pos_score:
        return "ranking_failure"

    # ROI-based patterns (only if metrics are available)
    full_mse      = metrics.get("future_image_smooth_l1") or metrics.get("mse")
    gripper_mse   = metrics.get("roi/gripper_mse")
    goal_mse      = metrics.get("roi/goal_mse")
    gripper_lpips = metrics.get("roi/gripper_lpips")
    goal_lpips    = metrics.get("roi/goal_lpips")

    if full_mse and full_mse > 0:
        # 3. Gripper miss
        if gripper_mse and gripper_mse / full_mse > _T_GRASP_MISS:
            return "grasp_miss"
        # 4. Goal misalignment
        if goal_mse and goal_mse / full_mse > _T_PLACE_MISALIGN:
            return "place_misalignment"

    # 5. Arm-only bias: gripper well-predicted but goal region poorly predicted
    if gripper_mse and goal_mse and gripper_mse > 0:
        if goal_mse / gripper_mse > _T_ARM_ONLY:
            return "wm_arm_only_bias"

    # 6. Transport drift: check multi-step if available
    ms_gripper = metrics.get("roi/multi_step_gripper_mse")
    if isinstance(ms_gripper, list) and len(ms_gripper) >= 2:
        if ms_gripper[-1] / (ms_gripper[0] + 1e-8) > _T_TRANSPORT_DRIFT:
            return "transport_drift"

    # 7. High confidence + poor reconstruction
    if score_range > _T_CONFIDENT_POOR and full_mse and full_mse > 0.05:
        return "false_progress"

    return "other"


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _build_rows(
    jsonl_records: List[Dict],
    metrics: Dict,
    task: str,
    suite: str,
    run_name: str,
) -> List[Dict]:
    rows: List[Dict] = []

    for rec in jsonl_records:
        step = rec.get("step", -1)
        for item in rec.get("per_item", []):
            item_id = item.get("item_id", -1)
            scores  = item.get("scores", [])
            pos_rank = sorted(scores, reverse=True).index(scores[0]) if scores else -1

            suggested = _suggest_error_type(scores, metrics)

            rows.append({
                "task":               task,
                "suite":              suite,
                "run_name":           run_name,
                "episode_id":         "",
                "window_id":          item_id,
                "step_t":             step,
                "success":            "",
                "positive_rank":      pos_rank,
                "chosen_id":          0,
                "pred_path":          "",
                "gt_path":            "",
                "suggested_error_type": suggested,
                "manual_error_type":  "",
                "notes":              "",
            })

    return rows


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _find_jsonl_and_metrics(sweep_dir: Path):
    """Yield (jsonl_path, metrics_path, task_suite, run_name) from a sweep dir."""
    for jsonl in sorted(sweep_dir.rglob("rank_eval/rank_eval_candidates.jsonl")):
        run_dir    = jsonl.parent.parent
        run_name   = run_dir.name
        task_suite = "unknown"
        for part in run_dir.parts:
            if part in ("spatial", "object", "goal", "long"):
                task_suite = part
                break

        # find corresponding full-eval metrics.json
        metrics_candidates = sorted(run_dir.rglob("eval_reports/**/metrics.json"))
        metrics_path = metrics_candidates[0] if metrics_candidates else None

        yield jsonl, metrics_path, task_suite, run_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_OUTPUT_COLS = [
    "task", "suite", "run_name",
    "episode_id", "window_id", "step_t",
    "success", "positive_rank", "chosen_id",
    "pred_path", "gt_path",
    "suggested_error_type", "manual_error_type", "notes",
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl",     type=str, help="Single JSONL ranking file.")
    src.add_argument("--sweep-dir", type=str, help="Root sweep directory (auto-discover).")

    parser.add_argument("--metrics", type=str, default="",
                        help="Path to metrics.json (used with --jsonl).")
    parser.add_argument("--task",    type=str, default="unknown")
    parser.add_argument("--suite",   type=str, default="unknown")
    parser.add_argument("--run-name",type=str, default="")
    parser.add_argument("--output",  type=str, default="taxonomy_template.csv")
    args = parser.parse_args(argv)

    all_rows: List[Dict] = []

    if args.jsonl:
        jsonl_path   = Path(args.jsonl)
        metrics_path = Path(args.metrics) if args.metrics else None
        metrics      = _read_json(metrics_path) if metrics_path else {}
        records      = _read_jsonl(jsonl_path)
        run_name     = args.run_name or jsonl_path.parent.parent.name
        rows = _build_rows(records, metrics, args.task, args.suite, run_name)
        all_rows.extend(rows)
    else:
        sweep = Path(args.sweep_dir)
        for jsonl, met_p, suite, run_name in _find_jsonl_and_metrics(sweep):
            metrics = _read_json(met_p) if met_p else {}
            records = _read_jsonl(jsonl)
            rows    = _build_rows(records, metrics, "", suite, run_name)
            all_rows.extend(rows)

    if not all_rows:
        print("No rows generated. Check --jsonl / --sweep-dir paths.", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_OUTPUT_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows → {out}")
    # Error type distribution
    from collections import Counter
    dist = Counter(r["suggested_error_type"] for r in all_rows)
    for typ, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {typ:<30} {cnt:>5} ({100*cnt/len(all_rows):.1f}%)")


if __name__ == "__main__":
    main()
