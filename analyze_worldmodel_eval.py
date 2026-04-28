#!/usr/bin/env python3
"""Aggregate WorldModel evaluation metrics from checkpoint or sweep directories.

Discovers and merges:
  rank_eval/rank_eval_metrics.json      -- latest tiered ranking metrics per run
  rank_eval/rank_eval_candidates.jsonl  -- full step-history of ranking metrics
  eval_reports/**/metrics.json          -- full-eval image reconstruction metrics

Outputs (written to --output-dir, default: same as --sweep-dir or --checkpoint-dir):
  worldmodel_eval_summary.csv           -- one row per (run, task_suite)
  worldmodel_eval_summary.json          -- same as JSON
  worldmodel_eval_ranking_history.csv   -- JSONL history flattened (all steps x runs)

Compatible with both:
  checkpoints/libero/FocusedWM/**       (residual worldmodel)
  checkpoints/libero/WorldModel/**      (baseline worldmodel)

Usage examples
--------------
  # Summarise a single run
  python analyze_worldmodel_eval.py \\
      --checkpoint-dir checkpoints/libero/FocusedWM/baseline/spatial/s42/run_name/

  # Summarise a whole sweep
  python analyze_worldmodel_eval.py \\
      --sweep-dir checkpoints/libero/FocusedWM/sweep/baseline_v3/

  # Quick subset smoke-test
  python analyze_worldmodel_eval.py \\
      --sweep-dir checkpoints/libero/FocusedWM/sweep/baseline_v3/ \\
      --protocol configs/libero/eval_protocol_v1.yaml \\
      --output-dir /tmp/wm_analysis/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol config loader (optional — graceful fallback if yaml not installed)
# ---------------------------------------------------------------------------

def _load_protocol(protocol_path: Optional[str]) -> dict:
    if protocol_path is None:
        return {}
    p = Path(protocol_path)
    if not p.exists():
        logger.warning("Protocol file not found: %s", p)
        return {}
    try:
        import yaml  # type: ignore
        with p.open() as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        logger.warning("PyYAML not available; protocol config ignored.")
        return {}


# ---------------------------------------------------------------------------
# Run identity extraction from path
# ---------------------------------------------------------------------------

def _parse_run_identity(path: Path) -> Dict[str, str]:
    """Extract metadata from a checkpoint directory path.

    Handles two known layouts:
      .../FocusedWM/<variant>/<task_suite>/s<seed>/<run_name>/
      .../WorldModel/<task_suite>/
    """
    parts = list(path.parts)
    identity: Dict[str, str] = {
        "run_dir":    str(path),
        "model_type": "unknown",
        "variant":    "unknown",
        "task_suite": "unknown",
        "seed":       "unknown",
        "run_name":   path.name,
    }

    for i, part in enumerate(parts):
        if part == "FocusedWM":
            identity["model_type"] = "residual"
            if i + 1 < len(parts):
                identity["variant"]    = parts[i + 1]
            if i + 2 < len(parts):
                identity["task_suite"] = parts[i + 2]
            if i + 3 < len(parts) and parts[i + 3].startswith("s"):
                identity["seed"]       = parts[i + 3].lstrip("s")
            if i + 4 < len(parts):
                identity["run_name"]   = parts[i + 4]
            break
        if part == "WorldModel":
            identity["model_type"] = "baseline"
            if i + 1 < len(parts):
                identity["task_suite"] = parts[i + 1]
            break

    return identity


# ---------------------------------------------------------------------------
# Metric file discovery
# ---------------------------------------------------------------------------

def _find_rank_metrics_json(root: Path) -> List[Path]:
    return sorted(root.rglob("rank_eval/rank_eval_metrics.json"))


def _find_rank_jsonl(root: Path) -> List[Path]:
    return sorted(root.rglob("rank_eval/rank_eval_candidates.jsonl"))


def _find_full_eval_metrics(root: Path) -> List[Path]:
    return sorted(root.rglob("eval_reports/**/metrics.json"))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _read_rank_metrics_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        doc = json.loads(p.read_text())
        return {"step": doc.get("step", -1), **doc.get("metrics", {})}
    except Exception as exc:
        logger.warning("Could not parse %s: %s", p, exc)
        return None


def _read_full_eval_metrics(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        logger.warning("Could not parse %s: %s", p, exc)
        return None


def _read_rank_jsonl(p: Path) -> List[Dict[str, Any]]:
    """Return list of records from JSONL; one record per training step."""
    records = []
    try:
        with p.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.debug("JSONL parse error at %s line %d: %s", p, lineno, exc)
    except Exception as exc:
        logger.warning("Could not read %s: %s", p, exc)
    return records


# ---------------------------------------------------------------------------
# Aggregation — per-run summary row
# ---------------------------------------------------------------------------

def _expand_multistep(prefix: str, lst: List[Any]) -> Dict[str, Any]:
    """Expand a multi-step list to first/last/max/drift_ratio scalars."""
    out: Dict[str, Any] = {}
    if not isinstance(lst, list) or len(lst) == 0:
        return out
    fvals = [float(v) for v in lst if v is not None and v == v]  # drop None / NaN
    if not fvals:
        return out
    out[f"{prefix}_first"]       = fvals[0]
    out[f"{prefix}_last"]        = fvals[-1]
    out[f"{prefix}_max"]         = max(fvals)
    out[f"{prefix}_drift_ratio"] = fvals[-1] / (fvals[0] + 1e-8)
    return out


def _build_run_summary(run_dir: Path, identity: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Merge latest ranking metrics + latest full-eval metrics for one run."""
    row: Dict[str, Any] = dict(identity)

    rank_path = run_dir / "rank_eval" / "rank_eval_metrics.json"
    if rank_path.exists():
        rm = _read_rank_metrics_json(rank_path)
        if rm:
            row["rank_step"] = rm.pop("step", -1)
            # Derive metric_family from K stored in rank_eval_dataset.json if available
            ds_path = run_dir / "rank_eval" / "rank_eval_dataset.json"
            K = 2  # default: baseline pairwise
            if ds_path.exists():
                try:
                    K = json.loads(ds_path.read_text()).get("metadata", {}).get("K", 2)
                except Exception:
                    pass
            row["metric_family"] = "tiered" if K >= 4 else "pairwise"
            row["ranking_K"]     = K
            for k, v in rm.items():
                row[f"rank/{k}"] = v

    # full-eval metrics — pick the first discovered metrics.json under eval_reports
    for m_path in sorted(run_dir.rglob("eval_reports/**/metrics.json")):
        fm = _read_full_eval_metrics(m_path)
        if fm:
            # infer task_suite from path if possible
            suite = "unknown"
            for part in m_path.parts:
                if part in ("spatial", "object", "goal", "long"):
                    suite = part
                    break
            for k, v in fm.items():
                if isinstance(v, list):
                    # Expand multi-step lists (U5)
                    row.update(_expand_multistep(f"fulleval_{suite}/{k}", v))
                else:
                    row[f"fulleval_{suite}/{k}"] = v
            break  # take only the first / closest

    if len(row) <= len(identity):
        logger.debug("No metrics found under %s", run_dir)
        return None

    return row


# ---------------------------------------------------------------------------
# JSONL history table (one row per step per run)
# ---------------------------------------------------------------------------

def _build_history_rows(
    run_dir: Path,
    identity: Dict[str, str],
) -> List[Dict[str, Any]]:
    jsonl_path = run_dir / "rank_eval" / "rank_eval_candidates.jsonl"
    if not jsonl_path.exists():
        return []
    records = _read_rank_jsonl(jsonl_path)
    rows = []
    for rec in records:
        K = rec.get("K", 2)
        metric_family = "tiered" if K >= 4 else "pairwise"
        # task_suite from JSONL record (U4 fix); fall back to identity
        task_suite_from_rec = rec.get("task_suite", "") or identity.get("task_suite", "unknown")

        hist_row: Dict[str, Any] = {
            "step":          rec.get("step", -1),
            "model_type":    rec.get("model_type", identity.get("model_type", "")),
            "task_name":     rec.get("task_name", ""),
            "task_suite":    task_suite_from_rec,
            "n_items":       rec.get("n_items", -1),
            "K":             K,
            "metric_family": metric_family,
            "saved_at":      rec.get("saved_at", ""),
        }
        hist_row.update(identity)
        # Override task_suite with record value (more specific than path-inferred)
        hist_row["task_suite"] = task_suite_from_rec

        for k, v in rec.get("metrics", {}).items():
            if isinstance(v, list):
                hist_row.update(_expand_multistep(f"tiered/{k}", v))
            else:
                hist_row[f"tiered/{k}"] = v

        rows.append(hist_row)
    return rows


def _build_suite_breakdown(summary_rows: List[Dict[str, Any]]) -> None:
    """Print pairwise vs tiered metric averages, grouped by task_suite."""
    from collections import defaultdict

    PAIRWISE_KEYS = ["rank/pairwise_acc", "rank/mean_margin"]
    TIERED_KEYS   = [
        "rank/strict_order_acc",
        "rank/acc_success_gt_nearsuccess",
        "rank/spearman_tier_corr",
        "rank/margin_success_minus_failure",
    ]
    ROI_KEYS = ["fulleval_spatial/roi/gripper_mse", "fulleval_spatial/roi/goal_mse"]

    by_suite: Dict[str, List[Dict]] = defaultdict(list)
    for row in summary_rows:
        suite = row.get("task_suite", "unknown")
        by_suite[suite].append(row)

    if not by_suite:
        return

    print("\n===== Suite Breakdown =====")
    for suite in sorted(by_suite):
        rows = by_suite[suite]
        family_counts: Dict[str, int] = {}
        for r in rows:
            fam = r.get("metric_family", "unknown")
            family_counts[fam] = family_counts.get(fam, 0) + 1

        print(f"\n  Suite: {suite}  (n={len(rows)}, families={family_counts})")
        for keys, label in [
            (PAIRWISE_KEYS, "  [pairwise metrics]"),
            (TIERED_KEYS,   "  [tiered metrics]"),
            (ROI_KEYS,      "  [ROI metrics]"),
        ]:
            avail = [k for k in keys if any(k in r for r in rows)]
            if not avail:
                continue
            print(f"  {label}")
            for k in avail:
                vals = [r[k] for r in rows if isinstance(r.get(k), float)]
                if vals:
                    import statistics
                    mean = statistics.mean(vals)
                    print(f"    {k:<45} mean={mean:.4f}  n={len(vals)}")
    print("=" * 27 + "\n")


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------

def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        logger.warning("No rows to write to %s", path)
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d rows → %s", len(rows), path)


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Wrote → %s", path)


# ---------------------------------------------------------------------------
# Human-readable table print
# ---------------------------------------------------------------------------

_DISPLAY_KEYS = [
    "model_type", "variant", "task_suite", "seed", "run_name",
    "rank_step",
    # Keys from rank_eval_metrics.json metrics dict (no tiered/ prefix there)
    "rank/strict_order_acc",
    "rank/acc_success_gt_nearsuccess",
    "rank/spearman_tier_corr",
    "rank/margin_success_minus_failure",
    "fulleval_spatial/future_image_smooth_l1",
    "fulleval_spatial/roi/gripper_l1",
    "fulleval_spatial/roi/object_l1",
    "fulleval_spatial/roi/goal_l1",
]


def _print_summary_table(rows: List[Dict[str, Any]]) -> None:
    present_keys = [k for k in _DISPLAY_KEYS
                    if any(k in row for row in rows)]
    if not present_keys:
        logger.info("(no display keys found in summary rows)")
        return

    header = "  ".join(f"{k:<28}" for k in present_keys)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for row in rows:
        cells = []
        for k in present_keys:
            val = row.get(k, "")
            if isinstance(val, float):
                cells.append(f"{val:<28.4f}")
            else:
                cells.append(f"{str(val):<28}")
        print("  ".join(cells))
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--checkpoint-dir", metavar="DIR",
        help="Single training-run checkpoint directory to analyse.",
    )
    src_group.add_argument(
        "--sweep-dir", metavar="DIR",
        help="Root directory containing multiple run sub-directories.",
    )
    parser.add_argument(
        "--protocol", metavar="YAML",
        default=None,
        help="Path to eval_protocol_v1.yaml (optional; used for display key ordering).",
    )
    parser.add_argument(
        "--output-dir", metavar="DIR", default=None,
        help="Directory to write output files. Defaults to the source directory.",
    )
    parser.add_argument(
        "--no-history", action="store_true",
        help="Skip writing the JSONL ranking-history CSV (large files).",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress the printed summary table.",
    )
    args = parser.parse_args(argv)

    _load_protocol(args.protocol)  # future: use for display key ordering

    # Resolve roots
    if args.checkpoint_dir:
        roots = [Path(args.checkpoint_dir)]
    else:
        sweep_root = Path(args.sweep_dir)
        # Heuristic: a run dir contains at least one of these sentinel files/dirs
        _SENTINELS = {"config_dump.json", "train_metrics.jsonl", "rank_eval"}
        roots = []
        for candidate in sorted(sweep_root.rglob("config_dump.json")):
            roots.append(candidate.parent)
        if not roots:
            # Fallback: any directory that directly contains rank_eval/
            for candidate in sorted(sweep_root.rglob("rank_eval")):
                roots.append(candidate.parent)
        roots = sorted(set(roots))

    if not roots:
        logger.error("No run directories found. Check --checkpoint-dir / --sweep-dir.")
        sys.exit(1)

    logger.info("Found %d run director%s.", len(roots), "y" if len(roots) == 1 else "ies")

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.checkpoint_dir) if args.checkpoint_dir else Path(args.sweep_dir)
    )

    # Build summary rows
    summary_rows: List[Dict[str, Any]] = []
    for run_dir in roots:
        identity = _parse_run_identity(run_dir)
        row = _build_run_summary(run_dir, identity)
        if row:
            summary_rows.append(row)

    logger.info("Built summary for %d run(s).", len(summary_rows))

    # Print table
    if not args.quiet and summary_rows:
        _print_summary_table(summary_rows)
        _build_suite_breakdown(summary_rows)

    # Write summary CSV / JSON
    _write_csv(summary_rows, out_dir / "worldmodel_eval_summary.csv")
    _write_json(summary_rows, out_dir / "worldmodel_eval_summary.json")

    # Build ranking-history rows (from JSONL)
    if not args.no_history:
        history_rows: List[Dict[str, Any]] = []
        for run_dir in roots:
            identity = _parse_run_identity(run_dir)
            history_rows.extend(_build_history_rows(run_dir, identity))
        if history_rows:
            _write_csv(history_rows, out_dir / "worldmodel_eval_ranking_history.csv")
        else:
            logger.info("No JSONL ranking history found (run training with tiered eval to generate).")


if __name__ == "__main__":
    main()
