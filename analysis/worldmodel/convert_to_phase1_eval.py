#!/usr/bin/env python3
"""Convert Phase 0 AR-Pixel WM eval outputs into the Phase 1 eval schema.

This is intentionally a converter, not a new AR-Pixel inference path.  The
Phase 0 AR-token WorldModel keeps its native evaluator; this script makes its
outputs comparable in Phase 1 tables without pretending it is a
PixelResidualWorldModel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def stat_mean(block: dict[str, Any], key: str) -> float:
    v = block.get(key, {})
    if isinstance(v, dict):
        return float(v.get("mean", math.nan))
    return float(v) if v is not None else math.nan


def find_reports(root: Path, label: str) -> list[Path]:
    reports = sorted(root.glob(f"**/eval_report__{label}__*.json"))
    if not reports and label == "trained":
        reports = sorted(root.glob("**/eval_report__trained__*.json"))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0-results", required=True, help="Phase0 result root or worldmodel subdir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", default="trained", choices=["trained", "base"])
    parser.add_argument("--task-suite", default="spatial")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--phase0-compatible", action="store_true", default=True)
    args = parser.parse_args()

    root = Path(args.phase0_results)
    wm_root = root / "worldmodel" if (root / "worldmodel").exists() else root
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reports = find_reports(wm_root, args.label)
    task_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []

    for report_path in reports:
        report = load_json(report_path)
        cfg = report.get("config", {})
        task_indices = cfg.get("selected_task_indices") or []
        task_id = int(task_indices[0]) if task_indices else len(task_rows)
        task_name = f"task{task_id}"
        per_task_path = report.get("per_task_summary_path")
        if per_task_path:
            per_task = load_json(Path(per_task_path)).get("per_task", {}).get(str(task_id), {})
            task_name = per_task.get("task_name", task_name)

        rollout = report.get("rollout_fidelity", {}).get("overall", {})
        action = report.get("action_sensitivity", {})
        row = {
            "model_name": "phase0_ar_pixel_converted",
            "model_family": "phase0_ar_pixel",
            "model_generation": "phase0_ar_pixel_converted",
            "target_mode": "ar_pixel",
            "metric_source": "converted_phase0_result",
            "window_manifest_hash": "",
            "task_index": task_id,
            "task_name": task_name,
            "full_mse": stat_mean(rollout, "mse"),
            "full_lpips": stat_mean(rollout, "lpips"),
            "gripper_mse": stat_mean(rollout, "roi/gripper_mse"),
            "gripper_lpips": stat_mean(rollout, "roi/gripper_lpips"),
            "goal_mse": stat_mean(rollout, "roi/goal_mse"),
            "goal_lpips": stat_mean(rollout, "roi/goal_lpips"),
            "dynamic_mse": math.nan,
            "dynamic_lpips": math.nan,
            "copy_current_full_mse": math.nan,
            "copy_current_full_lpips": math.nan,
            "pairwise_acc": float(action.get("per_window_pairwise_acc", math.nan)),
            "correct_lpips": stat_mean(action, "correct_lpips"),
            "shuffled_lpips": stat_mean(action, "shuffled_lpips"),
            "lpips_gap": stat_mean(action, "lpips_gap"),
            "lpips_gap_min": float(action.get("lpips_gap", {}).get("min", math.nan)) if isinstance(action.get("lpips_gap"), dict) else math.nan,
            "reverse_windows": int(action.get("num_losses", 0)),
            "num_windows": int(action.get("num_windows", report.get("rollout_fidelity", {}).get("num_windows", 0))),
        }
        task_rows.append(row)

        for item in action.get("per_window_records", []):
            ranking_rows.append({
                "task_name": task_name,
                "task_index": task_id,
                "window_id": item.get("case_id"),
                "window_phase": "phase0",
                "episode_length": "",
                "episode_file": "",
                "frame_indices": "",
                "action_indices": "",
                "correct_lpips": item.get("correct_lpips"),
                "shuffled_lpips": item.get("shuffled_lpips"),
                "lpips_gap": item.get("lpips_gap"),
                "pairwise_win": bool(item.get("win", 0)),
            })

    def mean_key(key: str) -> float:
        xs = [float(r[key]) for r in task_rows if r.get(key) is not None and not math.isnan(float(r[key]))]
        return float(np.mean(xs)) if xs else math.nan

    wins = sum(1 for r in ranking_rows if r.get("pairwise_win"))
    n_rank = len(ranking_rows)
    gaps = [float(r["lpips_gap"]) for r in ranking_rows if r.get("lpips_gap") is not None]
    metrics = {
        "model_name": "phase0_ar_pixel_converted",
        "model_family": "phase0_ar_pixel",
        "model_generation": "phase0_ar_pixel_converted",
        "target_mode": "ar_pixel",
        "metric_source": "converted_phase0_result",
        "window_manifest_hash": "",
        "full_mse": mean_key("full_mse"),
        "full_lpips": mean_key("full_lpips"),
        "gripper_mse": mean_key("gripper_mse"),
        "gripper_lpips": mean_key("gripper_lpips"),
        "goal_mse": mean_key("goal_mse"),
        "goal_lpips": mean_key("goal_lpips"),
        "dynamic_mse": math.nan,
        "dynamic_lpips": math.nan,
        "copy_current_full_mse": math.nan,
        "copy_current_full_lpips": math.nan,
        "pairwise_acc": wins / max(n_rank, 1),
        "correct_lpips": mean_key("correct_lpips"),
        "shuffled_lpips": mean_key("shuffled_lpips"),
        "lpips_gap": float(np.mean(gaps)) if gaps else mean_key("lpips_gap"),
        "lpips_gap_min": float(np.min(gaps)) if gaps else mean_key("lpips_gap_min"),
        "reverse_windows": n_rank - wins,
        "num_windows": int(sum(r.get("num_windows", 0) for r in task_rows)),
        "num_ranking_windows": n_rank,
        "phase0_compatible": bool(args.phase0_compatible),
        "metric_source": "converted_phase0_result",
        "window_manifest_hash": "",
    }
    (out / "aggregate_metrics.json").write_text(json.dumps({"condition": "phase0_ar_pixel_converted", "metrics": metrics}, indent=2, allow_nan=True), encoding="utf-8")

    if task_rows:
        with (out / "metrics_by_task.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
            w.writeheader()
            w.writerows(task_rows)
        with (out / "ranking_by_task.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["task_name", "task_index", "correct_lpips", "shuffled_lpips", "lpips_gap", "lpips_gap_min", "pairwise_acc", "reverse_windows", "num_windows"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in task_rows:
                w.writerow({k: r.get(k, "") for k in fields})
    if ranking_rows:
        with (out / "ranking_by_window.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ranking_rows[0].keys()))
            w.writeheader()
            w.writerows(ranking_rows)

    protocol = {
        "task_suite": args.task_suite,
        "selected_task_indices": sorted({r["task_index"] for r in task_rows}),
        "num_eval_windows": metrics["num_windows"],
        "eval_horizon": 7,
        "segment_length": 9,
        "action_start_offset": 0,
        "current_frame_index": "phase0 native evaluator",
        "future_frame_index": "phase0 native evaluator",
        "use_terminal_frame_only": False,
        "negative_type": "same_task_shuffle/native_phase0",
        "same_task_shuffle": True,
        "shuffle_seed": None,
        "window_seed": None,
        "lpips_input_range": "[-1,1]",
        "image_range_before_lpips": "phase0 native evaluator",
        "ranking_gap_definition": "lpips_gap = shuffled_lpips - correct_lpips",
        "pairwise_unit": "window",
        "roi_crop_size": "phase0 native ROI",
        "gripper_roi_method": "phase0 native motion ROI",
        "goal_roi_method": "phase0 native goal ROI",
        "rollout_mode": "phase0 AR-Pixel native evaluator",
        "decode_chunk_size": 2,
        "checkpoint_path": args.checkpoint_path,
        "tokenizer_path": args.tokenizer_path,
        "target_mode": "ar_pixel",
        "model_generation": "phase0_ar_pixel_converted",
        "model_family": "phase0_ar_pixel",
        "phase0_compatible": bool(args.phase0_compatible),
    }
    (out / "eval_protocol_config.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "config_used.json").write_text(json.dumps({"condition": "phase0_ar_pixel_converted", "eval_protocol": protocol}, indent=2), encoding="utf-8")
    print(f"Wrote Phase0 AR-Pixel converted eval to {out}")


if __name__ == "__main__":
    main()
