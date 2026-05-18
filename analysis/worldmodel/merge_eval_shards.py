#!/usr/bin/env python3
"""Merge per-shard evaluation results from multi-GPU evaluation.

Usage:
    python analysis/worldmodel/merge_eval_shards.py \
        --output-dir results/baseline_ar_pixel_wm/spatial \
        --num-shards 2

Reads shard_N_of_M/rows.jsonl from each shard, merges all rows, then
re-runs aggregation and writes final results to --output-dir.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from worldmodel.dynquery.utils import aggregate_phase1_metrics


def _row_vals(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if isinstance(r.get(key), float) and not np.isnan(r[key])]


def merge(output_dir: Path, num_shards: int) -> None:
    rows: list[dict] = []
    missing: list[int] = []

    for i in range(num_shards):
        shard_dir = output_dir / f"shard_{i}_of_{num_shards}"
        jsonl_path = shard_dir / "rows.jsonl"
        if not jsonl_path.exists():
            missing.append(i)
            print(f"[WARN] Missing: {jsonl_path}")
            continue
        shard_rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        print(f"  shard {i}: {len(shard_rows)} windows")
        rows.extend(shard_rows)

    if missing:
        print(f"[ERROR] {len(missing)} shard(s) missing: {missing}. Aborting merge.", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("[ERROR] No rows found across all shards.", file=sys.stderr)
        sys.exit(1)

    # Sort by global_window_id to restore original order
    rows.sort(key=lambda r: int(r.get("global_window_id", r.get("window_id", 0)) or 0))
    print(f"Merged {len(rows)} windows total. Re-aggregating ...")

    # Re-run the same aggregation logic as the main eval script
    metrics = aggregate_phase1_metrics(rows, str(output_dir), "phase0_ar_pixel_direct_eval")

    # Reward stability
    _rft_proxies = _row_vals(rows, "rft_reward_proxy")
    _rft_gaps    = _row_vals(rows, "rft_reward_gap")
    rft_proxy_std   = float(np.std(_rft_proxies))  if _rft_proxies else float("nan")
    rft_proxy_range = float(np.ptp(_rft_proxies))  if _rft_proxies else float("nan")
    rft_gap_std     = float(np.std(_rft_gaps))     if _rft_gaps    else float("nan")

    # Read metadata from any shard's aggregate_metrics.json for passthrough fields
    passthrough: dict = {}
    for i in range(num_shards):
        src = output_dir / f"shard_{i}_of_{num_shards}" / "aggregate_metrics.json"
        if src.exists():
            src_data = json.loads(src.read_text())
            passthrough = src_data.get("metrics", {})
            break

    metrics.update({
        "model_name": "phase0_ar_pixel_direct_eval",
        "model_family": "phase0_ar_pixel",
        "model_generation": "phase0_ar_pixel",
        "target_mode": "ar_pixel",
        "metric_source": "direct_eval_on_phase1_manifest",
        "window_manifest_hash": passthrough.get("window_manifest_hash", ""),
        "phase0_compatible": True,
        "negative_eval_types": passthrough.get("negative_eval_types", []),
        "default_negative_type": passthrough.get("default_negative_type", "mixed"),
        "temporal_shift_max": passthrough.get("temporal_shift_max", 3),
        "action_noise_std": passthrough.get("action_noise_std", 0.15),
        "rft_reward_gap_mean": metrics.get("rft_reward_gap", float("nan")),
        "rft_reward_proxy_std":   rft_proxy_std,
        "rft_reward_proxy_range": rft_proxy_range,
        "rft_reward_gap_std":     rft_gap_std,
        "pairwise_acc_score":        float("nan"),
        "score_gap_mean":            float("nan"),
        "score_gap_min":             float("nan"),
        "pearson_rft_score_corr":    float("nan"),
        "spearman_rft_score_corr":   float("nan"),
        "dynamic_mask_iou_gt_mean":       float("nan"),
        "dynamic_mask_precision_gt_mean": float("nan"),
        "dynamic_mask_recall_gt_mean":    float("nan"),
    })

    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps({"condition": "phase0_ar_pixel_direct_eval", "metrics": metrics}, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    # Merge rows.jsonl
    (output_dir / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, allow_nan=True) for r in rows) + "\n",
        encoding="utf-8",
    )

    print(f"Merge complete → {output_dir}/aggregate_metrics.json  (num_windows={metrics.get('num_windows')})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()
    merge(Path(args.output_dir), args.num_shards)


if __name__ == "__main__":
    main()
