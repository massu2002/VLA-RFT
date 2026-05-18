#!/usr/bin/env python3
"""Average aggregate_metrics.json across multiple seeds.

Usage:
    python analysis/worldmodel/average_seed_metrics.py \
        --output-dir results/baseline_ar_pixel_wm/spatial \
        --seeds 42,43,44

Reads:
    output_dir/seed_{N}/aggregate_metrics.json  for each seed N

Writes:
    output_dir/aggregate_metrics_multiseed.json
        metrics keys: mean values
        {key}_std keys: standard deviations
        {key}_seeds: per-seed values list
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("metrics", data)


def average(output_dir: Path, seeds: list[int]) -> dict:
    per_seed: list[dict] = []
    missing: list[int] = []

    for seed in seeds:
        p = output_dir / f"seed_{seed}" / "aggregate_metrics.json"
        if not p.exists():
            missing.append(seed)
            print(f"[WARN] Missing: {p}")
            continue
        per_seed.append(load_metrics(p))
        print(f"  seed {seed}: loaded ({p})")

    if missing:
        print(f"[ERROR] {len(missing)} seed(s) missing: {missing}", file=sys.stderr)
        sys.exit(1)

    if not per_seed:
        print("[ERROR] No seed results found.", file=sys.stderr)
        sys.exit(1)

    # Collect all numeric keys from the first seed
    all_keys = [k for k, v in per_seed[0].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)]

    out: dict = {}

    for key in all_keys:
        vals = []
        for m in per_seed:
            v = m.get(key)
            if v is not None and isinstance(v, (int, float)) and not math.isnan(float(v)):
                vals.append(float(v))

        if vals:
            mean = float(np.mean(vals))
            std  = float(np.std(vals))
            out[key]            = mean
            out[f"{key}_std"]   = std
            out[f"{key}_seeds"] = vals
        else:
            out[key]            = float("nan")
            out[f"{key}_std"]   = float("nan")
            out[f"{key}_seeds"] = []

    # Passthrough non-numeric fields from first seed
    for key, val in per_seed[0].items():
        if key not in out and not isinstance(val, (int, float)):
            out[key] = val

    out["num_seeds"]    = len(per_seed)
    out["seeds"]        = seeds
    out["num_windows"]  = int(sum(m.get("num_windows", 0) for m in per_seed))

    result = {
        "condition": per_seed[0].get("condition", "phase0_ar_pixel_direct_eval"),
        "num_seeds": len(per_seed),
        "seeds": seeds,
        "metrics": out,
    }

    dest = output_dir / "aggregate_metrics_multiseed.json"
    dest.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Multi-seed metrics written → {dest}  (seeds={seeds}, total_windows={out['num_windows']})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seeds", required=True, help="comma-separated seed list, e.g. 42,43,44")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    average(Path(args.output_dir), seeds)


if __name__ == "__main__":
    main()
