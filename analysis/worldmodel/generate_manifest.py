#!/usr/bin/env python3
"""Generate a standalone evaluation window manifest from RLDS data.

Creates a window_manifest.json compatible with
analysis/evaluate_phase0_ar_pixel_on_manifest.py, without requiring
a pre-existing Phase1 DynQuery evaluation run.

Usage (from repo root):
    python scripts/generate_standalone_manifest.py \\
        --task-suite spatial \\
        --data-root data/modified_libero_rlds \\
        --output results/baseline_ar_pixel_wm/spatial/window_manifest.json \\
        --eval-horizon 8 \\
        --num-windows 200 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow_datasets as tfds

from worldmodel.datasets.libero.data import resolve_dataset_name


def _decode(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if hasattr(x, "numpy"):
        v = x.numpy()
        return v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else str(v)
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-suite", default="spatial")
    ap.add_argument("--data-root", default="data/modified_libero_rlds")
    ap.add_argument("--output", required=True)
    ap.add_argument("--eval-horizon", type=int, default=8)
    ap.add_argument("--num-windows", type=int, default=200)
    ap.add_argument("--windows-per-task", type=int, default=0,
                    help="Override per-task window count (0 = num_windows / num_tasks)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-episode-length", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    dataset_name = resolve_dataset_name(args.task_suite)
    print(f"Loading {dataset_name} from {args.data_root} ...")
    ds = tfds.load(dataset_name, data_dir=args.data_root, split="train", shuffle_files=False)

    # --- Enumerate episodes (no image loading — cardinality only) ---
    print("Enumerating episodes (fast path: no image loading) ...")
    task_episodes: dict[str, list[dict]] = {}

    for idx, ep in enumerate(ds):
        path = _decode(ep["episode_metadata"]["file_path"])
        length = int(ep["steps"].cardinality().numpy())
        basename = os.path.basename(path)
        ep_id = os.path.splitext(basename)[0]
        task_key = ep_id[:-5] if ep_id.endswith("_demo") else ep_id

        min_len = max(args.min_episode_length, args.eval_horizon + 2)
        if length < min_len:
            continue

        task_episodes.setdefault(task_key, []).append({
            "episode_index": idx,
            "episode_file": path,
            "episode_id": ep_id,
            "episode_length": length,
        })
        if (idx + 1) % 50 == 0:
            print(f"  ... {idx + 1} episodes, {len(task_episodes)} tasks")

    n_tasks = len(task_episodes)
    total_eps = sum(len(v) for v in task_episodes.values())
    print(f"Found {total_eps} valid episodes across {n_tasks} tasks")

    # --- Stable task ordering (alphabetical) ---
    sorted_keys = sorted(task_episodes.keys())
    task_id_map = {k: i for i, k in enumerate(sorted_keys)}

    # --- Windows per task ---
    wins_per_task = args.windows_per_task if args.windows_per_task > 0 else max(1, args.num_windows // n_tasks)
    print(f"Windows per task: {wins_per_task}  (total target: {wins_per_task * n_tasks})")

    # --- Phase sampling parameters ---
    PHASES = ["early", "middle", "late"]
    PHASE_CENTER = {"early": 0.18, "middle": 0.50, "late": 0.82}
    PHASE_WIDTH  = {"early": 0.16, "middle": 0.20, "late": 0.16}

    windows = []
    global_id = 0

    for task_key in sorted_keys:
        task_id = task_id_map[task_key]
        task_name = task_key.replace("_", " ")
        eps = task_episodes[task_key]

        per_phase = [wins_per_task // 3] * 3
        per_phase[0] += wins_per_task - sum(per_phase)

        local_id = 0
        for phase_idx, phase in enumerate(PHASES):
            c = PHASE_CENTER[phase]
            w = PHASE_WIDTH[phase]
            H = args.eval_horizon

            for _ in range(per_phase[phase_idx]):
                ep_info = rng.choice(eps)
                L = ep_info["episode_length"]

                lo = max(0, int((c - w / 2) * L))
                hi = min(L - H - 1, int((c + w / 2) * L))
                if lo > hi:
                    lo, hi = 0, max(0, L - H - 1)
                if lo > hi:
                    continue

                current = rng.randint(lo, hi)
                windows.append({
                    "global_window_id": global_id,
                    "local_window_id": local_id,
                    "task_id": task_id,
                    "task_name": task_name,
                    "episode_file": ep_info["episode_file"],
                    "episode_index": ep_info["episode_index"],
                    "episode_id": ep_info["episode_id"],
                    "episode_length": L,
                    "window_position": phase,
                    "current_frame_index": current,
                    "history_frame_indices": [],
                    "future_frame_indices": list(range(current + 1, current + H + 1)),
                    "action_indices": list(range(current, current + H)),
                    "history_length": 0,
                })
                global_id += 1
                local_id += 1

        print(f"  Task {task_id:2d}  '{task_name[:55]:55s}'  {local_id} windows")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": {
            "model_generation": "standalone",
            "target_mode": "ar_pixel",
            "task_suite": args.task_suite,
            "eval_horizon": args.eval_horizon,
            "window_position_mode": "phase_sampling",
            "num_eval_windows": len(windows),
            "seed": args.seed,
            "windows_per_task": wins_per_task,
            "negative_eval_types": ["same_phase", "temporal_shift", "action_noise", "mixed"],
            "default_negative_type": "mixed",
            "temporal_shift_max": 3,
            "action_noise_std": 0.15,
        },
        "num_windows": len(windows),
        "windows": windows,
        "skipped_history": [],
    }

    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(windows)} windows → {out_path}")


if __name__ == "__main__":
    main()
