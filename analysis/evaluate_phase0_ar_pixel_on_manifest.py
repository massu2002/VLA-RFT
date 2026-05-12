#!/usr/bin/env python3
"""Evaluate Phase0 AR-Pixel world model on a Phase1 window manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.nn.functional as F

from worldmodel.datasets.libero.data import resolve_dataset_name
from worldmodel.eval_roi_utils import (
    compute_roi_metrics_np,
    get_goal_roi_center,
    get_roi_half,
    load_roi_config,
    motion_com_np,
)
from worldmodel.libero.visualize import (
    _load_model,
    _load_trained_model,
    compute_sequence_metrics_all,
    rollout_episode_single_pass,
)
from worldmodel.residual_worldmodel.pixel_residual_utils import aggregate_phase1_metrics, get_lpips_fn


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_bytes(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if hasattr(x, "decode"):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def episode_file_path(ep: dict) -> str:
    return decode_bytes(ep.get("episode_metadata", {}).get("file_path", ""))


def episode_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def build_episode_lookup(ds) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for idx, ep in enumerate(tfds.as_numpy(ds)):
        path = episode_file_path(ep)
        out[path] = ep
        out[os.path.basename(path)] = ep
        out[episode_id(path)] = ep
        out[str(idx)] = ep
    return out


def episode_arrays(ep: dict) -> tuple[np.ndarray, np.ndarray]:
    steps = list(ep["steps"])
    imgs = np.stack([s["observation"]["image"] for s in steps], axis=0).astype(np.uint8)
    acts = np.stack([s["action"] for s in steps], axis=0).astype(np.float32)
    return imgs, acts


def find_episode(lookup: dict[str, dict], rec: dict, prefix: str = "") -> dict:
    keys = [
        str(rec.get(f"{prefix}episode_index", "")),
        rec.get(f"{prefix}episode_file", ""),
        os.path.basename(str(rec.get(f"{prefix}episode_file", ""))),
        str(rec.get(f"{prefix}episode_id", "")),
    ]
    for key in keys:
        if key in lookup:
            return lookup[key]
    raise KeyError(f"Could not find episode for manifest row {rec.get('global_window_id')} prefix={prefix}")


def slice_phase0_window(ep: dict, current: int, horizon: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    imgs, acts = episode_arrays(ep)
    frames = [imgs[i] for i in range(current, current + horizon + 1)]
    actions = [acts[i] for i in range(current, current + horizon)]
    if len(frames) != horizon + 1 or len(actions) != horizon:
        raise ValueError(f"Bad phase0 slice current={current}, horizon={horizon}")
    return frames, actions


def masked_mse_np(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(np.float32)
    if m.sum() <= 0:
        return float("nan")
    diff = (a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0) ** 2
    return float((diff * m[:, :, None]).sum() / (m.sum() * a.shape[-1]))


def dynamic_mask_np(current: np.ndarray, future: np.ndarray, threshold: float, dilate: int) -> np.ndarray:
    diff = np.abs(future.astype(np.float32) / 255.0 - current.astype(np.float32) / 255.0).mean(axis=2)
    mask = (diff > threshold).astype(np.float32)
    if dilate > 1:
        t = torch.from_numpy(mask)[None, None]
        t = F.max_pool2d(t, kernel_size=dilate, stride=1, padding=dilate // 2)
        mask = t[0, 0].numpy()
    return mask


def horizon_avg_mae(pred_frames: list[np.ndarray], gt_frames: list[np.ndarray]) -> float:
    maes = [
        float(np.mean(np.abs(p.astype(np.float32) / 255.0 - g.astype(np.float32) / 255.0)))
        for p, g in zip(pred_frames, gt_frames)
    ]
    return float(np.mean(maes)) if maes else float("nan")


def lpips_np(lpips_fn, a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(lpips_fn(a, b))
    except Exception:
        return float("nan")


def dynamic_metrics(
    pred_frames: list[np.ndarray],
    gt_frames: list[np.ndarray],
    current: np.ndarray,
    lpips_fn,
    threshold: float,
    dilate: int,
) -> tuple[float, float, float]:
    mses, lpips_vals, static = [], [], []
    for pred, gt in zip(pred_frames, gt_frames):
        mask = dynamic_mask_np(current, gt, threshold, dilate)
        mses.append(masked_mse_np(pred, gt, mask))
        static.append(masked_mse_np(pred, current, 1.0 - mask))
        masked_pred = (pred.astype(np.float32) * mask[:, :, None] + current.astype(np.float32) * (1.0 - mask[:, :, None])).astype(np.uint8)
        masked_gt = (gt.astype(np.float32) * mask[:, :, None] + current.astype(np.float32) * (1.0 - mask[:, :, None])).astype(np.uint8)
        lpips_vals.append(lpips_np(lpips_fn, masked_pred, masked_gt))
    return float(np.nanmean(mses)), float(np.nanmean(lpips_vals)), float(np.nanmean(static))


def copy_metrics(
    gt_frames: list[np.ndarray],
    current: np.ndarray,
    lpips_model,
    device: torch.device,
    lpips_fn,
    threshold: float,
    dilate: int,
    roi_half: int,
    goal_center,
) -> dict:
    copies = [current for _ in gt_frames]
    seq = compute_sequence_metrics_all(gt_frames, copies, lpips_model, device)
    gripper_center = motion_com_np(current, gt_frames[-1])
    roi = compute_roi_metrics_np(copies, gt_frames, lpips_fn, gripper_center, goal_center, roi_half)
    dyn_mse, dyn_lpips, _ = dynamic_metrics(copies, gt_frames, current, lpips_fn, threshold, dilate)
    return {
        "copy_current_full_mse": float(seq["avg_mse"]),
        "copy_current_full_lpips": float(seq["avg_lpips"]),
        "copy_current_mse": float(seq["avg_mse"]),
        "copy_current_lpips": float(seq["avg_lpips"]),
        "copy_current_gripper_mse": roi.get("roi/gripper_mse", float("nan")),
        "copy_current_gripper_lpips": roi.get("roi/gripper_lpips", float("nan")),
        "copy_current_dynamic_mse": dyn_mse,
        "copy_current_dynamic_lpips": dyn_lpips,
    }


def load_phase0_model(args: argparse.Namespace, device: torch.device):
    model_dir = Path(args.phase0_ar_pixel_ckpt)
    base_dir = Path(args.phase0_ar_pixel_config) if args.phase0_ar_pixel_config else Path(args.base_model_dir)
    if (model_dir / "model.safetensors").exists():
        return _load_trained_model(str(base_dir), str(model_dir), args.tokenizer_ckpt, device)
    return _load_model(str(model_dir), args.tokenizer_ckpt, device)


_NEGATIVE_TYPE_ALIASES = {
    "same_task_same_phase_other_window": "same_phase",
    "same_task_other_window": "same_phase",
    "same_phase": "same_phase",
    "temporal_shift": "temporal_shift",
    "temporal_perturbation": "temporal_shift",
    "action_noise": "action_noise",
    "policy_like_perturbation": "action_noise",
    "mixed": "mixed",
}


def parse_negative_eval_types(raw: str) -> list[str]:
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        key = part.strip()
        if not key:
            continue
        norm = _NEGATIVE_TYPE_ALIASES.get(key)
        if norm is None:
            raise ValueError(f"Unknown negative eval type: {key}")
        if norm not in out:
            out.append(norm)
    return out or ["same_phase", "temporal_shift", "action_noise", "mixed"]


def negative_metric_suffix(neg_type: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in neg_type).strip("_")


def temporal_shift_actions(actions, rng: random.Random, max_shift: int):
    arr = np.asarray(actions)
    if len(arr) <= 1:
        return actions
    shift_cap = max(1, min(max_shift, len(arr) - 1))
    shift = rng.choice([s for s in range(-shift_cap, shift_cap + 1) if s != 0])
    shifted = np.empty_like(arr)
    if shift > 0:
        shifted[:shift] = arr[0]
        shifted[shift:] = arr[:-shift]
    else:
        s = -shift
        shifted[:-s] = arr[s:]
        shifted[-s:] = arr[-1]
    return shifted


def noisy_actions(actions, rng: random.Random, noise_std: float):
    arr = np.asarray(actions)
    scale = np.nanstd(arr, axis=0, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    noise_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    return arr + noise_rng.normal(0.0, noise_std * scale, size=arr.shape).astype(arr.dtype)


def save_ranking_by_task(rows: list[dict], out: Path) -> None:
    groups: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        key = str(row.get("task_name", "unknown"))
        groups.setdefault(key, {"correct": [], "shuffled": [], "gap": [], "win": []})
        groups[key]["correct"].append(float(row.get("correct_lpips", float("nan"))))
        groups[key]["shuffled"].append(float(row.get("shuffled_lpips", float("nan"))))
        groups[key]["gap"].append(float(row.get("lpips_gap", float("nan"))))
        groups[key]["win"].append(1.0 if row.get("pairwise_win") else 0.0)
    fields = [
        "task_name", "correct_lpips", "shuffled_lpips", "lpips_gap",
        "lpips_gap_min", "pairwise_acc", "reverse_windows", "num_windows",
    ]
    with (out / "ranking_by_task.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for task, vals in sorted(groups.items()):
            gaps = np.asarray(vals["gap"], dtype=np.float32)
            wins = np.asarray(vals["win"], dtype=np.float32)
            writer.writerow({
                "task_name": task,
                "correct_lpips": float(np.nanmean(vals["correct"])),
                "shuffled_lpips": float(np.nanmean(vals["shuffled"])),
                "lpips_gap": float(np.nanmean(gaps)),
                "lpips_gap_min": float(np.nanmin(gaps)),
                "pairwise_acc": float(np.nanmean(wins)),
                "reverse_windows": int(np.sum(wins < 0.5)),
                "num_windows": int(len(wins)),
            })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-ar-pixel-ckpt", required=True)
    ap.add_argument("--phase0-ar-pixel-config", default="")
    ap.add_argument("--base-model-dir", default="checkpoints/libero/WorldModel/spatial")
    ap.add_argument("--tokenizer-ckpt", required=True)
    ap.add_argument("--window-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--task-suite", default="spatial")
    ap.add_argument("--data-root", default="/localdata/modified_libero_rlds")
    ap.add_argument("--eval-horizon", type=int, default=7)
    ap.add_argument("--decode-chunk-size", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dynamic-threshold", type=float, default=0.05)
    ap.add_argument("--dynamic-dilate-kernel", type=int, default=7)
    ap.add_argument("--num-shuffle-reps", type=int, default=3,
                    help="Number of negative action samples per window (same default as v4 eval)")
    ap.add_argument("--negative-eval-types", default=os.environ.get("NEGATIVE_EVAL_TYPES", "same_phase,temporal_shift,action_noise,mixed"))
    ap.add_argument("--temporal-shift-max", type=int, default=int(os.environ.get("TEMPORAL_SHIFT_MAX", "3")))
    ap.add_argument("--action-noise-std", type=float, default=float(os.environ.get("ACTION_NOISE_STD", "0.15")))
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed for negative sampling RNG")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    manifest_path = Path(args.window_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest_path)
    (out / "window_manifest_used.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "window_manifest_hash.txt").write_text(manifest_hash + "\n", encoding="utf-8")

    proto = manifest.get("protocol", {})
    if proto.get("task_suite") != args.task_suite:
        raise ValueError(f"task_suite mismatch: manifest={proto.get('task_suite')} current={args.task_suite}")
    if int(proto.get("eval_horizon", args.eval_horizon)) != int(args.eval_horizon):
        raise ValueError(f"eval_horizon mismatch: manifest={proto.get('eval_horizon')} current={args.eval_horizon}")

    dataset_name = resolve_dataset_name(args.task_suite)
    ds = tfds.load(dataset_name, data_dir=args.data_root, split="train", shuffle_files=False)
    lookup = build_episode_lookup(ds)

    model = load_phase0_model(args, device).eval()
    lpips_model = __import__("lpips").LPIPS(net="alex").to(device).eval()
    lpips_fn = get_lpips_fn(device=str(device))
    roi_config = load_roi_config()

    rows = []
    windows = manifest.get("windows", [])
    if args.smoke:
        windows = windows[: max(1, min(2, len(windows)))]
    negative_eval_types = parse_negative_eval_types(args.negative_eval_types)
    default_negative_type = "mixed" if "mixed" in negative_eval_types else negative_eval_types[0]

    # Build per-task window list for negative sampling (same strategy as v4 eval:
    # sample actions from another window of the same task, args.num_shuffle_reps times).
    task_wins_by_id: dict[int, list[dict]] = defaultdict(list)
    for w in windows:
        task_wins_by_id[int(w["task_id"])].append(w)

    for rec in windows:
        task_id = int(rec["task_id"])
        task_name = rec.get("task_name", f"task{task_id}")
        current = int(rec["current_frame_index"])
        ep = find_episode(lookup, rec)
        frames, actions = slice_phase0_window(ep, current, args.eval_horizon)

        pred, gt = rollout_episode_single_pass(model, frames, actions, args.decode_chunk_size)

        # Negative sampling: evaluate multiple negative designs so RFT reward
        # sensitivity can be checked at different difficulty levels.
        same_task = [w for w in task_wins_by_id[task_id]
                     if w.get("global_window_id") != rec.get("global_window_id")]
        same_phase = [w for w in same_task
                      if w.get("window_position") == rec.get("window_position")]
        neg_rng = random.Random(args.seed * 10000 + int(rec.get("global_window_id", 0)))

        seq = compute_sequence_metrics_all(gt, pred, lpips_model, device)
        current_img = frames[0]
        roi_half = get_roi_half(args.task_suite, task_id, roi_config)
        goal_center = get_goal_roi_center(args.task_suite, task_id, roi_config)
        gripper_center = motion_com_np(current_img, gt[-1])
        roi = compute_roi_metrics_np(pred, gt, lpips_fn, gripper_center, goal_center, roi_half)
        dyn_mse, dyn_lpips, static_mse = dynamic_metrics(
            pred, gt, current_img, lpips_fn, args.dynamic_threshold, args.dynamic_dilate_kernel
        )
        cmet = copy_metrics(
            gt, current_img, lpips_model, device, lpips_fn,
            args.dynamic_threshold, args.dynamic_dilate_kernel, roi_half, goal_center
        )
        _h_avg_lpips = float(seq["avg_lpips"])
        _h_avg_mse = float(seq["avg_mse"])
        _h_avg_mae = horizon_avg_mae(pred, gt)
        _rft_proxy = -(_h_avg_lpips + _h_avg_mae)

        neg_stats: dict[str, dict[str, float]] = {}
        for neg_type in negative_eval_types:
            neg_lpips_list: list[float] = []
            neg_mae_list: list[float] = []
            for _ in range(args.num_shuffle_reps):
                concrete_type = neg_type
                if concrete_type == "mixed":
                    concrete_type = neg_rng.choice(["same_phase", "temporal_shift", "action_noise"])
                if concrete_type == "same_phase":
                    candidates = same_phase or same_task
                    if candidates:
                        neg_rec = neg_rng.choice(candidates)
                        neg_ep_i = find_episode(lookup, neg_rec)
                        _, neg_acts_i = slice_phase0_window(neg_ep_i, int(neg_rec["current_frame_index"]), args.eval_horizon)
                    else:
                        neg_acts_i = temporal_shift_actions(actions, neg_rng, args.temporal_shift_max)
                elif concrete_type == "temporal_shift":
                    neg_acts_i = temporal_shift_actions(actions, neg_rng, args.temporal_shift_max)
                elif concrete_type == "action_noise":
                    neg_acts_i = noisy_actions(actions, neg_rng, args.action_noise_std)
                else:
                    raise ValueError(f"Unknown negative type: {concrete_type}")

                pred_neg_i, _ = rollout_episode_single_pass(model, frames, neg_acts_i, args.decode_chunk_size)
                seq_neg_i = compute_sequence_metrics_all(gt, pred_neg_i, lpips_model, device)
                neg_lpips_list.append(float(seq_neg_i["avg_lpips"]))
                neg_mae_list.append(horizon_avg_mae(pred_neg_i, gt))
            _neg_lpips = float(np.nanmean(neg_lpips_list)) if neg_lpips_list else float("nan")
            _neg_mae = float(np.nanmean(neg_mae_list)) if neg_mae_list else float("nan")
            _neg_proxy = -(_neg_lpips + _neg_mae)
            neg_stats[neg_type] = {
                "lpips": _neg_lpips,
                "mae": _neg_mae,
                "rft_proxy": _neg_proxy,
                "rft_gap": _rft_proxy - _neg_proxy,
            }

        default_stats = neg_stats[default_negative_type]
        _shuffled_lpips = default_stats["lpips"]
        _rft_reward_gap = default_stats["rft_gap"]
        _copy_mse = cmet["copy_current_full_mse"]
        row = {
            "model_name": "phase0_ar_pixel_direct_eval",
            "model_family": "phase0_ar_pixel",
            "model_generation": "phase0_ar_pixel",
            "target_mode": "ar_pixel",
            "metric_source": "direct_eval_on_phase1_manifest",
            "window_manifest_hash": manifest_hash,
            "task_index": task_id,
            "task_name": task_name,
            "window_id": rec.get("local_window_id", rec.get("global_window_id")),
            "global_window_id": rec.get("global_window_id"),
            "negative_window_id": rec.get("negative_window_id"),
            "window_phase": rec.get("window_position"),
            "episode_length": rec.get("episode_length", ""),
            "episode_file": rec.get("episode_file", ""),
            "frame_indices": json.dumps([current] + rec.get("future_frame_indices", [])),
            "action_indices": json.dumps(rec.get("action_indices", [])),
            # Fields expected by aggregate_phase1_metrics
            "horizon_avg_lpips": _h_avg_lpips,
            "horizon_avg_mae": _h_avg_mae,
            "horizon_avg_mse": _h_avg_mse,
            "rft_reward_proxy": _rft_proxy,
            "copy_current_horizon_avg_mse": _copy_mse,
            "horizon_mse_over_copy": _h_avg_mse / _copy_mse if _copy_mse > 0 else float("nan"),
            "rft_reward_gap": _rft_reward_gap,
            "pairwise_win_rft": _rft_reward_gap > 0,
            # v4-specific fields are not applicable for Phase0
            "score_gap": float("nan"),
            "fuser_mask_entropy": float("nan"),
            "fuser_mask_overlap": float("nan"),
            "dynamic_mask_entropy": float("nan"),
            "dynamic_mask_overlap": float("nan"),
            "future_dynamic_query_norm": float("nan"),
            # Legacy / auxiliary fields kept for ranking_by_task and diagnostics
            "correct_lpips": _h_avg_lpips,
            "shuffled_lpips": _shuffled_lpips,
            "lpips_gap": _shuffled_lpips - _h_avg_lpips,
            "pairwise_win": _h_avg_lpips < _shuffled_lpips,
            "full_mse": _h_avg_mse,
            "full_lpips": _h_avg_lpips,
            "gripper_mse": roi.get("roi/gripper_mse", float("nan")),
            "gripper_lpips": roi.get("roi/gripper_lpips", float("nan")),
            "goal_mse": roi.get("roi/goal_mse", float("nan")),
            "goal_lpips": roi.get("roi/goal_lpips", float("nan")),
            "dynamic_mse": dyn_mse,
            "dynamic_lpips": dyn_lpips,
            "static_consistency_mse": static_mse,
            **cmet,
        }
        row["negative_metric_type"] = default_negative_type
        for neg_type, stats in neg_stats.items():
            suffix = negative_metric_suffix(neg_type)
            row[f"rft_reward_gap_{suffix}"] = stats["rft_gap"]
            row[f"pairwise_win_rft_{suffix}"] = stats["rft_gap"] > 0
            row[f"score_shuffle_{suffix}"] = float("nan")
            row[f"score_gap_{suffix}"] = float("nan")
            row[f"pairwise_win_score_{suffix}"] = False
        rows.append(row)

    metrics = aggregate_phase1_metrics(rows, str(out), "phase0_ar_pixel_direct_eval")
    save_ranking_by_task(rows, out)
    metrics.update({
        "model_name": "phase0_ar_pixel_direct_eval",
        "model_family": "phase0_ar_pixel",
        "model_generation": "phase0_ar_pixel",
        "target_mode": "ar_pixel",
        "metric_source": "direct_eval_on_phase1_manifest",
        "window_manifest_hash": manifest_hash,
        "phase0_compatible": True,
        "negative_eval_types": negative_eval_types,
        "default_negative_type": default_negative_type,
        "temporal_shift_max": args.temporal_shift_max,
        "action_noise_std": args.action_noise_std,
        # Alias: summary scripts read rft_reward_gap_mean; aggregate_phase1_metrics
        # stores the mean under plain "rft_reward_gap" (in scalar_keys).
        "rft_reward_gap_mean": metrics.get("rft_reward_gap", float("nan")),
        # Phase0 has no ActionFutureScorer — set explicitly to NaN for consistency.
        "pairwise_acc_score": float("nan"),
        "score_gap_mean":     float("nan"),
        "score_gap_min":      float("nan"),
    })
    (out / "aggregate_metrics.json").write_text(
        json.dumps({"condition": "phase0_ar_pixel_direct_eval", "metrics": metrics}, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    protocol = {
        **proto,
        "checkpoint_path": args.phase0_ar_pixel_ckpt,
        "tokenizer_path": args.tokenizer_ckpt,
        "target_mode": "ar_pixel",
        "model_generation": "phase0_ar_pixel",
        "model_family": "phase0_ar_pixel",
        "metric_source": "direct_eval_on_phase1_manifest",
        "window_manifest_hash": manifest_hash,
        "rollout_mode": "phase0 AR-Pixel single_pass on Phase1 manifest",
        "decode_chunk_size": args.decode_chunk_size,
        "phase0_compatible": True,
        "negative_eval_types": negative_eval_types,
        "default_negative_type": default_negative_type,
        "temporal_shift_max": args.temporal_shift_max,
        "action_noise_std": args.action_noise_std,
    }
    (out / "eval_protocol_config.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "config_used.json").write_text(json.dumps({"condition": "phase0_ar_pixel_direct_eval", "eval_protocol": protocol}, indent=2), encoding="utf-8")
    (out / "copy_current_metrics.json").write_text(json.dumps({k: metrics.get(k) for k in metrics if k.startswith("copy_current")}, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Wrote Phase0 AR-Pixel direct eval to {out}")


if __name__ == "__main__":
    main()
