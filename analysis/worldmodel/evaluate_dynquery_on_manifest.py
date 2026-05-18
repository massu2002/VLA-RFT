#!/usr/bin/env python3
"""Evaluate DynQueryWorldModel on a Phase1 window manifest.

Produces the same aggregate_metrics.json format as evaluate_ar_pixel_on_manifest.py
so that report_ja.py and average_seed_metrics.py work unchanged.

Usage:
    python analysis/worldmodel/evaluate_dynquery_on_manifest.py \
        --model-dir checkpoints/libero/DynQueryWorldModel/core_sweep/spatial/dq_baseline/s42/final \
        --window-manifest results/phase1/DynQueryWorldModel_core_sweep/dq_baseline/window_manifest.json \
        --output-dir results/phase1/DynQueryWorldModel_core_sweep/dq_baseline \
        --task-suite spatial \
        --data-root data/modified_libero_rlds

Key differences from evaluate_ar_pixel_on_manifest.py:
  - Loads DynQueryWorldModel via DynQueryWorldModel.load_pretrained()
  - Slices K history frames + current + H future GT frames for rollout
  - Adds DynQuery-specific metrics: dynamic_mask_iou_gt, ranking_score
"""

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
from tqdm import tqdm

from worldmodel.datasets.libero.data import resolve_dataset_name
from worldmodel.dynquery.model import DynQueryWorldModel
from worldmodel.dynquery.utils import aggregate_phase1_metrics, get_lpips_fn
from worldmodel.eval_roi_utils import (
    compute_roi_metrics_np,
    get_goal_roi_center,
    get_roi_half,
    load_roi_config,
    motion_com_np,
)
from worldmodel.libero.visualize import compute_sequence_metrics_all


# ---------------------------------------------------------------------------
# Shared helpers (identical to evaluate_ar_pixel_on_manifest.py)
# ---------------------------------------------------------------------------

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


def build_episode_lookup_selective(ds, needed_indices: set[int]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    remaining = set(needed_indices)
    for idx, ep in enumerate(ds):
        if idx not in needed_indices:
            continue
        path = ep["episode_metadata"]["file_path"].numpy().decode("utf-8", errors="ignore")
        steps_np = []
        for step in ep["steps"]:
            step_np: dict = {}
            for k, v in step.items():
                if k == "observation":
                    step_np["observation"] = {k2: v2.numpy() for k2, v2 in v.items()}
                else:
                    step_np[k] = v.numpy() if hasattr(v, "numpy") else v
            steps_np.append(step_np)
        ep_np = {"episode_metadata": {"file_path": path.encode()}, "steps": steps_np}
        out[path] = ep_np
        out[os.path.basename(path)] = ep_np
        out[episode_id(path)] = ep_np
        out[str(idx)] = ep_np
        remaining.discard(idx)
        if not remaining:
            break
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


def masked_mse_np(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(np.float32)
    if m.sum() <= 0:
        return float("nan")
    diff = (a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0) ** 2
    return float((diff * m[:, :, None]).sum() / (m.sum() * a.shape[-1]))


def masked_mae_np(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(np.float32)
    if m.sum() <= 0:
        return float("nan")
    diff = np.abs(a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0)
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
        mp = (pred.astype(np.float32) * mask[:, :, None] + current.astype(np.float32) * (1.0 - mask[:, :, None])).astype(np.uint8)
        mg = (gt.astype(np.float32) * mask[:, :, None] + current.astype(np.float32) * (1.0 - mask[:, :, None])).astype(np.uint8)
        lpips_vals.append(lpips_np(lpips_fn, mp, mg))
    return float(np.nanmean(mses)), float(np.nanmean(lpips_vals)), float(np.nanmean(static))


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
            raise ValueError(f"Unknown negative eval type: {key!r}")
        if norm not in out:
            out.append(norm)
    return out or ["same_phase", "temporal_shift", "action_noise"]


def negative_metric_suffix(neg_type: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in neg_type).strip("_")


def temporal_shift_actions(actions: np.ndarray, rng: random.Random, max_shift: int) -> np.ndarray:
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


def noisy_actions(actions: np.ndarray, rng: random.Random, noise_std: float) -> np.ndarray:
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
    fields = ["task_name", "correct_lpips", "shuffled_lpips", "lpips_gap",
              "lpips_gap_min", "pairwise_acc", "reverse_windows", "num_windows"]
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


# ---------------------------------------------------------------------------
# DynQuery-specific helpers
# ---------------------------------------------------------------------------

def slice_dynquery_window(
    ep: dict,
    current_idx: int,
    K: int,
    H: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Slice episode into (all_frames[K+1+H], all_actions[K+H], gt_frames[H]).

    Returns:
        pixels_list : K history + 1 current + H future GT frames (uint8 HWC)
        actions_list: K history + H horizon actions
        gt_frames   : H future GT frames (uint8 HWC), for metric computation
    """
    imgs, acts = episode_arrays(ep)
    T = len(imgs)

    # History: clamp to [0, current_idx)
    hist_start = max(0, current_idx - K)
    hist_frames = list(imgs[hist_start:current_idx])
    hist_acts   = list(acts[hist_start:current_idx])

    # Pad history to exactly K frames if episode too short
    while len(hist_frames) < K:
        hist_frames.insert(0, imgs[0])
        hist_acts.insert(0, acts[0])

    current_frame = imgs[current_idx]
    future_end = min(current_idx + H + 1, T)
    future_frames = list(imgs[current_idx + 1 : future_end])
    horizon_acts  = list(acts[current_idx : current_idx + H])

    if len(future_frames) != H or len(horizon_acts) != H:
        raise ValueError(
            f"Insufficient frames: current={current_idx} H={H} T={T} "
            f"got future={len(future_frames)} acts={len(horizon_acts)}"
        )

    pixels_list  = hist_frames + [current_frame] + future_frames  # K+1+H
    actions_list = hist_acts   + horizon_acts                      # K+H
    return pixels_list, actions_list, future_frames


def rollout_dynquery(
    model: DynQueryWorldModel,
    pixels_list: list[np.ndarray],
    actions_list: list[np.ndarray],
    K: int,
    H: int,
    device: torch.device,
) -> tuple[list[np.ndarray], dict]:
    """Run DynQuery rollout and return (pred_uint8_list, extra_metrics_dict).

    pred_uint8_list: H predicted frames as uint8 HWC numpy arrays.
    extra_metrics_dict: DynQuery-specific outputs (gate, ranking_score, …).
    """
    # Stack into tensors
    pixels_np  = np.stack(pixels_list, axis=0)       # [K+1+H, H, W, C] uint8
    actions_np = np.stack(actions_list, axis=0)      # [K+H, action_dim]

    pixels_t  = torch.from_numpy(pixels_np).unsqueeze(0).to(device)    # [1, K+1+H, H, W, C]
    actions_t = torch.from_numpy(actions_np).unsqueeze(0).to(device)   # [1, K+H, action_dim]

    with torch.no_grad():
        out = model.rollout(pixels_t, actions_t, H)

    # pred_future: [1, H, 3, H_img, W_img] float [0,1] CHW → convert to HWC uint8
    pred_f = out["pred_future"][0]  # [H, 3, H_img, W_img]
    pred_np = (pred_f.float().permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    pred_list = [pred_np[h] for h in range(H)]

    # DynQuery-specific extras
    extras: dict = {}

    # Dynamic residual gate: [1, H, 1, H_img, W_img] — used for mask IoU
    if "dynamic_residual_gate" in out:
        extras["dynamic_residual_gate"] = out["dynamic_residual_gate"][0]  # [H, 1, H_img, W_img]

    # ActionFutureScorer ranking score: [1] scalar
    if out.get("ranking_score") is not None:
        extras["ranking_score"] = float(out["ranking_score"][0].item())

    return pred_list, extras


def compute_mask_iou_metrics(
    pred_gate: torch.Tensor,  # [H, 1, H_img, W_img] float in [0,1]
    current_img: np.ndarray,  # HWC uint8
    gt_frames: list[np.ndarray],  # H × HWC uint8
    threshold: float,
    dilate: int,
    gate_threshold: float = 0.3,
) -> dict[str, float]:
    """Compute IoU / precision / recall between model gate and GT dynamic mask."""
    ious, precs, recs = [], [], []
    H = len(gt_frames)
    for h in range(H):
        gt_mask = dynamic_mask_np(current_img, gt_frames[h], threshold, dilate)  # HW float
        pred_mask = (pred_gate[h, 0].float().cpu().numpy() > gate_threshold).astype(np.float32)

        inter = (pred_mask * gt_mask).sum()
        union = np.clip(pred_mask + gt_mask, 0, 1).sum()
        iou   = float(inter / union) if union > 0 else float("nan")
        prec  = float(inter / pred_mask.sum()) if pred_mask.sum() > 0 else float("nan")
        rec   = float(inter / gt_mask.sum())   if gt_mask.sum()  > 0 else float("nan")
        ious.append(iou); precs.append(prec); recs.append(rec)

    return {
        "dynamic_mask_iou_gt":       float(np.nanmean(ious)),
        "dynamic_mask_precision_gt": float(np.nanmean(precs)),
        "dynamic_mask_recall_gt":    float(np.nanmean(recs)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="Path to DynQuery final checkpoint directory (contains dynquery_config.json)")
    ap.add_argument("--window-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--task-suite", default="spatial")
    ap.add_argument("--data-root", default="data/modified_libero_rlds")
    ap.add_argument("--eval-horizon", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dynamic-threshold", type=float, default=0.05)
    ap.add_argument("--dynamic-dilate-kernel", type=int, default=7)
    ap.add_argument("--num-shuffle-reps", type=int, default=1)
    ap.add_argument("--negative-eval-types",
                    default=os.environ.get("NEGATIVE_EVAL_TYPES", "same_phase,temporal_shift,action_noise"))
    ap.add_argument("--temporal-shift-max", type=int,
                    default=int(os.environ.get("TEMPORAL_SHIFT_MAX", "3")))
    ap.add_argument("--action-noise-std", type=float,
                    default=float(os.environ.get("ACTION_NOISE_STD", "0.15")))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--condition-name", default="",
                    help="Label written to aggregate_metrics.json (defaults to model dir basename)")
    args = ap.parse_args()

    condition_name = args.condition_name or Path(args.model_dir).parent.name

    out_base = Path(args.output_dir)
    out = out_base / f"shard_{args.shard_index}_of_{args.num_shards}" if args.num_shards > 1 else out_base
    out.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── Load manifest ────────────────────────────────────────────────────────
    manifest_path = Path(args.window_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest_path)
    (out / "window_manifest_used.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "window_manifest_hash.txt").write_text(manifest_hash + "\n", encoding="utf-8")

    proto = manifest.get("protocol", {})
    if proto.get("task_suite") and proto["task_suite"] != args.task_suite:
        raise ValueError(f"task_suite mismatch: manifest={proto['task_suite']} current={args.task_suite}")

    all_windows = manifest.get("windows", [])
    windows = all_windows
    if args.smoke:
        windows = windows[: max(1, min(2, len(windows)))]
    if args.num_shards > 1:
        windows = windows[args.shard_index :: args.num_shards]
        print(f"[Shard {args.shard_index}/{args.num_shards}] {len(windows)} windows assigned (round-robin)")

    # ── Load data ────────────────────────────────────────────────────────────
    dataset_name = resolve_dataset_name(args.task_suite)
    ds = tfds.load(dataset_name, data_dir=args.data_root, split="train", shuffle_files=False)
    needed_indices = {int(w["episode_index"]) for w in all_windows if "episode_index" in w}
    print(f"Loading {len(needed_indices)} episodes selectively …")
    lookup = build_episode_lookup_selective(ds, needed_indices)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading DynQueryWorldModel from {args.model_dir} …")
    model = DynQueryWorldModel.load_pretrained(args.model_dir).to(device).eval()
    K = model.cfg.history_length
    H = args.eval_horizon
    print(f"  K={K}  H={H}  scorer={'ON' if model.action_future_scorer else 'OFF'}")

    lpips_model = __import__("lpips").LPIPS(net="alex").to(device).eval()
    lpips_fn    = get_lpips_fn(device=str(device))
    roi_config  = load_roi_config()

    # ── Negative sampling pool ────────────────────────────────────────────────
    negative_eval_types = parse_negative_eval_types(args.negative_eval_types)
    _POOLED_NEG_TYPES   = ["same_phase", "temporal_shift", "action_noise"]
    default_negative_type = "pooled_equal"

    task_wins_by_id: dict[int, list[dict]] = defaultdict(list)
    for w in all_windows:
        task_wins_by_id[int(w["task_id"])].append(w)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    rows: list[dict] = []
    pbar = tqdm(windows, desc="Evaluating windows", unit="win", dynamic_ncols=True)

    for rec in pbar:
        task_id   = int(rec["task_id"])
        task_name = rec.get("task_name", f"task{task_id}")
        current_idx = int(rec["current_frame_index"])
        ep = find_episode(lookup, rec)

        pixels_list, actions_list, gt_frames = slice_dynquery_window(ep, current_idx, K, H)
        current_img = pixels_list[K]  # current frame (after K history frames)

        # ── Correct-action rollout ────────────────────────────────────────
        pred, extras = rollout_dynquery(model, pixels_list, actions_list, K, H, device)

        seq = compute_sequence_metrics_all(gt_frames, pred, lpips_model, device)
        roi_half    = get_roi_half(args.task_suite, task_id, roi_config)
        goal_center = get_goal_roi_center(args.task_suite, task_id, roi_config)
        gripper_center = motion_com_np(current_img, gt_frames[-1])
        roi = compute_roi_metrics_np(pred, gt_frames, lpips_fn, gripper_center, goal_center, roi_half)

        dyn_mse, dyn_lpips, static_mse = dynamic_metrics(
            pred, gt_frames, current_img, lpips_fn,
            args.dynamic_threshold, args.dynamic_dilate_kernel,
        )
        _h_avg_lpips = float(seq["avg_lpips"])
        _h_avg_mse   = float(seq["avg_mse"])
        _h_avg_mae   = horizon_avg_mae(pred, gt_frames)
        _rft_proxy   = -(_h_avg_lpips + _h_avg_mae)

        # Per-step LPIPS / MAE
        _lpips_per_frame = seq.get("lpips_per_frame", np.array([]))
        lpips_step1 = float(_lpips_per_frame[0]) if len(_lpips_per_frame) > 0 else float("nan")
        lpips_step4 = float(_lpips_per_frame[3]) if len(_lpips_per_frame) > 3 else float("nan")
        lpips_step8 = float(_lpips_per_frame[7]) if len(_lpips_per_frame) > 7 else float("nan")
        _mae_per_step = [
            float(np.mean(np.abs(p.astype(np.float32) / 255.0 - g.astype(np.float32) / 255.0)))
            for p, g in zip(pred, gt_frames)
        ]
        mae_step1 = _mae_per_step[0] if _mae_per_step else float("nan")
        mae_step4 = _mae_per_step[3] if len(_mae_per_step) > 3 else float("nan")
        mae_step8 = _mae_per_step[7] if len(_mae_per_step) > 7 else float("nan")

        # Dynamic / static masked MAE
        _dyn_mae_vals, _static_mae_vals = [], []
        for _p, _g in zip(pred, gt_frames):
            _dm = dynamic_mask_np(current_img, _g, args.dynamic_threshold, args.dynamic_dilate_kernel)
            _dyn_mae_vals.append(masked_mae_np(_p, _g, _dm))
            _static_mae_vals.append(masked_mae_np(_p, current_img, 1.0 - _dm))
        dynamic_region_mae_gt  = float(np.nanmean(_dyn_mae_vals))  if _dyn_mae_vals  else float("nan")
        static_consistency_mae = float(np.nanmean(_static_mae_vals)) if _static_mae_vals else float("nan")

        # DynQuery-specific: mask IoU
        if "dynamic_residual_gate" in extras:
            mask_metrics = compute_mask_iou_metrics(
                extras["dynamic_residual_gate"], current_img, gt_frames,
                args.dynamic_threshold, args.dynamic_dilate_kernel,
            )
        else:
            mask_metrics = {"dynamic_mask_iou_gt": float("nan"),
                            "dynamic_mask_precision_gt": float("nan"),
                            "dynamic_mask_recall_gt": float("nan")}

        ranking_score_correct = extras.get("ranking_score", float("nan"))

        pbar.set_postfix({
            "task": task_name[:25],
            "lpips": f"{_h_avg_lpips:.4f}",
            "mae":   f"{_h_avg_mae:.4f}",
            "rft":   f"{_rft_proxy:.4f}",
        })

        # ── Negative-action rollouts ──────────────────────────────────────
        same_task  = [w for w in task_wins_by_id[task_id]
                      if w.get("global_window_id") != rec.get("global_window_id")]
        same_phase = [w for w in same_task
                      if w.get("window_position") == rec.get("window_position")]
        neg_rng = random.Random(args.seed * 10000 + int(rec.get("global_window_id", 0)))

        neg_stats: dict[str, dict[str, float]] = {}
        score_gaps: dict[str, float] = {}

        for neg_type in negative_eval_types:
            neg_lpips_list: list[float] = []
            neg_mae_list:   list[float] = []
            neg_score_list: list[float] = []

            for _ in range(args.num_shuffle_reps):
                concrete = neg_type
                if concrete == "mixed":
                    concrete = neg_rng.choice(["same_phase", "temporal_shift", "action_noise"])

                if concrete == "same_phase":
                    candidates = same_phase or same_task
                    if candidates:
                        neg_rec = neg_rng.choice(candidates)
                        neg_ep  = find_episode(lookup, neg_rec)
                        neg_pixels, neg_actions, _ = slice_dynquery_window(
                            neg_ep, int(neg_rec["current_frame_index"]), K, H
                        )
                        # Replace only horizon actions (keep same history/current frames)
                        neg_horizon_actions = neg_actions[K:]
                        neg_actions_merged  = actions_list[:K] + neg_horizon_actions
                    else:
                        neg_horizon_shifted = temporal_shift_actions(
                            np.stack(actions_list[K:]), neg_rng, args.temporal_shift_max
                        )
                        neg_actions_merged = actions_list[:K] + [neg_horizon_shifted[i] for i in range(H)]
                elif concrete == "temporal_shift":
                    neg_horizon_shifted = temporal_shift_actions(
                        np.stack(actions_list[K:]), neg_rng, args.temporal_shift_max
                    )
                    neg_actions_merged = actions_list[:K] + [neg_horizon_shifted[i] for i in range(H)]
                elif concrete == "action_noise":
                    neg_horizon_noisy = noisy_actions(
                        np.stack(actions_list[K:]), neg_rng, args.action_noise_std
                    )
                    neg_actions_merged = actions_list[:K] + [neg_horizon_noisy[i] for i in range(H)]
                else:
                    raise ValueError(f"Unknown negative type: {concrete!r}")

                neg_pixels_full = pixels_list[:K + 1] + pixels_list[K + 1:]
                pred_neg, extras_neg = rollout_dynquery(
                    model, neg_pixels_full, neg_actions_merged, K, H, device
                )
                seq_neg = compute_sequence_metrics_all(gt_frames, pred_neg, lpips_model, device)
                neg_lpips_list.append(float(seq_neg["avg_lpips"]))
                neg_mae_list.append(horizon_avg_mae(pred_neg, gt_frames))
                if not math.isnan(ranking_score_correct) and "ranking_score" in extras_neg:
                    neg_score_list.append(extras_neg["ranking_score"])

            _neg_lpips = float(np.nanmean(neg_lpips_list)) if neg_lpips_list else float("nan")
            _neg_mae   = float(np.nanmean(neg_mae_list))   if neg_mae_list   else float("nan")
            _neg_proxy = -(_neg_lpips + _neg_mae)
            neg_stats[neg_type] = {
                "lpips": _neg_lpips, "mae": _neg_mae,
                "rft_proxy": _neg_proxy,
                "rft_gap": _rft_proxy - _neg_proxy,
            }
            if neg_score_list:
                score_gaps[neg_type] = ranking_score_correct - float(np.nanmean(neg_score_list))

        # Pooled pairwise (equal weight across same_phase / temporal_shift / action_noise)
        _pool_types = [t for t in _POOLED_NEG_TYPES if t in neg_stats]
        if _pool_types:
            _pooled_neg_lpips = float(np.nanmean([neg_stats[t]["lpips"] for t in _pool_types]))
            _pooled_neg_mae   = float(np.nanmean([neg_stats[t]["mae"]   for t in _pool_types]))
            _rft_reward_gap   = _rft_proxy - (-(_pooled_neg_lpips + _pooled_neg_mae))
            _shuffled_lpips   = _pooled_neg_lpips
        else:
            _pooled_neg_lpips = float("nan")
            _rft_reward_gap   = float("nan")
            _shuffled_lpips   = float("nan")

        # Pooled score gap
        _pooled_score_gap = float(np.nanmean(list(score_gaps.values()))) if score_gaps else float("nan")
        _score_gap_min    = float(np.nanmin(list(score_gaps.values())))  if score_gaps else float("nan")

        row: dict = {
            "model_name":       condition_name,
            "model_family":     "dynquery",
            "model_generation": "dynquery",
            "target_mode":      "temporal_query_residual",
            "metric_source":    "direct_eval_on_phase1_manifest",
            "window_manifest_hash": manifest_hash,
            "task_index":       task_id,
            "task_name":        task_name,
            "window_id":        rec.get("local_window_id", rec.get("global_window_id")),
            "global_window_id": rec.get("global_window_id"),
            "window_phase":     rec.get("window_position"),
            "episode_length":   rec.get("episode_length", ""),
            "episode_file":     rec.get("episode_file", ""),
            "frame_indices":    json.dumps([current_idx] + rec.get("future_frame_indices", [])),
            "action_indices":   json.dumps(rec.get("action_indices", [])),
            # Shared metrics (identical naming to AR-Pixel eval)
            "horizon_avg_lpips": _h_avg_lpips,
            "horizon_avg_mae":   _h_avg_mae,
            "horizon_avg_mse":   _h_avg_mse,
            "rft_reward_proxy":  _rft_proxy,
            "rft_reward_gap":    _rft_reward_gap,
            "pairwise_win_rft":  _rft_reward_gap > 0,
            "lpips_step1": lpips_step1, "lpips_step4": lpips_step4, "lpips_step8": lpips_step8,
            "mae_step1":   mae_step1,   "mae_step4":   mae_step4,   "mae_step8":   mae_step8,
            "dynamic_region_mse_gt":   dyn_mse,
            "dynamic_region_mae_gt":   dynamic_region_mae_gt,
            "dynamic_region_lpips_gt": dyn_lpips,
            "static_consistency_mse":  static_mse,
            "static_consistency_mae":  static_consistency_mae,
            "roi/gripper_mse":   roi.get("roi/gripper_mse",   float("nan")),
            "roi/gripper_mae":   roi.get("roi/gripper_mae",   float("nan")),
            "roi/gripper_lpips": roi.get("roi/gripper_lpips", float("nan")),
            "roi/goal_mse":      roi.get("roi/goal_mse",      float("nan")),
            "roi/goal_mae":      roi.get("roi/goal_mae",      float("nan")),
            "roi/goal_lpips":    roi.get("roi/goal_lpips",    float("nan")),
            # DynQuery-specific
            "dynamic_mask_iou_gt":       mask_metrics["dynamic_mask_iou_gt"],
            "dynamic_mask_precision_gt": mask_metrics["dynamic_mask_precision_gt"],
            "dynamic_mask_recall_gt":    mask_metrics["dynamic_mask_recall_gt"],
            "ranking_score":             ranking_score_correct,
            "score_gap":                 _pooled_score_gap,
            # Legacy aliases (for ranking_by_task backward compat)
            "correct_lpips":  _h_avg_lpips,
            "shuffled_lpips": _shuffled_lpips,
            "lpips_gap":      _shuffled_lpips - _h_avg_lpips if not math.isnan(_shuffled_lpips) else float("nan"),
            "pairwise_win":   _h_avg_lpips < _shuffled_lpips if not math.isnan(_shuffled_lpips) else False,
            "full_mse":   _h_avg_mse,
            "full_lpips": _h_avg_lpips,
            "gripper_mse":   roi.get("roi/gripper_mse",   float("nan")),
            "gripper_lpips": roi.get("roi/gripper_lpips", float("nan")),
            "goal_mse":      roi.get("roi/goal_mse",      float("nan")),
            "goal_lpips":    roi.get("roi/goal_lpips",    float("nan")),
            "dynamic_mse":   dyn_mse,
            "dynamic_lpips": dyn_lpips,
            "negative_metric_type": f"pooled({','.join(_pool_types)})" if _pool_types else "none",
            "copy_current_horizon_avg_mse": float("nan"),
            "horizon_mse_over_copy":        float("nan"),
            "fuser_mask_entropy":    float("nan"),
            "fuser_mask_overlap":    float("nan"),
            "dynamic_mask_entropy":  float("nan"),
            "dynamic_mask_overlap":  float("nan"),
            "future_dynamic_query_norm": float("nan"),
        }

        for neg_type, stats in neg_stats.items():
            sfx = negative_metric_suffix(neg_type)
            row[f"rft_reward_gap_{sfx}"]    = stats["rft_gap"]
            row[f"pairwise_win_rft_{sfx}"]  = stats["rft_gap"] > 0
            row[f"score_gap_{sfx}"]         = score_gaps.get(neg_type, float("nan"))
            row[f"score_shuffle_{sfx}"]     = float("nan")
            row[f"pairwise_win_score_{sfx}"] = (
                score_gaps.get(neg_type, float("nan")) > 0
                if not math.isnan(score_gaps.get(neg_type, float("nan"))) else False
            )
        rows.append(row)

    # ── Aggregate & write ─────────────────────────────────────────────────────
    (out / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, allow_nan=True) for r in rows) + "\n", encoding="utf-8"
    )

    metrics = aggregate_phase1_metrics(rows, str(out), condition_name)
    save_ranking_by_task(rows, out)

    def _row_vals(key: str) -> list[float]:
        return [r[key] for r in rows if isinstance(r.get(key), float) and not np.isnan(r[key])]

    _rft_proxies = _row_vals("rft_reward_proxy")
    _rft_gaps    = _row_vals("rft_reward_gap")
    rft_proxy_std   = float(np.std(_rft_proxies)) if _rft_proxies else float("nan")
    rft_proxy_range = float(np.ptp(_rft_proxies)) if _rft_proxies else float("nan")
    rft_gap_std     = float(np.std(_rft_gaps))    if _rft_gaps    else float("nan")

    _score_gaps_all  = _row_vals("score_gap")
    _score_gap_mean  = float(np.nanmean(_score_gaps_all)) if _score_gaps_all else float("nan")
    _score_gap_min_g = float(np.nanmin(_score_gaps_all))  if _score_gaps_all else float("nan")

    _mask_ious = _row_vals("dynamic_mask_iou_gt")
    _mask_prec = _row_vals("dynamic_mask_precision_gt")
    _mask_rec  = _row_vals("dynamic_mask_recall_gt")

    metrics.update({
        "model_name":       condition_name,
        "model_family":     "dynquery",
        "model_generation": "dynquery",
        "target_mode":      "temporal_query_residual",
        "metric_source":    "direct_eval_on_phase1_manifest",
        "window_manifest_hash": manifest_hash,
        "phase0_compatible":    True,
        "negative_eval_types":  negative_eval_types,
        "default_negative_type": default_negative_type,
        "temporal_shift_max":   args.temporal_shift_max,
        "action_noise_std":     args.action_noise_std,
        "rft_reward_gap_mean":  metrics.get("rft_reward_gap", float("nan")),
        "rft_reward_proxy_std":   rft_proxy_std,
        "rft_reward_proxy_range": rft_proxy_range,
        "rft_reward_gap_std":     rft_gap_std,
        "score_gap_mean":  _score_gap_mean,
        "score_gap_min":   _score_gap_min_g,
        "dynamic_mask_iou_gt_mean":       float(np.nanmean(_mask_ious)) if _mask_ious else float("nan"),
        "dynamic_mask_precision_gt_mean": float(np.nanmean(_mask_prec)) if _mask_prec else float("nan"),
        "dynamic_mask_recall_gt_mean":    float(np.nanmean(_mask_rec))  if _mask_rec  else float("nan"),
        # Placeholder for Pearson/Spearman (require cross-window pairing; set to NaN here)
        "pearson_rft_score_corr":  float("nan"),
        "spearman_rft_score_corr": float("nan"),
    })

    (out / "aggregate_metrics.json").write_text(
        json.dumps({"condition": condition_name, "metrics": metrics}, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    cfg_backup = {}
    try:
        cfg_backup = {"K": K, "H": H, "model_dir": args.model_dir,
                      "use_action_future_scorer": model.action_future_scorer is not None}
    except Exception:
        pass
    protocol = {
        **proto,
        "checkpoint_path": args.model_dir,
        "target_mode": "temporal_query_residual",
        "model_generation": "dynquery",
        "model_family": "dynquery",
        "metric_source": "direct_eval_on_phase1_manifest",
        "window_manifest_hash": manifest_hash,
        "rollout_mode": "DynQueryWorldModel.rollout()",
        "phase0_compatible": True,
        "negative_eval_types": negative_eval_types,
        "default_negative_type": default_negative_type,
        "temporal_shift_max": args.temporal_shift_max,
        "action_noise_std":   args.action_noise_std,
        **cfg_backup,
    }
    (out / "eval_protocol_config.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done. Wrote DynQuery eval to {out}")


if __name__ == "__main__":
    main()
