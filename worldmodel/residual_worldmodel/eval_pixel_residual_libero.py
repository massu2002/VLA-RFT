"""World model evaluation for PixelResidualWorldModel on LIBERO.

Produces per-window metrics and saves to:
    {output_dir}/aggregate_metrics.json
    {output_dir}/metrics_by_task.csv
    {output_dir}/ranking_by_window.csv
    {output_dir}/ranking_by_task.csv
    {output_dir}/config_used.json
    {output_dir}/debug/     (if --save-debug-images)

Metrics computed per window:
  World model quality:
    full_mse, full_lpips
    gripper_mse, gripper_lpips
    goal_mse, goal_lpips
    dynamic_mse, dynamic_lpips
    static_consistency_mse

  Ranking (GT action vs same-task shuffled action):
    correct_lpips, shuffled_lpips, lpips_gap, pairwise_win

Usage:
    python -m worldmodel.residual_worldmodel.eval_pixel_residual_libero \\
        --task-suite spatial \\
        --model-dir checkpoints/libero/PixelResidualWM/spatial/pixel_residual_v1 \\
        --data-root data/modified_libero_rlds \\
        --output-dir results/phase1/residual_worldmodel/pixel_residual \\
        --num-eval-windows 200 \\
        --condition-name pixel_residual

Dry-run:
    python -m worldmodel.residual_worldmodel.eval_pixel_residual_libero \\
        ... --dry-run-windows 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow_datasets as tfds
import torch

from ..datasets.libero.data import resolve_dataset_name
from .pixel_residual_config import PixelResidualConfig
from .pixel_residual_model import PixelResidualWorldModel
from .pixel_residual_utils import (
    aggregate_phase1_metrics,
    compute_dynamic_mask,
    extract_roi_crops,
    get_lpips_fn,
    motion_center_of_mass,
    save_debug_images,
)
from ..eval_roi_utils import (
    load_roi_config,
    get_goal_roi_center,
    get_roi_half,
    motion_com_np,
    roi_crop_np,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PixelResidualWorldModel on LIBERO data."
    )
    parser.add_argument("--task-suite", type=str, required=True,
                        choices=["spatial", "object", "goal", "10"])
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing pixel_residual_config.json + *.pt files.")
    parser.add_argument("--data-root", type=str, default="data/modified_libero_rlds")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--condition-name", type=str, default="",
                        help="Label for this condition (e.g. 'pixel_residual').")

    parser.add_argument("--num-eval-windows", type=int, default=200,
                        help="Total windows to evaluate (spread across tasks).")
    parser.add_argument("--eval-horizon",    type=int, default=7,
                        help="Rollout horizon H (capped by model's action_horizon).")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--device",          type=str, default="auto")
    parser.add_argument("--heldout-ratio",   type=float, default=0.2)

    parser.add_argument("--task-indices",    type=str, default="",
                        help="Comma-separated task indices (empty = all tasks).")

    parser.add_argument("--dry-run-windows", type=int, default=0,
                        help="Evaluate only first N windows per task for quick sanity check.")
    parser.add_argument("--save-debug-images", action="store_true", default=False)
    parser.add_argument("--lpips-batch-size", type=int, default=4,
                        help="Batch size for LPIPS computation (reduce if OOM).")

    # Ranking evaluation
    parser.add_argument("--num-ranking-windows", type=int, default=100,
                        help="Number of windows to use for ranking evaluation.")
    parser.add_argument("--num-shuffle-reps",    type=int, default=3,
                        help="How many shuffled-action rollouts per window for ranking.")

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _to_uint8_np(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] float [0,1] → [H, W, C] uint8."""
    t = t.detach().cpu().float().clamp(0, 1)
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    return (t.numpy() * 255).astype(np.uint8)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0) ** 2))


# ---------------------------------------------------------------------------
# Per-window evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_window(
    model: PixelResidualWorldModel,
    pixels_np: np.ndarray,    # [T+1, H, W, C] uint8
    actions_np: np.ndarray,   # [T, action_dim] float32
    horizon: int,
    device: torch.device,
    lpips_fn,
    goal_center: tuple,       # (cy, cx) in [0,1]
    roi_half: int,
    cfg: PixelResidualConfig,
    save_debug: bool,
    debug_dir: str,
    window_id: int,
    task_name: str,
) -> Optional[Dict]:
    """Run rollout and compute metrics for a single window."""
    try:
        T_plus_1 = pixels_np.shape[0]
        H_avail = T_plus_1 - 2
        H = min(horizon, H_avail)
        if H < 1:
            return None

        pixels_t  = torch.from_numpy(pixels_np).unsqueeze(0).to(device)    # [1, T+1, H, W, C]
        actions_t = torch.from_numpy(actions_np).unsqueeze(0).to(device)   # [1, T, action_dim]

        out = model.rollout(pixels_t, actions_t, horizon=H)

        pred_future   = out["pred_future"][0]    # [H, 3, H_img, W_img]
        current_img   = out["current_image"][0]  # [3, H_img, W_img]
        future_gt_t   = out["future_gt"][0]      # [H, 3, H_img, W_img]
        residual_pred = out["residual_pred"][0]  # [H, 3, H_img, W_img]
        dyn_mask      = out["dynamic_mask"][0]   # [H, 1, H_img, W_img]

        current_np = _to_uint8_np(current_img)  # [H, W, 3]

        # Use LAST future step as representative
        pred_last = _to_uint8_np(pred_future[-1])
        gt_last   = _to_uint8_np(future_gt_t[-1])

        # ---- Full-image metrics (last step) -----
        full_mse   = _mse(pred_last, gt_last)
        full_lpips = float(lpips_fn(pred_last, gt_last))

        # ---- Gripper ROI (motion CoM of current→last_future) -----
        cy_t, cx_t = motion_center_of_mass(
            current_img.unsqueeze(0),
            future_gt_t[-1:],
        )
        cy_v = cy_t[0].item()
        cx_v = cx_t[0].item()

        g_pred = roi_crop_np(pred_last, cy_v, cx_v, roi_half)
        g_gt   = roi_crop_np(gt_last,   cy_v, cx_v, roi_half)
        gripper_mse   = _mse(g_pred, g_gt)
        gripper_lpips = float(lpips_fn(g_pred, g_gt))

        # ---- Goal ROI (fixed per-task center) -----
        l_pred = roi_crop_np(pred_last, goal_center[0], goal_center[1], roi_half)
        l_gt   = roi_crop_np(gt_last,   goal_center[0], goal_center[1], roi_half)
        goal_mse   = _mse(l_pred, l_gt)
        goal_lpips = float(lpips_fn(l_pred, l_gt))

        # ---- Dynamic / static metrics (last step) -----
        dm_last = dyn_mask[-1].float()   # [1, H_img, W_img]
        dm_np   = dm_last[0].cpu().numpy()   # [H_img, W_img]

        pred_last_f = pred_future[-1].cpu().float()   # [3, H, W]
        gt_last_f   = future_gt_t[-1].cpu().float()   # [3, H, W]
        curr_last_f = current_img.cpu().float()        # [3, H, W]

        dynamic_mse = float("nan")
        static_mse  = float("nan")
        if dm_np.sum() > 0:
            dyn_mask_ch = torch.from_numpy(dm_np).unsqueeze(0)  # [1, H, W]
            dyn_mask_ch3 = dyn_mask_ch.unsqueeze(0).expand(1, 3, -1, -1)  # [1, 3, H, W]
            p_dyn = (pred_last_f.unsqueeze(0) * dyn_mask_ch).sum() / dyn_mask_ch3.sum().clamp(1)
            g_dyn = (gt_last_f.unsqueeze(0)   * dyn_mask_ch).sum() / dyn_mask_ch3.sum().clamp(1)
            # Per-pixel MSE in dynamic region
            dyn_diff = ((pred_last_f - gt_last_f) ** 2) * dyn_mask_ch
            dynamic_mse = float(dyn_diff.sum() / (dyn_mask_ch.sum() * 3).clamp(1))

            static_mask  = 1.0 - dyn_mask_ch
            static_diff  = ((pred_last_f - curr_last_f) ** 2) * static_mask
            static_mse   = float(static_diff.sum() / (static_mask.sum() * 3).clamp(1))

        # ---- Dynamic region LPIPS (if region large enough) -----
        dynamic_lpips = float("nan")
        if dm_np.sum() > 64:
            mask_bool = dm_np > 0.5
            rows = np.any(mask_bool, axis=1)
            cols = np.any(mask_bool, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            p_crop = pred_last[rmin:rmax+1, cmin:cmax+1]
            g_crop = gt_last[rmin:rmax+1,   cmin:cmax+1]
            if p_crop.size > 0 and g_crop.size > 0:
                try:
                    dynamic_lpips = float(lpips_fn(p_crop, g_crop))
                except Exception:
                    pass

        row: Dict = {
            "task_name":           task_name,
            "window_id":           window_id,
            "full_mse":            full_mse,
            "full_lpips":          full_lpips,
            "gripper_mse":         gripper_mse,
            "gripper_lpips":       gripper_lpips,
            "goal_mse":            goal_mse,
            "goal_lpips":          goal_lpips,
            "dynamic_mse":         dynamic_mse,
            "dynamic_lpips":       dynamic_lpips,
            "static_consistency_mse": static_mse,
            # Ranking fields filled in separately
            "correct_lpips":       full_lpips,
            "shuffled_lpips":      float("nan"),
            "lpips_gap":           float("nan"),
            "pairwise_win":        False,
        }

        # ---- Debug images (first window only or if explicitly requested) -----
        if save_debug and (window_id < 3):
            save_debug_images(
                output_dir      = os.path.join(debug_dir, f"task_{task_name}_win{window_id}"),
                current_image   = current_img,
                future_image    = future_gt_t[-1],
                residual_target = (future_gt_t[-1] - current_img),
                residual_pred   = residual_pred[-1],
                pred_future     = pred_future[-1],
                dynamic_mask    = dyn_mask[-1],
                roi_crop_current = _roi_crop_tensor(current_img, cy_v, cx_v, cfg.roi_crop_size),
                roi_crop_future  = _roi_crop_tensor(future_gt_t[-1], cy_v, cx_v, cfg.roi_crop_size),
                roi_crop_pred    = _roi_crop_tensor(pred_future[-1], cy_v, cx_v, cfg.roi_crop_size),
            )

        return row

    except Exception as exc:
        logger.warning("Error in window %d (task=%s): %s", window_id, task_name, exc)
        logger.debug(traceback.format_exc())
        return None


def _roi_crop_tensor(img: torch.Tensor, cy: float, cx: float, size: int) -> torch.Tensor:
    """[3, H, W] float → [3, size, size] float (bilinear crop)."""
    import torch.nn.functional as F
    H, W = img.shape[-2:]
    half = size // 2
    cy_px = int(cy * H)
    cx_px = int(cx * W)
    y0 = max(0, cy_px - half); y1 = min(H, cy_px + half)
    x0 = max(0, cx_px - half); x1 = min(W, cx_px + half)
    crop = img[:, y0:y1, x0:x1].unsqueeze(0)
    return F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False).squeeze(0)


# ---------------------------------------------------------------------------
# Ranking evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_rollout_lpips(
    model: PixelResidualWorldModel,
    pixels_t: torch.Tensor,    # [1, T+1, H, W, C]
    actions_t: torch.Tensor,   # [1, T, action_dim]
    horizon: int,
    lpips_fn,
    device: torch.device,
) -> float:
    """Return full-image LPIPS of predicted last future frame vs GT."""
    out = model.rollout(pixels_t, actions_t, horizon=horizon)
    pred_last = _to_uint8_np(out["pred_future"][0, -1])
    gt_last   = _to_uint8_np(out["future_gt"][0, -1])
    return float(lpips_fn(pred_last, gt_last))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = str(output_dir / "debug")

    condition_name = args.condition_name or Path(args.model_dir).name

    # Load model
    logger.info("Loading model from %s", args.model_dir)
    model = PixelResidualWorldModel.load_pretrained(args.model_dir).to(device).eval()
    cfg   = model.cfg
    logger.info("target_mode=%s  horizon=%d", cfg.target_mode, args.eval_horizon)

    # Save config
    import dataclasses
    with open(output_dir / "config_used.json", "w") as f:
        json.dump({"condition": condition_name,
                   "model_dir": args.model_dir,
                   "cfg": dataclasses.asdict(cfg)}, f, indent=2)

    # LPIPS
    lpips_fn = get_lpips_fn(device=str(device))

    # ROI config
    roi_config = load_roi_config()

    # Load LIBERO benchmark for task names
    try:
        from libero.libero import benchmark as libero_bench
        bench = libero_bench.get_benchmark_dict()
    except Exception:
        bench = {}

    # Resolve task indices
    dataset_name = resolve_dataset_name(args.task_suite)
    if args.task_indices:
        task_indices = [int(x.strip()) for x in args.task_indices.split(",") if x.strip()]
    else:
        # Enumerate tasks from benchmark
        task_indices = list(range(10))

    # Stream windows per task
    all_rows: List[Dict] = []
    windows_per_task = max(1, args.num_eval_windows // max(len(task_indices), 1))
    if args.dry_run_windows > 0:
        windows_per_task = min(windows_per_task, args.dry_run_windows)

    horizon = min(args.eval_horizon, cfg.action_horizon)

    logger.info("Evaluating %d tasks × %d windows = ~%d total",
                len(task_indices), windows_per_task, len(task_indices) * windows_per_task)

    for task_idx in task_indices:
        try:
            task_name = f"task{task_idx}"
            try:
                bench_key = f"libero_{args.task_suite}"
                if bench_key in bench:
                    task_names_list = bench[bench_key].get_task_names()
                    if task_idx < len(task_names_list):
                        task_name = task_names_list[task_idx]
            except Exception:
                pass

            goal_center = get_goal_roi_center(args.task_suite, task_idx, roi_config)
            roi_half    = get_roi_half(args.task_suite, task_idx, roi_config)

            logger.info("Task %d (%s): loading %d windows ...", task_idx, task_name, windows_per_task)

            ds = tfds.load(dataset_name, data_dir=args.data_root, split="train",
                           shuffle_files=False)
            ds_list = list(ds.take(50))  # cap episode count for speed
            random.shuffle(ds_list)

            window_count = 0
            for ep in ds_list:
                if window_count >= windows_per_task:
                    break
                steps = ep["steps"]
                imgs    = steps["observation"]["image"].numpy()     # [T_ep, H, W, C]
                acts    = steps["action"].numpy()                   # [T_ep, action_dim]
                T_ep = imgs.shape[0]
                seg = cfg.action_horizon + 2  # = T+1

                for start in range(0, T_ep - seg, max(1, seg // 2)):
                    if window_count >= windows_per_task:
                        break
                    pix_win = imgs[start: start + seg]              # [seg, H, W, C]
                    act_win = acts[start: start + seg - 1]          # [seg-1, action_dim]
                    if pix_win.shape[0] < seg:
                        continue

                    row = _eval_window(
                        model=model,
                        pixels_np=pix_win,
                        actions_np=act_win,
                        horizon=horizon,
                        device=device,
                        lpips_fn=lpips_fn,
                        goal_center=goal_center,
                        roi_half=roi_half,
                        cfg=cfg,
                        save_debug=args.save_debug_images,
                        debug_dir=debug_dir,
                        window_id=window_count,
                        task_name=task_name,
                    )
                    if row is None:
                        continue

                    # ---- Ranking: compare GT vs shuffled action -----
                    if window_count < args.num_ranking_windows:
                        pix_t = torch.from_numpy(pix_win).unsqueeze(0).to(device)
                        act_t = torch.from_numpy(act_win).unsqueeze(0).to(device)
                        correct_lpips = row["full_lpips"]

                        shuf_lpips_list = []
                        for _ in range(args.num_shuffle_reps):
                            perm = np.random.permutation(act_win.shape[0])
                            act_shuf = torch.from_numpy(act_win[perm]).unsqueeze(0).to(device)
                            lv = _compute_rollout_lpips(model, pix_t, act_shuf, horizon,
                                                        lpips_fn, device)
                            shuf_lpips_list.append(lv)

                        shuffled_lpips = float(np.mean(shuf_lpips_list))
                        lpips_gap = shuffled_lpips - correct_lpips
                        row["correct_lpips"]  = correct_lpips
                        row["shuffled_lpips"] = shuffled_lpips
                        row["lpips_gap"]      = lpips_gap
                        row["pairwise_win"]   = lpips_gap > 0

                    all_rows.append(row)
                    window_count += 1

            logger.info("  Task %d: %d windows evaluated.", task_idx, window_count)

        except Exception as exc:
            logger.warning("Error evaluating task %d: %s", task_idx, exc)
            logger.debug(traceback.format_exc())

    # ---- Aggregate and save ----
    agg = aggregate_phase1_metrics(all_rows, str(output_dir), condition_name)

    # Also save ranking_by_task.csv
    _save_ranking_by_task(all_rows, output_dir)

    logger.info("=== Phase 1 eval complete: %s ===", condition_name)
    logger.info("  full_mse=%.6f  gripper_mse=%.6f  pairwise_acc=%.4f  n_windows=%d",
                agg.get("full_mse", float("nan")),
                agg.get("gripper_mse", float("nan")),
                agg.get("pairwise_acc", float("nan")),
                agg.get("num_windows", 0))
    logger.info("  → %s", output_dir)


def _save_ranking_by_task(rows: List[Dict], output_dir: Path) -> None:
    by_task: Dict[str, Dict] = {}
    for row in rows:
        t = row.get("task_name", "unknown")
        if t not in by_task:
            by_task[t] = {"correct": [], "shuffled": [], "gap": [], "win": []}
        if not np.isnan(row.get("correct_lpips", float("nan"))):
            by_task[t]["correct"].append(row["correct_lpips"])
            by_task[t]["shuffled"].append(row["shuffled_lpips"])
            by_task[t]["gap"].append(row["lpips_gap"])
            by_task[t]["win"].append(float(row["pairwise_win"]))

    task_rows = []
    for t, d in sorted(by_task.items()):
        if not d["correct"]:
            continue
        task_rows.append({
            "task_name":        t,
            "correct_lpips_mean":  float(np.mean(d["correct"])),
            "shuffled_lpips_mean": float(np.mean(d["shuffled"])),
            "lpips_gap_mean":      float(np.mean(d["gap"])),
            "lpips_gap_min":       float(np.min(d["gap"])),
            "pairwise_acc":        float(np.mean(d["win"])),
            "reverse_windows":     int(sum(1 for v in d["win"] if v == 0)),
            "num_windows":         len(d["win"]),
        })

    if task_rows:
        with open(output_dir / "ranking_by_task.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
            w.writeheader()
            w.writerows(task_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
