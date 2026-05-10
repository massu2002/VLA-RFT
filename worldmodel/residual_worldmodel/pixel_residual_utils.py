"""Utilities for PixelResidualWorldModel.

Sections:
  1. Dynamic mask computation (pixel-diff based)
  2. Gripper ROI (motion center-of-mass crop)
  3. Masked loss helpers
  4. Debug image saver
  5. Phase 1 metrics aggregator (evaluation)
  6. LPIPS wrapper (lazy import)

All functions accept torch.Tensor images in float [0, 1] with shape [B, C, H, W]
unless noted otherwise.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Dynamic mask
# ===========================================================================

def compute_dynamic_mask(
    current_image: torch.Tensor,   # [B, 3, H, W] float [0, 1]
    future_image: torch.Tensor,    # [B, 3, H, W] float [0, 1]
    threshold: float = 0.05,
    dilate_kernel: int = 7,
) -> torch.Tensor:
    """Binary mask of pixels that changed significantly between frames.

    Algorithm:
      1. Compute per-pixel L1 difference averaged over channels.
      2. Threshold to create binary mask.
      3. Dilate with max-pool to include surrounding context.

    Returns:
        [B, 1, H, W] float32 binary mask (1 = dynamic, 0 = static).
    """
    # diff: [B, H, W]
    diff = (future_image.float() - current_image.float()).abs().mean(dim=1)
    mask = (diff > threshold).float().unsqueeze(1)   # [B, 1, H, W]

    if dilate_kernel > 1:
        k = dilate_kernel
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)

    return mask   # [B, 1, H, W]


# ===========================================================================
# 2. Gripper ROI (motion center-of-mass crop)
# ===========================================================================

def motion_center_of_mass(
    current_image: torch.Tensor,   # [B, 3, H, W] float [0, 1]
    future_image: torch.Tensor,    # [B, 3, H, W] float [0, 1]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (cy, cx) in [0, 1] for each batch item.

    Uses the channel-mean absolute difference as a weight map and computes
    the weighted centroid (center-of-mass of motion).

    Returns:
        cy [B] float — normalized row position in [0, 1]
        cx [B] float — normalized column position in [0, 1]
    """
    B, C, H, W = current_image.shape
    diff = (future_image.float() - current_image.float()).abs().mean(dim=1)  # [B, H, W]
    total = diff.sum(dim=[1, 2]).clamp(min=1e-6)  # [B]

    device = diff.device
    ys = torch.linspace(0, 1, H, device=device)  # [H]
    xs = torch.linspace(0, 1, W, device=device)  # [W]

    cy = (diff * ys[None, :, None]).sum(dim=[1, 2]) / total  # [B]
    cx = (diff * xs[None, None, :]).sum(dim=[1, 2]) / total  # [B]
    return cy, cx


def extract_roi_crops(
    images: torch.Tensor,           # [B, C, H, W]
    cy: torch.Tensor,               # [B] center row in [0, 1]
    cx: torch.Tensor,               # [B] center col in [0, 1]
    roi_size: int = 64,
) -> torch.Tensor:
    """Extract square ROI crops centred on (cy, cx) for each batch item.

    Crops are padded if the ROI extends outside the image boundary.
    All crops are bilinearly resized to roi_size × roi_size.

    Returns:
        [B, C, roi_size, roi_size] float
    """
    B, C, H, W = images.shape
    half = roi_size // 2
    crops = []
    for b in range(B):
        cy_px = int(cy[b].item() * H)
        cx_px = int(cx[b].item() * W)
        y0 = max(0, cy_px - half)
        y1 = min(H, cy_px + half)
        x0 = max(0, cx_px - half)
        x1 = min(W, cx_px + half)
        if y1 <= y0 or x1 <= x0:
            y0, y1, x0, x1 = 0, min(H, roi_size), 0, min(W, roi_size)
        crop = images[b:b+1, :, y0:y1, x0:x1]  # [1, C, h, w]
        crop_resized = F.interpolate(
            crop, size=(roi_size, roi_size), mode="bilinear", align_corners=False
        )
        crops.append(crop_resized)
    return torch.cat(crops, dim=0)  # [B, C, roi_size, roi_size]


# ===========================================================================
# 3. Masked loss helpers
# ===========================================================================

def masked_mse_loss(
    pred: torch.Tensor,    # [B, C, H, W]
    target: torch.Tensor,  # [B, C, H, W]
    mask: torch.Tensor,    # [B, 1, H, W] float, 1 = included
) -> torch.Tensor:
    """MSE loss restricted to masked pixels.

    Returns scalar.  Gracefully returns 0 if the mask is entirely zero.
    """
    total = mask.sum().clamp(min=1.0)
    diff_sq = ((pred - target.detach().to(pred.dtype)) ** 2) * mask.to(pred.dtype)
    return diff_sq.sum() / (total * pred.shape[1])


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """L1 loss restricted to masked pixels."""
    total = mask.sum().clamp(min=1.0)
    diff_abs = (pred - target.detach().to(pred.dtype)).abs() * mask.to(pred.dtype)
    return diff_abs.sum() / (total * pred.shape[1])


# ===========================================================================
# 4. Debug image saver
# ===========================================================================

def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] float or [H, W, C] → [H, W, C] uint8."""
    t = t.detach().cpu().float()
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    t = t.clamp(0, 1)
    return (t.numpy() * 255).astype(np.uint8)


def _residual_to_uint8(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] float residual → [H, W, C] uint8 (0 = -1, 128 = 0, 255 = +1)."""
    t = t.detach().cpu().float()
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    t = (t + 1.0) / 2.0
    return (t.clamp(0, 1).numpy() * 255).astype(np.uint8)


def save_debug_images(
    output_dir: str,
    current_image: torch.Tensor,           # [3, H, W] float [0, 1]
    future_image: torch.Tensor,            # [3, H, W] float [0, 1]
    residual_target: torch.Tensor,         # [3, H, W] float (can be negative)
    residual_pred: Optional[torch.Tensor], # [3, H, W] float or None
    pred_future: Optional[torch.Tensor],   # [3, H, W] float [0, 1] or None
    dynamic_mask: Optional[torch.Tensor],  # [1, H, W] float binary or None
    roi_crop_current: Optional[torch.Tensor],  # [3, roi, roi] or None
    roi_crop_future: Optional[torch.Tensor],   # [3, roi, roi] or None
    roi_crop_pred: Optional[torch.Tensor],     # [3, roi, roi] or None
    prefix: str = "sample",
) -> None:
    """Save per-sample debug images to *output_dir*."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        logger.warning("Pillow not available; cannot save debug images.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _save(arr: np.ndarray, name: str) -> None:
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        img = PILImage.fromarray(arr)
        img.save(os.path.join(output_dir, f"{prefix}_{name}.png"))

    _save(_tensor_to_uint8(current_image), "current_image")
    _save(_tensor_to_uint8(future_image), "future_image")
    _save(_residual_to_uint8(residual_target), "residual_target")

    if residual_pred is not None:
        _save(_residual_to_uint8(residual_pred), "residual_pred")
    if pred_future is not None:
        _save(_tensor_to_uint8(pred_future), "pred_future")
    if dynamic_mask is not None:
        _save((dynamic_mask[0].detach().cpu().float().numpy() * 255).astype(np.uint8)[:, :, None],
              "dynamic_mask")
    if roi_crop_current is not None:
        _save(_tensor_to_uint8(roi_crop_current), "gripper_roi_crop_current")
    if roi_crop_future is not None:
        _save(_tensor_to_uint8(roi_crop_future), "gripper_roi_crop_future")
    if roi_crop_pred is not None:
        _save(_tensor_to_uint8(roi_crop_pred), "gripper_roi_crop_pred")


# ===========================================================================
# 5. Phase 1 metrics aggregator
# ===========================================================================

def compute_image_metrics_np(
    pred: np.ndarray,   # [H, W, 3] uint8
    gt: np.ndarray,     # [H, W, 3] uint8
    lpips_fn=None,
) -> Dict[str, float]:
    """Compute MSE and LPIPS for a single image pair."""
    pred_f = pred.astype(np.float32) / 255.0
    gt_f   = gt.astype(np.float32)   / 255.0
    mse = float(np.mean((pred_f - gt_f) ** 2))
    lpips_val = float("nan")
    if lpips_fn is not None:
        try:
            lpips_val = float(lpips_fn(pred, gt))
        except Exception:
            pass
    return {"mse": mse, "lpips": lpips_val}


def aggregate_phase1_metrics(
    per_window_rows: List[Dict],
    output_dir: str,
    condition_name: str,
) -> Dict:
    """Aggregate per-window metrics into aggregate_metrics.json, csv files.

    per_window_rows: list of dicts with keys:
        task_name, window_id,
        full_mse, full_lpips,
        gripper_mse, gripper_lpips,
        goal_mse, goal_lpips,
        dynamic_mse, dynamic_lpips,
        static_consistency_mse,
        correct_lpips, shuffled_lpips, lpips_gap, pairwise_win

    Saves:
        {output_dir}/aggregate_metrics.json
        {output_dir}/metrics_by_task.csv
        {output_dir}/ranking_by_window.csv
        {output_dir}/config_used.json   (condition_name only; full config saved by caller)
    """
    import csv

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---- aggregate all windows ----
    scalar_keys = [
        "full_mse", "full_lpips",
        "gripper_mse", "gripper_lpips",
        "goal_mse", "goal_lpips",
        "dynamic_mse", "dynamic_lpips",
        "static_consistency_mse",
        "copy_current_mse", "copy_current_lpips",
        "copy_current_full_mse", "copy_current_full_lpips",
        "copy_current_gripper_mse", "copy_current_gripper_lpips",
        "copy_current_dynamic_mse", "copy_current_dynamic_lpips",
        "model_vs_copy_full_mse_delta",
        "model_vs_copy_gripper_mse_delta",
        "model_vs_copy_dynamic_mse_delta",
        "full_mse_over_copy_current_mse",
        "residual_abs_mean", "residual_abs_max",
        "write_mask_mean", "write_mask_max",
        "correct_lpips", "shuffled_lpips", "lpips_gap",
    ]

    agg: Dict[str, List[float]] = {k: [] for k in scalar_keys}
    pairwise_wins = 0
    num_windows = 0
    num_ranking_windows = 0

    for row in per_window_rows:
        num_windows += 1
        for k in scalar_keys:
            v = row.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                agg[k].append(float(v))
        is_ranked = row.get("shuffled_lpips") is not None
        if is_ranked and not (isinstance(row.get("shuffled_lpips"), float) and np.isnan(row.get("shuffled_lpips"))):
            num_ranking_windows += 1
            if row.get("pairwise_win"):
                pairwise_wins += 1

    def _mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    agg_metrics = {k: _mean(agg[k]) for k in scalar_keys}
    agg_metrics["pairwise_acc"] = pairwise_wins / max(num_ranking_windows, 1)
    agg_metrics["reverse_windows"] = num_ranking_windows - pairwise_wins
    agg_metrics["num_windows"] = num_windows
    agg_metrics["num_ranking_windows"] = num_ranking_windows

    # lpips_gap_min
    gaps = agg.get("lpips_gap", [])
    agg_metrics["lpips_gap_min"] = float(np.min(gaps)) if gaps else float("nan")

    with open(os.path.join(output_dir, "aggregate_metrics.json"), "w") as f:
        json.dump({"condition": condition_name, "metrics": agg_metrics}, f, indent=2)

    # ---- per-task breakdown ----
    by_task: Dict[str, Dict[str, List[float]]] = {}
    for row in per_window_rows:
        t = row.get("task_name", "unknown")
        if t not in by_task:
            by_task[t] = {k: [] for k in scalar_keys + ["pairwise_win"]}
        for k in scalar_keys:
            v = row.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                by_task[t][k].append(float(v))
        by_task[t]["pairwise_win"].append(float(row.get("pairwise_win", 0)))

    task_rows = []
    for t, d in sorted(by_task.items()):
        trow = {"task_name": t}
        for k in scalar_keys:
            trow[k] = _mean(d[k])
        trow["pairwise_acc"] = _mean(d["pairwise_win"])
        trow["num_windows"] = len(d["pairwise_win"])
        task_rows.append(trow)

    if task_rows:
        with open(os.path.join(output_dir, "metrics_by_task.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
            w.writeheader()
            w.writerows(task_rows)

    # ---- per-window ranking ----
    ranking_keys = [
        "task_name", "task_index", "window_id", "window_phase",
        "episode_length", "episode_file",
        "frame_indices", "action_indices",
        "correct_lpips", "shuffled_lpips", "lpips_gap", "pairwise_win",
    ]
    ranking_rows = []
    for row in per_window_rows:
        v = row.get("shuffled_lpips")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ranking_rows.append({k: row.get(k) for k in ranking_keys})
    if ranking_rows:
        with open(os.path.join(output_dir, "ranking_by_window.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ranking_keys)
            w.writeheader()
            w.writerows(ranking_rows)

    with open(os.path.join(output_dir, "config_used.json"), "w") as f:
        json.dump({"condition": condition_name}, f, indent=2)

    logger.info("[phase1_metrics] %s: pairwise_acc=%.4f  full_mse=%.6f  gripper_mse=%.6f  n=%d",
                condition_name,
                agg_metrics["pairwise_acc"],
                agg_metrics.get("full_mse", float("nan")),
                agg_metrics.get("gripper_mse", float("nan")),
                num_windows)

    return agg_metrics


# ===========================================================================
# 6. LPIPS wrapper (lazy import)
# ===========================================================================

_LPIPS_FN = None   # module-level singleton


def get_lpips_fn(device: str = "cpu", net: str = "alex"):
    """Return a cached LPIPS callable that accepts uint8 numpy [H, W, 3]."""
    global _LPIPS_FN
    if _LPIPS_FN is not None:
        return _LPIPS_FN

    try:
        import lpips as _lpips_lib
        _loss_fn = _lpips_lib.LPIPS(net=net, verbose=False).to(device).eval()
    except ImportError:
        logger.warning("lpips not installed; LPIPS metrics will be NaN.")
        _LPIPS_FN = lambda pred, gt: float("nan")  # noqa: E731
        return _LPIPS_FN

    def _lpips(pred_np: np.ndarray, gt_np: np.ndarray) -> float:
        """pred_np, gt_np: [H, W, 3] uint8 → scalar float."""
        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(arr).float() / 255.0  # [H, W, 3]
            t = t.permute(2, 0, 1).unsqueeze(0)        # [1, 3, H, W]
            return t.to(device) * 2 - 1                # [-1, 1] for LPIPS

        with torch.no_grad():
            score = _loss_fn(_to_tensor(pred_np), _to_tensor(gt_np))
        return float(score.item())

    _LPIPS_FN = _lpips
    return _LPIPS_FN
