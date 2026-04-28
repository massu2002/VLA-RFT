"""Shared ROI metric and ranking JSONL utilities.

Used by both baseline worldmodel (worldmodel/libero/visualize.py) and
residual worldmodel (worldmodel/residual_worldmodel/focused_visualize.py)
to ensure identical metric keys and JSONL schema for fair comparison.

ROI metric key convention (same for both model families):
  roi/gripper_mse   roi/gripper_lpips  roi/gripper_psnr  roi/gripper_ssim
  roi/object_mse    roi/object_lpips   roi/object_psnr   roi/object_ssim
  roi/goal_mse      roi/goal_lpips     roi/goal_psnr     roi/goal_ssim
  roi/multi_step_gripper_mse    (list, len=horizon)
  roi/multi_step_gripper_lpips  (list, len=horizon)
  roi/multi_step_goal_mse       (list, len=horizon)
  roi/multi_step_goal_lpips     (list, len=horizon)

Ranking JSONL schema (null for fields unavailable in a given model family):
  {
    "step": int,
    "model_type": "baseline" | "residual" | "residual_focused",
    "task_name": str,
    "task_suite": str,
    "n_items": int,
    "K": int,
    "tiers": [str, ...],
    "modes": [str, ...],
    "metrics": { <key>: float | null, ... },
    "score_breakdown": {
      "dino_cosine": float | null,
      "image_l1": float | null,
      "combined": float | null
    },
    "per_item": [{"item_id": int, "scores": [float, ...]}],
    "saved_at": str
  }
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    logger.warning("skimage not available; PSNR/SSIM ROI metrics will be NaN.")


# ---------------------------------------------------------------------------
# ROI config loader
# ---------------------------------------------------------------------------

_DEFAULT_ROI_CONFIG: Dict = {}


def load_roi_config(config_path: Optional[str] = None) -> Dict:
    """Load per-task goal ROI coordinates from JSON.

    Returns an empty dict on failure; callers fall back to fixed defaults.
    """
    path = Path(config_path) if config_path else Path(
        __file__
    ).parent.parent / "configs" / "libero" / "roi_coords_v1.json"
    if not path.exists():
        logger.debug("roi_coords_v1.json not found at %s; using defaults.", path)
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not load ROI config %s: %s", path, exc)
        return {}


def get_goal_roi_center(
    task_suite: str,
    task_index: int,
    roi_config: Optional[Dict] = None,
) -> Tuple[float, float]:
    """Return (y, x) ∈ [0,1] for the goal ROI of a specific task.

    Priority: per-task override → suite default → global default (0.30, 0.50).
    """
    cfg = roi_config or _DEFAULT_ROI_CONFIG
    suite_cfg = cfg.get(task_suite, {})

    task_key = str(task_index)
    if task_key in suite_cfg:
        entry = suite_cfg[task_key]
        return float(entry.get("goal_roi_y", 0.30)), float(entry.get("goal_roi_x", 0.50))

    default = suite_cfg.get("_default", {})
    return float(default.get("goal_roi_y", 0.30)), float(default.get("goal_roi_x", 0.50))


def get_roi_half(task_suite: str, task_index: int, roi_config: Optional[Dict] = None) -> int:
    """Return roi_half_pixels for a task, falling back to 40."""
    cfg = roi_config or _DEFAULT_ROI_CONFIG
    suite_cfg = cfg.get(task_suite, {})
    task_key = str(task_index)
    if task_key in suite_cfg:
        return int(suite_cfg[task_key].get("roi_half_pixels", 40))
    default = suite_cfg.get("_default", {})
    return int(default.get("roi_half_pixels", 40))


# ---------------------------------------------------------------------------
# Numpy ROI helpers (baseline-compatible, no torch dependency)
# ---------------------------------------------------------------------------

def motion_com_np(
    frame_a: np.ndarray,   # [H, W, 3] uint8
    frame_b: np.ndarray,   # [H, W, 3] uint8
) -> Tuple[float, float]:
    """Center-of-mass of |frame_b - frame_a|, normalised to (y, x) ∈ [0, 1].

    Used as gripper / object proxy when a focus map is not available (baseline).
    """
    diff = np.abs(frame_b.astype(np.float32) - frame_a.astype(np.float32))
    weight = diff.mean(axis=2)  # [H, W]
    total = float(weight.sum())
    if total < 1e-6:
        return 0.5, 0.5
    H, W = weight.shape
    ys = np.arange(H, dtype=np.float32) / max(H - 1, 1)
    xs = np.arange(W, dtype=np.float32) / max(W - 1, 1)
    com_y = float((weight * ys[:, None]).sum() / total)
    com_x = float((weight * xs[None, :]).sum() / total)
    return com_y, com_x


def roi_crop_np(
    frame: np.ndarray,    # [H, W, 3] uint8 or float
    cy_frac: float,       # center y ∈ [0, 1]
    cx_frac: float,       # center x ∈ [0, 1]
    half_px: int,         # half-size in pixels
) -> np.ndarray:
    """Return a square crop of ``frame`` centred on (cy_frac, cx_frac)."""
    H, W = frame.shape[:2]
    cy = int(cy_frac * H)
    cx = int(cx_frac * W)
    y0 = max(0, cy - half_px);  y1 = min(H, cy + half_px)
    x0 = max(0, cx - half_px);  x1 = min(W, cx + half_px)
    if y1 <= y0 or x1 <= x0:
        return frame
    return frame[y0:y1, x0:x1]


def _psnr_np(gt: np.ndarray, pred: np.ndarray) -> float:
    if not _HAS_SKIMAGE:
        return float("nan")
    gt_f  = gt.astype(np.float32)  / 255.0 if gt.dtype == np.uint8  else gt.astype(np.float32)
    pred_f = pred.astype(np.float32) / 255.0 if pred.dtype == np.uint8 else pred.astype(np.float32)
    try:
        return float(peak_signal_noise_ratio(gt_f, pred_f, data_range=1.0))
    except Exception:
        return float("nan")


def _ssim_np(gt: np.ndarray, pred: np.ndarray) -> float:
    if not _HAS_SKIMAGE:
        return float("nan")
    gt_f  = gt.astype(np.float32)  / 255.0 if gt.dtype == np.uint8  else gt.astype(np.float32)
    pred_f = pred.astype(np.float32) / 255.0 if pred.dtype == np.uint8 else pred.astype(np.float32)
    try:
        return float(structural_similarity(gt_f, pred_f, channel_axis=2, data_range=1.0))
    except Exception:
        return float("nan")


def compute_roi_metrics_np(
    pred_frames: List[np.ndarray],          # [H, W, 3] uint8, len = horizon
    gt_frames:   List[np.ndarray],          # [H, W, 3] uint8, len = horizon
    lpips_fn,                                # callable(pred_t, gt_t) -> float
    gripper_center: Tuple[float, float],    # (y, x) ∈ [0, 1]
    goal_center:    Tuple[float, float],    # (y, x) ∈ [0, 1]
    roi_half: int = 40,
) -> Dict:
    """Compute ROI metrics for a single rollout window.

    Returns a dict with per-frame lists (for multi-step) and scalar averages.
    Keys use the shared convention:
      roi/<region>_<metric>  (scalar average)
      roi/multi_step_<region>_<metric>  (list, one value per frame)
    """
    n = min(len(pred_frames), len(gt_frames))
    if n == 0:
        return {}

    gripper_mse_list:   List[float] = []
    gripper_psnr_list:  List[float] = []
    gripper_ssim_list:  List[float] = []
    gripper_lpips_list: List[float] = []
    goal_mse_list:      List[float] = []
    goal_psnr_list:     List[float] = []
    goal_ssim_list:     List[float] = []
    goal_lpips_list:    List[float] = []

    for i in range(n):
        pred = pred_frames[i]
        gt   = gt_frames[i]

        # gripper crop
        g_pred = roi_crop_np(pred, gripper_center[0], gripper_center[1], roi_half)
        g_gt   = roi_crop_np(gt,   gripper_center[0], gripper_center[1], roi_half)
        gp_f   = g_pred.astype(np.float32) / 255.0
        gg_f   = g_gt.astype(np.float32)   / 255.0

        gripper_mse_list.append(float(np.mean((gp_f - gg_f) ** 2)))
        gripper_psnr_list.append(_psnr_np(g_gt, g_pred))
        gripper_ssim_list.append(_ssim_np(g_gt, g_pred))
        try:
            gripper_lpips_list.append(float(lpips_fn(g_pred, g_gt)))
        except Exception:
            gripper_lpips_list.append(float("nan"))

        # goal crop
        l_pred = roi_crop_np(pred, goal_center[0], goal_center[1], roi_half)
        l_gt   = roi_crop_np(gt,   goal_center[0], goal_center[1], roi_half)
        lp_f   = l_pred.astype(np.float32) / 255.0
        lg_f   = l_gt.astype(np.float32)   / 255.0

        goal_mse_list.append(float(np.mean((lp_f - lg_f) ** 2)))
        goal_psnr_list.append(_psnr_np(l_gt, l_pred))
        goal_ssim_list.append(_ssim_np(l_gt, l_pred))
        try:
            goal_lpips_list.append(float(lpips_fn(l_pred, l_gt)))
        except Exception:
            goal_lpips_list.append(float("nan"))

    def _mean(lst: List[float]) -> float:
        valid = [v for v in lst if not (v != v)]   # drop NaN
        return float(np.mean(valid)) if valid else float("nan")

    return {
        # scalars (averages over horizon)
        "roi/gripper_mse":   _mean(gripper_mse_list),
        "roi/gripper_lpips": _mean(gripper_lpips_list),
        "roi/gripper_psnr":  _mean(gripper_psnr_list),
        "roi/gripper_ssim":  _mean(gripper_ssim_list),
        "roi/goal_mse":      _mean(goal_mse_list),
        "roi/goal_lpips":    _mean(goal_lpips_list),
        "roi/goal_psnr":     _mean(goal_psnr_list),
        "roi/goal_ssim":     _mean(goal_ssim_list),
        # multi-step per-frame lists
        "roi/multi_step_gripper_mse":    gripper_mse_list,
        "roi/multi_step_gripper_lpips":  gripper_lpips_list,
        "roi/multi_step_goal_mse":       goal_mse_list,
        "roi/multi_step_goal_lpips":     goal_lpips_list,
    }


# ---------------------------------------------------------------------------
# Shared JSONL appender (baseline + residual compatible)
# ---------------------------------------------------------------------------

#: Fixed set of metric keys that appear in every JSONL line (null if absent).
RANKING_METRIC_KEYS: List[str] = [
    "strict_order_acc",
    "pairwise_acc",
    "top1_acc",
    "mean_margin",
    "pos_score_mean",
    "neg_score_mean",
    "hardest_negative_margin",
    "acc_success_gt_nearsuccess",
    "acc_nearsuccess_gt_failure",
    "acc_success_gt_failure",
    "spearman_tier_corr",
    "margin_success_minus_nearsuccess",
    "margin_nearsuccess_minus_failure",
    "margin_success_minus_failure",
    "tier_score_success",
    "tier_score_nearsuccess",
    "tier_score_failure",
]

#: Fixed score-breakdown keys (null for unavailable models).
SCORE_BREAKDOWN_KEYS: List[str] = ["dino_cosine", "image_l1", "combined"]


def _normalise_metrics(metrics: Dict) -> Dict:
    """Ensure all RANKING_METRIC_KEYS are present; fill missing with None."""
    return {k: metrics.get(k) for k in RANKING_METRIC_KEYS}


def _normalise_score_breakdown(breakdown: Optional[Dict]) -> Dict:
    if not breakdown:
        return {k: None for k in SCORE_BREAKDOWN_KEYS}
    return {k: breakdown.get(k) for k in SCORE_BREAKDOWN_KEYS}


def append_ranking_jsonl(
    out_path: Path,
    step: int,
    model_type: str,                      # "baseline" | "residual" | "residual_focused"
    task_name: str,
    task_suite: str,
    n_items: int,
    K: int,
    tiers: List[str],
    modes: List[str],
    metrics: Dict,
    per_item: List[Dict],                 # [{"item_id": int, "scores": [float, ...]}]
    score_breakdown: Optional[Dict] = None,
) -> None:
    """Append one JSON line to *out_path* (creating/appending the file).

    Both baseline and residual call this function; fields that a model cannot
    populate are set to ``null`` via ``_normalise_metrics``.
    """
    record = {
        "step":            step,
        "model_type":      model_type,
        "task_name":       task_name,
        "task_suite":      task_suite,
        "n_items":         n_items,
        "K":               K,
        "tiers":           tiers,
        "modes":           modes,
        "metrics":         _normalise_metrics(metrics),
        "score_breakdown": _normalise_score_breakdown(score_breakdown),
        "per_item":        per_item,
        "saved_at":        datetime.datetime.utcnow().isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.info("Appended ranking JSONL → %s (step=%d, n=%d)", out_path, step, n_items)
