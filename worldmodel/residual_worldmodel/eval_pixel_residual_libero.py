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
        --data-root /localdata/modified_libero_rlds \\
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
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


_FALLBACK_TASK_NAMES = {
    "spatial": [
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl on the ramekin and place it on the plate",
        "pick up the black bowl next to the cookie box and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl next to the plate and place it on the plate",
        "pick up the black bowl on the wooden cabinet and place it on the plate",
    ],
}


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
    parser.add_argument("--data-root", type=str, default="/localdata/modified_libero_rlds")
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
    parser.add_argument("--phase0-compatible", action="store_true",
                        default=os.environ.get("PHASE0_COMPATIBLE", "0") == "1",
                        help="Use Phase 0-compatible window/action/ranking protocol.")
    parser.add_argument("--window-position-mode", type=str,
                        default=os.environ.get("WINDOW_POSITION_MODE", "random"),
                        choices=["random", "episode_phases"],
                        help=(
                            "Window sampling protocol. 'random' preserves the existing behavior. "
                            "'episode_phases' samples early/middle/late windows from each episode."
                        ))
    parser.add_argument("--num-eval-episodes-per-task", type=int,
                        default=int(os.environ.get("NUM_EVAL_EPISODES_PER_TASK", "0")),
                        help=(
                            "When --window-position-mode=episode_phases, number of episodes per task. "
                            "0 derives it from --num-eval-windows."
                        ))

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


def _decode_bytes(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if hasattr(x, "decode"):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _normalize_name(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _episode_file_path(ep: Dict) -> str:
    meta = ep.get("episode_metadata", {})
    return _decode_bytes(meta.get("file_path", ""))


def _episode_matches_task(ep: Dict, task_name: str) -> bool:
    file_name = _normalize_name(os.path.basename(_episode_file_path(ep)))
    task_key = _normalize_name(task_name)
    return task_key in file_name


def _collect_task_windows(
    ds,
    task_name: str,
    cfg: PixelResidualConfig,
    windows_per_task: int,
    window_position_mode: str,
    episodes_per_task: int,
    phase0_compatible: bool,
    rng: random.Random,
    seed: int,
    eval_horizon: int,
) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Collect windows for one task.

    Non-compatible mode preserves the original Phase 1 sliding-window protocol.
    Phase0-compatible mode mirrors Phase 0 more closely: filter episodes by
    task name, sample one random horizon window per episode, and prepend one
    previous frame/action so PixelResidualWorldModel's internal frame_1/action_1
    alignment corresponds to Phase 0's current frame/action_0.
    """
    windows: List[Tuple[np.ndarray, np.ndarray, Dict]] = []
    episodes_iter = tfds.as_numpy(ds) if phase0_compatible else tfds.as_numpy(ds.take(50))
    episodes = list(episodes_iter)
    rng.shuffle(episodes)
    phase_episodes_added = 0

    for ep in episodes:
        if len(windows) >= windows_per_task and window_position_mode != "episode_phases":
            break
        if (
            window_position_mode == "episode_phases"
            and episodes_per_task > 0
            and phase_episodes_added >= episodes_per_task
        ):
            break
        if phase0_compatible and not _episode_matches_task(ep, task_name):
            continue

        steps = list(ep["steps"])
        imgs = np.stack([s["observation"]["image"] for s in steps], axis=0)
        acts = np.stack([s["action"] for s in steps], axis=0)
        T_ep = imgs.shape[0]

        if phase0_compatible:
            H = min(eval_horizon, cfg.action_horizon)
            # Need one prepended frame/action because model.rollout consumes
            # pixels[:, 1] as current and actions[:, 1] as action_0.
            if T_ep < H + 2:
                continue
            max_phase0_start = T_ep - H - 1

            if window_position_mode == "episode_phases":
                phase_starts = [
                    ("early", 1),
                    ("middle", 1 + (max_phase0_start - 1) // 2),
                    ("late", max_phase0_start),
                ]
            else:
                phase_starts = [("random", rng.randint(1, max_phase0_start))]

            used_starts = set()
            added_this_episode = 0
            for window_phase, phase0_start in phase_starts:
                if phase0_start in used_starts:
                    continue
                used_starts.add(phase0_start)
                pix_win = imgs[phase0_start - 1: phase0_start + H + 1]
                act_win = acts[phase0_start - 1: phase0_start + H]
                if pix_win.shape[0] == H + 2 and act_win.shape[0] == H + 1:
                    windows.append((
                        pix_win,
                        act_win,
                        {
                            "episode_file": _episode_file_path(ep),
                            "window_phase": window_phase,
                            "phase0_start": int(phase0_start),
                            "episode_length": int(T_ep),
                            "frame_indices": list(range(phase0_start, phase0_start + H + 1)),
                            "action_indices": list(range(phase0_start, phase0_start + H)),
                        },
                    ))
                    added_this_episode += 1
            if added_this_episode:
                phase_episodes_added += 1
            continue

        seg = cfg.action_horizon + 2
        max_start = T_ep - seg
        if max_start < 0:
            continue
        if window_position_mode == "episode_phases":
            phase_starts = [
                ("early", 0),
                ("middle", max_start // 2),
                ("late", max_start),
            ]
        else:
            phase_starts = [("random", start) for start in range(0, T_ep - seg, max(1, seg // 2))]

        used_starts = set()
        added_this_episode = 0
        for window_phase, start in phase_starts:
            if len(windows) >= windows_per_task and window_position_mode != "episode_phases":
                break
            if start in used_starts:
                continue
            used_starts.add(start)
            pix_win = imgs[start: start + seg]
            act_win = acts[start: start + seg - 1]
            if pix_win.shape[0] < seg:
                continue
            windows.append((
                pix_win,
                act_win,
                {
                    "episode_file": _episode_file_path(ep),
                    "window_phase": window_phase,
                    "phase1_start": int(start),
                    "episode_length": int(T_ep),
                    "frame_indices": list(range(start + 1, start + cfg.action_horizon + 2)),
                    "action_indices": list(range(start + 1, start + cfg.action_horizon + 1)),
                },
            ))
            added_this_episode += 1
        if window_position_mode == "episode_phases" and added_this_episode:
            phase_episodes_added += 1

    return windows


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
        write_mask    = out.get("write_mask")
        write_mask_h  = write_mask[0] if write_mask is not None else None

        current_np = _to_uint8_np(current_img)  # [H, W, 3]

        # Use LAST future step as representative
        pred_last = _to_uint8_np(pred_future[-1])
        gt_last   = _to_uint8_np(future_gt_t[-1])
        current_last = _to_uint8_np(current_img)

        # ---- Full-image metrics (last step) -----
        full_mse   = _mse(pred_last, gt_last)
        full_lpips = float(lpips_fn(pred_last, gt_last))
        copy_current_mse = _mse(current_last, gt_last)
        copy_current_lpips = float(lpips_fn(current_last, gt_last))

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
            "copy_current_mse":     copy_current_mse,
            "copy_current_lpips":   copy_current_lpips,
            "residual_abs_mean":    float(residual_pred[-1].detach().cpu().float().abs().mean()),
            "residual_abs_max":     float(residual_pred[-1].detach().cpu().float().abs().amax()),
            "write_mask_mean":      float(write_mask_h[-1].detach().cpu().float().mean()) if write_mask_h is not None else float("nan"),
            "write_mask_max":       float(write_mask_h[-1].detach().cpu().float().amax()) if write_mask_h is not None else float("nan"),
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
    terminal_only: bool = True,
) -> float:
    """Return full-image LPIPS for rollout.

    Phase 1's original ranking used only the terminal frame. Phase 0 action
    sensitivity uses the rollout average over all predicted future frames.
    """
    out = model.rollout(pixels_t, actions_t, horizon=horizon)
    if terminal_only:
        pred_last = _to_uint8_np(out["pred_future"][0, -1])
        gt_last   = _to_uint8_np(out["future_gt"][0, -1])
        return float(lpips_fn(pred_last, gt_last))

    vals = []
    for h in range(out["pred_future"].shape[1]):
        pred_h = _to_uint8_np(out["pred_future"][0, h])
        gt_h   = _to_uint8_np(out["future_gt"][0, h])
        vals.append(float(lpips_fn(pred_h, gt_h)))
    return float(np.mean(vals)) if vals else float("nan")


def _save_eval_protocol_config(
    args: argparse.Namespace,
    output_dir: Path,
    condition_name: str,
    cfg: PixelResidualConfig,
    task_indices: List[int],
    horizon: int,
    checkpoint_path: str,
) -> Dict:
    protocol = {
        "task_suite": args.task_suite,
        "selected_task_indices": task_indices,
        "num_eval_windows": args.num_eval_windows,
        "eval_horizon": args.eval_horizon,
        "effective_eval_horizon": horizon,
        "segment_length": horizon + 2 if args.phase0_compatible else cfg.action_horizon + 2,
        "action_start_offset": 0 if args.phase0_compatible else 1,
        "frame_index_alignment": (
            "phase0 current=sampled frame, target=future frames 1..H"
            if args.phase0_compatible
            else "phase1/model native current=pixels[1], target=pixels[2:]"
        ),
        "negative_type": (
            "same_task_other_window" if args.phase0_compatible else "same_window_temporal_permutation"
        ),
        "same_task_shuffle": bool(args.phase0_compatible),
        "shuffle_seed": args.seed,
        "window_seed": args.seed,
        "lpips_input_range": "[-1,1]",
        "image_range_before_lpips": "uint8 [0,255] converted to [0,1]",
        "use_terminal_frame_only": not bool(args.phase0_compatible),
        "ranking_gap_definition": "lpips_gap = shuffled_lpips - correct_lpips",
        "pairwise_unit": "window",
        "roi_crop_size": int(getattr(cfg, "roi_crop_size", 80)),
        "gripper_roi_method": "motion_center_of_mass(current, final_future)",
        "goal_roi_method": "configs/libero/roi_coords_v1.json with default center",
        "rollout_mode": "single_pass PixelResidualWorldModel.rollout",
        "checkpoint_path": checkpoint_path,
        "tokenizer_path": None,
        "target_mode": cfg.target_mode,
        "model_generation": getattr(cfg, "model_generation", ""),
        "condition_name": condition_name,
        "phase0_compatible": bool(args.phase0_compatible),
        "window_position_mode": args.window_position_mode,
        "num_eval_episodes_per_task": args.num_eval_episodes_per_task,
        "num_ranking_windows": args.num_ranking_windows,
        "num_shuffle_reps": args.num_shuffle_reps,
        "heldout_ratio": args.heldout_ratio,
        "split_mode": "all/train stream; phase0-compatible filters by task filename",
    }
    (output_dir / "eval_protocol_config.json").write_text(json.dumps(protocol, indent=2))
    return protocol


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
                   "phase0_compatible": bool(args.phase0_compatible),
                   "cfg": dataclasses.asdict(cfg)}, f, indent=2)

    # LPIPS
    lpips_fn = get_lpips_fn(device=str(device))

    # ROI config
    roi_config = load_roi_config()

    # Load LIBERO benchmark for task names
    try:
        from libero.libero import benchmark as libero_bench
        bench = libero_bench.get_benchmark_dict()
        bench_key = f"libero_{args.task_suite}"
        task_names_all = bench[bench_key]().get_task_names() if bench_key in bench else []
    except Exception:
        bench = {}
        task_names_all = []
    if not task_names_all:
        task_names_all = _FALLBACK_TASK_NAMES.get(args.task_suite, [])

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
    episodes_per_task = 0
    if args.window_position_mode == "episode_phases":
        episodes_per_task = args.num_eval_episodes_per_task
        if episodes_per_task <= 0:
            # Each selected episode contributes early/middle/late windows.
            episodes_per_task = max(1, int(np.ceil(windows_per_task / 3.0)))
        args.num_eval_episodes_per_task = episodes_per_task

    horizon = min(args.eval_horizon, cfg.action_horizon)
    protocol = _save_eval_protocol_config(
        args=args,
        output_dir=output_dir,
        condition_name=condition_name,
        cfg=cfg,
        task_indices=task_indices,
        horizon=horizon,
        checkpoint_path=args.model_dir,
    )

    if args.window_position_mode == "episode_phases":
        logger.info("Evaluating %d tasks × %d episodes × 3 phase windows = ~%d total",
                    len(task_indices), episodes_per_task, len(task_indices) * episodes_per_task * 3)
    else:
        logger.info("Evaluating %d tasks × %d windows = ~%d total",
                    len(task_indices), windows_per_task, len(task_indices) * windows_per_task)
    logger.info("phase0_compatible=%s negative_type=%s terminal_frame_only=%s window_position_mode=%s",
                args.phase0_compatible,
                protocol["negative_type"],
                protocol["use_terminal_frame_only"],
                args.window_position_mode)

    ds = tfds.load(dataset_name, data_dir=args.data_root, split="train",
                   shuffle_files=False)
    rng = random.Random(args.seed)
    ranked_windows = 0

    for task_idx in task_indices:
        try:
            task_name = f"task{task_idx}"
            try:
                if task_idx < len(task_names_all):
                    task_name = task_names_all[task_idx]
            except Exception:
                pass

            goal_center = get_goal_roi_center(args.task_suite, task_idx, roi_config)
            roi_half    = get_roi_half(args.task_suite, task_idx, roi_config)

            logger.info("Task %d (%s): loading %d windows ...", task_idx, task_name, windows_per_task)

            task_windows = _collect_task_windows(
                ds=ds,
                task_name=task_name,
                cfg=cfg,
                windows_per_task=windows_per_task,
                window_position_mode=args.window_position_mode,
                episodes_per_task=episodes_per_task,
                phase0_compatible=args.phase0_compatible,
                rng=rng,
                seed=args.seed,
                eval_horizon=args.eval_horizon,
            )
            if args.phase0_compatible and len(task_windows) < 2:
                logger.warning(
                    "Task %d has only %d phase0-compatible windows; ranking negatives may be unavailable.",
                    task_idx, len(task_windows)
                )

            window_count = 0
            for local_idx, (pix_win, act_win, win_meta) in enumerate(task_windows):

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
                row.update({
                    "task_index": task_idx,
                    "window_phase": win_meta.get("window_phase", "random"),
                    "episode_length": win_meta.get("episode_length", ""),
                    "episode_file": win_meta.get("episode_file", ""),
                    "frame_indices": json.dumps(win_meta.get("frame_indices", [])),
                    "action_indices": json.dumps(win_meta.get("action_indices", [])),
                })

                # ---- Ranking: compare GT vs shuffled action -----
                if ranked_windows < args.num_ranking_windows:
                    pix_t = torch.from_numpy(pix_win).unsqueeze(0).to(device)
                    act_t = torch.from_numpy(act_win).unsqueeze(0).to(device)
                    terminal_only = not bool(args.phase0_compatible)
                    correct_lpips = (
                        row["full_lpips"] if terminal_only
                        else _compute_rollout_lpips(
                            model, pix_t, act_t, horizon, lpips_fn, device,
                            terminal_only=False,
                        )
                    )

                    shuf_lpips_list = []
                    for rep in range(args.num_shuffle_reps):
                        if args.phase0_compatible and len(task_windows) > 1:
                            neg_idx = rng.randrange(len(task_windows) - 1)
                            if neg_idx >= local_idx:
                                neg_idx += 1
                            neg_act_win = task_windows[neg_idx][1]
                            act_shuf = torch.from_numpy(neg_act_win).unsqueeze(0).to(device)
                        else:
                            perm = np.random.permutation(act_win.shape[0])
                            act_shuf = torch.from_numpy(act_win[perm]).unsqueeze(0).to(device)
                        lv = _compute_rollout_lpips(
                            model, pix_t, act_shuf, horizon, lpips_fn, device,
                            terminal_only=terminal_only,
                        )
                        shuf_lpips_list.append(lv)

                    shuffled_lpips = float(np.mean(shuf_lpips_list))
                    lpips_gap = shuffled_lpips - correct_lpips
                    row["correct_lpips"]  = correct_lpips
                    row["shuffled_lpips"] = shuffled_lpips
                    row["lpips_gap"]      = lpips_gap
                    row["pairwise_win"]   = lpips_gap > 0
                    ranked_windows += 1

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
    _save_phase_breakdowns(all_rows, output_dir)

    # Re-save full configs after aggregation because older utility versions
    # wrote a minimal config_used.json.
    with open(output_dir / "config_used.json", "w") as f:
        json.dump({"condition": condition_name,
                   "model_dir": args.model_dir,
                   "phase0_compatible": bool(args.phase0_compatible),
                   "eval_protocol": protocol,
                   "cfg": dataclasses.asdict(cfg)}, f, indent=2)
    _save_eval_protocol_config(
        args=args,
        output_dir=output_dir,
        condition_name=condition_name,
        cfg=cfg,
        task_indices=task_indices,
        horizon=horizon,
        checkpoint_path=args.model_dir,
    )

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


def _save_phase_breakdowns(rows: List[Dict], output_dir: Path) -> None:
    """Save metrics/ranking grouped by episode phase.

    Phase labels are "early", "middle", "late" when
    --window-position-mode=episode_phases, and "random" for the legacy sampler.
    """
    scalar_keys = [
        "full_mse", "full_lpips",
        "gripper_mse", "gripper_lpips",
        "goal_mse", "goal_lpips",
        "dynamic_mse", "dynamic_lpips",
        "static_consistency_mse",
        "copy_current_mse", "copy_current_lpips",
        "residual_abs_mean", "residual_abs_max",
        "write_mask_mean", "write_mask_max",
    ]

    def _is_valid(v) -> bool:
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    def _collect(keys: Tuple[str, ...]) -> Dict[Tuple[str, ...], Dict[str, List[float]]]:
        grouped: Dict[Tuple[str, ...], Dict[str, List[float]]] = {}
        for row in rows:
            group_key = tuple(str(row.get(k, "unknown")) for k in keys)
            if group_key not in grouped:
                grouped[group_key] = {k: [] for k in scalar_keys + ["correct_lpips", "shuffled_lpips", "lpips_gap", "pairwise_win"]}
                grouped[group_key]["num_windows"] = []
            grouped[group_key]["num_windows"].append(1.0)
            for k in scalar_keys + ["correct_lpips", "shuffled_lpips", "lpips_gap"]:
                v = row.get(k)
                if _is_valid(v):
                    grouped[group_key][k].append(float(v))
            v = row.get("shuffled_lpips")
            if _is_valid(v):
                grouped[group_key]["pairwise_win"].append(float(row.get("pairwise_win", 0)))
        return grouped

    def _rows_for(grouped: Dict[Tuple[str, ...], Dict[str, List[float]]], keys: Tuple[str, ...]) -> List[Dict]:
        out = []
        for group_key, vals in sorted(grouped.items()):
            row = {k: v for k, v in zip(keys, group_key)}
            for k in scalar_keys:
                row[k] = _mean(vals[k])
            row["correct_lpips_mean"] = _mean(vals["correct_lpips"])
            row["shuffled_lpips_mean"] = _mean(vals["shuffled_lpips"])
            row["lpips_gap_mean"] = _mean(vals["lpips_gap"])
            row["lpips_gap_min"] = float(np.min(vals["lpips_gap"])) if vals["lpips_gap"] else float("nan")
            row["pairwise_acc"] = _mean(vals["pairwise_win"])
            row["reverse_windows"] = int(sum(1 for v in vals["pairwise_win"] if v == 0))
            row["num_windows"] = int(sum(vals["num_windows"]))
            row["num_ranking_windows"] = len(vals["pairwise_win"])
            out.append(row)
        return out

    phase_rows = _rows_for(_collect(("window_phase",)), ("window_phase",))
    if phase_rows:
        with open(output_dir / "metrics_by_phase.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(phase_rows[0].keys()))
            w.writeheader()
            w.writerows(phase_rows)

    task_phase_rows = _rows_for(_collect(("task_name", "window_phase")), ("task_name", "window_phase"))
    if task_phase_rows:
        with open(output_dir / "metrics_by_task_phase.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(task_phase_rows[0].keys()))
            w.writeheader()
            w.writerows(task_phase_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
