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
import hashlib
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
    parser.add_argument("--action-ablation", action="store_true",
                        default=os.environ.get("ACTION_ABLATION", "0") == "1",
                        help="Evaluate correct/shuffle/zero/random/permuted actions.")
    parser.add_argument("--save-debug-visuals", action="store_true",
                        default=os.environ.get("SAVE_DEBUG_VISUALS", "0") == "1",
                        help="Save residual/write-mask/action-condition debug visuals.")
    parser.add_argument("--debug-num-tasks", type=int,
                        default=int(os.environ.get("DEBUG_NUM_TASKS", "3")))
    parser.add_argument("--debug-windows-per-task", type=int,
                        default=int(os.environ.get("DEBUG_WINDOWS_PER_TASK", "3")))
    parser.add_argument("--window-manifest", type=str,
                        default=os.environ.get("WINDOW_MANIFEST", ""),
                        help="Existing window_manifest.json to reuse.")
    parser.add_argument("--use-window-manifest", action="store_true",
                        default=os.environ.get("USE_WINDOW_MANIFEST", "0") == "1",
                        help="Use --window-manifest instead of sampling new windows.")
    parser.add_argument("--allow-protocol-mismatch", action="store_true",
                        default=os.environ.get("ALLOW_PROTOCOL_MISMATCH", "0") == "1",
                        help="Warn instead of failing when manifest protocol differs.")

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


def _masked_mse_uint8(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mask_f = mask.astype(np.float32)
    if mask_f.sum() <= 0:
        return float("nan")
    diff = (a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0) ** 2
    return float((diff * mask_f[:, :, None]).sum() / (mask_f.sum() * a.shape[-1]))


def _lpips_safe(lpips_fn, a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(lpips_fn(a, b))
    except Exception:
        return float("nan")


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


def _episode_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


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
    episodes = list(enumerate(episodes_iter))
    rng.shuffle(episodes)
    phase_episodes_added = 0

    for episode_index, ep in episodes:
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
                            "episode_index": int(episode_index),
                            "episode_id": _episode_id_from_path(_episode_file_path(ep)),
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
                    "episode_index": int(episode_index),
                    "episode_id": _episode_id_from_path(_episode_file_path(ep)),
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


def _manifest_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _protocol_for_manifest(args: argparse.Namespace, task_indices: List[int], horizon: int) -> Dict:
    return {
        "task_suite": args.task_suite,
        "selected_task_indices": task_indices,
        "num_eval_windows": args.num_eval_windows,
        "eval_horizon": horizon,
        "segment_length": horizon + 2,
        "action_start_offset": 0,
        "negative_type": "same_task_other_window" if args.phase0_compatible else "same_window_temporal_permutation",
        "same_task_shuffle": bool(args.phase0_compatible),
        "window_position_mode": args.window_position_mode,
        "num_eval_episodes_per_task": args.num_eval_episodes_per_task,
        "window_seed": args.seed,
        "shuffle_seed": args.seed,
        "phase0_compatible": bool(args.phase0_compatible),
        "lpips_input_range": "[-1,1]",
        "use_terminal_frame_only": not bool(args.phase0_compatible),
        "pairwise_unit": "window",
        "roi_crop_size": 64,
        "gripper_roi_method": "motion_center_of_mass(current, final_future)",
    }


def _make_manifest_record(
    global_id: int,
    task_id: int,
    task_name: str,
    local_id: int,
    win_meta: Dict,
    horizon: int,
) -> Dict:
    current = int(win_meta.get("phase0_start", int(win_meta.get("phase1_start", 0)) + 1))
    return {
        "global_window_id": int(global_id),
        "task_id": int(task_id),
        "task_name": task_name,
        "local_window_id": int(local_id),
        "episode_id": win_meta.get("episode_id", _episode_id_from_path(win_meta.get("episode_file", ""))),
        "episode_index": int(win_meta.get("episode_index", -1)),
        "episode_file": win_meta.get("episode_file", ""),
        "episode_length": int(win_meta.get("episode_length", 0) or 0),
        "window_position": win_meta.get("window_phase", "random"),
        "current_frame_index": current,
        "future_frame_indices": list(range(current + 1, current + horizon + 1)),
        "action_indices": list(range(current, current + horizon)),
        "negative_type": "same_task_other_window",
        "negative_task_id": None,
        "negative_episode_id": None,
        "negative_window_id": None,
        "negative_current_frame_index": None,
        "negative_action_indices": [],
        "roi_crop_size": 64,
        "gripper_roi_method": "motion_center_of_mass(current, final_future)",
    }


def _assign_manifest_negatives(records: List[Dict], rng: random.Random) -> None:
    by_task: Dict[int, List[Dict]] = {}
    for rec in records:
        by_task.setdefault(int(rec["task_id"]), []).append(rec)
    for task_id, arr in by_task.items():
        for self_pos, rec in enumerate(arr):
            if len(arr) <= 1:
                neg = rec
            else:
                neg_pos = rng.randrange(len(arr) - 1)
                if neg_pos >= self_pos:
                    neg_pos += 1
                neg = arr[neg_pos]
            assert int(neg["task_id"]) == int(task_id)
            rec["negative_task_id"] = int(neg["task_id"])
            rec["negative_episode_id"] = neg.get("episode_id")
            rec["negative_window_id"] = int(neg["global_window_id"])
            rec["negative_current_frame_index"] = int(neg["current_frame_index"])
            rec["negative_action_indices"] = list(neg["action_indices"])


def _save_window_manifest(output_dir: Path, protocol: Dict, records: List[Dict]) -> Tuple[Path, str]:
    task_counts: Dict[str, int] = {}
    for rec in records:
        key = str(rec["task_id"])
        task_counts[key] = task_counts.get(key, 0) + 1
    payload = {
        "protocol": protocol,
        "num_windows": len(records),
        "windows_per_task": task_counts,
        "windows": records,
    }
    json_path = output_dir / "window_manifest.json"
    jsonl_path = output_dir / "window_manifest.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    digest = _manifest_hash(json_path)
    (output_dir / "window_manifest_hash.txt").write_text(digest + "\n", encoding="utf-8")
    return json_path, digest


def _load_window_manifest(path: str) -> Tuple[Dict, str]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data, _manifest_hash(p)


def _check_manifest_protocol(
    manifest: Dict,
    args: argparse.Namespace,
    task_indices: List[int],
    horizon: int,
) -> None:
    proto = manifest.get("protocol", {})
    expected = _protocol_for_manifest(args, task_indices, horizon)
    keys = [
        "task_suite",
        "eval_horizon",
        "action_start_offset",
        "negative_type",
        "phase0_compatible",
    ]
    mismatches = []
    for key in keys:
        if proto.get(key) != expected.get(key):
            mismatches.append(f"{key}: manifest={proto.get(key)!r} current={expected.get(key)!r}")
    if sorted(proto.get("selected_task_indices", [])) != sorted(task_indices):
        mismatches.append(
            f"selected_task_indices: manifest={proto.get('selected_task_indices')!r} current={task_indices!r}"
        )
    if mismatches and not args.allow_protocol_mismatch:
        raise ValueError("Window manifest protocol mismatch:\n  " + "\n  ".join(mismatches))
    if mismatches:
        logger.warning("Window manifest protocol mismatch allowed:\n  %s", "\n  ".join(mismatches))


def _build_episode_lookup(ds) -> Dict[str, Dict]:
    lookup: Dict[str, Dict] = {}
    for idx, ep in enumerate(tfds.as_numpy(ds)):
        path = _episode_file_path(ep)
        lookup[path] = ep
        lookup[os.path.basename(path)] = ep
        lookup[_episode_id_from_path(path)] = ep
        lookup[str(idx)] = ep
    return lookup


def _slice_window_from_episode(ep: Dict, current: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    steps = list(ep["steps"])
    imgs = np.stack([s["observation"]["image"] for s in steps], axis=0)
    acts = np.stack([s["action"] for s in steps], axis=0)
    pix_win = imgs[current - 1: current + horizon + 1]
    act_win = acts[current - 1: current + horizon]
    if pix_win.shape[0] != horizon + 2 or act_win.shape[0] != horizon + 1:
        raise ValueError(
            f"Bad manifest slice current={current}, horizon={horizon}, "
            f"pix={pix_win.shape}, act={act_win.shape}"
        )
    return pix_win, act_win


def _windows_from_manifest(ds, manifest: Dict, horizon: int) -> Dict[int, List[Tuple[np.ndarray, np.ndarray, Dict]]]:
    lookup = _build_episode_lookup(ds)
    out: Dict[int, List[Tuple[np.ndarray, np.ndarray, Dict]]] = {}
    for rec in manifest.get("windows", []):
        ep = (
            lookup.get(str(rec.get("episode_index", "")))
            or lookup.get(rec.get("episode_file", ""))
            or lookup.get(os.path.basename(rec.get("episode_file", "")))
            or lookup.get(str(rec.get("episode_id", "")))
        )
        if ep is None:
            raise KeyError(
                f"Could not find episode for manifest row {rec.get('global_window_id')}: "
                f"{rec.get('episode_file')}"
            )
        pix_win, act_win = _slice_window_from_episode(ep, int(rec["current_frame_index"]), horizon)
        meta = {
            "episode_file": rec.get("episode_file", ""),
            "episode_index": rec.get("episode_index", -1),
            "episode_id": rec.get("episode_id", ""),
            "window_phase": rec.get("window_position", "manifest"),
            "phase0_start": int(rec["current_frame_index"]),
            "episode_length": rec.get("episode_length", ""),
            "frame_indices": [int(rec["current_frame_index"])] + [int(x) for x in rec.get("future_frame_indices", [])],
            "action_indices": [int(x) for x in rec.get("action_indices", [])],
            "manifest_record": rec,
        }
        out.setdefault(int(rec["task_id"]), []).append((pix_win, act_win, meta))
    return out


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
        g_copy = roi_crop_np(current_last, cy_v, cx_v, roi_half)
        gripper_mse   = _mse(g_pred, g_gt)
        gripper_lpips = float(lpips_fn(g_pred, g_gt))
        copy_current_gripper_mse   = _mse(g_copy, g_gt)
        copy_current_gripper_lpips = _lpips_safe(lpips_fn, g_copy, g_gt)

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
        copy_current_dynamic_mse = float("nan")
        static_mse  = float("nan")
        if dm_np.sum() > 0:
            dyn_mask_ch = torch.from_numpy(dm_np).unsqueeze(0)  # [1, H, W]
            dyn_mask_ch3 = dyn_mask_ch.unsqueeze(0).expand(1, 3, -1, -1)  # [1, 3, H, W]
            p_dyn = (pred_last_f.unsqueeze(0) * dyn_mask_ch).sum() / dyn_mask_ch3.sum().clamp(1)
            g_dyn = (gt_last_f.unsqueeze(0)   * dyn_mask_ch).sum() / dyn_mask_ch3.sum().clamp(1)
            # Per-pixel MSE in dynamic region
            dyn_diff = ((pred_last_f - gt_last_f) ** 2) * dyn_mask_ch
            dynamic_mse = float(dyn_diff.sum() / (dyn_mask_ch.sum() * 3).clamp(1))
            copy_dyn_diff = ((curr_last_f - gt_last_f) ** 2) * dyn_mask_ch
            copy_current_dynamic_mse = float(copy_dyn_diff.sum() / (dyn_mask_ch.sum() * 3).clamp(1))

            static_mask  = 1.0 - dyn_mask_ch
            static_diff  = ((pred_last_f - curr_last_f) ** 2) * static_mask
            static_mse   = float(static_diff.sum() / (static_mask.sum() * 3).clamp(1))

        # ---- Dynamic region LPIPS (if region large enough) -----
        dynamic_lpips = float("nan")
        copy_current_dynamic_lpips = float("nan")
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
                    c_crop = current_last[rmin:rmax+1, cmin:cmax+1]
                    copy_current_dynamic_lpips = float(lpips_fn(c_crop, g_crop))
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
            "copy_current_full_mse": copy_current_mse,
            "copy_current_full_lpips": copy_current_lpips,
            "copy_current_gripper_mse": copy_current_gripper_mse,
            "copy_current_gripper_lpips": copy_current_gripper_lpips,
            "copy_current_dynamic_mse": copy_current_dynamic_mse,
            "copy_current_dynamic_lpips": copy_current_dynamic_lpips,
            "model_vs_copy_full_mse_delta": full_mse - copy_current_mse,
            "model_vs_copy_gripper_mse_delta": gripper_mse - copy_current_gripper_mse,
            "model_vs_copy_dynamic_mse_delta": (
                dynamic_mse - copy_current_dynamic_mse
                if not np.isnan(dynamic_mse) and not np.isnan(copy_current_dynamic_mse)
                else float("nan")
            ),
            "full_mse_over_copy_current_mse": full_mse / max(copy_current_mse, 1e-12),
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


@torch.no_grad()
def _rollout_terminal_bundle(
    model: PixelResidualWorldModel,
    pixels_t: torch.Tensor,
    actions_t: torch.Tensor,
    horizon: int,
) -> Dict:
    out = model.rollout(pixels_t, actions_t, horizon=horizon)
    return {
        "out": out,
        "pred": _to_uint8_np(out["pred_future"][0, -1]),
        "gt": _to_uint8_np(out["future_gt"][0, -1]),
        "current": _to_uint8_np(out["current_image"][0]),
        "residual": out["residual_pred"][0, -1].detach().cpu().float(),
        "write_mask": (
            out["write_mask"][0, -1].detach().cpu().float()
            if out.get("write_mask") is not None else None
        ),
        "dynamic_mask": out["dynamic_mask"][0, -1].detach().cpu().float(),
    }


def _condition_metrics(
    bundle: Dict,
    correct_bundle: Dict,
    lpips_fn,
    goal_center: tuple,
    roi_half: int,
) -> Dict[str, float]:
    pred = bundle["pred"]
    gt = bundle["gt"]
    current = bundle["current"]
    dm = bundle["dynamic_mask"][0].numpy()
    cy_t, cx_t = motion_center_of_mass(
        torch.from_numpy(current).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
        torch.from_numpy(gt).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
    )
    cy_v, cx_v = cy_t[0].item(), cx_t[0].item()
    g_pred = roi_crop_np(pred, cy_v, cx_v, roi_half)
    g_gt = roi_crop_np(gt, cy_v, cx_v, roi_half)
    dyn_mse = _masked_mse_uint8(pred, gt, dm) if dm.sum() > 0 else float("nan")
    dyn_lpips = float("nan")
    if dm.sum() > 64:
        mask_bool = dm > 0.5
        rows = np.any(mask_bool, axis=1)
        cols = np.any(mask_bool, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            dyn_lpips = _lpips_safe(lpips_fn, pred[rmin:rmax+1, cmin:cmax+1], gt[rmin:rmax+1, cmin:cmax+1])
    return {
        "full_mse": _mse(pred, gt),
        "full_lpips": _lpips_safe(lpips_fn, pred, gt),
        "gripper_mse": _mse(g_pred, g_gt),
        "gripper_lpips": _lpips_safe(lpips_fn, g_pred, g_gt),
        "dynamic_mse": dyn_mse,
        "dynamic_lpips": dyn_lpips,
        "pred_vs_correct_mse": _mse(pred, correct_bundle["pred"]),
        "pred_vs_correct_lpips": _lpips_safe(lpips_fn, pred, correct_bundle["pred"]),
    }


def _make_action_variants(
    act_win: np.ndarray,
    task_windows: List[Tuple[np.ndarray, np.ndarray, Dict]],
    local_idx: int,
    rng: random.Random,
) -> Dict[str, np.ndarray]:
    variants = {"correct": act_win}
    if len(task_windows) > 1:
        rec = task_windows[local_idx][2].get("manifest_record", {})
        neg_id = rec.get("negative_window_id")
        neg_act = None
        if neg_id is not None:
            for _, cand_act, cand_meta in task_windows:
                cand_rec = cand_meta.get("manifest_record", {})
                if int(cand_rec.get("global_window_id", -1)) == int(neg_id):
                    neg_act = cand_act
                    break
        if neg_act is None:
            neg_idx = rng.randrange(len(task_windows) - 1)
            if neg_idx >= local_idx:
                neg_idx += 1
            neg_act = task_windows[neg_idx][1]
        variants["same_task_shuffle"] = neg_act
    else:
        variants["same_task_shuffle"] = act_win[np.random.permutation(act_win.shape[0])]
    variants["zero_action"] = np.zeros_like(act_win)
    variants["random_action"] = rng.random() * np.ones_like(act_win)
    lo = np.nanmin(act_win, axis=0, keepdims=True)
    hi = np.nanmax(act_win, axis=0, keepdims=True)
    variants["random_action"] = np.asarray(
        np.random.default_rng(rng.randint(0, 2**31 - 1)).uniform(lo, hi, size=act_win.shape),
        dtype=act_win.dtype,
    )
    perm = np.arange(act_win.shape[0])
    if len(perm) > 1:
        rng.shuffle(perm)
    variants["temporal_permutation"] = act_win[perm]
    return variants


def _save_action_ablation_outputs(rows: List[Dict], output_dir: Path) -> None:
    if not rows:
        return
    out = output_dir / "action_ablation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "action_ablation_by_window.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_task_cond: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for r in rows:
        key = (str(r["task_id"]), str(r["condition"]))
        by_task_cond.setdefault(key, {k: [] for k in ["full_mse", "full_lpips", "gripper_mse", "gripper_lpips", "dynamic_mse", "dynamic_lpips", "pred_vs_correct_mse", "pred_vs_correct_lpips"]})
        for k in by_task_cond[key]:
            v = r.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                by_task_cond[key][k].append(float(v))
    task_rows = []
    for (task_id, cond), vals in sorted(by_task_cond.items()):
        row = {"task_id": task_id, "condition": cond, "num_windows": len(next(iter(vals.values()), []))}
        for k, xs in vals.items():
            row[k] = float(np.mean(xs)) if xs else float("nan")
        task_rows.append(row)
    if task_rows:
        with open(out / "action_ablation_by_task.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
            w.writeheader()
            w.writerows(task_rows)

    window_groups: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        key = (str(r["task_id"]), str(r["window_id"]))
        window_groups.setdefault(key, {})[str(r["condition"])] = float(r["full_lpips"])
    correct_best = 0
    total = 0
    for conds in window_groups.values():
        if "correct" not in conds:
            continue
        total += 1
        if conds["correct"] <= min(conds.values()):
            correct_best += 1

    def _mean_cond(cond: str, key: str) -> float:
        vals = [float(r[key]) for r in rows if r["condition"] == cond and r.get(key) is not None and not (isinstance(r.get(key), float) and np.isnan(r.get(key)))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "correct_best_rate": correct_best / max(total, 1),
        "num_windows": total,
        "correct_vs_shuffle_pred_diff_mse": _mean_cond("same_task_shuffle", "pred_vs_correct_mse"),
        "correct_vs_zero_pred_diff_mse": _mean_cond("zero_action", "pred_vs_correct_mse"),
        "correct_vs_random_pred_diff_mse": _mean_cond("random_action", "pred_vs_correct_mse"),
        "warning_action_insensitive": (
            correct_best / max(total, 1) <= 0.5
            or _mean_cond("same_task_shuffle", "pred_vs_correct_mse") < 1e-5
            or _mean_cond("zero_action", "pred_vs_correct_mse") < 1e-5
        ),
    }
    (out / "action_ablation_summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Action Ablation Summary",
        "",
        f"- correct best rate: {summary['correct_best_rate']:.4f}",
        f"- correct vs shuffle pred diff MSE: {summary['correct_vs_shuffle_pred_diff_mse']:.6g}",
        f"- correct vs zero pred diff MSE: {summary['correct_vs_zero_pred_diff_mse']:.6g}",
        f"- correct vs random pred diff MSE: {summary['correct_vs_random_pred_diff_mse']:.6g}",
        f"- warning_action_insensitive: {summary['warning_action_insensitive']}",
    ]
    (out / "action_ablation_summary.md").write_text("\n".join(lines) + "\n")


def _save_priority_debug_visuals(
    output_dir: Path,
    task_idx: int,
    window_id: int,
    condition_bundles: Dict[str, Dict],
    row: Dict,
    goal_center: tuple,
    roi_half: int,
) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    correct = condition_bundles.get("correct")
    if correct is None:
        return
    shuffle = condition_bundles.get("same_task_shuffle", correct)
    zero = condition_bundles.get("zero_action")
    out_dir = output_dir / "debug_visuals" / f"task_{task_idx}" / f"window_{window_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_img(arr: np.ndarray, name: str) -> None:
        if arr.ndim == 2:
            Image.fromarray(arr.astype(np.uint8)).save(out_dir / name)
        else:
            Image.fromarray(arr.astype(np.uint8)).save(out_dir / name)

    def residual_vis(t: torch.Tensor) -> np.ndarray:
        arr = t.permute(1, 2, 0).numpy()
        return (((arr + 1.0) / 2.0).clip(0, 1) * 255).astype(np.uint8)

    def residual_abs_vis(t: torch.Tensor) -> np.ndarray:
        arr = t.abs().mean(0).numpy()
        arr = arr / max(float(arr.max()), 1e-8)
        return (arr * 255).astype(np.uint8)

    def mask_vis(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if t is None:
            return None
        arr = t.squeeze().numpy()
        arr = arr / max(float(arr.max()), 1e-8)
        return (arr * 255).astype(np.uint8)

    def overlay(base: np.ndarray, mask: Optional[np.ndarray], color=(255, 0, 0)) -> Optional[np.ndarray]:
        if mask is None:
            return None
        m = mask.astype(np.float32) / 255.0
        out = base.astype(np.float32).copy()
        col = np.array(color, dtype=np.float32)
        out = out * (1.0 - 0.45 * m[:, :, None]) + col[None, None, :] * (0.45 * m[:, :, None])
        return out.clip(0, 255).astype(np.uint8)

    save_img(correct["current"], "current.png")
    save_img(correct["gt"], "future.png")
    save_img(correct["pred"], "pred_correct.png")
    save_img(shuffle["pred"], "pred_shuffle.png")
    if zero is not None:
        save_img(zero["pred"], "pred_zero.png")
    save_img(residual_vis(correct["residual"]), "residual_correct.png")
    save_img(residual_vis(shuffle["residual"]), "residual_shuffle.png")
    save_img(residual_abs_vis(correct["residual"]), "residual_abs_correct.png")
    save_img(residual_abs_vis(shuffle["residual"]), "residual_abs_shuffle.png")
    save_img(residual_abs_vis(correct["residual"] - shuffle["residual"]), "residual_diff_correct_minus_shuffle.png")
    dyn = mask_vis(correct["dynamic_mask"])
    if dyn is not None:
        save_img(dyn, "dynamic_mask.png")
    wc = mask_vis(correct["write_mask"])
    ws = mask_vis(shuffle["write_mask"])
    if wc is not None:
        save_img(wc, "write_mask_correct.png")
    if ws is not None:
        save_img(ws, "write_mask_shuffle.png")
    if wc is not None and ws is not None:
        save_img(np.abs(wc.astype(np.int16) - ws.astype(np.int16)).astype(np.uint8), "write_mask_diff.png")
    ow = overlay(correct["current"], wc, color=(255, 64, 0))
    od = overlay(correct["current"], dyn, color=(0, 180, 255))
    if ow is not None:
        save_img(ow, "overlay_write_mask_on_current.png")
    if od is not None:
        save_img(od, "overlay_dynamic_mask_on_current.png")

    cy_t, cx_t = motion_center_of_mass(
        torch.from_numpy(correct["current"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
        torch.from_numpy(correct["gt"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
    )
    cy_v, cx_v = cy_t[0].item(), cx_t[0].item()
    save_img(roi_crop_np(correct["current"], cy_v, cx_v, roi_half), "gripper_roi_current.png")
    save_img(roi_crop_np(correct["gt"], cy_v, cx_v, roi_half), "gripper_roi_future.png")
    save_img(roi_crop_np(correct["pred"], cy_v, cx_v, roi_half), "gripper_roi_pred_correct.png")
    save_img(roi_crop_np(shuffle["pred"], cy_v, cx_v, roi_half), "gripper_roi_pred_shuffle.png")

    stats = {
        "task_id": task_idx,
        "window_id": window_id,
        "full_mse_correct": row.get("full_mse"),
        "full_mse_shuffle": None,
        "gripper_mse_correct": row.get("gripper_mse"),
        "gripper_mse_shuffle": None,
        "dynamic_mse_correct": row.get("dynamic_mse"),
        "dynamic_mse_shuffle": None,
        "lpips_correct": row.get("correct_lpips"),
        "lpips_shuffle": row.get("shuffled_lpips"),
        "lpips_gap": row.get("lpips_gap"),
        "copy_current_mse": row.get("copy_current_mse"),
        "write_mask_mean_correct": float(correct["write_mask"].mean()) if correct.get("write_mask") is not None else float("nan"),
        "write_mask_max_correct": float(correct["write_mask"].max()) if correct.get("write_mask") is not None else float("nan"),
        "write_mask_mean_shuffle": float(shuffle["write_mask"].mean()) if shuffle.get("write_mask") is not None else float("nan"),
        "write_mask_max_shuffle": float(shuffle["write_mask"].max()) if shuffle.get("write_mask") is not None else float("nan"),
        "residual_abs_mean_correct": float(correct["residual"].abs().mean()),
        "residual_abs_max_correct": float(correct["residual"].abs().max()),
        "residual_abs_mean_shuffle": float(shuffle["residual"].abs().mean()),
        "residual_abs_max_shuffle": float(shuffle["residual"].abs().max()),
    }
    (out_dir / "debug_stats.json").write_text(json.dumps(stats, indent=2))


def _save_eval_protocol_config(
    args: argparse.Namespace,
    output_dir: Path,
    condition_name: str,
    cfg: PixelResidualConfig,
    task_indices: List[int],
    horizon: int,
    checkpoint_path: str,
    window_manifest_path: str = "",
    window_manifest_hash: str = "",
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
        "action_ablation": bool(args.action_ablation),
        "save_debug_visuals": bool(args.save_debug_visuals),
        "debug_num_tasks": args.debug_num_tasks,
        "debug_windows_per_task": args.debug_windows_per_task,
        "use_window_manifest": bool(args.use_window_manifest),
        "window_manifest_path": window_manifest_path,
        "window_manifest_hash": window_manifest_hash,
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
    action_ablation_rows: List[Dict] = []
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
    manifest_hash = ""
    manifest_path = ""
    protocol = _save_eval_protocol_config(
        args=args,
        output_dir=output_dir,
        condition_name=condition_name,
        cfg=cfg,
        task_indices=task_indices,
        horizon=horizon,
        checkpoint_path=args.model_dir,
        window_manifest_path=manifest_path,
        window_manifest_hash=manifest_hash,
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
    windows_by_task: Dict[int, List[Tuple[np.ndarray, np.ndarray, Dict]]] = {}
    if args.use_window_manifest:
        if not args.window_manifest:
            raise ValueError("--use-window-manifest requires --window-manifest")
        manifest, manifest_hash = _load_window_manifest(args.window_manifest)
        _check_manifest_protocol(manifest, args, task_indices, horizon)
        windows_by_task = _windows_from_manifest(ds, manifest, horizon)
        manifest_path = str(Path(args.window_manifest))
        (output_dir / "window_manifest_used.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "window_manifest_hash.txt").write_text(manifest_hash + "\n", encoding="utf-8")
        logger.info("Using window manifest %s sha256=%s", manifest_path, manifest_hash)
    else:
        manifest_records: List[Dict] = []
        global_id = 0
        for task_idx in task_indices:
            task_name = f"task{task_idx}"
            try:
                if task_idx < len(task_names_all):
                    task_name = task_names_all[task_idx]
            except Exception:
                pass
            logger.info("Sampling windows for task %d (%s): target=%d", task_idx, task_name, windows_per_task)
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
            for local_idx, (_, _, win_meta) in enumerate(task_windows):
                rec = _make_manifest_record(global_id, task_idx, task_name, local_idx, win_meta, horizon)
                win_meta["manifest_record"] = rec
                manifest_records.append(rec)
                global_id += 1
            windows_by_task[task_idx] = task_windows
        _assign_manifest_negatives(manifest_records, rng)
        protocol_for_manifest = _protocol_for_manifest(args, task_indices, horizon)
        manifest_path_obj, manifest_hash = _save_window_manifest(output_dir, protocol_for_manifest, manifest_records)
        manifest_path = str(manifest_path_obj)
        (output_dir / "window_manifest_used.json").write_text(
            (output_dir / "window_manifest.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        logger.info("Saved window manifest %s sha256=%s", manifest_path, manifest_hash)

    # Re-save protocol now that the manifest hash is known.
    protocol = _save_eval_protocol_config(
        args=args,
        output_dir=output_dir,
        condition_name=condition_name,
        cfg=cfg,
        task_indices=task_indices,
        horizon=horizon,
        checkpoint_path=args.model_dir,
        window_manifest_path=manifest_path,
        window_manifest_hash=manifest_hash,
    )
    global_action_by_window: Dict[int, np.ndarray] = {}
    for arr in windows_by_task.values():
        for _, act, meta in arr:
            rec = meta.get("manifest_record")
            if rec:
                global_action_by_window[int(rec["global_window_id"])] = act

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

            task_windows = windows_by_task.get(task_idx, [])
            logger.info("Task %d (%s): evaluating %d manifest windows ...", task_idx, task_name, len(task_windows))
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
                    "global_window_id": win_meta.get("manifest_record", {}).get("global_window_id", ""),
                    "negative_window_id": win_meta.get("manifest_record", {}).get("negative_window_id", ""),
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
                            rec = win_meta.get("manifest_record", {})
                            neg_id = rec.get("negative_window_id")
                            neg_act_win = global_action_by_window.get(int(neg_id)) if neg_id is not None else None
                            if neg_act_win is None:
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

                # ---- Action ablation / priority-A debug visuals ----
                if args.action_ablation or args.save_debug_visuals:
                    pix_t = torch.from_numpy(pix_win).unsqueeze(0).to(device)
                    variants = _make_action_variants(act_win, task_windows, local_idx, rng)
                    bundles: Dict[str, Dict] = {}
                    for cond, acts_np in variants.items():
                        act_cond_t = torch.from_numpy(acts_np).unsqueeze(0).to(device)
                        bundles[cond] = _rollout_terminal_bundle(model, pix_t, act_cond_t, horizon)
                    correct_bundle = bundles["correct"]
                    if args.action_ablation:
                        for cond, bundle in bundles.items():
                            m = _condition_metrics(bundle, correct_bundle, lpips_fn, goal_center, roi_half)
                            action_ablation_rows.append({
                                "model_name": condition_name,
                                "task_id": task_idx,
                                "task_name": task_name,
                                "window_id": window_count,
                                "condition": cond,
                                **m,
                            })
                    if (
                        args.save_debug_visuals
                        and task_idx < args.debug_num_tasks
                        and window_count < args.debug_windows_per_task
                    ):
                        _save_priority_debug_visuals(
                            output_dir=output_dir,
                            task_idx=task_idx,
                            window_id=window_count,
                            condition_bundles=bundles,
                            row=row,
                            goal_center=goal_center,
                            roi_half=roi_half,
                        )

                all_rows.append(row)
                window_count += 1

            logger.info("  Task %d: %d windows evaluated.", task_idx, window_count)

        except Exception as exc:
            logger.warning("Error evaluating task %d: %s", task_idx, exc)
            logger.debug(traceback.format_exc())

    # ---- Aggregate and save ----
    agg = aggregate_phase1_metrics(all_rows, str(output_dir), condition_name)
    agg.update({
        "model_name": condition_name,
        "model_family": "phase1",
        "model_generation": getattr(cfg, "model_generation", ""),
        "target_mode": cfg.target_mode,
        "metric_source": "direct_eval_on_phase1_manifest" if args.use_window_manifest else "phase1_eval_manifest_generated",
        "window_manifest_hash": manifest_hash,
    })
    with open(output_dir / "aggregate_metrics.json", "w") as f:
        json.dump({"condition": condition_name, "metrics": agg}, f, indent=2)

    # Also save ranking_by_task.csv
    _save_ranking_by_task(all_rows, output_dir)
    _save_phase_breakdowns(all_rows, output_dir)
    _save_action_ablation_outputs(action_ablation_rows, output_dir)

    # Re-save full configs after aggregation because older utility versions
    # wrote a minimal config_used.json.
    with open(output_dir / "config_used.json", "w") as f:
        json.dump({"condition": condition_name,
                   "model_dir": args.model_dir,
                   "phase0_compatible": bool(args.phase0_compatible),
                   "eval_protocol": protocol,
                   "cfg": dataclasses.asdict(cfg)}, f, indent=2)
    protocol = _save_eval_protocol_config(
        args=args,
        output_dir=output_dir,
        condition_name=condition_name,
        cfg=cfg,
        task_indices=task_indices,
        horizon=horizon,
        checkpoint_path=args.model_dir,
        window_manifest_path=manifest_path,
        window_manifest_hash=manifest_hash,
    )
    with open(output_dir / "config_used.json", "w") as f:
        json.dump({"condition": condition_name,
                   "model_dir": args.model_dir,
                   "phase0_compatible": bool(args.phase0_compatible),
                   "eval_protocol": protocol,
                   "cfg": dataclasses.asdict(cfg)}, f, indent=2)

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
        "copy_current_full_mse", "copy_current_full_lpips",
        "copy_current_gripper_mse", "copy_current_gripper_lpips",
        "copy_current_dynamic_mse", "copy_current_dynamic_lpips",
        "model_vs_copy_full_mse_delta",
        "model_vs_copy_gripper_mse_delta",
        "model_vs_copy_dynamic_mse_delta",
        "full_mse_over_copy_current_mse",
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
