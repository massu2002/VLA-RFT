"""Evaluation script for TemporalDynamicQueryResidualWM (v4) on LIBERO.

Outputs (per condition directory):
    aggregate_metrics.json    — aggregated metrics
    metrics_by_task.csv
    metrics_by_phase.csv
    metrics_by_task_by_phase.csv
    ranking_by_window.csv     — per-window ranking (RFT reward + score)
    action_ablation/
    debug_visuals/task_N/window_M/
    window_manifest.json      — reusable eval manifest with history_frame_indices

Metrics in aggregate_metrics.json:
    Reconstruction quality (horizon-averaged, RFT-aligned):
        horizon_avg_lpips        — mean LPIPS over H steps (component of RFT reward)
        horizon_avg_mae          — mean MAE  over H steps (component of RFT reward)
        horizon_avg_mse          — mean MSE  over H steps
        rft_reward_proxy         — -(horizon_avg_lpips + horizon_avg_mae)
        copy_current_horizon_avg_mse — trivial baseline MSE
        horizon_mse_over_copy    — ratio; >1 means model worse than copy-current

    Ranking accuracy:
        pairwise_acc_rft         — fraction(rft_reward_correct > rft_reward_shuffled)
        pairwise_acc_score       — fraction(score_correct > score_shuffled)
                                   v4b only; NaN for v4a (no ActionFutureScorer)

    Ranking gaps:
        rft_reward_gap_mean/min  — correct − shuffled RFT reward
        score_gap_mean/min       — correct − shuffled scorer output

    Model internals:
        fuser_mask_entropy/overlap
        dynamic_mask_entropy/overlap
        future_dynamic_query_norm

    Operational:
        skipped_history_windows  — windows dropped due to insufficient history

Usage:
    python -m worldmodel.residual_worldmodel.eval_v4_temporal_query_libero \\
        --task-suite spatial \\
        --model-dir checkpoints/libero/PixelResidualWM/spatial/temporal_query_residual/v4b/s42/final \\
        --data-root /localdata/modified_libero_rlds \\
        --output-dir results/phase1/residual_worldmodel/v4b_spatial
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow_datasets as tfds
import torch

from ..datasets.libero.data import resolve_dataset_name
from .models.temporal_query_residual_wm import TemporalDynamicQueryResidualWM
from .pixel_residual_utils import (
    aggregate_phase1_metrics,
    compute_dynamic_mask,
    get_lpips_fn,
    motion_center_of_mass,
)
from ..eval_roi_utils import (
    load_roi_config,
    get_goal_roi_center,
    get_roi_half,
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
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate TemporalDynamicQueryResidualWM (v4) on LIBERO."
    )
    parser.add_argument("--task-suite",      type=str, required=True,
                        choices=["spatial", "object", "goal", "10"])
    parser.add_argument("--model-dir",       type=str, required=True)
    parser.add_argument("--data-root",       type=str, default="/localdata/modified_libero_rlds")
    parser.add_argument("--output-dir",      type=str, required=True)
    parser.add_argument("--condition-name",  type=str, default="")

    parser.add_argument("--num-eval-windows",     type=int, default=200)
    parser.add_argument("--num-ranking-windows",  type=int, default=100)
    parser.add_argument("--num-shuffle-reps",     type=int, default=3)
    parser.add_argument("--negative-eval-types",  type=str,
                        default=os.environ.get("NEGATIVE_EVAL_TYPES", "same_phase,temporal_shift,action_noise,mixed"),
                        help="Comma-separated negatives: same_phase, temporal_shift, action_noise, mixed")
    parser.add_argument("--temporal-shift-max",   type=int,
                        default=int(os.environ.get("TEMPORAL_SHIFT_MAX", "3")))
    parser.add_argument("--action-noise-std",     type=float,
                        default=float(os.environ.get("ACTION_NOISE_STD", "0.15")))
    parser.add_argument("--eval-horizon",         type=int, default=7)
    parser.add_argument("--eval-batch-size",      type=int, default=4)
    parser.add_argument("--seed",                 type=int, default=42)
    parser.add_argument("--device",               type=str, default="auto")

    parser.add_argument("--task-indices",         type=str, default="")
    parser.add_argument("--dry-run-windows",      type=int, default=0)
    parser.add_argument("--save-debug-images",    action="store_true", default=False)
    parser.add_argument("--save-debug-visuals",   action="store_true",
                        default=os.environ.get("SAVE_DEBUG_VISUALS", "0") == "1")
    parser.add_argument("--debug-num-tasks",      type=int,
                        default=int(os.environ.get("DEBUG_NUM_TASKS", "3")))
    parser.add_argument("--debug-windows-per-task", type=int,
                        default=int(os.environ.get("DEBUG_WINDOWS_PER_TASK", "3")))

    parser.add_argument("--action-ablation",      action="store_true",
                        default=os.environ.get("ACTION_ABLATION", "0") == "1")
    parser.add_argument("--window-position-mode", type=str,
                        default=os.environ.get("WINDOW_POSITION_MODE", "episode_phases"),
                        choices=["random", "episode_phases"])
    parser.add_argument("--num-eval-episodes-per-task", type=int,
                        default=int(os.environ.get("NUM_EVAL_EPISODES_PER_TASK", "0")))
    parser.add_argument("--window-manifest",      type=str,
                        default=os.environ.get("WINDOW_MANIFEST", ""))
    parser.add_argument("--use-window-manifest",  action="store_true",
                        default=os.environ.get("USE_WINDOW_MANIFEST", "0") == "1")

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _to_uint8_np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().float().clamp(0, 1)
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    return (t.numpy() * 255).astype(np.uint8)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0) ** 2))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0)))


def _mask_entropy_np(mask: torch.Tensor) -> float:
    """Mean per-query entropy of a [..., Q, N] softmax mask (higher = more uniform)."""
    p = mask.float().clamp(min=1e-9)
    return float(-(p * torch.log(p)).sum(dim=-1).mean().item())


def _mask_overlap_np(mask: torch.Tensor) -> float:
    """Mean off-diagonal pairwise cosine similarity across Q queries in [..., Q, N]."""
    flat = mask.float().reshape(-1, mask.shape[-2], mask.shape[-1])  # [B, Q, N]
    Q = flat.shape[1]
    if Q < 2:
        return 0.0
    norm = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    sim  = torch.bmm(norm, norm.transpose(1, 2))                      # [B, Q, Q]
    eye  = torch.eye(Q, device=mask.device, dtype=sim.dtype).unsqueeze(0)
    n_off = flat.shape[0] * Q * (Q - 1)
    return float((sim * (1.0 - eye)).sum().item() / max(n_off, 1))


def _lpips_safe(fn, a, b) -> float:
    try:
        return float(fn(a, b))
    except Exception:
        return float("nan")


def _decode_bytes(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _ep_file(ep: Dict) -> str:
    return _decode_bytes(ep.get("episode_metadata", {}).get("file_path", ""))


def _ep_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------------------------------------------------------
# Window collection with history support
# ---------------------------------------------------------------------------

def _collect_task_windows_v4(
    episodes: List[Tuple[int, Dict]],
    task_name: str,
    K: int,
    H: int,
    windows_per_task: int,
    window_position_mode: str,
    episodes_per_task: int,
    rng: random.Random,
) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Collect windows including K history frames.

    Window layout [total frames = K+2+H]:
        imgs[s:s+K]         → history frames
        imgs[s+K]           → context slot (unused)
        imgs[s+K+1]         → current frame
        imgs[s+K+2:s+K+2+H] → future frames

    Action layout [total = K+1+H]:
        acts[s:s+K+1]       → history+context (skipped by model)
        acts[s+K+1:s+K+1+H] → prediction horizon

    Args:
        episodes: Pre-filtered list of (global_idx, ep_dict) for this task only.
    """
    seg = K + 2 + H     # total frames per window
    act_seg = K + 1 + H  # total actions per window

    windows: List[Tuple[np.ndarray, np.ndarray, Dict]] = []
    episodes = list(episodes)
    rng.shuffle(episodes)

    phase_names = ("early", "middle", "later")
    phase_targets: Optional[Dict[str, int]] = None
    phase_counts: Dict[str, int] = {p: 0 for p in phase_names}
    if window_position_mode == "episode_phases":
        base = windows_per_task // len(phase_names)
        rem = windows_per_task % len(phase_names)
        phase_targets = {
            phase: base + (1 if i < rem else 0)
            for i, phase in enumerate(phase_names)
        }

    for ep_idx, ep in episodes:
        if len(windows) >= windows_per_task and window_position_mode != "episode_phases":
            break
        if window_position_mode == "episode_phases":
            if phase_targets and all(phase_counts[p] >= phase_targets[p] for p in phase_names):
                break

        steps = list(ep["steps"])
        imgs = np.stack([s["observation"]["image"] for s in steps], axis=0)
        acts = np.stack([s["action"] for s in steps], axis=0)
        T_ep = imgs.shape[0]

        if T_ep < seg:
            continue

        max_start = T_ep - seg
        if window_position_mode == "episode_phases":
            # Pick one random start from each third of the episode start range.
            e0, e1 = 0,              max(0, max_start // 3)
            m0, m1 = max_start // 3, max(max_start // 3, 2 * max_start // 3)
            l0, l1 = 2 * max_start // 3, max_start
            phase_starts = [
                ("early",  rng.randint(e0, max(e0, e1))),
                ("middle", rng.randint(m0, max(m0, m1))),
                ("later",  rng.randint(l0, max(l0, l1))),
            ]
        else:
            stride = max(1, seg // 2)
            phase_starts = [("random", s) for s in range(0, max_start + 1, stride)]

        used = set()
        added = 0
        for phase, start in phase_starts:
            if len(windows) >= windows_per_task and window_position_mode != "episode_phases":
                break
            if phase_targets and phase_counts.get(phase, 0) >= phase_targets.get(phase, 0):
                continue
            if start in used:
                continue
            used.add(start)
            pix_win = imgs[start: start + seg]
            act_win = acts[start: start + act_seg]
            if pix_win.shape[0] < seg or act_win.shape[0] < act_seg:
                continue

            current_abs = start + K + 1
            windows.append((
                pix_win, act_win,
                {
                    "episode_file":          _ep_file(ep),
                    "episode_index":         int(ep_idx),
                    "episode_id":            _ep_id(_ep_file(ep)),
                    "episode_length":        int(T_ep),
                    "window_phase":          phase,
                    "start":                 int(start),
                    "current_frame_index":   int(current_abs),
                    "history_frame_indices": list(range(current_abs - K, current_abs)),
                    "future_frame_indices":  list(range(current_abs + 1, current_abs + 1 + H)),
                    "action_indices":        list(range(current_abs, current_abs + H)),
                    "history_length":        K,
                },
            ))
            added += 1
            if phase in phase_counts:
                phase_counts[phase] += 1

    return windows


# ---------------------------------------------------------------------------
# Per-window evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_window_v4(
    model: TemporalDynamicQueryResidualWM,
    pixels_np: np.ndarray,    # [K+2+H, H_img, W_img, C] uint8
    actions_np: np.ndarray,   # [K+1+H, action_dim]
    horizon: int,
    device: torch.device,
    lpips_fn,
    goal_center: tuple,
    roi_half: int,
    window_id: int,
    task_name: str,
) -> Optional[Dict]:
    try:
        pixels_t  = torch.from_numpy(pixels_np).unsqueeze(0).to(device)
        actions_t = torch.from_numpy(actions_np).unsqueeze(0).to(device)

        out = model.rollout(pixels_t, actions_t, horizon=horizon)

        pred_future     = out["pred_future"][0]           # [H, 3, H_img, W_img]
        current_img     = out["current_image"][0]         # [3, H_img, W_img]
        future_gt_t     = out["future_gt"][0]             # [H, 3, H_img, W_img]
        fuser_masks     = out["fuser_masks"][0]           # [H, Q, N]
        future_dq       = out["future_dynamic_queries"][0]  # [H, Q, D]
        dynamic_masks_t = out["dynamic_masks"][0]         # [K+1, Q, N]
        ranking_score   = out.get("ranking_score")
        if ranking_score is not None:
            ranking_score = ranking_score[0].item()

        curr_np = _to_uint8_np(current_img)

        # Horizon-averaged metrics over all H steps — directly matches
        # phase1_residual_reward = -(LPIPS + MAE) averaged across the horizon.
        _h_lpips, _h_mae, _h_mse, _copy_mse = [], [], [], []
        for _t in range(horizon):
            _pt = _to_uint8_np(pred_future[_t])
            _gt = _to_uint8_np(future_gt_t[_t])
            _h_lpips.append(_lpips_safe(lpips_fn, _pt, _gt))
            _h_mae.append(_mae(_pt, _gt))
            _h_mse.append(_mse(_pt, _gt))
            _copy_mse.append(_mse(curr_np, _gt))

        horizon_avg_lpips            = float(np.nanmean(_h_lpips)) if _h_lpips else float("nan")
        horizon_avg_mae              = float(np.nanmean(_h_mae))   if _h_mae   else float("nan")
        horizon_avg_mse              = float(np.nanmean(_h_mse))   if _h_mse   else float("nan")
        rft_reward_proxy             = -(horizon_avg_lpips + horizon_avg_mae)
        copy_current_horizon_avg_mse = float(np.nanmean(_copy_mse)) if _copy_mse else float("nan")
        horizon_mse_over_copy        = horizon_avg_mse / max(copy_current_horizon_avg_mse, 1e-12)

        # Model internal structure metrics
        fuser_mask_entropy   = _mask_entropy_np(fuser_masks)
        fuser_mask_overlap   = _mask_overlap_np(fuser_masks)
        dynamic_mask_entropy = _mask_entropy_np(dynamic_masks_t)
        dynamic_mask_overlap = _mask_overlap_np(dynamic_masks_t)
        future_dq_norm       = float(future_dq.float().norm(dim=-1).mean().item())

        # score_correct: ActionFutureScorer output (v4b only); NaN for v4a (no scorer)
        score_correct_init = float(ranking_score) if ranking_score is not None else float("nan")

        row: Dict = {
            "task_name":                     task_name,
            "window_id":                     window_id,
            # Reconstruction quality (horizon-averaged, RFT-aligned)
            "horizon_avg_lpips":             horizon_avg_lpips,
            "horizon_avg_mae":               horizon_avg_mae,
            "horizon_avg_mse":               horizon_avg_mse,
            "rft_reward_proxy":              rft_reward_proxy,
            "copy_current_horizon_avg_mse":  copy_current_horizon_avg_mse,
            "horizon_mse_over_copy":         horizon_mse_over_copy,
            # Model internal structure
            "fuser_mask_entropy":            fuser_mask_entropy,
            "fuser_mask_overlap":            fuser_mask_overlap,
            "dynamic_mask_entropy":          dynamic_mask_entropy,
            "dynamic_mask_overlap":          dynamic_mask_overlap,
            "future_dynamic_query_norm":     future_dq_norm,
            # Ranking fields (filled in ranking block below)
            "score_correct":                 score_correct_init,
            "score_shuffle":                 float("nan"),
            "score_gap":                     float("nan"),
            "pairwise_win_score":            False,
            "rft_reward_gap":                float("nan"),
            "pairwise_win_rft":              False,
        }
        return row

    except Exception as exc:
        logger.warning("Error in window %d (task=%s): %s", window_id, task_name, exc)
        logger.debug(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Action variants
# ---------------------------------------------------------------------------

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


def _parse_negative_eval_types(raw: str) -> List[str]:
    out: List[str] = []
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


def _negative_metric_suffix(neg_type: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in neg_type).strip("_")


def _prediction_action_start(act_win: np.ndarray, horizon: int) -> int:
    return max(0, act_win.shape[0] - horizon)


def _select_same_phase_action(
    task_windows: List[Tuple],
    local_idx: int,
    phase: str,
    horizon: int,
    rng: random.Random,
) -> np.ndarray:
    candidates = [
        i for i, (_, _, meta) in enumerate(task_windows)
        if i != local_idx and str(meta.get("window_phase", "")) == phase
    ]
    if not candidates:
        candidates = [i for i in range(len(task_windows)) if i != local_idx]
    if candidates:
        return task_windows[rng.choice(candidates)][1]

    act_win = task_windows[local_idx][1]
    neg = np.array(act_win, copy=True)
    pred_start = _prediction_action_start(neg, horizon)
    pred = neg[pred_start:]
    if len(pred) > 1:
        perm = np.arange(len(pred))
        rng.shuffle(perm)
        neg[pred_start:] = pred[perm]
    return neg


def _make_temporal_shift_action(
    act_win: np.ndarray,
    horizon: int,
    rng: random.Random,
    max_shift: int,
) -> np.ndarray:
    neg = np.array(act_win, copy=True)
    pred_start = _prediction_action_start(neg, horizon)
    pred = np.array(neg[pred_start:], copy=True)
    if pred.shape[0] <= 1:
        return neg
    shift_cap = max(1, min(max_shift, pred.shape[0] - 1))
    shift = rng.choice([s for s in range(-shift_cap, shift_cap + 1) if s != 0])
    shifted = np.empty_like(pred)
    if shift > 0:
        shifted[:shift] = pred[0]
        shifted[shift:] = pred[:-shift]
    else:
        s = -shift
        shifted[:-s] = pred[s:]
        shifted[-s:] = pred[-1]
    neg[pred_start:] = shifted
    return neg


def _make_action_noise_action(
    act_win: np.ndarray,
    horizon: int,
    rng: random.Random,
    noise_std: float,
) -> np.ndarray:
    neg = np.array(act_win, copy=True)
    pred_start = _prediction_action_start(neg, horizon)
    pred = np.array(neg[pred_start:], copy=True)
    if pred.size == 0:
        return neg
    scale = np.nanstd(pred, axis=0, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    noise_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    neg[pred_start:] = pred + noise_rng.normal(0.0, noise_std * scale, size=pred.shape).astype(pred.dtype)
    return neg


def _make_negative_action_v4(
    neg_type: str,
    act_win: np.ndarray,
    task_windows: List[Tuple],
    local_idx: int,
    phase: str,
    horizon: int,
    rng: random.Random,
    temporal_shift_max: int,
    action_noise_std: float,
) -> np.ndarray:
    if neg_type == "mixed":
        neg_type = rng.choice(["same_phase", "temporal_shift", "action_noise"])
    if neg_type == "same_phase":
        return _select_same_phase_action(task_windows, local_idx, phase, horizon, rng)
    if neg_type == "temporal_shift":
        return _make_temporal_shift_action(act_win, horizon, rng, temporal_shift_max)
    if neg_type == "action_noise":
        return _make_action_noise_action(act_win, horizon, rng, action_noise_std)
    raise ValueError(f"Unknown negative action type: {neg_type}")


def _make_action_variants_v4(
    act_win: np.ndarray,
    task_windows: List[Tuple],
    local_idx: int,
    rng: random.Random,
) -> Dict[str, np.ndarray]:
    variants = {"correct": act_win}
    if len(task_windows) > 1:
        neg_idx = rng.randrange(len(task_windows) - 1)
        if neg_idx >= local_idx:
            neg_idx += 1
        variants["same_task_shuffle"] = task_windows[neg_idx][1]
    else:
        perm = np.random.permutation(act_win.shape[0])
        variants["same_task_shuffle"] = act_win[perm]
    variants["zero_action"]   = np.zeros_like(act_win)
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


# ---------------------------------------------------------------------------
# Debug visualization (v4)
# ---------------------------------------------------------------------------

def _save_v4_debug_visuals(
    output_dir: Path,
    task_idx: int,
    window_id: int,
    model: TemporalDynamicQueryResidualWM,
    pixels_t: torch.Tensor,
    actions_t: torch.Tensor,
    neg_actions_t: torch.Tensor,
    out_correct: Dict,
    out_shuffle: Dict,
    row: Dict,
    roi_half: int,
) -> None:
    try:
        from PIL import Image
    except ImportError:
        return

    out_dir = output_dir / "debug_visuals" / f"task_{task_idx}" / f"window_{window_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(arr: np.ndarray, name: str):
        Image.fromarray(arr.astype(np.uint8)).save(out_dir / name)

    def _residual_vis(t: torch.Tensor) -> np.ndarray:
        arr = t.float().permute(1, 2, 0).cpu().numpy()
        return (((arr + 1.0) / 2.0).clip(0, 1) * 255).astype(np.uint8)

    def _mask_vis(t: torch.Tensor) -> np.ndarray:
        arr = t.float().squeeze().cpu().numpy()
        arr = arr / max(float(arr.max()), 1e-8)
        return (arr * 255).astype(np.uint8)

    def _overlay(base_np: np.ndarray, mask_np: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
        m = mask_np.astype(np.float32) / 255.0
        out = base_np.astype(np.float32).copy()
        col = np.array(color, np.float32)
        out = out * (1 - 0.45 * m[:, :, None]) + col[None, None, :] * (0.45 * m[:, :, None])
        return out.clip(0, 255).astype(np.uint8)

    K = model.cfg.history_length
    pixels_f = pixels_t[0].float().permute(0, 3, 1, 2) / 255.0

    # History frames
    for k in range(K):
        _save(_to_uint8_np(pixels_f[k]), f"history_{k}.png")
    _save(_to_uint8_np(pixels_f[K + 1]), "current.png")
    _save(_to_uint8_np(pixels_f[K + 2]), "future.png")

    # Predictions
    _save(_to_uint8_np(out_correct["pred_future"][0, -1]), "pred_correct.png")
    _save(_to_uint8_np(out_shuffle["pred_future"][0, -1]), "pred_shuffle.png")

    # Residual abs
    res_c = out_correct["residual_pred"][0, -1]
    res_s = out_shuffle["residual_pred"][0, -1]
    _save(
        (res_c.float().abs().mean(0).cpu().numpy() / max(float(res_c.abs().max()), 1e-8) * 255).astype(np.uint8),
        "residual_abs_correct.png",
    )
    _save(
        ((res_c - res_s).float().abs().mean(0).cpu().numpy() / max(float((res_c - res_s).abs().max()), 1e-8) * 255).astype(np.uint8),
        "pred_diff_correct_minus_shuffle.png",
    )

    # Dynamic masks per query
    dyn_masks = out_correct.get("dynamic_masks")  # [1, K+1, Q, N]
    if dyn_masks is not None:
        dm = dyn_masks[0, -1]   # [Q, N]
        N = dm.shape[1]
        sp = int(N ** 0.5)
        for q in range(dm.shape[0]):
            _save(_mask_vis(dm[q].reshape(sp, sp).unsqueeze(0)), f"dynamic_mask_q{q}.png")

    # Fuser masks
    fuser = out_correct.get("fuser_masks")   # [1, H, Q, N]
    if fuser is not None:
        for t_idx in range(min(3, fuser.shape[1])):
            for q in range(fuser.shape[2]):
                sp = int(fuser.shape[3] ** 0.5)
                _save(_mask_vis(fuser[0, t_idx, q].reshape(sp, sp).unsqueeze(0)),
                      f"fuser_mask_t{t_idx}_q{q}.png")

    # Overlays
    curr_np = _to_uint8_np(pixels_f[K + 1])
    if dyn_masks is not None:
        dm_mean = dyn_masks[0, -1].float().mean(0)
        dm_sp = dm_mean.reshape(int(dm_mean.shape[0] ** 0.5), -1)
        import torch.nn.functional as F
        dm_img = F.interpolate(dm_sp.unsqueeze(0).unsqueeze(0), size=curr_np.shape[:2]).squeeze().numpy()
        _save(_overlay(curr_np, (dm_img / max(dm_img.max(), 1e-8) * 255).astype(np.uint8)),
              "overlay_dynamic_mask_on_current.png")
    if fuser is not None:
        fm_mean = fuser[0, -1].float().mean(0)
        sp = int(fm_mean.shape[0] ** 0.5)
        import torch.nn.functional as F
        fm_img = F.interpolate(fm_mean.reshape(sp, sp).unsqueeze(0).unsqueeze(0),
                               size=curr_np.shape[:2]).squeeze().numpy()
        _save(_overlay(curr_np, (fm_img / max(fm_img.max(), 1e-8) * 255).astype(np.uint8), (0, 180, 255)),
              "overlay_fuser_mask_on_current.png")

    # Debug stats JSON
    stats = {
        "task_idx":               task_idx,
        "window_id":              window_id,
        "ranking_score_correct":  row.get("ranking_score_correct", float("nan")),
        "ranking_score_shuffle":  row.get("score_shuffle", float("nan")),
        "score_gap":              row.get("score_gap", float("nan")),
        "lpips_correct":          row.get("correct_lpips", float("nan")),
        "lpips_shuffle":          row.get("shuffled_lpips", float("nan")),
        "lpips_gap":              row.get("lpips_gap", float("nan")),
        "fuser_mask_mean":        row.get("fuser_mask_mean", float("nan")),
        "fuser_mask_max":         row.get("fuser_mask_max", float("nan")),
        "dynamic_mask_mean":      row.get("dynamic_mask_mean", float("nan")),
        "future_dynamic_query_norm": row.get("future_dynamic_query_norm", float("nan")),
    }
    (out_dir / "debug_stats.json").write_text(json.dumps(stats, indent=2))


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

    condition_name = args.condition_name or Path(args.model_dir).name

    # Load v4 model
    logger.info("Loading v4 model from %s", args.model_dir)
    model = TemporalDynamicQueryResidualWM.load_pretrained(args.model_dir).to(device).eval()
    cfg = model.cfg
    K = cfg.history_length
    H = min(args.eval_horizon, cfg.action_horizon)
    logger.info("history_length=%d  action_horizon=%d  eval_horizon=%d", K, cfg.action_horizon, H)
    logger.info("use_action_future_scorer=%s  num_dynamic_queries=%d",
                cfg.use_action_future_scorer, cfg.num_dynamic_queries)
    logger.info(
        "v4 frame layout (per window, seg_len=%d):\n"
        "  pixels[0:%d]       → history  (episode idx: start … start+%d)\n"
        "  pixels[%d]          → context slot (NOT fed to model)\n"
        "  pixels[%d]          → current frame (episode idx: start+%d)\n"
        "  pixels[%d:%d]      → future GT  (H=%d steps)\n"
        "  *** context gap: history[-1]=start+%d, current=start+%d → 2-step gap via context slot ***",
        K + 2 + H,
        K, K - 1,
        K,
        K + 1, K + 1,
        K + 2, K + 2 + H, H,
        K - 1, K + 1,
    )

    # Save config
    import dataclasses
    with open(output_dir / "config_used.json", "w") as f:
        json.dump({
            "condition": condition_name,
            "model_dir": args.model_dir,
            "model_generation": "v4",
            "cfg": dataclasses.asdict(cfg),
        }, f, indent=2)

    lpips_fn  = get_lpips_fn(device=str(device))
    roi_config = load_roi_config()

    # Task names
    try:
        from libero.libero import benchmark as lb
        bench = lb.get_benchmark_dict()
        key = f"libero_{args.task_suite}"
        task_names_all = bench[key]().get_task_names() if key in bench else []
    except Exception:
        task_names_all = []
    if not task_names_all:
        task_names_all = _FALLBACK_TASK_NAMES.get(args.task_suite, [])

    dataset_name = resolve_dataset_name(args.task_suite)
    task_indices = (
        [int(x.strip()) for x in args.task_indices.split(",") if x.strip()]
        if args.task_indices else list(range(10))
    )

    windows_per_task = max(1, args.num_eval_windows // max(len(task_indices), 1))
    if args.dry_run_windows > 0:
        windows_per_task = min(windows_per_task, args.dry_run_windows)

    episodes_per_task = args.num_eval_episodes_per_task
    if args.window_position_mode == "episode_phases" and episodes_per_task <= 0:
        episodes_per_task = max(1, int(np.ceil(windows_per_task / 3.0)))

    ds = tfds.load(dataset_name, data_dir=args.data_root, split="train", shuffle_files=False)
    rng = random.Random(args.seed)
    negative_eval_types = _parse_negative_eval_types(args.negative_eval_types)
    default_negative_type = "mixed" if "mixed" in negative_eval_types else negative_eval_types[0]
    logger.info("negative_eval_types=%s  default=%s", ",".join(negative_eval_types), default_negative_type)

    # Build per-task episode index once, filtering by episode file path so that
    # each task_idx only sees episodes whose file names match the task name.
    logger.info("Loading all episodes to build per-task index ...")
    all_episodes_raw = list(enumerate(tfds.as_numpy(ds)))
    episodes_by_task: Dict[str, List[Tuple[int, Dict]]] = {}
    for task_idx_tmp in task_indices:
        tname = task_names_all[task_idx_tmp] if task_idx_tmp < len(task_names_all) else f"task{task_idx_tmp}"
        task_key = tname.lower().replace(" ", "_")
        filtered = [
            (gi, ep) for gi, ep in all_episodes_raw
            if task_key in os.path.basename(_ep_file(ep)).lower()
        ]
        episodes_by_task[tname] = filtered
        logger.info("  task %d (%s): %d matching episodes", task_idx_tmp, tname, len(filtered))
    del all_episodes_raw  # free memory

    all_rows: List[Dict] = []
    action_ablation_rows: List[Dict] = []
    skipped_history = 0
    ranked_windows = 0

    manifest_records: List[Dict] = []
    global_id = 0

    for task_idx in task_indices:
        task_name = task_names_all[task_idx] if task_idx < len(task_names_all) else f"task{task_idx}"
        goal_center = get_goal_roi_center(args.task_suite, task_idx, roi_config)
        roi_half    = get_roi_half(args.task_suite, task_idx, roi_config)

        logger.info("Collecting windows for task %d (%s) ...", task_idx, task_name)
        task_windows = _collect_task_windows_v4(
            episodes=episodes_by_task.get(task_name, []),
            task_name=task_name,
            K=K, H=H,
            windows_per_task=windows_per_task,
            window_position_mode=args.window_position_mode,
            episodes_per_task=episodes_per_task,
            rng=rng,
        )
        logger.info("  Task %d: %d windows collected.", task_idx, len(task_windows))

        window_count = 0
        for local_idx, (pix_win, act_win, win_meta) in enumerate(task_windows):

            row = _eval_window_v4(
                model=model,
                pixels_np=pix_win,
                actions_np=act_win,
                horizon=H,
                device=device,
                lpips_fn=lpips_fn,
                goal_center=goal_center,
                roi_half=roi_half,
                window_id=window_count,
                task_name=task_name,
            )
            if row is None:
                skipped_history += 1
                continue

            row.update({
                "task_index":      task_idx,
                "global_window_id": global_id,
                "window_phase":    win_meta.get("window_phase", "random"),
                "episode_length":  win_meta.get("episode_length", ""),
                "episode_file":    win_meta.get("episode_file", ""),
                "frame_indices":   json.dumps(win_meta.get("future_frame_indices", [])),
                "action_indices":  json.dumps(win_meta.get("action_indices", [])),
                "history_frame_indices": json.dumps(win_meta.get("history_frame_indices", [])),
            })

            # Record manifest entry
            manifest_records.append({
                "global_window_id":      global_id,
                "task_id":               task_idx,
                "task_name":             task_name,
                "episode_file":          win_meta.get("episode_file", ""),
                "episode_index":         win_meta.get("episode_index", -1),
                "episode_id":            win_meta.get("episode_id", ""),
                "episode_length":        win_meta.get("episode_length", 0),
                "window_position":       win_meta.get("window_phase", "random"),
                "current_frame_index":   win_meta.get("current_frame_index", -1),
                "history_frame_indices": win_meta.get("history_frame_indices", []),
                "future_frame_indices":  win_meta.get("future_frame_indices", []),
                "action_indices":        win_meta.get("action_indices", []),
                "history_length":        K,
            })
            global_id += 1

            # ---- RFT reward + score-based ranking (all eval windows) ----
            pix_t = torch.from_numpy(pix_win).unsqueeze(0).to(device)
            phase = str(win_meta.get("window_phase", "unknown"))

            for neg_type in negative_eval_types:
                neg_suffix = _negative_metric_suffix(neg_type)
                shuf_rft_reward_list = []
                shuf_score_list = []    # ActionFutureScorer scores across reps (v4b)

                for _ in range(args.num_shuffle_reps):
                    neg_act = _make_negative_action_v4(
                        neg_type=neg_type,
                        act_win=act_win,
                        task_windows=task_windows,
                        local_idx=local_idx,
                        phase=phase,
                        horizon=H,
                        rng=rng,
                        temporal_shift_max=args.temporal_shift_max,
                        action_noise_std=args.action_noise_std,
                    )

                    act_shuf_t = torch.from_numpy(neg_act).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out_shuf = model.rollout(pix_t, act_shuf_t, horizon=H)

                    _s_h_lpips, _s_h_mae = [], []
                    for _t in range(H):
                        _p_t = _to_uint8_np(out_shuf["pred_future"][0, _t])
                        _g_t = _to_uint8_np(out_shuf["future_gt"][0, _t])
                        _s_h_lpips.append(_lpips_safe(lpips_fn, _p_t, _g_t))
                        _s_h_mae.append(_mae(_p_t, _g_t))
                    _s_avg_lpips = float(np.nanmean(_s_h_lpips)) if _s_h_lpips else float("nan")
                    _s_avg_mae   = float(np.nanmean(_s_h_mae))   if _s_h_mae   else float("nan")
                    shuf_rft_reward_list.append(-(_s_avg_lpips + _s_avg_mae))

                    if cfg.use_action_future_scorer:
                        _s_score = out_shuf.get("ranking_score")
                        if _s_score is not None:
                            shuf_score_list.append(_s_score[0].item())

                shuffled_rft_reward = float(np.nanmean(shuf_rft_reward_list))
                rft_reward_gap = row["rft_reward_proxy"] - shuffled_rft_reward
                row[f"rft_reward_gap_{neg_suffix}"] = rft_reward_gap
                row[f"pairwise_win_rft_{neg_suffix}"] = rft_reward_gap > 0

                if cfg.use_action_future_scorer:
                    score_correct_v = row["score_correct"]
                    score_neg_v = float(np.nanmean(shuf_score_list)) if shuf_score_list else float("nan")
                    score_gap_v = (score_correct_v - score_neg_v
                                   if not (np.isnan(score_correct_v) or np.isnan(score_neg_v))
                                   else float("nan"))
                    row[f"score_shuffle_{neg_suffix}"] = score_neg_v
                    row[f"score_gap_{neg_suffix}"] = score_gap_v
                    row[f"pairwise_win_score_{neg_suffix}"] = (
                        score_correct_v > score_neg_v if not np.isnan(score_correct_v) else False
                    )
                else:
                    row[f"score_shuffle_{neg_suffix}"] = float("nan")
                    row[f"score_gap_{neg_suffix}"] = float("nan")
                    row[f"pairwise_win_score_{neg_suffix}"] = False

            # Backward-compatible representative metrics use the mixed negative
            # when available; otherwise they use the first requested negative.
            default_suffix = _negative_metric_suffix(default_negative_type)
            row["negative_metric_type"] = default_negative_type
            row["rft_reward_gap"] = row.get(f"rft_reward_gap_{default_suffix}", float("nan"))
            row["pairwise_win_rft"] = row.get(f"pairwise_win_rft_{default_suffix}", False)
            row["score_shuffle"] = row.get(f"score_shuffle_{default_suffix}", float("nan"))
            row["score_gap"] = row.get(f"score_gap_{default_suffix}", float("nan"))
            row["pairwise_win_score"] = row.get(f"pairwise_win_score_{default_suffix}", False)

            ranked_windows += 1

            # ---- Action ablation / debug visuals ----
            if args.action_ablation or args.save_debug_visuals:
                variants = _make_action_variants_v4(act_win, task_windows, local_idx, rng)

                bundles: Dict[str, Dict] = {}
                for cond, acts_np in variants.items():
                    act_t = torch.from_numpy(acts_np).unsqueeze(0).to(device)
                    with torch.no_grad():
                        b_out = model.rollout(pix_t, act_t, horizon=H)
                    bundles[cond] = b_out

                if args.action_ablation:
                    correct_b = bundles["correct"]
                    for cond, b_out in bundles.items():
                        pred = _to_uint8_np(b_out["pred_future"][0, -1])
                        gt   = _to_uint8_np(b_out["future_gt"][0, -1])
                        pred_c = _to_uint8_np(correct_b["pred_future"][0, -1])
                        sc = b_out.get("ranking_score")
                        action_ablation_rows.append({
                            "model_name":  condition_name,
                            "task_id":     task_idx,
                            "task_name":   task_name,
                            "window_id":   window_count,
                            "condition":   cond,
                            "full_mse":    _mse(pred, gt),
                            "full_lpips":  _lpips_safe(lpips_fn, pred, gt),
                            "pred_vs_correct_mse":   _mse(pred, pred_c),
                            "ranking_score": sc[0].item() if sc is not None else float("nan"),
                        })

                if (args.save_debug_visuals
                        and task_idx < args.debug_num_tasks
                        and window_count < args.debug_windows_per_task):
                    act_shuf_t = torch.from_numpy(variants.get("same_task_shuffle", act_win)).unsqueeze(0).to(device)
                    _save_v4_debug_visuals(
                        output_dir=output_dir,
                        task_idx=task_idx,
                        window_id=window_count,
                        model=model,
                        pixels_t=torch.from_numpy(pix_win).unsqueeze(0).to(device),
                        actions_t=torch.from_numpy(act_win).unsqueeze(0).to(device),
                        neg_actions_t=act_shuf_t,
                        out_correct=bundles["correct"],
                        out_shuffle=bundles.get("same_task_shuffle", bundles["correct"]),
                        row=row,
                        roi_half=roi_half,
                    )

            all_rows.append(row)
            window_count += 1

        logger.info("  Task %d: %d windows evaluated.", task_idx, window_count)

    # ---- Save window manifest with history extension ----
    protocol_info = {
        "model_generation":     "v4",
        "target_mode":          "temporal_query_residual",
        "task_suite":           args.task_suite,
        "history_length":       K,
        "eval_horizon":         H,
        "window_position_mode": args.window_position_mode,
        "num_eval_windows":     args.num_eval_windows,
        "context_gap_frames":   2,
        "negative_eval_types":  negative_eval_types,
        "default_negative_type": default_negative_type,
        "temporal_shift_max":   args.temporal_shift_max,
        "action_noise_std":     args.action_noise_std,
    }
    with open(output_dir / "eval_protocol_config.json", "w") as f:
        json.dump(protocol_info, f, indent=2)

    manifest_payload = {
        "protocol": protocol_info,
        "num_windows":     len(manifest_records),
        "windows":         manifest_records,
        "skipped_history": skipped_history,
    }
    (output_dir / "window_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- Aggregate metrics ----
    agg = aggregate_phase1_metrics(all_rows, str(output_dir), condition_name)

    def _valid_mean(key):
        vals = [r[key] for r in all_rows if not np.isnan(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    def _valid_min(key):
        vals = [r[key] for r in all_rows if not np.isnan(r.get(key, float("nan")))]
        return float(np.min(vals)) if vals else float("nan")

    # pairwise_acc_score: exclude windows where scorer returned NaN (score_shuffle = NaN).
    # Consistent with the per-task gating in aggregate_phase1_metrics.
    _score_rows = [r for r in all_rows if not np.isnan(r.get("score_shuffle", float("nan")))]
    _pairwise_acc_score = (float(np.mean([r["pairwise_win_score"] for r in _score_rows]))
                           if _score_rows else float("nan"))

    agg.update({
        "model_generation":          "v4",
        "target_mode":               "temporal_query_residual",
        "history_length":            K,
        "num_dynamic_queries":       cfg.num_dynamic_queries,
        "use_motion_bias":           cfg.use_motion_bias,
        "use_action_future_scorer":  cfg.use_action_future_scorer,
        "lambda_rank":               cfg.lambda_rank,
        "negative_eval_types":        negative_eval_types,
        "default_negative_type":      default_negative_type,
        "temporal_shift_max":         args.temporal_shift_max,
        "action_noise_std":           args.action_noise_std,
        "pairwise_acc_score":        _pairwise_acc_score,
        "pairwise_acc_rft":          agg.get("pairwise_acc_rft", float("nan")),
        "score_gap_mean":            _valid_mean("score_gap"),
        "score_gap_min":             _valid_min("score_gap"),
        "rft_reward_gap_mean":       _valid_mean("rft_reward_gap"),
        "rft_reward_gap_min":        _valid_min("rft_reward_gap"),
        "fuser_mask_entropy":        _valid_mean("fuser_mask_entropy"),
        "fuser_mask_overlap":        _valid_mean("fuser_mask_overlap"),
        "dynamic_mask_entropy":      _valid_mean("dynamic_mask_entropy"),
        "dynamic_mask_overlap":      _valid_mean("dynamic_mask_overlap"),
        "future_dynamic_query_norm": _valid_mean("future_dynamic_query_norm"),
        "skipped_history_windows":   skipped_history,
    })

    with open(output_dir / "aggregate_metrics.json", "w") as f:
        json.dump({"condition": condition_name, "metrics": agg}, f, indent=2)

    # Per-window ranking CSV
    rank_rows = [r for r in all_rows if not np.isnan(r.get("rft_reward_gap", float("nan")))]
    if rank_rows:
        with open(output_dir / "ranking_score_by_window.csv", "w", newline="") as f:
            fieldnames = [
                "task_name", "window_id", "window_phase",
                "score_correct", "score_shuffle", "score_gap", "pairwise_win_score",
                "rft_reward_proxy", "rft_reward_gap", "pairwise_win_rft",
                "horizon_avg_lpips", "horizon_avg_mae", "horizon_avg_mse",
            ]
            for neg_type in negative_eval_types:
                neg_suffix = _negative_metric_suffix(neg_type)
                fieldnames.extend([
                    f"score_shuffle_{neg_suffix}",
                    f"score_gap_{neg_suffix}",
                    f"pairwise_win_score_{neg_suffix}",
                    f"rft_reward_gap_{neg_suffix}",
                    f"pairwise_win_rft_{neg_suffix}",
                ])
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows([{k: r.get(k) for k in w.fieldnames} for r in rank_rows])

    # Action ablation
    if action_ablation_rows:
        abl_dir = output_dir / "action_ablation"
        abl_dir.mkdir(exist_ok=True)
        with open(abl_dir / "action_ablation_by_window.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(action_ablation_rows[0].keys()))
            w.writeheader()
            w.writerows(action_ablation_rows)

        conds = set(r["condition"] for r in action_ablation_rows)
        wins = sum(
            1 for r in action_ablation_rows
            if r["condition"] == "correct"
            and r["horizon_avg_lpips"] <= min(
                rr["horizon_avg_lpips"]
                for rr in action_ablation_rows
                if rr["task_id"] == r["task_id"] and rr["window_id"] == r["window_id"]
            )
        )
        total = sum(1 for r in action_ablation_rows if r["condition"] == "correct")
        abl_summary = {"correct_best_rate_horizon_lpips": wins / max(total, 1), "num_windows": total}
        for cond in conds:
            sc_rows = [r["ranking_score"] for r in action_ablation_rows
                       if r["condition"] == cond and not np.isnan(r.get("ranking_score", float("nan")))]
            abl_summary[f"score_{cond}_mean"] = float(np.mean(sc_rows)) if sc_rows else float("nan")
        (abl_dir / "action_ablation_summary.json").write_text(json.dumps(abl_summary, indent=2))

    logger.info("=== v4 eval complete: %s ===", condition_name)
    logger.info("  horizon_avg_lpips=%.4f  horizon_avg_mae=%.4f  horizon_avg_mse=%.4f  n=%d",
                agg.get("horizon_avg_lpips", float("nan")),
                agg.get("horizon_avg_mae", float("nan")),
                agg.get("horizon_avg_mse", float("nan")),
                agg.get("num_windows", 0))
    logger.info("  rft_reward_proxy=%.4f  horizon_mse_over_copy=%.4f",
                agg.get("rft_reward_proxy", float("nan")),
                agg.get("horizon_mse_over_copy", float("nan")))
    logger.info("  pairwise_acc_rft=%.4f  pairwise_acc_score=%.4f  rft_gap=%.4f  score_gap=%.4f",
                agg.get("pairwise_acc_rft", float("nan")),
                agg.get("pairwise_acc_score", float("nan")),
                agg.get("rft_reward_gap_mean", float("nan")),
                agg.get("score_gap_mean", float("nan")))
    logger.info("  fuser_entropy=%.4f  fuser_overlap=%.4f  dyn_entropy=%.4f  dyn_overlap=%.4f  skipped=%d",
                agg.get("fuser_mask_entropy", float("nan")),
                agg.get("fuser_mask_overlap", float("nan")),
                agg.get("dynamic_mask_entropy", float("nan")),
                agg.get("dynamic_mask_overlap", float("nan")),
                skipped_history)
    logger.info("  → %s", output_dir)


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
