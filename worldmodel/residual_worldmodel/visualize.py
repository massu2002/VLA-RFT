"""Visualization and evaluation for LatentResidualWorldModel on LIBERO data.

Produces the same outputs as worldmodel/libero/visualize.py, plus three
additional ROI-focused MSE metrics:

  motion_mse    — MSE restricted to pixels where |gt[t] - gt[t-1]| > threshold
                  (captures moving objects and robot arm motion)
  foreground_mse— MSE restricted to pixels where |gt[t] - frame_0| > threshold
                  (captures non-background = anything that changed from context)
  robot_roi_mse — MSE restricted to a fixed bottom-crop of the frame
                  (heuristic ROI where the robot arm typically appears in LIBERO)

Rollout modes
-------------
  current_anchor_ctx  — single-pass prediction of frames[2..T] given
                        ctx_tokens(frame_0) and z_curr(frame_1).
  adjacent_delta      — teacher-forced step-by-step prediction of frames[0..T-1].
                        (uses GT previous embedding at each step, measures 1-step quality)

Usage:
    python -m worldmodel.residual_worldmodel.visualize \\
        --task-suite spatial \\
        --model-dir checkpoints/libero/ResidualWorldModel/spatial/20240422_sweep_ca_baseline \\
        --visual-tokenizer checkpoints/libero/WorldModel/Tokenizer \\
        --output-dir evals/residual_worldmodel/spatial
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lpips
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow_datasets as tfds
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from libero.libero import benchmark

from ..datasets.libero.data import resolve_dataset_name
from .model import LatentResidualWorldModel


# =========================================================
# Argument parser
# =========================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LatentResidualWorldModel with per-frame and ROI-focused metrics."
    )

    # --- Dataset ---
    parser.add_argument(
        "--task-suite",
        type=str,
        choices=["spatial", "object", "goal", "10"],
        required=True,
    )
    parser.add_argument("--data-root", type=str, default="data/modified_libero_rlds")

    # --- Model ---
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to trained ResidualWorldModel directory (contains predictor.pt + config JSON).",
    )
    parser.add_argument("--visual-tokenizer", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # --- Eval controls ---
    parser.add_argument("--num-eval-windows", type=int, default=100)
    parser.add_argument("--segment-length", type=int, default=8,
                        help="Frames per window (must match training segment_length).")
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["heldout", "all", "fallback_all"],
        default="heldout",
    )
    parser.add_argument("--heldout-ratio", type=float, default=0.2)
    parser.add_argument("--task-indices", type=str, default="",
                        help='Comma-separated task indices within the suite, e.g. "0,1,4".')
    parser.add_argument("--eval-batch-size", type=int, default=1,
                        help="Eval batch size (keep at 1 for per-frame decoding).")

    # --- Casebook ---
    parser.add_argument("--display-frames", type=int, default=6)
    parser.add_argument("--save-casebook-count", type=int, default=5)

    # --- Full-episode ---
    parser.add_argument("--num-full-episodes-per-task", type=int, default=1)
    parser.add_argument("--full-episode-index", type=int, default=0)
    parser.add_argument(
        "--full-episode-split-mode",
        type=str,
        choices=["heldout", "all", "fallback_all"],
        default="fallback_all",
    )
    parser.add_argument("--full-episode-display-cols", type=int, default=6)
    parser.add_argument("--save-full-episode-frames", action="store_true", default=False)

    # --- ROI parameters ---
    parser.add_argument(
        "--motion-threshold", type=float, default=0.05,
        help="Normalized pixel difference threshold for motion ROI mask (0-1 scale).",
    )
    parser.add_argument(
        "--fg-threshold", type=float, default=0.05,
        help="Normalized pixel difference threshold for foreground ROI mask (0-1 scale).",
    )
    parser.add_argument(
        "--robot-roi-frac", type=float, default=0.35,
        help="Bottom fraction of frame used as robot ROI (e.g. 0.35 = bottom 35%%).",
    )
    parser.add_argument(
        "--roi-min-pixels", type=int, default=50,
        help="Minimum mask pixels required to compute ROI MSE (else reported as NaN).",
    )

    return parser


# =========================================================
# Data containers
# =========================================================


@dataclass
class EvalWindow:
    task_name: str
    task_index: int
    episode_file: str
    start: int
    frames: List[np.ndarray]   # len = segment_length
    actions: List[np.ndarray]  # len = segment_length - 1


# =========================================================
# Basic helpers
# =========================================================


def _suite_key(task_suite: str) -> str:
    return f"libero_{task_suite}"


def get_task_names(task_suite: str) -> List[str]:
    benchmarks = benchmark.get_benchmark_dict()
    suite_key = _suite_key(task_suite)
    if suite_key not in benchmarks:
        raise ValueError(f"Unsupported task suite: {task_suite}")
    return benchmarks[suite_key]().get_task_names()


def parse_task_indices(args: argparse.Namespace) -> List[int]:
    task_names = get_task_names(args.task_suite)
    if not args.task_indices.strip():
        return list(range(len(task_names)))
    out: List[int] = []
    for x in args.task_indices.split(","):
        x = x.strip()
        if not x:
            continue
        idx = int(x)
        if idx < 0 or idx >= len(task_names):
            raise ValueError(f"task index out of range: {idx}")
        out.append(idx)
    return sorted(set(out))


def _decode_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _device_for_run(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _cleanup(*objs) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _is_heldout(file_path: str, heldout_ratio: float) -> bool:
    if heldout_ratio <= 0.0:
        return False
    if heldout_ratio >= 1.0:
        return True
    return (_stable_hash_int(file_path) % 10000) / 10000.0 < heldout_ratio


def _select_frame_indices(num_frames: int, max_frames: int) -> List[int]:
    if num_frames <= max_frames:
        return list(range(num_frames))
    return np.linspace(0, num_frames - 1, num=max_frames, dtype=int).tolist()


# =========================================================
# Model loading
# =========================================================

# Fields removed from ResidualWorldModelConfig in past refactors
_OBSOLETE_CONFIG_FIELDS = {"adj_delta_aux_weight", "ctx_tokens_only"}
# Fields that are inferred at runtime from the tokenizer — not needed from JSON
_RUNTIME_INFERRED_FIELDS = {"n_dyn_tokens", "dyn_token_dim", "ctx_dim"}


def _resolve_checkpoint_dir(model_path: Path) -> Path:
    """Return the directory that contains model weights (predictor.pt or model.safetensors)."""
    # Prefer weights saved directly in the root dir
    if (model_path / "predictor.pt").exists() or (model_path / "model.safetensors").exists():
        return model_path

    # Fall back to the latest numbered checkpoint subdir
    checkpoints = sorted(
        [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints:
        chosen = checkpoints[-1]
        print(f"[info] Using checkpoint: {chosen}")
        return chosen

    return model_path


def _load_config_dict(ckpt_dir: Path, model_root: Path) -> dict:
    """Load config dict, searching in priority order.

    Priority:
      1. <ckpt_dir>/residual_worldmodel_config.json  (saved by save_pretrained)
      2. <model_root>/residual_worldmodel_config.json
      3. <model_root>/residual_worldmodel_training_summary.json  (fallback)
    """
    candidates = [
        ckpt_dir / "residual_worldmodel_config.json",
        model_root / "residual_worldmodel_config.json",
        ckpt_dir / "residual_worldmodel_training_summary.json",
        model_root / "residual_worldmodel_training_summary.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            # training_summary wraps config under "config" key
            if "config" in raw and isinstance(raw["config"], dict):
                cfg_dict = raw["config"]
                print(f"[info] Loaded config from training summary: {path}")
            else:
                cfg_dict = raw
                print(f"[info] Loaded config from: {path}")
            return cfg_dict

    raise FileNotFoundError(
        f"No config file found. Searched:\n" + "\n".join(f"  {p}" for p in candidates)
    )


def _sanitize_config(cfg_dict: dict) -> dict:
    """Drop obsolete/runtime fields and return a clean dict for ResidualWorldModelConfig."""
    skip = _OBSOLETE_CONFIG_FIELDS | _RUNTIME_INFERRED_FIELDS
    return {k: v for k, v in cfg_dict.items() if k not in skip}


def _load_predictor_weights(model: "LatentResidualWorldModel", ckpt_dir: Path) -> None:
    """Load predictor weights from predictor.pt or model.safetensors (auto-detected)."""
    predictor_pt = ckpt_dir / "predictor.pt"
    safetensors_path = ckpt_dir / "model.safetensors"

    if predictor_pt.exists():
        state = torch.load(str(predictor_pt), map_location="cpu")
        model.predictor.load_state_dict(state, strict=True)
        print(f"[info] Loaded predictor from: {predictor_pt}")
        return

    if safetensors_path.exists():
        from safetensors.torch import load_file as load_safetensors
        full_sd = load_safetensors(str(safetensors_path))
        # Extract and strip the "predictor." prefix
        pred_sd = {
            k[len("predictor."):]: v
            for k, v in full_sd.items()
            if k.startswith("predictor.")
        }
        if not pred_sd:
            raise RuntimeError(
                f"No 'predictor.*' keys found in {safetensors_path}. "
                f"Available prefixes: {sorted({k.split('.')[0] for k in full_sd})}"
            )
        model.predictor.load_state_dict(pred_sd, strict=True)
        print(f"[info] Loaded predictor from safetensors: {safetensors_path} ({len(pred_sd)} keys)")
        return

    raise FileNotFoundError(
        f"No predictor weights found in {ckpt_dir}. "
        f"Expected 'predictor.pt' or 'model.safetensors'."
    )


def load_model(
    model_dir: str,
    visual_tokenizer_path: str,
    device: torch.device,
) -> "LatentResidualWorldModel":
    """Load a LatentResidualWorldModel from a training output directory.

    Handles two checkpoint formats:
      1. predictor.pt + residual_worldmodel_config.json  (saved by save_pretrained)
      2. model.safetensors + residual_worldmodel_training_summary.json  (HF Trainer fallback)

    Also tolerates old config formats (drops obsolete fields, adds defaults for new ones).
    """
    from .config import ResidualWorldModelConfig

    model_path = Path(model_dir).resolve()
    ckpt_dir = _resolve_checkpoint_dir(model_path)

    cfg_dict = _load_config_dict(ckpt_dir, model_path)
    cfg_dict = _sanitize_config(cfg_dict)

    # Build config — unknown fields from old checkpoints are silently dropped above;
    # new fields not present get their dataclass defaults.
    valid_fields = {f.name for f in ResidualWorldModelConfig.__dataclass_fields__.values()}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    cfg = ResidualWorldModelConfig(**cfg_dict)
    cfg.visual_tokenizer_path = visual_tokenizer_path

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = LatentResidualWorldModel(
        visual_tokenizer_path=visual_tokenizer_path,
        cfg=cfg,
        torch_dtype=torch_dtype,
    )

    _load_predictor_weights(model, ckpt_dir)

    model.to(device)
    model.eval()
    return model


# =========================================================
# ROI utilities
# =========================================================


def compute_motion_mask(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Binary mask of pixels with large temporal change (uint8 HxWx3 → bool HxW)."""
    diff = np.abs(
        curr_frame.astype(np.float32) - prev_frame.astype(np.float32)
    ).mean(axis=2) / 255.0
    return diff > threshold


def compute_foreground_mask(
    context_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Binary mask of pixels different from the static context frame (uint8 → bool HxW)."""
    diff = np.abs(
        curr_frame.astype(np.float32) - context_frame.astype(np.float32)
    ).mean(axis=2) / 255.0
    return diff > threshold


def robot_roi_slice(H: int, W: int, frac: float = 0.35) -> Tuple[slice, slice]:
    """Bottom `frac` fraction of the frame — heuristic robot-arm region for LIBERO."""
    row_start = int(H * (1.0 - frac))
    return slice(row_start, H), slice(0, W)


def roi_mse(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    min_pixels: int,
) -> float:
    """MSE within binary mask (HxW bool). Returns NaN if mask is too small."""
    if mask.sum() < min_pixels:
        return float("nan")
    gt_f = gt.astype(np.float32) / 255.0
    pred_f = pred.astype(np.float32) / 255.0
    diff = (pred_f - gt_f) ** 2   # [H, W, 3]
    return float(diff[mask].mean())


# =========================================================
# Decode latents → frames
# =========================================================


@torch.no_grad()
def decode_latents_to_frames(
    model: LatentResidualWorldModel,
    ctx_tokens: torch.Tensor,   # [1, 1, N_ctx] int
    pred_latents: torch.Tensor, # [1, H, flat_dim] float
) -> List[np.ndarray]:
    """Convert predicted flat latents to decoded uint8 RGB frames."""
    B, H, _ = pred_latents.shape
    D = model.cfg.dyn_token_dim
    N = model.cfg.n_dyn_tokens

    # Nearest-neighbour quantization: flat float → discrete indices
    pred_codes = pred_latents.float().reshape(B * H, N, D)
    pred_indices = model.visual_tokenizer.dynamics_quantize.codes_to_indices(
        pred_codes.reshape(-1, D)
    ).reshape(B, H, N)  # [B, H, N_dyn]

    # Decode: ctx_tokens [B, 1, N_ctx], pred_indices [B, H, N_dyn]
    # detokenize returns [B, H+1, C, img_H, img_W] (prepends context frame)
    decoded = model.visual_tokenizer.detokenize(ctx_tokens, pred_indices)
    decoded = decoded[:, 1:].clamp(0.0, 1.0)  # [B, H, C, img_H, img_W]

    frames = []
    for h in range(H):
        arr = decoded[0, h].permute(1, 2, 0).cpu().float().numpy()
        frames.append((arr * 255.0).clip(0, 255).astype(np.uint8))
    return frames


# =========================================================
# Rollout functions
# =========================================================


@torch.no_grad()
def rollout_current_anchor_ctx(
    model: LatentResidualWorldModel,
    frames: List[np.ndarray],   # len = segment_length (e.g. 8)
    actions: List[np.ndarray],  # len = segment_length - 1 (e.g. 7)
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Single-pass rollout for current_anchor_ctx mode.

    Returns (pred_frames, gt_frames) where both cover frames[2..T].
    Requires segment_length >= 3.
    """
    T_plus_1 = len(frames)
    if T_plus_1 < 3:
        raise ValueError("current_anchor_ctx rollout requires segment_length >= 3.")

    # Stack and move to device
    pixels_np = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [T+1, H, W, 3]
    pixels_t = (
        torch.from_numpy(pixels_np)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)               # [1, T+1, C, H, W]
        .to(device)
    )

    # Encode all frames
    # tokenize(pixels [B, T+1, ...]) returns:
    #   ctx_tokens [1, 1, N_ctx]  — frame_0 as static context
    #   dyn_tokens [1, T, N_dyn]  — frames 1..T as dynamic tokens (frame_0 excluded)
    ctx_tokens, dyn_tokens = model._encode_both(pixels_t)
    dyn_flat = model._dyn_tokens_to_flat(dyn_tokens)   # [1, T, flat_dim] — frames 1..T

    z_curr = dyn_flat[:, 0, :]    # [1, flat_dim] — embed(frame_1)
    # z_future = dyn_flat[:, 1:, :] — embeds of frames 2..T

    ctx_summary = model._dequantize_ctx(ctx_tokens)    # [1, D_ctx]

    acts_np = np.stack(actions, axis=0).astype(np.float32)  # [T, action_dim]
    acts_t = torch.from_numpy(acts_np).unsqueeze(0).to(device)  # [1, T, action_dim]
    # actions[1:] = actions that transition from frame_1 → frame_2, ..., frame_{T-1} → frame_T
    acts_future = model._normalize_actions(acts_t[:, 1:, :])  # [1, T-1, action_dim]

    pred_dtype = next(model.predictor.parameters()).dtype
    pred_cum_delta = model.predictor(
        z_curr.to(pred_dtype),
        ctx_summary.to(pred_dtype),
        acts_future.to(pred_dtype),
    )  # [1, T-1, flat_dim]

    # pred_latents[h] ≈ embed(frame_{h+2}) for h=0..T-2
    pred_latents = z_curr.unsqueeze(1).to(pred_dtype) + pred_cum_delta  # [1, T-1, flat_dim]

    pred_frames = decode_latents_to_frames(model, ctx_tokens[:, :1], pred_latents)

    # GT: frames 2..T  (T-1 frames, matching pred_latents)
    gt_frames = list(frames[2:])

    return pred_frames, gt_frames


@torch.no_grad()
def rollout_adjacent_delta(
    model: LatentResidualWorldModel,
    frames: List[np.ndarray],   # len = segment_length  (= T+1 frames, indexed 0..T)
    actions: List[np.ndarray],  # len = segment_length - 1  (= T actions, indexed 0..T-1)
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Teacher-forced rollout for adjacent_delta mode, matching training exactly.

    tokenize(pixels [B, T+1, ...]) returns:
      ctx_tokens [B, 1, N_ctx]  — frame_0 as static context
      dyn_tokens [B, T, N_dyn]  — frames 1..T  (frame_0 is excluded from dynamic tokens)

    Training predictor:
      dyn_prev[t] = [zero, embed(frame_1), ..., embed(frame_{T-1})]
      pred_latents[t] = dyn_prev[t] + pred_delta[t] ≈ embed(frame_{t+1})

    Therefore pred_frames[t] predicts frame_{t+1}, and gt_frames = frames[1..T].
    Returns (pred_frames, gt_frames) — T pairs each predicting the next frame.
    """
    T_plus_1 = len(frames)
    T = T_plus_1 - 1

    pixels_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
    pixels_t = (
        torch.from_numpy(pixels_np)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        .to(device)
    )

    # ctx_tokens [1, 1, N_ctx] — frame_0 static context
    # dyn_tokens [1, T, N_dyn] — embeds of frames 1..T
    ctx_tokens, dyn_tokens = model._encode_both(pixels_t)
    dyn_flat = model._dyn_tokens_to_flat(dyn_tokens)  # [1, T, flat_dim] — frames 1..T

    flat_dim = dyn_flat.shape[-1]
    zero = torch.zeros(1, 1, flat_dim, device=device, dtype=dyn_flat.dtype)
    # dyn_all[0]=zero, dyn_all[1]=embed(frame_1), ..., dyn_all[T]=embed(frame_T)
    dyn_all = torch.cat([zero, dyn_flat], dim=1)   # [1, T+1, flat_dim]
    # dyn_prev[t] = embed(frame_t) for t=0..T-1  (with frame_0 = zero)
    dyn_prev = dyn_all[:, :T, :]                   # [1, T, flat_dim]

    acts_np = np.stack(actions, axis=0).astype(np.float32)
    acts_t = torch.from_numpy(acts_np).unsqueeze(0).to(device)
    acts_norm = model._normalize_actions(acts_t)   # [1, T, action_dim]

    pred_dtype = next(model.predictor.parameters()).dtype
    pred_delta = model.predictor(
        dyn_prev.to(pred_dtype), acts_norm.to(pred_dtype)
    )  # [1, T, flat_dim]

    # pred_latents[t] = dyn_prev[t] + pred_delta[t] ≈ embed(frame_{t+1})
    pred_latents = (dyn_prev.to(pred_dtype) + pred_delta)  # [1, T, flat_dim]

    pred_frames = decode_latents_to_frames(model, ctx_tokens[:, :1], pred_latents)

    # GT: frames[1..T]  — pred_latents[t] predicts frame_{t+1}
    gt_frames = list(frames[1:])

    return pred_frames, gt_frames


@torch.no_grad()
def rollout(
    model: LatentResidualWorldModel,
    frames: List[np.ndarray],
    actions: List[np.ndarray],
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Dispatch rollout based on model's residual_target_mode."""
    if model.cfg.residual_target_mode == "current_anchor_ctx":
        return rollout_current_anchor_ctx(model, frames, actions, device)
    else:
        return rollout_adjacent_delta(model, frames, actions, device)


# =========================================================
# Metrics
# =========================================================


def compute_metrics_full(
    gt_frames: List[np.ndarray],
    pred_frames: List[np.ndarray],
    context_frame: np.ndarray,
    lpips_model,
    device: torch.device,
    motion_threshold: float,
    fg_threshold: float,
    robot_roi_frac: float,
    roi_min_pixels: int,
) -> Dict:
    """Compute per-frame global + ROI metrics.

    Returns dict with:
      - global: mse, psnr, ssim, lpips per frame + averages
      - motion_roi_mse: MSE on motion mask per frame
      - fg_roi_mse: MSE on foreground mask per frame
      - robot_roi_mse: MSE on fixed bottom-crop per frame
    """
    mses, psnrs, ssims, lpips_list = [], [], [], []
    motion_mses, fg_mses, robot_mses = [], [], []

    H, W = gt_frames[0].shape[:2]
    robot_rs, robot_cs = robot_roi_slice(H, W, robot_roi_frac)
    # context frame for foreground mask
    ctx_f = context_frame

    for t, (gt, pred) in enumerate(zip(gt_frames, pred_frames)):
        gt_f = gt.astype(np.float32) / 255.0
        pred_f = pred.astype(np.float32) / 255.0

        mse = float(np.mean((pred_f - gt_f) ** 2))
        psnr = float(peak_signal_noise_ratio(gt_f, pred_f, data_range=1.0))
        ssim = float(structural_similarity(gt_f, pred_f, channel_axis=2, data_range=1.0))

        gt_t = torch.from_numpy(gt).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        pred_t = torch.from_numpy(pred).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        lp = float(lpips_model(pred_t * 2.0 - 1.0, gt_t * 2.0 - 1.0).item())

        mses.append(mse)
        psnrs.append(psnr)
        ssims.append(ssim)
        lpips_list.append(lp)

        # Motion ROI: temporal difference in GT frames
        if t == 0:
            prev_gt = context_frame
        else:
            prev_gt = gt_frames[t - 1]
        motion_mask = compute_motion_mask(prev_gt, gt, motion_threshold)
        motion_mses.append(roi_mse(gt, pred, motion_mask, roi_min_pixels))

        # Foreground ROI: differs from context frame
        fg_mask = compute_foreground_mask(ctx_f, gt, fg_threshold)
        fg_mses.append(roi_mse(gt, pred, fg_mask, roi_min_pixels))

        # Robot ROI: fixed bottom crop
        robot_mask = np.zeros((H, W), dtype=bool)
        robot_mask[robot_rs, robot_cs] = True
        robot_mses.append(roi_mse(gt, pred, robot_mask, roi_min_pixels))

    def _avg(lst: List[float]) -> float:
        valid = [v for v in lst if not math.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    return {
        "mse_per_frame": np.asarray(mses, dtype=np.float32),
        "psnr_per_frame": np.asarray(psnrs, dtype=np.float32),
        "ssim_per_frame": np.asarray(ssims, dtype=np.float32),
        "lpips_per_frame": np.asarray(lpips_list, dtype=np.float32),
        "motion_roi_mse_per_frame": np.asarray(motion_mses, dtype=np.float32),
        "fg_roi_mse_per_frame": np.asarray(fg_mses, dtype=np.float32),
        "robot_roi_mse_per_frame": np.asarray(robot_mses, dtype=np.float32),
        "avg_mse": _avg(mses),
        "avg_psnr": _avg(psnrs),
        "avg_ssim": _avg(ssims),
        "avg_lpips": _avg(lpips_list),
        "avg_motion_roi_mse": _avg(motion_mses),
        "avg_fg_roi_mse": _avg(fg_mses),
        "avg_robot_roi_mse": _avg(robot_mses),
    }


def _summarize(values: List[float]) -> Dict:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float32)
    if len(arr) == 0:
        return {k: float("nan") for k in ("mean", "std", "median", "min", "max", "worst10_mean")}
    k = max(1, math.ceil(0.1 * len(arr)))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "worst10_mean": float(np.mean(np.sort(arr)[-k:])),
    }


# =========================================================
# Window sampling
# =========================================================


def _episode_to_arrays(episode: Dict) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
    steps = list(episode["steps"])
    frames: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    for step in steps:
        frames.append(np.asarray(step["observation"]["image"], dtype=np.uint8))
        actions.append(np.asarray(step["action"], dtype=np.float32))
    episode_file = _decode_bytes(episode["episode_metadata"]["file_path"])
    return frames, actions[:-1], episode_file


def collect_eval_windows(args: argparse.Namespace) -> List[EvalWindow]:
    rng = random.Random(args.seed)
    dataset_name = resolve_dataset_name(args.task_suite)
    task_names = get_task_names(args.task_suite)
    normalized_task_names = [_normalize_name(t) for t in task_names]
    selected_task_indices = parse_task_indices(args)
    selected_set = set(selected_task_indices)
    horizon = args.segment_length - 1

    ds = tfds.load(dataset_name, data_dir=args.data_root, split="train", shuffle_files=False)

    def collect_candidates(mode: str):
        by_task: Dict[int, list] = {i: [] for i in selected_task_indices}
        for episode in tfds.as_numpy(ds):
            file_path = _decode_bytes(episode["episode_metadata"]["file_path"])
            base = _normalize_name(os.path.basename(file_path))
            task_idx = None
            for idx, key in enumerate(normalized_task_names):
                if key in base:
                    task_idx = idx
                    break
            if task_idx is None or task_idx not in selected_set:
                continue
            if mode == "heldout" and not _is_heldout(file_path, args.heldout_ratio):
                continue
            by_task[task_idx].append((task_names[task_idx], task_idx, episode))
        return by_task

    if args.split_mode == "fallback_all":
        candidates = collect_candidates("heldout")
        if sum(len(v) for v in candidates.values()) == 0:
            print("[warn] No heldout episodes. Falling back to all.")
            candidates = collect_candidates("all")
    else:
        candidates = collect_candidates(args.split_mode)

    for idx in selected_task_indices:
        rng.shuffle(candidates[idx])

    windows: List[EvalWindow] = []
    cursors = {i: 0 for i in selected_task_indices}

    while len(windows) < args.num_eval_windows:
        progressed = False
        for task_idx in selected_task_indices:
            arr = candidates[task_idx]
            if not arr:
                continue
            task_name, ti, episode = arr[cursors[task_idx] % len(arr)]
            frames, actions, episode_file = _episode_to_arrays(episode)
            cursors[task_idx] += 1

            if len(actions) < horizon:
                continue
            max_start = len(actions) - horizon
            start = rng.randint(0, max_start)
            w_frames = frames[start : start + horizon + 1]
            w_actions = actions[start : start + horizon]
            if len(w_frames) != horizon + 1:
                continue

            windows.append(
                EvalWindow(
                    task_name=task_name,
                    task_index=ti,
                    episode_file=episode_file,
                    start=start,
                    frames=w_frames,
                    actions=w_actions,
                )
            )
            progressed = True
            if len(windows) >= args.num_eval_windows:
                break
        if not progressed:
            break

    print(f"[info] Collected {len(windows)} eval windows.")
    return windows


# =========================================================
# Evaluation
# =========================================================


@torch.no_grad()
def evaluate_windows(
    model: LatentResidualWorldModel,
    windows: List[EvalWindow],
    lpips_model,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict]:
    """Run rollout + metrics for every window. Returns list of case records."""
    records = []
    for idx, w in enumerate(windows):
        try:
            pred_frames, gt_frames = rollout(model, w.frames, w.actions, device)
        except Exception as e:
            print(f"[warn] Window {idx} rollout failed: {e}")
            continue

        # context_frame = first frame of the window for ROI anchoring
        context_frame = w.frames[0]

        metrics = compute_metrics_full(
            gt_frames=gt_frames,
            pred_frames=pred_frames,
            context_frame=context_frame,
            lpips_model=lpips_model,
            device=device,
            motion_threshold=args.motion_threshold,
            fg_threshold=args.fg_threshold,
            robot_roi_frac=args.robot_roi_frac,
            roi_min_pixels=args.roi_min_pixels,
        )

        records.append(
            {
                "case_id": idx,
                "task_name": w.task_name,
                "task_index": w.task_index,
                "episode_file": w.episode_file,
                "start": w.start,
                "context_frame": w.frames[0],
                "pred_frames": pred_frames,
                "gt_frames": gt_frames,
                "metrics": metrics,
                "avg_mse": float(metrics["avg_mse"]),
                "avg_psnr": float(metrics["avg_psnr"]),
                "avg_ssim": float(metrics["avg_ssim"]),
                "avg_lpips": float(metrics["avg_lpips"]),
                "avg_motion_roi_mse": float(metrics["avg_motion_roi_mse"]),
                "avg_fg_roi_mse": float(metrics["avg_fg_roi_mse"]),
                "avg_robot_roi_mse": float(metrics["avg_robot_roi_mse"]),
            }
        )

        _cleanup(pred_frames, gt_frames, metrics)

    return records


def summarize_records(records: List[Dict]) -> Dict:
    def _gather(key):
        return [r[key] for r in records if not math.isnan(r.get(key, float("nan")))]

    return {
        "num_windows": len(records),
        "mse": _summarize(_gather("avg_mse")),
        "psnr": _summarize(_gather("avg_psnr")),
        "ssim": _summarize(_gather("avg_ssim")),
        "lpips": _summarize(_gather("avg_lpips")),
        "motion_roi_mse": _summarize(_gather("avg_motion_roi_mse")),
        "fg_roi_mse": _summarize(_gather("avg_fg_roi_mse")),
        "robot_roi_mse": _summarize(_gather("avg_robot_roi_mse")),
    }


# =========================================================
# Casebook rendering
# =========================================================


def _make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_f = gt.astype(np.float32) / 255.0
    pred_f = pred.astype(np.float32) / 255.0
    err = np.mean(np.abs(gt_f - pred_f), axis=2)
    err = np.clip(err / max(err.max(), 1e-6), 0.0, 1.0)
    err_rgb = np.stack([err, np.zeros_like(err), 1.0 - err], axis=2)
    return (err_rgb * 255.0).astype(np.uint8)


def _make_roi_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 100, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a colored overlay onto frame where mask is True."""
    out = frame.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return out.clip(0, 255).astype(np.uint8)


def render_casebook_figure(
    case: Dict,
    out_path: Path,
    display_frames: int,
    title_prefix: str,
    motion_threshold: float,
    fg_threshold: float,
    robot_roi_frac: float,
) -> None:
    gt_frames = case["gt_frames"]
    pred_frames = case["pred_frames"]
    context_frame = case["context_frame"]
    metrics = case["metrics"]
    indices = _select_frame_indices(len(pred_frames), display_frames)

    H, W = gt_frames[0].shape[:2]
    robot_rs, robot_cs = robot_roi_slice(H, W, robot_roi_frac)

    # Rows: GT | Pred | Error | Motion ROI | Foreground ROI | Robot ROI
    row_labels = ["GT", "Pred", "Error", "Motion\nROI", "FG\nROI", "Robot\nROI"]
    n_rows = len(row_labels)
    n_cols = len(indices)

    fig = plt.figure(figsize=(3.0 * n_cols, 2.8 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols)

    for col_i, frame_idx in enumerate(indices):
        gt = gt_frames[frame_idx]
        pred = pred_frames[frame_idx]
        prev_gt = context_frame if frame_idx == 0 else gt_frames[frame_idx - 1]

        motion_mask = compute_motion_mask(prev_gt, gt, motion_threshold)
        fg_mask = compute_foreground_mask(context_frame, gt, fg_threshold)
        robot_mask = np.zeros((H, W), dtype=bool)
        robot_mask[robot_rs, robot_cs] = True

        row_images = [
            gt,
            pred,
            _make_error_map(gt, pred),
            _make_roi_overlay(gt, motion_mask, color=(255, 80, 0)),
            _make_roi_overlay(gt, fg_mask, color=(0, 160, 255)),
            _make_roi_overlay(gt, robot_mask, color=(0, 220, 80)),
        ]

        for row_i, img in enumerate(row_images):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.imshow(img)
            ax.axis("off")

            if row_i == 0:
                ax.set_title(f"t={frame_idx + 1}", fontsize=9)
            elif row_i == 1:
                mse_v = metrics["mse_per_frame"][frame_idx]
                lp_v = metrics["lpips_per_frame"][frame_idx]
                ax.set_title(f"MSE {mse_v:.4f}\nLPIPS {lp_v:.4f}", fontsize=7)
            elif row_i == 3:
                mv = metrics["motion_roi_mse_per_frame"][frame_idx]
                ax.set_title(f"{mv:.4f}" if not math.isnan(mv) else "–", fontsize=7)
            elif row_i == 4:
                fv = metrics["fg_roi_mse_per_frame"][frame_idx]
                ax.set_title(f"{fv:.4f}" if not math.isnan(fv) else "–", fontsize=7)
            elif row_i == 5:
                rv = metrics["robot_roi_mse_per_frame"][frame_idx]
                ax.set_title(f"{rv:.4f}" if not math.isnan(rv) else "–", fontsize=7)

            if col_i == 0:
                ax.set_ylabel(
                    row_labels[row_i], rotation=0, labelpad=40, fontsize=9, va="center"
                )

    fig.suptitle(
        f"{title_prefix}\n"
        f"{case['task_name']} | start={case['start']}\n"
        f"MSE={case['avg_mse']:.4f} | PSNR={case['avg_psnr']:.2f} | "
        f"SSIM={case['avg_ssim']:.4f} | LPIPS={case['avg_lpips']:.4f}\n"
        f"Motion-ROI MSE={case['avg_motion_roi_mse']:.4f} | "
        f"FG-ROI MSE={case['avg_fg_roi_mse']:.4f} | "
        f"Robot-ROI MSE={case['avg_robot_roi_mse']:.4f}",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def rank_cases(records: List[Dict], key: str, count: int) -> Dict[str, List[Dict]]:
    if not records:
        return {"best": [], "median": [], "worst": []}
    valid = [r for r in records if not math.isnan(r.get(key, float("nan")))]
    if not valid:
        return {"best": [], "median": [], "worst": []}
    sorted_cases = sorted(valid, key=lambda x: x[key])
    best = sorted_cases[:count]
    worst = sorted_cases[-count:]
    center = len(sorted_cases) // 2
    half = count // 2
    median = sorted_cases[max(0, center - half) : max(0, center - half) + count]
    return {"best": best, "median": median, "worst": worst}


def render_casebook(
    ranked: Dict[str, List[Dict]],
    output_dir: Path,
    display_frames: int,
    motion_threshold: float,
    fg_threshold: float,
    robot_roi_frac: float,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for split_name, cases in ranked.items():
        saved[split_name] = []
        for i, case in enumerate(cases):
            out_path = output_dir / f"{split_name}_{i:02d}.png"
            render_casebook_figure(
                case=case,
                out_path=out_path,
                display_frames=display_frames,
                title_prefix=split_name.upper(),
                motion_threshold=motion_threshold,
                fg_threshold=fg_threshold,
                robot_roi_frac=robot_roi_frac,
            )
            saved[split_name].append(str(out_path))
    return saved


# =========================================================
# Full-episode rendering
# =========================================================


def load_episode_for_task(
    dataset_name: str,
    data_root: str,
    task_name: str,
    episode_index: int,
    split_mode: str = "all",
    heldout_ratio: float = 0.2,
) -> Tuple[Dict, str]:
    dataset = tfds.load(dataset_name, data_dir=data_root, split="train")
    task_key = _normalize_name(task_name)
    matched = []

    def _use(file_path):
        if split_mode == "heldout":
            return _is_heldout(file_path, heldout_ratio)
        return True

    for episode in tfds.as_numpy(dataset):
        file_path = _decode_bytes(episode["episode_metadata"]["file_path"])
        if task_key not in os.path.basename(file_path).lower():
            continue
        if split_mode == "fallback_all":
            if not _is_heldout(file_path, heldout_ratio):
                matched_all = True  # collect unconditionally below
            matched.append((episode, file_path))
        elif _use(file_path):
            matched.append((episode, file_path))

    if not matched:
        raise RuntimeError(f"No episodes found for task={task_name}, split_mode={split_mode}")
    if episode_index >= len(matched):
        raise RuntimeError(
            f"episode_index={episode_index} out of range (found {len(matched)}) "
            f"for task={task_name}"
        )
    episode, file_path = matched[episode_index]
    return episode, file_path


def render_full_episode_overview(
    task_suite: str,
    task_index: int,
    task_name: str,
    episode_file: str,
    gt_frames: List[np.ndarray],
    pred_frames: List[np.ndarray],
    metrics_list: List[Dict],  # one metrics dict per window
    output_path: Path,
    ncols: int = 6,
    motion_threshold: float = 0.05,
    fg_threshold: float = 0.05,
    robot_roi_frac: float = 0.35,
) -> None:
    """Render per-frame GT/Pred/Error/ROI grid for a full episode rollout."""
    num_frames = min(len(gt_frames), len(pred_frames))
    if num_frames == 0:
        return

    context_frame = gt_frames[0] if gt_frames else pred_frames[0]
    H, W = gt_frames[0].shape[:2]
    robot_rs, robot_cs = robot_roi_slice(H, W, robot_roi_frac)

    n_rows_per_frame = 4  # GT / Pred / Error / Motion ROI overlay
    nrows_grid = math.ceil(num_frames / ncols)
    total_rows = nrows_grid * n_rows_per_frame

    fig = plt.figure(
        figsize=(3.0 * ncols, 2.5 * total_rows), constrained_layout=True
    )
    gs = fig.add_gridspec(total_rows, ncols)

    # Aggregate metrics for title
    avg_mse = float(np.nanmean([m["avg_mse"] for m in metrics_list])) if metrics_list else float("nan")
    avg_psnr = float(np.nanmean([m["avg_psnr"] for m in metrics_list])) if metrics_list else float("nan")
    avg_motion = float(np.nanmean([m["avg_motion_roi_mse"] for m in metrics_list])) if metrics_list else float("nan")
    avg_fg = float(np.nanmean([m["avg_fg_roi_mse"] for m in metrics_list])) if metrics_list else float("nan")
    avg_robot = float(np.nanmean([m["avg_robot_roi_mse"] for m in metrics_list])) if metrics_list else float("nan")

    for idx in range(num_frames):
        r = idx // ncols
        c = idx % ncols
        gt = gt_frames[idx]
        pred = pred_frames[idx]
        prev_gt = context_frame if idx == 0 else gt_frames[idx - 1]
        motion_mask = compute_motion_mask(prev_gt, gt, motion_threshold)

        # Each frame gets n_rows_per_frame rows in the grid
        base_row = r * n_rows_per_frame

        for row_offset, img in enumerate([
            gt,
            pred,
            _make_error_map(gt, pred),
            _make_roi_overlay(gt, motion_mask, color=(255, 80, 0)),
        ]):
            ax = fig.add_subplot(gs[base_row + row_offset, c])
            ax.imshow(img)
            ax.axis("off")
            if row_offset == 0:
                ax.set_title(f"t={idx + 1}", fontsize=8)
            if c == 0:
                label = ["GT", "Pred", "Err", "Motion"][row_offset]
                ax.set_ylabel(label, rotation=0, labelpad=22, fontsize=8, va="center")

    fig.suptitle(
        f"{task_suite} / task{task_index + 1:02d} / {task_name}\n{episode_file}\n"
        f"avg MSE={avg_mse:.4f} | avg PSNR={avg_psnr:.2f} | "
        f"Motion-ROI={avg_motion:.4f} | FG-ROI={avg_fg:.4f} | Robot-ROI={avg_robot:.4f}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# =========================================================
# Main evaluation
# =========================================================


def run_evaluation(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = _device_for_run(args.device)
    model = load_model(args.model_dir, args.visual_tokenizer, device)
    mode = model.cfg.residual_target_mode
    model_label = _slugify(Path(args.model_dir).name)

    print(f"[info] Model dir   : {args.model_dir}")
    print(f"[info] Mode        : {mode}")
    print(f"[info] Device      : {device}")

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    selected_task_indices = parse_task_indices(args)
    windows = collect_eval_windows(args)

    # --- Per-window evaluation ---
    records = evaluate_windows(model, windows, lpips_model, device, args)

    summary = summarize_records(records)
    summary_path = output_root / f"summary__{model_label}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dir": args.model_dir,
                "task_suite": args.task_suite,
                "residual_target_mode": mode,
                "segment_length": args.segment_length,
                "num_windows": len(records),
                "roi_params": {
                    "motion_threshold": args.motion_threshold,
                    "fg_threshold": args.fg_threshold,
                    "robot_roi_frac": args.robot_roi_frac,
                    "roi_min_pixels": args.roi_min_pixels,
                },
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"[info] Summary written: {summary_path}")
    _print_summary(summary)

    # --- Casebook ---
    ranked = rank_cases(records, key="avg_lpips", count=args.save_casebook_count)
    casebook_dir = output_root / f"casebook__{model_label}"
    casebook_paths = render_casebook(
        ranked=ranked,
        output_dir=casebook_dir,
        display_frames=args.display_frames,
        motion_threshold=args.motion_threshold,
        fg_threshold=args.fg_threshold,
        robot_roi_frac=args.robot_roi_frac,
    )
    print(f"[info] Casebook written to: {casebook_dir}")

    # --- Per-task summary ---
    per_task: Dict[str, Dict] = {}
    task_records: Dict[int, List[Dict]] = {}
    for r in records:
        task_records.setdefault(r["task_index"], []).append(r)

    for ti, t_recs in task_records.items():
        per_task[str(ti)] = {
            "task_name": t_recs[0]["task_name"],
            "window_count": len(t_recs),
            "summary": summarize_records(t_recs),
        }

    per_task_path = output_root / f"per_task__{model_label}.json"
    with open(per_task_path, "w", encoding="utf-8") as f:
        json.dump(per_task, f, indent=2)
    print(f"[info] Per-task summary: {per_task_path}")

    # --- Full-episode rollout ---
    dataset_name = resolve_dataset_name(args.task_suite)
    task_names = get_task_names(args.task_suite)
    full_ep_summary = {}

    for task_idx in selected_task_indices:
        task_name = task_names[task_idx]
        per_ep_runs = []

        for k in range(args.num_full_episodes_per_task):
            ep_idx = args.full_episode_index + k
            try:
                episode, episode_file = load_episode_for_task(
                    dataset_name=dataset_name,
                    data_root=args.data_root,
                    task_name=task_name,
                    episode_index=ep_idx,
                    split_mode=args.full_episode_split_mode,
                    heldout_ratio=args.heldout_ratio,
                )
            except Exception as e:
                print(f"[warn] Could not load episode {ep_idx} for {task_name}: {e}")
                continue

            frames, actions, _ = _episode_to_arrays(episode)

            # Slide segment-length windows across the full episode and predict.
            # Stride differs by mode so that consecutive windows produce contiguous GT frames:
            #   adjacent_delta  predicts T   frames per window → stride = T   = seg-1
            #   current_anchor  predicts T-1 frames per window → stride = T-1 = seg-2
            # Using seg-1 for CA would skip one frame per window boundary (z_curr anchor).
            seg = args.segment_length
            is_ca = (mode == "current_anchor_ctx")
            stride = seg - 2 if is_ca else seg - 1
            stride = max(stride, 1)

            all_pred: List[np.ndarray] = []
            all_gt: List[np.ndarray] = []
            all_metrics: List[Dict] = []

            for start in range(0, len(actions) - (seg - 1), stride):
                w_frames = frames[start : start + seg]
                w_actions = actions[start : start + seg - 1]
                if len(w_frames) < seg:
                    break
                try:
                    pf, gf = rollout(model, w_frames, w_actions, device)
                except Exception as e:
                    print(f"[warn] Rollout failed at start={start}: {e}")
                    continue
                m = compute_metrics_full(
                    gt_frames=gf,
                    pred_frames=pf,
                    context_frame=w_frames[0],
                    lpips_model=lpips_model,
                    device=device,
                    motion_threshold=args.motion_threshold,
                    fg_threshold=args.fg_threshold,
                    robot_roi_frac=args.robot_roi_frac,
                    roi_min_pixels=args.roi_min_pixels,
                )
                all_pred.extend(pf)
                all_gt.extend(gf)
                all_metrics.append(m)

            if not all_pred:
                continue

            task_dir = (
                output_root
                / f"full_episode__{model_label}"
                / _slugify(task_name)
                / f"episode_{ep_idx:03d}"
            )
            task_dir.mkdir(parents=True, exist_ok=True)

            overview_path = task_dir / "overview.png"
            render_full_episode_overview(
                task_suite=args.task_suite,
                task_index=task_idx,
                task_name=task_name,
                episode_file=episode_file,
                gt_frames=all_gt,
                pred_frames=all_pred,
                metrics_list=all_metrics,
                output_path=overview_path,
                ncols=args.full_episode_display_cols,
                motion_threshold=args.motion_threshold,
                fg_threshold=args.fg_threshold,
                robot_roi_frac=args.robot_roi_frac,
            )

            if args.save_full_episode_frames:
                frames_dir = task_dir / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                for i, (g, p) in enumerate(zip(all_gt, all_pred)):
                    plt.imsave(frames_dir / f"gt_{i:04d}.png", g)
                    plt.imsave(frames_dir / f"pred_{i:04d}.png", p)
                    plt.imsave(frames_dir / f"err_{i:04d}.png", _make_error_map(g, p))

            ep_payload = {
                "task_index": task_idx,
                "task_name": task_name,
                "episode_index": ep_idx,
                "episode_file": episode_file,
                "frame_count": len(all_pred),
                "avg_mse": float(np.nanmean([m["avg_mse"] for m in all_metrics])),
                "avg_psnr": float(np.nanmean([m["avg_psnr"] for m in all_metrics])),
                "avg_ssim": float(np.nanmean([m["avg_ssim"] for m in all_metrics])),
                "avg_lpips": float(np.nanmean([m["avg_lpips"] for m in all_metrics])),
                "avg_motion_roi_mse": float(np.nanmean([m["avg_motion_roi_mse"] for m in all_metrics])),
                "avg_fg_roi_mse": float(np.nanmean([m["avg_fg_roi_mse"] for m in all_metrics])),
                "avg_robot_roi_mse": float(np.nanmean([m["avg_robot_roi_mse"] for m in all_metrics])),
                "overview_png": str(overview_path),
            }
            with open(task_dir / "metrics.json", "w") as f:
                json.dump(ep_payload, f, indent=2)

            per_ep_runs.append(ep_payload)
            _cleanup(frames, actions, all_pred, all_gt, all_metrics)

        full_ep_summary[str(task_idx)] = {"task_name": task_name, "episodes": per_ep_runs}

    full_ep_path = output_root / f"full_episode_summary__{model_label}.json"
    with open(full_ep_path, "w") as f:
        json.dump(full_ep_summary, f, indent=2)

    print(f"[info] Full-episode summary: {full_ep_path}")
    print(f"\n[done] All outputs in: {output_root}")


def _print_summary(summary: Dict) -> None:
    print("\n=== Evaluation Summary ===")
    for metric in ("mse", "psnr", "ssim", "lpips", "motion_roi_mse", "fg_roi_mse", "robot_roi_mse"):
        s = summary.get(metric, {})
        mean = s.get("mean", float("nan"))
        std = s.get("std", float("nan"))
        print(f"  {metric:20s}: mean={mean:.4f}  std={std:.4f}")
    print("==========================\n")


# =========================================================
# Entry point
# =========================================================


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
