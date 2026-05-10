"""Phase 1 residual world model inference helpers for VLA-RFT.

The RFT pipeline consumes a future-image prediction when computing its
existing reconstruction/perceptual reward.  This module keeps the target-mode
specific reconstruction in one place:

  pixel                       pred_future = model(current, actions)
  pixel_residual              pred_future = current + residual_pred
  pixel_residual_roi_dynamic  pred_future = current + write_mask * residual_pred
  temporal_query_residual     pred_future = current + dynamic_query_fused_residual
                              + optional ranking_score → rank_score / hybrid reward

Images are float [0, 1] unless noted otherwise.

RFT reward modes (v4+):
  visual      — negative MSE between pred_future last frame and target image
  rank_score  — normalized ActionFutureScorer output (falls back to visual if unavailable)
  hybrid      — alpha * visual_reward + beta * normalized_rank_score

Environment variables consumed by compute_rft_reward():
  WORLD_REWARD_TYPE        visual | rank_score | hybrid   (default: visual)
  RANK_REWARD_ALPHA        float weight for visual component in hybrid  (default: 0.2)
  RANK_REWARD_BETA         float weight for rank_score in hybrid        (default: 0.8)
  NORMALIZE_RANK_REWARD    1 to z-score normalise rank_score per batch  (default: 1)
  CLIP_RANK_REWARD         1 to clip rank_score to ±RANK_REWARD_CLIP_VALUE (default: 1)
  RANK_REWARD_CLIP_VALUE   clip magnitude                               (default: 5.0)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .pixel_residual_model import PixelResidualWorldModel
from .pixel_residual_utils import _tensor_to_uint8

logger = logging.getLogger(__name__)


def _dtype_from_name(name: str) -> torch.dtype:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    if name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _try_import_v4():
    """Lazy import of v4 model to avoid circular deps at module load time."""
    from .models.temporal_query_residual_wm import TemporalDynamicQueryResidualWM
    return TemporalDynamicQueryResidualWM


def load_phase1_world_model(
    checkpoint: str,
    *,
    target_mode: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> PixelResidualWorldModel:
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"WORLD_MODEL_CKPT does not exist: {checkpoint}")
    model = PixelResidualWorldModel.load_pretrained(checkpoint, torch_dtype=torch_dtype)
    if target_mode is not None and model.cfg.target_mode != target_mode:
        raise ValueError(
            f"target_mode mismatch: requested={target_mode} checkpoint={model.cfg.target_mode}"
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def _align_actions(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    actions = actions.float()
    if actions.shape[-1] == action_dim:
        return actions
    if actions.shape[-1] > action_dim:
        return actions[..., :action_dim]
    pad = torch.zeros(*actions.shape[:-1], action_dim - actions.shape[-1],
                      device=actions.device, dtype=actions.dtype)
    return torch.cat([actions, pad], dim=-1)


@torch.no_grad()
def predict_future_with_world_model(
    world_model: PixelResidualWorldModel,
    current_image: torch.Tensor,
    actions: torch.Tensor,
    *,
    horizon: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Predict future images with a Phase 1 world model.

    Args:
        world_model: Loaded PixelResidualWorldModel.
        current_image: [B, 3, H, W] float [0,1].
        actions: [B, H, A] action sequence.
        horizon: Optional number of future frames.

    Returns:
        Dictionary containing pred_future [B, H, 3, H, W], current_image, and
        optional write_mask/write_logits.
    """
    device = next(world_model.parameters()).device
    current_image = current_image.to(device).float().clamp(0, 1)
    actions = _align_actions(actions.to(device), int(world_model.cfg.action_dim))
    if horizon is not None:
        actions = actions[:, :horizon]
    H = actions.shape[1]

    # PixelResidualWorldModel.rollout follows the LIBERO training window layout:
    # frame_0 is a context copy, frame_1 is current, frames 2.. are GT future.
    # For inference-only smoke tests, future frames are placeholders equal to current.
    pixels_cf = torch.cat([current_image[:, None], current_image[:, None].repeat(1, H + 1, 1, 1, 1)], dim=1)
    pixels_u8 = (pixels_cf.clamp(0, 1) * 255.0).round().to(torch.uint8)
    pixels_u8 = pixels_u8.permute(0, 1, 3, 4, 2).contiguous()

    first_action = actions[:, 0:1]
    end_action = actions[:, -1:]
    actions_w_ctx = torch.cat([first_action, actions, end_action], dim=1)

    out = world_model.rollout(pixels_u8, actions_w_ctx, horizon=H)
    pred = out["pred_future"]
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Phase1 world model produced NaN or Inf pred_future")
    if pred.shape != (current_image.shape[0], H, *current_image.shape[1:]):
        raise ValueError(f"Unexpected pred_future shape: {tuple(pred.shape)}")
    return out


def load_v4_world_model(
    checkpoint: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
):
    """Load a TemporalDynamicQueryResidualWM (v4) from a checkpoint directory."""
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"WORLD_MODEL_CKPT does not exist: {checkpoint}")
    TemporalDynamicQueryResidualWM = _try_import_v4()
    model = TemporalDynamicQueryResidualWM.load_pretrained(checkpoint, torch_dtype=torch_dtype)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def load_world_model(
    checkpoint: str,
    *,
    model_generation: str = "v1",
    target_mode: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
):
    """Unified world model loader dispatching on model_generation.

    model_generation "v4" loads TemporalDynamicQueryResidualWM;
    all other values load PixelResidualWorldModel.
    """
    if model_generation == "v4":
        return load_v4_world_model(
            checkpoint, torch_dtype=torch_dtype, device=device
        )
    return load_phase1_world_model(
        checkpoint, target_mode=target_mode, torch_dtype=torch_dtype, device=device
    )


@torch.no_grad()
def predict_future_with_v4_model(
    world_model,
    current_image: torch.Tensor,
    actions: torch.Tensor,
    *,
    history_images: Optional[torch.Tensor] = None,
    horizon: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Predict future images with a v4 TemporalDynamicQueryResidualWM.

    Args:
        world_model: Loaded TemporalDynamicQueryResidualWM.
        current_image: [B, 3, H_img, W_img] float [0,1].
        actions: [B, H, A] action sequence (prediction horizon only).
        history_images: [B, K, 3, H_img, W_img] float [0,1], or None.
            When None, current_image is repeated K times as a placeholder.
        horizon: Optional cap on prediction horizon.

    Returns:
        Dictionary with pred_future [B, H, 3, H_img, W_img], current_image,
        ranking_score [B] or None, and v4 debug tensors.
    """
    device = next(world_model.parameters()).device
    current_image = current_image.to(device).float().clamp(0.0, 1.0)
    actions = _align_actions(actions.to(device), int(world_model.cfg.action_dim))
    if horizon is not None:
        actions = actions[:, :horizon]
    H = actions.shape[1]
    K = world_model.cfg.history_length
    B = current_image.shape[0]

    # Build history frames: use provided history or repeat current as placeholder.
    if history_images is not None:
        hist = history_images.to(device).float().clamp(0.0, 1.0)
        if hist.shape[1] < K:
            # Pad on the left with copies of the oldest frame if K is larger.
            pad = hist[:, :1].repeat(1, K - hist.shape[1], 1, 1, 1)
            hist = torch.cat([pad, hist], dim=1)
        elif hist.shape[1] > K:
            hist = hist[:, -K:]
    else:
        hist = current_image.unsqueeze(1).expand(B, K, *current_image.shape[1:])

    # Pixel layout: [hist_0..hist_{K-1} | context | current | future_0..future_{H-1}]
    # Future GT is unknown at inference time — fill with current as placeholder.
    context = current_image.unsqueeze(1)                          # [B,1,3,H,W]
    current_u = current_image.unsqueeze(1)                        # [B,1,3,H,W]
    future_placeholder = current_image.unsqueeze(1).expand(B, H, *current_image.shape[1:])

    # Concatenate and convert to uint8 channel-last [B, K+2+H, H_img, W_img, C].
    all_frames = torch.cat([hist, context, current_u, future_placeholder], dim=1)
    pixels_u8 = (all_frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    pixels_u8 = pixels_u8.permute(0, 1, 3, 4, 2).contiguous()   # → channel-last

    # Action layout for rollout: [B, K+1+H, A].
    # rollout uses actions[:, K+1:K+1+H]; preceding slots are dummies.
    dummy = actions[:, 0:1].expand(B, K + 1, -1)
    actions_full = torch.cat([dummy, actions], dim=1)             # [B, K+1+H, A]

    out = world_model.rollout(pixels_u8, actions_full, horizon=H)

    pred = out["pred_future"]
    if not torch.isfinite(pred).all():
        raise FloatingPointError("v4 world model produced NaN or Inf pred_future")
    return out


# ---------------------------------------------------------------------------
# RFT reward computation
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip() not in ("0", "false", "False", "no", "")


def compute_rft_reward(
    pred_future: torch.Tensor,
    target_image: Optional[torch.Tensor] = None,
    ranking_score: Optional[torch.Tensor] = None,
    *,
    reward_type: Optional[str] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    normalize_rank: Optional[bool] = None,
    clip_rank: Optional[bool] = None,
    rank_clip_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute RFT reward from prediction output.

    Reads defaults from environment variables when keyword args are None.

    Args:
        pred_future: [B, H, 3, H_img, W_img] float [0,1].  Visual reward uses
            the last predicted frame (pred_future[:, -1]).
        target_image: [B, 3, H_img, W_img] float [0,1].  Required for visual.
        ranking_score: [B] float from ActionFutureScorer, or None.
        reward_type: visual | rank_score | hybrid (env: WORLD_REWARD_TYPE).
        alpha: visual weight in hybrid (env: RANK_REWARD_ALPHA, default 0.2).
        beta: rank_score weight in hybrid (env: RANK_REWARD_BETA, default 0.8).
        normalize_rank: z-score normalise rank_score per batch (env: NORMALIZE_RANK_REWARD).
        clip_rank: clip rank_score to ±rank_clip_value (env: CLIP_RANK_REWARD).
        rank_clip_value: clip magnitude (env: RANK_REWARD_CLIP_VALUE, default 5.0).

    Returns:
        Dict with keys:
          reward           [B] — the final reward tensor
          reward_type      str — effective reward type used
          visual_reward    [B] or None
          rank_score_raw   [B] or None
          rank_score_norm  [B] or None
          visual_reward_mean, rank_score_mean, rank_score_std, hybrid_reward_mean
    """
    if reward_type is None:
        reward_type = os.environ.get("WORLD_REWARD_TYPE", "visual")
    if alpha is None:
        alpha = _env_float("RANK_REWARD_ALPHA", 0.2)
    if beta is None:
        beta = _env_float("RANK_REWARD_BETA", 0.8)
    if normalize_rank is None:
        normalize_rank = _env_bool("NORMALIZE_RANK_REWARD", True)
    if clip_rank is None:
        clip_rank = _env_bool("CLIP_RANK_REWARD", True)
    if rank_clip_value is None:
        rank_clip_value = _env_float("RANK_REWARD_CLIP_VALUE", 5.0)

    B = pred_future.shape[0]
    device = pred_future.device

    # --- visual reward: negative MSE vs target last predicted frame --------
    visual_reward: Optional[torch.Tensor] = None
    if target_image is not None:
        last_pred = pred_future[:, -1].float()          # [B, 3, H, W]
        target_f  = target_image.to(device).float().clamp(0.0, 1.0)
        mse = ((last_pred - target_f) ** 2).mean(dim=(1, 2, 3))   # [B]
        visual_reward = -mse                            # higher is better

    # --- rank_score normalisation ------------------------------------------
    rank_score_raw  = ranking_score
    rank_score_norm: Optional[torch.Tensor] = None
    if rank_score_raw is not None:
        rs = rank_score_raw.float().to(device)
        if normalize_rank and B > 1:
            std = rs.std().clamp(min=1e-6)
            rs = (rs - rs.mean()) / std
        if clip_rank:
            rs = rs.clamp(-rank_clip_value, rank_clip_value)
        rank_score_norm = rs

    # --- fallback logic ----------------------------------------------------
    effective_type = reward_type
    if reward_type in ("rank_score", "hybrid") and rank_score_norm is None:
        logger.warning(
            "WORLD_REWARD_TYPE=%s requested but ranking_score is not available "
            "(model may lack ActionFutureScorer). Falling back to visual reward.",
            reward_type,
        )
        effective_type = "visual"

    if effective_type == "visual" and visual_reward is None:
        raise ValueError(
            "visual reward requested but target_image was not provided"
        )

    # --- compute final reward ----------------------------------------------
    reward: torch.Tensor
    if effective_type == "visual":
        reward = visual_reward
    elif effective_type == "rank_score":
        reward = rank_score_norm
    else:  # hybrid
        v = visual_reward if visual_reward is not None else torch.zeros(B, device=device)
        reward = alpha * v + beta * rank_score_norm

    # --- diagnostics -------------------------------------------------------
    def _mean(t):
        return float(t.mean().item()) if t is not None else None

    def _std(t):
        return float(t.std().item()) if t is not None and t.numel() > 1 else None

    return {
        "reward":             reward,
        "reward_type":        effective_type,
        "visual_reward":      visual_reward,
        "rank_score_raw":     rank_score_raw,
        "rank_score_norm":    rank_score_norm,
        "visual_reward_mean": _mean(visual_reward),
        "rank_score_mean":    _mean(rank_score_norm),
        "rank_score_std":     _std(rank_score_norm),
        "hybrid_reward_mean": _mean(reward) if effective_type == "hybrid" else None,
    }


def _save_png(path: Path, tensor: torch.Tensor) -> None:
    from PIL import Image

    arr = _tensor_to_uint8(tensor)
    Image.fromarray(arr).save(path)


def run_smoke_test(
    *,
    checkpoint: str,
    target_mode: str,
    model_generation: str,
    output_dir: str,
    dtype: str = "bfloat16",
    device_name: str = "cpu",
    batch_size: int = 1,
    horizon: int = 8,
    image_size: int = 256,
    # v4-specific
    history_length: int = 0,
    reward_type: str = "visual",
) -> Dict[str, Any]:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch_dtype = _dtype_from_name(dtype)

    is_v4 = model_generation == "v4"

    if is_v4:
        model = load_v4_world_model(checkpoint, torch_dtype=torch_dtype, device=device)
        history_length = model.cfg.history_length
    else:
        model = load_phase1_world_model(
            checkpoint,
            target_mode=target_mode,
            torch_dtype=torch_dtype,
            device=device,
        )

    current = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    current[:, :, image_size // 4: image_size // 2, image_size // 4: image_size // 2] = 0.5
    actions = torch.zeros(batch_size, horizon, int(model.cfg.action_dim), device=device)

    if is_v4:
        history_images = None
        if history_length > 0:
            history_images = current.unsqueeze(1).expand(batch_size, history_length, -1, -1, -1)
        out = predict_future_with_v4_model(
            model, current, actions,
            history_images=history_images,
            horizon=horizon,
        )
    else:
        out = predict_future_with_world_model(model, current, actions, horizon=horizon)

    pred = out["pred_future"]
    write_mask = out.get("write_mask")
    ranking_score = out.get("ranking_score")

    stats: Dict[str, Any] = {
        "current_image_shape": list(current.shape),
        "actions_shape": list(actions.shape),
        "pred_future_shape": list(pred.shape),
        "pred_future_min": float(pred.min().item()),
        "pred_future_max": float(pred.max().item()),
        "pred_future_mean": float(pred.mean().item()),
        "target_mode": target_mode,
        "model_generation": model_generation,
        "checkpoint": checkpoint,
        "config_target_mode": model.cfg.target_mode,
        "use_residual_write_mask": bool(getattr(model.cfg, "use_residual_write_mask", False)),
    }
    if write_mask is not None:
        stats["write_mask_mean"] = float(write_mask.mean().item())
        stats["write_mask_max"] = float(write_mask.max().item())

    if is_v4:
        stats["history_length"] = history_length
        if ranking_score is not None:
            stats["ranking_score_mean"] = float(ranking_score.mean().item())
            stats["ranking_score_min"] = float(ranking_score.min().item())
            stats["ranking_score_max"] = float(ranking_score.max().item())

        reward_out = compute_rft_reward(
            pred,
            target_image=current,        # smoke test: target = current (zero motion)
            ranking_score=ranking_score,
            reward_type=reward_type,
        )
        stats["reward_type"] = reward_out["reward_type"]
        stats["visual_reward_mean"] = reward_out["visual_reward_mean"]
        stats["rank_score_mean"] = reward_out["rank_score_mean"]
        stats["rank_score_std"] = reward_out["rank_score_std"]
        stats["hybrid_reward_mean"] = reward_out["hybrid_reward_mean"]

        fuser_masks = out.get("fuser_masks")
        dynamic_masks = out.get("dynamic_masks")
        if fuser_masks is not None:
            stats["fuser_mask_mean"] = float(fuser_masks.float().mean().item())
        if dynamic_masks is not None:
            stats["dynamic_mask_mean"] = float(dynamic_masks.float().mean().item())

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_png(out_dir / "current_image.png", current[0])
    _save_png(out_dir / "pred_future.png", pred[0, -1])
    with open(out_dir / "debug_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-mode", default="temporal_query_residual")
    parser.add_argument("--model-generation", default="v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--history-length", type=int, default=0,
                        help="Number of history frames for v4 models")
    parser.add_argument("--reward-type", default="visual",
                        choices=["visual", "rank_score", "hybrid"],
                        help="RFT reward mode (WORLD_REWARD_TYPE env var overrides)")
    args = parser.parse_args()

    reward_type = os.environ.get("WORLD_REWARD_TYPE", args.reward_type)
    stats = run_smoke_test(
        checkpoint=args.checkpoint,
        target_mode=args.target_mode,
        model_generation=args.model_generation,
        output_dir=args.output_dir,
        dtype=args.dtype,
        device_name=args.device,
        horizon=args.horizon,
        image_size=args.image_size,
        history_length=args.history_length,
        reward_type=reward_type,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
