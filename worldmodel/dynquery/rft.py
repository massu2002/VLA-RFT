"""RFT inference helpers for DynQueryWorldModel.

Provides future-image prediction and reward computation for the RFT pipeline.

RFT reward modes:
  lpips_mae   — -(LPIPS + MAE) horizon-averaged (default; matches eval rft_reward_proxy)
  visual      — horizon-averaged negative MSE (fast fallback)
  rank_score  — normalized ActionFutureScorer output (stage_b only)
  hybrid      — alpha * -(LPIPS+MAE) + beta * rank_score_norm

Environment variables:
  WORLD_REWARD_TYPE      lpips_mae | visual | rank_score | hybrid  (default: lpips_mae)
  RANK_REWARD_ALPHA      visual component weight in hybrid           (default: 0.2)
  RANK_REWARD_BETA       rank_score weight in hybrid                 (default: 0.8)
  NORMALIZE_RANK_REWARD  1 = z-score normalise rank_score per batch  (default: 1)
  CLIP_RANK_REWARD       1 = clip rank_score to ±RANK_REWARD_CLIP_VALUE (default: 1)
  RANK_REWARD_CLIP_VALUE clip magnitude                               (default: 5.0)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .model import DynQueryWorldModel, _tensor_to_uint8

logger = logging.getLogger(__name__)

_LPIPS_MODELS: Dict[str, Any] = {}


def _get_lpips_model(device: torch.device):
    key = str(device)
    if key not in _LPIPS_MODELS:
        try:
            import lpips as _lpips_lib
        except ImportError:
            raise ImportError(
                "lpips package is required for reward_type='lpips_mae'. "
                "Install with: pip install lpips"
            )
        model = _lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()
        _LPIPS_MODELS[key] = model
        logger.info("LPIPS(AlexNet) loaded on %s for lpips_mae reward.", device)
    return _LPIPS_MODELS[key]


def _dtype_from_name(name: str) -> torch.dtype:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    if name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _align_actions(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    actions = actions.float()
    if actions.shape[-1] == action_dim:
        return actions
    if actions.shape[-1] > action_dim:
        return actions[..., :action_dim]
    pad = torch.zeros(*actions.shape[:-1], action_dim - actions.shape[-1],
                      device=actions.device, dtype=actions.dtype)
    return torch.cat([actions, pad], dim=-1)


def load_world_model(
    checkpoint: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> DynQueryWorldModel:
    """Load a DynQueryWorldModel from a checkpoint directory."""
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    model = DynQueryWorldModel.load_pretrained(checkpoint, torch_dtype=torch_dtype)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


# Backward-compat alias used by fsdp_workers.py.
load_v4_world_model = load_world_model


@torch.no_grad()
def predict_future(
    world_model: DynQueryWorldModel,
    current_image: torch.Tensor,
    actions: torch.Tensor,
    *,
    history_images: Optional[torch.Tensor] = None,
    horizon: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Predict future images with a DynQueryWorldModel.

    Args:
        world_model: Loaded DynQueryWorldModel.
        current_image: [B, 3, H_img, W_img] float [0,1].
        actions: [B, H, A] action sequence (prediction horizon).
        history_images: [B, K, 3, H_img, W_img] float [0,1], or None.
        horizon: Optional cap on prediction horizon.

    Returns:
        Dictionary with pred_future [B, H, 3, H_img, W_img],
        ranking_score [B] or None, and debug tensors.
    """
    device = next(world_model.parameters()).device
    current_image = current_image.to(device).float().clamp(0.0, 1.0)
    actions = _align_actions(actions.to(device), int(world_model.cfg.action_dim))
    if horizon is not None:
        actions = actions[:, :horizon]
    H = actions.shape[1]
    K = world_model.cfg.history_length
    B = current_image.shape[0]

    if history_images is not None:
        hist = history_images.to(device).float().clamp(0.0, 1.0)
        if hist.shape[1] < K:
            pad = hist[:, :1].repeat(1, K - hist.shape[1], 1, 1, 1)
            hist = torch.cat([pad, hist], dim=1)
        elif hist.shape[1] > K:
            hist = hist[:, -K:]
    else:
        hist = current_image.unsqueeze(1).expand(B, K, *current_image.shape[1:])

    current_u = current_image.unsqueeze(1)
    future_placeholder = current_image.unsqueeze(1).expand(B, H, *current_image.shape[1:])

    all_frames = torch.cat([hist, current_u, future_placeholder], dim=1)
    pixels_u8 = (all_frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    pixels_u8 = pixels_u8.permute(0, 1, 3, 4, 2).contiguous()

    dummy = actions[:, 0:1].expand(B, K, -1)
    actions_full = torch.cat([dummy, actions], dim=1)

    out = world_model.rollout(pixels_u8, actions_full, horizon=H)

    pred = out["pred_future"]
    if not torch.isfinite(pred).all():
        raise FloatingPointError("DynQueryWorldModel produced NaN or Inf pred_future")
    return out


# Backward-compat alias.
predict_future_with_v4_model = predict_future


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
    target_images: Optional[torch.Tensor] = None,
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

    Args:
        pred_future: [B, H, 3, H_img, W_img] float [0,1].
        target_images: [B, H, 3, H_img, W_img] float [0,1]. Required for visual/lpips_mae.
        ranking_score: [B] float from ActionFutureScorer, or None.
        reward_type: lpips_mae | visual | rank_score | hybrid.
        alpha/beta: hybrid weights.
        normalize_rank: z-score normalise rank_score per batch.
        clip_rank/rank_clip_value: clip rank_score.

    Returns:
        Dict with reward [B], reward_type str, visual_reward [B] or None,
        rank_score_raw [B] or None, rank_score_norm [B] or None.
    """
    if reward_type is None:
        reward_type = os.environ.get("WORLD_REWARD_TYPE", "lpips_mae")
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

    visual_reward: Optional[torch.Tensor] = None
    if target_images is not None:
        target_f = target_images.to(device).float().clamp(0.0, 1.0)
        B_val, H_val = pred_future.shape[0], pred_future.shape[1]
        mae_per_step = (pred_future - target_f).abs().mean(dim=(2, 3, 4))

        if reward_type in ("lpips_mae", "hybrid"):
            lpips_model = _get_lpips_model(device)
            pred_flat = (pred_future.reshape(B_val * H_val, *pred_future.shape[2:]) * 2 - 1).clamp(-1.0, 1.0)
            tgt_flat  = (target_f.reshape(B_val * H_val, *target_f.shape[2:]) * 2 - 1).clamp(-1.0, 1.0)
            with torch.no_grad():
                lpips_flat = lpips_model(pred_flat, tgt_flat).squeeze(-1).squeeze(-1).squeeze(-1)
            lpips_per_step = lpips_flat.reshape(B_val, H_val)
            visual_reward = -(lpips_per_step.mean(dim=1) + mae_per_step.mean(dim=1))
        else:
            mse_per_step = ((pred_future - target_f) ** 2).mean(dim=(2, 3, 4))
            visual_reward = -mse_per_step.mean(dim=1)

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

    effective_type = reward_type
    if reward_type in ("rank_score", "hybrid") and rank_score_norm is None:
        logger.warning(
            "WORLD_REWARD_TYPE=%s requested but ranking_score unavailable. Falling back to lpips_mae.",
            reward_type,
        )
        effective_type = "lpips_mae"

    if effective_type in ("visual", "lpips_mae", "hybrid") and visual_reward is None:
        raise ValueError(
            "visual/lpips_mae/hybrid reward requested but target_images was not provided"
        )

    if effective_type in ("visual", "lpips_mae"):
        reward = visual_reward
    elif effective_type == "rank_score":
        reward = rank_score_norm
    else:
        reward = alpha * visual_reward + beta * rank_score_norm

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


def _save_png(path: Path, t: torch.Tensor) -> None:
    try:
        from PIL import Image as _PILImage
        arr = _tensor_to_uint8(t)
        _PILImage.fromarray(arr).save(path)
    except ImportError:
        pass


def run_smoke_test(
    *,
    checkpoint: str,
    output_dir: str,
    dtype: str = "bfloat16",
    device_name: str = "cpu",
    batch_size: int = 1,
    horizon: int = 8,
    image_size: int = 256,
    reward_type: str = "visual",
) -> Dict[str, Any]:
    """Smoke test: load model, run a forward pass, write debug_stats.json."""
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch_dtype = _dtype_from_name(dtype)

    model = load_world_model(checkpoint, torch_dtype=torch_dtype, device=device)
    cfg = model.cfg

    current = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    current[:, :, image_size // 4: image_size // 2, image_size // 4: image_size // 2] = 0.5
    actions = torch.zeros(batch_size, horizon, int(cfg.action_dim), device=device)
    history_images = current.unsqueeze(1).expand(batch_size, cfg.history_length, -1, -1, -1)

    out = predict_future(model, current, actions, history_images=history_images, horizon=horizon)

    pred = out["pred_future"]
    ranking_score = out.get("ranking_score")

    target_images = current.unsqueeze(1).expand(-1, horizon, -1, -1, -1)
    reward_out = compute_rft_reward(
        pred,
        target_images=target_images,
        ranking_score=ranking_score,
        reward_type=reward_type,
    )

    stats: Dict[str, Any] = {
        "checkpoint":           checkpoint,
        "model_generation":     "dynquery",
        "target_mode":          cfg.target_mode,
        "history_length":       cfg.history_length,
        "current_image_shape":  list(current.shape),
        "actions_shape":        list(actions.shape),
        "pred_future_shape":    list(pred.shape),
        "pred_future_min":      float(pred.min().item()),
        "pred_future_max":      float(pred.max().item()),
        "pred_future_mean":     float(pred.mean().item()),
        "reward_type":          reward_out["reward_type"],
        "visual_reward_mean":   reward_out["visual_reward_mean"],
        "rank_score_mean":      reward_out["rank_score_mean"],
        "rank_score_std":       reward_out["rank_score_std"],
        "hybrid_reward_mean":   reward_out["hybrid_reward_mean"],
    }
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
    parser = argparse.ArgumentParser(description="DynQueryWorldModel smoke test")
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--target-mode",   default="temporal_query_residual",
                        help="Ignored (read from checkpoint); kept for CLI compatibility")
    parser.add_argument("--model-generation", default="dynquery",
                        help="Ignored (always dynquery); kept for CLI compatibility")
    parser.add_argument("--dtype",         default="bfloat16")
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--horizon",       type=int, default=8)
    parser.add_argument("--image-size",    type=int, default=256)
    parser.add_argument("--reward-type",   default="visual",
                        choices=["visual", "lpips_mae", "rank_score", "hybrid"])
    args = parser.parse_args()

    reward_type = os.environ.get("WORLD_REWARD_TYPE", args.reward_type)
    stats = run_smoke_test(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        dtype=args.dtype,
        device_name=args.device,
        horizon=args.horizon,
        image_size=args.image_size,
        reward_type=reward_type,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
