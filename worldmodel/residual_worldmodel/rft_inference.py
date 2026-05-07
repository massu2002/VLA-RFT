"""Phase 1 residual world model inference helpers for VLA-RFT.

The RFT pipeline consumes a future-image prediction when computing its
existing reconstruction/perceptual reward.  This module keeps the target-mode
specific reconstruction in one place:

  pixel                       pred_future = model(current, actions)
  pixel_residual              pred_future = current + residual_pred
  pixel_residual_roi_dynamic  pred_future = current + write_mask * residual_pred

Images are float [0, 1] unless noted otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .pixel_residual_model import PixelResidualWorldModel
from .pixel_residual_utils import _tensor_to_uint8


def _dtype_from_name(name: str) -> torch.dtype:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    if name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


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
) -> Dict[str, Any]:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = load_phase1_world_model(
        checkpoint,
        target_mode=target_mode,
        torch_dtype=_dtype_from_name(dtype),
        device=device,
    )
    current = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    current[:, :, image_size // 4: image_size // 2, image_size // 4: image_size // 2] = 0.5
    actions = torch.zeros(batch_size, horizon, int(model.cfg.action_dim), device=device)

    out = predict_future_with_world_model(model, current, actions, horizon=horizon)
    pred = out["pred_future"]
    write_mask = out.get("write_mask")

    stats = {
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
    parser.add_argument("--target-mode", required=True)
    parser.add_argument("--model-generation", default="v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    stats = run_smoke_test(
        checkpoint=args.checkpoint,
        target_mode=args.target_mode,
        model_generation=args.model_generation,
        output_dir=args.output_dir,
        dtype=args.dtype,
        device_name=args.device,
        horizon=args.horizon,
        image_size=args.image_size,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
