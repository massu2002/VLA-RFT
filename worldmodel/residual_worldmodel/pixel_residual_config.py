"""Configuration for PixelResidualWorldModel.

target_mode controls what the model predicts and which losses are applied:

  "pixel"
      Predict the full future image directly.
      L = MSE(pred_future, gt_future)

  "pixel_residual"
      Predict residual = future - current; reconstruct as current + residual.
      L = lambda_residual * MSE(residual_pred, residual_target)
        + lambda_image    * MSE(pred_future, gt_future)

  "pixel_residual_roi_dynamic"
      pixel_residual + region-specific losses.
      L = lambda_residual * L_residual
        + lambda_image    * L_image
        + lambda_dynamic  * L_dynamic   (masked in dynamic region)
        + lambda_gripper  * L_gripper   (masked in gripper ROI)
        + lambda_static   * L_static    (penalise residual outside dynamic)

# TODO(latent_residual): add "latent_residual" target_mode that operates on
#   the FSQ embedding space of a frozen visual tokenizer.  Hook point:
#   PixelResidualWorldModel._forward_latent_residual() — not yet implemented.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class PixelResidualConfig:
    # ------------------------------------------------------------------ Target
    # "pixel" | "pixel_residual" | "pixel_residual_roi_dynamic"
    target_mode: str = "pixel_residual"

    # ------------------------------------------------------------------ Paths
    action_ranges_path: str = "train/verl/ivideogpt/configs/libero_action_ranges.pth"

    # ---------------------------------------------------------------- Actions
    action_dim: int = 7
    action_horizon: int = 7    # number of future steps = segment_length - 2

    # ------------------------------------------------------------ Architecture
    # Encoder output channels (spatial 8×8 feature map)
    encoder_channels: int = 256
    # Number of spatial tokens = 8*8 = 64 (derived from encoder downsample ×5)
    n_spatial_tokens: int = 64

    # Predictor transformer
    hidden_dim: int = 256
    n_heads: int = 4
    n_pred_layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.0

    # Action encoder MLP hidden size
    action_emb_dim: int = 128

    # -------------------------------------------------------------- Image dims
    image_height: int = 256
    image_width: int = 256
    image_channels: int = 3

    # --------------------------------------------------------------- Precision
    autocast_dtype: str = "bf16"

    # ------------------------------------------------------------------ Loss λ
    lambda_residual: float = 1.0
    lambda_image: float = 0.25
    lambda_dynamic: float = 2.0
    lambda_gripper: float = 2.0
    lambda_static: float = 0.5

    # ---------------------------------------------------------- Dynamic mask
    # Threshold on per-pixel |future - current| mean to define "dynamic" region
    dynamic_threshold: float = 0.05
    # Max-pool dilation kernel for dynamic mask (odd int)
    dynamic_dilate_kernel: int = 7

    # ---------------------------------------------------------------- ROI crop
    # Gripper ROI crop size in pixels (square)
    roi_crop_size: int = 64

    # -------------------------------------------------------- Debug / dry-run
    # Save debug images to results/phase1/residual_worldmodel/debug/
    save_debug_images: bool = False
    debug_output_dir: str = "results/phase1/residual_worldmodel/debug"

    # Dry-run: limit evaluation to this many windows (0 = unlimited)
    dry_run_windows: int = 0


def add_pixel_residual_args(parser: argparse.ArgumentParser) -> None:
    """Attach PixelResidualWorldModel-specific arguments to an ArgumentParser."""
    g = parser.add_argument_group("pixel residual world model")

    g.add_argument("--target-mode", type=str,
                   choices=["pixel", "pixel_residual", "pixel_residual_roi_dynamic"],
                   default="pixel_residual",
                   help="What the model predicts and which losses apply.")

    g.add_argument("--action-ranges-path", type=str,
                   default="train/verl/ivideogpt/configs/libero_action_ranges.pth")
    g.add_argument("--action-dim", type=int, default=7)
    g.add_argument("--action-horizon", type=int, default=7,
                   help="Number of future steps to predict (= segment_length - 2).")

    g.add_argument("--encoder-channels", type=int, default=256,
                   help="Output channels of the CNN encoder (feature map 8×8).")
    g.add_argument("--hidden-dim", type=int, default=256)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--n-pred-layers", type=int, default=4)
    g.add_argument("--ffn-dim", type=int, default=1024)
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--action-emb-dim", type=int, default=128)

    g.add_argument("--image-height", type=int, default=256)
    g.add_argument("--image-width", type=int, default=256)

    g.add_argument("--lambda-residual", type=float, default=1.0)
    g.add_argument("--lambda-image", type=float, default=0.25)
    g.add_argument("--lambda-dynamic", type=float, default=2.0)
    g.add_argument("--lambda-gripper", type=float, default=2.0)
    g.add_argument("--lambda-static", type=float, default=0.5)

    g.add_argument("--dynamic-threshold", type=float, default=0.05,
                   help="Pixel-diff threshold for dynamic mask (env: DYNAMIC_THRESHOLD).")
    g.add_argument("--dynamic-dilate-kernel", type=int, default=7,
                   help="Max-pool dilation kernel size for dynamic mask (odd).")
    g.add_argument("--roi-crop-size", type=int, default=64,
                   help="Gripper ROI crop size in pixels (env: ROI_CROP_SIZE).")

    g.add_argument("--save-debug-images", action="store_true", default=False)
    g.add_argument("--debug-output-dir", type=str,
                   default="results/phase1/residual_worldmodel/debug")
    g.add_argument("--dry-run-windows", type=int, default=0,
                   help="Limit eval to N windows for dry-run (0 = unlimited).")
