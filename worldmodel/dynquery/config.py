"""Configuration for DynQueryWorldModel (Minimal Action-Conditioned Dynamic Query World Model).

Core modifications enabled by default:
  Core 1: DynamicQueryExtractor action-conditioned mask  (use_action_conditioned_mask=True)
  Core 2: TemporalResidualPredictor query_wise mode      (predictor_mode="query_wise")
  Core 3: PixelDecoder dynamic residual gate             (use_dynamic_residual_gate=True)
  Core 4: L_mask_dynamic + L_query_delta_sparse losses

Loss:
  L_total =
    lambda_image          * L_image       (MSE pred_future vs gt_future)
  + lambda_dynamic        * L_dynamic     (dynamic-masked MSE)
  + lambda_static         * L_static      (static-region consistency MSE)
  + lambda_query          * L_query       (pred vs GT future dynamic queries MSE)
  + lambda_rank           * L_rank        (multi-negative InfoNCE, optional)
  + lambda_mask_dynamic   * L_mask_dynamic (query mask union vs GT dynamic region BCE)
  + lambda_query_delta_sparse * L_query_delta_sparse (L2 sparsity on query deltas)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class DynQueryConfig:
    # ---------------------------------------------------------------- Identity
    target_mode: str = "temporal_query_residual"
    model_generation: str = "dynquery"

    # ---------------------------------------------------------------- Paths
    action_ranges_path: str = "train/verl/ivideogpt/configs/libero_action_ranges.pth"

    # ---------------------------------------------------------------- History
    history_length: int = 2           # K: history frames before current

    # ---------------------------------------------------------------- Queries
    num_dynamic_queries: int = 8      # Q: dynamic query vectors per frame

    # ------------------------------------------------------------ Architecture
    encoder_channels: int = 256
    n_spatial_tokens: int = 64        # 8×8 fixed for 256→8 encoder

    hidden_dim: int = 256
    n_heads: int = 4

    n_context_layers: int = 2         # TemporalResidualPredictor context self-attn
    n_fuser_layers: int = 2           # TokenFuser spatial-update layers
    n_scorer_layers: int = 2          # ActionFutureScorer cross-attn layers

    ffn_dim: int = 1024
    dropout: float = 0.0

    # ---------------------------------------------------------------- Actions
    action_dim: int = 7
    action_bins: int = 256
    action_horizon: int = 7
    action_emb_dim: int = 128
    action_conditioning_mode: str = "discrete_tokens"

    # -------------------------------------------------------------- Image dims
    image_height: int = 256
    image_width: int = 256
    image_channels: int = 3

    # --------------------------------------------------------------- Precision
    autocast_dtype: str = "bf16"

    # ------------------------------------------------------ Core feature flags
    # All enabled by default in the simplified model.
    use_motion_bias: bool = True
    use_action_future_scorer: bool = True
    use_action_conditioned_mask: bool = True   # Core 1
    predictor_mode: str = "query_wise"         # Core 2: "query_wise" | "linear_expand"
    use_dynamic_residual_gate: bool = True     # Core 3

    # -------------------------------------------------------------- Loss λ
    lambda_image: float = 0.1
    lambda_dynamic: float = 1.0
    lambda_static: float = 0.2
    lambda_query: float = 0.5
    lambda_rank: float = 1.0
    lambda_mask_dynamic: float = 0.1          # Core 4a
    lambda_query_delta_sparse: float = 0.001  # Core 4b

    # InfoNCE temperature for ranking loss
    rank_temperature: float = 0.07
    rank_margin: float = 0.1

    # ------------------------------------------- Negative action construction
    negative_type: str = "mixed"

    # ----------------------------------------------------- Output activation
    residual_output_activation: str = "tanh"
    residual_output_scale: float = 1.0
    pixel_output_activation: str = "sigmoid"

    # ---------------------------------------------------------- Dynamic mask
    dynamic_threshold: float = 0.05
    dynamic_dilate_kernel: int = 7

    # ---------------------------------------------------------------- ROI
    roi_crop_size: int = 64

    # --------------------------------------------------------- Debug / dry-run
    save_debug_images: bool = False
    debug_output_dir: str = "results/dynquery/debug"
    dry_run_windows: int = 0

    # ---------------------------------------- Deprecated / backward-compat fields
    # Kept so old checkpoint configs can be loaded without error; ignored by the model.
    lambda_sparse: float = 0.0       # deprecated: was mask entropy loss weight
    lambda_diversity: float = 0.0   # deprecated: was mask diversity loss weight
    fuser_mode: str = "single_mask"  # deprecated: only single_mask supported
    dynamic_gate_source: str = "fuser_masks"  # deprecated: always fuser_masks
    use_cumulative_query_delta: bool = False   # deprecated: not used
    use_action_conditioned_mask_v0: bool = False  # deprecated alias


# Backward-compat alias for code that still uses the old name.
TemporalQueryResidualConfig = DynQueryConfig


def add_dynquery_args(parser: argparse.ArgumentParser) -> None:
    """Attach DynQuery model arguments to an ArgumentParser."""
    g = parser.add_argument_group("dynquery world model")

    g.add_argument("--target-mode", type=str, default="temporal_query_residual")
    g.add_argument("--model-generation", type=str, default="dynquery")
    g.add_argument("--action-ranges-path", type=str,
                   default="train/verl/ivideogpt/configs/libero_action_ranges.pth")

    g.add_argument("--history-length", type=int, default=2)
    g.add_argument("--num-dynamic-queries", type=int, default=8)

    g.add_argument("--encoder-channels", type=int, default=256)
    g.add_argument("--hidden-dim", type=int, default=256)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--n-context-layers", type=int, default=2)
    g.add_argument("--n-fuser-layers", type=int, default=2)
    g.add_argument("--n-scorer-layers", type=int, default=2)
    g.add_argument("--ffn-dim", type=int, default=1024)
    g.add_argument("--dropout", type=float, default=0.0)

    g.add_argument("--action-dim", type=int, default=7)
    g.add_argument("--action-bins", type=int, default=256)
    g.add_argument("--action-horizon", type=int, default=7)
    g.add_argument("--action-emb-dim", type=int, default=128)
    g.add_argument("--action-conditioning-mode", type=str,
                   choices=["discrete_tokens", "continuous_mlp"],
                   default="discrete_tokens")

    g.add_argument("--image-height", type=int, default=256)
    g.add_argument("--image-width", type=int, default=256)
    g.add_argument("--autocast-dtype", type=str, default="bf16")

    # Core feature flags
    g.add_argument("--use-motion-bias", action="store_true", default=True)
    g.add_argument("--no-motion-bias", dest="use_motion_bias", action="store_false")
    g.add_argument("--use-action-future-scorer", action="store_true", default=True)
    g.add_argument("--no-action-future-scorer",
                   dest="use_action_future_scorer", action="store_false")
    g.add_argument("--use-action-conditioned-mask", action="store_true", default=True)
    g.add_argument("--no-action-conditioned-mask",
                   dest="use_action_conditioned_mask", action="store_false")
    g.add_argument("--predictor-mode", type=str, default="query_wise",
                   choices=["query_wise", "linear_expand"])
    g.add_argument("--use-dynamic-residual-gate", action="store_true", default=True)
    g.add_argument("--no-dynamic-residual-gate",
                   dest="use_dynamic_residual_gate", action="store_false")

    # Loss weights
    g.add_argument("--lambda-image", type=float, default=0.1)
    g.add_argument("--lambda-dynamic", type=float, default=1.0)
    g.add_argument("--lambda-static", type=float, default=0.2)
    g.add_argument("--lambda-query", type=float, default=0.5)
    g.add_argument("--lambda-rank", type=float, default=1.0)
    g.add_argument("--lambda-mask-dynamic", type=float, default=0.1)
    g.add_argument("--lambda-query-delta-sparse", type=float, default=0.001)

    g.add_argument("--rank-temperature", type=float, default=0.07)
    g.add_argument("--rank-margin", type=float, default=0.1)
    g.add_argument("--negative-type", type=str, default="mixed",
                   choices=["temporal_perm", "batch_roll", "mixed"])

    g.add_argument("--residual-output-activation", type=str,
                   choices=["tanh", "raw"], default="tanh")
    g.add_argument("--residual-output-scale", type=float, default=1.0)
    g.add_argument("--pixel-output-activation", type=str,
                   choices=["sigmoid", "clamp"], default="sigmoid")

    g.add_argument("--dynamic-threshold", type=float, default=0.05)
    g.add_argument("--dynamic-dilate-kernel", type=int, default=7)
    g.add_argument("--roi-crop-size", type=int, default=64)

    g.add_argument("--save-debug-images", action="store_true", default=False)
    g.add_argument("--debug-output-dir", type=str, default="results/dynquery/debug")
    g.add_argument("--dry-run-windows", type=int, default=0)


# Backward-compat alias.
add_v4_args = add_dynquery_args
