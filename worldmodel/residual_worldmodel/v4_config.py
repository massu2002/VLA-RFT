"""Configuration for TemporalDynamicQueryResidualWM (v4).

target_mode = "temporal_query_residual"
model_generation = "v4"

Architecture stages:
  v4a  — Temporal Dynamic Query WM without ranking head
         (USE_ACTION_FUTURE_SCORER=0)
  v4b  — v4a + Action-Future Ranking Head
         (USE_ACTION_FUTURE_SCORER=1, LAMBDA_RANK=1.0)
  v4c  — v4b + RFT rank_score reward (WORLD_REWARD_TYPE=hybrid)
         (experimental; enabled via WORLD_REWARD_TYPE env at RFT time)

Loss:
  L_total =
    lambda_image   * L_image       (pred_future vs gt_future MSE)
  + lambda_dynamic * L_dynamic_roi (dynamic-masked MSE)
  + lambda_static  * L_static      (static consistency MSE)
  + lambda_query   * L_query       (pred vs GT future dynamic queries MSE)
  + lambda_rank    * L_action_rank (multi-negative InfoNCE, temperature=rank_temperature)
  + lambda_sparse  * L_mask_sparse (sparsity on dynamic/fuser masks)

  L_action_rank uses in-batch InfoNCE: for each positive (action_i, future_q_i.detach()),
  all batch negatives act as shared contrastive targets.
  future_dynamic_queries are detached from the scorer to prevent L_rank from
  corrupting the reconstruction-focused query predictor (H3 fix).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TemporalQueryResidualConfig:
    # ---------------------------------------------------------------- Identity
    target_mode: str = "temporal_query_residual"
    model_generation: str = "v4"

    # ---------------------------------------------------------------- Paths
    action_ranges_path: str = "train/verl/ivideogpt/configs/libero_action_ranges.pth"

    # ---------------------------------------------------------------- History
    # K: number of history frames preceding current frame
    history_length: int = 2

    # ---------------------------------------------------------------- Queries
    # Q: number of dynamic query vectors per frame
    num_dynamic_queries: int = 8

    # ------------------------------------------------------------ Architecture
    # Encoder output channels; spatial map is 8×8 (N=64) after 5× stride-2 CNN
    encoder_channels: int = 256
    n_spatial_tokens: int = 64    # (8×8), fixed for 256→8 encoder

    # Transformer hidden dimension (shared across all sub-modules)
    hidden_dim: int = 256
    n_heads: int = 4

    # TemporalResidualPredictor: context self-attn layers
    n_context_layers: int = 2
    # TokenFuser: spatial-update transformer layers (0 = skip)
    n_fuser_layers: int = 2
    # ActionFutureScorer: cross-attn decoder layers
    n_scorer_layers: int = 2

    ffn_dim: int = 1024
    dropout: float = 0.0

    # ---------------------------------------------------------------- Actions
    action_dim: int = 7
    action_bins: int = 256
    action_horizon: int = 7
    action_emb_dim: int = 128
    # "discrete_tokens" mirrors AR-Pixel action tokenization; "continuous_mlp" is simpler
    action_conditioning_mode: str = "discrete_tokens"

    # -------------------------------------------------------------- Image dims
    image_height: int = 256
    image_width: int = 256
    image_channels: int = 3

    # --------------------------------------------------------------- Precision
    autocast_dtype: str = "bf16"

    # ---------------------------------------------------------- Feature flags
    # USE_MOTION_BIAS: add ||z_t - z_{t-1}|| bias to mask logits
    use_motion_bias: bool = False
    # USE_ACTION_FUTURE_SCORER: v4b — add ranking head
    use_action_future_scorer: bool = True

    # -------------------------------------------------------------- Loss λ
    lambda_image: float = 0.1
    lambda_dynamic: float = 1.0
    lambda_static: float = 0.2
    lambda_query: float = 0.5
    lambda_rank: float = 1.0
    lambda_sparse: float = 0.01

    # InfoNCE temperature for ranking loss (lower = sharper distribution)
    # Replaces softplus margin loss; rank_margin is retained for backward compat
    # but ignored when rank_temperature > 0.
    rank_temperature: float = 0.07
    rank_margin: float = 0.1

    # ---------------------------------------------------- Mask sparsity loss
    # lambda_sparse: weight on per-mask entropy (concentrate each mask)
    # lambda_diversity: weight on pairwise cosine similarity across queries
    #   (penalize overlap, encourage different queries to attend to different tokens)
    lambda_diversity: float = 0.01

    # ------------------------------------------- Negative action construction
    # "temporal_perm" : temporal shuffle of correct actions (original)
    # "batch_roll"    : roll the batch → each sample gets another episode's actions
    # "mixed"         : batch_roll when B>1, temporal_perm as fallback (default)
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
    debug_output_dir: str = "results/phase1/residual_worldmodel/debug"
    dry_run_windows: int = 0


def add_v4_args(parser: argparse.ArgumentParser) -> None:
    """Attach v4-specific arguments to an ArgumentParser."""
    g = parser.add_argument_group("v4 temporal query residual world model")

    g.add_argument("--target-mode", type=str,
                   default="temporal_query_residual")
    g.add_argument("--model-generation", type=str, default="v4")
    g.add_argument("--action-ranges-path", type=str,
                   default="train/verl/ivideogpt/configs/libero_action_ranges.pth")

    # History
    g.add_argument("--history-length", type=int, default=2,
                   help="K: number of history frames before current (env: HISTORY_LENGTH).")

    # Queries
    g.add_argument("--num-dynamic-queries", type=int, default=8,
                   help="Q: dynamic query vectors per frame (env: NUM_DYNAMIC_QUERIES).")

    # Architecture
    g.add_argument("--encoder-channels", type=int, default=256)
    g.add_argument("--hidden-dim", type=int, default=256)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--n-context-layers", type=int, default=2)
    g.add_argument("--n-fuser-layers", type=int, default=2)
    g.add_argument("--n-scorer-layers", type=int, default=2)
    g.add_argument("--ffn-dim", type=int, default=1024)
    g.add_argument("--dropout", type=float, default=0.0)

    # Actions
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

    # Feature flags
    g.add_argument("--use-motion-bias", action="store_true", default=False,
                   help="Add motion bias to DynamicQueryExtractor mask logits.")
    g.add_argument("--no-motion-bias", dest="use_motion_bias", action="store_false")
    g.add_argument("--use-action-future-scorer", action="store_true", default=True,
                   help="v4b: add ActionFutureScorer ranking head.")
    g.add_argument("--no-action-future-scorer",
                   dest="use_action_future_scorer", action="store_false")

    # Loss weights
    g.add_argument("--lambda-image", type=float, default=0.1)
    g.add_argument("--lambda-dynamic", type=float, default=1.0)
    g.add_argument("--lambda-static", type=float, default=0.2)
    g.add_argument("--lambda-query", type=float, default=0.5)
    g.add_argument("--lambda-rank",      type=float, default=1.0)
    g.add_argument("--lambda-sparse",   type=float, default=0.01,
                   help="Weight on mask entropy concentration loss.")
    g.add_argument("--lambda-diversity", type=float, default=0.01,
                   help="Weight on pairwise query overlap penalty.")
    g.add_argument("--rank-temperature", type=float, default=0.07,
                   help="InfoNCE temperature for ranking loss (env: RANK_TEMPERATURE).")
    g.add_argument("--rank-margin",     type=float, default=0.1)
    g.add_argument("--negative-type",   type=str,   default="mixed",
                   choices=["temporal_perm", "batch_roll", "mixed"])

    # Output activations
    g.add_argument("--residual-output-activation", type=str,
                   choices=["tanh", "raw"], default="tanh")
    g.add_argument("--residual-output-scale", type=float, default=1.0)
    g.add_argument("--pixel-output-activation", type=str,
                   choices=["sigmoid", "clamp"], default="sigmoid")

    # Dynamic mask
    g.add_argument("--dynamic-threshold", type=float, default=0.05)
    g.add_argument("--dynamic-dilate-kernel", type=int, default=7)
    g.add_argument("--roi-crop-size", type=int, default=64)

    g.add_argument("--save-debug-images", action="store_true", default=False)
    g.add_argument("--debug-output-dir", type=str,
                   default="results/phase1/residual_worldmodel/debug")
    g.add_argument("--dry-run-windows", type=int, default=0)
