from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResidualWorldModelConfig:
    """Configuration for LatentResidualWorldModel."""

    # --- Paths ---
    visual_tokenizer_path: str = ""
    action_ranges_path: str = "train/verl/ivideogpt/configs/libero_action_ranges.pth"

    # --- Action space ---
    action_dim: int = 7

    # --- Tokenizer ---
    # These are inferred from the loaded tokenizer at runtime; defaults match
    # the LIBERO CompressiveVQModelFSQ checkpoint (patch_size=4, 8×8 patches, FSQ dim=5).
    n_dyn_tokens: int = 64   # number of dynamic tokens per frame (8×8 patches)
    dyn_token_dim: int = 5   # FSQ code dimension per token
    tokenizer_micro_batch_size: Optional[int] = 4

    # --- Predictor architecture ---
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.0

    # --- Context token dimension (inferred from tokenizer at runtime) ---
    # ctx_tokens shape from CompressiveVQModelFSQ: [B, 1, ctx_dim]
    ctx_dim: int = 1024

    # --- Residual target mode ---
    # "adjacent_delta"      : baseline — zero-anchored adjacent diffs
    #                         gt_delta[t] = dyn_emb[t+1] - dyn_emb[t]
    # "current_anchor_ctx"  : main — cumulative residuals from z_curr, conditioned on ctx_tokens
    #                         gt_cum_delta[h] = z_future[h] - z_curr
    #                         anchor_token = emb_proj(z_curr) + ctx_proj(ctx_summary)
    residual_target_mode: str = "adjacent_delta"

    # --- Context source (current_anchor_ctx only) ---
    # "segment_initial"       : ctx_tokens from the first frame of the current segment window
    #                           (original behaviour — no episode metadata required).
    # "episode_initial_image" : ctx_tokens always from the first frame of the episode.
    #                           Requires include_episode_metadata=True in the dataset.
    ctx_source_mode: str = "segment_initial"

    # --- Temporal residual history (current_anchor_ctx only) ---
    # Number of within-window adjacent dynamic deltas to mean-pool into res_summary.
    # DISABLED (0) by default: in the current segment layout dyn_flat[B, T, D] contains
    # only frames 1..T, so dyn_flat[:, 0] = z_curr and dyn_flat[:, 1:] = z_future.
    # Any delta computed from dyn_flat pairs leaks future GT into the predictor, causing
    # train/rollout mismatch.  Enable only when a dedicated past-frame buffer is added.
    residual_history_len: int = 0

    # --- Reward-aligned loss (current_anchor_ctx only) ---
    # When True the primary loss is L_task_feature; latent MSE is demoted to auxiliary.
    use_reward_aligned_loss: bool = False

    # --- progress_head settings ---
    # "remaining_steps" : predict normalised remaining steps to episode success
    #                     (requires episode_length + window_start in batch).
    # "temporal_position": simple h/(H-1) position proxy (no episode metadata).
    progress_target_mode: str = "remaining_steps"
    normalize_remaining_steps: bool = True   # divide by episode_length when True

    # --- success_head settings ---
    # "goal_image_distance": predict latent distance to the episode goal image.
    #                        Requires episode_goal_pixels in batch.
    # "episode_displacement": cumulative displacement from z_curr (no episode metadata).
    success_target_mode: str = "goal_image_distance"
    goal_distance_space: str = "latent"          # currently only "latent" is supported
    goal_image_source: str = "episode_goal_image"  # source field name in batch

    # --- reward head architecture ---
    reward_head_hidden_dim: int = 128            # hidden dim of 2-layer MLP heads
    reward_head_activation: str = "gelu"         # "gelu" or "relu"

    # --- reward_proxy_head ---
    use_reward_proxy_head: bool = True           # include reward_proxy_head in training

    # --- Individual loss weights ---
    loss_weight_progress: float = 1.0      # progress head loss weight
    loss_weight_success: float = 0.5       # success head loss weight
    loss_weight_reward_proxy: float = 0.3  # reward_proxy head loss weight
    loss_weight_consistency: float = 0.1   # cumulative residual MSE (auxiliary)

    # --- Reconstruction loss (both modes, optional) ---
    recon_loss_weight: float = 0.0  # 0 = disabled; > 0 adds optional pixel recon loss

    # --- Horizon parameters ---
    # teacher_forced_horizon: how many future frames to predict in teacher-forced rollout
    #   (-1 = T-1, i.e. the full available future)
    teacher_forced_horizon: int = -1
    # autoregressive_horizon: number of steps for open-loop rollout_autoregressive()
    autoregressive_horizon: int = 4
    # reward_rollout_horizon: horizon used when computing score_dict for reward signals
    reward_rollout_horizon: int = 8
    # visualize_rollout_horizon: horizon used in visualize.py rollout (-1 = all)
    visualize_rollout_horizon: int = -1
    # action_start_offset: index into the action sequence where future actions begin
    #   (0 = use all T actions; 1 = skip action_0, use actions[1:] — matches CA training)
    action_start_offset: int = 1

    # --- Precision ---
    autocast_dtype: str = "bf16"

    @property
    def flat_dyn_dim(self) -> int:
        return self.n_dyn_tokens * self.dyn_token_dim


def add_residual_wm_args(parser: argparse.ArgumentParser) -> None:
    """Attach ResidualWorldModel-specific arguments to an existing ArgumentParser."""
    g = parser.add_argument_group("residual world model")

    # --- Paths / action space ---
    g.add_argument("--action-ranges-path", type=str,
                   default="train/verl/ivideogpt/configs/libero_action_ranges.pth",
                   help="Path to .pth with per-dim action min/max ranges.")
    g.add_argument("--action-dim", type=int, default=7)
    g.add_argument("--tokenizer-micro-batch-size", type=int, default=4,
                   help="Micro-batch size for tokenizer forward passes.")

    # --- Architecture ---
    g.add_argument("--hidden-dim", type=int, default=256,
                   help="Predictor transformer hidden dimension.")
    g.add_argument("--num-heads", type=int, default=4,
                   help="Number of attention heads in the predictor transformer.")
    g.add_argument("--num-layers", type=int, default=4,
                   help="Number of transformer layers in the predictor.")
    g.add_argument("--ffn-dim", type=int, default=1024,
                   help="Feed-forward dimension in the predictor transformer.")
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--ctx-dim", type=int, default=1024,
                   help="Dimension of static context tokens from the visual tokenizer.")

    # --- Mode ---
    g.add_argument("--residual-target-mode", type=str,
                   choices=["adjacent_delta", "current_anchor_ctx"],
                   default="adjacent_delta",
                   help=(
                       "'adjacent_delta': zero-anchored adjacent diffs (baseline). "
                       "'current_anchor_ctx': cumulative deltas from z_curr (main)."
                   ))

    # --- Context source ---
    g.add_argument("--ctx-source-mode", type=str,
                   choices=["segment_initial", "episode_initial_image"],
                   default="segment_initial",
                   help=(
                       "'segment_initial': ctx from first frame of current window (default). "
                       "'episode_initial_image': ctx always from episode frame_0 "
                       "(requires include_episode_metadata=True in dataset)."
                   ))

    # --- Temporal residual history ---
    g.add_argument("--residual-history-len", type=int, default=0,
                   help=(
                       "Adjacent dynamic deltas to pool into res_summary (0 = disabled). "
                       "Currently kept at 0: dyn_flat contains only frames 1..T so any "
                       "within-window delta leaks future GT into the predictor."
                   ))

    # --- Reward-aligned loss ---
    g.add_argument("--use-reward-aligned-loss", action="store_true", default=False,
                   help="Enable reward-aligned task-feature loss as primary loss.")

    # progress head
    g.add_argument("--progress-target-mode", type=str,
                   choices=["remaining_steps", "temporal_position"],
                   default="remaining_steps",
                   help="Target for progress_head.")
    g.add_argument("--normalize-remaining-steps", action="store_true", default=True,
                   help="Normalise remaining_steps by episode_length.")
    g.add_argument("--no-normalize-remaining-steps", dest="normalize_remaining_steps",
                   action="store_false")

    # success head
    g.add_argument("--success-target-mode", type=str,
                   choices=["goal_image_distance", "episode_displacement"],
                   default="goal_image_distance",
                   help="Target for success_head.")
    g.add_argument("--goal-distance-space", type=str, default="latent",
                   help="Space for goal distance computation ('latent' supported).")
    g.add_argument("--goal-image-source", type=str, default="episode_goal_image",
                   help="Batch field name for the goal image.")

    # reward head architecture
    g.add_argument("--reward-head-hidden-dim", type=int, default=128,
                   help="Hidden dim of 2-layer MLP reward heads.")
    g.add_argument("--reward-head-activation", type=str, default="gelu",
                   choices=["gelu", "relu"],
                   help="Activation for reward head MLP.")

    # reward proxy head
    g.add_argument("--use-reward-proxy-head", action="store_true", default=True)
    g.add_argument("--no-reward-proxy-head", dest="use_reward_proxy_head",
                   action="store_false")

    # loss weights
    g.add_argument("--loss-weight-progress", type=float, default=1.0)
    g.add_argument("--loss-weight-success", type=float, default=0.5)
    g.add_argument("--loss-weight-reward-proxy", type=float, default=0.3)
    g.add_argument("--loss-weight-consistency", type=float, default=0.1,
                   help="Weight for cumulative residual MSE (auxiliary).")

    # --- Reconstruction ---
    g.add_argument("--recon-loss-weight", type=float, default=0.0,
                   help="Weight for optional pixel reconstruction loss (0 = disabled).")

    # --- Horizon parameters ---
    g.add_argument("--teacher-forced-horizon", type=int, default=-1,
                   help="Future frames in teacher-forced rollout (-1 = full window).")
    g.add_argument("--autoregressive-horizon", type=int, default=4,
                   help="Steps for open-loop autoregressive rollout.")
    g.add_argument("--reward-rollout-horizon", type=int, default=8,
                   help="Horizon for score_dict computation in rollout APIs.")
    g.add_argument("--visualize-rollout-horizon", type=int, default=-1,
                   help="Horizon used in visualize.py rollout (-1 = all).")
    g.add_argument("--action-start-offset", type=int, default=1,
                   help="Index into action sequence where future actions begin (1 = skip action_0).")
