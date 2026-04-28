"""Configuration for ActionConditionedFocusedResidualWM (DINO-based)."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FocusedWMConfig:
    """Configuration for ActionConditionedFocusedResidualWM.

    Visual backbone: DINOv2 (non-optional).
    Task: current_image + action_chunk -> terminal future image.

    TODO(RFT): When connecting to policy optimization, expose
               `get_latent_features_for_rft()` in focused_model.py.
    """

    # ------------------------------------------------------------------ Paths
    action_ranges_path: str = ""
    dino_weights_path: str = ""  # local path; if empty, load via torch.hub

    # ----------------------------------------------------------- Model identity
    model_type: str = "ActionConditionedFocusedResidualWM"
    # Ablation variant tag — embedded in checkpoint dir names.
    # Presets: "full" | "no_focus" | "no_dino_loss" | "pixel_focus" | "image_rank"
    model_variant: str = "full"

    # ------------------------------------------------------- DINO backbone
    # Non-optional: DINO IS the visual encoder.
    dino_model_name: str = "dinov2_vits14"
    # Input size fed to DINO. Must be a multiple of the backbone patch size.
    # dinov2_vits14 patch_size=14 → 224 gives 16×16=256 patches.
    dino_input_size: int = 224
    # Feature dim per patch — inferred at runtime from the loaded backbone.
    # Set here as a default that matches dinov2_vits14.
    dino_feature_dim: int = 384
    # If True, DINO weights are completely frozen (recommended for stability).
    dino_frozen: bool = True
    # > 0: unfreeze last N transformer blocks only (ignored when dino_frozen=True).
    dino_finetune_last_n_layers: int = 0
    # Hub source string (change to load original DINO v1 models).
    dino_hub_source: str = "facebookresearch/dinov2"

    # ----------------------------------------- Derived spatial patch parameters
    # These are set at runtime by DinoFeatureExtractor.__init__().
    # Provide defaults that match dino_input_size=224 / patch_size=14.
    n_patches: int = 256   # (dino_input_size / patch_size)^2
    patch_hw: int = 16     # sqrt(n_patches)

    # --------------------------------------------------------- Action space
    action_dim: int = 7
    action_horizon: int = 7   # = segment_length - 1

    # --------------------------------------------------------- Architecture
    hidden_dim: int = 256
    action_emb_dim: int = 128
    n_action_enc_layers: int = 2
    n_pred_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.0

    # ---------------------------------------------------------- Image config
    image_height: int = 256
    image_width: int = 256
    image_channels: int = 3

    # ----------------------------------------------------- Focus head toggle
    use_focus_head: bool = True

    # -------------------------------------------------- Image residual options
    # If True, upsample focus_map to image space and gate pred_residual_image
    # before adding to current_image:
    #   pred_future = clamp(current + focus_img * residual, 0, 1)
    # If False (default), plain residual:
    #   pred_future = clamp(current + residual, 0, 1)
    use_focus_gated_image_residual: bool = False

    # ----------------------------------------- Decoder skip connection options
    # If True, a shallow CNN encodes current_image at two resolutions and injects
    # the features into the last 2 upsample stages of FutureImageDecoder.
    use_decoder_skip_connection: bool = True
    # Number of channels in each skip feature map.
    skip_base_channels: int = 32
    # Decoder upsampling mode:
    #   "convtranspose" : legacy ConvTranspose2d-based decoder
    #   "resize_conv"   : interpolate + conv + norm + activation
    decoder_upsample_mode: str = "convtranspose"
    # Interpolation mode used by resize_conv stages and refinement blocks.
    decoder_interp_mode: str = "bilinear"
    # If True, add light residual refinement after the highest-resolution stages.
    use_decoder_refine: bool = False
    # Number of high-resolution stages to refine (typically 1-2).
    num_decoder_refine_blocks: int = 2

    # ----------------------------------------- Dual focus mask (Phase 4)
    # If True, splits focus_logits into two masks:
    #   context_mask = sigmoid(focus_logits)          — soft/broad (predictor context)
    #   write_mask   = sigmoid(focus_logits / tau)    — sharp (token & image update)
    # If False, context_mask == write_mask (legacy single-mask behaviour).
    use_dual_focus_mask: bool = True
    # Temperature for write_mask sharpening.  Must be in (0, 1] — lower = sharper.
    write_mask_temperature: float = 0.5

    # ----------------------------------------- Optional auxiliary losses (Phase 4)
    # A. Foreground-weighted reconstruction (fg mask = upsampled change_target or write_mask)
    use_fg_recon_loss: bool = True
    fg_recon_weight: float = 0.1
    # B. Background residual penalty (suppresses residual in non-focus regions)
    use_bg_residual_penalty: bool = False
    bg_residual_weight: float = 0.01
    # C. Foreground gradient consistency (sharpens edges via gradient matching in fg)
    use_fg_grad_loss: bool = True
    fg_grad_weight: float = 0.02

    # --------------------------------------------------------- Loss weights
    # 1. Image reconstruction
    recon_loss_weight: float = 1.0

    # LPIPS perceptual loss (requires `pip install lpips`; expensive).
    use_lpips_loss: bool = False
    lpips_loss_weight: float = 0.1

    # 2. DINO future feature consistency
    use_dino_feature_loss: bool = True
    dino_feature_loss_weight: float = 0.1

    # 3. Focus supervision target selection (mutually exclusive recommended).
    # 3a. DINO-based (primary):  BCE(focus_map, normalize(||DINO_cur - DINO_fut||))
    use_dino_focus_supervision: bool = True
    # 3b. Pixel-based (ablation): BCE(focus_map, normalize(||cur_pixel - fut_pixel||))
    use_pixel_focus_supervision: bool = False
    focus_supervision_weight: float = 0.1
    # Threshold > 0 binarises the change target; 0 keeps it soft/normalised.
    change_target_threshold: float = 0.0

    # 4. Focus sparsity regularization.
    use_focus_sparsity: bool = True
    # "l1" = mean(focus), "entropy" = -p*log(p)-(1-p)*log(1-p)
    focus_sparsity_mode: str = "l1"
    focus_sparsity_weight: float = 0.01

    # ------------------------------------------------- Candidate ranking score
    # "dino_only"  — DINO cosine similarity of predicted vs GT future features
    # "image_only" — negative SmoothL1 of predicted vs GT future pixels
    # "combined"   — ranking_image_weight * image + (1 - w) * dino
    ranking_score_type: str = "dino_only"
    ranking_image_weight: float = 0.3   # used only for "combined"

    # -------------------------------------------- Negative generation (eval)
    negative_mode: str = "all"   # "roll" | "noise" | "shuffle" | "all"
    noise_std: float = 0.05
    num_action_candidates: int = 4

    # ---------------------------------------- Tiered 3-layer ranking eval
    # Replaces/extends the simple GT-vs-negative pairwise eval with a
    # 3-tier hierarchy: success / near_success / failure.
    # Set use_tiered_rank_eval=False to keep legacy-only behaviour.
    use_tiered_rank_eval: bool = True
    # 0 = use all items from probe_batches (no sub-sampling)
    num_rank_eval_items: int = 0
    num_near_success_candidates: int = 2
    num_failure_candidates: int = 3
    # near_success: GT + N(0, near_success_noise_std)
    near_success_noise_std: float = 0.05
    # failure: large noise / shuffle / roll (cycled)
    failure_noise_std: float = 0.30
    # artifact saving flags
    save_rank_eval_json: bool = True
    save_rank_eval_csv: bool = True
    save_rank_eval_plots: bool = True

    # -------------------------------- Fixed ranking benchmark (opt-in)
    # Set use_fixed_rank_eval_dataset=True to enable the paper-quality benchmark
    # that uses temporal-neighbor near-success and same-task hard negatives.
    use_fixed_rank_eval_dataset: bool = False
    # Non-empty: load benchmark from this path instead of auto-building it.
    rank_eval_dataset_path: str = ""
    # Force rebuild even if ranking_benchmark.pt already exists.
    regenerate_rank_eval_dataset: bool = False
    # How many episodes to include in the pool (more → richer hard negatives).
    num_benchmark_pool_episodes: int = 20
    # Anchor frames sampled per episode (interior windows only).
    num_benchmark_anchors_per_episode: int = 5
    # Comma-separated near-success candidate modes (tried in order per slot).
    near_success_modes_bench: str = "temporal_neighbor,small_noise"
    # Comma-separated failure candidate modes (cycled in order per slot).
    failure_modes_bench: str = "same_task_hard,same_task_mismatch,large_noise,shuffle"
    # Independent seed for the fixed benchmark (separate from training seed).
    fixed_rank_eval_seed: int = 1337

    # ------------------------------------------------------ Eval / viz schedule
    eval_every: int = 500
    save_viz_every: int = 1000

    # ------------------------------------------------------------ Debug mode
    # "fast_debug"  — tiny dataset, 10-step run for import/shape checks.
    # "normal"      — standard training.
    # "full_report" — per-step logging, more eval batches, all visualizations.
    debug_mode: str = "normal"

    # ---------------------------------------------------------------- Precision
    autocast_dtype: str = "bf16"

    # ----------------------------------------------- Backward compat (unused)
    visual_tokenizer_path: str = ""
    tokenizer_micro_batch_size: Optional[int] = None
    patch_feature_dim: int = 5  # FSQ legacy field; ignored by DINO model

    # -----------------------------------------------------------------------
    # Derived helpers (read-only)
    # -----------------------------------------------------------------------

    @property
    def n_decoder_upsample_steps(self) -> int:
        """Number of ×2 upsample steps needed: patch_hw → image_height."""
        ratio = self.image_height // self.patch_hw
        assert ratio > 0 and (ratio & (ratio - 1)) == 0, (
            f"image_height / patch_hw must be a power of 2, "
            f"got {self.image_height} / {self.patch_hw} = {ratio}"
        )
        return int(math.log2(ratio))

    def variant_tag(self) -> str:
        """Short string identifying this configuration for checkpoint naming."""
        parts = [self.model_variant, self.dino_model_name]
        if not self.use_focus_head:
            parts.append("nofocus")
        if not self.use_dino_feature_loss:
            parts.append("nodino")
        if self.use_pixel_focus_supervision and not self.use_dino_focus_supervision:
            parts.append("pixelfocus")
        if self.decoder_upsample_mode != "convtranspose":
            parts.append(f"dup={self.decoder_upsample_mode}")
        if self.decoder_interp_mode != "bilinear":
            parts.append(f"dint={self.decoder_interp_mode}")
        if self.use_decoder_refine:
            parts.append(f"drf={self.num_decoder_refine_blocks}")
        if self.use_dual_focus_mask and self.write_mask_temperature != 0.5:
            parts.append(f"wmt={self.write_mask_temperature}")
        if self.use_fg_recon_loss:
            parts.append(f"fgr={self.fg_recon_weight}")
        if self.use_bg_residual_penalty:
            parts.append(f"bgp={self.bg_residual_weight}")
        if self.use_fg_grad_loss:
            parts.append(f"fgg={self.fg_grad_weight}")
        parts.append(f"rank={self.ranking_score_type}")
        return "_".join(parts)

    def resolve_debug_defaults(self) -> None:
        """Override eval / viz schedule for fast_debug mode."""
        if self.debug_mode == "fast_debug":
            self.eval_every = 5
            self.save_viz_every = 10
        elif self.debug_mode == "full_report":
            self.eval_every = 100
            self.save_viz_every = 200


# ---------------------------------------------------------------------------
# Argparse helper — call from training entrypoints
# ---------------------------------------------------------------------------

def add_focused_wm_args(parser: argparse.ArgumentParser) -> None:
    """Attach all FocusedWM arguments to an ArgumentParser group."""
    g = parser.add_argument_group("focused world model")

    # DINO
    g.add_argument("--dino-model-name", type=str, default="dinov2_vits14",
                   help="DINOv2 model name: dinov2_vits14 | dinov2_vitb14 | …")
    g.add_argument("--dino-input-size", type=int, default=224,
                   help="Images resized to this before DINO (must be multiple of patch_size).")
    g.add_argument("--dino-weights-path", type=str, default="",
                   help="Local path to DINO weights; if empty, use torch.hub.")
    g.add_argument("--dino-hub-source", type=str, default="facebookresearch/dinov2")
    g.add_argument("--dino-frozen", action="store_true", default=True)
    g.add_argument("--dino-no-frozen", dest="dino_frozen", action="store_false")
    g.add_argument("--dino-finetune-last-n-layers", type=int, default=0)

    # Model variant (ablation)
    g.add_argument("--model-variant", type=str, default="full",
                   choices=["full", "no_focus", "no_dino_loss", "pixel_focus", "image_rank"])

    # Action
    g.add_argument("--action-dim", type=int, default=7)
    g.add_argument("--action-ranges-path", type=str, default="")

    # Architecture
    g.add_argument("--hidden-dim", type=int, default=256)
    g.add_argument("--action-emb-dim", type=int, default=128)
    g.add_argument("--n-action-enc-layers", type=int, default=2)
    g.add_argument("--n-pred-layers", type=int, default=4)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--ffn-dim", type=int, default=1024)
    g.add_argument("--dropout", type=float, default=0.0)

    # Image
    g.add_argument("--image-height", type=int, default=256)
    g.add_argument("--image-width", type=int, default=256)

    # Focus head
    g.add_argument("--use-focus-head", action="store_true", default=True)
    g.add_argument("--no-focus-head", dest="use_focus_head", action="store_false")

    # Decoder
    g.add_argument("--decoder-upsample-mode", type=str, default="convtranspose",
                   choices=["convtranspose", "resize_conv"])
    g.add_argument("--decoder-interp-mode", type=str, default="bilinear",
                   choices=["nearest", "bilinear", "bicubic"])
    g.add_argument("--use-decoder-refine", action="store_true", default=False)
    g.add_argument("--no-decoder-refine", dest="use_decoder_refine", action="store_false")
    g.add_argument("--num-decoder-refine-blocks", type=int, default=2)
    g.add_argument("--write-mask-temperature", type=float, default=0.5,
                   help="Temperature for dual focus write_mask sharpening.")

    # Loss weights
    g.add_argument("--recon-loss-weight", type=float, default=1.0)

    g.add_argument("--use-lpips-loss", action="store_true", default=False)
    g.add_argument("--lpips-loss-weight", type=float, default=0.1)

    g.add_argument("--use-dino-feature-loss", action="store_true", default=True)
    g.add_argument("--no-dino-feature-loss", dest="use_dino_feature_loss", action="store_false")
    g.add_argument("--dino-feature-loss-weight", type=float, default=0.1)

    g.add_argument("--use-dino-focus-supervision", action="store_true", default=True)
    g.add_argument("--no-dino-focus-supervision",
                   dest="use_dino_focus_supervision", action="store_false")
    g.add_argument("--use-pixel-focus-supervision", action="store_true", default=False)
    g.add_argument("--focus-supervision-weight", type=float, default=0.1)
    g.add_argument("--change-target-threshold", type=float, default=0.0)

    g.add_argument("--use-focus-sparsity", action="store_true", default=True)
    g.add_argument("--no-focus-sparsity", dest="use_focus_sparsity", action="store_false")
    g.add_argument("--focus-sparsity-mode", type=str, default="l1",
                   choices=["l1", "entropy"])
    g.add_argument("--focus-sparsity-weight", type=float, default=0.01)

    # Optional auxiliary losses
    g.add_argument("--use-fg-recon-loss", action="store_true", default=True)
    g.add_argument("--no-fg-recon-loss", dest="use_fg_recon_loss", action="store_false")
    g.add_argument("--fg-recon-weight", type=float, default=0.1)
    g.add_argument("--use-bg-residual-penalty", action="store_true", default=False)
    g.add_argument("--no-bg-residual-penalty",
                   dest="use_bg_residual_penalty", action="store_false")
    g.add_argument("--bg-residual-weight", type=float, default=0.01)
    g.add_argument("--use-fg-grad-loss", action="store_true", default=True)
    g.add_argument("--no-fg-grad-loss", dest="use_fg_grad_loss", action="store_false")
    g.add_argument("--fg-grad-weight", type=float, default=0.02)

    # Ranking
    g.add_argument("--ranking-score-type", type=str, default="dino_only",
                   choices=["dino_only", "image_only", "combined"])
    g.add_argument("--ranking-image-weight", type=float, default=0.3)

    # Negatives
    g.add_argument("--negative-mode", type=str, default="all",
                   choices=["roll", "noise", "shuffle", "all"])
    g.add_argument("--noise-std", type=float, default=0.05)
    g.add_argument("--num-action-candidates", type=int, default=4)

    # Tiered 3-layer ranking eval
    g.add_argument("--use-tiered-rank-eval", action="store_true", default=True)
    g.add_argument("--no-tiered-rank-eval",
                   dest="use_tiered_rank_eval", action="store_false")
    g.add_argument("--num-rank-eval-items", type=int, default=0,
                   help="Max items for tiered eval (0 = all probe batches).")
    g.add_argument("--num-near-success-candidates", type=int, default=2)
    g.add_argument("--num-failure-candidates", type=int, default=3)
    g.add_argument("--near-success-noise-std", type=float, default=0.05)
    g.add_argument("--failure-noise-std", type=float, default=0.30)
    g.add_argument("--save-rank-eval-json", action="store_true", default=True)
    g.add_argument("--no-save-rank-eval-json",
                   dest="save_rank_eval_json", action="store_false")
    g.add_argument("--save-rank-eval-csv", action="store_true", default=True)
    g.add_argument("--no-save-rank-eval-csv",
                   dest="save_rank_eval_csv", action="store_false")
    g.add_argument("--save-rank-eval-plots", action="store_true", default=True)
    g.add_argument("--no-save-rank-eval-plots",
                   dest="save_rank_eval_plots", action="store_false")

    # Fixed ranking benchmark
    g.add_argument("--use-fixed-rank-eval-dataset", action="store_true", default=False)
    g.add_argument("--no-fixed-rank-eval-dataset",
                   dest="use_fixed_rank_eval_dataset", action="store_false")
    g.add_argument("--rank-eval-dataset-path", type=str, default="",
                   help="Path to pre-built ranking_benchmark.pt; empty = auto-build.")
    g.add_argument("--regenerate-rank-eval-dataset", action="store_true", default=False,
                   help="Force rebuild even if ranking_benchmark.pt exists.")
    g.add_argument("--num-benchmark-pool-episodes", type=int, default=20)
    g.add_argument("--num-benchmark-anchors-per-episode", type=int, default=5)
    g.add_argument("--near-success-modes-bench", type=str,
                   default="temporal_neighbor,small_noise")
    g.add_argument("--failure-modes-bench", type=str,
                   default="same_task_hard,same_task_mismatch,large_noise,shuffle")
    g.add_argument("--fixed-rank-eval-seed", type=int, default=1337)

    # Eval / viz
    g.add_argument("--eval-every", type=int, default=500)
    g.add_argument("--save-viz-every", type=int, default=1000)

    # Debug
    g.add_argument("--debug-mode", type=str, default="normal",
                   choices=["fast_debug", "normal", "full_report"])
