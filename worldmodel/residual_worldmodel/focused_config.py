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

    # Ranking
    g.add_argument("--ranking-score-type", type=str, default="dino_only",
                   choices=["dino_only", "image_only", "combined"])
    g.add_argument("--ranking-image-weight", type=float, default=0.3)

    # Negatives
    g.add_argument("--negative-mode", type=str, default="all",
                   choices=["roll", "noise", "shuffle", "all"])
    g.add_argument("--noise-std", type=float, default=0.05)
    g.add_argument("--num-action-candidates", type=int, default=4)

    # Eval / viz
    g.add_argument("--eval-every", type=int, default=500)
    g.add_argument("--save-viz-every", type=int, default=1000)

    # Debug
    g.add_argument("--debug-mode", type=str, default="normal",
                   choices=["fast_debug", "normal", "full_report"])
