"""ActionConditionedFocusedResidualWM — DINO-based terminal future image prediction.

Architecture
------------
  current_image [B, C, H, W]       +      actions [B, H_steps, 7]
        │                                         │
  DinoFeatureExtractor              ActionChunkEncoder
  (DINOv2 frozen backbone               small transformer
   + learned linear proj)               mean-pool → [B, Da]
        │  patch_tokens [B, N, D]              │
        └──────────────┬───────────────────────┘
                       │
                  FocusHead → focus_map [B, N] ∈ [0, 1]
                       │
             ResidualFuturePredictor → future_tokens [B, N, D]
                       │
             FutureImageDecoder → pred_future [B, 3, H, W] ∈ [0, 1]

Losses (when future_pixels provided)
-------------------------------------
  recon          : SmoothL1(pred_future, gt_future)                    [always]
  dino_feature   : MSE(future_tokens, proj(DINO(gt_future)))          [optional]
  focus_sup      : BCE(focus_map, change_target)                      [optional]
    - dino mode  :   change_target = normalize(||DINO_cur - DINO_fut||)
    - pixel mode :   change_target = normalize(||cur_pixel - fut_pixel||)
  focus_sparsity : L1 or entropy of focus_map                         [optional]
  lpips          : LPIPS perceptual loss                               [optional]

TODO(RFT): Future connection — call get_latent_features_for_rft() from policy trainer
           to obtain (focus_map, future_tokens, pred_future_image) for reward shaping.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focused_config import FocusedWMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet normalisation constants for DINO
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# DinoFeatureExtractor — the sole visual encoder (non-optional)
# ---------------------------------------------------------------------------

class DinoFeatureExtractor(nn.Module):
    """Frozen (or partially fine-tuned) DINOv2 backbone + learned projection.

    Inputs must be float images in [0, 1]; normalisation to ImageNet stats is
    applied internally before feeding the backbone.

    The backbone is always DINOv2 (non-optional by design).  All downstream
    losses compare features in either the projected (hidden_dim) space or the
    raw DINO space — see extract_raw() vs forward().
    """

    def __init__(self, cfg: FocusedWMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ---- Load backbone -----------------------------------------------
        backbone = self._load_backbone(cfg)

        # ---- Freeze / partial fine-tune ----------------------------------
        if cfg.dino_frozen:
            backbone.requires_grad_(False)
            backbone.eval()
            logger.info("DINO backbone frozen (all params).")
        elif cfg.dino_finetune_last_n_layers > 0:
            backbone.requires_grad_(False)
            blocks = list(backbone.blocks)
            n_unfreeze = cfg.dino_finetune_last_n_layers
            for blk in blocks[-n_unfreeze:]:
                blk.requires_grad_(True)
            backbone.norm.requires_grad_(True)
            logger.info("DINO: unfroze last %d blocks + norm.", n_unfreeze)
        else:
            logger.info("DINO backbone fully trainable.")

        self.backbone = backbone

        # ---- Infer dims from loaded backbone -----------------------------
        dino_feat_dim = int(backbone.embed_dim)
        patch_size    = int(backbone.patch_embed.proj.kernel_size[0])
        n_patches     = (cfg.dino_input_size // patch_size) ** 2
        patch_hw      = int(math.sqrt(n_patches))

        # Update cfg in-place so downstream modules see correct values
        cfg.dino_feature_dim = dino_feat_dim
        cfg.n_patches        = n_patches
        cfg.patch_hw         = patch_hw

        logger.info(
            "DINO: model=%s  feat_dim=%d  patch_size=%d  "
            "input_size=%d  n_patches=%d  patch_hw=%d",
            cfg.dino_model_name, dino_feat_dim, patch_size,
            cfg.dino_input_size, n_patches, patch_hw,
        )

        # ---- Learned projection ------------------------------------------
        self.proj = nn.Linear(dino_feat_dim, cfg.hidden_dim)

        # ---- Normalisation buffers (non-trainable) -----------------------
        self.register_buffer(
            "_norm_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_norm_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    # ---- Construction helpers -------------------------------------------

    @staticmethod
    def _load_backbone(cfg: FocusedWMConfig) -> nn.Module:
        if cfg.dino_weights_path:
            backbone = torch.hub.load(
                cfg.dino_hub_source,
                cfg.dino_model_name,
                pretrained=False,
                verbose=False,
            )
            state = torch.load(cfg.dino_weights_path, map_location="cpu")
            backbone.load_state_dict(state, strict=False)
            logger.info("DINO loaded from local weights: %s", cfg.dino_weights_path)
        else:
            try:
                backbone = torch.hub.load(
                    cfg.dino_hub_source,
                    cfg.dino_model_name,
                    verbose=False,
                )
                logger.info("DINO loaded via torch.hub: %s/%s",
                            cfg.dino_hub_source, cfg.dino_model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load DINO backbone '{cfg.dino_model_name}' "
                    f"from '{cfg.dino_hub_source}'. "
                    "Set --dino-weights-path to load from a local file, or "
                    "ensure internet access for torch.hub."
                ) from exc
        return backbone

    # ---- Internal helpers -----------------------------------------------

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Float [0,1] [B,3,H,W] → ImageNet-normalised, resized to dino_input_size."""
        H = self.cfg.dino_input_size
        if images.shape[-2] != H or images.shape[-1] != H:
            images = F.interpolate(
                images, size=(H, H), mode="bilinear", align_corners=False
            )
        mean = self._norm_mean.to(images.dtype)
        std  = self._norm_std.to(images.dtype)
        return (images - mean) / std

    def train(self, mode: bool = True) -> "DinoFeatureExtractor":
        """Keep frozen backbone in eval() even when the parent model is set to train()."""
        super().train(mode)
        if self.cfg.dino_frozen and mode:
            self.backbone.eval()
        return self

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone; return patch tokens [B, N, D_dino]."""
        if self.cfg.dino_frozen:
            with torch.no_grad():
                out = self.backbone.forward_features(x)
        else:
            out = self.backbone.forward_features(x)
        return out["x_norm_patchtokens"].float()

    # ---- Public API -------------------------------------------------------

    @torch.no_grad()
    def extract_raw(self, images: torch.Tensor) -> torch.Tensor:
        """Extract raw DINO patch features (no projection).

        Used for loss computation targets — always no_grad regardless of
        backbone frozen/finetuned setting, because we treat these as targets.

        Args:
            images: [B, 3, H, W] float [0, 1]
        Returns:
            [B, N_patches, D_dino] float32
        """
        x = self._preprocess(images)
        out = self.backbone.forward_features(x)
        return out["x_norm_patchtokens"].float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINO features and project to hidden_dim.

        Args:
            images: [B, 3, H, W] float [0, 1]
        Returns:
            patch_tokens: [B, N_patches, hidden_dim]
        """
        x = self._preprocess(images)
        raw = self._run_backbone(x)                              # [B, N, D_dino] float32
        return self.proj(raw.to(self.proj.weight.dtype))         # [B, N, hidden_dim]

    def forward_with_raw(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (patch_tokens [B,N,D_h], raw_dino [B,N,D_dino]).

        Avoids a second backbone pass when both projected and raw are needed.
        Raw features are returned with no_grad applied (used as loss targets).
        """
        x = self._preprocess(images)
        raw = self._run_backbone(x)                              # [B, N, D_dino] float32
        projected = self.proj(raw.to(self.proj.weight.dtype))    # [B, N, hidden_dim]
        return projected, raw.detach()


# ---------------------------------------------------------------------------
# ActionChunkEncoder
# ---------------------------------------------------------------------------

class ActionChunkEncoder(nn.Module):
    """Encode an action chunk [B, H, action_dim] → embedding [B, action_emb_dim]."""

    def __init__(self, cfg: FocusedWMConfig) -> None:
        super().__init__()
        D  = cfg.hidden_dim
        Da = cfg.action_emb_dim

        self.input_proj = nn.Linear(cfg.action_dim, D)
        self.pos_emb    = nn.Parameter(torch.zeros(1, cfg.action_horizon, D))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_action_enc_layers)
        self.out_proj    = nn.Linear(D, Da)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """actions [B, H, action_dim] → action_emb [B, action_emb_dim]."""
        B, H, _ = actions.shape
        x = self.input_proj(actions)                  # [B, H, D]
        x = x + self.pos_emb[:, :H, :].to(x.dtype)
        x = self.transformer(x)                       # [B, H, D]
        return self.out_proj(x.mean(dim=1))           # [B, Da]


# ---------------------------------------------------------------------------
# FocusHead
# ---------------------------------------------------------------------------

class FocusHead(nn.Module):
    """Per-patch soft focus logits conditioned on action embedding.

    Returns logits of shape [B, N].
    Probability map can be obtained with torch.sigmoid(logits).
    """

    def __init__(self, cfg: FocusedWMConfig) -> None:
        super().__init__()
        D  = cfg.hidden_dim
        Da = cfg.action_emb_dim

        self.patch_proj  = nn.Linear(D, D)
        self.action_proj = nn.Linear(Da, D)
        self.scorer = nn.Sequential(
            nn.GELU(),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,  # [B, N, D]
        action_emb: torch.Tensor,    # [B, Da]
    ) -> torch.Tensor:               # [B, N] logits
        p = self.patch_proj(patch_tokens)              # [B, N, D]
        a = self.action_proj(action_emb).unsqueeze(1)  # [B, 1, D]
        return self.scorer(p + a).squeeze(-1)

# ---------------------------------------------------------------------------
# ResidualFuturePredictor
# ---------------------------------------------------------------------------

class ResidualFuturePredictor(nn.Module):
    """Predict delta patch tokens conditioned on action and focus map.

    Returns delta_tokens [B, N, D] — caller applies strict masked residual:
        future_tokens = current_tokens + focus.unsqueeze(-1) * delta_tokens

    focused_tokens = focus * F                    ← mask background before transformer
    x = [action_tok | focused_tokens]             ← action-conditioned foreground only
    x → transformer → delta_ctx [B, N, D]
    delta_tokens = fuser(concat[delta_ctx, focused_tokens])
                                                  ← fuser sees focused input, not full F
    """

    def __init__(self, cfg: FocusedWMConfig) -> None:
        super().__init__()
        D  = cfg.hidden_dim
        Da = cfg.action_emb_dim

        self.action_tok_proj = nn.Linear(Da, D)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_pred_layers)

        self.fuser = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        # Zero-init output linear: initial delta ≈ 0 for stable warm-up
        nn.init.zeros_(self.fuser[-1].weight)
        nn.init.zeros_(self.fuser[-1].bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,  # [B, N, D]
        action_emb: torch.Tensor,    # [B, Da]
        focus_map: torch.Tensor,     # [B, N]
    ) -> torch.Tensor:               # [B, N, D]  delta_tokens
        focus = focus_map.unsqueeze(-1)                                        # [B, N, 1]
        focused_tokens = focus * patch_tokens                                  # [B, N, D]

        action_tok = self.action_tok_proj(action_emb).unsqueeze(1)            # [B, 1, D]
        x = torch.cat([action_tok, focused_tokens], dim=1)                    # [B, N+1, D]
        x = self.transformer(x)                                                # [B, N+1, D]
        delta_ctx = x[:, 1:, :]                                               # [B, N, D]

        delta_tokens = self.fuser(torch.cat([delta_ctx, focused_tokens], dim=-1))  # [B, N, D]
        return delta_tokens


# ---------------------------------------------------------------------------
# CurrentImageSkipEncoder
# ---------------------------------------------------------------------------

class CurrentImageSkipEncoder(nn.Module):
    """Shallow CNN that extracts multi-scale spatial features from current_image.

    Produces skip features at two resolutions (H and H/2) that are injected
    into the last two upsample stages of FutureImageDecoder.

    Kept as float32 weights (not cast to bf16) so current_f [float32] can be
    fed directly.  Inside the decoder each skip tensor is cast to match the
    decoder feature dtype (e.g. bf16 during autocast).

    Output: dict  spatial_size → tensor [B, skip_base_channels, H', W']
    """

    def __init__(self, cfg: FocusedWMConfig) -> None:
        super().__init__()
        base = cfg.skip_base_channels

        # Full resolution: [B, 3, H, W] → [B, base, H, W]
        self.conv_full = nn.Sequential(
            nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.GELU(),
        )
        # Half resolution: [B, base, H, W] → [B, base, H/2, W/2]
        self.conv_half = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """x [B, 3, H, W] → {H: [B, base, H, W], H//2: [B, base, H//2, W//2]}"""
        s_full = self.conv_full(x)          # [B, base, H, W]
        s_half = self.conv_half(s_full)     # [B, base, H/2, W/2]
        return {x.shape[-1]: s_full, x.shape[-1] // 2: s_half}


# ---------------------------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------------------------

def _decoder_norm(channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels)


class DecoderUpsampleStage(nn.Module):
    """Single decoder upsample stage with legacy and resize-conv modes."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        upsample_mode: str,
        interp_mode: str,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.interp_mode = interp_mode

        if upsample_mode == "convtranspose":
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                _decoder_norm(out_ch),
                nn.GELU(),
            )
        elif upsample_mode == "resize_conv":
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
            self.norm = _decoder_norm(out_ch)
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported decoder_upsample_mode: {upsample_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample_mode == "convtranspose":
            return self.block(x)

        kwargs = {}
        if self.interp_mode != "nearest":
            kwargs["align_corners"] = False
        x = F.interpolate(x, scale_factor=2, mode=self.interp_mode, **kwargs)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class DecoderRefineBlock(nn.Module):
    """Light residual refinement block for high-resolution decoder features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _decoder_norm(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _decoder_norm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return self.act(residual + y)


# ---------------------------------------------------------------------------
# FutureImageDecoder
# ---------------------------------------------------------------------------

class FutureImageDecoder(nn.Module):
    """Decode future patch tokens → residual image [B, 3, H, W] (unconstrained).

    Caller adds residual to current_image and clamps to [0, 1]:
        pred_future_image = clamp(current_image + decoder(future_tokens, ...), 0, 1)

    When skip_features (dict: spatial_size → tensor) is provided, the last
    `n_skip_stages` upsample stages concat and fuse the matching skip feature.
    This injects high-frequency current-image detail into the residual path.

    Number of ×2 upsample steps is derived from cfg.n_decoder_upsample_steps.
    Works for any patch_hw that is a power-of-2 divisor of image_height.
    """

    def __init__(self, cfg: FocusedWMConfig) -> None:
        super().__init__()
        D       = cfg.hidden_dim
        n_steps = cfg.n_decoder_upsample_steps  # e.g. 4 for 16×16 → 256×256

        # Build upsample stages as explicit ModuleList so skip can be injected
        self.stages: nn.ModuleList = nn.ModuleList()
        self.stage_out_ch: list    = []
        self.patch_hw = cfg.patch_hw
        self.decoder_upsample_mode = cfg.decoder_upsample_mode
        self.decoder_interp_mode = cfg.decoder_interp_mode
        self.use_decoder_refine = cfg.use_decoder_refine

        in_ch = D
        for _ in range(n_steps):
            out_ch = max(in_ch // 2, 32)
            self.stages.append(
                DecoderUpsampleStage(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    upsample_mode=cfg.decoder_upsample_mode,
                    interp_mode=cfg.decoder_interp_mode,
                )
            )
            self.stage_out_ch.append(out_ch)
            in_ch = out_ch

        # Skip fusion convs — applied to the last min(2, n_steps) upsample stages.
        # skip_fuse[str(i)] fuses (stage_out_ch[i] + skip_ch) → stage_out_ch[i].
        n_skip = min(2, n_steps)
        self._skip_stage_ids: set = set(range(n_steps - n_skip, n_steps))

        if cfg.use_decoder_skip_connection:
            skip_ch = cfg.skip_base_channels
            self.skip_fuse: Optional[nn.ModuleDict] = nn.ModuleDict({
                str(i): nn.Sequential(
                    nn.Conv2d(self.stage_out_ch[i] + skip_ch,
                              self.stage_out_ch[i],
                              kernel_size=3, padding=1),
                    nn.GELU(),
                )
                for i in self._skip_stage_ids
            })
        else:
            self.skip_fuse = None

        max_refine = min(max(cfg.num_decoder_refine_blocks, 0), n_steps, 2)
        self._refine_stage_ids: set = set(range(n_steps - max_refine, n_steps))
        if self.use_decoder_refine and max_refine > 0:
            self.refine_blocks: Optional[nn.ModuleDict] = nn.ModuleDict({
                str(i): DecoderRefineBlock(
                    channels=self.stage_out_ch[i],
                )
                for i in self._refine_stage_ids
            })
        else:
            self.refine_blocks = None

        self.out_conv = nn.Conv2d(self.stage_out_ch[-1], 3, kernel_size=1)
        # Zero-init: initial residual ≈ 0 → pred_future ≈ current at init
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        future_tokens: torch.Tensor,
        skip_features: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Args:
            future_tokens  : [B, N, D]
            skip_features  : dict {spatial_size: [B, skip_ch, H', W']} or None
        Returns:
            pred_residual_image : [B, 3, H, W]  (unconstrained, raw residual)
        """
        B, N, D = future_tokens.shape
        ph = self.patch_hw
        x = future_tokens.permute(0, 2, 1).reshape(B, D, ph, ph)   # [B, D, ph, ph]

        for i, stage in enumerate(self.stages):
            x = stage(x)                                             # [B, C, H', W']

            if (self.skip_fuse is not None
                    and i in self._skip_stage_ids
                    and skip_features is not None):
                spatial = x.shape[-1]
                skip = skip_features.get(spatial)
                if skip is not None:
                    skip = skip.to(x.dtype)                          # match bf16/fp32
                    x = self.skip_fuse[str(i)](
                        torch.cat([x, skip], dim=1)
                    )                                                # [B, C, H', W']

            if self.refine_blocks is not None and i in self._refine_stage_ids:
                x = self.refine_blocks[str(i)](x)

        return self.out_conv(x)   # raw residual — no sigmoid


# ---------------------------------------------------------------------------
# Change target helpers
# ---------------------------------------------------------------------------

def compute_dino_change_target(
    current_raw_dino: torch.Tensor,   # [B, N, D_dino]
    future_raw_dino: torch.Tensor,    # [B, N, D_dino]
    threshold: float = 0.0,
) -> torch.Tensor:
    """Per-patch DINO feature distance → normalised change target [B, N] ∈ [0,1].

    Preferred over pixel-diff because DINO captures semantic changes.
    """
    diff = (current_raw_dino - future_raw_dino).norm(dim=-1, keepdim=False)  # [B, N]
    return _normalise_target(diff, threshold)


def compute_pixel_change_target(
    current_pixels: torch.Tensor,   # [B, 3, H, W] float [0,1]
    future_pixels: torch.Tensor,    # [B, 3, H, W] float [0,1]
    patch_hw: int,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Per-patch pixel L1 difference → normalised change target [B, N] ∈ [0,1]."""
    diff = (current_pixels - future_pixels).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
    pooled = F.adaptive_avg_pool2d(diff, (patch_hw, patch_hw))                # [B,1,ph,ph]
    pooled = pooled.squeeze(1).reshape(pooled.shape[0], -1)                   # [B, N]
    return _normalise_target(pooled, threshold)


def _normalise_target(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Normalise [B, N] values to [0, 1], optionally binarise."""
    if threshold > 0.0:
        return (x > threshold).float()
    vmin = x.amin(dim=1, keepdim=True)
    vmax = x.amax(dim=1, keepdim=True)
    return (x - vmin) / (vmax - vmin + 1e-8)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ActionConditionedFocusedResidualWM(nn.Module):
    """DINO-based action-conditioned world model — terminal future image prediction.

    Inputs
    ------
    current_pixels : [B, 3, H, W] float [0,1] OR [B, H, W, 3] uint8
    actions        : [B, H_steps, action_dim] float
    future_pixels  : (optional) same format as current_pixels

    Returns (dict)
    --------------
    loss                    : scalar  (0.0 when future_pixels is None)
    pred_future_image       : [B, 3, H, W] ∈ [0, 1]
    focus_map               : [B, N_patches] ∈ [0, 1]
    current_dino_features   : [B, N, D_dino]  — raw DINO features of current frame
    predicted_future_tokens : [B, N, hidden_dim] — current + focus * delta  (strict masked residual)
    predicted_delta_tokens  : [B, N, hidden_dim] — raw delta from ResidualFuturePredictor
    delta_norm              : scalar — mean patch-token L2 norm of delta
    fg_delta_norm           : scalar — focus-weighted mean delta norm (foreground)
    bg_delta_norm           : scalar — (1-focus)-weighted mean delta norm (background)
    predicted_residual_image: [B, 3, H, W] — raw decoder output (added to current_image)
    residual_l1_mean        : scalar — mean absolute residual pixel value
    residual_abs_max        : scalar — max absolute residual pixel value
    dino_change_target      : [B, N] ∈ [0,1] or None  — DINO-based change supervision target
    loss_components         : dict of individual (weighted) scalar losses
    """

    def __init__(
        self,
        cfg: FocusedWMConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Visual encoder — DINO (always present, non-optional)
        self.dino = DinoFeatureExtractor(cfg)

        # Trainable sub-modules
        self.action_encoder = ActionChunkEncoder(cfg)
        self.focus_head     = FocusHead(cfg) if cfg.use_focus_head else None
        self.predictor      = ResidualFuturePredictor(cfg)
        self.decoder        = FutureImageDecoder(cfg)

        # Shallow skip encoder for current_image → decoder skip features.
        # Kept in float32 so it directly accepts current_f [float32, 0-1].
        # The decoder casts skip features to its own dtype at fusion time.
        self.image_skip_encoder: Optional[nn.Module] = (
            CurrentImageSkipEncoder(cfg) if cfg.use_decoder_skip_connection else None
        )

        # LPIPS (optional — lazy-loaded on first use)
        self._lpips_fn: Optional[nn.Module] = None

        if torch_dtype is not None:
            trainable = [self.action_encoder, self.predictor, self.decoder]
            if self.focus_head is not None:
                trainable.append(self.focus_head)
            # Only project layer in DINO is trainable
            trainable.append(self.dino.proj)
            # image_skip_encoder intentionally excluded — stays float32
            for mod in trainable:
                mod.to(dtype=torch_dtype)

        # Action normalisation ranges (optional)
        if cfg.action_ranges_path:
            self.register_buffer(
                "action_ranges",
                torch.load(cfg.action_ranges_path, map_location="cpu"),
                persistent=False,
            )
        else:
            self.action_ranges = None

    # ------------------------------------------------------------------ Compat

    def gradient_checkpointing_enable(self, *args, **kwargs):
        pass

    def enable_input_require_grads(self):
        for p in self.parameters():
            if p.requires_grad:
                p.requires_grad_(True)

    # ------------------------------------------------------------------ Helpers

    def _prep_image(self, pixels: torch.Tensor) -> torch.Tensor:
        """Convert [B,H,W,C] uint8 or [B,C,H,W] float → [B,3,image_height,image_width] float [0,1]."""
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0
            pixels = pixels.permute(0, 3, 1, 2).contiguous()
        H, W = self.cfg.image_height, self.cfg.image_width
        if pixels.shape[-2] != H or pixels.shape[-1] != W:
            pixels = F.interpolate(pixels, size=(H, W), mode="bilinear", align_corners=False)
        return pixels

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_ranges is None:
            return actions
        ar  = self.action_ranges.to(actions.device)
        mn, mx = ar[:, 0], ar[:, 1]
        return torch.clamp((actions - mn) / (mx - mn + 1e-8), 0.0, 1.0)

    def _get_lpips(self, device: torch.device) -> nn.Module:
        if self._lpips_fn is None:
            try:
                import lpips
                self._lpips_fn = lpips.LPIPS(net="vgg").to(device)
                self._lpips_fn.requires_grad_(False)
                self._lpips_fn.eval()
            except ImportError as exc:
                raise ImportError(
                    "lpips package required for LPIPS loss: pip install lpips"
                ) from exc
        return self._lpips_fn

    # ------------------------------------------------------------------ Losses

    def _recon_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target.to(pred.dtype))

    def _lpips_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fn = self._get_lpips(pred.device)
        # LPIPS expects [-1, 1]
        p = pred.float() * 2.0 - 1.0
        t = target.float() * 2.0 - 1.0
        return fn(p, t).mean()

    def _dino_feature_loss(
        self,
        predicted_tokens: torch.Tensor,   # [B, N, hidden_dim]  predicted future
        gt_raw_dino: torch.Tensor,         # [B, N, D_dino]       from DINO(gt_future)
    ):
        # Project GT DINO features to hidden_dim using the same shared proj
        gt_projected = self.dino.proj(gt_raw_dino.to(predicted_tokens.dtype))  # [B,N,D_h]
        return F.mse_loss(predicted_tokens, gt_projected.detach()), gt_projected.detach()

    def _focus_supervision_loss(
        self,
        focus_logits: torch.Tensor,    # [B, N] logits
        change_target: torch.Tensor,   # [B, N] in [0, 1]
    ) -> torch.Tensor:
        target = change_target.to(device=focus_logits.device, dtype=focus_logits.dtype)
        return F.binary_cross_entropy_with_logits(focus_logits, target)

    def _focus_sparsity_loss(self, focus_map: torch.Tensor) -> torch.Tensor:
        if self.cfg.focus_sparsity_mode == "entropy":
            p = focus_map.clamp(1e-6, 1 - 1e-6)
            return -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        return focus_map.mean()  # L1 (mean ≈ L1 for sigmoid outputs)

    def _fg_grad_loss(
        self,
        pred: torch.Tensor,    # [B, 3, H, W]
        target: torch.Tensor,  # [B, 3, H, W]
        fg_mask: torch.Tensor, # [B, 1, H, W]  image-space weight
    ) -> torch.Tensor:
        """Foreground gradient consistency — match spatial gradients inside write_mask."""
        def _fd(x):
            return x[:, :, :, 1:] - x[:, :, :, :-1], x[:, :, 1:, :] - x[:, :, :-1, :]
        p_f = pred.to(target.dtype)
        pgx, pgy = _fd(p_f)
        tgx, tgy = _fd(target.to(pred.dtype))
        mfx = fg_mask[:, :, :, 1:].to(pred.dtype)
        mfy = fg_mask[:, :, 1:, :].to(pred.dtype)
        return (mfx * (pgx - tgx).abs()).mean() + (mfy * (pgy - tgy).abs()).mean()

    # ------------------------------------------------------------------ Forward

    def forward(
        self,
        current_pixels: torch.Tensor,
        actions: torch.Tensor,
        future_pixels: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> Dict[str, torch.Tensor]:

        current_f  = self._prep_image(current_pixels)
        actions_n  = self._normalize_actions(actions.float())

        # Cast actions to trainable module dtype.  During training this is handled
        # automatically by torch.autocast; outside autocast (ranking eval, metrics)
        # we must cast explicitly so bfloat16 linear layers receive the right dtype.
        _mdtype = self.action_encoder.input_proj.weight.dtype
        if actions_n.dtype != _mdtype:
            actions_n = actions_n.to(_mdtype)

        # ----- Encode current frame with DINO + project ------------------
        # forward_with_raw avoids a double backbone pass when raw is needed.
        current_tokens, current_raw = self.dino.forward_with_raw(current_f)
        # current_tokens: [B, N, hidden_dim]  (projected, grad enabled for proj)
        # current_raw:    [B, N, D_dino]      (detached, for loss targets)

        # ----- Action encoding -------------------------------------------
        H_act = min(actions_n.shape[1], self.cfg.action_horizon)
        action_emb = self.action_encoder(actions_n[:, :H_act, :])  # [B, Da]

        # ----- Focus masks (Phase 4: dual context / write mask) ----------
        focus_logits = None
        if self.focus_head is not None:
            focus_logits = self.focus_head(current_tokens, action_emb)   # [B, N] logits
            context_mask = torch.sigmoid(focus_logits)                   # [B, N] broad — predictor input
            if self.cfg.use_dual_focus_mask:
                write_mask = torch.sigmoid(
                    focus_logits / self.cfg.write_mask_temperature
                )                                                         # [B, N] sharp — update gate
            else:
                write_mask = context_mask
        else:
            ones = torch.ones(
                current_tokens.shape[0], current_tokens.shape[1],
                device=current_tokens.device, dtype=current_tokens.dtype,
            )
            context_mask = ones
            write_mask   = ones

        focus_map = context_mask   # legacy alias used for losses / diagnostics

        # ----- Strict masked residual prediction -------------------------
        delta_tokens  = self.predictor(current_tokens, action_emb, context_mask)
        future_tokens = current_tokens + write_mask.unsqueeze(-1) * delta_tokens

        # delta / token-shift statistics (diagnostic, no gradient flow)
        with torch.no_grad():
            _d      = delta_tokens.detach().float().norm(dim=-1)  # [B, N]
            _ctx    = context_mask.detach().float()               # [B, N] context (broad)
            _wrt    = write_mask.detach().float()                 # [B, N] write   (sharp)
            _bg_ctx = 1.0 - _ctx
            _bg_wrt = 1.0 - _wrt
            # context-mask weighted (predictor input side)
            delta_norm            = _d.mean()
            fg_delta_norm         = (_ctx * _d).sum() / (_ctx.sum() + 1e-8)
            bg_delta_norm         = (_bg_ctx * _d).sum() / (_bg_ctx.sum() + 1e-8)
            bg_fg_delta_ratio     = bg_delta_norm / (fg_delta_norm + 1e-6)
            # write-mask weighted (actual update side)
            fg_delta_norm_write     = (_wrt * _d).sum() / (_wrt.sum() + 1e-8)
            bg_delta_norm_write     = (_bg_wrt * _d).sum() / (_bg_wrt.sum() + 1e-8)
            bg_fg_delta_ratio_write = bg_delta_norm_write / (fg_delta_norm_write + 1e-6)
            # token shift = write * delta  (future - current = write * delta)
            _shift = (_wrt.unsqueeze(-1) * delta_tokens.detach().float()).norm(dim=-1)  # [B,N]
            future_token_shift_l2    = _shift.mean()
            fg_future_token_shift_l2 = (_wrt * _shift).sum() / (_wrt.sum() + 1e-8)
            bg_future_token_shift_l2 = (_bg_wrt * _shift).sum() / (_bg_wrt.sum() + 1e-8)
            # mask statistics
            write_mask_mean       = _wrt.mean()
            context_mask_mean     = _ctx.mean()
            write_context_gap     = (_wrt - _ctx).abs().mean()
            context_write_overlap = (_ctx.clamp(0, 1) * _wrt.clamp(0, 1)).mean()
            write_context_ratio   = write_mask_mean / (context_mask_mean + 1e-6)

        # ----- Decode residual image and compose final prediction ----------
        skip_features = (
            self.image_skip_encoder(current_f)
            if self.image_skip_encoder is not None else None
        )
        pred_residual_image = self.decoder(future_tokens, skip_features=skip_features)

        # Upsample write_mask to image space — used for Phase 4 composition and stats
        B_f, _, H_f, W_f = current_f.shape
        _ph = self.cfg.patch_hw
        write_mask_img = F.interpolate(
            write_mask.reshape(B_f, 1, _ph, _ph).to(current_f.dtype),
            size=(H_f, W_f), mode="bilinear", align_corners=False,
        )  # [B, 1, H, W]

        if self.cfg.use_dual_focus_mask:
            # Phase 4: gate residual with write_mask_img (sharp mask)
            pred_future_image = torch.clamp(
                current_f + write_mask_img * pred_residual_image.to(current_f.dtype), 0.0, 1.0,
            )
        elif self.cfg.use_focus_gated_image_residual:
            # Legacy: gate with context_mask upsampled to image space
            ctx_img = F.interpolate(
                context_mask.reshape(B_f, 1, _ph, _ph).to(current_f.dtype),
                size=(H_f, W_f), mode="bilinear", align_corners=False,
            )
            pred_future_image = torch.clamp(
                current_f + ctx_img * pred_residual_image.to(current_f.dtype), 0.0, 1.0,
            )
        else:
            pred_future_image = torch.clamp(
                current_f + pred_residual_image.to(current_f.dtype), 0.0, 1.0,
            )

        # residual statistics (diagnostic, no gradient flow)
        with torch.no_grad():
            _r  = pred_residual_image.detach().float()  # [B,3,H,W]
            _rabs = _r.abs()
            residual_l1_mean     = _rabs.mean()
            residual_l2_mean     = _r.pow(2).mean().sqrt()
            residual_abs_max     = _rabs.amax()
            residual_signed_mean = _r.mean()
            # Context-mask fg/bg residual decomposition
            _ctx_img = F.interpolate(
                context_mask.detach().float().reshape(B_f, 1, _ph, _ph),
                size=(H_f, W_f), mode="bilinear", align_corners=False,
            )  # [B,1,H,W]
            fg_residual_l1       = (_ctx_img * _rabs).sum() / (_ctx_img.sum() * 3 + 1e-8)
            bg_residual_l1       = ((1 - _ctx_img) * _rabs).sum() / (
                (1 - _ctx_img).sum() * 3 + 1e-8)
            bg_fg_residual_ratio = bg_residual_l1 / (fg_residual_l1 + 1e-6)
            # Write-mask fg/bg residual decomposition
            _wrt_img = write_mask_img.detach().float()
            fg_residual_l1_write       = (_wrt_img * _rabs).sum() / (_wrt_img.sum() * 3 + 1e-8)
            bg_residual_l1_write       = ((1 - _wrt_img) * _rabs).sum() / (
                (1 - _wrt_img).sum() * 3 + 1e-8)
            bg_fg_residual_ratio_write = bg_residual_l1_write / (fg_residual_l1_write + 1e-6)

        # ----- Losses (only when GT future is provided) ------------------
        lc: Dict[str, torch.Tensor] = {}
        total = torch.zeros(1, device=current_tokens.device, dtype=current_tokens.dtype)
        dino_change_target  = None
        gt_raw_dino         = None
        gt_projected_dino   = None
        future_f: Optional[torch.Tensor] = None   # set when future_pixels is not None

        if future_pixels is not None:
            future_f = self._prep_image(future_pixels)

            # --- GT DINO features (no_grad, used as targets) ----
            gt_raw_dino = self.dino.extract_raw(future_f)  # [B, N, D_dino]

            # 1. Reconstruction
            recon = self._recon_loss(pred_future_image, future_f)
            lc["recon"] = recon
            total = total + self.cfg.recon_loss_weight * recon

            # 2. LPIPS
            if self.cfg.use_lpips_loss:
                lp = self._lpips_loss(pred_future_image, future_f)
                lc["lpips"] = lp
                total = total + self.cfg.lpips_loss_weight * lp

            # 3. DINO feature consistency
            if self.cfg.use_dino_feature_loss:
                feat, gt_projected_dino = self._dino_feature_loss(future_tokens, gt_raw_dino)
                lc["dino_feature"] = feat
                total = total + self.cfg.dino_feature_loss_weight * feat

            # 4. Focus supervision
            if self.focus_head is not None and (
                self.cfg.use_dino_focus_supervision
                or self.cfg.use_pixel_focus_supervision
            ):
                if self.cfg.use_dino_focus_supervision:
                    dino_change_target = compute_dino_change_target(
                        current_raw, gt_raw_dino,
                        threshold=self.cfg.change_target_threshold,
                    )
                    change_target = dino_change_target
                else:  # pixel fallback (ablation)
                    change_target = compute_pixel_change_target(
                        current_f, future_f,
                        patch_hw=self.cfg.patch_hw,
                        threshold=self.cfg.change_target_threshold,
                    )

                focus_sup = self._focus_supervision_loss(focus_logits, change_target)
                lc["focus_supervision"] = focus_sup
                total = total + self.cfg.focus_supervision_weight * focus_sup

            # 5. Focus sparsity
            if self.focus_head is not None and self.cfg.use_focus_sparsity:
                focus_sp = self._focus_sparsity_loss(focus_map)
                lc["focus_sparsity"] = focus_sp
                total = total + self.cfg.focus_sparsity_weight * focus_sp

            # 6. Foreground reconstruction loss (gated by write_mask or change target)
            if self.cfg.use_fg_recon_loss:
                _fg_src = (
                    dino_change_target if dino_change_target is not None else write_mask
                )
                _fg_mask = F.interpolate(
                    _fg_src.detach().float().reshape(
                        future_f.shape[0], 1, self.cfg.patch_hw, self.cfg.patch_hw
                    ),
                    size=(future_f.shape[-2], future_f.shape[-1]),
                    mode="bilinear", align_corners=False,
                ).to(pred_future_image.dtype)
                fg_recon = (
                    _fg_mask * F.smooth_l1_loss(pred_future_image, future_f.to(pred_future_image.dtype), reduction="none")
                ).mean()
                lc["fg_recon"] = fg_recon
                total = total + self.cfg.fg_recon_weight * fg_recon

            # 7. Background residual penalty (suppress residual outside write_mask)
            if self.cfg.use_bg_residual_penalty:
                bg_mask_img = (1.0 - write_mask_img).detach()
                bg_pen = (
                    bg_mask_img * pred_residual_image.to(bg_mask_img.dtype).abs()
                ).mean()
                lc["bg_residual_penalty"] = bg_pen
                total = total + self.cfg.bg_residual_weight * bg_pen

            # 8. Foreground gradient consistency
            if self.cfg.use_fg_grad_loss:
                grad_loss = self._fg_grad_loss(pred_future_image, future_f, write_mask_img)
                lc["fg_grad"] = grad_loss
                total = total + self.cfg.fg_grad_weight * grad_loss

        lc["total"] = total

        return {
            # ---- Core outputs ------------------------------------------------
            "loss": total,
            "pred_future_image": pred_future_image,                # [B,3,H,W] ∈ [0,1]
            "predicted_residual_image": pred_residual_image,       # [B,3,H,W] raw residual
            # ---- Auxiliary pixel tensors (for trainer-side diagnostics) ------
            "current_pixels_float": current_f,                     # [B,3,H,W] float32 [0,1]
            "future_pixels_float": future_f,                       # [B,3,H,W] or None
            # ---- Focus masks (Phase 4) ----------------------------------------
            "context_mask": context_mask,                          # [B, N] broad (predictor)
            "write_mask": write_mask,                              # [B, N] sharp (update gate)
            "write_mask_img": write_mask_img,                      # [B, 1, H, W]
            "focus_map": focus_map,                                # [B, N] alias = context_mask
            # ---- Mask statistics ---------------------------------------------
            "write_mask_mean": write_mask_mean,
            "context_mask_mean": context_mask_mean,
            "write_context_gap": write_context_gap,
            "context_write_overlap": context_write_overlap,
            "write_context_ratio": write_context_ratio,
            # ---- Other tokens ------------------------------------------------
            "current_dino_features": current_raw,                  # [B, N, D_dino]
            "predicted_future_tokens": future_tokens,               # [B, N, hidden_dim]
            "predicted_delta_tokens": delta_tokens,                 # [B, N, hidden_dim]
            "gt_projected_dino": gt_projected_dino,                 # [B, N, hidden_dim] or None
            "dino_change_target": dino_change_target,               # [B, N] or None
            # ---- Token-update statistics — context-mask side (A) -------------
            "delta_norm": delta_norm,
            "fg_delta_norm": fg_delta_norm,
            "bg_delta_norm": bg_delta_norm,
            "bg_fg_delta_ratio": bg_fg_delta_ratio,
            # ---- Token-update statistics — write-mask side (A) ---------------
            "fg_delta_norm_write": fg_delta_norm_write,
            "bg_delta_norm_write": bg_delta_norm_write,
            "bg_fg_delta_ratio_write": bg_fg_delta_ratio_write,
            "future_token_shift_l2": future_token_shift_l2,
            "fg_future_token_shift_l2": fg_future_token_shift_l2,
            "bg_future_token_shift_l2": bg_future_token_shift_l2,
            # ---- Image-residual statistics — context-mask side (B) -----------
            "residual_l1_mean": residual_l1_mean,
            "residual_l2_mean": residual_l2_mean,
            "residual_abs_max": residual_abs_max,
            "residual_signed_mean": residual_signed_mean,
            "fg_residual_l1": fg_residual_l1,
            "bg_residual_l1": bg_residual_l1,
            "bg_fg_residual_ratio": bg_fg_residual_ratio,
            # ---- Image-residual statistics — write-mask side (B) -------------
            "fg_residual_l1_write": fg_residual_l1_write,
            "bg_residual_l1_write": bg_residual_l1_write,
            "bg_fg_residual_ratio_write": bg_fg_residual_ratio_write,
            # ---- Loss breakdown ----------------------------------------------
            "loss_components": lc,
        }

    # ------------------------------------------------------------------ RFT hook

    @torch.no_grad()
    def get_latent_features_for_rft(
        self,
        current_pixels: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """TODO(RFT): Exposes latent features for policy reward computation.

        Returns focus_map, predicted future tokens, and pred future image.
        Connect from the RFT policy trainer to use these as shaped rewards.
        """
        out = self.forward(current_pixels, actions, future_pixels=None)
        return {
            "focus_map": out["focus_map"],
            "predicted_future_tokens": out["predicted_future_tokens"],
            "pred_future_image": out["pred_future_image"],
            "current_dino_features": out["current_dino_features"],
        }

    # ------------------------------------------------------------------ Ranking

    @torch.no_grad()
    def rank_action_candidates(
        self,
        current_pixels: torch.Tensor,    # [B, 3, H, W] or [B, H, W, 3]
        action_candidates: torch.Tensor, # [B, K, H_steps, action_dim]
        future_pixels: torch.Tensor,     # [B, 3, H, W] or [B, H, W, 3]
    ) -> torch.Tensor:                   # [B, K] scores (higher = better)
        """Score K action candidates per sample using the configured ranking score.

        Scoring modes (cfg.ranking_score_type):
          "dino_only"  — cosine similarity of predicted vs GT future DINO features
          "image_only" — negative SmoothL1 of predicted vs GT future pixels
          "combined"   — weighted sum of both
        """
        B, K, H_steps, _ = action_candidates.shape
        current_f = self._prep_image(current_pixels)
        future_f  = self._prep_image(future_pixels)

        # GT future DINO features (shared for all candidates)
        gt_raw = self.dino.extract_raw(future_f)  # [B, N, D_dino]
        gt_proj = self.dino.proj(gt_raw.to(self.dino.proj.weight.dtype))  # [B, N, D_h]

        # Expand current frame across K candidates
        cur_exp = current_f.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(
            B * K, *current_f.shape[1:]
        )
        act_flat = action_candidates.reshape(B * K, H_steps, -1)

        out = self.forward(cur_exp, act_flat)
        pred_img    = out["pred_future_image"]      # [B*K, 3, H, W]
        pred_tokens = out["predicted_future_tokens"]  # [B*K, N, D_h]

        # GT reference (expanded for K candidates)
        gt_proj_exp = gt_proj.unsqueeze(1).expand(-1, K, -1, -1).reshape(
            B * K, *gt_proj.shape[1:]
        )  # [B*K, N, D_h]
        gt_img_exp = future_f.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(
            B * K, *future_f.shape[1:]
        )  # [B*K, 3, H, W]

        scores_flat: Optional[torch.Tensor] = None

        if self.cfg.ranking_score_type in ("dino_only", "combined"):
            # Cosine similarity over (N×D_h) flattened feature space
            a = pred_tokens.reshape(B * K, -1)
            b = gt_proj_exp.reshape(B * K, -1)
            dino_score = F.cosine_similarity(a, b, dim=-1)  # [B*K]
            scores_flat = dino_score

        if self.cfg.ranking_score_type in ("image_only", "combined"):
            img_loss = F.smooth_l1_loss(
                pred_img, gt_img_exp.to(pred_img.dtype), reduction="none"
            ).mean(dim=[1, 2, 3])  # [B*K]
            img_score = -img_loss

            if self.cfg.ranking_score_type == "combined":
                w = self.cfg.ranking_image_weight
                scores_flat = (1 - w) * scores_flat + w * img_score
            else:
                scores_flat = img_score

        return scores_flat.reshape(B, K)  # [B, K] — higher = better
