"""Pixel-Residual World Model.

Predicts future video frames directly in pixel space, conditioned on actions.
Three target modes are supported:

  "pixel"
      Full future-image prediction.  The decoder outputs a full image [0,1].
      L = MSE(pred_future, gt_future)

  "pixel_residual"
      Residual prediction.  The decoder outputs the pixel-level change
      (future - current), and the predicted future is reconstructed as:
          pred_future = clamp(current + residual_pred, 0, 1)
      L = λ_residual * MSE(residual_pred, residual_target)
        + λ_image    * MSE(pred_future, gt_future)

  "pixel_residual_roi_dynamic"
      Residual prediction with additional region-specific losses:
          L_dynamic  — MSE restricted to the dynamic mask
          L_gripper  — MSE restricted to the gripper ROI crop
          L_static   — MSE between pred_future and current_image outside the
                       dynamic mask (penalise spurious motion in static areas)

Architecture overview
---------------------
  PixelEncoder      CNN: [B, 3, 256, 256] → [B, N=64, C]     (5× stride-2 conv)
  ActionEncoder     MLP: [B, H, 7]        → [B, H, action_emb_dim]
  PixelPredictor    Causal transformer:
                       anchor(curr_tokens) + action_toks → pred_global [B, H, C]
                       broadcast: pred_tokens[h] = curr_tokens + pred_global[h]
  PixelDecoder      CNN: [B*H, N=64, C]   → [B*H, 3, 256, 256]  (5× 2× upsample)

The architecture is fully differentiable in pixel space; no external tokenizer
or DINO backbone is required.

TODO(latent_residual): Add "latent_residual" mode that routes through a frozen
  CompressiveVQModelFSQ encoder for feature extraction and predicts residuals
  in the FSQ embedding space.  Hook: PixelResidualWorldModel._forward_latent().
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pixel_residual_config import PixelResidualConfig
from .pixel_residual_utils import (
    compute_dynamic_mask,
    motion_center_of_mass,
    extract_roi_crops,
    masked_mse_loss,
)


# ===========================================================================
# Encoder — CNN  [B, 3, H, W] → [B, N=64, C]
# ===========================================================================

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.GELU(),
    )


class PixelEncoder(nn.Module):
    """5× stride-2 CNN: [B, 3, 256, 256] → [B, C, 8, 8] → [B, N=64, C]."""

    def __init__(self, cfg: PixelResidualConfig) -> None:
        super().__init__()
        C = cfg.encoder_channels
        self.net = nn.Sequential(
            _conv_block(3,     C // 8),   # [B, C/8, 128, 128]
            _conv_block(C // 8, C // 4),  # [B, C/4,  64,  64]
            _conv_block(C // 4, C // 2),  # [B, C/2,  32,  32]
            _conv_block(C // 2, C),        # [B, C,    16,  16]
            _conv_block(C,      C),        # [B, C,     8,   8]
        )
        self.out_channels = C
        # After 5 doublings: 256 / 32 = 8 → n_spatial_tokens = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] → [B, N=64, C]."""
        feat = self.net(x)           # [B, C, 8, 8]
        B, C, h, w = feat.shape
        return feat.reshape(B, C, h * w).permute(0, 2, 1)  # [B, 64, C]


# ===========================================================================
# Decoder — CNN  [B, N=64, C] → [B, 3, H, W]
# ===========================================================================

def _deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.GELU(),
    )


class PixelDecoder(nn.Module):
    """5× 2× upsample CNN: [B, N=64, C] → [B, 3, 256, 256].

    Output is raw (no sigmoid); the model applies clamping in its forward pass.
    """

    def __init__(self, cfg: PixelResidualConfig) -> None:
        super().__init__()
        C = cfg.encoder_channels
        self.net = nn.Sequential(
            _deconv_block(C,      C),        # [B, C,    16,  16]
            _deconv_block(C,      C // 2),   # [B, C/2,  32,  32]
            _deconv_block(C // 2, C // 4),   # [B, C/4,  64,  64]
            _deconv_block(C // 4, C // 8),   # [B, C/8, 128, 128]
            nn.ConvTranspose2d(C // 8, 3, kernel_size=4, stride=2, padding=1),  # [B, 3, 256, 256]
        )
        self.spatial_size = 8  # sqrt(64)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, N=64, C] → [B, 3, 256, 256] (raw, no activation)."""
        B, N, C_in = tokens.shape
        h = w = self.spatial_size
        x = tokens.permute(0, 2, 1).reshape(B, C_in, h, w)  # [B, C, 8, 8]
        return self.net(x)                                    # [B, 3, 256, 256]


# ===========================================================================
# Action encoder — MLP
# ===========================================================================

class ActionEncoder(nn.Module):
    """MLP: [B, H, action_dim] → [B, H, action_emb_dim]."""

    def __init__(self, cfg: PixelResidualConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.action_emb_dim),
            nn.GELU(),
            nn.Linear(cfg.action_emb_dim, cfg.action_emb_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions)


# ===========================================================================
# Pixel predictor — causal transformer
# ===========================================================================

class PixelResidualPredictor(nn.Module):
    """Causal transformer: curr_tokens + actions → per-step global delta.

    Sequence layout (length = H+1):
        token_0    = anchor(pool(curr_tokens))   ← static context
        token_h    = act_proj(action_{h-1})      ← future action  h=1..H

    Output:
        pred_global [B, H, D] — per-step global latent delta.
        Broadcast to spatial: pred_tokens[h] = curr_tokens + pred_global[h].
    """

    def __init__(self, cfg: PixelResidualConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim

        # Project pooled encoder output → hidden_dim anchor token
        self.curr_proj = nn.Linear(cfg.encoder_channels, D)

        # Project per-step action embedding → hidden_dim
        self.act_proj = nn.Linear(cfg.action_emb_dim, D)

        # Causal transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_pred_layers)

        # Output projection back to encoder_channels (spatial token delta)
        self.out_proj = nn.Linear(D, cfg.encoder_channels)

    def forward(
        self,
        curr_tokens: torch.Tensor,   # [B, N, C_enc]  N=64 spatial tokens
        act_emb: torch.Tensor,       # [B, H, action_emb_dim]
    ) -> torch.Tensor:
        """Returns pred_global: [B, H, C_enc]."""
        B, N, C_enc = curr_tokens.shape
        H = act_emb.shape[1]

        # Anchor: pool spatial tokens → [B, C_enc] → project → [B, 1, D]
        anchor = self.curr_proj(curr_tokens.mean(dim=1)).unsqueeze(1)   # [B, 1, D]

        # Action tokens: [B, H, action_emb_dim] → [B, H, D]
        act_toks = self.act_proj(act_emb)   # [B, H, D]

        # Concat: [B, H+1, D]
        seq = torch.cat([anchor, act_toks], dim=1)

        # Causal attention mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            H + 1, device=seq.device, dtype=seq.dtype
        )
        out = self.transformer(seq, mask=causal_mask, is_causal=True)  # [B, H+1, D]

        # Predictions from positions 1..H (skip anchor)
        pred_delta = self.out_proj(out[:, 1:, :])  # [B, H, C_enc]
        return pred_delta


# ===========================================================================
# Main model
# ===========================================================================

class PixelResidualWorldModel(nn.Module):
    """CNN-based Pixel-Residual World Model.

    Fully differentiable in pixel space; no external tokenizer or DINO required.

    Forward expects:
        pixels  [B, T+1, H, W, C] uint8  (channel-last, as from RLDS dataset)
        actions [B, T,   action_dim]     float32

    Frame convention (matches LatentResidualWorldModel current_anchor_ctx):
        frame_0          — context window start (not used here)
        frame_1          — current frame (anchor)
        frame_2 .. T     — future frames (targets)
        H = T - 1        — number of future steps predicted

    Normalization: images are converted to [0, 1] float internally.
    """

    def __init__(
        self,
        cfg: PixelResidualConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder    = PixelEncoder(cfg)
        self.act_encoder = ActionEncoder(cfg)
        self.predictor  = PixelResidualPredictor(cfg)
        self.decoder    = PixelDecoder(cfg)

        if torch_dtype is not None:
            for m in [self.encoder, self.act_encoder, self.predictor, self.decoder]:
                m = m.to(dtype=torch_dtype)

        # Action normalization ranges (loaded once onto CPU)
        try:
            ranges = torch.load(cfg.action_ranges_path, map_location="cpu")
            self.register_buffer("action_ranges", ranges, persistent=False)
        except Exception:
            # Fallback: unit range; normalization becomes identity
            self.register_buffer(
                "action_ranges",
                torch.stack([torch.zeros(cfg.action_dim), torch.ones(cfg.action_dim)], dim=1),
                persistent=False,
            )

    # ------------------------------------------------------------------
    # HuggingFace Trainer compatibility
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, *args, **kwargs):
        pass

    def enable_input_require_grads(self):
        for m in [self.encoder, self.act_encoder, self.predictor, self.decoder]:
            m.requires_grad_(True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        ar = self.action_ranges.to(actions.device)   # [action_dim, 2]
        mn, mx = ar[:, 0], ar[:, 1]
        return torch.clamp((actions - mn) / (mx - mn + 1e-8), 0.0, 1.0)

    def _autocast_ctx(self, device):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            self.cfg.autocast_dtype, torch.float32
        )
        # device may be a torch.device or string; use .type when available
        device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
        if device_type == "cpu":
            return torch.autocast(device_type="cpu", dtype=torch.float32)
        return torch.autocast(device_type=device_type, dtype=dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixels: torch.Tensor,    # [B, T+1, H_img, W_img, C] uint8  channel-last
        actions: torch.Tensor,   # [B, T, action_dim]
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Dispatch to the appropriate forward based on cfg.target_mode."""
        device = next(self.encoder.parameters()).device
        pixels  = pixels.to(device=device, non_blocking=True)
        actions = actions.to(device=device, non_blocking=True)

        # uint8 channel-last → float channel-first [0, 1]
        # pixels: [B, T+1, H, W, C] uint8 → [B, T+1, C, H, W] float [0,1]
        pixels_f = pixels.permute(0, 1, 4, 2, 3).contiguous().float() / 255.0

        B, T_plus_1 = pixels_f.shape[:2]
        T = T_plus_1 - 1
        H = T - 1   # number of future steps

        if H < 1:
            raise ValueError(
                f"Need segment_length >= 3 to have ≥1 future step, got T={T}."
            )

        # current: frame_1;  future_gt: frames 2..T
        current   = pixels_f[:, 1]          # [B, 3, H_img, W_img]
        future_gt = pixels_f[:, 2:]         # [B, H, 3, H_img, W_img]

        # Actions: skip action_0 (same offset as LatentResidualWorldModel CA mode)
        acts = self._normalize_actions(actions[:, 1:1 + H, :])  # [B, H, action_dim]

        mode = self.cfg.target_mode
        if mode == "pixel":
            return self._forward_pixel(current, future_gt, acts, B, H, device)
        elif mode == "pixel_residual":
            return self._forward_pixel_residual(current, future_gt, acts, B, H, device)
        elif mode == "pixel_residual_roi_dynamic":
            return self._forward_pixel_residual_roi_dynamic(
                current, future_gt, acts, B, H, device
            )
        else:
            raise ValueError(f"Unknown target_mode: {mode!r}")

    # ------------------------------------------------------------------
    # Forward — pixel  (full-image prediction, no residual)
    # ------------------------------------------------------------------

    def _encode_decode(
        self,
        current: torch.Tensor,   # [B, 3, H, W]
        acts: torch.Tensor,      # [B, H, action_dim]
        B: int,
        H: int,
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared encode → predict → decode step.

        Returns:
            pred_tokens  [B, H, N, C_enc]   — predicted spatial tokens per step
            pred_decoded [B, H, 3, H_img, W_img] — decoded images (raw, no clamp)
        """
        dtype = next(self.encoder.parameters()).dtype

        with self._autocast_ctx(device):
            curr_tokens = self.encoder(current.to(dtype))       # [B, N, C]
            act_emb     = self.act_encoder(acts.to(dtype))      # [B, H, emb_dim]
            pred_global = self.predictor(curr_tokens, act_emb)  # [B, H, C]

        # Broadcast global delta to all spatial tokens (residual in token space)
        # curr_tokens: [B, N, C]; pred_global: [B, H, C] → [B, H, 1, C]
        pred_tokens = (
            curr_tokens.unsqueeze(1) + pred_global.unsqueeze(2)
        )  # [B, H, N, C]

        # Decode each future-step token map to an image
        N, C_enc = pred_tokens.shape[2], pred_tokens.shape[3]
        pred_flat = pred_tokens.reshape(B * H, N, C_enc)        # [B*H, N, C]
        with self._autocast_ctx(device):
            decoded_flat = self.decoder(pred_flat.to(dtype))    # [B*H, 3, H_img, W_img]

        H_img, W_img = decoded_flat.shape[-2:]
        pred_decoded = decoded_flat.reshape(B, H, 3, H_img, W_img).float()

        return pred_tokens, pred_decoded

    def _forward_pixel(
        self,
        current: torch.Tensor,    # [B, 3, H, W]
        future_gt: torch.Tensor,  # [B, H, 3, H, W]
        acts: torch.Tensor,
        B: int,
        H: int,
        device,
    ) -> Dict:
        _, pred_decoded = self._encode_decode(current, acts, B, H, device)
        pred_future = pred_decoded.clamp(0.0, 1.0)
        loss = F.mse_loss(pred_future, future_gt.detach().float())
        return {"loss": loss}

    # ------------------------------------------------------------------
    # Forward — pixel_residual
    # ------------------------------------------------------------------

    def _forward_pixel_residual(
        self,
        current: torch.Tensor,
        future_gt: torch.Tensor,
        acts: torch.Tensor,
        B: int,
        H: int,
        device,
    ) -> Dict:
        _, pred_decoded = self._encode_decode(current, acts, B, H, device)

        # pred_decoded treated as residual estimate
        residual_pred   = pred_decoded                                   # [B, H, 3, H, W]
        residual_target = (future_gt - current.unsqueeze(1)).float()     # [B, H, 3, H, W]
        pred_future     = (current.unsqueeze(1) + residual_pred).clamp(0.0, 1.0)

        L_residual = F.mse_loss(residual_pred, residual_target.detach())
        L_image    = F.mse_loss(pred_future,   future_gt.detach().float())

        loss = (self.cfg.lambda_residual * L_residual
                + self.cfg.lambda_image * L_image)
        return {
            "loss": loss,
            "loss_residual": L_residual.detach(),
            "loss_image":    L_image.detach(),
        }

    # ------------------------------------------------------------------
    # Forward — pixel_residual_roi_dynamic
    # ------------------------------------------------------------------

    def _forward_pixel_residual_roi_dynamic(
        self,
        current: torch.Tensor,
        future_gt: torch.Tensor,
        acts: torch.Tensor,
        B: int,
        H: int,
        device,
    ) -> Dict:
        _, pred_decoded = self._encode_decode(current, acts, B, H, device)

        residual_pred   = pred_decoded
        residual_target = (future_gt - current.unsqueeze(1)).float()
        pred_future     = (current.unsqueeze(1) + residual_pred).clamp(0.0, 1.0)
        future_gt_f     = future_gt.float()

        # ---- Base residual + image losses --------------------------------
        L_residual = F.mse_loss(residual_pred, residual_target.detach())
        L_image    = F.mse_loss(pred_future,   future_gt_f.detach())

        # ---- Dynamic mask (from GT; per future step) --------------------
        # Compute mask for each future step independently, then stack.
        dyn_masks = []
        for h in range(H):
            dm = compute_dynamic_mask(
                current,                   # [B, 3, H, W]
                future_gt_f[:, h],         # [B, 3, H, W]
                threshold=self.cfg.dynamic_threshold,
                dilate_kernel=self.cfg.dynamic_dilate_kernel,
            )                              # [B, 1, H, W]
            dyn_masks.append(dm)
        dyn_mask_seq = torch.stack(dyn_masks, dim=1)  # [B, H, 1, H_img, W_img]

        # ---- Dynamic region loss ----------------------------------------
        L_dynamic = torch.tensor(0.0, device=device)
        for h in range(H):
            dm = dyn_mask_seq[:, h]   # [B, 1, H, W]
            L_dynamic = L_dynamic + masked_mse_loss(
                pred_future[:, h],   # [B, 3, H, W]
                future_gt_f[:, h],
                dm,
            )
        L_dynamic = L_dynamic / max(H, 1)

        # ---- Static consistency loss ------------------------------------
        # Penalise pred_future deviating from current in static regions.
        L_static = torch.tensor(0.0, device=device)
        for h in range(H):
            static_mask = 1.0 - dyn_mask_seq[:, h]   # [B, 1, H, W]
            L_static = L_static + masked_mse_loss(
                pred_future[:, h],
                current,
                static_mask,
            )
        L_static = L_static / max(H, 1)

        # ---- Gripper ROI loss -------------------------------------------
        # Use motion CoM between current and the LAST future frame as gripper proxy.
        cy, cx = motion_center_of_mass(current, future_gt_f[:, -1])

        L_gripper = torch.tensor(0.0, device=device)
        for h in range(H):
            roi_pred = extract_roi_crops(pred_future[:, h], cy, cx,
                                         self.cfg.roi_crop_size)   # [B, 3, roi, roi]
            roi_gt   = extract_roi_crops(future_gt_f[:, h], cy, cx,
                                         self.cfg.roi_crop_size)
            L_gripper = L_gripper + F.mse_loss(roi_pred, roi_gt.detach())
        L_gripper = L_gripper / max(H, 1)

        # ---- Total loss -------------------------------------------------
        loss = (self.cfg.lambda_residual * L_residual
                + self.cfg.lambda_image   * L_image
                + self.cfg.lambda_dynamic * L_dynamic
                + self.cfg.lambda_gripper * L_gripper
                + self.cfg.lambda_static  * L_static)

        return {
            "loss":           loss,
            "loss_residual":  L_residual.detach(),
            "loss_image":     L_image.detach(),
            "loss_dynamic":   L_dynamic.detach(),
            "loss_gripper":   L_gripper.detach(),
            "loss_static":    L_static.detach(),
        }

    # ------------------------------------------------------------------
    # Rollout (no-grad, for evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        pixels: torch.Tensor,    # [B, T+1, H_img, W_img, C] uint8
        actions: torch.Tensor,   # [B, T, action_dim]
        horizon: int = -1,
    ) -> Dict:
        """Predict future images without computing losses.

        Returns dict with:
            pred_future   [B, H, 3, H_img, W_img] float [0, 1]
            current_image [B, 3, H_img, W_img]    float [0, 1]
            future_gt     [B, H, 3, H_img, W_img] float [0, 1]
            residual_pred [B, H, 3, H_img, W_img] float (may be outside [0,1])
            dynamic_mask  [B, H, 1, H_img, W_img] float binary (or None)
        """
        device = next(self.encoder.parameters()).device
        pixels_f = pixels.to(device).permute(0, 1, 4, 2, 3).float() / 255.0

        B, T_plus_1 = pixels_f.shape[:2]
        T = T_plus_1 - 1
        H_full = T - 1
        H = H_full if horizon < 0 else min(horizon, H_full)

        current   = pixels_f[:, 1]         # [B, 3, H_img, W_img]
        future_gt = pixels_f[:, 2:2 + H]   # [B, H, 3, H_img, W_img]

        acts = self._normalize_actions(
            actions.to(device)[:, 1:1 + H, :]
        )  # [B, H, action_dim]

        _, pred_decoded = self._encode_decode(current, acts, B, H, device)

        if self.cfg.target_mode == "pixel":
            pred_future   = pred_decoded.clamp(0.0, 1.0)
            residual_pred = pred_decoded - current.unsqueeze(1)
        else:
            residual_pred = pred_decoded
            pred_future   = (current.unsqueeze(1) + residual_pred).clamp(0.0, 1.0)

        # Dynamic masks (optional, for visualization)
        dyn_masks = []
        for h in range(H):
            dm = compute_dynamic_mask(
                current, future_gt[:, h],
                threshold=self.cfg.dynamic_threshold,
                dilate_kernel=self.cfg.dynamic_dilate_kernel,
            )
            dyn_masks.append(dm)
        dynamic_mask = torch.stack(dyn_masks, dim=1)   # [B, H, 1, H_img, W_img]

        return {
            "pred_future":   pred_future,
            "current_image": current,
            "future_gt":     future_gt,
            "residual_pred": residual_pred,
            "dynamic_mask":  dynamic_mask,
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        import dataclasses
        os.makedirs(save_directory, exist_ok=True)

        def _unwrap(m):
            return m.module if hasattr(m, "module") else m

        torch.save(_unwrap(self.encoder).state_dict(),
                   os.path.join(save_directory, "encoder.pt"))
        torch.save(_unwrap(self.act_encoder).state_dict(),
                   os.path.join(save_directory, "act_encoder.pt"))
        torch.save(_unwrap(self.predictor).state_dict(),
                   os.path.join(save_directory, "predictor.pt"))
        torch.save(_unwrap(self.decoder).state_dict(),
                   os.path.join(save_directory, "decoder.pt"))

        with open(os.path.join(save_directory, "pixel_residual_config.json"),
                  "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=2)

    @classmethod
    def load_pretrained(
        cls,
        save_directory: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "PixelResidualWorldModel":
        import dataclasses

        with open(os.path.join(save_directory, "pixel_residual_config.json"),
                  encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = PixelResidualConfig(**cfg_dict)

        model = cls(cfg, torch_dtype=torch_dtype)

        def _load(module, fname):
            state = torch.load(os.path.join(save_directory, fname), map_location="cpu")
            module.load_state_dict(state, strict=True)

        _load(model.encoder,     "encoder.pt")
        _load(model.act_encoder, "act_encoder.pt")
        _load(model.predictor,   "predictor.pt")
        _load(model.decoder,     "decoder.pt")
        return model
