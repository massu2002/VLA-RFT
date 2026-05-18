"""DynQueryWorldModel — Minimal Action-Conditioned Dynamic Query World Model.

A minimal action-conditioned dynamic-query residual world model:
it keeps the original PixelEncoder → DynamicQueryExtractor →
TemporalResidualPredictor → TokenFuser → PixelDecoder pipeline,
but adds action-conditioned dynamic masks, query-wise future prediction,
and dynamic residual gating to focus prediction on action-relevant dynamic regions.

Core modifications over the base world model:
  Core 1: DynamicQueryExtractor receives action_tokens for action-conditioned masks
  Core 2: TemporalResidualPredictor uses query_wise mode (per-step, per-query)
  Core 3: PixelDecoder output gated by dynamic region mask derived from fuser_masks
  Core 4: L_mask_dynamic + L_query_delta_sparse losses

Forward I/O (training)
----------------------
  Input:
    pixels           [B, K+1+H, H_img, W_img, C]  uint8  channel-last
                     layout: [hist_0 .. hist_{K-1} | current | future_0 .. future_{H-1}]
    actions          [B, K+H, action_dim]
    negative_actions [B, K+H, action_dim] optional (for ranking loss)
  Output: dict — see forward() docstring.

Rollout I/O (evaluation)
------------------------
  Same input convention; returns pred_future + debug tensors.

Checkpoint format
-----------------
  dynquery_config.json  (new)  or  v4_config.json  (legacy — still readable)
  encoder.pt, decoder.pt, act_encoder.pt, spatial_proj.pt,
  spatial_unproj.pt, act_proj.pt, query_extractor.pt,
  temporal_predictor.pt, token_fuser.pt, action_future_scorer.pt
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DynQueryConfig


# ===========================================================================
# Pixel encoder / decoder (CNN backbone)
# ===========================================================================

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.GELU(),
    )


def _deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.GELU(),
    )


class PixelEncoder(nn.Module):
    """5× stride-2 CNN: [B, 3, 256, 256] → [B, N=64, C]."""

    def __init__(self, cfg: DynQueryConfig) -> None:
        super().__init__()
        C = cfg.encoder_channels
        self.net = nn.Sequential(
            _conv_block(3,      C // 8),
            _conv_block(C // 8, C // 4),
            _conv_block(C // 4, C // 2),
            _conv_block(C // 2, C),
            _conv_block(C,      C),
        )
        self.out_channels = C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] → [B, N=64, C]."""
        feat = self.net(x)
        B, C, h, w = feat.shape
        return feat.reshape(B, C, h * w).permute(0, 2, 1)


class PixelDecoder(nn.Module):
    """5× 2× upsample CNN: [B, N=64, C] → [B, 3, 256, 256] (raw, no activation)."""

    def __init__(self, cfg: DynQueryConfig) -> None:
        super().__init__()
        C = cfg.encoder_channels
        self.net = nn.Sequential(
            _deconv_block(C,      C),
            _deconv_block(C,      C // 2),
            _deconv_block(C // 2, C // 4),
            _deconv_block(C // 4, C // 8),
            nn.ConvTranspose2d(C // 8, 3, kernel_size=4, stride=2, padding=1),
        )
        self.spatial_size = 8

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, N=64, C] → [B, 3, 256, 256] (raw, no activation)."""
        B, N, C_in = tokens.shape
        h = w = self.spatial_size
        x = tokens.permute(0, 2, 1).reshape(B, C_in, h, w)
        return self.net(x)


class ActionEncoder(nn.Module):
    """Action encoder supporting discrete_tokens and continuous_mlp modes."""

    def __init__(self, cfg: DynQueryConfig) -> None:
        super().__init__()
        self.mode = getattr(cfg, "action_conditioning_mode", "continuous_mlp")
        self.action_dim = cfg.action_dim
        self.action_bins = int(getattr(cfg, "action_bins", 256))
        self.action_emb_dim = cfg.action_emb_dim
        if self.mode == "discrete_tokens":
            self.emb = nn.Embedding(self.action_bins, cfg.action_emb_dim)
            self.net = nn.Sequential(
                nn.Linear(cfg.action_dim * cfg.action_emb_dim, cfg.action_emb_dim),
                nn.GELU(),
                nn.Linear(cfg.action_emb_dim, cfg.action_emb_dim),
            )
        elif self.mode == "continuous_mlp":
            self.net = nn.Sequential(
                nn.Linear(cfg.action_dim, cfg.action_emb_dim),
                nn.GELU(),
                nn.Linear(cfg.action_emb_dim, cfg.action_emb_dim),
            )
        else:
            raise ValueError(f"Unknown action_conditioning_mode: {self.mode!r}")

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if self.mode == "discrete_tokens":
            ids = torch.floor(actions.clamp(0.0, 1.0) * self.action_bins).long()
            ids = ids.clamp(0, self.action_bins - 1)
            emb = self.emb(ids).reshape(*ids.shape[:2], -1)
            return self.net(emb)
        return self.net(actions)


# ===========================================================================
# Utility functions
# ===========================================================================

def compute_dynamic_mask(
    current_image: torch.Tensor,
    future_image: torch.Tensor,
    threshold: float = 0.05,
    dilate_kernel: int = 7,
) -> torch.Tensor:
    """Binary mask of pixels that changed between frames.

    Returns [B, 1, H, W] float32 (1 = dynamic, 0 = static).
    """
    diff = (future_image.float() - current_image.float()).abs().mean(dim=1)
    mask = (diff > threshold).float().unsqueeze(1)
    if dilate_kernel > 1:
        k = dilate_kernel
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
    return mask


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss restricted to masked pixels. Returns 0 if mask is all zero."""
    total = mask.sum().clamp(min=1.0)
    diff_sq = ((pred - target.detach().to(pred.dtype)) ** 2) * mask.to(pred.dtype)
    return diff_sq.sum() / (total * pred.shape[1])


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] float → [H, W, C] uint8."""
    t = t.detach().cpu().float()
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    return (t.clamp(0, 1).numpy() * 255).astype(np.uint8)


# ===========================================================================
# DynamicQueryExtractor
# ===========================================================================

class DynamicQueryExtractor(nn.Module):
    """Extract Q dynamic queries from K+1 spatial token maps.

    Core 1: Uses current_shared_mask computed from z_t + optional action summary.
    The mask is applied uniformly to all frames, keeping query index q consistent
    across time.  Motion bias (use_motion_bias) adds ||z_t - z_{t-1}|| to logits.
    Action conditioning (use_action_conditioned_mask) adds action summary as a
    spatial bias, steering masks toward action-relevant regions.
    """

    def __init__(self, cfg: DynQueryConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim
        Q = cfg.num_dynamic_queries

        self.mask_proj = nn.Linear(D, D)
        self.mask_head = nn.Linear(D, Q)
        self.query_proj = nn.Linear(D, D)

        self.use_motion_bias = cfg.use_motion_bias
        if cfg.use_motion_bias:
            self.motion_bias_head = nn.Linear(D, Q)

        self.use_action_conditioned_mask = cfg.use_action_conditioned_mask
        if cfg.use_action_conditioned_mask:
            self.action_cond_proj = nn.Linear(D, D)

        self.Q = Q
        self.D = D

    def forward(
        self,
        z_seq: torch.Tensor,                           # [B, K+1, N, D]
        action_tokens: Optional[torch.Tensor] = None,  # [B, H, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (dynamic_queries [B,K+1,Q,D], dynamic_masks [B,K+1,Q,N])."""
        B, T, N, D = z_seq.shape
        z_t = z_seq[:, -1]                              # [B, N, D]

        h = self.mask_proj(z_t)                         # [B, N, D]

        # Core 1: bias spatial attention toward action-relevant regions
        if self.use_action_conditioned_mask and action_tokens is not None:
            act_summary = action_tokens.mean(dim=1)                           # [B, D]
            act_cond = self.action_cond_proj(act_summary).unsqueeze(1)        # [B, 1, D]
            h = h + act_cond                                                  # [B, N, D]

        mask_logits = self.mask_head(h).permute(0, 2, 1)   # [B, Q, N]

        if self.use_motion_bias and T > 1:
            z_prev = z_seq[:, -2]
            motion = (z_t - z_prev).abs()
            bias = self.motion_bias_head(motion).permute(0, 2, 1)            # [B, Q, N]
            mask_logits = mask_logits + bias

        shared_mask = torch.softmax(mask_logits, dim=-1)        # [B, Q, N]
        masks = shared_mask.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, Q, N]

        z_proj = self.query_proj(z_seq)                         # [B, T, N, D]
        queries = torch.einsum("btqn,btnd->btqd", masks, z_proj)

        return queries, masks


# ===========================================================================
# TemporalResidualPredictor
# ===========================================================================

class TemporalResidualPredictor(nn.Module):
    """Predict future dynamic queries from history queries and action sequence.

    Core 2: query_wise mode (default) builds per-(step, query) seed tokens from
    current_queries + action_tokens + step_emb, cross-attends to encoded context,
    and predicts a per-query delta.  Returns both future_dynamic_queries and
    future_delta_queries (the raw delta, used for L_query_delta_sparse).

    Legacy linear_expand mode is kept for backward compatibility.
    """

    def __init__(self, cfg: DynQueryConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim
        Q = cfg.num_dynamic_queries
        H = cfg.action_horizon

        ctx_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(ctx_layer, cfg.n_context_layers)

        self.step_emb = nn.Embedding(H + 1, D)

        self.predictor_mode = cfg.predictor_mode  # "query_wise" | "linear_expand"

        # Only create components for the active mode to avoid DDP unused-parameter errors.
        if cfg.predictor_mode == "query_wise":
            self.qw_cross_attn = nn.MultiheadAttention(
                D, cfg.n_heads, dropout=cfg.dropout, batch_first=True
            )
            self.qw_cross_norm = nn.LayerNorm(D)
            self.query_delta_head = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, D),
            )
        else:
            # linear_expand mode
            self.cross_attn = nn.MultiheadAttention(D, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(D)
            self.out_proj   = nn.Linear(D, Q * D)
        self.Q = Q
        self.D = D

    def forward(
        self,
        current_queries: torch.Tensor,   # [B, Q, D]
        residual_queries: torch.Tensor,  # [B, K, Q, D]
        action_tokens: torch.Tensor,     # [B, H, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (future_dynamic_queries [B,H,Q,D], future_delta_queries [B,H,Q,D])."""
        B, Q, D = current_queries.shape
        K = residual_queries.shape[1]
        H = action_tokens.shape[1]

        # Shared context: current queries + K residual deltas → [B, (K+1)*Q, D]
        resid_flat = residual_queries.reshape(B, K * Q, D)
        context = torch.cat([current_queries, resid_flat], dim=1)
        context = self.context_encoder(context)

        step_ids = torch.arange(H, device=action_tokens.device)
        step_emb = self.step_emb(step_ids).unsqueeze(0).expand(B, -1, -1)  # [B, H, D]

        if self.predictor_mode == "query_wise":
            # Core 2: per-(step, query) seed → cross-attend → delta
            future_seed = (
                current_queries[:, None, :, :]   # [B, 1, Q, D]
                + action_tokens[:, :, None, :]   # [B, H, 1, D]
                + step_emb[:, :, None, :]        # [B, H, 1, D]
            )                                    # [B, H, Q, D]
            future_tokens = future_seed.reshape(B, H * Q, D)
            attn_out, _ = self.qw_cross_attn(
                query=future_tokens, key=context, value=context
            )
            future_out = self.qw_cross_norm(attn_out + future_tokens)
            future_out = future_out.reshape(B, H, Q, D)
            delta = self.query_delta_head(future_out)              # [B, H, Q, D]
            future_dynamic_queries = current_queries[:, None, :, :] + delta
        else:
            # Fallback linear_expand mode
            horizon = action_tokens + step_emb                     # [B, H, D]
            attn_out, _ = self.cross_attn(query=horizon, key=context, value=context)
            horizon_out = self.cross_norm(attn_out + horizon)
            future_flat = self.out_proj(horizon_out)
            future_dynamic_queries = future_flat.reshape(B, H, Q, D)
            delta = future_dynamic_queries - current_queries[:, None, :, :]

        return future_dynamic_queries, delta


# ===========================================================================
# TokenFuser
# ===========================================================================

class TokenFuser(nn.Module):
    """Fuse future dynamic queries into current spatial tokens (single_mask mode).

    For each horizon step h:
        fuser_mask[h, q, n] = softmax_n( q[h,q] · k[n] / sqrt(D) )
        update[h, n]        = sum_q fuser_mask[h,q,n] * W_v(q[h,q])
        z_future[h, n]      = z_current[n] + update[h, n]

    Core 3 (gate) is applied in DynQueryWorldModel.forward(), not here.
    """

    def __init__(self, cfg: DynQueryConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim

        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.scale = D ** -0.5

        if cfg.n_fuser_layers > 0:
            fuser_layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_dim,
                dropout=cfg.dropout, batch_first=True, norm_first=True,
            )
            self.refine = nn.TransformerEncoder(fuser_layer, cfg.n_fuser_layers)
        else:
            self.refine = None

        self.D = D

    def forward(
        self,
        current_spatial_tokens: torch.Tensor,  # [B, N, D]
        future_dynamic_queries: torch.Tensor,  # [B, H, Q, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (future_spatial [B,H,N,D], fuser_masks [B,H,Q,N])."""
        B, N, D = current_spatial_tokens.shape
        _, H, Q, _ = future_dynamic_queries.shape

        k = self.k_proj(current_spatial_tokens)    # [B, N, D]
        q = self.q_proj(future_dynamic_queries)    # [B, H, Q, D]
        v = self.v_proj(future_dynamic_queries)    # [B, H, Q, D]

        attn = torch.einsum("bhqd,bnd->bhqn", q, k) * self.scale
        fuser_masks = torch.softmax(attn, dim=-1)  # [B, H, Q, N]

        update = torch.einsum("bhqn,bhqd->bhnd", fuser_masks, v)   # [B, H, N, D]
        future_spatial = current_spatial_tokens.unsqueeze(1) + update

        if self.refine is not None:
            future_spatial = self.refine(
                future_spatial.reshape(B * H, N, D)
            ).reshape(B, H, N, D)

        return future_spatial, fuser_masks


# ===========================================================================
# ActionFutureScorer  (stage_b)
# ===========================================================================

class ActionFutureScorer(nn.Module):
    """Score action candidates from future dynamic query evolution (stage_b).

    Higher score = model predicts this action leads to a dynamic future
    consistent with the current history context.
    """

    def __init__(self, cfg: DynQueryConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim

        self.act_query_proj = nn.Linear(D, D)

        if cfg.n_scorer_layers > 0:
            dec_layer = nn.TransformerDecoderLayer(
                d_model=D, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_dim,
                dropout=cfg.dropout, batch_first=True, norm_first=True,
            )
            self.cross_decoder = nn.TransformerDecoder(dec_layer, cfg.n_scorer_layers)
        else:
            self.cross_decoder = None

        self.score_mlp = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )
        self.D = D

    def forward(
        self,
        action_tokens: torch.Tensor,           # [B, H, D]
        future_dynamic_queries: torch.Tensor,  # [B, H, Q, D]
    ) -> Dict[str, torch.Tensor]:
        B, H, D = action_tokens.shape
        Q = future_dynamic_queries.shape[2]

        act_q = self.act_query_proj(action_tokens)
        mem = future_dynamic_queries.reshape(B, H * Q, D)

        if self.cross_decoder is not None:
            scored = self.cross_decoder(act_q, mem)
        else:
            scored = act_q

        pooled = scored.mean(dim=1)
        score = self.score_mlp(pooled).squeeze(-1)
        return {"ranking_score": score}


# ===========================================================================
# DynQueryWorldModel  (main model)
# ===========================================================================

class DynQueryWorldModel(nn.Module):
    """Minimal Action-Conditioned Dynamic Query World Model.

    Forward expects:
        pixels  [B, K+1+H, H_img, W_img, C]  uint8  channel-last
        actions [B, K+H, action_dim]

    Frame layout:
        pixels[:, 0:K]       — history frames  (K = history_length)
        pixels[:, K]         — current frame
        pixels[:, K+1:K+1+H] — future GT frames

    Action layout:
        actions[:, 0:K]      — history actions (skipped)
        actions[:, K:K+H]    — prediction horizon actions
    """

    def __init__(
        self,
        cfg: DynQueryConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        D = cfg.hidden_dim
        C_enc = cfg.encoder_channels

        self.encoder = PixelEncoder(cfg)
        self.decoder = PixelDecoder(cfg)
        self.act_encoder = ActionEncoder(cfg)

        self.spatial_proj   = nn.Linear(C_enc, D)
        self.spatial_unproj = nn.Linear(D, C_enc)
        self.act_proj       = nn.Linear(cfg.action_emb_dim, D)

        self.query_extractor    = DynamicQueryExtractor(cfg)
        self.temporal_predictor = TemporalResidualPredictor(cfg)
        self.token_fuser        = TokenFuser(cfg)

        if cfg.use_action_future_scorer:
            self.action_future_scorer: Optional[ActionFutureScorer] = ActionFutureScorer(cfg)
        else:
            self.action_future_scorer = None

        if torch_dtype is not None:
            self.to(dtype=torch_dtype)

        self._zero_init_decoder_output()

        try:
            ranges = torch.load(cfg.action_ranges_path, map_location="cpu")
            self.register_buffer("action_ranges", ranges, persistent=False)
        except Exception:
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
        for m in [self.encoder, self.act_encoder, self.query_extractor,
                  self.temporal_predictor, self.token_fuser, self.decoder]:
            m.requires_grad_(True)
        if self.action_future_scorer is not None:
            self.action_future_scorer.requires_grad_(True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _zero_init_decoder_output(self) -> None:
        last = self.decoder.net[-1]
        if isinstance(last, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        ar = self.action_ranges.to(actions.device)
        mn, mx = ar[:, 0], ar[:, 1]
        return torch.clamp((actions - mn) / (mx - mn + 1e-8), 0.0, 1.0)

    def _residual_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        mode = self.cfg.residual_output_activation
        if mode == "tanh":
            scale = max(float(self.cfg.residual_output_scale), 1e-6)
            return scale * torch.tanh(raw / scale)
        return raw

    def _autocast_ctx(self, device):
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
        dtype = dtype_map.get(self.cfg.autocast_dtype, torch.float32)
        dev_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
        if dev_type == "cpu":
            return torch.autocast(device_type="cpu", dtype=torch.float32)
        return torch.autocast(device_type=dev_type, dtype=dtype)

    def _encode_frame_seq(self, images: torch.Tensor, device) -> torch.Tensor:
        """[B, T, 3, H_img, W_img] → [B, T, N, D]."""
        B, T, C, H_img, W_img = images.shape
        dtype = next(self.encoder.parameters()).dtype
        with self._autocast_ctx(device):
            flat = images.reshape(B * T, C, H_img, W_img).to(dtype)
            enc = self.encoder(flat)
            N = enc.shape[1]
            proj = self.spatial_proj(enc)
        return proj.reshape(B, T, N, self.cfg.hidden_dim)

    def _decode_future_tokens(self, future_spatial: torch.Tensor, B: int, H: int, device) -> torch.Tensor:
        """[B, H, N, D] → [B, H, 3, H_img, W_img]."""
        N, D = future_spatial.shape[2], future_spatial.shape[3]
        dtype = next(self.decoder.parameters()).dtype
        with self._autocast_ctx(device):
            flat = future_spatial.reshape(B * H, N, D).to(dtype)
            dec_tokens = self.spatial_unproj(flat)
            decoded = self.decoder(dec_tokens)
        H_img, W_img = decoded.shape[-2:]
        return decoded.reshape(B, H, 3, H_img, W_img).float()

    def _compute_dynamic_gate(
        self,
        fuser_masks: torch.Tensor,  # [B, H, Q, N]
        B: int,
        H: int,
        H_img: int,
        W_img: int,
    ) -> torch.Tensor:
        """Compute dynamic residual gate [B, H, 1, H_img, W_img] from fuser_masks.

        Soft union of Q query masks → 8×8 spatial gate → upsample to image resolution.
        """
        # Soft union over Q: [B, H, N]
        dyn_tok_mask = 1.0 - torch.prod(1.0 - fuser_masks.clamp(0, 1).float(), dim=2)
        dyn_mask_8 = dyn_tok_mask.reshape(B * H, 1, 8, 8)
        dyn_mask_img = F.interpolate(
            dyn_mask_8, size=(H_img, W_img), mode="bilinear", align_corners=False
        ).reshape(B, H, 1, H_img, W_img)
        return dyn_mask_img

    def _encode_and_query(
        self,
        history_frames: torch.Tensor,
        current: torch.Tensor,
        acts: torch.Tensor,
        B: int,
        H: int,
        K: int,
        device,
    ) -> Dict:
        """Shared encode + query pipeline for forward() and rollout()."""
        dtype = next(self.encoder.parameters()).dtype

        # Build frame sequence [history_0..K-1, current]
        current_unsq = current.unsqueeze(1)
        all_frames = torch.cat([history_frames, current_unsq], dim=1) if K > 0 else current_unsq
        z_seq = self._encode_frame_seq(all_frames, device)    # [B, K+1, N, D]
        current_spatial = z_seq[:, -1]                        # [B, N, D]

        # Encode actions before query extraction (Core 1 needs action_tokens)
        with self._autocast_ctx(device):
            act_emb = self.act_encoder(acts.to(dtype))
            action_tokens = self.act_proj(act_emb)            # [B, H, D]

        # Core 1: extract dynamic queries with action conditioning
        dynamic_queries, dynamic_masks = self.query_extractor(z_seq, action_tokens)
        # dynamic_queries: [B, K+1, Q, D]   dynamic_masks: [B, K+1, Q, N]

        current_queries = dynamic_queries[:, -1]              # [B, Q, D]
        if K > 0:
            residual_queries = dynamic_queries[:, 1:] - dynamic_queries[:, :-1]  # [B, K, Q, D]
        else:
            Q_dim = self.cfg.num_dynamic_queries
            D_dim = self.cfg.hidden_dim
            residual_queries = torch.zeros(B, 1, Q_dim, D_dim, device=device, dtype=dtype)

        # Core 2: predict future queries (query_wise mode)
        future_dynamic_queries, future_delta_queries = self.temporal_predictor(
            current_queries, residual_queries, action_tokens
        )                                                     # [B, H, Q, D] each

        # Fuse into future spatial tokens
        future_spatial, fuser_masks = self.token_fuser(current_spatial, future_dynamic_queries)
        # future_spatial: [B, H, N, D]   fuser_masks: [B, H, Q, N]

        return {
            "z_seq":                  z_seq,
            "current_spatial":        current_spatial,
            "dynamic_queries":        dynamic_queries,
            "dynamic_masks":          dynamic_masks,
            "current_queries":        current_queries,
            "residual_queries":       residual_queries,
            "action_tokens":          action_tokens,
            "future_dynamic_queries": future_dynamic_queries,
            "future_delta_queries":   future_delta_queries,
            "future_spatial":         future_spatial,
            "fuser_masks":            fuser_masks,
        }

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        negative_actions: Optional[torch.Tensor] = None,
        return_debug: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training forward.

        Output dict keys:
          loss                    — scalar training loss
          loss_image / loss_dynamic / loss_static / loss_query / loss_rank
          loss_mask_dynamic / loss_query_delta_sparse
          dynamic_mask_area / query_delta_norm / dynamic_residual_gate_area
          pred_future             [B, H, 3, H_img, W_img]
          dynamic_residual_gate   [B, H, 1, H_img, W_img]
          dynamic_queries         [B, K+1, Q, D]
          dynamic_masks           [B, K+1, Q, N]
          current_queries         [B, Q, D]
          residual_queries        [B, K, Q, D]
          future_dynamic_queries  [B, H, Q, D]
          future_delta_queries    [B, H, Q, D]
          future_spatial          [B, H, N, D]
          fuser_masks             [B, H, Q, N]
          ranking_score           [B] or absent
        """
        device = next(self.encoder.parameters()).device
        pixels = pixels.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        pixels_f = pixels.permute(0, 1, 4, 2, 3).contiguous().float() / 255.0

        K = self.cfg.history_length
        B = pixels_f.shape[0]
        H = pixels_f.shape[1] - K - 1

        if H < 1:
            raise ValueError(
                f"forward: need at least K+2 frames, got {pixels_f.shape[1]} K={K} → H={H}"
            )

        history_frames = pixels_f[:, 0:K]
        current        = pixels_f[:, K]
        future_gt      = pixels_f[:, K + 1: K + 1 + H]
        H_img, W_img   = current.shape[-2], current.shape[-1]
        N = self.cfg.n_spatial_tokens

        acts = self._normalize_actions(actions[:, K: K + H])

        enc = self._encode_and_query(history_frames, current, acts, B, H, K, device)

        # Decode future spatial tokens → raw residual
        pred_decoded  = self._decode_future_tokens(enc["future_spatial"], B, H, device)
        residual_pred = self._residual_from_raw(pred_decoded)

        # Core 3: compute dynamic gate from fuser_masks (always from fuser_masks)
        dyn_mask_img = self._compute_dynamic_gate(
            enc["fuser_masks"], B, H, H_img, W_img
        ).to(residual_pred.dtype)                            # [B, H, 1, H_img, W_img]

        if self.cfg.use_dynamic_residual_gate:
            residual_pred = residual_pred * dyn_mask_img

        pred_future = (current.unsqueeze(1) + residual_pred).clamp(0.0, 1.0)

        # ----- Losses -----
        future_gt_f = future_gt.float()

        L_image = F.mse_loss(pred_future, future_gt_f.detach())

        # Compute per-step GT dynamic masks
        dyn_masks = [
            compute_dynamic_mask(
                current, future_gt_f[:, h],
                self.cfg.dynamic_threshold, self.cfg.dynamic_dilate_kernel,
            )
            for h in range(H)
        ]
        dyn_mask_seq = torch.stack(dyn_masks, dim=1)         # [B, H, 1, H_img, W_img]

        L_dynamic = sum(
            masked_mse_loss(pred_future[:, h], future_gt_f[:, h], dyn_mask_seq[:, h])
            for h in range(H)
        ) / max(H, 1)

        L_static = sum(
            masked_mse_loss(pred_future[:, h], current, 1.0 - dyn_mask_seq[:, h])
            for h in range(H)
        ) / max(H, 1)

        L_query = self._query_future_loss(
            future_gt_f, enc["dynamic_masks"][:, -1], enc["future_dynamic_queries"], device,
        )

        # Core 4a: L_mask_dynamic — encourage query masks to cover GT dynamic region
        _eps = 1e-6
        if self.cfg.lambda_mask_dynamic > 0.0:
            cur_dyn_masks = enc["dynamic_masks"][:, -1]             # [B, Q, N]
            mask_union = 1.0 - torch.prod(
                1.0 - cur_dyn_masks.clamp(0, 1).float(), dim=1
            )                                                        # [B, N]
            gt_dyn_any = dyn_mask_seq.max(dim=1).values              # [B, 1, H_img, W_img]
            gt_dyn_8 = F.interpolate(
                gt_dyn_any.float(), size=(8, 8), mode="area"
            ).reshape(B, N)
            # binary_cross_entropy is blocked at dispatch level under autocast (not just a dtype issue).
            # Disable autocast locally and compute in float32.
            _dev_type = next(self.encoder.parameters()).device.type
            with torch.autocast(device_type=_dev_type, enabled=False):
                L_mask_dynamic: torch.Tensor = F.binary_cross_entropy(
                    mask_union.clamp(_eps, 1 - _eps).float(), gt_dyn_8.detach().float()
                )
        else:
            L_mask_dynamic = pred_future.new_zeros(())

        # Core 4b: L_query_delta_sparse — keep query changes sparse
        if self.cfg.lambda_query_delta_sparse > 0.0:
            L_query_delta_sparse: torch.Tensor = (
                enc["future_delta_queries"].norm(dim=-1).mean()
            )
        else:
            L_query_delta_sparse = pred_future.new_zeros(())

        # Ranking loss (stage_b)
        L_rank: torch.Tensor = pred_future.new_zeros(())
        ranking_score: Optional[torch.Tensor] = None
        if self.action_future_scorer is not None:
            scorer_out = self.action_future_scorer(
                enc["action_tokens"],
                enc["future_dynamic_queries"].detach(),
            )
            ranking_score = scorer_out["ranking_score"]

            neg_acts = self._get_negative_actions(negative_actions, actions, K, H, device)
            if neg_acts is not None:
                dtype_model = next(self.encoder.parameters()).dtype
                with torch.no_grad():
                    with self._autocast_ctx(device):
                        neg_emb = self.act_encoder(neg_acts.to(dtype_model))
                        neg_action_tokens = self.act_proj(neg_emb)
                    neg_future_q, _ = self.temporal_predictor(
                        enc["current_queries"], enc["residual_queries"], neg_action_tokens
                    )
                neg_score = self.action_future_scorer(
                    neg_action_tokens, neg_future_q
                )["ranking_score"]

                temperature = self.cfg.rank_temperature
                B_rank = ranking_score.shape[0]
                if B_rank > 1:
                    logits = torch.cat([
                        ranking_score.unsqueeze(1),
                        neg_score.unsqueeze(0).expand(B_rank, -1),
                    ], dim=1) / temperature
                    labels = torch.zeros(B_rank, dtype=torch.long, device=device)
                    L_rank = F.cross_entropy(logits, labels)
                else:
                    L_rank = F.softplus((neg_score - ranking_score) / temperature).mean()

        cfg = self.cfg
        loss = (
            cfg.lambda_image             * L_image
            + cfg.lambda_dynamic         * L_dynamic
            + cfg.lambda_static          * L_static
            + cfg.lambda_query           * L_query
            + cfg.lambda_rank            * L_rank
            + cfg.lambda_mask_dynamic    * L_mask_dynamic
            + cfg.lambda_query_delta_sparse * L_query_delta_sparse
        )

        out: Dict[str, torch.Tensor] = {
            "loss":                    loss,
            "loss_image":              L_image.detach(),
            "loss_dynamic":            L_dynamic.detach(),
            "loss_static":             L_static.detach(),
            "loss_query":              L_query.detach(),
            "loss_rank":               L_rank.detach(),
            "loss_mask_dynamic":       L_mask_dynamic.detach(),
            "loss_query_delta_sparse": L_query_delta_sparse.detach(),
            # Diagnostics
            "dynamic_mask_area":           dyn_mask_seq.float().mean().detach(),
            "query_delta_norm":            enc["future_delta_queries"].norm(dim=-1).mean().detach(),
            "dynamic_residual_gate_area":  dyn_mask_img.float().mean().detach(),
            # Predictions (always included)
            "pred_future":             pred_future.detach(),
            "dynamic_residual_gate":   dyn_mask_img.detach(),
            # Internal tensors
            "dynamic_queries":         enc["dynamic_queries"].detach(),
            "dynamic_masks":           enc["dynamic_masks"].detach(),
            "current_queries":         enc["current_queries"].detach(),
            "residual_queries":        enc["residual_queries"].detach(),
            "future_dynamic_queries":  enc["future_dynamic_queries"].detach(),
            "future_delta_queries":    enc["future_delta_queries"].detach(),
            "future_spatial":          enc["future_spatial"].detach(),
            "fuser_masks":             enc["fuser_masks"].detach(),
        }

        if ranking_score is not None:
            out["ranking_score"] = ranking_score.detach()

        # Zero-action diagnostic (only when scorer is active)
        if (
            self.action_future_scorer is not None
            and os.environ.get("DISABLE_ZERO_ACTION_DIAGNOSTIC", "0") not in ("1", "true", "TRUE", "yes", "YES")
        ):
            dtype_model = next(self.encoder.parameters()).dtype
            with torch.no_grad():
                with self._autocast_ctx(device):
                    zero_emb = self.act_encoder(torch.zeros_like(acts).to(dtype_model))
                    zero_tokens = self.act_proj(zero_emb)
                zero_future_q, _ = self.temporal_predictor(
                    enc["current_queries"], enc["residual_queries"], zero_tokens
                )
            zero_score = self.action_future_scorer(zero_tokens, zero_future_q)["ranking_score"]
            out["score_zero_action"] = zero_score.mean().detach()

        return out

    def _get_negative_actions(
        self,
        negative_actions: Optional[torch.Tensor],
        actions: torch.Tensor,
        K: int,
        H: int,
        device,
    ) -> Optional[torch.Tensor]:
        if negative_actions is not None:
            neg = negative_actions.to(device, non_blocking=True)
            return self._normalize_actions(neg[:, K: K + H])

        acts_raw = actions[:, K: K + H]
        H_len = acts_raw.shape[1]
        if H_len < 2:
            return None
        perm = torch.randperm(H_len, device=device)
        if (perm == torch.arange(H_len, device=device)).all():
            perm = torch.roll(perm, 1)
        return self._normalize_actions(acts_raw[:, perm, :])

    def _query_future_loss(
        self,
        future_gt_f: torch.Tensor,
        current_mask: torch.Tensor,
        pred_future_queries: torch.Tensor,
        device,
    ) -> torch.Tensor:
        B, H, C, H_img, W_img = future_gt_f.shape
        z_future_seq = self._encode_frame_seq(future_gt_f, device)
        mask_h = current_mask.unsqueeze(1).expand(-1, H, -1, -1)
        gt_future_queries = torch.einsum("bhqn,bhnd->bhqd", mask_h, z_future_seq)
        return F.mse_loss(pred_future_queries, gt_future_queries.detach())

    # ------------------------------------------------------------------
    # Rollout (no-grad, for evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        horizon: int = -1,
    ) -> Dict:
        """Predict future images without computing losses.

        Returns:
            pred_future             [B, H, 3, H_img, W_img]
            future_dynamic_queries  [B, H, Q, D]
            future_delta_queries    [B, H, Q, D]
            future_spatial          [B, H, N, D]
            dynamic_masks           [B, K+1, Q, N]
            fuser_masks             [B, H, Q, N]
            dynamic_residual_gate   [B, H, 1, H_img, W_img]
            ranking_score           [B] or None
            current_image           [B, 3, H_img, W_img]
            future_gt               [B, H, 3, H_img, W_img]
        """
        device = next(self.encoder.parameters()).device
        pixels_f = pixels.to(device).permute(0, 1, 4, 2, 3).float() / 255.0
        actions = actions.to(device)

        K = self.cfg.history_length
        B = pixels_f.shape[0]
        H_full = pixels_f.shape[1] - K - 1
        H = H_full if horizon < 0 else min(horizon, H_full)
        H_img, W_img = pixels_f.shape[-2], pixels_f.shape[-1]

        history_frames = pixels_f[:, 0:K]
        current        = pixels_f[:, K]
        future_gt      = pixels_f[:, K + 1: K + 1 + H]

        acts = self._normalize_actions(actions[:, K: K + H])

        enc = self._encode_and_query(history_frames, current, acts, B, H, K, device)

        pred_decoded  = self._decode_future_tokens(enc["future_spatial"], B, H, device)
        residual_pred = self._residual_from_raw(pred_decoded)

        # Core 3: dynamic residual gate (same as forward())
        dyn_mask_img = self._compute_dynamic_gate(
            enc["fuser_masks"], B, H, H_img, W_img
        ).to(residual_pred.dtype)

        if self.cfg.use_dynamic_residual_gate:
            residual_pred = residual_pred * dyn_mask_img

        pred_future = (current.unsqueeze(1) + residual_pred).clamp(0.0, 1.0)

        ranking_score = None
        if self.action_future_scorer is not None:
            ranking_score = self.action_future_scorer(
                enc["action_tokens"], enc["future_dynamic_queries"]
            )["ranking_score"]

        return {
            "pred_future":            pred_future,
            "future_dynamic_queries": enc["future_dynamic_queries"],
            "future_delta_queries":   enc["future_delta_queries"],
            "future_spatial":         enc["future_spatial"],
            "dynamic_masks":          enc["dynamic_masks"],
            "fuser_masks":            enc["fuser_masks"],
            "dynamic_residual_gate":  dyn_mask_img,
            "ranking_score":          ranking_score,
            "current_image":          current,
            "future_gt":              future_gt,
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        os.makedirs(save_directory, exist_ok=True)

        def _unwrap(m):
            return m.module if hasattr(m, "module") else m

        _modules = {
            "encoder":            self.encoder,
            "decoder":            self.decoder,
            "act_encoder":        self.act_encoder,
            "spatial_proj":       self.spatial_proj,
            "spatial_unproj":     self.spatial_unproj,
            "act_proj":           self.act_proj,
            "query_extractor":    self.query_extractor,
            "temporal_predictor": self.temporal_predictor,
            "token_fuser":        self.token_fuser,
        }
        if self.action_future_scorer is not None:
            _modules["action_future_scorer"] = self.action_future_scorer

        for name, mod in _modules.items():
            torch.save(
                _unwrap(mod).state_dict(),
                os.path.join(save_directory, f"{name}.pt"),
            )

        cfg_dict = dataclasses.asdict(self.cfg)
        with open(os.path.join(save_directory, "dynquery_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

    @classmethod
    def load_pretrained(
        cls,
        save_directory: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "DynQueryWorldModel":
        for cfg_name in ("dynquery_config.json", "v4_config.json"):
            cfg_path = os.path.join(save_directory, cfg_name)
            if os.path.exists(cfg_path):
                break
        else:
            raise FileNotFoundError(
                f"No config file found in {save_directory} "
                f"(tried dynquery_config.json, v4_config.json)"
            )

        with open(cfg_path, encoding="utf-8") as f:
            cfg_dict = json.load(f)

        cfg_dict.setdefault("model_generation", "dynquery")
        # Filter to only known fields (forward-compat: ignore fields from future versions)
        known = {f.name for f in dataclasses.fields(DynQueryConfig)}
        cfg = DynQueryConfig(**{k: v for k, v in cfg_dict.items() if k in known})
        model = cls(cfg, torch_dtype=torch_dtype)

        def _load(module, fname):
            path = os.path.join(save_directory, fname)
            if os.path.exists(path):
                state = torch.load(path, map_location="cpu")
                module.load_state_dict(state, strict=False)  # strict=False for compat

        _load(model.encoder,            "encoder.pt")
        _load(model.decoder,            "decoder.pt")
        _load(model.act_encoder,        "act_encoder.pt")
        _load(model.spatial_proj,       "spatial_proj.pt")
        _load(model.spatial_unproj,     "spatial_unproj.pt")
        _load(model.act_proj,           "act_proj.pt")
        _load(model.query_extractor,    "query_extractor.pt")
        _load(model.temporal_predictor, "temporal_predictor.pt")
        _load(model.token_fuser,        "token_fuser.pt")
        if model.action_future_scorer is not None:
            _load(model.action_future_scorer, "action_future_scorer.pt")

        return model


# Backward-compat alias.
TemporalDynamicQueryResidualWM = DynQueryWorldModel
