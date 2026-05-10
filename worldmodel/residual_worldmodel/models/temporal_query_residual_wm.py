"""TemporalDynamicQueryResidualWM — v4 world model.

Architecture
------------
Instead of predicting dense pixel residuals directly, v4 extracts a small set
of *dynamic queries* from the history image sequence and rolls them forward
using the action sequence, then fuses the resulting future queries back into
the current spatial feature map to decode future images.

Stages
------
  DynamicQueryExtractor   : [B,K+1,N,D] → dynamic_queries [B,K+1,Q,D]
                                        + dynamic_masks   [B,K+1,Q,N]
  TemporalResidualPredictor: history queries + actions → future_dynamic_queries [B,H,Q,D]
  TokenFuser              : current spatial tokens + future_dynamic_queries → future_spatial [B,H,N,D]
  PixelDecoder            : future_spatial → pred_future [B,H,3,H,W]
  ActionFutureScorer      : (v4b) actions + future_dynamic_queries → ranking_score [B]

Forward I/O (training)
----------------------
  Input:
    pixels           [B, K+2+H, H_img, W_img, C]  uint8  channel-last
                     layout: [hist_0 .. hist_{K-1} | context | current | future_0 .. future_{H-1}]
    actions          [B, K+1+H, action_dim]
    negative_actions [B, K+1+H, action_dim] optional (for ranking loss)
  Output:
    loss, loss_image, loss_dynamic, loss_static, loss_query, loss_rank, loss_sparse
    pred_future, future_spatial_tokens, future_dynamic_queries,
    dynamic_masks, fuser_masks, ranking_score (optional)

Rollout I/O (no-grad evaluation)
---------------------------------
  Same input convention; returns pred_future + debug tensors.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..v4_config import TemporalQueryResidualConfig
from ..pixel_residual_model import PixelEncoder, PixelDecoder, ActionEncoder
from ..pixel_residual_utils import compute_dynamic_mask, masked_mse_loss


# ---------------------------------------------------------------------------
# Mask regularisation helpers
# ---------------------------------------------------------------------------

def _mask_entropy(mask: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy of softmax masks [B, ..., Q, N].

    Entropy = -sum_n p_n * log(p_n).
    Minimising this encourages each query mask to concentrate on few tokens
    (peaked / sparse attention) rather than collapsing to uniform.

    For a uniform distribution over N=64 tokens, entropy ≈ log(64) ≈ 4.15.
    For a near-delta mask, entropy ≈ 0.
    """
    p = mask.clamp(min=1e-9)
    return -(p * torch.log(p)).sum(dim=-1).mean()


def _mask_diversity(mask: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine similarity across the Q query dimension.

    Penalises different queries attending to the same spatial tokens.
    Minimising this encourages Q distinct attention patterns.
    Shape: mask [B, ..., Q, N] → flattened to [M, Q, N].

    Loss is scaled by Q/8 so that higher Q configurations (e.g. Q=16)
    receive proportionally stronger diversity enforcement and don't collapse.
    """
    Q, N = mask.shape[-2], mask.shape[-1]
    if Q < 2:
        return mask.new_zeros(())
    flat = mask.reshape(-1, Q, N).float()               # [M, Q, N]
    norm = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-9)  # [M, Q, N]
    sim  = torch.bmm(norm, norm.transpose(1, 2))        # [M, Q, Q]
    # off-diagonal only (exclude self-similarity = 1)
    eye  = torch.eye(Q, device=mask.device, dtype=sim.dtype).unsqueeze(0)
    n_off = flat.shape[0] * Q * (Q - 1)
    base  = (sim * (1.0 - eye)).sum() / max(n_off, 1)
    # Q-proportional scaling: Q=8 → ×1.0, Q=16 → ×2.0
    return base * (Q / 8.0)


# ===========================================================================
# DynamicQueryExtractor
# ===========================================================================

class DynamicQueryExtractor(nn.Module):
    """Extract Q dynamic queries from K+1 spatial token maps.

    Uses *current_shared_mask*: soft attention masks are computed solely from
    z_t (the current frame), then applied uniformly to all frames z_{t-K}..z_t.
    This keeps query index q consistent across time, easing temporal reasoning.

    Optional USE_MOTION_BIAS: adds ||z_t - z_{t-1}|| to mask logits so that
    highly-moving spatial positions are more likely to be selected.
    """

    def __init__(self, cfg: TemporalQueryResidualConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim
        Q = cfg.num_dynamic_queries

        self.mask_proj = nn.Linear(D, D)
        self.mask_head = nn.Linear(D, Q)        # spatial token → Q logit
        self.query_proj = nn.Linear(D, D)

        self.use_motion_bias = cfg.use_motion_bias
        if cfg.use_motion_bias:
            self.motion_bias_head = nn.Linear(D, Q)

        self.Q = Q
        self.D = D

    def forward(
        self, z_seq: torch.Tensor               # [B, K+1, N, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            dynamic_queries  [B, K+1, Q, D]
            dynamic_masks    [B, K+1, Q, N]   (softmax weights, shared from z_t)
        """
        B, T, N, D = z_seq.shape
        z_t = z_seq[:, -1]                       # [B, N, D]

        # Mask logits from current frame tokens
        h = self.mask_proj(z_t)                  # [B, N, D]
        mask_logits = self.mask_head(h).permute(0, 2, 1)  # [B, Q, N]

        if self.use_motion_bias and T > 1:
            z_prev = z_seq[:, -2]                # [B, N, D]
            motion = (z_t - z_prev).abs()        # [B, N, D]
            bias = self.motion_bias_head(motion).permute(0, 2, 1)  # [B, Q, N]
            mask_logits = mask_logits + bias

        # Softmax over N spatial tokens → soft attention mask
        shared_mask = torch.softmax(mask_logits, dim=-1)   # [B, Q, N]

        # Expand to all time steps
        masks = shared_mask.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, Q, N]

        # Extract queries: for each frame t and query q, sum over N spatial tokens
        z_proj = self.query_proj(z_seq)          # [B, T, N, D]
        # [B,T,Q,N] x [B,T,N,D] -> [B,T,Q,D]
        queries = torch.einsum("btqn,btnd->btqd", masks, z_proj)

        return queries, masks


# ===========================================================================
# TemporalResidualPredictor
# ===========================================================================

class TemporalResidualPredictor(nn.Module):
    """Predict future dynamic queries from history queries and action sequence.

    Context tokens: current_queries + flattened K residual_queries
    Horizon tokens: action_tokens + learned step embedding
    Cross-attention: horizon queries attend to encoded context
    Output: [B, H, Q, D]
    """

    def __init__(self, cfg: TemporalQueryResidualConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim
        Q = cfg.num_dynamic_queries
        H = cfg.action_horizon

        # Context self-attention
        ctx_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(ctx_layer, cfg.n_context_layers)

        # Learned step embedding
        self.step_emb = nn.Embedding(H + 1, D)

        # Cross-attention: horizon attends to context
        self.cross_attn = nn.MultiheadAttention(
            D, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(D)

        # Project [B, H, D] → [B, H, Q*D]
        self.out_proj = nn.Linear(D, Q * D)

        self.Q = Q
        self.D = D

    def forward(
        self,
        current_queries: torch.Tensor,    # [B, Q, D]
        residual_queries: torch.Tensor,   # [B, K, Q, D]  K ≥ 1
        action_tokens: torch.Tensor,      # [B, H, D]
    ) -> torch.Tensor:
        """Returns future_dynamic_queries [B, H, Q, D]."""
        B, Q, D = current_queries.shape
        K = residual_queries.shape[1]
        H = action_tokens.shape[1]

        # Build context: [current_queries | residual flat]
        resid_flat = residual_queries.reshape(B, K * Q, D)         # [B, K*Q, D]
        context = torch.cat([current_queries, resid_flat], dim=1)  # [B, (K+1)*Q, D]
        context = self.context_encoder(context)                     # [B, (K+1)*Q, D]

        # Horizon tokens: action + step embedding
        step_ids = torch.arange(H, device=action_tokens.device)
        step_emb = self.step_emb(step_ids).unsqueeze(0).expand(B, -1, -1)  # [B, H, D]
        horizon = action_tokens + step_emb                          # [B, H, D]

        # Cross-attention
        attn_out, _ = self.cross_attn(query=horizon, key=context, value=context)
        horizon_out = self.cross_norm(attn_out + horizon)           # [B, H, D]

        # Expand to Q queries per step
        future_flat = self.out_proj(horizon_out)                    # [B, H, Q*D]
        future_dynamic_queries = future_flat.reshape(B, H, Q, D)   # [B, H, Q, D]
        return future_dynamic_queries


# ===========================================================================
# TokenFuser
# ===========================================================================

class TokenFuser(nn.Module):
    """Fuse future dynamic queries into current spatial tokens.

    For each horizon step h:
        fuser_mask[h, q, n] = softmax_n( q[h,q] · k[n] / sqrt(D) )
        update[h, n]        = sum_q fuser_mask[h,q,n] * W_v(q[h,q])
        z_future[h, n]      = z_current[n] + update[h, n]

    Optionally refined by n_fuser_layers of self-attention.
    """

    def __init__(self, cfg: TemporalQueryResidualConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim

        self.q_proj = nn.Linear(D, D)    # project dynamic queries to query space
        self.k_proj = nn.Linear(D, D)    # project spatial tokens to key space
        self.v_proj = nn.Linear(D, D)    # project dynamic queries to value space
        self.scale = D ** -0.5

        if cfg.n_fuser_layers > 0:
            fuser_layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=cfg.n_heads,
                dim_feedforward=cfg.ffn_dim,
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
        """
        Returns:
            future_spatial_tokens  [B, H, N, D]
            fuser_masks            [B, H, Q, N]
        """
        B, N, D = current_spatial_tokens.shape
        _, H, Q, _ = future_dynamic_queries.shape

        k = self.k_proj(current_spatial_tokens)    # [B, N, D]
        q = self.q_proj(future_dynamic_queries)    # [B, H, Q, D]
        v = self.v_proj(future_dynamic_queries)    # [B, H, Q, D]

        # Attention weights: [B, H, Q, N]
        attn = torch.einsum("bhqd,bnd->bhqn", q, k) * self.scale
        fuser_masks = torch.softmax(attn, dim=-1)  # [B, H, Q, N]

        # Weighted sum of value embeddings: [B, H, N, D]
        update = torch.einsum("bhqn,bhqd->bhnd", fuser_masks, v)

        # Add to current tokens (broadcast over H)
        future_spatial = current_spatial_tokens.unsqueeze(1) + update  # [B, H, N, D]

        if self.refine is not None:
            BH = B * H
            future_spatial = self.refine(
                future_spatial.reshape(BH, N, D)
            ).reshape(B, H, N, D)

        return future_spatial, fuser_masks


# ===========================================================================
# ActionFutureScorer
# ===========================================================================

class ActionFutureScorer(nn.Module):
    """Score action candidates from future dynamic query evolution (v4b).

    action_tokens  [B, H, D] → projected query
    future_dynamic_queries [B, H, Q, D] → cross-attended memory
    → pooled → MLP → scalar score [B]

    Higher score = model predicts this action leads to a more "dynamic" future
    consistent with the current history context.
    """

    def __init__(self, cfg: TemporalQueryResidualConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim

        self.act_query_proj = nn.Linear(D, D)

        if cfg.n_scorer_layers > 0:
            dec_layer = nn.TransformerDecoderLayer(
                d_model=D, nhead=cfg.n_heads,
                dim_feedforward=cfg.ffn_dim,
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

        # Queries from actions
        act_q = self.act_query_proj(action_tokens)   # [B, H, D]

        # Memory: flatten future dynamic queries [B, H*Q, D]
        mem = future_dynamic_queries.reshape(B, H * Q, D)

        if self.cross_decoder is not None:
            scored = self.cross_decoder(act_q, mem)  # [B, H, D]
        else:
            scored = act_q

        # Pool over H steps, project to scalar
        pooled = scored.mean(dim=1)                              # [B, D]
        score = self.score_mlp(pooled).squeeze(-1)               # [B]
        return {"ranking_score": score}


# ===========================================================================
# TemporalDynamicQueryResidualWM  (main model)
# ===========================================================================

class TemporalDynamicQueryResidualWM(nn.Module):
    """v4 Temporal Dynamic Query Residual World Model.

    Fully differentiable in pixel space; reuses PixelEncoder / PixelDecoder
    from the existing v1/v3 pipeline.

    Forward expects:
        pixels  [B, K+2+H, H_img, W_img, C]  uint8  channel-last
        actions [B, K+1+H, action_dim]

    Frame layout:
        pixels[:, 0:K]       — history frames  (K = history_length)
        pixels[:, K]         — context slot    (unused; mirrors v1/v3 frame_0)
        pixels[:, K+1]       — current frame
        pixels[:, K+2:K+2+H] — future GT frames

    Action layout:
        actions[:, 0:K+1]    — history + context actions (skipped)
        actions[:, K+1:K+1+H]— prediction horizon actions

    Checkpoint files saved:
        v4_config.json
        encoder.pt, decoder.pt, act_encoder.pt, spatial_proj.pt
        spatial_unproj.pt, act_proj.pt
        query_extractor.pt, temporal_predictor.pt
        token_fuser.pt
        action_future_scorer.pt  (if use_action_future_scorer)
    """

    def __init__(
        self,
        cfg: TemporalQueryResidualConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        D = cfg.hidden_dim
        C_enc = cfg.encoder_channels

        # Shared encoder/decoder (identical to v1/v3)
        self.encoder = PixelEncoder(cfg)
        self.decoder = PixelDecoder(cfg)

        # Action encoder (duck-typed; uses same fields as PixelResidualConfig)
        self.act_encoder = ActionEncoder(cfg)

        # Project encoder output channels → hidden_dim
        self.spatial_proj = nn.Linear(C_enc, D)
        # Project hidden_dim → encoder channels (before decoder)
        self.spatial_unproj = nn.Linear(D, C_enc)
        # Project action_emb_dim → hidden_dim
        self.act_proj = nn.Linear(cfg.action_emb_dim, D)

        # v4 sub-modules
        self.query_extractor = DynamicQueryExtractor(cfg)
        self.temporal_predictor = TemporalResidualPredictor(cfg)
        self.token_fuser = TokenFuser(cfg)

        if cfg.use_action_future_scorer:
            self.action_future_scorer: Optional[ActionFutureScorer] = ActionFutureScorer(cfg)
        else:
            self.action_future_scorer = None

        if torch_dtype is not None:
            self.to(dtype=torch_dtype)

        # Zero-init decoder output so residual starts near zero
        self._zero_init_decoder_output()

        # Action normalization ranges
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

    def _encode_frame_seq(
        self,
        images: torch.Tensor,    # [B, T, 3, H_img, W_img] float [0,1]
        device,
    ) -> torch.Tensor:           # [B, T, N, D]
        B, T, C, H_img, W_img = images.shape
        dtype = next(self.encoder.parameters()).dtype
        with self._autocast_ctx(device):
            flat = images.reshape(B * T, C, H_img, W_img).to(dtype)
            enc = self.encoder(flat)                          # [B*T, N, C_enc]
            N = enc.shape[1]
            proj = self.spatial_proj(enc)                    # [B*T, N, D]
        return proj.reshape(B, T, N, self.cfg.hidden_dim)    # [B, T, N, D]

    def _decode_future_tokens(
        self,
        future_spatial: torch.Tensor,  # [B, H, N, D]
        B: int,
        H: int,
        device,
    ) -> torch.Tensor:                 # [B, H, 3, H_img, W_img] float
        N, D = future_spatial.shape[2], future_spatial.shape[3]
        dtype = next(self.decoder.parameters()).dtype
        with self._autocast_ctx(device):
            flat = future_spatial.reshape(B * H, N, D).to(dtype)
            dec_tokens = self.spatial_unproj(flat)           # [B*H, N, C_enc]
            decoded = self.decoder(dec_tokens)               # [B*H, 3, H_img, W_img]
        H_img, W_img = decoded.shape[-2:]
        return decoded.reshape(B, H, 3, H_img, W_img).float()

    # ------------------------------------------------------------------
    # Core computation shared by forward() and rollout()
    # ------------------------------------------------------------------

    def _encode_and_query(
        self,
        history_frames: torch.Tensor,   # [B, K, 3, H_img, W_img]
        current: torch.Tensor,          # [B, 3, H_img, W_img]
        acts: torch.Tensor,             # [B, H, action_dim] normalized
        B: int,
        H: int,
        K: int,
        device,
    ) -> Dict:
        dtype = next(self.encoder.parameters()).dtype

        # Build [B, K+1, 3, H, W] sequence: history + current
        current_unsq = current.unsqueeze(1)                  # [B, 1, 3, H, W]
        if K > 0:
            all_frames = torch.cat([history_frames, current_unsq], dim=1)
        else:
            all_frames = current_unsq
        # Shape: [B, K+1, 3, H_img, W_img]

        z_seq = self._encode_frame_seq(all_frames, device)   # [B, K+1, N, D]
        current_spatial = z_seq[:, -1]                       # [B, N, D]

        # Dynamic query extraction
        dynamic_queries, dynamic_masks = self.query_extractor(z_seq)
        # [B, K+1, Q, D], [B, K+1, Q, N]

        current_queries = dynamic_queries[:, -1]             # [B, Q, D]
        if K > 0:
            # Temporal residuals between consecutive frames
            residual_queries = dynamic_queries[:, 1:] - dynamic_queries[:, :-1]
        else:
            Q = self.cfg.num_dynamic_queries
            D = self.cfg.hidden_dim
            residual_queries = torch.zeros(B, 1, Q, D, device=device, dtype=dtype)

        # Action encoding
        with self._autocast_ctx(device):
            act_emb = self.act_encoder(acts.to(dtype))       # [B, H, action_emb_dim]
            action_tokens = self.act_proj(act_emb)           # [B, H, D]

        # Temporal prediction
        future_dynamic_queries = self.temporal_predictor(
            current_queries, residual_queries, action_tokens
        )  # [B, H, Q, D]

        # Token fusion
        future_spatial, fuser_masks = self.token_fuser(
            current_spatial, future_dynamic_queries
        )  # [B, H, N, D], [B, H, Q, N]

        return {
            "z_seq": z_seq,
            "current_spatial": current_spatial,
            "dynamic_queries": dynamic_queries,
            "dynamic_masks": dynamic_masks,
            "current_queries": current_queries,
            "residual_queries": residual_queries,
            "action_tokens": action_tokens,
            "future_dynamic_queries": future_dynamic_queries,
            "future_spatial": future_spatial,
            "fuser_masks": fuser_masks,
        }

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        pixels: torch.Tensor,                              # [B, K+2+H, H, W, C] uint8
        actions: torch.Tensor,                             # [B, K+1+H, action_dim]
        negative_actions: Optional[torch.Tensor] = None,  # [B, K+1+H, action_dim]
        return_debug: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.encoder.parameters()).device
        pixels = pixels.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        # uint8 [B, T+1, H, W, C] → float [B, T+1, C, H, W]
        pixels_f = pixels.permute(0, 1, 4, 2, 3).contiguous().float() / 255.0

        K = self.cfg.history_length
        B = pixels_f.shape[0]
        T_plus_1 = pixels_f.shape[1]
        H = T_plus_1 - K - 2   # future steps

        if H < 1:
            raise ValueError(
                f"v4 forward: need at least K+3 frames, got T+1={T_plus_1} K={K} → H={H}"
            )

        history_frames = pixels_f[:, 0:K]           # [B, K, 3, H_img, W_img]
        current        = pixels_f[:, K + 1]          # [B, 3, H_img, W_img]
        future_gt      = pixels_f[:, K + 2: K + 2 + H]  # [B, H, 3, H_img, W_img]

        acts = self._normalize_actions(actions[:, K + 1: K + 1 + H])  # [B, H, action_dim]

        enc = self._encode_and_query(history_frames, current, acts, B, H, K, device)

        # Decode future images
        pred_decoded = self._decode_future_tokens(enc["future_spatial"], B, H, device)
        residual_pred = self._residual_from_raw(pred_decoded)
        pred_future = (current.unsqueeze(1) + residual_pred).clamp(0.0, 1.0)

        # ---- Loss computation ----
        future_gt_f = future_gt.float()

        L_image = F.mse_loss(pred_future, future_gt_f.detach())

        # Dynamic masks per horizon step
        dyn_masks = []
        for h in range(H):
            dm = compute_dynamic_mask(
                current, future_gt_f[:, h],
                self.cfg.dynamic_threshold, self.cfg.dynamic_dilate_kernel,
            )
            dyn_masks.append(dm)
        dyn_mask_seq = torch.stack(dyn_masks, dim=1)   # [B, H, 1, H_img, W_img]

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

        # Ranking loss (v4b) — multi-negative InfoNCE with detached future queries
        #
        # H3 fix: future_dynamic_queries are detached before the scorer so that
        # L_rank does not corrupt the reconstruction-focused query predictor.
        # Gradients from L_rank flow only through the scorer and act_encoder/act_proj.
        #
        # Loss: InfoNCE with in-batch negatives.
        # For each sample i, denominator = exp(pos_i/τ) + Σ_j exp(neg_j/τ)
        # where neg_j are all B negative scores in the batch.
        L_rank = torch.tensor(0.0, device=device, dtype=torch.float32)
        ranking_score: Optional[torch.Tensor] = None
        if self.action_future_scorer is not None:
            # Positive path: detach future_q to isolate scorer from reconstruction grad
            scorer_out = self.action_future_scorer(
                enc["action_tokens"],
                enc["future_dynamic_queries"].detach(),
            )
            ranking_score = scorer_out["ranking_score"]   # [B]

            neg_acts = self._get_negative_actions(negative_actions, actions, K, H, device)
            if neg_acts is not None:
                dtype_model = next(self.encoder.parameters()).dtype
                # Negative path: compute without gradients; scorer still learns
                # because its own parameters have requires_grad=True
                with torch.no_grad():
                    with self._autocast_ctx(device):
                        neg_emb = self.act_encoder(neg_acts.to(dtype_model))
                        neg_action_tokens = self.act_proj(neg_emb)
                    neg_future_q = self.temporal_predictor(
                        enc["current_queries"], enc["residual_queries"], neg_action_tokens
                    )
                neg_score = self.action_future_scorer(
                    neg_action_tokens, neg_future_q
                )["ranking_score"]   # [B]

                # Multi-negative InfoNCE:
                # logits [B, B+1]: col 0 = positive, cols 1..B = all batch negatives
                temperature = self.cfg.rank_temperature
                B_rank = ranking_score.shape[0]
                if B_rank > 1:
                    logits = torch.cat([
                        ranking_score.unsqueeze(1),              # [B, 1]
                        neg_score.unsqueeze(0).expand(B_rank, -1),  # [B, B]
                    ], dim=1) / temperature                      # [B, B+1]
                    labels = torch.zeros(B_rank, dtype=torch.long, device=device)
                    L_rank = F.cross_entropy(logits, labels)
                else:
                    # B=1 fallback: binary logistic loss
                    L_rank = F.softplus((neg_score - ranking_score) / temperature).mean()

        # Mask regularisation
        # L_entropy  : encourage concentration (low entropy = peaked mask)
        # L_diversity: penalize overlap between different query masks
        L_entropy  = _mask_entropy(enc["dynamic_masks"])  + _mask_entropy(enc["fuser_masks"])
        L_diversity = _mask_diversity(enc["dynamic_masks"]) + _mask_diversity(enc["fuser_masks"])

        loss = (
            self.cfg.lambda_image     * L_image
            + self.cfg.lambda_dynamic * L_dynamic
            + self.cfg.lambda_static  * L_static
            + self.cfg.lambda_query   * L_query
            + self.cfg.lambda_rank    * L_rank
            + self.cfg.lambda_sparse  * L_entropy
            + self.cfg.lambda_diversity * L_diversity
        )

        # copy-current collapse監視: pred_future と current の比較
        copy_mse = F.mse_loss(
            current.unsqueeze(1).expand_as(future_gt_f),
            future_gt_f.detach(),
        )
        mse_over_copy = (L_image / copy_mse.clamp(min=1e-12)).detach()

        out: Dict[str, torch.Tensor] = {
            "loss":            loss,
            "loss_image":      L_image.detach(),
            "loss_dynamic":    L_dynamic.detach(),
            "loss_static":     L_static.detach(),
            "loss_query":      L_query.detach(),
            "loss_rank":       L_rank.detach() if isinstance(L_rank, torch.Tensor) else torch.tensor(L_rank),
            "loss_entropy":    L_entropy.detach(),
            "loss_diversity":  L_diversity.detach(),
            "copy_current_mse": copy_mse.detach(),
            "mse_over_copy":   mse_over_copy,
        }

        if ranking_score is not None:
            out["ranking_score"] = ranking_score

        # zero-action baseline: score with all-zero actions (診断用、lossには使わない)
        # It adds an extra temporal predictor + scorer forward pass per step, so large sweeps
        # can disable it with DISABLE_ZERO_ACTION_DIAGNOSTIC=1.
        if (
            self.action_future_scorer is not None
            and os.environ.get("DISABLE_ZERO_ACTION_DIAGNOSTIC", "0") not in ("1", "true", "TRUE", "yes", "YES")
        ):
            dtype_model = next(self.encoder.parameters()).dtype
            with torch.no_grad():
                zero_emb = self.act_encoder(torch.zeros_like(acts).to(dtype_model))
                zero_tokens = self.act_proj(zero_emb)
            zero_future_q = self.temporal_predictor(
                enc["current_queries"], enc["residual_queries"], zero_tokens
            )
            zero_score = self.action_future_scorer(zero_tokens, zero_future_q)["ranking_score"]
            out["score_zero_action"] = zero_score.mean().detach()

        if return_debug:
            out.update({
                "pred_future":            pred_future.detach(),
                "future_spatial_tokens":  enc["future_spatial"].detach(),
                "future_dynamic_queries": enc["future_dynamic_queries"].detach(),
                "dynamic_masks":          enc["dynamic_masks"].detach(),
                "fuser_masks":            enc["fuser_masks"].detach(),
            })

        return out

    def _get_negative_actions(
        self,
        negative_actions: Optional[torch.Tensor],
        actions: torch.Tensor,
        K: int,
        H: int,
        device,
    ) -> Optional[torch.Tensor]:
        """Return normalized negative actions [B, H, action_dim] or None.

        V4Collator constructs all negative types (same_task_other_window,
        temporal_permutation, zero_random) and passes them via negative_actions.
        This method simply normalizes and slices the provided tensor.

        Fallback (B=1 or training without collator): temporal permutation of
        the correct horizon actions.
        """
        if negative_actions is not None:
            neg = negative_actions.to(device, non_blocking=True)
            return self._normalize_actions(neg[:, K + 1: K + 1 + H])

        # B=1 fallback: temporal permutation
        acts_raw = actions[:, K + 1: K + 1 + H]   # [B, H, action_dim]
        H_len = acts_raw.shape[1]
        if H_len < 2:
            return None
        perm = torch.randperm(H_len, device=device)
        if (perm == torch.arange(H_len, device=device)).all():
            perm = torch.roll(perm, 1)
        return self._normalize_actions(acts_raw[:, perm, :])

    def _query_future_loss(
        self,
        future_gt_f: torch.Tensor,      # [B, H, 3, H_img, W_img]
        current_mask: torch.Tensor,     # [B, Q, N] shared mask from z_t
        pred_future_queries: torch.Tensor,  # [B, H, Q, D]
        device,
    ) -> torch.Tensor:
        """MSE between predicted and GT-derived future dynamic queries."""
        B, H, C, H_img, W_img = future_gt_f.shape
        D = self.cfg.hidden_dim

        z_future_seq = self._encode_frame_seq(future_gt_f, device)  # [B, H, N, D]

        # Apply current shared mask to derive GT future queries
        mask_h = current_mask.unsqueeze(1).expand(-1, H, -1, -1)   # [B, H, Q, N]
        gt_future_queries = torch.einsum("bhqn,bhnd->bhqd", mask_h, z_future_seq)

        return F.mse_loss(pred_future_queries, gt_future_queries.detach())

    # ------------------------------------------------------------------
    # Rollout (no-grad, for evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        pixels: torch.Tensor,    # [B, K+2+H, H_img, W_img, C] uint8
        actions: torch.Tensor,   # [B, K+1+H, action_dim]
        horizon: int = -1,
    ) -> Dict:
        """Predict future images without computing losses.

        Returns:
            pred_future             [B, H, 3, H_img, W_img]  float [0,1]
            current_image           [B, 3, H_img, W_img]
            future_gt               [B, H, 3, H_img, W_img]
            residual_pred           [B, H, 3, H_img, W_img]
            dynamic_mask            [B, H, 1, H_img, W_img]  binary
            fuser_masks             [B, H, Q, N]
            dynamic_masks           [B, K+1, Q, N]
            future_dynamic_queries  [B, H, Q, D]
            ranking_score           [B] or None
        """
        device = next(self.encoder.parameters()).device
        pixels_f = pixels.to(device).permute(0, 1, 4, 2, 3).float() / 255.0
        actions = actions.to(device)

        K = self.cfg.history_length
        B = pixels_f.shape[0]
        T_plus_1 = pixels_f.shape[1]
        H_full = T_plus_1 - K - 2
        H = H_full if horizon < 0 else min(horizon, H_full)

        history_frames = pixels_f[:, 0:K]
        current        = pixels_f[:, K + 1]
        future_gt      = pixels_f[:, K + 2: K + 2 + H]

        acts = self._normalize_actions(actions[:, K + 1: K + 1 + H])

        enc = self._encode_and_query(history_frames, current, acts, B, H, K, device)

        pred_decoded = self._decode_future_tokens(enc["future_spatial"], B, H, device)
        residual_pred = self._residual_from_raw(pred_decoded)
        pred_future = (current.unsqueeze(1) + residual_pred).clamp(0.0, 1.0)

        # GT dynamic masks for visualization
        dyn_masks = []
        for h in range(H):
            dyn_masks.append(compute_dynamic_mask(
                current, future_gt[:, h],
                self.cfg.dynamic_threshold, self.cfg.dynamic_dilate_kernel,
            ))
        dynamic_mask = torch.stack(dyn_masks, dim=1)   # [B, H, 1, H_img, W_img]

        ranking_score = None
        if self.action_future_scorer is not None:
            scorer_out = self.action_future_scorer(
                enc["action_tokens"], enc["future_dynamic_queries"]
            )
            ranking_score = scorer_out["ranking_score"]

        return {
            "pred_future":            pred_future,
            "current_image":          current,
            "future_gt":              future_gt,
            "residual_pred":          residual_pred,
            "dynamic_mask":           dynamic_mask,
            "fuser_masks":            enc["fuser_masks"],
            "dynamic_masks":          enc["dynamic_masks"],
            "future_dynamic_queries": enc["future_dynamic_queries"],
            "ranking_score":          ranking_score,
            # v1/v3 compat keys (None for v4)
            "write_mask":             None,
            "write_logits":           None,
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        import dataclasses
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

        with open(os.path.join(save_directory, "v4_config.json"), "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=2)

    @classmethod
    def load_pretrained(
        cls,
        save_directory: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "TemporalDynamicQueryResidualWM":
        cfg_path = os.path.join(save_directory, "v4_config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = TemporalQueryResidualConfig(**cfg_dict)
        model = cls(cfg, torch_dtype=torch_dtype)

        def _load(module, fname):
            path = os.path.join(save_directory, fname)
            if os.path.exists(path):
                state = torch.load(path, map_location="cpu")
                module.load_state_dict(state, strict=True)

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
