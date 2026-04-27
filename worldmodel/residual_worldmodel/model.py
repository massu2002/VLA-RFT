"""Latent Residual World Model.

Two residual target modes (controlled by ResidualWorldModelConfig.residual_target_mode):

  adjacent_delta  (baseline)
    Predict step-by-step adjacent differences in FSQ embedding space.
    Target: gt_delta[t] = dyn_emb[t+1] - dyn_emb[t]  (zero-anchored at t=0)
    Predictor input: (prev_dyn_emb, action) — ResidualPredictor.

  current_anchor_ctx  (main)
    Use frame_1's dynamic latent (z_curr) as anchor, conditioned on the static
    context tokens (ctx_tokens) from frame_0.
    Target: gt_cum_delta[h] = z_future[h] - z_curr  for h = 0..T-2
    Predictor: anchor_token = emb_proj(z_curr) + ctx_proj(ctx_summary),
               then causal transformer over [anchor | future_actions].
    Reconstruction: pred_latent[h] = z_curr + pred_cum_delta[h].

Extending
---------
- ROI / progress-value / auxiliary heads: add inside _forward_current_anchor_ctx().
- Reconstruction loss: set config.recon_loss_weight > 0.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ivideogpt.ctx_tokenizer.compressive_vq_model import CompressiveVQModelFSQ

from .config import ResidualWorldModelConfig
from .losses import (
    compute_remaining_steps_labels,
    compute_goal_distance_labels,
    compute_motion_proxy_labels,
)


# ---------------------------------------------------------------------------
# Utility: micro-batched tokenizer call
# ---------------------------------------------------------------------------

def _batch_tokenize(visual_tokenizer, pixels, micro_batch_size: Optional[int]):
    """Call visual_tokenizer.tokenize() with optional micro-batching."""
    if micro_batch_size is None:
        return visual_tokenizer.tokenize(pixels)

    B = pixels.shape[0]
    ctx_list, dyn_list = [], []
    for i in range(0, B, micro_batch_size):
        c, d = visual_tokenizer.tokenize(pixels[i : i + micro_batch_size])
        ctx_list.append(c)
        dyn_list.append(d)
    return torch.cat(ctx_list, dim=0), torch.cat(dyn_list, dim=0)


# ---------------------------------------------------------------------------
# Predictor — adjacent_delta baseline
# ---------------------------------------------------------------------------

class ResidualPredictor(nn.Module):
    """Causal transformer for adjacent-delta prediction (baseline).

    At step t the input is (prev_dyn_emb, action[t]).  Output is the predicted
    adjacent delta: pred_delta[t] ≈ dyn_emb[t] - dyn_emb[t-1].

    Sequence length = T (same as the number of dynamic frames).
    """

    def __init__(self, cfg: ResidualWorldModelConfig) -> None:
        super().__init__()
        flat_dim = cfg.flat_dyn_dim

        self.emb_proj = nn.Linear(flat_dim, cfg.hidden_dim)
        self.act_proj = nn.Linear(cfg.action_dim, cfg.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.out_proj = nn.Linear(cfg.hidden_dim, flat_dim)

    def forward(
        self,
        dyn_prev: torch.Tensor,   # [B, T, flat_dyn_dim]
        actions: torch.Tensor,    # [B, T, action_dim]
    ) -> torch.Tensor:
        """Returns pred_delta: [B, T, flat_dyn_dim]."""
        T = actions.shape[1]
        x = self.emb_proj(dyn_prev) + self.act_proj(actions)  # [B, T, hidden_dim]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device, dtype=x.dtype
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.out_proj(x)  # [B, T, flat_dyn_dim]


# ---------------------------------------------------------------------------
# Predictor — current_anchor_ctx (main)
# ---------------------------------------------------------------------------

class CurrentAnchorCtxPredictor(nn.Module):
    """Causal transformer for current-anchor cumulative residual prediction.

    Conditions on the current dynamic latent (z_curr), the static context
    tokens from frame_0 (ctx_summary), and optionally a temporal residual
    history summary (res_summary).

    Anchor token construction:
        anchor_tok = emb_proj(z_curr) + ctx_proj(ctx_summary) [+ res_proj(res_summary)]
    where ctx_summary = mean-pooled FSQ embeddings of ctx_tokens  [B, D_ctx].

    Sequence layout (length = H+1):
        token_0   = anchor_tok                  ← z_curr + ctx [+ history] context
        token_h   = act_proj(action_{h-1})      ← future action (h = 1..H)

    Output:
        pred_cum_delta = out_proj(output[:, 1:, :])  [B, H, flat_dim]

    Causality: position h attends to token_0 (anchor) and tokens 1..h-1 (actions).
    """

    def __init__(self, cfg: ResidualWorldModelConfig) -> None:
        super().__init__()
        flat_dim = cfg.flat_dyn_dim

        self.emb_proj = nn.Linear(flat_dim, cfg.hidden_dim)        # z_curr → hidden
        self.ctx_proj = nn.Linear(cfg.ctx_dim, cfg.hidden_dim)     # ctx_summary → hidden
        self.act_proj = nn.Linear(cfg.action_dim, cfg.hidden_dim)  # action → hidden

        # Optional temporal residual history projection (enabled when history_len > 0)
        self.use_res_history = cfg.residual_history_len > 0
        if self.use_res_history:
            self.res_proj = nn.Linear(flat_dim, cfg.hidden_dim)    # res_summary → hidden

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.out_proj = nn.Linear(cfg.hidden_dim, flat_dim)

    def forward(
        self,
        z_curr: torch.Tensor,                        # [B, flat_dyn_dim]
        ctx_summary: torch.Tensor,                   # [B, ctx_dim]
        actions: torch.Tensor,                       # [B, H, action_dim]
        res_summary: torch.Tensor | None = None,     # [B, flat_dyn_dim] or None
    ) -> torch.Tensor:
        """Returns pred_cum_delta: [B, H, flat_dyn_dim]."""
        H = actions.shape[1]

        # Anchor: z_curr + ctx context, optionally + residual history
        anchor_hidden = self.emb_proj(z_curr) + self.ctx_proj(ctx_summary)
        if self.use_res_history and res_summary is not None:
            anchor_hidden = anchor_hidden + self.res_proj(res_summary)
        anchor_tok = anchor_hidden.unsqueeze(1)      # [B, 1, hidden_dim]

        action_toks = self.act_proj(actions)         # [B, H, hidden_dim]

        x = torch.cat([anchor_tok, action_toks], dim=1)  # [B, H+1, hidden_dim]

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            H + 1, device=x.device, dtype=x.dtype
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        # Predictions from positions 1..H (skip anchor output token)
        return self.out_proj(x[:, 1:, :])  # [B, H, flat_dyn_dim]


# ---------------------------------------------------------------------------
# Reward-aligned prediction heads
# ---------------------------------------------------------------------------

def _make_head_mlp(in_dim: int, hidden_dim: int, activation: str) -> nn.Sequential:
    """2-layer MLP regression head: Linear → activation → Linear → scalar."""
    act = nn.GELU() if activation == "gelu" else nn.ReLU()
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        act,
        nn.Linear(hidden_dim, 1),
    )


class RewardAlignedHeads(nn.Module):
    """2-layer MLP regression heads projecting predicted latents to task-relevant scalars.

    Applied to pred_latent_seq [B, H, flat_dim] during training.

    Heads:
        progress_head      — predicts normalised remaining steps to episode success [0, 1]
        success_head       — predicts latent distance to goal image [0, 1]
        reward_proxy_head  — predicts step-wise motion magnitude proxy (optional)

    Architecture per head: Linear(flat_dim, hidden_dim) → activation → Linear(hidden_dim, 1)
    No sigmoid — raw regression to match the [0, 1] target range.
    """

    def __init__(self, cfg: ResidualWorldModelConfig) -> None:
        super().__init__()
        flat_dim   = cfg.flat_dyn_dim
        hidden_dim = cfg.reward_head_hidden_dim
        activation = cfg.reward_head_activation
        self.progress_head = _make_head_mlp(flat_dim, hidden_dim, activation)
        self.success_head  = _make_head_mlp(flat_dim, hidden_dim, activation)
        self.use_reward_proxy_head = cfg.use_reward_proxy_head
        if cfg.use_reward_proxy_head:
            self.reward_proxy_head = _make_head_mlp(flat_dim, hidden_dim, activation)

    def forward(self, latents: torch.Tensor) -> dict:
        """
        Args:
            latents: [B, H, flat_dim]
        Returns:
            dict with 'progress' [B, H], 'success' [B, H],
            and optionally 'reward_proxy' [B, H].
        """
        out = {
            "progress": self.progress_head(latents).squeeze(-1),   # [B, H]
            "success":  self.success_head(latents).squeeze(-1),    # [B, H]
        }
        if self.use_reward_proxy_head:
            out["reward_proxy"] = self.reward_proxy_head(latents).squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LatentResidualWorldModel(nn.Module):
    """World model that predicts future video as latent residuals.

    residual_target_mode = "adjacent_delta"  (baseline)
        Uses ResidualPredictor.  Predicts adjacent diffs zero-anchored at t=0.

    residual_target_mode = "current_anchor_ctx"  (main)
        Uses CurrentAnchorCtxPredictor.
        z_curr = dyn_emb of frame_1; ctx_summary = mean-pooled ctx_tokens of frame_0.
        Predicts cumulative deltas: pred_cum_delta[h] ≈ z_future[h] - z_curr.
        Reconstruction: pred_latent[h] = z_curr + pred_cum_delta[h].
    """

    def __init__(
        self,
        visual_tokenizer_path: str,
        cfg: ResidualWorldModelConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Frozen visual tokenizer
        self.visual_tokenizer = CompressiveVQModelFSQ.from_pretrained(
            visual_tokenizer_path, torch_dtype=torch.float32
        )
        self.visual_tokenizer.requires_grad_(False)
        self.visual_tokenizer.eval()

        # Infer token/embedding dimensions from the loaded tokenizer
        cfg.n_dyn_tokens = 64  # 8×8 patches (patch_size=4 on 32×32 feature map)
        cfg.dyn_token_dim = len(self.visual_tokenizer.dyn_fsq_levels)  # 5 for default
        # ctx_dim: float embedding dim per context token after FSQ dequantization
        cfg.ctx_dim = len(self.visual_tokenizer.vq_fsq_levels)

        # Trainable predictor — selected by residual_target_mode
        if cfg.residual_target_mode == "current_anchor_ctx":
            self.predictor = CurrentAnchorCtxPredictor(cfg)
        else:
            # "adjacent_delta" baseline (default)
            self.predictor = ResidualPredictor(cfg)

        if torch_dtype is not None:
            self.predictor = self.predictor.to(dtype=torch_dtype)

        # Optional reward-aligned heads (current_anchor_ctx only)
        self.reward_heads: RewardAlignedHeads | None = None
        if cfg.use_reward_aligned_loss and cfg.residual_target_mode == "current_anchor_ctx":
            self.reward_heads = RewardAlignedHeads(cfg)
            if torch_dtype is not None:
                self.reward_heads = self.reward_heads.to(dtype=torch_dtype)

        # Action normalization ranges — loaded once onto CPU; moved to device in forward
        self.register_buffer(
            "action_ranges",
            torch.load(cfg.action_ranges_path, map_location="cpu"),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Context / dynamic encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_ctx_from_frame(self, frame_f: torch.Tensor) -> torch.Tensor:
        """Encode a single frame as ctx_tokens [B, 1, N_ctx].

        Args:
            frame_f: [B, C, H, W] float [0, 1].
        Returns:
            ctx_tokens [B, 1, N_ctx] int indices.
        """
        B = frame_f.shape[0]
        device = "cuda" if frame_f.is_cuda else "cpu"
        # tokenize requires a 2-frame sequence (context + ≥1 future).
        # Use the same frame as a dummy future; only ctx_tokens is extracted.
        seq = frame_f.unsqueeze(1).expand(-1, 2, -1, -1, -1)  # [B, 2, C, H, W]
        with self._autocast_ctx(device):
            ctx_tokens, _ = _batch_tokenize(
                self.visual_tokenizer, seq, self.cfg.tokenizer_micro_batch_size
            )
        return ctx_tokens  # [B, 1, N_ctx]

    @torch.no_grad()
    def _encode_dyn_from_frame(self, frame_f: torch.Tensor) -> torch.Tensor:
        """Encode a single frame as a dynamic flat latent [B, flat_dim].

        Args:
            frame_f: [B, C, H, W] float [0, 1].
        Returns:
            dyn_flat [B, flat_dim].
        """
        B = frame_f.shape[0]
        device = "cuda" if frame_f.is_cuda else "cpu"
        # tokenize requires context_length=1 + ≥1 future; use the same frame for context.
        seq = frame_f.unsqueeze(1).expand(-1, 2, -1, -1, -1)  # [B, 2, C, H, W]
        with self._autocast_ctx(device):
            _, dyn_tokens = _batch_tokenize(
                self.visual_tokenizer, seq, self.cfg.tokenizer_micro_batch_size
            )
        # dyn_tokens: [B, 1, N_dyn] (only the single future frame)
        dyn_emb = self._dequantize(dyn_tokens)  # [B, 1, N_dyn, D]
        flat_dim = self.cfg.n_dyn_tokens * dyn_emb.shape[-1]
        return dyn_emb.reshape(B, flat_dim).float()  # [B, flat_dim]

    def _autocast_ctx(self, device: str):
        dtype_name = self.cfg.autocast_dtype
        if device == "cpu":
            return torch.autocast(device_type="cpu", dtype=torch.float32)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype_name, torch.float32)
        return torch.autocast(device_type=device, dtype=dtype)

    @torch.no_grad()
    def _encode(self, pixels: torch.Tensor):
        """Encode pixels; return only dyn_tokens (adjacent_delta path)."""
        device = "cuda" if pixels.is_cuda else "cpu"
        with self._autocast_ctx(device):
            _ctx, dyn_tokens = _batch_tokenize(
                self.visual_tokenizer, pixels, self.cfg.tokenizer_micro_batch_size
            )
        return dyn_tokens  # [B, T, N_dyn]

    @torch.no_grad()
    def _encode_both(self, pixels: torch.Tensor):
        """Encode pixels; return (ctx_tokens, dyn_tokens) for current_anchor_ctx path."""
        device = "cuda" if pixels.is_cuda else "cpu"
        with self._autocast_ctx(device):
            ctx_tokens, dyn_tokens = _batch_tokenize(
                self.visual_tokenizer, pixels, self.cfg.tokenizer_micro_batch_size
            )
        return ctx_tokens, dyn_tokens  # [B, N_ctx, ctx_dim], [B, T, N_dyn]

    @torch.no_grad()
    def _dequantize_ctx(self, ctx_tokens: torch.Tensor) -> torch.Tensor:
        """ctx_tokens [B, N_frames, N_ctx] int → ctx_summary [B, D_ctx] float.

        Dequantizes context token indices to float FSQ embeddings via
        self.quantize.indices_to_codes(), then mean-pools over all spatial tokens
        to produce a single compact context vector per sample.
        """
        B = ctx_tokens.shape[0]
        flat = ctx_tokens.reshape(B, -1)  # [B, N_frames * N_ctx]
        emb = self.visual_tokenizer.quantize.indices_to_codes(flat)  # [B, N_total, D_ctx]
        return emb.mean(dim=1).float()  # [B, D_ctx]

    @torch.no_grad()
    def _dequantize(self, dyn_tokens: torch.Tensor) -> torch.Tensor:
        """Convert discrete token indices to continuous FSQ embeddings.

        Returns dyn_emb: [B, T, N_dyn, D_fsq]
        """
        B, T, N = dyn_tokens.shape
        flat = dyn_tokens.reshape(B * T, N)
        emb = self.visual_tokenizer.dynamics_quantize.indices_to_codes(flat)
        return emb.reshape(B, T, N, -1)

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize continuous actions to [0, 1] using precomputed ranges."""
        ar = self.action_ranges.to(actions.device)
        min_v, max_v = ar[:, 0], ar[:, 1]
        return torch.clamp((actions - min_v) / (max_v - min_v + 1e-8), 0.0, 1.0)

    def _dyn_tokens_to_flat(self, dyn_tokens: torch.Tensor) -> torch.Tensor:
        """dyn_tokens [B,T,N] → dyn_flat [B,T,flat_dim] (float)."""
        dyn_emb = self._dequantize(dyn_tokens)  # [B, T, N, D]
        B, T = dyn_tokens.shape[:2]
        flat_dim = self.cfg.n_dyn_tokens * dyn_emb.shape[-1]
        return dyn_emb.reshape(B, T, flat_dim).float()

    # ------------------------------------------------------------------
    # HuggingFace Trainer compatibility
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, *args, **kwargs):
        pass

    def enable_input_require_grads(self):
        self.predictor.requires_grad_(True)

    # ------------------------------------------------------------------
    # Forward — dispatcher
    # ------------------------------------------------------------------

    def forward(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        episode_init_pixels: torch.Tensor | None = None,  # [B, H, W, C] uint8 — episode frame_0
        episode_goal_pixels: torch.Tensor | None = None,  # [B, H, W, C] uint8 — episode last frame
        window_start: torch.Tensor | None = None,         # [B] long
        episode_length: torch.Tensor | None = None,       # [B] long
        **kwargs,
    ) -> dict:
        """Dispatch to the appropriate forward based on residual_target_mode."""
        if not torch.is_tensor(pixels):
            pixels = torch.as_tensor(pixels)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)

        device = next(self.predictor.parameters()).device
        pixels = pixels.to(device=device, non_blocking=True)
        actions = actions.to(device=device, non_blocking=True)

        # channel-last uint8 → channel-first float [0, 1]
        pixels_f = pixels.permute(0, 1, 4, 2, 3).contiguous().float() / 255.0

        if self.cfg.residual_target_mode == "current_anchor_ctx":
            # Preprocess optional episode-level images
            ep_init_f = ep_goal_f = None
            if episode_init_pixels is not None:
                ep_init_f = episode_init_pixels.to(device=device, non_blocking=True)
                ep_init_f = ep_init_f.permute(0, 3, 1, 2).contiguous().float() / 255.0
            if episode_goal_pixels is not None:
                ep_goal_f = episode_goal_pixels.to(device=device, non_blocking=True)
                ep_goal_f = ep_goal_f.permute(0, 3, 1, 2).contiguous().float() / 255.0
            ws = window_start.to(device=device) if window_start is not None else None
            el = episode_length.to(device=device) if episode_length is not None else None
            return self._forward_current_anchor_ctx(
                pixels_f, actions, device,
                ep_init_f=ep_init_f, ep_goal_f=ep_goal_f,
                window_start=ws, episode_length=el,
            )
        else:
            return self._forward_adjacent_delta(pixels_f, actions, device)

    # ------------------------------------------------------------------
    # Forward — adjacent_delta baseline
    # ------------------------------------------------------------------

    def _forward_adjacent_delta(
        self,
        pixels_f: torch.Tensor,  # [B, T+1, C, H, W] float [0,1]
        actions: torch.Tensor,   # [B, T, action_dim]
        device,
    ) -> dict:
        """Baseline: predict zero-anchored adjacent deltas.

        gt_delta[t] = dyn_emb[t+1] - dyn_emb[t]  (dyn_emb[-1] = 0)
        pred via ResidualPredictor(dyn_prev, actions).
        """
        B, T_plus_1 = pixels_f.shape[:2]
        T = T_plus_1 - 1

        dyn_tokens = self._encode(pixels_f)             # [B, T, N]
        dyn_flat = self._dyn_tokens_to_flat(dyn_tokens) # [B, T, flat_dim]
        flat_dim = dyn_flat.shape[-1]

        # Zero anchor prepended to create the "previous embedding" sequence
        zero = torch.zeros(B, 1, flat_dim, device=device, dtype=dyn_flat.dtype)
        dyn_all = torch.cat([zero, dyn_flat], dim=1)  # [B, T+1, flat_dim]

        gt_delta = dyn_all[:, 1:] - dyn_all[:, :-1]   # [B, T, flat_dim]
        dyn_prev = dyn_all[:, :T]                       # [B, T, flat_dim]

        acts_norm = self._normalize_actions(actions)    # [B, T, action_dim]

        pred_dtype = next(self.predictor.parameters()).dtype
        pred_delta = self.predictor(
            dyn_prev.to(pred_dtype), acts_norm.to(pred_dtype)
        )  # [B, T, flat_dim]

        loss = F.mse_loss(pred_delta, gt_delta.detach().to(pred_delta.dtype))

        if self.cfg.recon_loss_weight > 0.0:
            recon_loss = self._reconstruction_loss_adj(
                dyn_tokens, dyn_all, pred_delta, B, T, device
            )
            loss = loss + self.cfg.recon_loss_weight * recon_loss

        return {"loss": loss}

    # ------------------------------------------------------------------
    # Forward — current_anchor_ctx (main)
    # ------------------------------------------------------------------

    def _forward_current_anchor_ctx(
        self,
        pixels_f: torch.Tensor,              # [B, T+1, C, H, W] float [0,1]
        actions: torch.Tensor,               # [B, T, action_dim]
        device,
        ep_init_f: torch.Tensor | None = None,   # [B, C, H, W] episode initial frame
        ep_goal_f: torch.Tensor | None = None,   # [B, C, H, W] episode goal frame
        window_start: torch.Tensor | None = None, # [B] long
        episode_length: torch.Tensor | None = None, # [B] long
    ) -> dict:
        """Current-anchor cumulative residual prediction, conditioned on ctx_tokens.

        Frame layout (segment_length = T+1):
            frame_0        — segment start (used as ctx when ctx_source_mode='segment_initial')
            frame_1        — z_curr: dynamic anchor
            frame_2 .. T   — z_future: prediction targets (T-1 frames)

        ctx_tokens source is controlled by cfg.ctx_source_mode:
            'segment_initial'       — ctx from frame_0 of the current segment (default)
            'episode_initial_image' — ctx from ep_init_f (episode's very first frame)

        Targets:
            gt_cum_delta[h] = z_future[h] - z_curr   for h = 0..T-2

        Requires segment_length >= 3 (T >= 2).
        """
        B, T_plus_1 = pixels_f.shape[:2]
        T = T_plus_1 - 1

        if T < 2:
            raise ValueError(
                f"current_anchor_ctx mode requires segment_length >= 3 (T >= 2), got T={T}."
            )

        # --- Encode segment ---
        seg_ctx_tokens, dyn_tokens = self._encode_both(pixels_f)
        # seg_ctx_tokens: [B, 1, N_ctx]  — from segment frame_0
        # dyn_tokens:     [B, T, N_dyn]  — from frames 1..T

        # --- ctx_tokens source ---
        # When ctx_source_mode='episode_initial_image', ctx_tokens always come from
        # the episode's very first frame (ep_init_f), giving a stable global context
        # regardless of where the segment window falls in the episode.
        if self.cfg.ctx_source_mode == "episode_initial_image" and ep_init_f is not None:
            ctx_tokens = self._encode_ctx_from_frame(ep_init_f)  # [B, 1, N_ctx]
        else:
            ctx_tokens = seg_ctx_tokens  # [B, 1, N_ctx] — segment frame_0

        dyn_flat = self._dyn_tokens_to_flat(dyn_tokens)  # [B, T, flat_dim]

        z_curr   = dyn_flat[:, 0, :]   # [B, flat_dim]   — frame_1 anchor
        z_future = dyn_flat[:, 1:, :]  # [B, T-1, flat_dim] — frame_2..T targets

        H = T - 1  # number of future steps

        # Cumulative residuals from current anchor (ground truth)
        gt_cum_delta = z_future - z_curr.unsqueeze(1)  # [B, H, flat_dim]

        # Dequantize integer ctx indices → float FSQ embeddings, then mean-pool
        ctx_summary = self._dequantize_ctx(ctx_tokens)  # [B, D_ctx]

        # Actions that transition from frame_1 onward (skip action_0)
        acts_future_norm = self._normalize_actions(
            actions[:, self.cfg.action_start_offset:self.cfg.action_start_offset + H, :]
        )  # [B, H, action_dim]

        # --- Temporal residual history ---
        # Disabled by default (residual_history_len=0): within-window deltas leak future
        # GT frames into the predictor (see _compute_res_summary docstring).
        res_summary = self._compute_res_summary(dyn_flat, device)  # None when disabled

        pred_dtype = next(self.predictor.parameters()).dtype
        pred_cum_delta = self.predictor(
            z_curr.to(pred_dtype),
            ctx_summary.to(pred_dtype),
            acts_future_norm.to(pred_dtype),
            res_summary.to(pred_dtype) if res_summary is not None else None,
        )  # [B, H, flat_dim]

        # pred_latent_seq[h] = z_curr + pred_cum_delta[h] ≈ embed(frame_{h+2})
        pred_latent_seq = (
            z_curr.unsqueeze(1).to(pred_cum_delta.dtype) + pred_cum_delta
        )  # [B, H, flat_dim]

        # --- Goal latent (for success_head) ---
        goal_latent: torch.Tensor | None = None
        if (self.cfg.use_reward_aligned_loss
                and self.reward_heads is not None
                and self.cfg.success_target_mode == "goal_image_distance"
                and ep_goal_f is not None):
            goal_latent = self._encode_dyn_from_frame(ep_goal_f)  # [B, flat_dim]

        # --- Loss computation ---
        if self.cfg.use_reward_aligned_loss and self.reward_heads is not None:
            loss, loss_components = self._reward_aligned_loss(
                pred_cum_delta, gt_cum_delta, pred_latent_seq,
                z_curr, z_future, dyn_tokens, B, T, device,
                window_start=window_start,
                episode_length=episode_length,
                goal_latent=goal_latent,
            )
        else:
            # Original behaviour: cumulative residual MSE as primary loss
            loss = F.mse_loss(pred_cum_delta, gt_cum_delta.detach().to(pred_cum_delta.dtype))
            loss_components = {}

        if self.cfg.recon_loss_weight > 0.0:
            recon_loss = self._reconstruction_loss_ca(
                dyn_tokens[:, 1:], pred_latent_seq, B, H, device
            )
            loss = loss + self.cfg.recon_loss_weight * recon_loss

        return {"loss": loss, **loss_components}

    # ------------------------------------------------------------------
    # Temporal residual history helper
    # ------------------------------------------------------------------

    def _compute_res_summary(
        self,
        dyn_flat: torch.Tensor,  # [B, T, flat_dim] — frames 1..T
        device,
    ) -> torch.Tensor | None:
        """Mean-pool adjacent deltas from the first residual_history_len pairs.

        CURRENTLY DISABLED (residual_history_len == 0 by default).

        Why disabled: dyn_flat[B, T, D] contains frames 1..T where
          dyn_flat[:, 0] = z_curr  (frame_1, the anchor)
          dyn_flat[:, 1:] = z_future  (frames 2..T, the prediction targets)
        Any delta computed from dyn_flat[:, k+1] - dyn_flat[:, k] for k >= 0
        uses at least one future frame, leaking GT into the predictor and causing
        train/rollout mismatch.

        To re-enable safely: extend the data pipeline to supply a dedicated buffer
        of past dynamic frames (before frame_0) so that history deltas are
        computed from frames entirely preceding the current segment.

        Returns None when residual_history_len == 0 (disabled).
        """
        H_hist = self.cfg.residual_history_len
        if H_hist <= 0:
            return None
        T = dyn_flat.shape[1]
        # Number of adjacent pairs available within the window
        n_pairs = min(H_hist, T - 1)
        if n_pairs <= 0:
            return torch.zeros(dyn_flat.shape[0], dyn_flat.shape[-1], device=device,
                               dtype=dyn_flat.dtype)
        # delta[k] = dyn_flat[:, k+1] - dyn_flat[:, k]  for k = 0..n_pairs-1
        # NOTE: all pairs here involve future frames — do not enable without a
        #       past-frame buffer (see docstring above).
        deltas = dyn_flat[:, 1:1 + n_pairs, :] - dyn_flat[:, :n_pairs, :]  # [B, n_pairs, flat_dim]
        return deltas.mean(dim=1)  # [B, flat_dim]

    # ------------------------------------------------------------------
    # Reward-aligned loss helper
    # ------------------------------------------------------------------

    def _reward_aligned_loss(
        self,
        pred_cum_delta: torch.Tensor,          # [B, H, flat_dim]
        gt_cum_delta: torch.Tensor,            # [B, H, flat_dim]
        pred_latent_seq: torch.Tensor,         # [B, H, flat_dim]
        z_curr: torch.Tensor,                  # [B, flat_dim]
        z_future: torch.Tensor,                # [B, H, flat_dim]  GT future latents
        dyn_tokens: torch.Tensor,              # [B, T, N_dyn]
        B: int,
        T: int,
        device,
        window_start: torch.Tensor | None = None,   # [B] long
        episode_length: torch.Tensor | None = None, # [B] long
        goal_latent: torch.Tensor | None = None,    # [B, flat_dim]
    ) -> torch.Tensor:
        """Reward-aligned loss: task-feature heads as primary, latent MSE as auxiliary.

        Primary:   L_task_feature = w_prog * L_progress + w_succ * L_success
                                    [+ w_proxy * L_reward_proxy]
        Auxiliary: w_consistency * L_consistency  (cumulative residual MSE)

        progress_head target:
            - 'remaining_steps': normalised remaining steps to episode end
            - 'temporal_position': fallback h/(H-1) when episode metadata is unavailable

        success_head target:
            - 'goal_image_distance': normalised latent distance to goal image
            - 'episode_displacement': fallback cumulative displacement from z_curr
        """
        H = pred_latent_seq.shape[1]
        pred_dtype = pred_latent_seq.dtype
        cfg = self.cfg

        # --- GT labels (computed from GT latents / metadata, detached) ---
        with torch.no_grad():
            # Progress label
            if (cfg.progress_target_mode == "remaining_steps"
                    and window_start is not None and episode_length is not None):
                gt_progress = compute_remaining_steps_labels(
                    window_start, episode_length, H, device, torch.float32,
                    normalize=cfg.normalize_remaining_steps,
                    action_start_offset=cfg.action_start_offset,
                )  # [B, H]
            else:
                # Fallback: simple temporal position
                h_idx = torch.linspace(1.0, 0.0, steps=H, device=device)
                gt_progress = h_idx.unsqueeze(0).expand(B, -1)  # [B, H], 1→0 (remaining frac)

            # Success label
            if (cfg.success_target_mode == "goal_image_distance"
                    and goal_latent is not None):
                gt_success = compute_goal_distance_labels(
                    z_future.float(), goal_latent.float(), normalize=True
                )  # [B, H]  — distance normalised to [0, 1], 1=far from goal
            else:
                # Fallback: cumulative displacement from z_curr, normalised
                cum_disp = (z_future - z_curr.unsqueeze(1)).norm(dim=-1)  # [B, H]
                init_disp = cum_disp[:, 0:1].clamp(min=1e-6)
                gt_success = (cum_disp / init_disp).clamp(0.0, 1.0)

            # Reward proxy label (motion magnitude)
            gt_motion = compute_motion_proxy_labels(z_future.float(), z_curr.float())  # [B, H]

        # --- Apply heads to predicted latents ---
        scores = self.reward_heads(pred_latent_seq.to(pred_dtype))  # dict of [B, H]

        progress_loss = F.huber_loss(
            scores["progress"], gt_progress.to(pred_dtype).detach(), delta=0.5
        )
        success_loss = F.huber_loss(
            scores["success"], gt_success.to(pred_dtype).detach(), delta=0.5
        )

        L_task_feature = (
            cfg.loss_weight_progress * progress_loss
            + cfg.loss_weight_success * success_loss
        )

        reward_proxy_loss: torch.Tensor | None = None
        if cfg.use_reward_proxy_head and "reward_proxy" in scores:
            reward_proxy_loss = F.huber_loss(
                scores["reward_proxy"], gt_motion.to(pred_dtype).detach(), delta=0.5
            )
            L_task_feature = L_task_feature + cfg.loss_weight_reward_proxy * reward_proxy_loss

        # Cumulative residual MSE demoted to auxiliary
        L_consistency = F.mse_loss(pred_cum_delta, gt_cum_delta.detach().to(pred_dtype))

        total = L_task_feature + cfg.loss_weight_consistency * L_consistency

        # Return individual components (detached floats) alongside total for logging
        components: dict = {
            "loss_progress":    progress_loss.detach().float(),
            "loss_success":     success_loss.detach().float(),
            "loss_consistency": L_consistency.detach().float(),
        }
        if reward_proxy_loss is not None:
            components["loss_reward_proxy"] = reward_proxy_loss.detach().float()

        return total, components

    # ------------------------------------------------------------------
    # Reconstruction loss helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reconstruction_loss_adj(
        self,
        dyn_tokens: torch.Tensor,
        dyn_all: torch.Tensor,
        pred_delta: torch.Tensor,
        B: int,
        T: int,
        device,
    ) -> torch.Tensor:
        """Pixel reconstruction loss for adjacent_delta mode."""
        pred_emb = dyn_all[:, :T].to(pred_delta.dtype) + pred_delta  # [B, T, flat_dim]

        D = self.cfg.dyn_token_dim
        N = self.cfg.n_dyn_tokens
        pred_indices = self.visual_tokenizer.dynamics_quantize.codes_to_indices(
            pred_emb.float().reshape(B * T, N, D).reshape(-1, D)
        ).reshape(B, T, N)

        gt_decoded = self.visual_tokenizer.detokenize(
            dyn_tokens[:, :1], dyn_tokens
        )[:, 1:]
        pred_decoded = self.visual_tokenizer.detokenize(
            pred_indices[:, :1], pred_indices
        )[:, 1:]

        return F.mse_loss(pred_decoded.clamp(0, 1), gt_decoded.clamp(0, 1).detach())

    @torch.no_grad()
    def _reconstruction_loss_ca(
        self,
        dyn_tokens_future: torch.Tensor,  # [B, H, N_dyn]  GT future token indices
        pred_latent_seq: torch.Tensor,    # [B, H, flat_dim] predicted future latents
        B: int,
        H: int,
        device,
    ) -> torch.Tensor:
        """Pixel reconstruction loss for current_anchor_ctx mode."""
        D = self.cfg.dyn_token_dim
        N = self.cfg.n_dyn_tokens
        pred_indices = self.visual_tokenizer.dynamics_quantize.codes_to_indices(
            pred_latent_seq.float().reshape(B * H, N, D).reshape(-1, D)
        ).reshape(B, H, N)

        gt_decoded = self.visual_tokenizer.detokenize(
            dyn_tokens_future[:, :1], dyn_tokens_future
        )[:, 1:]
        pred_decoded = self.visual_tokenizer.detokenize(
            pred_indices[:, :1], pred_indices
        )[:, 1:]

        return F.mse_loss(pred_decoded.clamp(0, 1), gt_decoded.clamp(0, 1).detach())

    # ------------------------------------------------------------------
    # Rollout API (for verified reward generation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout_teacher_forced(
        self,
        pixels: torch.Tensor,    # [B, T+1, C, H, W] float [0,1] or uint8
        actions: torch.Tensor,   # [B, T, action_dim]
        horizon: int = -1,
    ) -> dict:
        """Teacher-forced rollout for current_anchor_ctx mode.

        At each step the predictor receives GT latents as context (no error accumulation).
        Useful for training-aligned evaluation and reward supervision.

        Args:
            pixels:  [B, T+1, C, H, W] in [0,1] or uint8 channel-last.
            actions: [B, T, action_dim].
            horizon: number of future frames to predict (-1 = T-1 = full window).

        Returns dict with keys:
            pred_latent_seq   [B, H, flat_dim]
            pred_cum_delta    [B, H, flat_dim]
            gt_latent_seq     [B, H, flat_dim]
            score_dict        {progress, success, reward_proxy} each [B, H] if heads present
        """
        if self.cfg.residual_target_mode != "current_anchor_ctx":
            raise RuntimeError("rollout_teacher_forced requires residual_target_mode='current_anchor_ctx'.")

        # Normalise pixels to float [0,1] channel-first
        if pixels.dtype == torch.uint8:
            pixels_f = pixels.permute(0, 1, 4, 2, 3).float() / 255.0
        else:
            pixels_f = pixels if pixels.shape[2] < pixels.shape[-1] else pixels

        device = next(self.predictor.parameters()).device
        pixels_f = pixels_f.to(device)
        actions = actions.to(device)

        B, T_plus_1 = pixels_f.shape[:2]
        T = T_plus_1 - 1
        H = (T - 1) if horizon < 0 else min(horizon, T - 1)
        if H < 1:
            raise ValueError(f"horizon={horizon} results in H={H} < 1.")

        ctx_tokens, dyn_tokens = self._encode_both(pixels_f)
        dyn_flat = self._dyn_tokens_to_flat(dyn_tokens)  # [B, T, flat_dim]
        z_curr   = dyn_flat[:, 0, :]                     # [B, flat_dim]
        z_future = dyn_flat[:, 1:1 + H, :]               # [B, H, flat_dim]

        ctx_summary = self._dequantize_ctx(ctx_tokens)
        acts_future = self._normalize_actions(actions[:, 1:1 + H, :])  # [B, H, action_dim]
        res_summary = self._compute_res_summary(dyn_flat, device)

        pred_dtype = next(self.predictor.parameters()).dtype
        pred_cum_delta = self.predictor(
            z_curr.to(pred_dtype),
            ctx_summary.to(pred_dtype),
            acts_future.to(pred_dtype),
            res_summary.to(pred_dtype) if res_summary is not None else None,
        )  # [B, H, flat_dim]

        pred_latent_seq = z_curr.unsqueeze(1).to(pred_dtype) + pred_cum_delta  # [B, H, flat_dim]

        score_dict: dict = {}
        if self.reward_heads is not None:
            score_dict = self.reward_heads(pred_latent_seq)

        return {
            "pred_latent_seq":  pred_latent_seq,
            "pred_cum_delta":   pred_cum_delta,
            "gt_latent_seq":    z_future.to(pred_dtype),
            "score_dict":       score_dict,
        }

    @torch.no_grad()
    def rollout_autoregressive(
        self,
        ctx_tokens: torch.Tensor,  # [B, 1, N_ctx] int
        z_curr: torch.Tensor,      # [B, flat_dim] float
        actions: torch.Tensor,     # [B, H, action_dim] float (unnormalised)
        horizon: int = -1,
    ) -> dict:
        """Open-loop autoregressive rollout for current_anchor_ctx mode.

        Each step feeds back the previously predicted latent as the new z_curr
        (anchor shifts forward). No GT frames are used after the initial z_curr.

        Args:
            ctx_tokens: static context from frame_0, shape [B, 1, N_ctx].
            z_curr:     initial dynamic anchor (embed of frame_1), [B, flat_dim].
            actions:    future actions [B, H, action_dim] (unnormalised).
            horizon:    steps to roll out (-1 = use cfg.autoregressive_horizon).

        Returns dict with keys:
            pred_latent_seq  [B, H, flat_dim]
            pred_cum_delta   [B, H, flat_dim]  (cumulative from original z_curr)
            score_dict       {progress, success, reward_proxy} each [B, H] if heads present
        """
        if self.cfg.residual_target_mode != "current_anchor_ctx":
            raise RuntimeError("rollout_autoregressive requires residual_target_mode='current_anchor_ctx'.")

        device = next(self.predictor.parameters()).device
        H = self.cfg.autoregressive_horizon if horizon < 0 else horizon
        if H < 1:
            raise ValueError(f"horizon={H} < 1.")

        ctx_tokens = ctx_tokens.to(device)
        z_anchor   = z_curr.to(device)
        actions    = actions.to(device)
        acts_norm  = self._normalize_actions(actions[:, :H, :])  # [B, H, action_dim]
        ctx_summary = self._dequantize_ctx(ctx_tokens)

        pred_dtype = next(self.predictor.parameters()).dtype

        # Single-pass prediction (same as teacher-forced but without GT for res_summary)
        pred_cum_delta = self.predictor(
            z_anchor.to(pred_dtype),
            ctx_summary.to(pred_dtype),
            acts_norm.to(pred_dtype),
            None,  # no GT history available in open-loop
        )  # [B, H, flat_dim]

        pred_latent_seq = z_anchor.unsqueeze(1).to(pred_dtype) + pred_cum_delta  # [B, H, flat_dim]

        score_dict: dict = {}
        if self.reward_heads is not None:
            score_dict = self.reward_heads(pred_latent_seq)

        return {
            "pred_latent_seq": pred_latent_seq,
            "pred_cum_delta":  pred_cum_delta,
            "score_dict":      score_dict,
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        import dataclasses
        os.makedirs(save_directory, exist_ok=True)

        # Unwrap DDP if needed
        predictor = self.predictor
        if hasattr(predictor, "module"):
            predictor = predictor.module

        torch.save(
            predictor.state_dict(),
            os.path.join(save_directory, "predictor.pt"),
        )

        if self.reward_heads is not None:
            heads = self.reward_heads
            if hasattr(heads, "module"):
                heads = heads.module
            torch.save(
                heads.state_dict(),
                os.path.join(save_directory, "reward_heads.pt"),
            )

        with open(
            os.path.join(save_directory, "residual_worldmodel_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=2)

    @classmethod
    def load_pretrained(
        cls,
        save_directory: str,
        visual_tokenizer_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "LatentResidualWorldModel":
        with open(
            os.path.join(save_directory, "residual_worldmodel_config.json"),
            encoding="utf-8",
        ) as f:
            cfg_dict = json.load(f)
        cfg = ResidualWorldModelConfig(**cfg_dict)
        cfg.visual_tokenizer_path = visual_tokenizer_path

        model = cls(visual_tokenizer_path, cfg, torch_dtype=torch_dtype)
        state = torch.load(
            os.path.join(save_directory, "predictor.pt"), map_location="cpu"
        )
        model.predictor.load_state_dict(state, strict=True)
        return model
