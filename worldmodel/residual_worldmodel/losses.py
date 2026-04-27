"""Loss functions for Latent Residual World Model and FocusedResidualWM.

Organised in sections:
  1. Legacy latent-space losses (used by LatentResidualWorldModel)
  2. DINO-based losses (used by ActionConditionedFocusedResidualWM)
  3. Focus metrics
  4. Reward-aligned label computation

All DINO-based losses expect pre-extracted feature tensors — the loss functions
themselves are pure torch, without backbone calls.  This keeps them testable
without a GPU and composable in any training loop.

Teacher-focus API
-----------------
Future extensions can pass teacher_focus [B, N] ∈ [0,1] to
dino_focus_supervision_loss() to mix a top-down teacher signal with the
self-supervised DINO-diff target.  The API is reserved but unused today.

TODO(RFT): When connecting to RFT, add a reward_from_dino_score() helper
           that converts the candidate DINO ranking score to a per-step reward.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def latent_residual_loss(
    pred_delta: torch.Tensor,
    gt_delta: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error between predicted and ground-truth latent residuals."""
    return F.mse_loss(pred_delta, gt_delta.detach().to(pred_delta.dtype))


def reconstruction_loss(
    pred_frames: torch.Tensor,
    gt_frames: torch.Tensor,
) -> torch.Tensor:
    """Pixel-level MSE between decoded predicted and ground-truth frames."""
    return F.mse_loss(
        pred_frames.clamp(0.0, 1.0),
        gt_frames.clamp(0.0, 1.0).detach().to(pred_frames.dtype),
    )


def combined_loss(
    pred_delta: torch.Tensor,
    gt_delta: torch.Tensor,
    pred_frames: torch.Tensor | None = None,
    gt_frames: torch.Tensor | None = None,
    recon_weight: float = 0.0,
) -> torch.Tensor:
    """Weighted combination of residual and optional reconstruction losses."""
    loss = latent_residual_loss(pred_delta, gt_delta)
    if recon_weight > 0.0 and pred_frames is not None and gt_frames is not None:
        loss = loss + recon_weight * reconstruction_loss(pred_frames, gt_frames)
    return loss


# ---------------------------------------------------------------------------
# Reward-aligned label computation
# ---------------------------------------------------------------------------


def compute_remaining_steps_labels(
    window_start: torch.Tensor,   # [B] long — index of window start in episode
    episode_length: torch.Tensor, # [B] long — total episode length
    H: int,                       # number of future steps being predicted
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool = True,
    action_start_offset: int = 1,
) -> torch.Tensor:
    """Remaining steps to episode end for each predicted future frame.

    Frame layout (action_start_offset=1):
        window_start          = episode index of segment frame_0
        z_curr                = frame_1  (episode index window_start + 1)
        z_future[h] / pred[h] = frame_{h+2}  (episode index window_start + h + 2)

    The formula uses abs_idx = window_start + action_start_offset + h, which is
    window_start + h + 1 — one step earlier than the actual frame_{h+2} index.
    This introduces a mild +1 bias in remaining (labels are slightly optimistic).
    The bias is consistent across all h and does not affect relative ordering,
    so regression learning is unaffected in practice.

    remaining[b, h] = episode_length[b] - (window_start[b] + action_start_offset + h + 1)

    Args:
        window_start:   [B] absolute start index of the window in the episode.
        episode_length: [B] total episode steps.
        H:              number of predicted future frames.
        normalize:      if True, divide by episode_length so values are in [0, 1].
        action_start_offset: offset added to window_start (1 for CA = skip action_0).
    Returns:
        [B, H] float tensor; clamped to [0, 1] when normalize=True.
    """
    B = window_start.shape[0]
    # h_idx[h] = h for h = 0..H-1
    h_idx = torch.arange(H, device=device, dtype=torch.float32)  # [H]

    # Absolute episode index of each predicted frame
    abs_idx = (window_start.float() + action_start_offset).unsqueeze(1) + h_idx.unsqueeze(0)
    # [B, H]

    remaining = episode_length.float().unsqueeze(1) - abs_idx - 1.0  # [B, H]
    remaining = remaining.clamp(min=0.0)

    if normalize:
        remaining = remaining / episode_length.float().unsqueeze(1).clamp(min=1.0)
        remaining = remaining.clamp(0.0, 1.0)

    return remaining.to(dtype)


def compute_goal_distance_labels(
    z_future: torch.Tensor,   # [B, H, flat_dim]  GT future latents
    goal_latent: torch.Tensor, # [B, flat_dim]      goal image latent
    eps: float = 1e-6,
    normalize: bool = True,
) -> torch.Tensor:
    """Latent distance from each predicted frame to the episode goal image.

    goal_distance[b, h] = ||z_future[b, h] - goal_latent[b]|| / normalizer

    When normalize=True, divide by the initial distance (z_future[0] to goal)
    so that the label starts near 1 (far from goal) and decreases toward 0.

    Returns:
        [B, H] float, in [0, 1] when normalized.
    """
    diff = z_future.float() - goal_latent.float().unsqueeze(1)   # [B, H, flat_dim]
    dist = diff.norm(dim=-1)                                       # [B, H]

    if normalize:
        init_dist = dist[:, 0:1].clamp(min=eps)                   # [B, 1]
        dist = (dist / init_dist).clamp(0.0, 2.0) / 2.0           # normalise to [0, 1]

    return dist.to(z_future.dtype)


def compute_motion_proxy_labels(
    z_future: torch.Tensor,   # [B, H, flat_dim]
    z_curr: torch.Tensor,     # [B, flat_dim]
    eps: float = 1e-6,
) -> torch.Tensor:
    """Step-wise motion magnitude as reward proxy, normalised to ~[0, 1].

    Used for the reward_proxy_head when use_reward_proxy_head=True.
    """
    seq = torch.cat([z_curr.unsqueeze(1), z_future], dim=1)  # [B, H+1, flat_dim]
    deltas = seq[:, 1:, :] - seq[:, :-1, :]                  # [B, H, flat_dim]
    motion = deltas.norm(dim=-1)                               # [B, H]
    mean_motion = motion.mean(dim=1, keepdim=True).clamp(min=eps)
    return (motion / mean_motion).clamp(0.0, 3.0) / 3.0       # ~[0, 1]


# ===========================================================================
# Section 2: DINO-based losses for ActionConditionedFocusedResidualWM
# ===========================================================================


def dino_feature_consistency_loss(
    predicted_tokens: torch.Tensor,   # [B, N, D_h]  — from ResidualFuturePredictor
    gt_projected_dino: torch.Tensor,  # [B, N, D_h]  — proj(DINO(gt_future)), detached
) -> torch.Tensor:
    """MSE between predicted future tokens and projected GT DINO features.

    Both tensors must already be in the same projected hidden_dim space.
    The caller is responsible for applying the DINO projection to the GT
    features before passing them here.

    Args:
        predicted_tokens:  [B, N, D_h] from ResidualFuturePredictor.
        gt_projected_dino: [B, N, D_h] = dino.proj(DINO(gt_future)), no_grad.
    Returns:
        scalar MSE loss.
    """
    return F.mse_loss(predicted_tokens, gt_projected_dino.detach().to(predicted_tokens.dtype))


def dino_focus_supervision_loss(
    focus_probs: torch.Tensor,     # [B, N] ∈ [0, 1]  — sigmoid-probability focus map
    change_target: torch.Tensor,   # [B, N] ∈ [0, 1]  — DINO-diff or pixel-diff target
    teacher_focus: Optional[torch.Tensor] = None,  # [B, N] reserved for future use
    teacher_alpha: float = 0.0,    # mix weight for teacher signal (0 = ignore)
) -> torch.Tensor:
    """Binary cross-entropy between a probability focus map and the change target.

    Args:
        focus_probs:   [B, N] ∈ [0, 1] — sigmoid output of FocusHead.
                       Cast to float32 before BCE to remain safe under autocast.
        change_target: [B, N] ∈ [0, 1] — per-patch change magnitude (DINO-diff
                       or pixel-diff), normalised to [0, 1].  DINO-diff is
                       preferred because it captures semantic change.
        teacher_focus / teacher_alpha: reserved API for future top-down teacher
            supervision.  Set teacher_alpha > 0 to blend in a teacher signal.
            Currently unused (teacher_alpha defaults to 0).

    Note on autocast safety:
        F.binary_cross_entropy is unsafe under fp16/bf16 autocast (inputs may
        overflow).  Casting to float32 explicitly avoids this.  The model's
        training loop uses binary_cross_entropy_with_logits on raw logits, which
        is numerically equivalent and natively autocast-safe.  This standalone
        function operates on post-sigmoid probabilities, hence the cast approach.

    TODO(RFT): teacher signal can come from task-completion reward or
               a pre-trained policy's attention maps.
    """
    # Cast to float32 to be safe under bf16/fp16 autocast
    probs  = focus_probs.float().clamp(1e-7, 1 - 1e-7)
    target = change_target.detach().float()
    if teacher_focus is not None and teacher_alpha > 0.0:
        target = (1 - teacher_alpha) * target + teacher_alpha * teacher_focus.detach().float()
        target = target.clamp(0.0, 1.0)
    return F.binary_cross_entropy(probs, target)


def focus_sparsity_loss(
    focus_map: torch.Tensor,   # [B, N] ∈ [0, 1]
    mode: str = "l1",
) -> torch.Tensor:
    """Regularise the focus map to be sparse.

    Modes:
      "l1"      — mean(focus_map)  [encourages values toward 0]
      "entropy" — binary-entropy(-p*log(p)-(1-p)*log(1-p)); maximised at 0.5,
                  pushing map toward 0 or 1 (deterministic focus).
    """
    if mode == "entropy":
        p = focus_map.clamp(1e-6, 1 - 1e-6)
        return -(p * p.log() + (1 - p) * (1 - p).log()).mean()
    return focus_map.mean()


def image_reconstruction_loss(
    pred_image: torch.Tensor,    # [B, 3, H, W] ∈ [0, 1]
    gt_image: torch.Tensor,      # [B, 3, H, W] ∈ [0, 1]
    mode: str = "smooth_l1",
) -> torch.Tensor:
    """Pixel-level reconstruction loss.

    Modes: "smooth_l1" (default, robust to outliers) | "l1" | "mse"
    """
    pred = pred_image.clamp(0.0, 1.0)
    gt   = gt_image.clamp(0.0, 1.0).detach().to(pred_image.dtype)
    if mode == "l1":
        return F.l1_loss(pred, gt)
    if mode == "mse":
        return F.mse_loss(pred, gt)
    return F.smooth_l1_loss(pred, gt)


# ===========================================================================
# Section 3: Focus evaluation metrics (no-grad, used in eval loops)
# ===========================================================================


@torch.no_grad()
def focus_metrics(
    focus_map: torch.Tensor,      # [B, N] ∈ [0, 1]
    change_target: Optional[torch.Tensor] = None,  # [B, N] ∈ [0, 1]
    threshold: float = 0.5,
) -> dict:
    """Compute focus diagnostic metrics.

    Returns a dict with scalar float values:
      focus_mean       — average activation
      focus_sparsity   — fraction of patches below threshold (sparse = high)
      focus_entropy    — binary entropy of focus_map (lower = more decisive)
      iou_vs_change    — IoU between binarised focus_map and change_target
      dice_vs_change   — Dice coefficient
    """
    fm = focus_map.float()
    result = {
        "focus_mean":     fm.mean().item(),
        "focus_sparsity": (fm < threshold).float().mean().item(),
        "focus_entropy":  _binary_entropy_mean(fm).item(),
    }
    if change_target is not None:
        ct = change_target.float()
        pred_bin = (fm >= threshold).float()
        gt_bin   = (ct >= threshold).float()
        result["iou_vs_change"]  = _batch_iou(pred_bin, gt_bin).item()
        result["dice_vs_change"] = _batch_dice(pred_bin, gt_bin).item()
    return result


def _binary_entropy_mean(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


def _batch_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean IoU over batch. pred, gt: [B, N] binary."""
    inter = (pred * gt).sum(dim=1)
    union = (pred + gt - pred * gt).sum(dim=1).clamp(min=1)
    return (inter / union).mean()


def _batch_dice(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Dice over batch. pred, gt: [B, N] binary."""
    inter = (pred * gt).sum(dim=1)
    denom = (pred.sum(dim=1) + gt.sum(dim=1)).clamp(min=1)
    return (2 * inter / denom).mean()


@torch.no_grad()
def image_reconstruction_metrics(
    pred_image: torch.Tensor,   # [B, 3, H, W] ∈ [0, 1]
    gt_image: torch.Tensor,     # [B, 3, H, W] ∈ [0, 1]
) -> dict:
    """Compute per-batch image quality metrics.

    Returns:
      l1, smooth_l1, mse — scalar values
    """
    pred = pred_image.clamp(0.0, 1.0).float()
    gt   = gt_image.clamp(0.0, 1.0).float()
    return {
        "future_image_l1":        F.l1_loss(pred, gt).item(),
        "future_image_smooth_l1": F.smooth_l1_loss(pred, gt).item(),
        "future_image_mse":       F.mse_loss(pred, gt).item(),
    }


@torch.no_grad()
def dino_feature_metrics(
    predicted_tokens: torch.Tensor,   # [B, N, D_h]
    gt_projected_dino: torch.Tensor,  # [B, N, D_h]
) -> dict:
    """DINO feature distance metrics for logging.

    Returns:
      dino_feature_mse        — MSE in projected space
      dino_cosine_similarity  — mean cosine sim over (B × N) pairs
    """
    p  = predicted_tokens.float().reshape(-1, predicted_tokens.shape[-1])
    g  = gt_projected_dino.float().reshape(-1, gt_projected_dino.shape[-1])
    mse  = F.mse_loss(p, g).item()
    cos  = F.cosine_similarity(p, g, dim=-1).mean().item()
    return {
        "dino_feature_mse": mse,
        "dino_cosine_similarity": cos,
    }
