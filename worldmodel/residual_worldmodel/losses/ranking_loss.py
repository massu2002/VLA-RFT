"""Action ranking loss for v4b ActionFutureScorer.

Trains the ranking head so that:
    score(correct actions) > score(negative actions) + margin

Using the softplus margin loss:
    L_rank = softplus(margin - score_pos + score_neg).mean()

Negative actions can be:
  - same-task other-window actions      (explicit negative_actions tensor)
  - temporal permutation of correct     (fallback; always available)
  - batch-internal same-task sampling   (optional; requires task_ids)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionRankingLoss(nn.Module):
    """Softplus margin ranking loss for action quality scoring.

    Args:
        margin: minimum gap enforced between positive and negative scores.
        reduction: "mean" or "sum".
    """

    def __init__(self, margin: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        score_pos: torch.Tensor,          # [B] score for correct actions
        score_neg: torch.Tensor,          # [B] score for negative actions
    ) -> torch.Tensor:
        """Return scalar ranking loss."""
        loss = F.softplus(self.margin - score_pos + score_neg)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

    def pairwise_accuracy(
        self,
        score_pos: torch.Tensor,
        score_neg: torch.Tensor,
    ) -> float:
        """Fraction of pairs where score_pos > score_neg."""
        with torch.no_grad():
            wins = (score_pos > score_neg).float()
        return float(wins.mean().item())

    def score_gap(
        self,
        score_pos: torch.Tensor,
        score_neg: torch.Tensor,
    ) -> torch.Tensor:
        """score_pos - score_neg; positive gap = correct ordering."""
        return score_pos - score_neg


def make_temporal_permutation_negatives(
    actions: torch.Tensor,          # [B, H, action_dim]
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Permute temporal axis of actions to produce a negative sample.

    Ensures the permutation is not identity.
    Returns [B, H, action_dim].
    """
    B, H, D = actions.shape
    if H < 2:
        return actions.clone()

    perm = torch.randperm(H, generator=rng, device=actions.device)
    # Avoid identity
    identity = torch.arange(H, device=actions.device)
    if (perm == identity).all():
        perm = torch.roll(perm, 1)

    return actions[:, perm, :]


def make_batch_negative_actions(
    actions: torch.Tensor,    # [B, H, action_dim]
    task_ids: torch.Tensor,   # [B] int, same-task windows should match
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample negative actions from same-task other-window within the batch.

    For each item b, finds another item b' with task_ids[b'] == task_ids[b]
    and b' != b, and uses actions[b'] as the negative.
    Falls back to temporal permutation when no same-task partner exists.
    """
    B = actions.shape[0]
    neg = actions.clone()
    for b in range(B):
        tid = task_ids[b].item()
        candidates = [i for i in range(B) if i != b and task_ids[i].item() == tid]
        if candidates:
            idx = candidates[torch.randint(len(candidates), (1,), generator=rng).item()]
            neg[b] = actions[idx]
        else:
            neg[b] = make_temporal_permutation_negatives(actions[b:b+1], rng=rng)[0]
    return neg
