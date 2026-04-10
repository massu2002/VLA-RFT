"""Minimal xformers.ops stub used by Diffusers attention modules."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


class AttentionBias:
    """Placeholder base class used only for type checks."""


class AttentionOpBase:
    """Placeholder base class used only for type checks."""


class MemoryEfficientAttentionFlashAttentionOp:
    """Placeholder used by Diffusers imports."""


def _apply_attention_bias(scores: torch.Tensor, attn_bias: Optional[torch.Tensor]) -> torch.Tensor:
    if attn_bias is None:
        return scores

    if hasattr(attn_bias, "materialize"):
        attn_bias = attn_bias.materialize(scores.shape, dtype=scores.dtype, device=scores.device)

    if not torch.is_tensor(attn_bias):
        return scores

    if attn_bias.dtype == torch.bool:
        scores = scores.masked_fill(~attn_bias, torch.finfo(scores.dtype).min)
    else:
        scores = scores + attn_bias.to(device=scores.device, dtype=scores.dtype)
    return scores


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, op=None):
    """A small, correct-enough fallback for the attention paths used here."""

    del op

    if query.shape[-1] != key.shape[-1] or key.shape[-1] != value.shape[-1]:
        raise ValueError("query, key, and value must share the same head dimension")

    head_dim = query.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    scores = _apply_attention_bias(scores, attn_bias)
    probs = torch.softmax(scores, dim=-1)
    if p and p > 0:
        probs = F.dropout(probs, p=p)
    return torch.matmul(probs, value)


__all__ = [
    "AttentionBias",
    "AttentionOpBase",
    "MemoryEfficientAttentionFlashAttentionOp",
    "memory_efficient_attention",
]
