
"""A minimal flash_attn stub for the current environment.

The installed flash-attn binary is incompatible with the PyTorch build in this
workspace. We only need a tiny subset of the API so that Transformers and
Diffusers can import the world-model tokenizer checkpoints.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .bert_padding import index_first_axis, pad_input, unpad_input

__version__ = "2.1.0"


def _attention(query, key, value, attn_mask=None, causal=False, softmax_scale=None, dropout_p=0.0):
    if query.shape[-1] != key.shape[-1] or key.shape[-1] != value.shape[-1]:
        raise ValueError("query, key, and value must share the same head dimension")

    head_dim = query.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + attn_mask.to(dtype=scores.dtype, device=scores.device)
    if causal:
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(torch.ones(q_len, k_len, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    if dropout_p and dropout_p > 0:
        probs = F.dropout(probs, p=dropout_p)
    return torch.matmul(probs, value)


def flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False, window_size=None, alibi_slopes=None, deterministic=False):
    del window_size, alibi_slopes, deterministic
    return _attention(query, key, value, causal=causal, softmax_scale=softmax_scale, dropout_p=dropout_p)


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    alibi_slopes=None,
    deterministic=False,
):
    del window_size, alibi_slopes, deterministic, max_seqlen_q, max_seqlen_k
    outputs = []
    for i in range(cu_seqlens_q.numel() - 1):
        q_start = int(cu_seqlens_q[i].item())
        q_end = int(cu_seqlens_q[i + 1].item())
        k_start = int(cu_seqlens_k[i].item())
        k_end = int(cu_seqlens_k[i + 1].item())
        q_i = q[q_start:q_end].unsqueeze(0)
        k_i = k[k_start:k_end].unsqueeze(0)
        v_i = v[k_start:k_end].unsqueeze(0)
        out_i = _attention(q_i, k_i, v_i, causal=causal, softmax_scale=softmax_scale, dropout_p=dropout_p)
        outputs.append(out_i.squeeze(0))
    return torch.cat(outputs, dim=0)
