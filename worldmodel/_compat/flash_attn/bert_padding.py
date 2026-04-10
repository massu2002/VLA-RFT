
"""Padding helpers used by Transformers' flash attention shim."""

from __future__ import annotations

from typing import Tuple

import torch


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, indices.to(device=x.device, dtype=torch.long))


def pad_input(x: torch.Tensor, indices: torch.Tensor, batch_size: int, seqlen: int) -> torch.Tensor:
    out = torch.zeros((batch_size * seqlen, *x.shape[1:]), device=x.device, dtype=x.dtype)
    out.index_copy_(0, indices.to(device=x.device, dtype=torch.long), x)
    return out.view(batch_size, seqlen, *x.shape[1:])


def unpad_input(x: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten().to(dtype=torch.int32)
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    unpadded = x.reshape(-1, *x.shape[2:]).index_select(0, indices.to(device=x.device, dtype=torch.long))
    return unpadded, indices, cu_seqlens, max_seqlen_in_batch
