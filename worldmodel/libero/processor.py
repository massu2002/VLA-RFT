from __future__ import annotations

from typing import Optional

import torch


def compute_position_id_with_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.cumsum(mask, dim=-1) - 1, min=0)


def batch_forward(batch_size, input_tensor, forward):
    outputs = []
    for i in range(0, input_tensor.shape[0], batch_size):
        outputs.append(forward(input_tensor[i : i + batch_size]))
    return torch.cat(outputs, dim=0)


def batch_forward2(batch_size, input1, input2, forward):
    outputs = []
    for i in range(0, input1.shape[0], batch_size):
        outputs.append(forward(input1[i : i + batch_size], input2[i : i + batch_size]))
    return torch.cat(outputs, dim=0)


def batch_forward3(batch_size, input_tensor, forward):
    outputs_1, outputs_2 = [], []
    for i in range(0, input_tensor.shape[0], batch_size):
        out_1, out_2 = forward(input_tensor[i : i + batch_size])
        outputs_1.append(out_1)
        outputs_2.append(out_2)
    return torch.cat(outputs_1, dim=0), torch.cat(outputs_2, dim=0)


class ContextMultiStepPredictionProcessor:
    """Matches the context multi-step processor used in the paper codepath."""

    def __init__(self, config, visual_tokenizer):
        self.config = config
        self.visual_tokenizer = visual_tokenizer
        self.action_ranges = torch.load(config.action_ranges_path, map_location="cpu")

    def _autocast_device(self, tensor: torch.Tensor) -> str:
        return "cuda" if tensor.is_cuda else "cpu"

    def _autocast_dtype(self, tensor: torch.Tensor) -> torch.dtype:
        if not tensor.is_cuda:
            return torch.float32

        dtype_name = getattr(self.config, "autocast_dtype", "bf16")
        if dtype_name == "bf16":
            return torch.bfloat16
        if dtype_name == "fp16":
            return torch.float16
        return torch.float32

    def _discretize_actions(self, actions, num_bins=256):
        if actions.dim() == 3:
            b, t = actions.shape[:2]
            actions = actions.reshape(b * t, -1)
        else:
            b, t = None, None

        action_ranges = self.action_ranges.to(actions.device)
        max_values, min_values = action_ranges[:, 1], action_ranges[:, 0]
        actions = torch.clip((actions - min_values) / (max_values - min_values + 1e-8), 0, 1)
        actions = torch.floor(actions * num_bins).to(torch.int32).clip(0, num_bins - 1)

        if b is not None and t is not None:
            actions = actions.reshape(b, t, *actions.shape[1:])
        return actions

    @torch.no_grad()
    def detokenize(self, ctx_tokens, tokens):
        with torch.autocast(device_type=self._autocast_device(ctx_tokens), dtype=self._autocast_dtype(ctx_tokens)):
            if self.config.tokenizer_micro_batch_size is not None:
                return batch_forward2(
                    self.config.tokenizer_micro_batch_size,
                    ctx_tokens,
                    tokens,
                    lambda x, y: self.visual_tokenizer.detokenize(x, y),
                )
            return self.visual_tokenizer.detokenize(ctx_tokens, tokens)

    def _create_response(self, ground_truth_tokens):
        device = ground_truth_tokens.device
        b, t, n = ground_truth_tokens.shape

        input_ids = torch.cat(
            [
                torch.ones((b, t, 1), dtype=ground_truth_tokens.dtype, device=device) * self.config.bos_token_id,
                ground_truth_tokens,
            ],
            dim=2,
        ).reshape(b, -1)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.float32, device=device)
        loss_mask = torch.ones(input_ids.shape, dtype=torch.float32, device=device)

        def singles(value, dtype):
            return torch.ones((b, 1), device=device, dtype=dtype) * value

        input_ids = torch.cat([input_ids, singles(self.config.eos_token_id, input_ids.dtype)], dim=1)
        attention_mask = torch.cat([attention_mask, singles(1.0, attention_mask.dtype)], dim=1)
        loss_mask = torch.cat([loss_mask, singles(1.0, loss_mask.dtype)], dim=1)

        labels = input_ids
        return input_ids, attention_mask, loss_mask, labels

    @torch.no_grad()
    def __call__(self, pixels, actions, return_interpolated=False, return_ctx_tokens=False):
        self.action_ranges = self.action_ranges.to(pixels.device)

        # pixels: [B, T+1, C, H, W]
        # actions: [B, T, action_dim]
        b, num_frames = pixels.shape[:2]
        num_actions = actions.shape[1]
        if num_frames != num_actions + 1:
            raise ValueError(
                f"Expected num_frames == num_actions + 1, got {num_frames} and {num_actions}"
            )

        context_length = self.config.context_length

        autocast_device = self._autocast_device(pixels)
        autocast_dtype = self._autocast_dtype(pixels)
        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
            if self.config.tokenizer_micro_batch_size is not None:
                ctx_tokens, dyn_tokens = batch_forward3(
                    self.config.tokenizer_micro_batch_size,
                    pixels,
                    lambda x: self.visual_tokenizer.tokenize(x),
                )
            else:
                ctx_tokens, dyn_tokens = self.visual_tokenizer.tokenize(pixels)

        # token id range を分離
        ctx_tokens = ctx_tokens[:, :context_length] + self.config.visual_token_num
        action_tokens = self._discretize_actions(actions, self.config.action_bins)
        action_tokens = action_tokens + self.config.visual_token_num * 2

        # dyn_tokens は各 frame に対応している前提
        # o_i を prefix に置き、o_{i+1:i+T} を予測対象にする
        future_dyn_tokens = dyn_tokens  # そのまま future 側として使う

        prefix_tokens = ctx_tokens.reshape(b, -1)

        # 各step: action_t のあとに future_image_{t+1}
        step_tokens = torch.cat(
            [
                action_tokens,
                future_dyn_tokens,
            ],
            dim=-1,
        )  # [B, T, n_act + n_dyn]

        input_ids = torch.cat(
            [
                prefix_tokens,
                step_tokens.reshape(b, -1),
            ],
            dim=-1,
        )

        # labels: action は全部マスク、future image だけ supervise
        step_labels = torch.full_like(step_tokens, -100)
        n_act = action_tokens.shape[-1]
        step_labels[..., n_act:] = future_dyn_tokens

        labels = torch.cat(
            [
                torch.full_like(prefix_tokens, -100),
                step_labels.reshape(b, -1),
            ],
            dim=-1,
        )

        attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
        position_ids = compute_position_id_with_mask(attention_mask)

        output_dict = {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels.long(),
        }

        if return_interpolated:
            if return_ctx_tokens:
                return output_dict, pixels, ctx_tokens
            return output_dict, pixels
        if return_ctx_tokens:
            return output_dict, ctx_tokens
        return output_dict
