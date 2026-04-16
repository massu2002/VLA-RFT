
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ivideogpt.ctx_tokenizer.compressive_vq_model import CompressiveVQModelFSQ

from .processor import ContextMultiStepPredictionProcessor


@dataclass
class WorldModelRuntimeConfig:
    action_ranges_path: str
    tokenizer_micro_batch_size: Optional[int]
    context_length: int
    action_dim: int
    action_bins: int
    max_length: int
    visual_token_num: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    tokens_per_frame: int
    autocast_dtype: str = "bf16"
    processor_type: str = "ctx_msp"
    use_img_gt_ac: bool = False
    interact: bool = False


class LiberoWorldModelTrainer(nn.Module):
    """A thin wrapper that tokenizes LIBERO windows and trains the causal LM."""

    def __init__(
        self,
        lm_path: str,
        visual_tokenizer_path: str,
        runtime_cfg: WorldModelRuntimeConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = True,
        load_pretrained_weights: bool = False,
    ) -> None:
        super().__init__()
        self.lm_path = lm_path
        self.visual_tokenizer_path = visual_tokenizer_path
        self.runtime_cfg = runtime_cfg
        self.load_pretrained_weights = load_pretrained_weights

        self.text_tokenizer = AutoTokenizer.from_pretrained(lm_path, use_fast=True)
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.text_tokenizer.padding_side = "right"

        self.visual_tokenizer = CompressiveVQModelFSQ.from_pretrained(visual_tokenizer_path, torch_dtype=torch.float32)
        self.visual_tokenizer.requires_grad_(False)
        self.visual_tokenizer.eval()

        processor_cfg = SimpleNamespace(**asdict(runtime_cfg))
        self.processor = ContextMultiStepPredictionProcessor(processor_cfg, self.visual_tokenizer)

        if load_pretrained_weights:
            self.lm = AutoModelForCausalLM.from_pretrained(
                lm_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
        else:
            config = AutoConfig.from_pretrained(lm_path)
            self.lm = AutoModelForCausalLM.from_config(config)
            if torch_dtype is not None:
                self.lm = self.lm.to(dtype=torch_dtype)
        self.config = self.lm.config
        self.lm.config.use_cache = False
        if gradient_checkpointing:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.gradient_checkpointing = gradient_checkpointing

    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.lm.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        return self.lm.enable_input_require_grads()

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def forward(self, pixels=None, actions=None, **kwargs):
        if pixels is None or actions is None:
            raise ValueError("pixels and actions are required")

        if not torch.is_tensor(pixels):
            pixels = torch.as_tensor(pixels)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)

        device = next(self.lm.parameters()).device
        pixels = pixels.to(device=device, non_blocking=True)
        actions = actions.to(device=device, non_blocking=True)

        pixels = pixels.permute(0, 1, 4, 2, 3).contiguous().float() / 255.0

        model_inputs = self.processor(pixels, actions)

        outputs = self.lm(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs["labels"],
            **kwargs,
        )
        return outputs

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.lm.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)
        self.text_tokenizer.save_pretrained(save_directory)
        with open(os.path.join(save_directory, "worldmodel_runtime_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.runtime_cfg), f, indent=2)
