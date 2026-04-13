# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/config.py

import enum
import json
import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

from transformers import PretrainedConfig

# Add for verl
import vllm.config as _vllm_config_mod
from vllm.config import ModelConfig as VLLMBaseModelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.loader import BaseModelLoader

logger = init_logger(__name__)


def _drop_attr(obj, name):
    try:
        setattr(obj, name, None)
    except Exception:
        pass

    try:
        obj.__dict__.pop(name, None)
    except Exception:
        pass

    for internal_name in ("_internal_dict", "internal_dict"):
        try:
            d = getattr(obj, internal_name, None)
            if d is not None and hasattr(d, "pop"):
                d.pop(name, None)
        except Exception:
            pass


def _rope_type(x):
    if not isinstance(x, dict):
        return None
    return x.get("rope_type", x.get("type"))


def _sanitize_rope_in_place(obj, prefix="root"):
    if obj is None:
        return

    rp = getattr(obj, "rope_parameters", None)
    if isinstance(rp, dict):
        rp = dict(rp)
        rp_type = _rope_type(rp)

        if "type" in rp and "rope_type" not in rp:
            rp["rope_type"] = rp.pop("type")

        # default だけ no-scaling として落とす
        if rp_type == "default":
            _drop_attr(obj, "rope_parameters")
        else:
            try:
                setattr(obj, "rope_parameters", rp)
            except Exception:
                pass

            # rope_scaling が未設定なら補助的に移す
            rs = getattr(obj, "rope_scaling", None)
            if rs is None:
                try:
                    setattr(obj, "rope_scaling", dict(rp))
                except Exception:
                    pass

    elif rp is not None:
        _drop_attr(obj, "rope_parameters")

    rs = getattr(obj, "rope_scaling", None)
    if isinstance(rs, dict):
        rs = dict(rs)
        rs_type = _rope_type(rs)

        if "type" in rs and "rope_type" not in rs:
            rs["rope_type"] = rs.pop("type")

        # default だけ落とす
        if rs_type == "default":
            _drop_attr(obj, "rope_scaling")
        else:
            try:
                setattr(obj, "rope_scaling", rs)
            except Exception:
                pass

    elif rs is not None:
        _drop_attr(obj, "rope_scaling")

    for child_name in ("text_config", "language_config", "llm_config", "decoder_config", "generator_config"):
        child = getattr(obj, child_name, None)
        if child is not None and child is not obj:
            _sanitize_rope_in_place(child, f"{prefix}.{child_name}")


def _safe_to_dict(cfg):
    try:
        return cfg.to_dict()
    except Exception:
        return None


def _log_rope_tree_from_dict(d, prefix="root"):
    if isinstance(d, dict):
        rs = d.get("rope_scaling", "__missing__")
        rp = d.get("rope_parameters", "__missing__")

        if rs != "__missing__" or rp != "__missing__":
            logger.warning(
                "DEBUG ROPE %s | rope_scaling=%r | rope_parameters=%r",
                prefix, None if rs == "__missing__" else rs, None if rp == "__missing__" else rp
            )

        for k, v in d.items():
            if isinstance(v, dict):
                _log_rope_tree_from_dict(v, f"{prefix}.{k}")
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        _log_rope_tree_from_dict(item, f"{prefix}.{k}[{i}]")


if not getattr(_vllm_config_mod, "_verl_rope_fix_installed", False):
    _orig_get_and_verify_max_len = _vllm_config_mod._get_and_verify_max_len

    def _patched_get_and_verify_max_len(*args, **kwargs):
        hf_config = kwargs.get("hf_config", None)
        if hf_config is None and len(args) >= 1:
            hf_config = args[0]

        if hf_config is not None:
            logger.warning("========== ROPE FIX before _get_and_verify_max_len ==========")
            logger.warning(
                "ROPE FIX incoming class=%s _name_or_path=%r",
                hf_config.__class__.__name__,
                getattr(hf_config, "_name_or_path", None),
            )

            before = _safe_to_dict(hf_config)
            if before is not None:
                _log_rope_tree_from_dict(before, "before_fix")

            _sanitize_rope_in_place(hf_config)

            after = _safe_to_dict(hf_config)
            if after is not None:
                _log_rope_tree_from_dict(after, "after_fix")

        return _orig_get_and_verify_max_len(*args, **kwargs)

    _vllm_config_mod._get_and_verify_max_len = _patched_get_and_verify_max_len
    _vllm_config_mod._verl_rope_fix_installed = True


class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    MEGATRON = "megatron"
    HF = "hf"
    DTENSOR = "dtensor"
    DUMMY_HF = "dummy_hf"
    DUMMY_MEGATRON = "dummy_megatron"
    DUMMY_DTENSOR = "dummy_dtensor"


class ModelConfig(VLLMBaseModelConfig):
    def __init__(self, hf_config: PretrainedConfig, *args, **kwargs) -> None:
        safe_hf_config = copy.deepcopy(hf_config)
        _sanitize_rope_in_place(safe_hf_config)

        super().__init__(
            model=safe_hf_config._name_or_path,
            tokenizer=safe_hf_config._name_or_path,
            *args,
            **kwargs
        )
        self.hf_config = safe_hf_config


@dataclass
class LoadConfig:
    """
    download_dir: Directory to download and load the weights, default to the
        default cache directory of huggingface.
    load_format: The format of the model weights to load:
        "auto" will try to load the weights in the safetensors format and
            fall back to the pytorch bin format if safetensors format is
            not available.
        "pt" will load the weights in the pytorch bin format.
        "safetensors" will load the weights in the safetensors format.
        "npcache" will load the weights in pytorch format and store
            a numpy cache to speed up the loading.
        "dummy" will initialize the weights with random values, which is
            mainly for profiling.
        "tensorizer" will use CoreWeave's tensorizer library for
            fast weight loading.
        "bitsandbytes" will load nf4 type weights.
    ignore_patterns: The list of patterns to ignore when loading the model.
        Default to "original/**/*" to avoid repeated loading of llama's
        checkpoints.

    """

    load_format: Union[str, LoadFormat, "BaseModelLoader"] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(default_factory=dict)
    ignore_patterns: Optional[Union[List[str], str]] = None

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = json.loads(model_loader_extra_config)
        self._verify_load_format()

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info("Ignoring the following patterns when downloading weights: %s", self.ignore_patterns)
        else:
            self.ignore_patterns = ["original/**/*"]

    def _verify_load_format(self) -> None:
        if not isinstance(self.load_format, str):
            return

        load_format = self.load_format.lower()
        self.load_format = LoadFormat(load_format)

        rocm_not_supported_load_format: List[str] = []
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f for f in LoadFormat.__members__ if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(f"load format '{load_format}' is not supported in ROCm. "
                             f"Supported load formats are "
                             f"{rocm_supported_load_format}")
