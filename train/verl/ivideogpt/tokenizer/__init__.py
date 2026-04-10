
"""Lazy tokenizer registry for the world-model codepaths."""

from __future__ import annotations

from importlib import import_module


class _LazyTokenizerRegistry(dict):
    def __init__(self):
        super().__init__({
            "cnn": ("ivideogpt.tokenizer.vq_model", "CNNFSQModel256"),
            "ctx_cnn": ("ivideogpt.ctx_tokenizer.compressive_vq_model", "CompressiveVQModelFSQ"),
        })

    def __getitem__(self, key):
        module_name, attr_name = super().__getitem__(key)
        module = import_module(module_name)
        return getattr(module, attr_name)


TOKENIZER = _LazyTokenizerRegistry()

__all__ = ["TOKENIZER"]
