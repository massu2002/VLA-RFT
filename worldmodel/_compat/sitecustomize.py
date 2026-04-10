
"""Compatibility tweaks for the world-model training entrypoints."""

from __future__ import annotations

import importlib.util

_original_find_spec = importlib.util.find_spec


def _patched_find_spec(name, package=None):
    if name in {"flash_attn", "flash_attn_2_cuda"}:
        return None
    return _original_find_spec(name, package)


importlib.util.find_spec = _patched_find_spec
