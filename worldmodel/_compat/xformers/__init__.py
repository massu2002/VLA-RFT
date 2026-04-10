"""Lightweight xformers compatibility shim.

The project environment ships with an xformers build that is incompatible with
the installed PyTorch version. Diffusers imports xformers during tokenizer
construction, so we provide a tiny local stub that satisfies the imports used
by the world-model tokenizer.
"""

from . import ops  # noqa: F401

__all__ = ["ops"]
__version__ = "0.0.0"
