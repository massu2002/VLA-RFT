"""Residual World Model variants for VLA-RFT.

Latent-space model (Phase 0 baseline):
    LatentResidualWorldModel  — model.py
    ResidualWorldModelConfig  — config.py

Pixel-space residual model (Phase 1):
    PixelResidualWorldModel   — pixel_residual_model.py
    PixelResidualConfig       — pixel_residual_config.py

Training (LIBERO):
    python -m worldmodel.residual_worldmodel.train_libero --task-suite spatial ...
    python -m worldmodel.residual_worldmodel.train_pixel_residual_libero \\
        --task-suite spatial --output-dir checkpoints/libero/PixelResidualWM/...
"""

from .config import ResidualWorldModelConfig
from .model import LatentResidualWorldModel
from .pixel_residual_config import PixelResidualConfig
from .pixel_residual_model import PixelResidualWorldModel

__all__ = [
    "ResidualWorldModelConfig",
    "LatentResidualWorldModel",
    "PixelResidualConfig",
    "PixelResidualWorldModel",
]
