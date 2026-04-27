"""Latent Residual World Model.

Predicts future video frames as incremental latent residuals in the FSQ
embedding space of the frozen visual tokenizer, conditioned on actions.

Key classes:
    LatentResidualWorldModel  — model.py
    ResidualWorldModelConfig  — config.py

Training (LIBERO):
    python -m worldmodel.residual_worldmodel.train_libero --task-suite spatial ...
"""

from .config import ResidualWorldModelConfig
from .model import LatentResidualWorldModel

__all__ = ["ResidualWorldModelConfig", "LatentResidualWorldModel"]
