# Backward-compatibility shim — real implementation lives in worldmodel/core/model.py
from ..core.model import (  # noqa: F401
    WorldModelRuntimeConfig,
    WorldModelTrainer,
    WorldModelTrainer as LiberoWorldModelTrainer,
)
