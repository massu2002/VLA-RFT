# Backward-compatibility shim — real implementation lives in worldmodel/core/processor.py
from ..core.processor import (  # noqa: F401
    ContextMultiStepPredictionProcessor,
    compute_position_id_with_mask,
    batch_forward,
    batch_forward2,
    batch_forward3,
)
