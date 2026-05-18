"""Dynamic Query World Model (DynQuery).

Core architectural innovation: DynamicQueryExtractor extracts Q query vectors
from K+1 history frames via shared soft-attention masks, then rolls them forward
through actions to predict future images without a dense pixel predictor.

Stages:
  DynamicQueryExtractor    — history frames → dynamic queries + masks
  TemporalResidualPredictor — queries + actions → future dynamic queries
  TokenFuser               — fuse future queries into current spatial tokens
  PixelDecoder             — future spatial tokens → predicted future images
  ActionFutureScorer       — (optional) rank action sequences vs future queries

Usage:
    python -m worldmodel.dynquery.train --task-suite spatial --output-dir <path>
    python -m worldmodel.dynquery.eval  --task-suite spatial --model-dir <path>
"""

from .config import DynQueryConfig, add_dynquery_args
from .model import DynQueryWorldModel

__all__ = [
    "DynQueryConfig",
    "add_dynquery_args",
    "DynQueryWorldModel",
]
