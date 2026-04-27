# Backward-compatibility shim — real implementation lives in worldmodel/datasets/libero/data.py
from ..datasets.libero.data import (  # noqa: F401
    LIBERO_TASK_TO_DATASET,
    SampleWindow,
    RldsIterableDataset,
    RldsIterableDataset as LiberoWorldModelIterableDataset,
    resolve_dataset_name,
)
