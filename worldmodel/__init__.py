"""
World model training and evaluation package.

Structure
---------
core/
    Dataset-agnostic model wrapper (WorldModelTrainer) and token processor.
datasets/
    Per-dataset data loaders (RLDS streaming).
    datasets/libero/  — LIBERO task-suite data loader
libero/
    LIBERO-specific visualization and backward-compat shims for train/data/model.

Entry points
------------
Training (generic):
    python -m worldmodel.train --task-suite spatial ...   # LIBERO shorthand
    python -m worldmodel.train --dataset-name my_dataset_no_noops ...

Visualization (LIBERO):
    python -m worldmodel.libero.visualize ...
    python -m worldmodel.libero.visualize_base ...

residual_worldmodel/
    LatentResidualWorldModel — predicts future frames as latent residuals in FSQ
    embedding space of the frozen visual tokenizer, conditioned on actions.
    Training (LIBERO):
        python -m worldmodel.residual_worldmodel.train_libero --task-suite spatial ...
"""
