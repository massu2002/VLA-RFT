"""Training entrypoint for PixelResidualWorldModel on LIBERO RLDS data.

Usage (from repo root):
    python -m worldmodel.residual_worldmodel.train_pixel_residual_libero \\
        --task-suite spatial \\
        --data-root data/modified_libero_rlds \\
        --output-dir checkpoints/libero/PixelResidualWM/spatial/pixel_residual_v1 \\
        --target-mode pixel_residual \\
        --max-steps 50000

Dry-run (smoke test):
    DRY_RUN=1 python -m worldmodel.residual_worldmodel.train_pixel_residual_libero \\
        --task-suite spatial --output-dir /tmp/pxr_smoke --max-steps 5

Environment variable overrides (all also available as CLI flags):
    TARGET_MODE, DYNAMIC_THRESHOLD, ROI_CROP_SIZE,
    LAMBDA_RESIDUAL, LAMBDA_IMAGE, LAMBDA_DYNAMIC, LAMBDA_GRIPPER, LAMBDA_STATIC
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from transformers import Trainer, TrainingArguments

from ..datasets.libero.data import RldsIterableDataset, resolve_dataset_name
from .pixel_residual_config import PixelResidualConfig, add_pixel_residual_args
from .pixel_residual_model import PixelResidualWorldModel


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PixelResidualWorldModel on LIBERO RLDS data."
    )

    # Dataset
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--task-suite", type=str,
                               choices=["spatial", "object", "goal", "10"])
    dataset_group.add_argument("--dataset-name", type=str)
    parser.add_argument("--data-root", type=str, default="data/modified_libero_rlds")

    parser.add_argument("--output-dir", type=str, required=True)

    # Training
    parser.add_argument("--max-steps",            type=int,   default=50000)
    parser.add_argument("--segment-length",        type=int,   default=8,
                        help="Frames per window (= T+1).  H = segment_length - 2.")
    parser.add_argument("--batch-size-per-device", type=int,   default=4)
    parser.add_argument("--grad-accum",            type=int,   default=4)
    parser.add_argument("--global-batch-size",     type=int,   default=None)
    parser.add_argument("--learning-rate",         type=float, default=1e-4)
    parser.add_argument("--warmup-ratio",          type=float, default=0.02)
    parser.add_argument("--weight-decay",          type=float, default=0.0)
    parser.add_argument("--adam-beta1",            type=float, default=0.9)
    parser.add_argument("--adam-beta2",            type=float, default=0.999)
    parser.add_argument("--adam-epsilon",          type=float, default=1e-8)
    parser.add_argument("--max-grad-norm",         type=float, default=1.0)
    parser.add_argument("--optim",                 type=str,   default="adamw_torch")
    parser.add_argument("--seed",                  type=int,   default=42)
    parser.add_argument("--save-steps",            type=int,   default=5000)
    parser.add_argument("--logging-steps",         type=int,   default=10)
    parser.add_argument("--save-total-limit",      type=int,   default=3)
    parser.add_argument("--num-workers",           type=int,   default=0)
    parser.add_argument("--lr-scheduler-type",     type=str,   default="cosine")
    parser.add_argument("--precision",             type=str,
                        choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--tf32",    action="store_true",  default=False)
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")

    # Model-specific
    add_pixel_residual_args(parser)

    return parser


# ---------------------------------------------------------------------------
# Custom Trainer — logs individual loss components
# ---------------------------------------------------------------------------

_LOSS_KEYS = ["loss_residual", "loss_image", "loss_dynamic", "loss_gripper", "loss_static"]


class PixelResidualTrainer(Trainer):
    """Accumulates per-component losses between log intervals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_accum: dict = {}
        self._loss_steps: int  = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        for k in _LOSS_KEYS:
            if k in outputs:
                v = outputs[k]
                self._loss_accum[k] = self._loss_accum.get(k, 0.0) + (
                    v.item() if isinstance(v, torch.Tensor) else float(v)
                )
        self._loss_steps += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, **kwargs):
        if self._loss_steps > 0 and "loss" in logs:
            for k, total in self._loss_accum.items():
                logs[k] = round(total / self._loss_steps, 6)
            self._loss_accum.clear()
            self._loss_steps = 0
        super().log(logs, **kwargs)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class SimpleCollator:
    def __call__(self, batch):
        return {
            "pixels":  torch.stack([item["pixels"]  for item in batch], dim=0),
            "actions": torch.stack([item["actions"] for item in batch], dim=0),
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_world_size() -> int:
    # torchrun sets WORLD_SIZE before the Python process starts,
    # before torch.distributed.init_process_group() is called by HF Trainer.
    # Reading the env var is the only reliable way to get the true world size
    # at argument-parse time.
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1:
        return ws
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _resolve_precision(precision: str):
    if precision == "bf16":
        return torch.bfloat16, True, False, "bf16"
    if precision == "fp16":
        return torch.float16, False, True, "fp16"
    if precision == "fp32":
        return torch.float32, False, False, "fp32"
    if torch.cuda.is_available():
        return torch.bfloat16, True, False, "bf16"
    return torch.float32, False, False, "fp32"


def _save_loss_curves(log_history: List[dict], output_dir: Path) -> None:
    """Save loss_curves.png with one subplot per tracked loss."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    keys = ["loss"] + [k for k in _LOSS_KEYS if any(k in e for e in train_entries)]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        steps = [e["step"] for e in train_entries if key in e]
        vals  = [e[key]    for e in train_entries if key in e]
        if steps:
            ax.plot(steps, vals, linewidth=1.2)
        ax.set_title(key); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Allow env-var overrides before parsing
    for env_key, arg_name in [
        ("TARGET_MODE",       "--target-mode"),
        ("DYNAMIC_THRESHOLD", "--dynamic-threshold"),
        ("ROI_CROP_SIZE",     "--roi-crop-size"),
        ("LAMBDA_RESIDUAL",   "--lambda-residual"),
        ("LAMBDA_IMAGE",      "--lambda-image"),
        ("LAMBDA_DYNAMIC",    "--lambda-dynamic"),
        ("LAMBDA_GRIPPER",    "--lambda-gripper"),
        ("LAMBDA_STATIC",     "--lambda-static"),
    ]:
        val = os.environ.get(env_key)
        if val and arg_name not in __import__("sys").argv:
            __import__("sys").argv.extend([arg_name, val])

    # Dry-run shortcut: tiny defaults
    if os.environ.get("DRY_RUN", "0") in ("1", "true"):
        defaults_override = ["--max-steps", "5", "--batch-size-per-device", "2",
                             "--save-steps", "5", "--logging-steps", "1"]
        __import__("sys").argv.extend(
            a for a in defaults_override if a not in __import__("sys").argv
        )

    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    world_size = _resolve_world_size()

    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.global_batch_size is not None:
        args.grad_accum = max(
            1,
            math.ceil(args.global_batch_size / (args.batch_size_per_device * world_size)),
        )

    torch_dtype, use_bf16, use_fp16, autocast_dtype = _resolve_precision(args.precision)

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset_name = (
        resolve_dataset_name(args.task_suite)
        if args.task_suite else args.dataset_name
    )

    raw_chunk_length = args.segment_length
    if raw_chunk_length < 3:
        raise ValueError("--segment-length must be >= 3 (need >= 1 future step).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build model config ---
    cfg = PixelResidualConfig(
        target_mode        = args.target_mode,
        action_ranges_path = args.action_ranges_path,
        action_dim         = args.action_dim,
        action_horizon     = raw_chunk_length - 2,
        encoder_channels   = args.encoder_channels,
        hidden_dim         = args.hidden_dim,
        n_heads            = args.n_heads,
        n_pred_layers      = args.n_pred_layers,
        ffn_dim            = args.ffn_dim,
        dropout            = args.dropout,
        action_emb_dim     = args.action_emb_dim,
        autocast_dtype     = autocast_dtype,
        lambda_residual    = args.lambda_residual,
        lambda_image       = args.lambda_image,
        lambda_dynamic     = args.lambda_dynamic,
        lambda_gripper     = args.lambda_gripper,
        lambda_static      = args.lambda_static,
        dynamic_threshold  = args.dynamic_threshold,
        dynamic_dilate_kernel = args.dynamic_dilate_kernel,
        roi_crop_size      = args.roi_crop_size,
        save_debug_images  = args.save_debug_images,
        debug_output_dir   = args.debug_output_dir,
    )

    model = PixelResidualWorldModel(cfg, torch_dtype=torch_dtype)

    train_dataset = RldsIterableDataset(
        dataset_name       = dataset_name,
        data_dir           = args.data_root,
        raw_chunk_length   = raw_chunk_length,
        seed               = args.seed,
        shuffle_episodes   = True,
        shuffle_windows    = True,
    )

    # num_workers: with DDP each GPU process handles its own data loading in
    # parallel, so num_workers=0 is acceptable.  The user may override via
    # --num-workers; we warn if num_workers > 0 + DDP (TF/fork incompatibility).
    num_workers = args.num_workers
    if num_workers > 0 and world_size > 1:
        print(
            f"[train_pxr] WARNING: num_workers={num_workers} with DDP (world_size={world_size}) "
            "may conflict with TensorFlow TFDS. Set --num-workers 0 if hangs occur."
        )

    training_kwargs = dict(
        output_dir                  = str(output_dir),
        per_device_train_batch_size = args.batch_size_per_device,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.learning_rate,
        weight_decay                = args.weight_decay,
        warmup_ratio                = args.warmup_ratio,
        lr_scheduler_type           = args.lr_scheduler_type,
        optim                       = args.optim,
        adam_beta1                  = args.adam_beta1,
        adam_beta2                  = args.adam_beta2,
        adam_epsilon                = args.adam_epsilon,
        max_grad_norm               = args.max_grad_norm,
        max_steps                   = args.max_steps,
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        logging_steps               = args.logging_steps,
        save_steps                  = args.save_steps,
        save_total_limit            = args.save_total_limit,
        report_to                   = ["tensorboard"],
        remove_unused_columns       = False,
        dataloader_num_workers      = num_workers,
        logging_dir                 = str(output_dir / "logs"),
        seed                        = args.seed,
        dataloader_pin_memory       = torch.cuda.is_available(),
        # Needed for HF Trainer to correctly detect DDP when called via torchrun
        local_rank                  = int(os.environ.get("LOCAL_RANK", "-1")),
    )
    if world_size > 1:
        training_kwargs["ddp_find_unused_parameters"] = False

    trainer = PixelResidualTrainer(
        model         = model,
        args          = TrainingArguments(**training_kwargs),
        train_dataset = train_dataset,
        data_collator = SimpleCollator(),
    )

    print(f"[train_pxr] target_mode={cfg.target_mode}  "
          f"steps={args.max_steps}  bs/dev={args.batch_size_per_device}  "
          f"grad_accum={args.grad_accum}  world_size={world_size}  "
          f"local_rank={os.environ.get('LOCAL_RANK', '0')}")

    trainer.train()
    trainer.save_model(str(output_dir))

    if trainer.is_world_process_zero():
        # Guarantee the compact per-module checkpoints exist
        unwrapped = trainer.model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        unwrapped.save_pretrained(str(output_dir))

        _save_loss_curves(trainer.state.log_history, output_dir)

        import dataclasses
        summary = {
            "model_type":            "PixelResidualWorldModel",
            "target_mode":           cfg.target_mode,
            "task_suite":            args.task_suite,
            "dataset_name":          dataset_name,
            "max_steps":             args.max_steps,
            "segment_length":        args.segment_length,
            "world_size":            world_size,
            "batch_size_per_device": args.batch_size_per_device,
            "grad_accum":            args.grad_accum,
            "precision":             args.precision,
            "config":                dataclasses.asdict(cfg),
            "timestamp":             datetime.now().isoformat(),
        }
        with open(output_dir / "pixel_residual_training_summary.json",
                  "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[train_pxr] training complete → {output_dir}")


if __name__ == "__main__":
    main()
