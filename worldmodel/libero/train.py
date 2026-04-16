from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import LiberoWorldModelIterableDataset, resolve_dataset_name
from .model import LiberoWorldModelTrainer, WorldModelRuntimeConfig


def build_parser():
    parser = argparse.ArgumentParser(description="Train the LIBERO world model with offline RLDS data.")
    parser.add_argument("--task-suite", type=str, choices=["spatial", "object", "goal", "10"], required=True)
    parser.add_argument("--data-root", type=str, default="data/modified_libero_rlds")
    parser.add_argument("--model-template", "--base-model", dest="model_template", type=str, required=True)
    parser.add_argument("--visual-tokenizer", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=150000)
    parser.add_argument("--segment-length", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=1)
    parser.add_argument("--tokenizer-micro-batch-size", type=int, default=4)
    parser.add_argument("--batch-size-per-device", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-scheduler-type", type=str, default="constant")
    parser.add_argument("--precision", type=str, choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--tf32", action="store_true", default=False)
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.add_argument("--load-pretrained-weights", action="store_true", default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    return parser


class SimpleCollator:
    def __call__(self, batch):
        pixels = torch.stack([item["pixels"] for item in batch], dim=0)
        actions = torch.stack([item["actions"] for item in batch], dim=0)
        return {"pixels": pixels, "actions": actions}


def resolve_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def resolve_precision(precision: str) -> tuple[torch.dtype, bool, bool, str]:
    if precision == "bf16":
        return torch.bfloat16, True, False, "bf16"
    if precision == "fp16":
        return torch.float16, False, True, "fp16"
    if precision == "fp32":
        return torch.float32, False, False, "fp32"

    if torch.cuda.is_available():
        return torch.float16, False, True, "fp16"
    return torch.float32, False, False, "fp32"


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    world_size = resolve_world_size()

    if args.batch_size_per_device < 1:
        raise ValueError("--batch-size-per-device must be at least 1")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be at least 1")
    if args.global_batch_size is not None:
        if args.global_batch_size < 1:
            raise ValueError("--global-batch-size must be at least 1")
        args.grad_accum = max(1, math.ceil(args.global_batch_size / (args.batch_size_per_device * world_size)))

    torch_dtype, use_bf16, use_fp16, autocast_dtype = resolve_precision(args.precision)

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    task_dataset_name = resolve_dataset_name(args.task_suite)
    raw_chunk_length = args.segment_length
    if raw_chunk_length < 2:
        raise ValueError("--segment-length must be at least 2")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_template, use_fast=True)
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    runtime_cfg = WorldModelRuntimeConfig(
        action_ranges_path="train/verl/ivideogpt/configs/libero_action_ranges.pth",
        tokenizer_micro_batch_size=args.tokenizer_micro_batch_size,
        context_length=args.context_length,
        action_dim=7,
        action_bins=256,
        max_length=1663,
        visual_token_num=-1,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        tokens_per_frame=64,
        autocast_dtype=autocast_dtype,
        processor_type="ctx_msp",
        use_img_gt_ac=False,
        interact=False,
    )

    model = LiberoWorldModelTrainer(
        lm_path=args.model_template,
        visual_tokenizer_path=args.visual_tokenizer,
        runtime_cfg=runtime_cfg,
        torch_dtype=torch_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        load_pretrained_weights=args.load_pretrained_weights,
    )
    runtime_cfg.visual_token_num = int(model.visual_tokenizer.num_vq_embeddings)
    model.processor.config.visual_token_num = runtime_cfg.visual_token_num

    train_dataset = LiberoWorldModelIterableDataset(
        dataset_name=task_dataset_name,
        data_dir=args.data_root,
        raw_chunk_length=raw_chunk_length,
        seed=args.seed,
        shuffle_episodes=True,
        shuffle_windows=True,
    )

    training_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        logging_dir=str(output_dir / "logs"),
        seed=args.seed,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    if world_size > 1:
        training_kwargs["ddp_find_unused_parameters"] = False

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SimpleCollator(),
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if trainer.is_world_process_zero():
        summary = {
            "task_suite": args.task_suite,
            "dataset_name": task_dataset_name,
            "model_template": args.model_template,
            "load_pretrained_weights": args.load_pretrained_weights,
            "visual_tokenizer": args.visual_tokenizer,
            "max_steps": args.max_steps,
            "segment_length": args.segment_length,
            "raw_chunk_length": raw_chunk_length,
            "world_size": world_size,
            "batch_size_per_device": args.batch_size_per_device,
            "grad_accum": args.grad_accum,
            "global_batch_size": args.batch_size_per_device * args.grad_accum * world_size,
            "precision": args.precision,
            "resolved_torch_dtype": str(torch_dtype),
            "autocast_dtype": autocast_dtype,
            "bf16": use_bf16,
            "fp16": use_fp16,
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_dir / "worldmodel_training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
