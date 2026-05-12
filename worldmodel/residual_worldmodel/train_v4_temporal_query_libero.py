"""Training entry point for TemporalDynamicQueryResidualWM (v4) on LIBERO RLDS.

Usage:
    python -m worldmodel.residual_worldmodel.train_v4_temporal_query_libero \\
        --task-suite spatial \\
        --output-dir checkpoints/libero/PixelResidualWM/spatial/temporal_query_residual/v4a/s42 \\
        --history-length 2 \\
        --num-dynamic-queries 8 \\
        --no-action-future-scorer \\
        --max-steps 50000

    # v4b (with ranking head)
    python -m worldmodel.residual_worldmodel.train_v4_temporal_query_libero \\
        --task-suite spatial \\
        --output-dir checkpoints/libero/.../v4b/s42 \\
        --use-action-future-scorer \\
        --lambda-rank 1.0

Smoke test:
    SMOKE=1 python -m worldmodel.residual_worldmodel.train_v4_temporal_query_libero \\
        --task-suite spatial --output-dir /tmp/v4_smoke --max-steps 5

Environment variable overrides (mirrors existing train_pixel_residual_libero.py):
    HISTORY_LENGTH, NUM_DYNAMIC_QUERIES
    USE_ACTION_FUTURE_SCORER, LAMBDA_RANK, RANK_MARGIN
    LAMBDA_IMAGE, LAMBDA_DYNAMIC, LAMBDA_STATIC, LAMBDA_QUERY, LAMBDA_SPARSE
    USE_MOTION_BIAS, DYNAMIC_THRESHOLD, ACTION_CONDITIONING_MODE
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from transformers import Trainer, TrainingArguments

from ..datasets.libero.data import RldsIterableDataset, resolve_dataset_name
from .v4_config import TemporalQueryResidualConfig, add_v4_args
from .models.temporal_query_residual_wm import TemporalDynamicQueryResidualWM


# ---------------------------------------------------------------------------
# Loss keys logged per-step
# ---------------------------------------------------------------------------

# Model output keys (accumulated from outputs["..."])
_V4_LOSS_KEYS = [
    "loss_image", "loss_dynamic", "loss_static",
    "loss_query", "loss_rank",
    "loss_entropy", "loss_diversity",          # mask 正則化 (改良版 L_sparse)
    "copy_current_mse", "mse_over_copy",       # copy-current collapse 監視
    "score_zero_action",                       # zero-action baseline (ActionFutureScorer)
]

# Collator diagnostic keys (accumulated from batch metadata, NOT model outputs)
# Logged with same interval as loss; useful for checking negative quality.
_V4_NEG_STAT_KEYS = [
    "neg_task_match_rate",        # fraction of negatives from same task as positive
    "num_same_task_negative",     # count: same_task_other_window negatives
    "num_temporal_perm_negative", # count: temporal permutation negatives
    "num_action_noise_negative",  # count: action_noise negatives
    "num_zero_random_negative",   # count: zero or random action negatives
    "num_batch_roll_fallback",    # count: batch-roll fallback (no same-task candidate)
]


# ---------------------------------------------------------------------------
# Custom Trainer
# ---------------------------------------------------------------------------

class V4Trainer(Trainer):
    """Accumulates v4 per-component losses and negative-type stats between log intervals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_accum: dict = {}
        self._loss_steps: int = 0

    @staticmethod
    def _to_log_scalar(value, *, reduce: str = "mean") -> float:
        """Convert scalar or multi-element tensors to a Python float for logging."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return 0.0
            value_f = value.detach().float()
            if reduce == "sum":
                return float(value_f.sum().item())
            return float(value_f.mean().item())
        return float(value)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop collator diagnostic stats before passing inputs to model.
        # These are Python scalars (not tensors), prefixed with "_neg_".
        neg_stats = {k: inputs.pop(k) for k in list(inputs) if k.startswith("_neg_")}

        outputs = model(**inputs)
        loss = outputs["loss"]

        # Accumulate model output components
        for k in _V4_LOSS_KEYS:
            if k in outputs:
                v = outputs[k]
                self._loss_accum[k] = self._loss_accum.get(k, 0.0) + self._to_log_scalar(v)

        # Accumulate collator negative-type stats (key: strip leading "_neg_")
        # "_neg_task_match_rate" → "neg_task_match_rate"
        # "_neg_num_same_task"   → "neg_num_same_task"  etc.
        for k, v in neg_stats.items():
            stat_key = k[1:]   # strip exactly one leading "_"
            reduce = "mean" if stat_key == "neg_task_match_rate" else "sum"
            self._loss_accum[stat_key] = (
                self._loss_accum.get(stat_key, 0.0)
                + self._to_log_scalar(v, reduce=reduce)
            )

        self._loss_steps += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, **kwargs):
        if self._loss_steps > 0 and "loss" in logs:
            for k, total in self._loss_accum.items():
                logs[k] = round(total / self._loss_steps, 6)

            # copy-current collapse 警告
            moc = logs.get("mse_over_copy", None)
            if moc is not None and moc >= 1.0:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "[copy-current collapse] mse_over_copy=%.4f >= 1.0 at step %s "
                    "— model is worse than copy-current. Check zero-init decoder / lambda_image.",
                    moc, logs.get("step", "?"),
                )

            # negative quality 警告: same-task率が低い = cross-task easy negativeばかり
            ntmr = logs.get("neg_task_match_rate", None)
            if ntmr is not None and ntmr < 0.3:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "[low same-task negative rate] neg_task_match_rate=%.3f < 0.3 at step %s "
                    "— most negatives are cross-task (too easy). "
                    "Increase batch size or verify task_id extraction.",
                    ntmr, logs.get("step", "?"),
                )

            self._loss_accum.clear()
            self._loss_steps = 0
        super().log(logs, **kwargs)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class V4Collator:
    """Collate RLDS batch items and construct per-sample mixed negatives.

    Negative mix (default / overridable by NEGATIVE_MIX env var):
      same_task_other_window : 0.50  — same task, different window (hard negative)
      temporal_permutation   : 0.25  — same episode, shuffled horizon (hard negative)
      zero_random            : 0.25  — zero or uniform-random actions (easy anchor)

    same_task fallback:
      If no same-task candidate exists in the batch, falls back to batch-roll
      (actions[(i+1)%B]).  neg_task_match_rate tracks how often same-task succeeded.

    Why this matters for Phase 1:
      ActionFutureScorer must learn "does this action sequence match this history+state?"
      Cross-task negatives (easy) make the scorer learn task identity instead.
      same_task_other_window forces it to discriminate within-task action quality.
    """

    DEFAULT_NEGATIVE_MIX = "same_task_other_window:0.5,temporal_permutation:0.25,zero_random:0.25"
    VALID_NEGATIVE_TYPES = {"same_task_other_window", "temporal_permutation", "zero_random", "action_noise"}
    DEFAULT_ACTION_NOISE_STD = 0.15

    def __init__(
        self,
        history_length: int,
        action_horizon: int,
        negative_mix: str = "",
        action_noise_std: float = DEFAULT_ACTION_NOISE_STD,
    ) -> None:
        self.K = history_length
        self.H = action_horizon
        self.action_noise_std = float(os.environ.get("ACTION_NOISE_STD", action_noise_std))

        spec = os.environ.get("NEGATIVE_MIX", negative_mix or "")
        if not spec.strip():
            neg_type = os.environ.get("NEGATIVE_TYPE", "mixed").strip()
            neg_type_aliases = {
                "same_task": "same_task_other_window",
                "temporal_perm": "temporal_permutation",
                "batch_roll": "same_task_other_window",
            }
            neg_type = neg_type_aliases.get(neg_type, neg_type)
            if neg_type == "mixed":
                spec = self.DEFAULT_NEGATIVE_MIX
            elif neg_type in self.VALID_NEGATIVE_TYPES:
                spec = f"{neg_type}:1.0"
            else:
                warnings.warn(
                    f"Unknown NEGATIVE_TYPE='{neg_type}', falling back to default mixed negative mix.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                spec = self.DEFAULT_NEGATIVE_MIX
        aliases = {
            "same_task": "same_task_other_window",
        }
        raw: dict = {}
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            k, v = part.split(":", 1)
            key = k.strip()
            if key in aliases:
                canonical = aliases[key]
                warnings.warn(
                    f"NEGATIVE_MIX key '{key}' is deprecated; use '{canonical}'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                key = canonical
            raw[key] = raw.get(key, 0.0) + float(v.strip())
        total = sum(raw.values()) or 1.0
        mix = {k: v / total for k, v in raw.items()}

        # Build sorted cumulative thresholds for the four types
        self._thresholds: list = []
        cumsum = 0.0
        for name in ["same_task_other_window", "temporal_permutation", "action_noise", "zero_random"]:
            cumsum += mix.get(name, 0.0)
            self._thresholds.append((cumsum, name))

    def _sample_neg_type(self, r: float) -> str:
        for threshold, name in self._thresholds:
            if r < threshold:
                return name
        return "zero_random"

    def __call__(self, batch: list) -> dict:
        import random as _rng

        pixels  = torch.stack([item["pixels"]  for item in batch], dim=0)
        actions = torch.stack([item["actions"] for item in batch], dim=0)
        B, T_act, D_act = actions.shape
        K, H = self.K, self.H

        # task_id: use sentinel unique per sample if not available (no false matches)
        task_ids = [
            item["task_id"].item()
            if "task_id" in item and isinstance(item["task_id"], torch.Tensor)
            else -(i + 1)        # unique negative → never matches
            for i, item in enumerate(batch)
        ]

        negative_actions = actions.clone()
        num_same_task    = 0
        num_temp_perm    = 0
        num_action_noise = 0
        num_zero_random  = 0
        num_roll_fallbk  = 0
        same_task_count  = 0   # negatives whose task == positive's task

        for i in range(B):
            neg_type = self._sample_neg_type(_rng.random())
            tid = task_ids[i]

            if neg_type == "same_task_other_window":
                candidates = [j for j in range(B) if j != i and task_ids[j] == tid]
                if candidates:
                    j = _rng.choice(candidates)
                    negative_actions[i] = actions[j]
                    num_same_task += 1
                    same_task_count += 1
                else:
                    # No same-task candidate in batch — fall back to batch-roll
                    j = (i + 1) % B
                    negative_actions[i] = actions[j]
                    num_roll_fallbk += 1
                    if task_ids[j] == tid:
                        same_task_count += 1

            elif neg_type == "temporal_permutation":
                # Permute only the prediction horizon slice [K+1 : K+1+H]
                perm = torch.randperm(H)
                if H > 1 and (perm == torch.arange(H)).all():
                    perm = torch.roll(perm, 1)
                negative_actions[i, K + 1: K + 1 + H] = (
                    actions[i, K + 1: K + 1 + H][perm]
                )
                num_temp_perm += 1
                same_task_count += 1   # same episode → same task

            elif neg_type == "action_noise":
                # Add Gaussian noise scaled by per-dim std of the prediction horizon.
                # Matches eval _make_action_noise_action: use actual std when > 1e-6,
                # fall back to 1.0 only for near-zero std (static actions).
                # Same episode → same task.
                horizon_acts = actions[i, K + 1: K + 1 + H]   # [H, D]
                scale = horizon_acts.std(dim=0)                 # [D]
                scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
                noise = torch.randn(H, D_act, dtype=actions.dtype) * scale * self.action_noise_std
                negative_actions[i, K + 1: K + 1 + H] = horizon_acts + noise
                num_action_noise += 1
                same_task_count += 1   # same episode → same task

            else:  # zero_random
                horizon_acts = actions[:, K + 1: K + 1 + H]   # [B, H, D]
                if _rng.random() < 0.5:
                    negative_actions[i, K + 1: K + 1 + H] = torch.zeros(
                        H, D_act, dtype=actions.dtype
                    )
                else:
                    lo = horizon_acts.flatten(0, 1).min(0)[0]   # [D]
                    hi = horizon_acts.flatten(0, 1).max(0)[0]   # [D]
                    negative_actions[i, K + 1: K + 1 + H] = (
                        torch.rand(H, D_act, dtype=actions.dtype) * (hi - lo + 1e-6) + lo
                    )
                num_zero_random += 1
                # zero/random: does not count as same-task match

        neg_task_match_rate = same_task_count / max(B, 1)

        return {
            "pixels":           pixels,
            "actions":          actions,
            "negative_actions": negative_actions,
            # Collator diagnostic stats — stripped in V4Trainer.compute_loss.
            # Shape [1] (not scalar) so accelerate can torch.cat them across DDP devices.
            "_neg_task_match_rate":         torch.tensor([neg_task_match_rate], dtype=torch.float32),
            "_neg_num_same_task":           torch.tensor([num_same_task],       dtype=torch.long),
            "_neg_num_temporal_perm":       torch.tensor([num_temp_perm],       dtype=torch.long),
            "_neg_num_action_noise":        torch.tensor([num_action_noise],    dtype=torch.long),
            "_neg_num_zero_random":         torch.tensor([num_zero_random],     dtype=torch.long),
            "_neg_num_batch_roll_fallback": torch.tensor([num_roll_fallbk],    dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Utilities (mirrors train_pixel_residual_libero.py)
# ---------------------------------------------------------------------------

def _resolve_world_size() -> int:
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
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    all_plot_keys = _V4_LOSS_KEYS + _V4_NEG_STAT_KEYS
    keys = ["loss"] + [k for k in all_plot_keys if any(k in e for e in train_entries)]
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
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TemporalDynamicQueryResidualWM (v4) on LIBERO RLDS data."
    )

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--task-suite", type=str,
                               choices=["spatial", "object", "goal", "10"])
    dataset_group.add_argument("--dataset-name", type=str)
    parser.add_argument("--data-root", type=str, default="/localdata/modified_libero_rlds")
    parser.add_argument("--output-dir", type=str, required=True)

    # Training loop
    parser.add_argument("--max-steps",            type=int,   default=50000)
    parser.add_argument("--segment-length",        type=int,   default=0,
                        help="Override total segment length. 0 = auto (history+horizon+2).")
    parser.add_argument("--batch-size-per-device", type=int,   default=1)
    parser.add_argument("--grad-accum",            type=int,   default=8)
    parser.add_argument("--global-batch-size",     type=int,   default=None)
    parser.add_argument("--learning-rate",         type=float, default=5e-5)
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

    # Warm-start: load weights from a prior checkpoint before training begins.
    # Weights are loaded with strict=False so mismatched keys (e.g. new scorer
    # weights in v4b initialised from a v4a checkpoint) are safely skipped.
    parser.add_argument("--init-from-checkpoint", type=str, default="",
                        help="Path to a saved model dir (contains config.json + pytorch_model.bin). "
                             "Loaded with strict=False before training; useful for 2-stage training.")

    # Negative sampling
    parser.add_argument("--action-noise-std", type=float, default=V4Collator.DEFAULT_ACTION_NOISE_STD,
                        help="Std multiplier for action_noise training negatives.")

    # v4-specific model args
    add_v4_args(parser)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Env-var overrides (mirrors existing scripts)
    _ENV_MAP = [
        ("HISTORY_LENGTH",             "--history-length"),
        ("NUM_DYNAMIC_QUERIES",        "--num-dynamic-queries"),
        ("LAMBDA_IMAGE",               "--lambda-image"),
        ("LAMBDA_DYNAMIC",             "--lambda-dynamic"),
        ("LAMBDA_STATIC",              "--lambda-static"),
        ("LAMBDA_QUERY",               "--lambda-query"),
        ("LAMBDA_RANK",                "--lambda-rank"),
        ("LAMBDA_SPARSE",              "--lambda-sparse"),
        ("LAMBDA_DIVERSITY",           "--lambda-diversity"),
        ("RANK_TEMPERATURE",           "--rank-temperature"),
        ("RANK_MARGIN",                "--rank-margin"),
        ("DYNAMIC_THRESHOLD",          "--dynamic-threshold"),
        ("ROI_CROP_SIZE",              "--roi-crop-size"),
        ("ACTION_CONDITIONING_MODE",   "--action-conditioning-mode"),
        ("RESIDUAL_OUTPUT_ACTIVATION", "--residual-output-activation"),
        ("RESIDUAL_OUTPUT_SCALE",      "--residual-output-scale"),
        ("PIXEL_OUTPUT_ACTIVATION",    "--pixel-output-activation"),
        ("INIT_FROM_CHECKPOINT",       "--init-from-checkpoint"),
        ("ACTION_NOISE_STD",           "--action-noise-std"),
    ]
    import sys
    argv = sys.argv
    for env_key, arg_name in _ENV_MAP:
        val = os.environ.get(env_key)
        if val and arg_name not in argv:
            argv.extend([arg_name, val])

    _BOOL_ENV_MAP = [
        ("USE_ACTION_FUTURE_SCORER", "--use-action-future-scorer", "--no-action-future-scorer"),
        ("USE_MOTION_BIAS",          "--use-motion-bias",          "--no-motion-bias"),
    ]
    for env_key, yes_arg, no_arg in _BOOL_ENV_MAP:
        val = os.environ.get(env_key)
        if val and yes_arg not in argv and no_arg not in argv:
            if val in ("0", "false", "FALSE", "False", "no", "NO"):
                argv.append(no_arg)
            else:
                argv.append(yes_arg)

    # SMOKE / DRY_RUN shortcut
    if os.environ.get("SMOKE", "0") in ("1", "true") or os.environ.get("DRY_RUN", "0") in ("1", "true"):
        for a, v in [("--max-steps", "5"), ("--batch-size-per-device", "2"),
                     ("--save-steps", "5"), ("--logging-steps", "1")]:
            if a not in argv:
                argv.extend([a, v])

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

    # Resolve dataset name
    dataset_name = (
        resolve_dataset_name(args.task_suite)
        if args.task_suite else args.dataset_name
    )

    # Segment length: K + 1 (context slot) + 1 (current) + H (future)
    K = args.history_length
    H = args.action_horizon
    auto_seg_len = K + H + 2   # total frames per window
    if args.segment_length > 0:
        seg_len = args.segment_length
        if seg_len < auto_seg_len:
            raise ValueError(
                f"--segment-length {seg_len} < required {auto_seg_len} "
                f"for history_length={K} action_horizon={H}"
            )
    else:
        seg_len = auto_seg_len

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- Issue 3
    # フレームレイアウト確認ログ
    # history[-1] と current の間に context slot(1フレーム)が存在するため、
    # エピソード上では 2ステップの gap がある。residual_queries はこの gap を
    # 含んだ差分を学習することに注意。
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="[%(levelname)s] %(message)s")
    logger = _log.getLogger(__name__)
    logger.info(
        "v4 frame layout (seg_len=%d):\n"
        "  pixels[:,   0:%d]   → history  (episode idx: start … start+%d)\n"
        "  pixels[:,   %d]     → context slot (NOT used by model)\n"
        "  pixels[:,   %d]     → current frame (episode idx: start+%d)\n"
        "  pixels[:, %d:%d]   → future GT  (H=%d steps)\n"
        "  *** history[-1] to current GAP = 2 episode steps (context slot skipped) ***\n"
        "  actions[:,  0:%d]   → history+context (skipped by model)\n"
        "  actions[:, %d:%d]  → prediction horizon (H=%d steps)",
        seg_len,
        K, K - 1,
        K,
        K + 1, K + 1,
        K + 2, K + 2 + H, H,
        K + 1,
        K + 1, K + 1 + H, H,
    )
    # ----------------------------------------------------------------

    # Build v4 config
    cfg = TemporalQueryResidualConfig(
        target_mode              = args.target_mode,
        model_generation         = args.model_generation,
        action_ranges_path       = args.action_ranges_path,
        history_length           = args.history_length,
        num_dynamic_queries      = args.num_dynamic_queries,
        encoder_channels         = args.encoder_channels,
        hidden_dim               = args.hidden_dim,
        n_heads                  = args.n_heads,
        n_context_layers         = args.n_context_layers,
        n_fuser_layers           = args.n_fuser_layers,
        n_scorer_layers          = args.n_scorer_layers,
        ffn_dim                  = args.ffn_dim,
        dropout                  = args.dropout,
        action_dim               = args.action_dim,
        action_bins              = args.action_bins,
        action_horizon           = H,
        action_emb_dim           = args.action_emb_dim,
        action_conditioning_mode = args.action_conditioning_mode,
        autocast_dtype           = autocast_dtype,
        use_motion_bias          = args.use_motion_bias,
        use_action_future_scorer = args.use_action_future_scorer,
        lambda_image             = args.lambda_image,
        lambda_dynamic           = args.lambda_dynamic,
        lambda_static            = args.lambda_static,
        lambda_query             = args.lambda_query,
        lambda_rank              = args.lambda_rank,
        lambda_sparse            = args.lambda_sparse,
        lambda_diversity         = args.lambda_diversity,
        rank_temperature         = args.rank_temperature,
        rank_margin              = args.rank_margin,
        residual_output_activation = args.residual_output_activation,
        residual_output_scale    = args.residual_output_scale,
        pixel_output_activation  = args.pixel_output_activation,
        dynamic_threshold        = args.dynamic_threshold,
        dynamic_dilate_kernel    = args.dynamic_dilate_kernel,
        roi_crop_size            = args.roi_crop_size,
        save_debug_images        = args.save_debug_images,
        debug_output_dir         = args.debug_output_dir,
    )

    model = TemporalDynamicQueryResidualWM(cfg, torch_dtype=torch_dtype)

    # 2-stage warm-start: load per-module weights from a prior checkpoint.
    # Each module is loaded independently with strict=False so that:
    #   - modules absent in the source (e.g. action_future_scorer for v4a→v4b)
    #     remain randomly initialised.
    #   - modules present in both source and target are loaded exactly.
    if args.init_from_checkpoint:
        ckpt_path = Path(args.init_from_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"--init-from-checkpoint path does not exist: {ckpt_path}"
            )
        _module_names = [
            ("encoder",            model.encoder),
            ("decoder",            model.decoder),
            ("act_encoder",        model.act_encoder),
            ("spatial_proj",       model.spatial_proj),
            ("spatial_unproj",     model.spatial_unproj),
            ("act_proj",           model.act_proj),
            ("query_extractor",    model.query_extractor),
            ("temporal_predictor", model.temporal_predictor),
            ("token_fuser",        model.token_fuser),
        ]
        if model.action_future_scorer is not None:
            _module_names.append(("action_future_scorer", model.action_future_scorer))
        loaded, skipped = [], []
        for mod_name, mod in _module_names:
            pt_file = ckpt_path / f"{mod_name}.pt"
            if pt_file.exists():
                state = torch.load(str(pt_file), map_location="cpu")
                result = mod.load_state_dict(state, strict=False)
                loaded.append(mod_name)
                if result.missing_keys or result.unexpected_keys:
                    logger.info(
                        "  %s: missing=%s unexpected=%s",
                        mod_name, result.missing_keys, result.unexpected_keys,
                    )
            else:
                skipped.append(mod_name)
        logger.info(
            "Warm-start from %s: loaded=%s  skipped(not in src)=%s",
            ckpt_path, loaded, skipped,
        )

    train_dataset = RldsIterableDataset(
        dataset_name     = dataset_name,
        data_dir         = args.data_root,
        raw_chunk_length = seg_len,
        seed             = args.seed,
        shuffle_episodes = True,
        shuffle_windows  = True,
    )

    num_workers = args.num_workers
    if num_workers > 0 and world_size > 1:
        import warnings
        warnings.warn(
            f"--num-workers={num_workers} with DDP (world_size={world_size}) may cause "
            "TF/fork compatibility issues. Consider --num-workers 0."
        )

    training_args = TrainingArguments(
        output_dir              = str(output_dir),
        max_steps               = args.max_steps,
        per_device_train_batch_size = args.batch_size_per_device,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate           = args.learning_rate,
        lr_scheduler_type       = args.lr_scheduler_type,
        warmup_ratio            = args.warmup_ratio,
        weight_decay            = args.weight_decay,
        adam_beta1              = args.adam_beta1,
        adam_beta2              = args.adam_beta2,
        adam_epsilon            = args.adam_epsilon,
        max_grad_norm           = args.max_grad_norm,
        optim                   = args.optim,
        seed                    = args.seed,
        save_steps              = args.save_steps,
        save_total_limit        = args.save_total_limit,
        logging_steps           = args.logging_steps,
        logging_dir             = str(output_dir / "logs"),
        dataloader_num_workers  = num_workers,
        bf16                    = use_bf16,
        fp16                    = use_fp16,
        tf32                    = args.tf32,
        report_to               = [],
        remove_unused_columns   = False,
        label_names             = [],
        ddp_find_unused_parameters = False,
    )

    neg_mix_spec = os.environ.get("NEGATIVE_MIX", V4Collator.DEFAULT_NEGATIVE_MIX)
    logger.info("Negative mix: %s", neg_mix_spec)

    trainer = V4Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        data_collator   = V4Collator(
            history_length  = cfg.history_length,
            action_horizon  = cfg.action_horizon,
            action_noise_std = args.action_noise_std,
        ),
    )

    trainer.train()

    # Save final checkpoint
    final_dir = output_dir / "final"
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(str(final_dir))

    # Save run metadata
    run_meta = {
        "target_mode":       cfg.target_mode,
        "model_generation":  cfg.model_generation,
        "history_length":    cfg.history_length,
        "num_dynamic_queries": cfg.num_dynamic_queries,
        "use_action_future_scorer": cfg.use_action_future_scorer,
        "use_motion_bias":   cfg.use_motion_bias,
        "segment_length":    seg_len,
        "max_steps":         args.max_steps,
        "task_suite":        args.task_suite,
        "output_dir":        str(output_dir),
        "timestamp":         datetime.now().isoformat(),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    _save_loss_curves(trainer.state.log_history, output_dir)

    import logging
    logging.getLogger(__name__).info(
        "v4 training complete. Checkpoint: %s", final_dir
    )


if __name__ == "__main__":
    main()
