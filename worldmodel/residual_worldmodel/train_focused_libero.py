"""Training entrypoint for ActionConditionedFocusedResidualWM on LIBERO.

Design goals
------------
* DINO-based visual encoder (non-optional).
* Robust training loop: NaN/Inf guard, gradient clipping, AMP safety.
* Multiple checkpoint types: latest, best_recon, best_dino_feature, best_rank.
* Structured logging: jsonl per step + CSV summary + terminal.
* Inline action-ranking evaluation with DINO feature scores (primary).
* Ablation-friendly: variant_tag() embedded in run directory name.

Usage (from repo root)
----------------------
    python -m worldmodel.residual_worldmodel.train_focused_libero \
        --task-suite spatial \
        --data-root data/modified_libero_rlds \
        --output-dir checkpoints/libero/FocusedWM/spatial/run0 \
        --max-steps 15000

TODO(RFT): To connect to policy training, import ActionConditionedFocusedResidualWM
           and call .get_latent_features_for_rft() from the reward shaping step.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from ..datasets.libero.data import RldsIterableDataset, resolve_dataset_name
from .focused_config import FocusedWMConfig, add_focused_wm_args
from .focused_model import (
    ActionConditionedFocusedResidualWM,
    compute_dino_change_target,
    compute_pixel_change_target,
)
from .losses import (
    dino_feature_metrics,
    focus_metrics,
    image_reconstruction_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FocusedWindowDataset(IterableDataset):
    """Yields (current_pixels, future_pixels, actions) triples.

    current_pixels: frame[0]      [H, W, C] uint8
    future_pixels:  frame[T-1]    [H, W, C] uint8
    actions:        actions[0:T-1] [T-1, 7] float32

    Segment_length = T (total frames), action_horizon = T - 1.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        segment_length: int,
        seed: int = 42,
        image_key: str = "image",
    ) -> None:
        super().__init__()
        self.inner = RldsIterableDataset(
            dataset_name=dataset_name,
            data_dir=data_dir,
            raw_chunk_length=segment_length,
            seed=seed,
            shuffle_episodes=True,
            shuffle_windows=True,
            window_stride=1,
            image_key=image_key,
            include_episode_metadata=False,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for batch in self.inner:
            pixels  = batch["pixels"]   # [T, H, W, C] uint8
            actions = batch["actions"]  # [T-1, 7] float32
            if pixels.shape[0] < 2:
                continue
            yield {
                "current_pixels": pixels[0],    # [H, W, C]
                "future_pixels":  pixels[-1],   # [H, W, C]
                "actions":        actions,       # [T-1, 7]
            }


# ---------------------------------------------------------------------------
# Negative generation (for action ranking evaluation)
# ---------------------------------------------------------------------------

_NEG_CYCLE = ["roll", "noise", "shuffle"]


def _build_negatives(
    actions: torch.Tensor,   # [B, H, 7]
    n_neg: int,
    mode: str,
    noise_std: float,
) -> List[torch.Tensor]:
    types = (
        [_NEG_CYCLE[i % len(_NEG_CYCLE)] for i in range(n_neg)]
        if mode == "all"
        else [mode] * n_neg
    )
    negs = []
    roll_shift = 0
    for t in types:
        if t == "roll":
            roll_shift += 1
            negs.append(torch.roll(actions, roll_shift, dims=0))
        elif t == "noise":
            negs.append(actions + torch.randn_like(actions) * noise_std)
        else:  # shuffle
            perm = torch.randperm(actions.shape[1], device=actions.device)
            negs.append(actions[:, perm, :])
    return negs


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return torch.distributed.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return torch.distributed.get_world_size() if _is_dist() else 1


def _is_main() -> bool:
    return _rank() == 0


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([s[k] for s in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Appends metric dicts to a .jsonl file and a .csv file."""

    def __init__(self, output_dir: str) -> None:
        self._dir    = Path(output_dir)
        self._jsonl  = self._dir / "train_metrics.jsonl"
        self._csv    = self._dir / "train_metrics.csv"
        self._writer: Optional[csv.DictWriter] = None
        self._csv_fh = None
        self._fieldnames: Optional[list] = None

    def log(self, metrics: dict) -> None:
        with self._jsonl.open("a") as f:
            f.write(json.dumps(metrics) + "\n")

        if self._fieldnames is None:
            self._fieldnames = list(metrics.keys())
            self._csv_fh = self._csv.open("w", newline="")
            self._writer = csv.DictWriter(self._csv_fh, fieldnames=self._fieldnames, extrasaction="ignore")
            self._writer.writeheader()

        self._writer.writerow({k: metrics.get(k, "") for k in self._fieldnames})
        self._csv_fh.flush()

    def close(self) -> None:
        if self._csv_fh is not None:
            self._csv_fh.close()


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Saves multiple checkpoint types and prunes old ones."""

    _TYPES = ["latest", "best_recon", "best_dino_feature", "best_rank", "final"]

    def __init__(self, output_dir: str, save_total_limit: int = 3) -> None:
        self._dir   = Path(output_dir)
        self._limit = save_total_limit
        self._best: Dict[str, float] = {
            "best_recon":        float("inf"),
            "best_dino_feature": float("inf"),
            "best_rank":        -float("inf"),
        }
        self._history: Dict[str, List[str]] = {t: [] for t in self._TYPES}

    def save(
        self,
        model: nn.Module,
        step: int,
        metrics: dict,
        cfg: FocusedWMConfig,
        force_tag: Optional[str] = None,
    ) -> None:
        """Save 'latest' checkpoint; conditionally save best-type checkpoints."""
        tags = ["latest"]

        recon = metrics.get("recon")
        if recon is not None and recon < self._best["best_recon"]:
            self._best["best_recon"] = recon
            tags.append("best_recon")

        dino_feat = metrics.get("dino_feature")
        if dino_feat is not None and dino_feat < self._best["best_dino_feature"]:
            self._best["best_dino_feature"] = dino_feat
            tags.append("best_dino_feature")

        rank_acc = metrics.get("pairwise_acc")
        if rank_acc is not None and rank_acc > self._best["best_rank"]:
            self._best["best_rank"] = rank_acc
            tags.append("best_rank")

        if force_tag and force_tag not in tags:
            tags.append(force_tag)

        for tag in tags:
            ckpt_dir = self._dir / f"checkpoint-{tag}-step{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            state = model.state_dict()
            torch.save(state, ckpt_dir / "model.pt")

            meta = {**metrics, "step": step, "checkpoint_type": tag}
            (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))

            cfg_dict = {
                k: v for k, v in vars(cfg).items()
                if not callable(v) and not k.startswith("_")
            }
            (ckpt_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))

            self._history[tag].append(str(ckpt_dir))
            self._prune(tag)

            logger.info("[ckpt] saved %s → %s", tag, ckpt_dir)

    def _prune(self, tag: str) -> None:
        history = self._history[tag]
        while len(history) > self._limit:
            old = history.pop(0)
            import shutil
            try:
                shutil.rmtree(old)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Inline action-ranking evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_action_rank_eval(
    model: ActionConditionedFocusedResidualWM,
    probe_batches: List[Dict[str, torch.Tensor]],
    device: torch.device,
    cfg: FocusedWMConfig,
) -> dict:
    """Compute action-ranking metrics on fixed probe batches.

    Positive = GT action.
    Negatives = _build_negatives(GT, n_neg, mode, noise_std).
    Primary score: DINO feature cosine similarity (or image/combined per cfg).

    Returns dict with:
      pairwise_acc, top1_acc, mean_margin,
      pos_score_mean, neg_score_mean,
      hardest_negative_margin.
    """
    model_inner = model
    model_inner.eval()

    n_neg = cfg.num_action_candidates - 1
    all_pairwise, all_top1, all_margins = [], [], []
    all_pos_scores, all_neg_scores, all_hard_margins = [], [], []

    for batch in probe_batches:
        cur  = batch["current_pixels"].to(device)
        fut  = batch["future_pixels"].to(device)
        acts = batch["actions"].to(device)

        negatives = _build_negatives(acts, n_neg, cfg.negative_mode, cfg.noise_std)
        all_candidates = [acts] + negatives  # [1 + n_neg] each [B, H, 7]

        # Stack candidates → [B, K, H, 7]
        K  = len(all_candidates)
        B  = acts.shape[0]
        act_stacked = torch.stack(all_candidates, dim=1)  # [B, K, H, 7]

        scores = model_inner.rank_action_candidates(cur, act_stacked, fut)  # [B, K]
        pos_scores = scores[:, 0]   # [B]
        neg_scores = scores[:, 1:]  # [B, n_neg]

        pairwise = (pos_scores.unsqueeze(1) > neg_scores).float().mean().item()
        top1     = (pos_scores.unsqueeze(1) > neg_scores.max(dim=1, keepdim=True).values).float().squeeze(1).mean().item()
        margin   = (pos_scores.unsqueeze(1) - neg_scores).mean().item()
        hard_margin = (pos_scores - neg_scores.max(dim=1).values).mean().item()

        all_pairwise.append(pairwise)
        all_top1.append(top1)
        all_margins.append(margin)
        all_pos_scores.append(pos_scores.mean().item())
        all_neg_scores.append(neg_scores.mean().item())
        all_hard_margins.append(hard_margin)

    return {
        "pairwise_acc":          sum(all_pairwise) / len(all_pairwise),
        "top1_acc":              sum(all_top1) / len(all_top1),
        "mean_margin":           sum(all_margins) / len(all_margins),
        "pos_score_mean":        sum(all_pos_scores) / len(all_pos_scores),
        "neg_score_mean":        sum(all_neg_scores) / len(all_neg_scores),
        "hardest_negative_margin": sum(all_hard_margins) / len(all_hard_margins),
    }


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class FocusedWMTrainer:
    """Robust custom training loop for ActionConditionedFocusedResidualWM.

    Features:
      * NaN/Inf guard (raises RuntimeError with step info in debug mode)
      * Gradient clipping
      * AMP (bfloat16 or float16) via torch.autocast + GradScaler
      * Multiple checkpoint types
      * jsonl + CSV metric logging
      * Inline action-ranking eval every cfg.eval_every steps
      * debug_mode support (fast_debug / normal / full_report)
    """

    def __init__(self, cfg: FocusedWMConfig, args: argparse.Namespace) -> None:
        self.cfg  = cfg
        self.args = args
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        _dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.torch_dtype = _dtype_map.get(args.precision, torch.bfloat16)
        self.use_amp = args.precision in ("bf16", "fp16")

        # --- Model ---
        logger.info("Building ActionConditionedFocusedResidualWM (%s) …", cfg.model_variant)
        self.model = ActionConditionedFocusedResidualWM(cfg, torch_dtype=self.torch_dtype)
        self.model.to(self.device)

        if _is_dist():
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[_rank()]
            )

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Trainable parameters: %s", f"{n_params:,}")

        # --- Optimiser ---
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )

        total_steps = args.max_steps
        warmup      = int(total_steps * args.warmup_ratio)

        def _lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
        self.scaler    = torch.cuda.amp.GradScaler(
            enabled=(args.precision == "fp16")
        )

        # --- Dataset ---
        dataset_name = resolve_dataset_name(args.task_suite)
        limit = args.debug_steps if cfg.debug_mode == "fast_debug" else None
        dataset = FocusedWindowDataset(
            dataset_name=dataset_name,
            data_dir=args.data_root,
            segment_length=args.segment_length,
            seed=args.seed,
        )
        self.loader = DataLoader(
            dataset,
            batch_size=args.batch_size_per_device,
            collate_fn=_collate,
            num_workers=args.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        # --- Probe batches for action-rank eval ---
        self._probe_batches: Optional[List[Dict[str, torch.Tensor]]] = None
        self._n_probe = args.num_rank_eval_batches

        # --- Output / logging ---
        self._out = Path(args.output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._metric_logger = MetricsLogger(str(self._out)) if _is_main() else None
        self._ckpt_manager  = CheckpointManager(
            str(self._out), save_total_limit=args.save_total_limit
        ) if _is_main() else None

        # Running state
        self._step        = 0
        self._grad_accum  = max(1, args.global_batch_size // (
            args.batch_size_per_device * _world_size()
        ))
        self._accum_count = 0
        self._accum_loss  = 0.0
        self._accum_comps: Dict[str, float] = {}

        # Best metric tracking (for terminal report)
        self._best: Dict[str, float] = {}

        logger.info(
            "Trainer ready | device=%s  dtype=%s  grad_accum=%d  max_steps=%d",
            self.device, self.torch_dtype, self._grad_accum, args.max_steps,
        )

    # ------------------------------------------------------------------ Helpers

    def _model_inner(self) -> ActionConditionedFocusedResidualWM:
        return self.model.module if _is_dist() else self.model

    def _check_finite(self, loss: torch.Tensor, step: int) -> None:
        if not torch.isfinite(loss):
            msg = f"Non-finite loss ({loss.item()}) at step {step}."
            if self.cfg.debug_mode == "fast_debug":
                raise RuntimeError(msg)
            logger.warning(msg + " Skipping update.")
            raise _SkipStep(msg)

    # ------------------------------------------------------------------ Training

    def _train_micro_step(self, batch: Dict[str, torch.Tensor]) -> dict:
        """Single micro-step (before gradient accumulation boundary)."""
        self.model.train()

        cur  = batch["current_pixels"].to(self.device, non_blocking=True)
        fut  = batch["future_pixels"].to(self.device, non_blocking=True)
        acts = batch["actions"].to(self.device, non_blocking=True)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.torch_dtype,
            enabled=self.use_amp,
        ):
            out  = self._model_inner()(cur, acts, fut)
            loss = out["loss"].mean()

        self._check_finite(loss, self._step)

        # Scale for gradient accumulation
        scaled_loss = loss / self._grad_accum
        self.scaler.scale(scaled_loss).backward()

        comps = {k: v.item() for k, v in out["loss_components"].items()
                 if hasattr(v, "item") and k != "total"}
        return {"loss": loss.item(), **comps}

    def _optimizer_step(self) -> float:
        """Apply gradients; return current grad norm."""
        self.scaler.unscale_(self.optimizer)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_grad_norm
            )
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm

    def _maybe_eval(self) -> dict:
        """Run action-ranking eval if it's time. Returns {} if skipped."""
        if not self.args.run_rank_eval:
            return {}
        if self._step % self.cfg.eval_every != 0 or self._step == 0:
            return {}
        if not _is_main():
            return {}

        # Collect probe batches lazily
        if self._probe_batches is None:
            self._probe_batches = self._collect_probe_batches(self._n_probe)

        rank_metrics = run_action_rank_eval(
            self._model_inner(),
            self._probe_batches,
            self.device,
            self.cfg,
        )
        return {f"rank/{k}": v for k, v in rank_metrics.items()}

    def _collect_probe_batches(self, n: int) -> List[Dict[str, torch.Tensor]]:
        """Collect n batches from the data loader for use as fixed probe set."""
        logger.info("Collecting %d probe batches for action-ranking eval …", n)
        probes = []
        it = iter(self.loader)
        for _ in range(n):
            try:
                probes.append(next(it))
            except StopIteration:
                break
        logger.info("Collected %d probe batches.", len(probes))
        return probes

    @torch.no_grad()
    def _compute_feature_metrics(self, batch: Dict[str, torch.Tensor]) -> dict:
        """No-grad forward pass to compute DINO cosine-sim and focus metrics."""
        model = self._model_inner()
        model.eval()
        cur  = batch["current_pixels"].to(self.device, non_blocking=True)
        fut  = batch["future_pixels"].to(self.device, non_blocking=True)
        acts = batch["actions"].to(self.device, non_blocking=True)
        try:
            out = model(cur, acts, fut)
        except Exception:
            model.train()
            return {}
        model.train()

        result: dict = {}

        pred_tok = out.get("predicted_future_tokens")
        gt_proj  = out.get("gt_projected_dino")
        if pred_tok is not None and gt_proj is not None:
            dm = dino_feature_metrics(pred_tok, gt_proj)
            result.update({f"feat/{k}": v for k, v in dm.items()})

        fm   = out.get("focus_map")
        ctgt = out.get("dino_change_target")
        if fm is not None:
            fmet = focus_metrics(fm, ctgt)
            result.update({f"focus/{k}": v for k, v in fmet.items()})

        return result

    def _log_step(self, step_metrics: dict, grad_norm: float, eval_metrics: dict,
                  feature_metrics: dict | None = None) -> None:
        metrics = {
            "step": self._step,
            "lr":   self.scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
            **step_metrics,
            **eval_metrics,
            **(feature_metrics or {}),
        }

        if not _is_main():
            return

        # --- Persist to file ---
        if self._metric_logger:
            self._metric_logger.log(metrics)

        # --- Best tracking (higher = better for rank; lower = better for losses) ---
        _rank_keys   = {"rank/pairwise_acc", "pairwise_acc", "rank/top1_acc", "top1_acc"}
        _loss_keys   = {"recon", "dino_feature"}
        for k, v in metrics.items():
            if k in _rank_keys:
                clean = k.replace("rank/", "")
                old = self._best.get(clean)
                if old is None or v > old:
                    if old is not None:
                        logger.info("[Best] %s  %.4f → %.4f", clean, old, v)
                    self._best[clean] = v
            elif k in _loss_keys:
                old = self._best.get(k)
                if old is None or v < old:
                    if old is not None:
                        logger.info("[Best] %s  %.4f → %.4f", k, old, v)
                    self._best[k] = v

        # --- [Train] line (every logging_steps) ---
        if self._step % self.args.logging_steps == 0:
            train_parts = [f"step={self._step}"]
            for k, label in [
                ("loss",             "loss"),
                ("recon",            "recon"),
                ("dino_feature",     "dino"),
                ("focus_supervision","focus"),
                ("focus_sparsity",   "sparsity"),
            ]:
                if k in metrics:
                    train_parts.append(f"{label}={metrics[k]:.4f}")
            train_parts.append(f"lr={metrics['lr']:.2e}")
            logger.info("[Train]   %s", " | ".join(train_parts))

            # [Feature] line
            feat_parts = []
            for k, label in [
                ("feat/dino_cosine_similarity", "cosine"),
                ("feat/dino_feature_mse",       "feat_mse"),
            ]:
                if k in metrics:
                    feat_parts.append(f"{label}={metrics[k]:.4f}")
            if feat_parts:
                logger.info("[Feature] %s", " | ".join(feat_parts))

            # [Focus] line
            focus_parts = []
            for k, label in [
                ("focus/focus_mean",    "mean"),
                ("focus/focus_entropy", "entropy"),
                ("focus/iou_vs_change", "iou"),
                ("focus/dice_vs_change","dice"),
            ]:
                if k in metrics:
                    focus_parts.append(f"{label}={metrics[k]:.4f}")
            if focus_parts:
                logger.info("[Focus]   %s", " | ".join(focus_parts))

        # --- [Eval] line (fires whenever action-ranking eval was run) ---
        if eval_metrics:
            eval_parts = [f"step={self._step}"]
            for k, label in [
                ("rank/pairwise_acc",           "pairwise"),
                ("rank/top1_acc",               "top1"),
                ("rank/mean_margin",            "margin"),
                ("rank/pos_score_mean",         "pos"),
                ("rank/neg_score_mean",         "neg"),
                ("rank/hardest_negative_margin","hard_margin"),
            ]:
                if k in metrics:
                    eval_parts.append(f"{label}={metrics[k]:.4f}")
            logger.info("[Eval]    %s", " | ".join(eval_parts))

    # ------------------------------------------------------------------ Main loop

    def train(self) -> None:
        loader_iter = iter(self.loader)
        self.optimizer.zero_grad()

        max_steps = (
            self.args.debug_steps
            if self.cfg.debug_mode == "fast_debug"
            else self.args.max_steps
        )

        logger.info("Starting training for %d steps (debug_mode=%s).",
                    max_steps, self.cfg.debug_mode)

        accum_metrics: Dict[str, float] = {}
        accum_count   = 0

        while self._step < max_steps:
            # --- Fetch batch ---------
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)
                batch       = next(loader_iter)

            # --- Micro step ----------
            try:
                step_comps = self._train_micro_step(batch)
            except _SkipStep:
                self.optimizer.zero_grad()
                continue

            for k, v in step_comps.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v
            accum_count += 1

            # --- Gradient update boundary ----------
            is_update_step = (
                accum_count == self._grad_accum
                or self._step == max_steps - 1
            )
            if is_update_step:
                grad_norm = self._optimizer_step()

                avg = {k: v / accum_count for k, v in accum_metrics.items()}
                accum_metrics = {}
                accum_count   = 0

                eval_metrics = self._maybe_eval()

                feature_metrics: dict = {}
                log_now = (_is_main() and self._step % self.args.logging_steps == 0)
                if log_now:
                    feature_metrics = self._compute_feature_metrics(batch)

                self._log_step(avg, grad_norm, eval_metrics, feature_metrics)

                # Save step visualizations
                viz_every = getattr(self.cfg, "save_viz_every", 0)
                if _is_main() and viz_every > 0 and self._step % viz_every == 0:
                    try:
                        from .focused_visualize import save_step_visualizations
                        save_step_visualizations(
                            output_dir=str(self._out),
                            step=self._step,
                            model=self._model_inner(),
                            batch=batch,
                            device=self.device,
                            cfg=self.cfg,
                        )
                    except Exception as _viz_err:
                        logger.warning("Visualization failed at step %d: %s", self._step, _viz_err)

                # Save checkpoint + update training curves plot
                if self.args.save_steps > 0 and self._step % self.args.save_steps == 0:
                    ckpt_metrics = {**avg, **{k.replace("rank/", ""): v
                                              for k, v in eval_metrics.items()}}
                    if _is_main() and self._ckpt_manager:
                        self._ckpt_manager.save(
                            self._model_inner(), self._step, ckpt_metrics, self.cfg
                        )
                        try:
                            from .focused_visualize import plot_training_curves
                            plot_training_curves(str(self._out))
                        except Exception as _plot_err:
                            logger.warning("Training curve plot failed: %s", _plot_err)

            self._step += 1

        # --- Final save ---
        if _is_main():
            if self._ckpt_manager:
                self._ckpt_manager.save(
                    self._model_inner(), self._step, {}, self.cfg,
                    force_tag="final"
                )
            if self._metric_logger:
                self._metric_logger.close()
            # Plot training curves from the completed jsonl log
            try:
                from .focused_visualize import plot_training_curves
                plot_training_curves(str(self._out))
            except Exception as _plot_err:
                logger.warning("Training curve plot failed: %s", _plot_err)
            self._print_summary()

    def _print_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("Training complete.")
        if self._best:
            logger.info("  Best metrics:")
            # Print in a meaningful order
            ordered = ["recon", "dino_feature", "pairwise_acc", "top1_acc"]
            shown = set()
            for k in ordered:
                if k in self._best:
                    logger.info("    %-28s %.5f", k, self._best[k])
                    shown.add(k)
            for k, v in sorted(self._best.items()):
                if k not in shown:
                    logger.info("    %-28s %.5f", k, v)
        logger.info("  Output: %s", self._out)
        logger.info("=" * 60)


class _SkipStep(Exception):
    pass


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ActionConditionedFocusedResidualWM on LIBERO"
    )

    # Data
    parser.add_argument("--task-suite", type=str, default="spatial",
                        choices=["spatial", "object", "goal", "10", "long"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--segment-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)

    # Training
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--batch-size-per-device", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)

    # Ranking eval
    parser.add_argument("--run-rank-eval", action="store_true", default=True)
    parser.add_argument("--no-rank-eval", dest="run_rank_eval", action="store_false")
    parser.add_argument("--num-rank-eval-batches", type=int, default=32)

    # Debug
    parser.add_argument("--debug-steps", type=int, default=20,
                        help="Steps to run in fast_debug mode.")

    # FocusedWM architecture
    add_focused_wm_args(parser)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Initialize distributed process group when running under torchrun.
    # Must happen before any _is_main() / _rank() calls so that rank-0
    # guards work correctly across all processes.
    _dist_world = int(os.environ.get("WORLD_SIZE", "1"))
    if _dist_world > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress INFO from worker processes — only rank-0 output goes to the log.
    if int(os.environ.get("RANK", "0")) != 0:
        logging.getLogger().setLevel(logging.WARNING)

    args = parse_args()

    action_horizon = args.segment_length - 1

    cfg = FocusedWMConfig(
        action_ranges_path=args.action_ranges_path,
        dino_weights_path=args.dino_weights_path,
        dino_model_name=args.dino_model_name,
        dino_input_size=args.dino_input_size,
        dino_frozen=args.dino_frozen,
        dino_finetune_last_n_layers=args.dino_finetune_last_n_layers,
        dino_hub_source=args.dino_hub_source,
        model_variant=args.model_variant,
        action_dim=args.action_dim,
        action_horizon=action_horizon,
        hidden_dim=args.hidden_dim,
        action_emb_dim=args.action_emb_dim,
        n_action_enc_layers=args.n_action_enc_layers,
        n_pred_layers=args.n_pred_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        image_height=args.image_height,
        image_width=args.image_width,
        use_focus_head=args.use_focus_head,
        recon_loss_weight=args.recon_loss_weight,
        use_lpips_loss=args.use_lpips_loss,
        lpips_loss_weight=args.lpips_loss_weight,
        use_dino_feature_loss=args.use_dino_feature_loss,
        dino_feature_loss_weight=args.dino_feature_loss_weight,
        use_dino_focus_supervision=args.use_dino_focus_supervision,
        use_pixel_focus_supervision=args.use_pixel_focus_supervision,
        focus_supervision_weight=args.focus_supervision_weight,
        change_target_threshold=args.change_target_threshold,
        use_focus_sparsity=args.use_focus_sparsity,
        focus_sparsity_mode=args.focus_sparsity_mode,
        focus_sparsity_weight=args.focus_sparsity_weight,
        ranking_score_type=args.ranking_score_type,
        ranking_image_weight=args.ranking_image_weight,
        negative_mode=args.negative_mode,
        noise_std=args.noise_std,
        num_action_candidates=args.num_action_candidates,
        eval_every=args.eval_every,
        save_viz_every=args.save_viz_every,
        debug_mode=args.debug_mode,
        autocast_dtype=args.precision,
    )
    cfg.resolve_debug_defaults()

    if args.global_batch_size is None:
        args.global_batch_size = args.batch_size_per_device * _world_size()

    # Embed variant tag in output directory
    out_base = Path(args.output_dir)
    if cfg.model_variant != "full" or cfg.dino_model_name != "dinov2_vits14":
        tagged = out_base.parent / (out_base.name + "_" + cfg.variant_tag())
        args.output_dir = str(tagged)
        logger.info("Output dir adjusted to: %s", args.output_dir)

    trainer = FocusedWMTrainer(cfg, args)
    trainer.train()

    if _dist_world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
