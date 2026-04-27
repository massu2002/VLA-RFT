"""LIBERO training entrypoint for LatentResidualWorldModel.

Usage:
    python -m worldmodel.residual_worldmodel.train_libero \\
        --task-suite spatial \\
        --data-root data/modified_libero_rlds \\
        --visual-tokenizer checkpoints/libero/WorldModel/Tokenizer \\
        --output-dir checkpoints/libero/ResidualWorldModel/spatial/run01

The training loop is structurally identical to worldmodel/train.py —
only the model class and model-specific arguments differ.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from ..datasets.libero.data import RldsIterableDataset, resolve_dataset_name
from .config import ResidualWorldModelConfig, add_residual_wm_args
from .model import LatentResidualWorldModel


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train LatentResidualWorldModel on LIBERO RLDS data."
    )

    # --- Dataset ---
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--task-suite",
        type=str,
        choices=["spatial", "object", "goal", "10"],
        help="LIBERO task suite short name.",
    )
    dataset_group.add_argument(
        "--dataset-name",
        type=str,
        help="RLDS dataset name (TFDS) used directly.",
    )

    parser.add_argument("--data-root", type=str, default="data/modified_libero_rlds")
    parser.add_argument("--visual-tokenizer", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    # --- Training ---
    parser.add_argument("--max-steps", type=int, default=150000)
    parser.add_argument("--segment-length", type=int, default=8,
                        help="Number of frames per training window (= T+1).")
    parser.add_argument("--batch-size-per-device", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    parser.add_argument("--precision", type=str,
                        choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--tf32", action="store_true", default=False)
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")

    # --- Residual world model specific ---
    add_residual_wm_args(parser)

    # --- Action-ranking evaluation ---
    parser.add_argument("--run-action-rank-eval", action="store_true", default=False,
                        help="Enable periodic action-ranking evaluation during training.")
    parser.add_argument("--action-rank-eval-every", type=int, default=500,
                        help="Run action-ranking eval every N global steps.")
    parser.add_argument("--num-action-rank-eval-batches", type=int, default=16,
                        help="Number of probe batches for action-ranking evaluation.")
    parser.add_argument("--num-action-candidates", type=int, default=2,
                        help="Total candidates per state (1 pos + N-1 neg).")
    parser.add_argument("--action-rank-score-progress-weight", type=float, default=1.0,
                        help="Weight of terminal progress in ranking score.")
    parser.add_argument("--action-rank-score-success-weight", type=float, default=0.5,
                        help="Weight of terminal success in ranking score.")
    parser.add_argument("--best-action-rank-metric", type=str, default="pairwise_acc",
                        choices=["pairwise_acc", "top1_acc", "mean_margin"],
                        help="Primary metric for best-model tracking.")
    parser.add_argument("--save-best-action-rank-model", action="store_true", default=True,
                        help="Save best checkpoint based on action-ranking metric.")
    parser.add_argument("--no-save-best-action-rank-model",
                        dest="save_best_action_rank_model", action="store_false")
    parser.add_argument("--action-negative-mode", type=str, default="all",
                        choices=["roll", "noise", "shuffle", "all"],
                        help=(
                            "Negative candidate generation: "
                            "roll=batch-shift, noise=GT+Gaussian, "
                            "shuffle=time-step permutation, all=cycle through all three."
                        ))
    parser.add_argument("--action-negative-noise-std", type=float, default=0.05,
                        help="Std of Gaussian noise for 'noise' negative candidates.")

    return parser


# ---------------------------------------------------------------------------
# Custom Trainer — logs individual loss components
# ---------------------------------------------------------------------------

_LOSS_KEYS = ["loss_progress", "loss_success", "loss_consistency", "loss_reward_proxy"]


class ResidualWMTrainer(Trainer):
    """Trainer subclass that accumulates individual loss components and injects
    them into the periodic log (alongside the standard 'loss' key).

    Works by:
      1. compute_loss() extracts loss_* tensors returned by model.forward()
         and accumulates them between logging steps.
      2. log() drains the accumulator and appends averaged values to `logs`
         before delegating to the parent implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_accum: dict = {}
        self._loss_steps: int = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        # Accumulate individual loss components (detached floats)
        for k in _LOSS_KEYS:
            if k in outputs:
                v = outputs[k]
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                self._loss_accum[k] = self._loss_accum.get(k, 0.0) + val
        self._loss_steps += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, **kwargs):
        # Inject averaged individual losses when the periodic train loss is logged
        if self._loss_steps > 0 and "loss" in logs:
            for k, total in self._loss_accum.items():
                logs[k] = round(total / self._loss_steps, 6)
            self._loss_accum.clear()
            self._loss_steps = 0
        super().log(logs, **kwargs)


# ---------------------------------------------------------------------------
# Post-training: loss curve plot + final metrics JSON
# ---------------------------------------------------------------------------

def _save_loss_curves(log_history: List[dict], output_dir: Path) -> None:
    """Save loss_curves.png with one subplot per tracked loss."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available — skipping loss_curves.png")
        return

    # Collect train entries (have "loss" key but not "eval_loss")
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_entries  = [e for e in log_history if "eval_loss" in e]

    all_keys = ["loss"] + [k for k in _LOSS_KEYS
                           if any(k in e for e in train_entries)]

    n = len(all_keys)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, key in zip(axes, all_keys):
        # train
        steps = [e["step"] for e in train_entries if key in e]
        vals  = [e[key]    for e in train_entries if key in e]
        if steps:
            ax.plot(steps, vals, label="train", linewidth=1.2)
        # eval (eval uses "eval_" prefix)
        eval_key = "eval_" + key
        esteps = [e["step"] for e in eval_entries if eval_key in e]
        evals  = [e[eval_key] for e in eval_entries if eval_key in e]
        if esteps:
            ax.plot(esteps, evals, label="eval", linewidth=1.2, linestyle="--")
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if steps or esteps:
            ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = output_dir / "loss_curves.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[info] loss_curves.png saved → {out_path}")


def _save_final_metrics(log_history: List[dict], output_dir: Path) -> None:
    """Save final_metrics.json with last/best values for each tracked loss."""
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_entries  = [e for e in log_history if "eval_loss" in e]

    metrics: dict = {}

    # --- train ---
    if train_entries:
        last = train_entries[-1]
        metrics["num_train_steps"] = last.get("step", len(train_entries))
        metrics["final_train_total_loss"] = last.get("loss")
        for k in _LOSS_KEYS:
            if k in last:
                metrics[f"final_train_{k}"] = last[k]

        all_losses = [e["loss"] for e in train_entries if "loss" in e]
        metrics["best_train_total_loss"] = min(all_losses) if all_losses else None

        # Full history
        metrics["train_history"] = [
            {k: e[k] for k in ["step", "loss"] + _LOSS_KEYS if k in e}
            for e in train_entries
        ]

    # --- eval ---
    if eval_entries:
        last_eval = eval_entries[-1]
        metrics["num_eval_steps"] = len(eval_entries)
        metrics["final_eval_total_loss"] = last_eval.get("eval_loss")
        for k in _LOSS_KEYS:
            ek = "eval_" + k
            if ek in last_eval:
                metrics[f"final_eval_{k}"] = last_eval[ek]

        all_eval = [e["eval_loss"] for e in eval_entries if "eval_loss" in e]
        metrics["best_eval_total_loss"] = min(all_eval) if all_eval else None

        metrics["eval_history"] = [
            {k: e[k] for k in ["step", "eval_loss"]
             + ["eval_" + kk for kk in _LOSS_KEYS] if k in e}
            for e in eval_entries
        ]

    out_path = output_dir / "final_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[info] final_metrics.json saved → {out_path}")


# ---------------------------------------------------------------------------
# Action-ranking evaluation
# ---------------------------------------------------------------------------

_NEG_CYCLE = ["roll", "noise", "shuffle"]


def _build_negatives(
    actions: torch.Tensor,   # [B, T, action_dim]
    n_neg: int,
    mode: str,               # "roll" | "noise" | "shuffle" | "all"
    noise_std: float,
) -> tuple[list[torch.Tensor], dict[str, int]]:
    """Generate n_neg negative action candidates.

    Modes:
      "roll"    — batch-dimension shift (other samples' GT actions within the probe batch)
      "noise"   — GT actions + Gaussian noise (same trajectory, imperfect execution)
      "shuffle" — GT actions with time steps randomly permuted
      "all"     — cycle through [roll, noise, shuffle]

    NOTE: True same-task negative (guaranteeing the negative comes from the same task
    as the positive) requires task_id in the batch.  The current data pipeline does not
    expose task_id, so "roll" is the closest approximation: since probe batches come from
    a single task-suite loader, most rolled samples share the same suite.  Proper
    same-task grouping is left as future work requiring data.py extension.

    Returns:
        neg_list: list of n_neg tensors [B, T, action_dim]
        counts:   {"roll": int, "noise": int, "shuffle": int}
    """
    counts: dict[str, int] = {"roll": 0, "noise": 0, "shuffle": 0}
    neg_list: list[torch.Tensor] = []

    types = ([_NEG_CYCLE[i % len(_NEG_CYCLE)] for i in range(n_neg)]
             if mode == "all" else [mode] * n_neg)

    roll_shift = 0
    for neg_type in types:
        if neg_type == "roll":
            roll_shift += 1
            neg = torch.roll(actions, roll_shift, dims=0)
        elif neg_type == "noise":
            neg = actions + torch.randn_like(actions) * noise_std
        else:  # "shuffle" — permute time steps
            perm = torch.randperm(actions.shape[1], device=actions.device)
            neg = actions[:, perm, :]
        neg_list.append(neg)
        counts[neg_type] += 1

    return neg_list, counts


def _collect_probe_batches(
    dataset,
    collator,
    n_batches: int,
    batch_size: int,
) -> List[dict]:
    """Consume the first n_batches from dataset into CPU tensors for use as probe set."""
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator, num_workers=0
    )
    batches: List[dict] = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batches.append({k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()})
    return batches


@torch.no_grad()
def evaluate_action_ranking(
    model,                   # unwrapped LatentResidualWorldModel
    probe_batches: List[dict],
    progress_weight: float = 1.0,
    success_weight: float = 0.5,
    num_candidates: int = 2,
    negative_mode: str = "all",
    noise_std: float = 0.05,
) -> dict:
    """Action-ranking evaluation using open-loop rollout.

    For each probe batch:
      - positive candidate : GT actions from the batch
      - negative candidates: generated by _build_negatives() per negative_mode

    Score per candidate: -(progress_w * terminal_progress + success_w * terminal_success)
    Higher score = predicted to be closer to goal.

    Metrics:
      pairwise_acc            fraction of (pos, neg) pairs where pos outscores neg
      top1_acc                fraction where pos has highest score among all candidates
      mean_margin             mean(pos_score - mean(neg_scores))
      pos/neg_terminal_progress  mean terminal progress_head output
      pos/neg_terminal_success   mean terminal success_head output
    """
    if model.reward_heads is None:
        return {}
    if model.cfg.residual_target_mode != "current_anchor_ctx":
        return {}

    device = next(model.predictor.parameters()).device
    H = model.cfg.reward_rollout_horizon
    n_neg = max(1, num_candidates - 1)

    all_pos_prog, all_neg_prog = [], []
    all_pos_succ, all_neg_succ = [], []
    all_pos_score, all_neg_scores = [], []
    total_counts: dict[str, int] = {"roll": 0, "noise": 0, "shuffle": 0}

    for batch in probe_batches:
        pixels  = batch["pixels"].to(device)   # [B, T+1, H, W, C] uint8
        actions = batch["actions"].to(device)  # [B, T, action_dim]
        B = pixels.shape[0]
        if B < 2:
            continue

        # Encode state: z_curr and ctx_tokens
        pixels_f = pixels.permute(0, 1, 4, 2, 3).float() / 255.0
        ctx_tokens, dyn_tokens = model._encode_both(pixels_f)
        dyn_flat = model._dyn_tokens_to_flat(dyn_tokens)
        z_curr = dyn_flat[:, 0]  # [B, flat_dim]

        # Use episode initial image ctx when configured
        if (model.cfg.ctx_source_mode == "episode_initial_image"
                and "episode_init_pixels" in batch):
            ep_init_f = (batch["episode_init_pixels"].to(device)
                         .permute(0, 3, 1, 2).float() / 255.0)
            ctx_tokens = model._encode_ctx_from_frame(ep_init_f)

        def _terminal_scores(act):
            out = model.rollout_autoregressive(ctx_tokens, z_curr, act, horizon=H)
            sd = out["score_dict"]
            prog = sd["progress"][:, -1].float().cpu()  # [B]
            succ = sd["success"][:, -1].float().cpu()   # [B]
            score = -(progress_weight * prog + success_weight * succ)
            return prog, succ, score

        pos_prog, pos_succ, pos_score = _terminal_scores(actions)

        neg_acts, counts = _build_negatives(actions, n_neg, negative_mode, noise_std)
        for k, v in counts.items():
            total_counts[k] += v

        neg_prog_list, neg_succ_list, neg_score_list = [], [], []
        for neg_act in neg_acts:
            p, s, sc = _terminal_scores(neg_act)
            neg_prog_list.append(p)
            neg_succ_list.append(s)
            neg_score_list.append(sc)

        # Stack negatives: [n_neg, B]
        neg_scores_t = torch.stack(neg_score_list, dim=0)  # [n_neg, B]

        all_pos_prog.append(pos_prog)
        all_neg_prog.append(torch.stack(neg_prog_list).mean(0))
        all_pos_succ.append(pos_succ)
        all_neg_succ.append(torch.stack(neg_succ_list).mean(0))
        all_pos_score.append(pos_score)
        all_neg_scores.append(neg_scores_t)

    if not all_pos_score:
        return {}

    # Log candidate breakdown
    total_neg = sum(total_counts.values())
    print(
        f"[action_rank] candidates: 1 pos + {n_neg} neg  |  "
        f"total_neg={total_neg}  "
        + "  ".join(f"{k}={v}" for k, v in total_counts.items() if v > 0)
    )

    pos_score_all = torch.cat(all_pos_score)               # [N]
    neg_scores_all = torch.cat(all_neg_scores, dim=1)      # [n_neg, N]

    # pairwise_acc: pos beats each individual neg
    pairwise = (pos_score_all.unsqueeze(0) > neg_scores_all).float().mean().item()
    # top1_acc: pos beats ALL negs simultaneously
    top1 = (pos_score_all.unsqueeze(0) > neg_scores_all).all(dim=0).float().mean().item()
    # mean_margin: pos_score - mean(neg_scores)
    margin = (pos_score_all - neg_scores_all.mean(dim=0)).mean().item()

    return {
        "pairwise_acc":           round(pairwise, 6),
        "top1_acc":               round(top1, 6),
        "mean_margin":            round(margin, 6),
        "pos_terminal_progress":  round(torch.cat(all_pos_prog).mean().item(), 6),
        "neg_terminal_progress":  round(torch.cat(all_neg_prog).mean().item(), 6),
        "pos_terminal_success":   round(torch.cat(all_pos_succ).mean().item(), 6),
        "neg_terminal_success":   round(torch.cat(all_neg_succ).mean().item(), 6),
    }


class ActionRankingCallback(TrainerCallback):
    """TrainerCallback that runs action-ranking evaluation at fixed step intervals.

    Also tracks the best checkpoint according to `best_metric` and saves it to
    `output_dir/best_action_rank/`.
    """

    _METRIC_HIGHER_IS_BETTER = {
        "pairwise_acc": True,
        "top1_acc":     True,
        "mean_margin":  True,
    }

    def __init__(
        self,
        probe_batches: List[dict],
        output_dir: Path,
        eval_every: int,
        progress_weight: float,
        success_weight: float,
        num_candidates: int,
        best_metric: str,
        save_best: bool,
        negative_mode: str = "all",
        noise_std: float = 0.05,
    ):
        self.probe_batches   = probe_batches
        self.output_dir      = output_dir
        self.eval_every      = eval_every
        self.progress_weight = progress_weight
        self.success_weight  = success_weight
        self.num_candidates  = num_candidates
        self.best_metric     = best_metric
        self.save_best       = save_best
        self.negative_mode   = negative_mode
        self.noise_std       = noise_std

        self._best: dict | None = None   # best metrics dict (includes global_step)
        self._history: List[dict] = []   # all eval results in chronological order

    def _is_better(self, metrics: dict) -> bool:
        if self._best is None:
            return True
        higher = self._METRIC_HIGHER_IS_BETTER.get(self.best_metric, True)
        cur  = metrics.get(self.best_metric, float("-inf"))
        prev = self._best.get(self.best_metric, float("-inf"))
        if cur != prev:
            return cur > prev if higher else cur < prev
        # Tiebreak: top1_acc then mean_margin
        for tb in ["top1_acc", "mean_margin"]:
            c, p = metrics.get(tb, 0.0), self._best.get(tb, 0.0)
            if c != p:
                return c > p
        return False

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return
        if not state.is_local_process_zero:
            return

        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.eval()
        metrics = evaluate_action_ranking(
            unwrapped,
            self.probe_batches,
            progress_weight=self.progress_weight,
            success_weight=self.success_weight,
            num_candidates=self.num_candidates,
            negative_mode=self.negative_mode,
            noise_std=self.noise_std,
        )
        unwrapped.train()

        if not metrics:
            return

        metrics["global_step"] = state.global_step
        print(
            f"[action_rank | step={state.global_step}] "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "global_step")
        )

        self._history.append(dict(metrics))
        self._save_rank_curves()
        self._save_rank_history_json()

        if self._is_better(metrics):
            self._best = dict(metrics)
            if self.save_best:
                self._save_best(unwrapped, metrics)

    def _save_best(self, model, metrics: dict) -> None:
        best_dir = self.output_dir / "best_action_rank"
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(best_dir))
        out_path = best_dir / "best_action_rank_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[action_rank] best updated → {best_dir}  "
              f"{self.best_metric}={metrics[self.best_metric]:.4f}")

    def _save_rank_curves(self) -> None:
        """Save action_rank_eval_curves.png with one line per metric."""
        if not self._history:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warn] matplotlib not available — skipping action_rank_eval_curves.png")
            return

        _RANK_METRICS = [
            "pairwise_acc",
            "top1_acc",
            "mean_margin",
            "pos_terminal_progress",
            "neg_terminal_progress",
            "pos_terminal_success",
            "neg_terminal_success",
        ]

        steps = [e["global_step"] for e in self._history]
        fig, ax = plt.subplots(figsize=(10, 5))
        for key in _RANK_METRICS:
            vals = [e.get(key) for e in self._history]
            if any(v is not None for v in vals):
                ax.plot(steps, vals, marker="o", markersize=3, linewidth=1.2, label=key)

        ax.set_title("Action-Ranking Evaluation Metrics")
        ax.set_xlabel("global_step")
        ax.set_ylabel("metric value")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = self.output_dir / "action_rank_eval_curves.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[action_rank] curves saved → {out_path}")

    def _save_rank_history_json(self) -> None:
        """Save action_rank_eval_history.json with full chronological history."""
        if not self._history:
            return
        out_path = self.output_dir / "action_rank_eval_history.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)

    def on_train_end(self, args, state, control, **kwargs):
        """Log the overall best result and save final curves/history."""
        if not state.is_local_process_zero:
            return
        self._save_rank_curves()
        self._save_rank_history_json()
        if self._best:
            print(
                f"[action_rank] best checkpoint: step={self._best['global_step']}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in self._best.items()
                            if k != "global_step")
            )


# ---------------------------------------------------------------------------
# Utilities (same pattern as worldmodel/train.py)
# ---------------------------------------------------------------------------

class SimpleCollator:
    def __call__(self, batch):
        out = {
            "pixels":  torch.stack([item["pixels"]  for item in batch], dim=0),
            "actions": torch.stack([item["actions"] for item in batch], dim=0),
        }
        # Episode metadata — only present when include_episode_metadata=True in dataset
        if "episode_init_pixels" in batch[0]:
            out["episode_init_pixels"] = torch.stack(
                [item["episode_init_pixels"] for item in batch], dim=0
            )
            out["episode_goal_pixels"] = torch.stack(
                [item["episode_goal_pixels"] for item in batch], dim=0
            )
            out["window_start"]   = torch.stack([item["window_start"]   for item in batch], dim=0)
            out["episode_length"] = torch.stack([item["episode_length"] for item in batch], dim=0)
        return out


def _resolve_world_size() -> int:
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
        return torch.float16, False, True, "fp16"
    return torch.float32, False, False, "fp32"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    world_size = _resolve_world_size()

    if args.batch_size_per_device < 1:
        raise ValueError("--batch-size-per-device must be >= 1")
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

    if args.task_suite is not None:
        dataset_name = resolve_dataset_name(args.task_suite)
    else:
        dataset_name = args.dataset_name

    raw_chunk_length = args.segment_length
    if raw_chunk_length < 2:
        raise ValueError("--segment-length must be >= 2")
    if args.residual_target_mode == "current_anchor_ctx" and raw_chunk_length < 3:
        raise ValueError(
            "--segment-length must be >= 3 when --residual-target-mode=current_anchor"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build model config ---
    cfg = ResidualWorldModelConfig(
        visual_tokenizer_path=args.visual_tokenizer,
        action_ranges_path=args.action_ranges_path,
        action_dim=args.action_dim,
        tokenizer_micro_batch_size=args.tokenizer_micro_batch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        recon_loss_weight=args.recon_loss_weight,
        ctx_dim=args.ctx_dim,
        residual_target_mode=args.residual_target_mode,
        autocast_dtype=autocast_dtype,
        # Context source
        ctx_source_mode=args.ctx_source_mode,
        # Temporal residual history
        residual_history_len=args.residual_history_len,
        # Reward-aligned loss
        use_reward_aligned_loss=args.use_reward_aligned_loss,
        progress_target_mode=args.progress_target_mode,
        normalize_remaining_steps=args.normalize_remaining_steps,
        success_target_mode=args.success_target_mode,
        goal_distance_space=args.goal_distance_space,
        goal_image_source=args.goal_image_source,
        reward_head_hidden_dim=args.reward_head_hidden_dim,
        reward_head_activation=args.reward_head_activation,
        use_reward_proxy_head=args.use_reward_proxy_head,
        loss_weight_progress=args.loss_weight_progress,
        loss_weight_success=args.loss_weight_success,
        loss_weight_reward_proxy=args.loss_weight_reward_proxy,
        loss_weight_consistency=args.loss_weight_consistency,
        # Horizon parameters
        teacher_forced_horizon=args.teacher_forced_horizon,
        autoregressive_horizon=args.autoregressive_horizon,
        reward_rollout_horizon=args.reward_rollout_horizon,
        visualize_rollout_horizon=args.visualize_rollout_horizon,
        action_start_offset=args.action_start_offset,
    )

    # --- Build model ---
    model = LatentResidualWorldModel(
        visual_tokenizer_path=args.visual_tokenizer,
        cfg=cfg,
        torch_dtype=torch_dtype,
    )

    # --- Build dataset ---
    # Episode metadata (init/goal pixels, window_start, episode_length) is only needed
    # for reward-aligned loss modes that use remaining_steps or goal_image_distance targets.
    need_episode_meta = (
        args.use_reward_aligned_loss
        and args.residual_target_mode == "current_anchor_ctx"
    )
    train_dataset = RldsIterableDataset(
        dataset_name=dataset_name,
        data_dir=args.data_root,
        raw_chunk_length=raw_chunk_length,
        seed=args.seed,
        shuffle_episodes=True,
        shuffle_windows=True,
        include_episode_metadata=need_episode_meta,
    )

    # --- Build HF Trainer ---
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

    collator = SimpleCollator()

    # --- Action-ranking probe batches (collected before training starts) ---
    callbacks = []
    use_action_rank = (
        args.run_action_rank_eval
        and cfg.residual_target_mode == "current_anchor_ctx"
        and cfg.use_reward_aligned_loss
    )
    if use_action_rank:
        print(f"[info] Collecting {args.num_action_rank_eval_batches} probe batches "
              f"for action-ranking eval (every {args.action_rank_eval_every} steps)...")
        probe_batches = _collect_probe_batches(
            train_dataset, collator,
            n_batches=args.num_action_rank_eval_batches,
            batch_size=args.batch_size_per_device,
        )
        print(f"[info] Collected {len(probe_batches)} probe batches.")
        callbacks.append(ActionRankingCallback(
            probe_batches   = probe_batches,
            output_dir      = output_dir,
            eval_every      = args.action_rank_eval_every,
            progress_weight = args.action_rank_score_progress_weight,
            success_weight  = args.action_rank_score_success_weight,
            num_candidates  = args.num_action_candidates,
            best_metric     = args.best_action_rank_metric,
            save_best       = args.save_best_action_rank_model,
            negative_mode   = args.action_negative_mode,
            noise_std       = args.action_negative_noise_std,
        ))

    trainer = ResidualWMTrainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    # Explicitly save predictor.pt + residual_worldmodel_config.json to the root output dir.
    # trainer.save_model() calls model.save_pretrained() but may only save model.safetensors
    # in DDP mode.  This call guarantees the compact predictor-only checkpoint exists.
    if trainer.is_world_process_zero():
        unwrapped = trainer.model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        unwrapped.save_pretrained(str(output_dir))

    if trainer.is_world_process_zero():
        _save_loss_curves(trainer.state.log_history, output_dir)
        _save_final_metrics(trainer.state.log_history, output_dir)

    if trainer.is_world_process_zero():
        import dataclasses
        summary = {
            "model_type": "LatentResidualWorldModel",
            "task_suite": args.task_suite,
            "dataset_name": dataset_name,
            "visual_tokenizer": args.visual_tokenizer,
            "max_steps": args.max_steps,
            "segment_length": args.segment_length,
            "world_size": world_size,
            "batch_size_per_device": args.batch_size_per_device,
            "grad_accum": args.grad_accum,
            "global_batch_size": args.batch_size_per_device * args.grad_accum * world_size,
            "precision": args.precision,
            "resolved_torch_dtype": str(torch_dtype),
            "config": dataclasses.asdict(cfg),
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_dir / "residual_worldmodel_training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
