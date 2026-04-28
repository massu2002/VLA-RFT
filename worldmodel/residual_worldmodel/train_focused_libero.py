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
from .tiered_rank_eval import (
    TieredEvalDataset,
    build_tiered_eval_dataset,
    run_tiered_rank_eval,
)
from .rank_benchmark import (
    RankingBenchmark,
    load_or_build_benchmark,
    run_ranking_benchmark_eval,
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
# Residual-analysis diagnostic helpers (A-F)
# ---------------------------------------------------------------------------

def _pearson_corr_batch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between x and y averaged over batch. x,y: [B, N]."""
    x = x.float();  y = y.float()
    xc = x - x.mean(dim=1, keepdim=True)
    yc = y - y.mean(dim=1, keepdim=True)
    num = (xc * yc).sum(dim=1)
    den = (xc.norm(dim=1) * yc.norm(dim=1)).clamp(min=1e-8)
    return (num / den).mean().item()


def _image_grad_l1(img: torch.Tensor) -> float:
    """Mean L1 spatial-gradient magnitude. img: [B,3,H,W] in [0,1]."""
    img = img.float()
    gx = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    gy = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
    return ((gx + gy) / 2).item()


def _binary_entropy_mean(p: torch.Tensor) -> float:
    """Mean binary entropy of a probability tensor [B, N]."""
    p = p.float().clamp(1e-6, 1 - 1e-6)
    return (-(p * p.log() + (1 - p) * (1 - p).log())).mean().item()


def _batch_iou_soft(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> float:
    """Mean IoU (binary at threshold) over batch. pred, gt: [B, N]."""
    pb = (pred.float() >= threshold).float()
    gb = (gt.float()   >= threshold).float()
    inter = (pb * gb).sum(dim=1)
    union = (pb + gb - pb * gb).sum(dim=1).clamp(min=1)
    return (inter / union).mean().item()


def _batch_dice_soft(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> float:
    """Mean Dice (binary at threshold) over batch. pred, gt: [B, N]."""
    pb = (pred.float() >= threshold).float()
    gb = (gt.float()   >= threshold).float()
    inter = (pb * gb).sum(dim=1)
    denom = (pb.sum(dim=1) + gb.sum(dim=1)).clamp(min=1)
    return (2 * inter / denom).mean().item()


def _compute_phase_diagnostics(
    out: dict,
    cfg,
    *,
    prefix: str = "diag/",
) -> dict:
    """Compute A-F diagnostic metrics from a full model forward() output dict.

    Covers both single-mask (legacy) and dual-mask (Phase 4) modes.
    All tensors are detached / float cast here — no gradient risk.
    """
    result: dict = {}

    focus_map     = out.get("focus_map")                     # [B, N]  = context_mask alias
    context_mask_t = out.get("context_mask")                 # [B, N]  (Phase 4)
    write_mask_t  = out.get("write_mask")                    # [B, N]  (Phase 4)
    delta_tok     = out.get("predicted_delta_tokens")        # [B, N, D]
    residual      = out.get("predicted_residual_image")      # [B, 3, H, W]
    pred_img      = out.get("pred_future_image")             # [B, 3, H, W]
    current_f     = out.get("current_pixels_float")          # [B, 3, H, W] float32
    future_f      = out.get("future_pixels_float")           # [B, 3, H, W] or None

    if focus_map is None or delta_tok is None or residual is None:
        return result

    # Resolve context/write masks (fall back to single focus_map in legacy mode)
    ctx_t = context_mask_t if context_mask_t is not None else focus_map
    wrt_t = write_mask_t   if write_mask_t   is not None else focus_map

    with torch.no_grad():
        ctx = ctx_t.detach().float()  # [B, N] context mask
        wrt = wrt_t.detach().float()  # [B, N] write   mask
        fg  = ctx                      # alias: "foreground" = context_mask
        bg  = 1.0 - fg
        bg_wrt = 1.0 - wrt

        # ---- A. Dual-mask basic statistics ------------------------------
        result[f"{prefix}context_mask_mean"]    = ctx.mean().item()
        result[f"{prefix}write_mask_mean"]      = wrt.mean().item()
        result[f"{prefix}context_mask_entropy"] = _binary_entropy_mean(ctx)
        result[f"{prefix}write_mask_entropy"]   = _binary_entropy_mean(wrt)
        result[f"{prefix}write_context_gap"]    = (wrt - ctx).abs().mean().item()
        result[f"{prefix}write_context_ratio"]  = (wrt.mean() / (ctx.mean() + 1e-6)).item()
        result[f"{prefix}context_write_overlap"] = (ctx.clamp(0, 1) * wrt.clamp(0, 1)).mean().item()

        # ---- A. Token-update locality — context + write weighted --------
        dt_f = delta_tok.detach().float()
        dn   = dt_f.norm(dim=-1)          # [B, N]

        ctx_fn = (ctx * dn).sum() / (ctx.sum() + 1e-8)
        ctx_bn = (bg  * dn).sum() / (bg.sum()  + 1e-8)
        wrt_fn = (wrt * dn).sum() / (wrt.sum() + 1e-8)
        wrt_bn = (bg_wrt * dn).sum() / (bg_wrt.sum() + 1e-8)

        result[f"{prefix}delta_norm"]                 = dn.mean().item()
        result[f"{prefix}context_fg_delta_norm"]      = ctx_fn.item()
        result[f"{prefix}context_bg_delta_norm"]      = ctx_bn.item()
        result[f"{prefix}context_bg_fg_delta_ratio"]  = (ctx_bn / (ctx_fn + 1e-6)).item()
        result[f"{prefix}write_fg_delta_norm"]        = wrt_fn.item()
        result[f"{prefix}write_bg_delta_norm"]        = wrt_bn.item()
        result[f"{prefix}write_bg_fg_delta_ratio"]    = (wrt_bn / (wrt_fn + 1e-6)).item()
        # Legacy keys (used by existing log lines)
        result[f"{prefix}fg_delta_norm"]              = ctx_fn.item()
        result[f"{prefix}bg_delta_norm"]              = ctx_bn.item()
        result[f"{prefix}bg_fg_delta_ratio"]          = (ctx_bn / (ctx_fn + 1e-6)).item()

        # Shift = actual token update = write * delta
        actual_shift = (wrt.unsqueeze(-1) * dt_f).norm(dim=-1)  # [B, N]
        result[f"{prefix}future_token_shift_l2"]   = actual_shift.mean().item()
        result[f"{prefix}context_fg_shift"]        = ((ctx    * actual_shift).sum() / (ctx.sum()    + 1e-8)).item()
        result[f"{prefix}context_bg_shift"]        = ((bg     * actual_shift).sum() / (bg.sum()     + 1e-8)).item()
        result[f"{prefix}write_fg_shift"]          = ((wrt    * actual_shift).sum() / (wrt.sum()    + 1e-8)).item()
        result[f"{prefix}write_bg_shift"]          = ((bg_wrt * actual_shift).sum() / (bg_wrt.sum() + 1e-8)).item()
        # Legacy token-shift keys
        result[f"{prefix}fg_token_shift_l2"]       = result[f"{prefix}context_fg_shift"]
        result[f"{prefix}bg_token_shift_l2"]       = result[f"{prefix}context_bg_shift"]

        # ---- A. Correlations: context + write vs delta / residual ------
        _r    = residual.detach().float()   # [B,3,H,W]
        _rabs = _r.abs()
        B_, _, H_, W_ = _r.shape
        ph    = cfg.patch_hw
        res_pool = F.adaptive_avg_pool2d(_rabs, (ph, ph))  # [B,3,ph,ph]
        res_e    = res_pool.mean(dim=1).reshape(B_, -1)     # [B, N]

        result[f"{prefix}corr_context_delta_norm"]       = _pearson_corr_batch(ctx, dn)
        result[f"{prefix}corr_write_delta_norm"]         = _pearson_corr_batch(wrt, dn)
        result[f"{prefix}corr_context_residual_energy"]  = _pearson_corr_batch(ctx, res_e)
        result[f"{prefix}corr_write_residual_energy"]    = _pearson_corr_batch(wrt, res_e)
        # Legacy
        result[f"{prefix}corr_focus_delta_norm"]         = result[f"{prefix}corr_context_delta_norm"]
        result[f"{prefix}corr_focus_residual_energy"]    = result[f"{prefix}corr_context_residual_energy"]

        # ---- D. Image-residual locality — context + write weighted ------
        ctx_2d   = ctx.reshape(B_, 1, ph, ph)
        ctx_img  = F.interpolate(ctx_2d, size=(H_, W_), mode="bilinear", align_corners=False)
        bg_ctx_img = 1.0 - ctx_img
        wrt_2d   = wrt.reshape(B_, 1, ph, ph)
        wrt_img  = F.interpolate(wrt_2d, size=(H_, W_), mode="bilinear", align_corners=False)
        bg_wrt_img = 1.0 - wrt_img

        ctx_fgr = (ctx_img   * _rabs).sum() / (ctx_img.sum()   * 3 + 1e-8)
        ctx_bgr = (bg_ctx_img * _rabs).sum() / (bg_ctx_img.sum() * 3 + 1e-8)
        wrt_fgr = (wrt_img   * _rabs).sum() / (wrt_img.sum()   * 3 + 1e-8)
        wrt_bgr = (bg_wrt_img * _rabs).sum() / (bg_wrt_img.sum() * 3 + 1e-8)

        result[f"{prefix}residual_l1_mean"]              = _rabs.mean().item()
        result[f"{prefix}residual_l2_mean"]              = _r.pow(2).mean().sqrt().item()
        result[f"{prefix}residual_abs_max"]              = _rabs.amax().item()
        result[f"{prefix}residual_signed_mean"]          = _r.mean().item()
        result[f"{prefix}context_fg_residual_l1"]        = ctx_fgr.item()
        result[f"{prefix}context_bg_residual_l1"]        = ctx_bgr.item()
        result[f"{prefix}context_bg_fg_residual_ratio"]  = (ctx_bgr / (ctx_fgr + 1e-6)).item()
        result[f"{prefix}write_fg_residual_l1"]          = wrt_fgr.item()
        result[f"{prefix}write_bg_residual_l1"]          = wrt_bgr.item()
        result[f"{prefix}write_bg_fg_residual_ratio"]    = (wrt_bgr / (wrt_fgr + 1e-6)).item()
        # Legacy keys
        result[f"{prefix}fg_residual_l1"]                = ctx_fgr.item()
        result[f"{prefix}bg_residual_l1"]                = ctx_bgr.item()
        result[f"{prefix}bg_fg_residual_ratio"]          = (ctx_bgr / (ctx_fgr + 1e-6)).item()

        # ---- B. Dual-mask vs change target (only when available) --------
        ctgt = out.get("dino_change_target")   # [B, N] or None
        if ctgt is not None:
            ct = ctgt.detach().float()
            result[f"{prefix}context_iou"]              = _batch_iou_soft(ctx, ct)
            result[f"{prefix}context_dice"]             = _batch_dice_soft(ctx, ct)
            result[f"{prefix}write_iou"]                = _batch_iou_soft(wrt, ct)
            result[f"{prefix}write_dice"]               = _batch_dice_soft(wrt, ct)
            result[f"{prefix}write_minus_context_iou"]  = (
                result[f"{prefix}write_iou"] - result[f"{prefix}context_iou"]
            )
            result[f"{prefix}write_minus_context_dice"] = (
                result[f"{prefix}write_dice"] - result[f"{prefix}context_dice"]
            )

        # ---- C, F, G: require GT future ---------------------------------
        if current_f is not None and future_f is not None and pred_img is not None:
            cur  = current_f.detach().float()
            fut  = future_f.detach().float()
            pred = pred_img.detach().float()

            # C: change magnitude ratio
            gt_change   = (fut - cur).abs().mean()
            pred_change = (pred - cur).abs().mean()
            result[f"{prefix}gt_future_change_l1"]    = gt_change.item()
            result[f"{prefix}pred_future_change_l1"]  = pred_change.item()
            result[f"{prefix}change_magnitude_ratio"] = (pred_change / (gt_change + 1e-6)).item()

            # F: recon error — context + write mask weighted
            err   = (pred - fut).abs()      # [B,3,H,W]
            err_sq = (pred - fut).pow(2)

            # Use change_target for fg mask if available, else write_mask
            if ctgt is not None:
                _ct = ctgt.detach().float().reshape(B_, 1, ph, ph)
                fg_mask = F.interpolate(_ct, size=(H_, W_), mode="bilinear", align_corners=False)
            else:
                fg_mask = wrt_img

            fgrl1 = (fg_mask      * err).sum()    / (fg_mask.sum()      * 3 + 1e-8)
            bgrl1 = ((1 - fg_mask) * err).sum()   / ((1 - fg_mask).sum() * 3 + 1e-8)
            fgmse = (fg_mask      * err_sq).sum() / (fg_mask.sum()      * 3 + 1e-8)
            bgmse = ((1 - fg_mask) * err_sq).sum() / ((1 - fg_mask).sum() * 3 + 1e-8)
            result[f"{prefix}fg_recon_l1"]        = fgrl1.item()
            result[f"{prefix}bg_recon_l1"]        = bgrl1.item()
            result[f"{prefix}fg_recon_mse"]       = fgmse.item()
            result[f"{prefix}bg_recon_mse"]       = bgmse.item()
            result[f"{prefix}fg_bg_recon_ratio"]  = (bgrl1 / (fgrl1 + 1e-6)).item()

            # Context-mask weighted recon
            ctx_fgl1 = (ctx_img     * err).sum() / (ctx_img.sum()     * 3 + 1e-8)
            ctx_bgl1 = (bg_ctx_img  * err).sum() / (bg_ctx_img.sum()  * 3 + 1e-8)
            # Write-mask weighted recon
            wrt_fgl1 = (wrt_img     * err).sum() / (wrt_img.sum()     * 3 + 1e-8)
            wrt_bgl1 = (bg_wrt_img  * err).sum() / (bg_wrt_img.sum()  * 3 + 1e-8)
            result[f"{prefix}fg_recon_l1_context"]      = ctx_fgl1.item()
            result[f"{prefix}bg_recon_l1_context"]      = ctx_bgl1.item()
            result[f"{prefix}fg_recon_l1_write"]        = wrt_fgl1.item()
            result[f"{prefix}bg_recon_l1_write"]        = wrt_bgl1.item()
            result[f"{prefix}fg_bg_recon_ratio_write"]  = (wrt_bgl1 / (wrt_fgl1 + 1e-6)).item()

            # G: sharpness
            pg = _image_grad_l1(pred)
            gg = _image_grad_l1(fut)
            result[f"{prefix}pred_grad_l1"]  = pg
            result[f"{prefix}gt_grad_l1"]    = gg
            result[f"{prefix}grad_l1_gap"]   = pg - gg

    return result


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

        # prefer strict_order_acc (tiered) over legacy pairwise_acc
        rank_acc = metrics.get("strict_order_acc") or metrics.get("pairwise_acc")
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
        # --- Tiered 3-layer eval dataset (built lazily from probe_batches) ---
        self._tiered_eval_dataset: Optional[TieredEvalDataset] = None
        # --- Fixed ranking benchmark (built/loaded once, opt-in) ---
        self._ranking_benchmark: Optional[RankingBenchmark] = None

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
        _STEP_STAT_KEYS = (
            # A: token update locality — context-mask side
            "delta_norm", "fg_delta_norm", "bg_delta_norm", "bg_fg_delta_ratio",
            "future_token_shift_l2", "fg_future_token_shift_l2", "bg_future_token_shift_l2",
            # A: token update locality — write-mask side (Phase 4)
            "fg_delta_norm_write", "bg_delta_norm_write", "bg_fg_delta_ratio_write",
            # B: image residual locality — context-mask side
            "residual_l1_mean", "residual_l2_mean", "residual_abs_max",
            "residual_signed_mean", "fg_residual_l1", "bg_residual_l1",
            "bg_fg_residual_ratio",
            # B: image residual locality — write-mask side (Phase 4)
            "fg_residual_l1_write", "bg_residual_l1_write", "bg_fg_residual_ratio_write",
            # Phase 4 mask statistics
            "write_mask_mean", "context_mask_mean", "write_context_gap",
            "context_write_overlap", "write_context_ratio",
        )
        extra_stats = {
            k: out[k].item()
            for k in _STEP_STAT_KEYS
            if k in out and hasattr(out[k], "item")
        }
        return {"loss": loss.item(), **comps, **extra_stats}

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

        # Collect probe batches lazily (shared between legacy and tiered eval)
        if self._probe_batches is None:
            self._probe_batches = self._collect_probe_batches(self._n_probe)

        # Legacy pairwise eval (always runs when run_rank_eval=True)
        rank_metrics = run_action_rank_eval(
            self._model_inner(),
            self._probe_batches,
            self.device,
            self.cfg,
        )
        result = {f"rank/{k}": v for k, v in rank_metrics.items()}

        # Tiered 3-layer eval (opt-in via cfg.use_tiered_rank_eval)
        if self.cfg.use_tiered_rank_eval:
            if self._tiered_eval_dataset is None:
                # Optionally limit number of items
                probe_for_tiered = self._probe_batches
                max_items = self.cfg.num_rank_eval_items
                if max_items > 0:
                    # Trim probe batches so total items ≤ max_items
                    trimmed, count = [], 0
                    for b in probe_for_tiered:
                        bs = b["current_pixels"].shape[0]
                        trimmed.append(b)
                        count += bs
                        if count >= max_items:
                            break
                    probe_for_tiered = trimmed

                self._tiered_eval_dataset = build_tiered_eval_dataset(
                    probe_batches=probe_for_tiered,
                    n_near_success=self.cfg.num_near_success_candidates,
                    n_failure=self.cfg.num_failure_candidates,
                    near_noise_std=self.cfg.near_success_noise_std,
                    fail_noise_std=self.cfg.failure_noise_std,
                    seed=getattr(self.args, "seed", 42),
                    task_name=getattr(self.args, "task_suite", ""),
                )
                logger.info(
                    "Built TieredEvalDataset: %d items, K=%d candidates "
                    "(n_ns=%d n_f=%d near_std=%.3f fail_std=%.3f)",
                    self._tiered_eval_dataset.n_items,
                    self._tiered_eval_dataset.K,
                    self.cfg.num_near_success_candidates,
                    self.cfg.num_failure_candidates,
                    self.cfg.near_success_noise_std,
                    self.cfg.failure_noise_std,
                )

            tiered_metrics = run_tiered_rank_eval(
                model=self._model_inner(),
                dataset=self._tiered_eval_dataset,
                device=self.device,
                cfg=self.cfg,
                output_dir=str(self._out),
                step=self._step,
                task_name=getattr(self.args, "task_suite", ""),
                task_suite=getattr(self.args, "task_suite", ""),
            )
            result.update(tiered_metrics)

        # Fixed ranking benchmark (opt-in via cfg.use_fixed_rank_eval_dataset)
        if self.cfg.use_fixed_rank_eval_dataset:
            if self._ranking_benchmark is None:
                try:
                    dataset_name = resolve_dataset_name(
                        getattr(self.args, "task_suite", "spatial")
                    )
                    self._ranking_benchmark = load_or_build_benchmark(
                        out_dir=str(self._out),
                        dataset_name=dataset_name,
                        data_dir=self.args.data_root,
                        segment_length=self.args.segment_length,
                        task_name=getattr(self.args, "task_suite", ""),
                        cfg=self.cfg,
                        seed=self.cfg.fixed_rank_eval_seed,
                        regenerate=self.cfg.regenerate_rank_eval_dataset,
                    )
                    logger.info(
                        "RankingBenchmark ready: %d items  K=%d",
                        self._ranking_benchmark.n_items,
                        self._ranking_benchmark.K,
                    )
                except Exception as _bench_err:
                    logger.warning(
                        "Failed to build/load RankingBenchmark: %s", _bench_err
                    )
                    self._ranking_benchmark = None

            if self._ranking_benchmark is not None:
                bench_metrics = run_ranking_benchmark_eval(
                    model=self._model_inner(),
                    benchmark=self._ranking_benchmark,
                    device=self.device,
                    cfg=self.cfg,
                    output_dir=str(self._out),
                    step=self._step,
                )
                result.update(bench_metrics)

        return result

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
        """No-grad forward pass: DINO/focus metrics + A-F residual diagnostics."""
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

        # DINO feature metrics
        pred_tok = out.get("predicted_future_tokens")
        gt_proj  = out.get("gt_projected_dino")
        if pred_tok is not None and gt_proj is not None:
            dm = dino_feature_metrics(pred_tok, gt_proj)
            result.update({f"feat/{k}": v for k, v in dm.items()})

        # Focus quality metrics
        fm   = out.get("focus_map")
        ctgt = out.get("dino_change_target")
        if fm is not None:
            fmet = focus_metrics(fm, ctgt)
            result.update({f"focus/{k}": v for k, v in fmet.items()})

        # A-F residual diagnostics (Phase 1-3 structural metrics)
        diag = _compute_phase_diagnostics(out, self.cfg, prefix="diag/")
        result.update(diag)

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
        _rank_keys   = {
            "rank/pairwise_acc", "pairwise_acc",
            "rank/top1_acc",     "top1_acc",
            # tiered metrics
            "tiered/strict_order_acc",
            "tiered/acc_success_gt_nearsuccess",
            "tiered/acc_nearsuccess_gt_failure",
            "tiered/acc_success_gt_failure",
            "tiered/spearman_tier_corr",
        }
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
                ("fg_recon",         "fg_recon"),
                ("bg_residual_penalty", "bg_pen"),
                ("fg_grad",          "fg_grad"),
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

            # [Mask] line — dual-mask statistics (A)
            mask_parts = []
            for k, label in [
                ("write_mask_mean",                  "wrt"),
                ("context_mask_mean",                "ctx"),
                ("write_context_ratio",              "wrt/ctx"),
                ("write_context_gap",                "gap"),
                ("diag/write_mask_entropy",          "wrt_ent"),
                ("diag/context_mask_entropy",        "ctx_ent"),
                ("diag/context_write_overlap",       "ovlp"),
                ("diag/write_iou",                   "wrt_iou"),
                ("diag/write_minus_context_iou",     "Δiou"),
            ]:
                if k in metrics:
                    mask_parts.append(f"{label}={metrics[k]:.4f}")
            if mask_parts:
                logger.info("[Mask]    %s", " | ".join(mask_parts))

            # [Delta] line — token-update locality (A, context + write mask)
            delta_parts = []
            for k, label in [
                ("bg_fg_delta_ratio",                "bg/fg_δ_ctx"),
                ("bg_fg_delta_ratio_write",          "bg/fg_δ_wrt"),
                ("diag/context_fg_shift",            "ctx_fg_sh"),
                ("diag/write_fg_shift",              "wrt_fg_sh"),
                ("diag/write_bg_shift",              "wrt_bg_sh"),
                ("diag/corr_context_delta_norm",     "corr_ctx_δ"),
                ("diag/corr_write_delta_norm",       "corr_wrt_δ"),
            ]:
                if k in metrics:
                    delta_parts.append(f"{label}={metrics[k]:.4f}")
            if delta_parts:
                logger.info("[Delta]   %s", " | ".join(delta_parts))

            # [Residual] line — image-residual + change + sharpness (D/C/F, context + write)
            res_parts = []
            for k, label in [
                ("bg_fg_residual_ratio",                  "bg/fg_res_ctx"),
                ("bg_fg_residual_ratio_write",            "bg/fg_res_wrt"),
                ("diag/write_bg_fg_residual_ratio",       "bg/fg_res_wrt(d)"),
                ("diag/change_magnitude_ratio",           "chg_ratio"),
                ("diag/fg_recon_l1_write",                "fg_err_wrt"),
                ("diag/bg_recon_l1_write",                "bg_err_wrt"),
                ("diag/fg_bg_recon_ratio_write",          "bg/fg_err_wrt"),
                ("diag/grad_l1_gap",                      "grad_gap"),
                ("diag/corr_context_residual_energy",     "corr_ctx_res"),
                ("diag/corr_write_residual_energy",       "corr_wrt_res"),
            ]:
                if k in metrics:
                    res_parts.append(f"{label}={metrics[k]:.4f}")
            if res_parts:
                logger.info("[Residual]%s", " | ".join(res_parts))

        # --- [Eval] line (fires whenever legacy ranking eval was run) ---
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

        # --- [TieredRank] line (fires whenever tiered eval was run) ---
        if any(k.startswith("tiered/") for k in eval_metrics):
            tr_parts = [f"step={self._step}"]
            for k, label in [
                ("tiered/strict_order_acc",             "strict"),
                ("tiered/acc_success_gt_nearsuccess",   "s>ns"),
                ("tiered/acc_nearsuccess_gt_failure",   "ns>f"),
                ("tiered/acc_success_gt_failure",       "s>f"),
                ("tiered/margin_success_minus_failure", "margin_s_f"),
                ("tiered/spearman_tier_corr",           "spearman"),
                ("tiered/tier_score_success",           "score_s"),
                ("tiered/tier_score_nearsuccess",       "score_ns"),
                ("tiered/tier_score_failure",           "score_f"),
            ]:
                if k in metrics:
                    tr_parts.append(f"{label}={metrics[k]:.4f}")
            logger.info("[TieredRank] %s", " | ".join(tr_parts))

        # --- [Benchmark] line (fires whenever fixed benchmark eval was run) ---
        if any(k.startswith("bench/") for k in eval_metrics):
            bm_parts = [f"step={self._step}"]
            for k, label in [
                ("bench/strict_order_acc",                  "strict"),
                ("bench/acc_success_gt_nearsuccess",        "s>ns"),
                ("bench/acc_nearsuccess_gt_failure",        "ns>f"),
                ("bench/spearman_tier_corr",                "spearman"),
                ("bench/acc_gt_temporal_neighbor",          "s>tn"),
                ("bench/acc_gt_same_task_hard",             "s>sth"),
                ("bench/acc_gt_same_task_mismatch",         "s>stm"),
                ("bench/mean_score_temporal_neighbor",      "sc_tn"),
                ("bench/mean_score_same_task_hard",         "sc_sth"),
            ]:
                if k in metrics:
                    bm_parts.append(f"{label}={metrics[k]:.4f}")
            logger.info("[Benchmark]  %s", " | ".join(bm_parts))

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
        decoder_upsample_mode=args.decoder_upsample_mode,
        decoder_interp_mode=args.decoder_interp_mode,
        use_decoder_refine=args.use_decoder_refine,
        num_decoder_refine_blocks=args.num_decoder_refine_blocks,
        write_mask_temperature=args.write_mask_temperature,
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
        use_fg_recon_loss=args.use_fg_recon_loss,
        fg_recon_weight=args.fg_recon_weight,
        use_bg_residual_penalty=args.use_bg_residual_penalty,
        bg_residual_weight=args.bg_residual_weight,
        use_fg_grad_loss=args.use_fg_grad_loss,
        fg_grad_weight=args.fg_grad_weight,
        ranking_score_type=args.ranking_score_type,
        ranking_image_weight=args.ranking_image_weight,
        negative_mode=args.negative_mode,
        noise_std=args.noise_std,
        num_action_candidates=args.num_action_candidates,
        # tiered eval
        use_tiered_rank_eval=args.use_tiered_rank_eval,
        num_rank_eval_items=args.num_rank_eval_items,
        num_near_success_candidates=args.num_near_success_candidates,
        num_failure_candidates=args.num_failure_candidates,
        near_success_noise_std=args.near_success_noise_std,
        failure_noise_std=args.failure_noise_std,
        save_rank_eval_json=args.save_rank_eval_json,
        save_rank_eval_csv=args.save_rank_eval_csv,
        save_rank_eval_plots=args.save_rank_eval_plots,
        # fixed ranking benchmark
        use_fixed_rank_eval_dataset=args.use_fixed_rank_eval_dataset,
        rank_eval_dataset_path=args.rank_eval_dataset_path,
        regenerate_rank_eval_dataset=args.regenerate_rank_eval_dataset,
        num_benchmark_pool_episodes=args.num_benchmark_pool_episodes,
        num_benchmark_anchors_per_episode=args.num_benchmark_anchors_per_episode,
        near_success_modes_bench=args.near_success_modes_bench,
        failure_modes_bench=args.failure_modes_bench,
        fixed_rank_eval_seed=args.fixed_rank_eval_seed,
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
