"""Visualization for ActionConditionedFocusedResidualWM.

Saves all visualizations under:
  {output_dir}/visualizations/libero/{task_suite}/
  {output_dir}/eval_reports/libero/{task_suite}/

Generated artifacts
-------------------
Per-step (called during training at save_viz_every):
  step_{N:06d}/          — directory per step
    batch_examples.png   — grid: current | GT future | pred future | focus overlay
    change_targets.png   — grid: DINO-change target overlay on current frame

Full evaluation (called offline or at end of training):
  full_eval/
    metrics.json         — image reconstruction + focus + DINO feature metrics
    casebook.png         — representative examples (best/median/worst by recon loss)
    focus_analysis.png   — focus map statistics plots
    candidate_ranking.png — candidate comparison grid
    per_task_summary.csv — per-task metrics

Usage
-----
    from worldmodel.residual_worldmodel.focused_visualize import (
        save_step_visualizations,
        run_full_evaluation,
    )
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..eval_roi_utils import load_roi_config, get_goal_roi_center, get_roi_half

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not available; visualizations will be skipped.")

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    """[3, H, W] float [0,1] or [H, W, C] uint8 → [H, W, 3] uint8 numpy."""
    t = tensor.detach().cpu()
    is_uint8 = (t.dtype == torch.uint8)
    t = t.float()
    if t.dim() == 3 and t.shape[0] == 3:   # [3, H, W] → [H, W, 3]
        t = t.permute(1, 2, 0)
    t = (t / 255.0) if is_uint8 else t.clamp(0.0, 1.0)
    return (t.numpy() * 255).astype(np.uint8)


def _overlay_heatmap(
    image_np: np.ndarray,        # [H, W, 3] uint8
    heatmap: np.ndarray,         # [H, W] float [0, 1]
    alpha: float = 0.5,
    colormap: str = "hot",
) -> np.ndarray:
    """Overlay a heatmap on an image.  Returns [H, W, 3] uint8."""
    if not _HAS_MPL:
        return image_np
    cmap = cm.get_cmap(colormap)
    heat_rgba = (cmap(heatmap) * 255).astype(np.uint8)[:, :, :3]
    blended = (alpha * heat_rgba + (1 - alpha) * image_np).astype(np.uint8)
    return blended


def _focus_map_to_spatial(
    focus_map: torch.Tensor,  # [N] or [B, N]
    patch_hw: int,
    image_hw: int,
) -> np.ndarray:
    """Upsample focus map to image resolution.  Returns [image_hw, image_hw] float [0,1]."""
    if focus_map.dim() == 2:
        focus_map = focus_map[0]  # take first batch item
    fm = focus_map.float().reshape(patch_hw, patch_hw)
    fm_img = F.interpolate(
        fm.unsqueeze(0).unsqueeze(0),
        size=(image_hw, image_hw),
        mode="bilinear",
        align_corners=False,
    ).squeeze().clamp(0.0, 1.0)
    return fm_img.cpu().numpy()


@torch.no_grad()
def _focus_weighted_recon(
    pred: torch.Tensor,        # [B, 3, H, W] float [0,1]
    gt: torch.Tensor,          # [B, 3, H, W] float [0,1]
    focus_map: torch.Tensor,   # [B, N] ∈ [0,1]
    patch_hw: int,
) -> float:
    """Mean squared error weighted by the focus map (object-centric error)."""
    H, W = pred.shape[-2], pred.shape[-1]
    # Upsample focus map to image resolution: [B, 1, H, W]
    fm = focus_map.float().reshape(focus_map.shape[0], 1, patch_hw, patch_hw)
    fm_img = F.interpolate(fm, size=(H, W), mode="bilinear", align_corners=False)
    per_pixel_sq = (pred.float() - gt.float()).pow(2).mean(dim=1, keepdim=True)  # [B,1,H,W]
    weight_sum = fm_img.sum().clamp(min=1e-6)
    return (per_pixel_sq * fm_img).sum().item() / weight_sum.item()


@torch.no_grad()
def _focus_center_of_mass(
    focus_map: torch.Tensor,  # [B, N] ∈ [0,1]
    patch_hw: int,
) -> Tuple[float, float]:
    """Mean center-of-mass (y, x) of focus map, normalised to [0, 1].
    Serves as a gripper-location proxy: high focus should follow the gripper."""
    fm = focus_map.float().reshape(-1, patch_hw, patch_hw)  # [B, Ph, Pw]
    B, Ph, Pw = fm.shape
    ys = torch.arange(Ph, device=fm.device, dtype=torch.float32) / max(Ph - 1, 1)
    xs = torch.arange(Pw, device=fm.device, dtype=torch.float32) / max(Pw - 1, 1)
    w = fm.sum(dim=(1, 2)).clamp(min=1e-6)  # [B]
    com_y = (fm * ys.view(1, Ph, 1)).sum(dim=(1, 2)) / w  # [B]
    com_x = (fm * xs.view(1, 1, Pw)).sum(dim=(1, 2)) / w  # [B]
    return float(com_y.mean().item()), float(com_x.mean().item())


@torch.no_grad()
def _motion_sensitive_recon(
    pred: torch.Tensor,          # [B, 3, H, W] float [0,1]
    gt: torch.Tensor,            # [B, 3, H, W] float [0,1]
    change_target: torch.Tensor, # [B, N] ∈ [0,1]  DINO change target
    patch_hw: int,
    threshold: float = 0.3,
) -> float:
    """Reconstruction error in high-motion patches only.
    Binarises change_target at threshold and computes MSE only on those patches."""
    H, W = pred.shape[-2], pred.shape[-1]
    mask = (change_target.float() >= threshold).float()  # [B, N]
    mask_img = mask.reshape(mask.shape[0], 1, patch_hw, patch_hw)
    mask_img = F.interpolate(mask_img, size=(H, W), mode="nearest")  # [B,1,H,W]
    per_pixel_sq = (pred.float() - gt.float()).pow(2).mean(dim=1, keepdim=True)
    weight_sum = mask_img.sum().clamp(min=1e-6)
    return (per_pixel_sq * mask_img).sum().item() / weight_sum.item()


@torch.no_grad()
def _batch_com(attn: torch.Tensor, patch_hw: int) -> torch.Tensor:
    """Per-sample center-of-mass of a patch-level attention map.

    Args:
        attn: [B, N] attention / focus values ∈ [0, 1]
        patch_hw: sqrt(N), i.e. number of patches along one spatial dimension

    Returns:
        [B, 2] float tensor with (y, x) ∈ [0, 1], top-left origin.
    """
    B = attn.shape[0]
    fm = attn.float().reshape(B, patch_hw, patch_hw)       # [B, Ph, Pw]
    ys = torch.arange(patch_hw, dtype=torch.float32, device=fm.device) / max(patch_hw - 1, 1)
    xs = torch.arange(patch_hw, dtype=torch.float32, device=fm.device) / max(patch_hw - 1, 1)
    w  = fm.sum(dim=(1, 2)).clamp(min=1e-6)               # [B]
    com_y = (fm * ys.view(1, patch_hw, 1)).sum(dim=(1, 2)) / w   # [B]
    com_x = (fm * xs.view(1, 1, patch_hw)).sum(dim=(1, 2)) / w   # [B]
    return torch.stack([com_y, com_x], dim=1)              # [B, 2]


@torch.no_grad()
def _roi_l1_batch(
    pred: torch.Tensor,       # [B, 3, H, W] float [0, 1]
    gt: torch.Tensor,         # [B, 3, H, W] float [0, 1]
    centers: torch.Tensor,    # [B, 2] float ∈ [0, 1] (y, x)
    roi_half: int = 40,       # half-size of square ROI in pixels
) -> float:
    """Mean per-sample L1 in per-sample ROI crops.

    Each sample gets its own crop centred on ``centers[i]``.  Crops are
    clamped to the image boundary, so edge samples are still meaningful.
    Falls back to full-image L1 if the crop collapses to zero area.
    """
    B, _, H, W = pred.shape
    vals: List[float] = []
    for i in range(B):
        cy = int(float(centers[i, 0].item()) * H)
        cx = int(float(centers[i, 1].item()) * W)
        y0 = max(0, cy - roi_half);  y1 = min(H, cy + roi_half)
        x0 = max(0, cx - roi_half);  x1 = min(W, cx + roi_half)
        if y1 <= y0 or x1 <= x0:
            vals.append(float(F.l1_loss(pred[i], gt[i]).item()))
        else:
            vals.append(float(
                F.l1_loss(pred[i, :, y0:y1, x0:x1],
                          gt[i, :, y0:y1, x0:x1]).item()
            ))
    return float(np.mean(vals))


def _grid_to_image(
    rows: List[List[np.ndarray]],  # List[row] where row = List[[H,W,3]]
    pad: int = 4,
    bg: int = 200,
) -> Optional[np.ndarray]:
    """Assemble a grid of images. Returns [H_total, W_total, 3] uint8."""
    if not rows:
        return None
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    H, W   = rows[0][0].shape[:2]
    grid_h = n_rows * H + (n_rows + 1) * pad
    grid_w = n_cols * W + (n_cols + 1) * pad
    canvas = np.full((grid_h, grid_w, 3), bg, dtype=np.uint8)
    for ri, row in enumerate(rows):
        for ci, img in enumerate(row):
            y = ri * (H + pad) + pad
            x = ci * (W + pad) + pad
            canvas[y : y + H, x : x + W] = img[:H, :W]
    return canvas


def _save_image(arr: np.ndarray, path: Path) -> None:
    if not _HAS_PIL:
        return
    PILImage.fromarray(arr).save(str(path))


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _apply_cmap_rgb(
    data: np.ndarray,
    cmap_name: str = "hot",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """Apply a named colormap to 2D data, return [H, W, 3] uint8."""
    if not _HAS_MPL:
        return np.zeros((*data.shape, 3), dtype=np.uint8)
    cmap = cm.get_cmap(cmap_name)
    normalized = np.clip(
        (data.astype(np.float64) - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0
    )
    return (cmap(normalized)[..., :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Matplotlib-based labeled grid (replaces raw PIL grid for all viz outputs)
# ---------------------------------------------------------------------------

def _mpl_save_grid(
    panels: List[List[Optional[np.ndarray]]],
    col_titles: List[str],
    row_labels: Optional[List[str]],
    title: str,
    out_path: "Path",
    cell_annotations: Optional[List[List[str]]] = None,
    heatmap_cmap: str = "hot",
    cell_size: float = 2.5,
    dpi: int = 100,
) -> None:
    """Save a matplotlib figure with labeled rows/columns.

    panels[row][col]:
      [H, W, 3] uint8  → shown as RGB image (imshow)
      [H, W] float32   → shown as heatmap (cmap, vmin=0, vmax=1)
      None             → empty cell

    cell_annotations[row][col]: optional text drawn at the bottom of each cell.
    """
    if not _HAS_MPL or not panels:
        return

    n_rows = len(panels)
    n_cols = max(len(r) for r in panels)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_size, 0.55 + n_rows * cell_size),
        squeeze=False,
        gridspec_kw={"wspace": 0.04, "hspace": 0.12},
    )
    fig.patch.set_facecolor("#f4f4f4")

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", color="#111111",
                     y=1.0, va="bottom")

    for ri in range(n_rows):
        for ci in range(n_cols):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#bbbbbb"); sp.set_linewidth(0.5)

            if ci < len(panels[ri]) and panels[ri][ci] is not None:
                p = panels[ri][ci]
                if p.ndim == 3:
                    ax.imshow(p)
                else:
                    ax.imshow(p, cmap=heatmap_cmap, vmin=0.0, vmax=1.0, aspect="auto")
            else:
                ax.axis("off")
                continue

            # Column title on first row
            if ri == 0 and ci < len(col_titles):
                ax.set_title(col_titles[ci], fontsize=8, fontweight="bold",
                             color="#222222", pad=3)

            # Row label to the left of first column
            if ci == 0 and row_labels and ri < len(row_labels):
                ax.text(-0.04, 0.5, row_labels[ri],
                        transform=ax.transAxes,
                        fontsize=7.5, color="#333333",
                        ha="right", va="center", multialignment="center",
                        fontfamily="monospace")

            # Optional per-cell annotation at bottom
            if cell_annotations and ri < len(cell_annotations) \
                    and ci < len(cell_annotations[ri]) \
                    and cell_annotations[ri][ci]:
                ax.text(0.5, -0.03, cell_annotations[ri][ci],
                        transform=ax.transAxes,
                        fontsize=7, color="#555555",
                        ha="center", va="top")

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Step-level visualization (lightweight, called during training)
# ---------------------------------------------------------------------------

def save_step_visualizations(
    output_dir: str,
    step: int,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    cfg,  # FocusedWMConfig
    max_samples: int = 8,
) -> None:
    """Save labeled matplotlib visualizations for a single training step.

    Outputs:
      batch_examples.png  — Current | GT Future | Predicted | Focus Map | Error Map
      change_targets.png  — Current | GT Future | Change Target | Focus Map (comparison)
    """
    if not _HAS_MPL:
        return

    out_step = _ensure(
        Path(output_dir) / "visualizations" / "libero" / f"step_{step:06d}"
    )

    m = model.module if hasattr(model, "module") else model
    m.eval()

    cur  = batch["current_pixels"][:max_samples].to(device)
    fut  = batch["future_pixels"][:max_samples].to(device)
    acts = batch["actions"][:max_samples].to(device)

    with torch.no_grad():
        out = m(cur, acts, fut)

    pred_imgs  = out["pred_future_image"]    # [B, 3, H, W]
    focus_maps = out["focus_map"]            # [B, N]
    change_tgt = out.get("dino_change_target")

    B        = pred_imgs.shape[0]
    patch_hw = cfg.patch_hw
    pred_hw  = pred_imgs.shape[-2:]

    cur_f = _prep_image_tensor_batch(cur)
    if cur_f.shape[-2:] != pred_hw:
        cur_f = F.interpolate(cur_f, size=pred_hw, mode="bilinear", align_corners=False)
    fut_f = _prep_image_tensor_batch(fut)
    if fut_f.shape[-2:] != pred_hw:
        fut_f = F.interpolate(fut_f, size=pred_hw, mode="bilinear", align_corners=False)
    image_hw = pred_hw[0]

    # --- batch_examples.png ---
    col_titles_main = [
        "Current Frame",
        "GT Future",
        "Predicted Future",
        "Focus Map\n(hot overlay on current)",
        "Error Map\n|pred − GT|  (hot)",
    ]
    panels_main   = []
    row_labels    = []
    annots_main   = []

    for i in range(B):
        cur_np  = _to_numpy_rgb(cur_f[i])
        fut_np  = _to_numpy_rgb(fut_f[i])
        pred_np = _to_numpy_rgb(pred_imgs[i])

        fm = _focus_map_to_spatial(focus_maps[i], patch_hw, image_hw)
        focus_ov = _overlay_heatmap(cur_np, fm, alpha=0.55, colormap="hot")

        err = (pred_imgs[i].float() - fut_f[i].to(pred_imgs.device).float())
        err_map = err.abs().mean(0).cpu().numpy()
        err_map = (err_map / (err_map.max() + 1e-8)).astype(np.float32)

        l1 = float(F.l1_loss(
            pred_imgs[i].float(),
            fut_f[i].to(pred_imgs.device).float(),
        ).item())

        panels_main.append([cur_np, fut_np, pred_np, focus_ov, err_map])
        row_labels.append(f"S{i + 1}\nL1={l1:.4f}")
        annots_main.append([None, None, None,
                            f"mean={fm.mean():.3f}", f"max={err_map.max():.3f}"])

    _mpl_save_grid(
        panels=panels_main,
        col_titles=col_titles_main,
        row_labels=row_labels,
        title=f"Step {step:,}  —  Training Batch Examples  (B={B})",
        out_path=out_step / "batch_examples.png",
        cell_annotations=annots_main,
        heatmap_cmap="hot",
    )

    # --- change_targets.png (only when available) ---
    if change_tgt is not None:
        col_titles_ct = [
            "Current Frame",
            "GT Future",
            "Change Target\n(DINO feature diff, viridis)",
            "Focus Map\n(hot overlay — should match change)",
        ]
        panels_ct  = []
        annots_ct  = []
        for i in range(B):
            cur_np = _to_numpy_rgb(cur_f[i])
            fut_np = _to_numpy_rgb(fut_f[i])
            ct  = _focus_map_to_spatial(change_tgt[i], patch_hw, image_hw).astype(np.float32)
            fm  = _focus_map_to_spatial(focus_maps[i], patch_hw, image_hw).astype(np.float32)
            panels_ct.append([cur_np, fut_np, ct, fm])
            annots_ct.append([None, None,
                              f"mean={ct.mean():.3f}", f"mean={fm.mean():.3f}"])

        _mpl_save_grid(
            panels=panels_ct,
            col_titles=col_titles_ct,
            row_labels=[f"S{i + 1}" for i in range(B)],
            title=f"Step {step:,}  —  Change Target vs Focus Map",
            out_path=out_step / "change_targets.png",
            cell_annotations=annots_ct,
            heatmap_cmap="viridis",
        )

    m.train()

    # Phase 4: dual-mask additional visualizations (degrade gracefully if no dual mask)
    try:
        save_dual_mask_visualizations(
            output_dir=output_dir,
            step=step,
            model=model,
            batch=batch,
            device=device,
            cfg=cfg,
            max_samples=max_samples,
        )
    except Exception as _dmv_err:
        logger.debug("dual_mask_visualizations skipped: %s", _dmv_err)


def _prep_image_tensor(pixels: torch.Tensor) -> torch.Tensor:
    """[H, W, C] uint8 or [3, H, W] float → [3, H, W] float [0,1]."""
    if pixels.dtype == torch.uint8:
        return pixels.float().permute(2, 0, 1) / 255.0
    return pixels


# ---------------------------------------------------------------------------
# Full evaluation (called offline or at training end)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_full_evaluation(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    cfg,                    # FocusedWMConfig
    output_dir: str,
    task_suite: str,
    max_batches: int = 200,
) -> Dict[str, float]:
    """Run full evaluation over a dataset and save metrics + visualizations.

    Returns:
        dict of aggregated metrics (mean over all batches).
    """
    from .losses import (
        image_reconstruction_metrics,
        focus_metrics,
        dino_feature_metrics,
    )

    out_eval = _ensure(Path(output_dir) / "eval_reports" / "libero" / task_suite)
    out_viz  = _ensure(Path(output_dir) / "visualizations" / "libero" / task_suite / "full_eval")

    # Task-aware goal ROI (Task B) — load once, apply per batch using suite default
    _roi_config = load_roi_config()
    _goal_y, _goal_x = get_goal_roi_center(task_suite, -1, _roi_config)   # suite default

    m = model.module if hasattr(model, "module") else model
    m.eval()

    agg: Dict[str, List[float]] = {}
    casebook: Dict[str, dict]   = {}   # step → {recon_loss, images}
    n_batches = 0

    for batch in data_loader:
        if n_batches >= max_batches:
            break

        cur  = batch["current_pixels"].to(device)
        fut  = batch["future_pixels"].to(device)
        acts = batch["actions"].to(device)

        out = m(cur, acts, fut)

        pred   = out["pred_future_image"]
        focus  = out["focus_map"]
        tokens = out["predicted_future_tokens"]
        change = out.get("dino_change_target")

        # GT future in [0,1] float for metrics; resize to match pred if needed
        fut_f = _prep_image_tensor_batch(fut)
        if fut_f.shape[-2:] != pred.shape[-2:]:
            fut_f = F.interpolate(
                fut_f, size=pred.shape[-2:], mode="bilinear", align_corners=False
            )

        # --- Reconstruction metrics ---
        im_met = image_reconstruction_metrics(pred, fut_f.to(pred.device))
        for k, v in im_met.items():
            agg.setdefault(k, []).append(v)

        # --- Focus metrics ---
        fm_met = focus_metrics(focus, change)
        for k, v in fm_met.items():
            agg.setdefault("focus_" + k if not k.startswith("focus") else k, []).append(v)

        # --- DINO feature metrics ---
        gt_raw  = m.dino.extract_raw(fut_f.to(pred.device))
        gt_proj = m.dino.proj(gt_raw.to(tokens.dtype))
        dn_met  = dino_feature_metrics(tokens, gt_proj)
        for k, v in dn_met.items():
            agg.setdefault(k, []).append(v)

        # --- Object-centric error (focus-weighted recon) ---
        fw_recon = _focus_weighted_recon(pred, fut_f.to(pred.device), focus, cfg.patch_hw)
        agg.setdefault("focus_weighted_recon", []).append(fw_recon)

        # --- Gripper-proxy: focus center of mass ---
        com_y, com_x = _focus_center_of_mass(focus, cfg.patch_hw)
        agg.setdefault("focus_com_y", []).append(com_y)
        agg.setdefault("focus_com_x", []).append(com_x)

        # --- ROI metrics (gripper / object / goal) ---
        roi_half = getattr(cfg, "roi_half_pixels", 40)
        wrt_t    = out.get("write_mask", focus)

        # gripper ROI: per-sample crop around write-mask center-of-mass
        gripper_centers = _batch_com(wrt_t.to(pred.device), cfg.patch_hw)
        agg.setdefault("roi/gripper_l1", []).append(
            _roi_l1_batch(pred, fut_f.to(pred.device), gripper_centers, roi_half)
        )

        # object ROI: per-sample crop around DINO change-target COM
        if change is not None:
            obj_centers = _batch_com(change.to(pred.device), cfg.patch_hw)
            agg.setdefault("roi/object_l1", []).append(
                _roi_l1_batch(pred, fut_f.to(pred.device), obj_centers, roi_half)
            )

        # goal ROI: task-aware position from roi_coords_v1.json (Task B)
        # Falls back to cfg.goal_roi_y/x if set, then to suite default from config.
        goal_y = getattr(cfg, "goal_roi_y", _goal_y)
        goal_x = getattr(cfg, "goal_roi_x", _goal_x)
        B_cur  = pred.shape[0]
        goal_centers = torch.tensor(
            [[goal_y, goal_x]] * B_cur, dtype=torch.float32, device=pred.device
        )
        agg.setdefault("roi/goal_l1", []).append(
            _roi_l1_batch(pred, fut_f.to(pred.device), goal_centers, roi_half)
        )

        # --- Motion-sensitive error ---
        if change is not None:
            ms_recon = _motion_sensitive_recon(pred, fut_f.to(pred.device), change, cfg.patch_hw)
            agg.setdefault("motion_sensitive_recon", []).append(ms_recon)

        # --- Dual-mask per-sample stats for analysis plots ---------------
        ctx_t = out.get("context_mask", focus)
        wrt_t = out.get("write_mask",   focus)
        ctx_mean = float(ctx_t.mean().item())
        wrt_mean = float(wrt_t.mean().item())
        agg.setdefault("context_mask_mean", []).append(ctx_mean)
        agg.setdefault("write_mask_mean",   []).append(wrt_mean)
        agg.setdefault("write_context_gap", []).append(
            float((wrt_t - ctx_t).abs().mean().item())
        )
        if change is not None:
            from .train_focused_libero import _batch_iou_soft, _batch_dice_soft
            agg.setdefault("write_iou",  []).append(_batch_iou_soft(wrt_t.float(), change.float()))
            agg.setdefault("context_iou",[]).append(_batch_iou_soft(ctx_t.float(), change.float()))

        # --- Collect casebook sample (first item in batch) ---------------
        recon_loss = im_met["future_image_smooth_l1"]
        casebook[f"batch_{n_batches:04d}"] = {
            "recon_loss":   recon_loss,
            "focus_mean":   fm_met.get("focus_mean", 0),
            "context_mean": ctx_mean,
            "write_mean":   wrt_mean,
            "write_context_gap": float((wrt_t[0] - ctx_t[0]).abs().mean().item()),
            "batch_idx": n_batches,
            # Store tensors for later rendering
            "cur_t":       cur[0].cpu(),
            "fut_t":       fut[0].cpu(),
            "pred_t":      pred[0].cpu(),
            "focus_t":     focus[0].cpu(),
            "context_t":   ctx_t[0].cpu(),
            "write_t":     wrt_t[0].cpu(),
            "change_t":    change[0].cpu() if change is not None else None,
            "residual_t":  out.get("predicted_residual_image", pred[:1])[0].cpu(),
        }

        n_batches += 1

    # --- Aggregate ---
    summary = {k: float(np.mean(v)) for k, v in agg.items()}
    summary["n_batches"] = n_batches

    # --- Save metrics.json ---
    (out_eval / "metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info("Full eval metrics saved: %s", out_eval / "metrics.json")

    # --- Save casebook visualizations ---
    if _HAS_PIL and casebook:
        _save_casebook(casebook, out_viz, cfg)
        _save_dual_mask_casebook(casebook, out_viz, cfg)
        _save_focus_analysis(casebook, out_viz)
        _save_dual_mask_analysis(agg, out_viz)
        if cfg.num_action_candidates > 1:
            _save_candidate_ranking_note(out_viz)

    return summary


def _prep_image_tensor_batch(pixels: torch.Tensor) -> torch.Tensor:
    """[B, H, W, C] uint8 or [B, 3, H, W] float → [B, 3, H, W] float."""
    if pixels.dtype == torch.uint8:
        return pixels.float().permute(0, 3, 1, 2) / 255.0
    return pixels


def _save_casebook(
    casebook: dict,
    out_dir: Path,
    cfg,
) -> None:
    """Save best / median / worst casebook as a labeled matplotlib figure."""
    if not _HAS_MPL:
        return

    entries = sorted(casebook.values(), key=lambda x: x["recon_loss"])
    n = len(entries)
    picks = [
        ("Best",   entries[0]),
        ("Median", entries[n // 2]),
        ("Worst",  entries[-1]),
    ]

    patch_hw = cfg.patch_hw
    has_change = any(e["change_t"] is not None for _, e in picks)

    col_titles = [
        "Current Frame",
        "GT Future",
        "Predicted Future",
        "Focus Map\n(hot overlay on current)",
        "Error Map\n|pred − GT|  (hot)",
    ]
    if has_change:
        col_titles.append("Change Target\n(DINO diff, viridis)")

    panels     = []
    row_labels = []
    annots     = []

    for label, entry in picks:
        pred_t   = entry["pred_t"]          # [3, H, W] float
        pred_hw  = pred_t.shape[-2:]
        image_hw = pred_hw[0]

        # Resize inputs to pred resolution for consistency
        cur_f = _prep_image_tensor(entry["cur_t"]).unsqueeze(0)
        fut_f = _prep_image_tensor(entry["fut_t"]).unsqueeze(0)
        if cur_f.shape[-2:] != pred_hw:
            cur_f = F.interpolate(cur_f, size=pred_hw, mode="bilinear", align_corners=False)
        if fut_f.shape[-2:] != pred_hw:
            fut_f = F.interpolate(fut_f, size=pred_hw, mode="bilinear", align_corners=False)

        cur_np  = _to_numpy_rgb(cur_f[0])
        fut_np  = _to_numpy_rgb(fut_f[0])
        pred_np = _to_numpy_rgb(pred_t)

        fm = _focus_map_to_spatial(entry["focus_t"], patch_hw, image_hw)
        focus_ov = _overlay_heatmap(cur_np, fm, alpha=0.55)

        err = (pred_t.float() - fut_f[0].float()).abs().mean(0).numpy()
        err_map = (err / (err.max() + 1e-8)).astype(np.float32)

        row   = [cur_np, fut_np, pred_np, focus_ov, err_map]
        annot = [
            None, None, None,
            f"focus_mean={fm.mean():.3f}",
            f"max_err={err_map.max():.3f}",
        ]

        if has_change:
            if entry["change_t"] is not None:
                ct = _focus_map_to_spatial(entry["change_t"], patch_hw, image_hw)
                row.append(ct.astype(np.float32))
                annot.append(f"ct_mean={ct.mean():.3f}")
            else:
                row.append(None); annot.append(None)

        panels.append(row)
        annots.append(annot)
        l1 = entry["recon_loss"]
        row_labels.append(f"{label}\nL1={l1:.4f}")

    _mpl_save_grid(
        panels=panels,
        col_titles=col_titles,
        row_labels=row_labels,
        title="Casebook  —  Best / Median / Worst Reconstruction",
        out_path=out_dir / "casebook.png",
        cell_annotations=annots,
        heatmap_cmap="hot",
    )


def _save_focus_analysis(casebook: dict, out_dir: Path) -> None:
    """Save a matplotlib figure with focus statistics."""
    if not _HAS_MPL:
        return

    recons = [e["recon_loss"] for e in casebook.values()]
    fmeans = [e["focus_mean"] for e in casebook.values()]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(fmeans, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("Focus map mean distribution")
    axes[0].set_xlabel("Mean focus activation")
    axes[0].set_ylabel("Count")

    axes[1].scatter(fmeans, recons, alpha=0.5, s=12, color="darkorange")
    axes[1].set_title("Focus mean vs. Reconstruction loss")
    axes[1].set_xlabel("Focus mean")
    axes[1].set_ylabel("Recon loss (SmoothL1)")

    fig.tight_layout()
    fig.savefig(str(out_dir / "focus_analysis.png"), dpi=120)
    plt.close(fig)
    logger.info("Focus analysis saved.")


# ---------------------------------------------------------------------------
# Dual-mask step-level visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_dual_mask_visualizations(
    output_dir: str,
    step: int,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    cfg,
    max_samples: int = 6,
) -> None:
    """Save dual-mask step visualizations.

    Outputs under step_{N:06d}/:
      dual_mask_examples.png        — context + write mask vs error/residual
      change_target_vs_dual_mask.png — change target vs context/write + difference
      dual_mask_casebook.png        — best/median/worst within this batch
      dual_mask_analysis.png        — per-patch histogram + scatter (single batch)
    """
    if not _HAS_MPL:
        return

    out_step = _ensure(
        Path(output_dir) / "visualizations" / "libero" / f"step_{step:06d}"
    )

    m = model.module if hasattr(model, "module") else model
    m.eval()

    B_in = min(max_samples, batch["current_pixels"].shape[0])
    cur  = batch["current_pixels"][:B_in].to(device)
    fut  = batch["future_pixels"][:B_in].to(device)
    acts = batch["actions"][:B_in].to(device)

    with torch.no_grad():
        out = m(cur, acts, fut)

    pred_imgs = out["pred_future_image"]                          # [B, 3, H, W]
    ctx_mask  = out.get("context_mask", out.get("focus_map"))     # [B, N]
    wrt_mask  = out.get("write_mask",   out.get("focus_map"))     # [B, N]
    residual  = out.get("predicted_residual_image")               # [B, 3, H, W]
    change_tgt = out.get("dino_change_target")                    # [B, N] or None
    wrt_img_t  = out.get("write_mask_img")                        # [B, 1, H, W] or None

    if ctx_mask is None:
        m.train()
        return

    B        = pred_imgs.shape[0]
    patch_hw = cfg.patch_hw
    pred_hw  = pred_imgs.shape[-2:]
    image_hw = pred_hw[0]

    cur_f = _prep_image_tensor_batch(cur)
    if cur_f.shape[-2:] != pred_hw:
        cur_f = F.interpolate(cur_f, size=pred_hw, mode="bilinear", align_corners=False)
    fut_f = _prep_image_tensor_batch(fut)
    if fut_f.shape[-2:] != pred_hw:
        fut_f = F.interpolate(fut_f, size=pred_hw, mode="bilinear", align_corners=False)

    # -----------------------------------------------------------------------
    # 1. dual_mask_examples.png
    # -----------------------------------------------------------------------
    has_change   = change_tgt is not None
    has_residual = residual is not None

    col_titles = [
        "Current Frame", "GT Future", "Predicted Future",
        "Context Mask\n(hot)", "Write Mask\n(Oranges)",
    ]
    if has_change:
        col_titles.append("Change Target\n(viridis)")
    col_titles.append("Error |pred−GT|\n(hot)")
    if has_residual:
        col_titles.append("|Pred Residual|\n(hot)")
        col_titles.append("Write×|Residual|\n(hot)")

    panels_ex  = []
    row_labels = []
    annots_ex  = []
    recon_l1s  = []

    for i in range(B):
        cur_np  = _to_numpy_rgb(cur_f[i])
        fut_np  = _to_numpy_rgb(fut_f[i])
        pred_np = _to_numpy_rgb(pred_imgs[i])

        ctx_sp  = _focus_map_to_spatial(ctx_mask[i], patch_hw, image_hw)
        wrt_sp  = _focus_map_to_spatial(wrt_mask[i], patch_hw, image_hw)
        ctx_ov  = _overlay_heatmap(cur_np, ctx_sp, alpha=0.55, colormap="hot")
        wrt_ov  = _overlay_heatmap(cur_np, wrt_sp, alpha=0.55, colormap="Oranges")

        err_t   = (pred_imgs[i].float() - fut_f[i].to(pred_imgs.device).float()).abs()
        err_map = err_t.mean(0).cpu().numpy()
        err_map = _apply_cmap_rgb(err_map / (err_map.max() + 1e-8), "hot")

        l1 = float(F.l1_loss(pred_imgs[i].float(), fut_f[i].to(pred_imgs.device).float()).item())
        recon_l1s.append(l1)
        ctx_m = float(ctx_mask[i].mean().item())
        wrt_m = float(wrt_mask[i].mean().item())

        row   = [cur_np, fut_np, pred_np, ctx_ov, wrt_ov]
        annot = [None, None, None, f"mean={ctx_m:.3f}", f"mean={wrt_m:.3f}"]

        if has_change:
            ct_sp = _focus_map_to_spatial(change_tgt[i], patch_hw, image_hw)
            row.append(_apply_cmap_rgb(ct_sp, "viridis"))
            annot.append(f"mean={ct_sp.mean():.3f}")

        row.append(err_map)
        annot.append(f"l1={l1:.4f}")

        if has_residual:
            res_sp = residual[i].float().abs().mean(0).cpu().numpy()  # [H, W]
            res_max = res_sp.max()
            row.append(_apply_cmap_rgb(res_sp / (res_max + 1e-8), "hot"))
            annot.append(f"max={res_max:.4f}")

            # write × |residual|
            if wrt_img_t is not None:
                wm_2d = wrt_img_t[i, 0].cpu().numpy()
            else:
                wm_2d = _focus_map_to_spatial(wrt_mask[i], patch_hw, image_hw)
            wm_res = wm_2d * (res_sp / (res_max + 1e-8))
            row.append(_apply_cmap_rgb(wm_res / (wm_res.max() + 1e-8), "hot"))
            annot.append(f"gap={abs(wrt_m - ctx_m):.3f}")

        panels_ex.append(row)
        annots_ex.append(annot)
        row_labels.append(f"S{i+1}\nL1={l1:.4f}\nctx={ctx_m:.3f}\nwrt={wrt_m:.3f}")

    _mpl_save_grid(
        panels=panels_ex,
        col_titles=col_titles,
        row_labels=row_labels,
        title=f"Step {step:,}  —  Dual Mask Examples  (B={B})",
        out_path=out_step / "dual_mask_examples.png",
        cell_annotations=annots_ex,
        heatmap_cmap="hot",
        cell_size=2.3,
    )

    # -----------------------------------------------------------------------
    # 2. change_target_vs_dual_mask.png
    # -----------------------------------------------------------------------
    if has_change:
        col_titles_ct = [
            "Current Frame", "GT Future", "Change Target\n(viridis)",
            "Context Mask\n(hot)", "Write Mask\n(Oranges)",
            "Write − Context\n(RdBu: blue<0<red)",
        ]
        panels_ct = []
        annots_ct = []
        for i in range(B):
            cur_np = _to_numpy_rgb(cur_f[i])
            fut_np = _to_numpy_rgb(fut_f[i])
            ct_sp  = _focus_map_to_spatial(change_tgt[i], patch_hw, image_hw).astype(np.float32)
            ctx_sp = _focus_map_to_spatial(ctx_mask[i],   patch_hw, image_hw).astype(np.float32)
            wrt_sp = _focus_map_to_spatial(wrt_mask[i],   patch_hw, image_hw).astype(np.float32)
            diff   = wrt_sp - ctx_sp   # ∈ [-1, 1]; render with RdBu centred at 0
            diff_rgb = _apply_cmap_rgb(diff, "RdBu_r", vmin=-1.0, vmax=1.0)

            panels_ct.append([
                cur_np, fut_np,
                _apply_cmap_rgb(ct_sp,  "viridis"),
                _apply_cmap_rgb(ctx_sp, "hot"),
                _apply_cmap_rgb(wrt_sp, "Oranges"),
                diff_rgb,
            ])
            annots_ct.append([
                None, None,
                f"mean={ct_sp.mean():.3f}",
                f"mean={ctx_sp.mean():.3f}",
                f"mean={wrt_sp.mean():.3f}",
                f"mean_gap={np.abs(diff).mean():.3f}",
            ])

        _mpl_save_grid(
            panels=panels_ct,
            col_titles=col_titles_ct,
            row_labels=[f"S{i+1}" for i in range(B)],
            title=f"Step {step:,}  —  Change Target vs Dual Mask",
            out_path=out_step / "change_target_vs_dual_mask.png",
            cell_annotations=annots_ct,
            heatmap_cmap="viridis",
            cell_size=2.3,
        )

    # -----------------------------------------------------------------------
    # 3. dual_mask_casebook.png  (best / median / worst within this batch)
    # -----------------------------------------------------------------------
    if B >= 3 and has_residual:
        order = sorted(range(B), key=lambda i: recon_l1s[i])
        picks = [("Best", order[0]), ("Median", order[B // 2]), ("Worst", order[-1])]

        cb_cols = [
            "Current", "GT Future", "Predicted",
            "Context Mask\n(hot)", "Write Mask\n(Oranges)",
            "Error |pred−GT|\n(hot)", "Change Target\n(viridis)",
            "|Pred Residual|\n(hot)",
        ]
        cb_panels = []; cb_annots = []; cb_labels = []

        for label, i in picks:
            cur_np  = _to_numpy_rgb(cur_f[i])
            fut_np  = _to_numpy_rgb(fut_f[i])
            pred_np = _to_numpy_rgb(pred_imgs[i])
            ctx_sp  = _focus_map_to_spatial(ctx_mask[i], patch_hw, image_hw)
            wrt_sp  = _focus_map_to_spatial(wrt_mask[i], patch_hw, image_hw)
            err_sp  = (pred_imgs[i].float() - fut_f[i].to(pred_imgs.device).float()).abs().mean(0).cpu().numpy()
            res_sp  = residual[i].float().abs().mean(0).cpu().numpy()

            row   = [cur_np, fut_np, pred_np,
                     _apply_cmap_rgb(ctx_sp, "hot"),
                     _apply_cmap_rgb(wrt_sp, "Oranges"),
                     _apply_cmap_rgb(err_sp / (err_sp.max() + 1e-8), "hot")]
            annot = [None, None, None,
                     f"mean={ctx_sp.mean():.3f}", f"mean={wrt_sp.mean():.3f}",
                     f"l1={recon_l1s[i]:.4f}"]

            if has_change:
                ct_sp = _focus_map_to_spatial(change_tgt[i], patch_hw, image_hw)
                row.append(_apply_cmap_rgb(ct_sp, "viridis"))
                annot.append(f"ct={ct_sp.mean():.3f}")
            else:
                row.append(None); annot.append(None)

            row.append(_apply_cmap_rgb(res_sp / (res_sp.max() + 1e-8), "hot"))
            annot.append(f"max={res_sp.max():.4f}")

            ctx_m = float(ctx_mask[i].mean().item())
            wrt_m = float(wrt_mask[i].mean().item())
            cb_panels.append(row)
            cb_annots.append(annot)
            cb_labels.append(
                f"{label}\nL1={recon_l1s[i]:.4f}\n"
                f"ctx={ctx_m:.3f}\nwrt={wrt_m:.3f}\n"
                f"gap={abs(wrt_m - ctx_m):.3f}"
            )

        _mpl_save_grid(
            panels=cb_panels,
            col_titles=cb_cols,
            row_labels=cb_labels,
            title=f"Step {step:,}  —  Dual Mask Casebook (batch best/median/worst)",
            out_path=out_step / "dual_mask_casebook.png",
            cell_annotations=cb_annots,
            heatmap_cmap="hot",
            cell_size=2.3,
        )

    # -----------------------------------------------------------------------
    # 4. dual_mask_analysis.png  (patch-level distributions within batch)
    # -----------------------------------------------------------------------
    if _HAS_MPL:
        try:
            _save_batch_dual_mask_analysis(
                ctx_mask=ctx_mask, wrt_mask=wrt_mask,
                change_tgt=change_tgt,
                pred_imgs=pred_imgs, fut_f=fut_f.to(pred_imgs.device),
                residual=residual,
                step=step,
                out_path=out_step / "dual_mask_analysis.png",
            )
        except Exception as _ana_err:
            logger.debug("dual_mask_analysis skipped: %s", _ana_err)

    m.train()


@torch.no_grad()
def _save_batch_dual_mask_analysis(
    ctx_mask: torch.Tensor,       # [B, N]
    wrt_mask: torch.Tensor,       # [B, N]
    change_tgt: Optional[torch.Tensor],  # [B, N] or None
    pred_imgs: torch.Tensor,      # [B, 3, H, W]
    fut_f: torch.Tensor,          # [B, 3, H, W]
    residual: Optional[torch.Tensor],    # [B, 3, H, W] or None
    step: int,
    out_path: "Path",
) -> None:
    """Patch-level distribution plots for a single batch (step-level)."""
    if not _HAS_MPL:
        return

    ctx_np = ctx_mask.float().cpu().numpy().ravel()   # [B*N]
    wrt_np = wrt_mask.float().cpu().numpy().ravel()
    gap_np = np.abs(wrt_np - ctx_np)

    n_cols = 4
    n_rows = 2 if change_tgt is not None else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.0))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Step {step:,} — Dual Mask Analysis (patch-level, B={ctx_mask.shape[0]})",
                 fontsize=10, fontweight="bold")

    def _hist(ax, data, title, color, xlabel="value"):
        ax.hist(data, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.axvline(float(np.mean(data)), color="black", linewidth=1.2, linestyle="--",
                   label=f"mean={np.mean(data):.3f}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    _hist(axes[0, 0], ctx_np, "Context Mask Distribution", "#4477aa")
    _hist(axes[0, 1], wrt_np, "Write Mask Distribution",   "#ee6677")
    _hist(axes[0, 2], gap_np, "|Write − Context| Gap",     "#228833")

    # Scatter: recon per sample vs mask means
    recon_per_sample = F.l1_loss(
        pred_imgs.float(), fut_f.float(), reduction="none"
    ).mean(dim=[1, 2, 3]).cpu().numpy()
    ctx_per_sample = ctx_mask.float().mean(dim=1).cpu().numpy()
    wrt_per_sample = wrt_mask.float().mean(dim=1).cpu().numpy()

    ax = axes[0, 3]
    ax.scatter(ctx_per_sample, recon_per_sample, c="#4477aa", alpha=0.8, s=25, label="context")
    ax.scatter(wrt_per_sample, recon_per_sample, c="#ee6677", alpha=0.8, s=25, marker="^", label="write")
    ax.set_title("Mask Mean vs Recon L1", fontsize=9, fontweight="bold")
    ax.set_xlabel("mask mean", fontsize=8); ax.set_ylabel("recon l1", fontsize=8)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    if change_tgt is not None and n_rows > 1:
        ct_np = change_tgt.float().cpu().numpy().ravel()

        _hist(axes[1, 0], ct_np, "Change Target Distribution", "#ccbb44")

        # write_iou per sample
        from .train_focused_libero import _batch_iou_soft, _batch_dice_soft
        write_iou_val  = _batch_iou_soft(wrt_mask.float(), change_tgt.float())
        ctx_iou_val    = _batch_iou_soft(ctx_mask.float(), change_tgt.float())
        ax = axes[1, 1]
        ax.bar(["context", "write"], [ctx_iou_val, write_iou_val],
               color=["#4477aa", "#ee6677"], alpha=0.85)
        ax.set_title("IoU vs Change Target", fontsize=9, fontweight="bold")
        ax.set_ylabel("IoU@0.5", fontsize=8); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        for xi, vi in enumerate([ctx_iou_val, write_iou_val]):
            ax.text(xi, vi + 0.02, f"{vi:.3f}", ha="center", fontsize=8)

        # Scatter: change_target vs write_mask per patch
        ax = axes[1, 2]
        subsample = min(len(wrt_np), 2000)
        idx = np.random.choice(len(wrt_np), subsample, replace=False)
        ax.scatter(ct_np[idx], wrt_np[idx], alpha=0.3, s=4, color="#ee6677", label="write")
        ax.scatter(ct_np[idx], ctx_np[idx], alpha=0.3, s=4, color="#4477aa", label="context")
        ax.set_title("Change Target vs Mask\n(patch-level scatter)", fontsize=9, fontweight="bold")
        ax.set_xlabel("change target", fontsize=8); ax.set_ylabel("mask value", fontsize=8)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[1, 3]
        ax.bar(["ctx_iou", "wrt_iou", "Δiou"],
               [ctx_iou_val, write_iou_val, write_iou_val - ctx_iou_val],
               color=["#4477aa", "#ee6677", "#228833"], alpha=0.85)
        ax.set_title("Write vs Context IoU Delta", fontsize=9, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Dual-mask full-eval casebook & analysis
# ---------------------------------------------------------------------------

def _save_dual_mask_casebook(casebook: dict, out_dir: Path, cfg) -> None:
    """Best / Median / Worst casebook with dual-mask columns."""
    if not _HAS_MPL:
        return

    entries = sorted(casebook.values(), key=lambda x: x["recon_loss"])
    n = len(entries)
    picks = [("Best", entries[0]), ("Median", entries[n // 2]), ("Worst", entries[-1])]

    patch_hw = cfg.patch_hw
    has_change   = any(e.get("change_t") is not None for _, e in picks)
    has_residual = any(e.get("residual_t") is not None for _, e in picks)

    col_titles = [
        "Current", "GT Future", "Predicted",
        "Context Mask\n(hot)", "Write Mask\n(Oranges)",
        "Error |pred−GT|\n(hot)",
    ]
    if has_change:
        col_titles.append("Change Target\n(viridis)")
    if has_residual:
        col_titles.append("|Pred Residual|\n(hot)")

    panels = []; row_labels = []; annots = []

    for label, entry in picks:
        pred_t   = entry["pred_t"]
        pred_hw  = pred_t.shape[-2:]
        image_hw = pred_hw[0]

        cur_f  = _prep_image_tensor(entry["cur_t"]).unsqueeze(0)
        fut_f  = _prep_image_tensor(entry["fut_t"]).unsqueeze(0)
        for t, hw in [(cur_f, pred_hw), (fut_f, pred_hw)]:
            if t.shape[-2:] != hw:
                t = F.interpolate(t, size=hw, mode="bilinear", align_corners=False)

        cur_np  = _to_numpy_rgb(F.interpolate(cur_f, size=pred_hw, mode="bilinear", align_corners=False)[0])
        fut_np  = _to_numpy_rgb(F.interpolate(fut_f, size=pred_hw, mode="bilinear", align_corners=False)[0])
        pred_np = _to_numpy_rgb(pred_t)

        ctx_sp  = _focus_map_to_spatial(entry.get("context_t", entry["focus_t"]), patch_hw, image_hw)
        wrt_sp  = _focus_map_to_spatial(entry.get("write_t",   entry["focus_t"]), patch_hw, image_hw)

        fut_np_f = fut_f[0].numpy() if fut_f.shape[-2:] == pred_hw else \
            F.interpolate(fut_f, size=pred_hw, mode="bilinear", align_corners=False)[0].numpy()
        err_sp  = np.abs(pred_t.numpy() - fut_np_f).mean(0)  # rough error

        row   = [cur_np, fut_np, pred_np,
                 _apply_cmap_rgb(ctx_sp, "hot"),
                 _apply_cmap_rgb(wrt_sp, "Oranges"),
                 _apply_cmap_rgb(err_sp / (err_sp.max() + 1e-8), "hot")]
        annot = [None, None, None,
                 f"ctx={ctx_sp.mean():.3f}", f"wrt={wrt_sp.mean():.3f}",
                 f"l1={entry['recon_loss']:.4f}"]

        if has_change:
            if entry.get("change_t") is not None:
                ct_sp = _focus_map_to_spatial(entry["change_t"], patch_hw, image_hw)
                row.append(_apply_cmap_rgb(ct_sp, "viridis"))
                annot.append(f"ct={ct_sp.mean():.3f}")
            else:
                row.append(None); annot.append(None)

        if has_residual and entry.get("residual_t") is not None:
            res_sp = entry["residual_t"].float().abs().mean(0).numpy()
            row.append(_apply_cmap_rgb(res_sp / (res_sp.max() + 1e-8), "hot"))
            gap = entry.get("write_context_gap", 0.0)
            annot.append(f"gap={gap:.3f}")

        panels.append(row)
        annots.append(annot)
        ctx_m = entry.get("context_mean", entry.get("focus_mean", 0))
        wrt_m = entry.get("write_mean",   entry.get("focus_mean", 0))
        row_labels.append(
            f"{label}\nL1={entry['recon_loss']:.4f}\n"
            f"ctx={ctx_m:.3f}\nwrt={wrt_m:.3f}\n"
            f"gap={entry.get('write_context_gap', 0):.3f}"
        )

    _mpl_save_grid(
        panels=panels,
        col_titles=col_titles,
        row_labels=row_labels,
        title="Dual Mask Casebook — Best / Median / Worst Reconstruction",
        out_path=out_dir / "dual_mask_casebook.png",
        cell_annotations=annots,
        heatmap_cmap="hot",
        cell_size=2.5,
    )


def _save_dual_mask_analysis(agg: Dict, out_dir: Path) -> None:
    """Multi-batch analysis scatter and histograms for dual-mask metrics."""
    if not _HAS_MPL:
        return

    recons     = agg.get("future_image_smooth_l1", [])
    ctx_means  = agg.get("context_mask_mean", [])
    wrt_means  = agg.get("write_mask_mean",   [])
    gaps       = agg.get("write_context_gap", [])
    write_ious = agg.get("write_iou",  [])
    ctx_ious   = agg.get("context_iou",[])

    if not recons:
        return

    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.8, n_rows * 3.2))
    fig.suptitle("Dual Mask Analysis — Full Evaluation", fontsize=11, fontweight="bold")

    def _hist(ax, data, title, color):
        if not data:
            ax.axis("off"); return
        ax.hist(data, bins=20, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(float(np.mean(data)), color="black", linewidth=1.5, linestyle="--",
                   label=f"μ={np.mean(data):.3f}")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    def _scatter(ax, xs, ys, xlabel, ylabel, title, color):
        if not xs or not ys:
            ax.axis("off"); return
        ax.scatter(xs, ys, alpha=0.5, s=15, color=color)
        ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(alpha=0.3)

    _hist(axes[0, 0], ctx_means, "Context Mask Mean Dist.", "#4477aa")
    _hist(axes[0, 1], wrt_means, "Write Mask Mean Dist.",   "#ee6677")
    _hist(axes[0, 2], gaps,      "|Write − Context| Gap",   "#228833")
    _scatter(axes[0, 3], wrt_means, recons, "write_mask_mean", "recon_l1",
             "Write Mean vs Recon", "#ee6677")

    _scatter(axes[1, 0], ctx_means, recons, "context_mask_mean", "recon_l1",
             "Context Mean vs Recon", "#4477aa")

    if write_ious:
        _scatter(axes[1, 1], write_ious, recons, "write_iou", "recon_l1",
                 "Write IoU vs Recon", "#aa3377")
        _hist(axes[1, 2], write_ious, "Write IoU vs Change Target", "#aa3377")
        if ctx_ious:
            delta_ious = [w - c for w, c in zip(write_ious, ctx_ious)]
            _scatter(axes[1, 3], delta_ious, recons, "Δiou (write−ctx)", "recon_l1",
                     "ΔIoU vs Recon", "#228833")
        else:
            axes[1, 3].axis("off")
    else:
        for ax in axes[1, 1:]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(out_dir / "dual_mask_analysis.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_dir / "dual_mask_analysis.png")


def _save_candidate_ranking_note(out_dir: Path) -> None:
    """Write a note file explaining how to run candidate ranking visualization."""
    note = (
        "Candidate ranking visualization requires running the model with K candidates.\n"
        "Use rank_action_candidates() from focused_model.py and pass the scored\n"
        "candidates to save_candidate_ranking() in this module.\n"
    )
    (out_dir / "candidate_ranking_README.txt").write_text(note)


def save_candidate_ranking(
    output_dir: str,
    step: int,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    cfg,
    n_viz: int = 4,
) -> None:
    """Visualize candidate action predictions with ranking scores.

    Columns: Current | GT Future | GT-action pred | Neg-1 pred | Neg-2 pred | …
    Row labels show per-sample scores for each candidate.
    """
    if not _HAS_MPL:
        return

    out = _ensure(
        Path(output_dir) / "visualizations" / "libero" / f"step_{step:06d}"
    )

    from .train_focused_libero import _build_negatives
    m = model.module if hasattr(model, "module") else model
    m.eval()

    cur  = batch["current_pixels"][:n_viz].to(device)
    fut  = batch["future_pixels"][:n_viz].to(device)
    acts = batch["actions"][:n_viz].to(device)

    n_neg     = cfg.num_action_candidates - 1
    negatives = _build_negatives(acts, n_neg, cfg.negative_mode, cfg.noise_std)
    all_acts  = [acts] + negatives   # K candidates (index 0 = GT)
    K         = len(all_acts)
    B         = cur.shape[0]

    act_stacked = torch.stack(all_acts, dim=1)   # [B, K, H, 7]
    scores      = m.rank_action_candidates(cur, act_stacked, fut)  # [B, K]

    _pred_hw = (cfg.image_height, cfg.image_height)
    cur_f = _prep_image_tensor_batch(cur)
    if cur_f.shape[-2:] != _pred_hw:
        cur_f = F.interpolate(cur_f, size=_pred_hw, mode="bilinear", align_corners=False)
    fut_f = _prep_image_tensor_batch(fut)
    if fut_f.shape[-2:] != _pred_hw:
        fut_f = F.interpolate(fut_f, size=_pred_hw, mode="bilinear", align_corners=False)

    col_titles = (
        ["Current Frame", "GT Future"]
        + ["GT-action\n(should rank #1)"]
        + [f"Neg {k}\n(should rank low)" for k in range(1, K)]
    )

    panels     = []
    row_labels = []
    annots     = []

    for i in range(B):
        cur_np = _to_numpy_rgb(cur_f[i])
        fut_np = _to_numpy_rgb(fut_f[i])

        row   = [cur_np, fut_np]
        annot = [None, None]

        for k in range(K):
            with torch.no_grad():
                pred_k = m(cur[i:i+1], all_acts[k][i:i+1])["pred_future_image"][0]
            row.append(_to_numpy_rgb(pred_k))
            score_val = scores[i, k].item()
            annot.append(f"score = {score_val:.3f}")

        panels.append(row)
        annots.append(annot)

        pos_score  = scores[i, 0].item()
        neg_scores = [scores[i, k].item() for k in range(1, K)]
        ranked_correctly = pos_score > max(neg_scores) if neg_scores else True
        marker = "✓" if ranked_correctly else "✗"
        row_labels.append(
            f"S{i + 1} {marker}\nGT={pos_score:.3f}\n"
            + "\n".join(f"n{k}={s:.3f}" for k, s in enumerate(neg_scores, 1))
        )

    _mpl_save_grid(
        panels=panels,
        col_titles=col_titles,
        row_labels=row_labels,
        title=f"Step {step:,}  —  Action Candidate Ranking  (GT = index 0)",
        out_path=out / "candidate_ranking.png",
        cell_annotations=annots,
        heatmap_cmap="hot",
        cell_size=2.2,
    )

    m.train()


# ---------------------------------------------------------------------------
# Ranking eval curve plot (tiered 3-layer metrics over training steps)
# ---------------------------------------------------------------------------

def plot_ranking_eval_curves(output_dir: str) -> None:
    """Read train_metrics.jsonl and save ranking_eval_curves.png.

    Plots the tiered ranking metrics (strict_order_acc, tier pairwise accuracy,
    tier margins, tier mean scores, Spearman correlation) alongside the legacy
    pairwise/top1 metrics for comparison.
    """
    if not _HAS_MPL:
        return

    import json as _json
    jsonl_path = Path(output_dir) / "train_metrics.jsonl"
    if not jsonl_path.exists():
        return

    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(_json.loads(line))
                except _json.JSONDecodeError:
                    continue
    if not records:
        return

    # Skip if no tiered metrics exist yet
    if not any("tiered/strict_order_acc" in r for r in records):
        logger.debug("No tiered/* metrics in train_metrics.jsonl; "
                     "skipping ranking_eval_curves.png")
        return

    def _xy(key):
        pts = [(r["step"], r[key]) for r in records if key in r]
        return ([p[0] for p in pts], [p[1] for p in pts]) if pts else ([], [])

    def _plot(ax, key, label, color=None, marker=None, linestyle="-"):
        xs, ys = _xy(key)
        if xs:
            ax.plot(xs, ys, label=label, color=color, linestyle=linestyle,
                    marker=marker, markersize=3, linewidth=1.2)

    def _finalize(axes):
        for ax in axes.ravel():
            if not ax.lines:
                ax.axis("off")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        f"Ranking Eval Curves — {Path(output_dir).name}",
        fontsize=12, fontweight="bold",
    )

    # ── Row 0: tiered accuracy ────────────────────────────────────────────
    ax = axes[0, 0]
    _plot(ax, "tiered/strict_order_acc",
          "strict_order", color="crimson", marker="o")
    ax.set_title("Strict 3-way order accuracy")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    _plot(ax, "tiered/acc_success_gt_nearsuccess",
          "s > ns",  color="darkorange",  marker="s")
    _plot(ax, "tiered/acc_nearsuccess_gt_failure",
          "ns > f",  color="goldenrod",   marker="^")
    _plot(ax, "tiered/acc_success_gt_failure",
          "s > f",   color="darkgreen",   marker="D")
    ax.set_title("Tier pairwise accuracy")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    _plot(ax, "tiered/spearman_tier_corr",
          "Spearman ρ", color="purple", marker="o")
    ax.set_title("Spearman tier correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    # ── Row 1: tier margins ───────────────────────────────────────────────
    ax = axes[1, 0]
    _plot(ax, "tiered/margin_success_minus_failure",
          "success − failure", color="#c0392b", marker="D")
    ax.set_title("Success − Failure margin")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    _plot(ax, "tiered/margin_success_minus_nearsuccess",
          "s − ns", color="#e67e22")
    _plot(ax, "tiered/margin_nearsuccess_minus_failure",
          "ns − f", color="#9b59b6")
    ax.set_title("All tier margins")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    _plot(ax, "tiered/tier_score_success",
          "success",      color="#2ecc71")
    _plot(ax, "tiered/tier_score_nearsuccess",
          "near_success", color="#f39c12")
    _plot(ax, "tiered/tier_score_failure",
          "failure",      color="#e74c3c")
    ax.set_title("Mean DINO score per tier")
    ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    # ── Row 2: legacy comparison ──────────────────────────────────────────
    ax = axes[2, 0]
    _plot(ax, "rank/pairwise_acc",
          "pairwise (legacy)",  color="steelblue",       marker="o")
    _plot(ax, "tiered/pairwise_acc",
          "pairwise (tiered)",  color="cornflowerblue",  marker="s", linestyle="--")
    ax.set_title("Pairwise accuracy: legacy vs tiered")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Step")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    _plot(ax, "rank/top1_acc",
          "top1 (legacy)", color="seagreen")
    _plot(ax, "tiered/top1_acc",
          "top1 (tiered)", color="mediumseagreen", linestyle="--")
    ax.set_title("Top-1 accuracy: legacy vs tiered")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Step")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2, 2]
    _plot(ax, "rank/pos_score_mean",
          "pos score (legacy)", color="darkgreen")
    _plot(ax, "rank/neg_score_mean",
          "neg score (legacy)", color="darkred")
    ax.set_title("Legacy pos/neg mean scores")
    ax.set_xlabel("Step"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Row 3: fixed benchmark metrics (only when bench/* keys exist) ─────
    has_bench = any("bench/strict_order_acc" in r for r in records)
    if has_bench:
        fig.set_size_inches(15, 16)
        outer = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35) if False else None
        # Extend the figure by replacing it with a 4-row version
        plt.close(fig)
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        fig.suptitle(
            f"Ranking Eval Curves — {Path(output_dir).name}",
            fontsize=12, fontweight="bold",
        )

        # Re-draw rows 0-2
        ax = axes[0, 0]
        _plot(ax, "tiered/strict_order_acc", "strict_order", color="crimson", marker="o")
        ax.set_title("Strict 3-way order accuracy")
        ax.set_ylim(0, 1.05); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        _plot(ax, "tiered/acc_success_gt_nearsuccess", "s > ns", color="darkorange", marker="s")
        _plot(ax, "tiered/acc_nearsuccess_gt_failure", "ns > f", color="goldenrod",  marker="^")
        _plot(ax, "tiered/acc_success_gt_failure",     "s > f",  color="darkgreen",  marker="D")
        ax.set_title("Tier pairwise accuracy")
        ax.set_ylim(0, 1.05); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 2]
        _plot(ax, "tiered/spearman_tier_corr", "Spearman ρ", color="purple", marker="o")
        ax.set_title("Spearman tier correlation")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1, 0]
        _plot(ax, "tiered/margin_success_minus_failure", "success − failure",
              color="#c0392b", marker="D")
        ax.set_title("Success − Failure margin")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1, 1]
        _plot(ax, "tiered/margin_success_minus_nearsuccess", "s − ns", color="#e67e22")
        _plot(ax, "tiered/margin_nearsuccess_minus_failure",  "ns − f", color="#9b59b6")
        ax.set_title("All tier margins")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1, 2]
        _plot(ax, "tiered/tier_score_success",     "success",      color="#2ecc71")
        _plot(ax, "tiered/tier_score_nearsuccess", "near_success", color="#f39c12")
        _plot(ax, "tiered/tier_score_failure",     "failure",      color="#e74c3c")
        ax.set_title("Mean DINO score per tier")
        ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[2, 0]
        _plot(ax, "rank/pairwise_acc",   "pairwise (legacy)", color="steelblue",      marker="o")
        _plot(ax, "tiered/pairwise_acc", "pairwise (tiered)", color="cornflowerblue", marker="s",
              linestyle="--")
        ax.set_title("Pairwise accuracy: legacy vs tiered")
        ax.set_ylim(0, 1.05); ax.set_xlabel("Step")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[2, 1]
        _plot(ax, "rank/top1_acc",   "top1 (legacy)", color="seagreen")
        _plot(ax, "tiered/top1_acc", "top1 (tiered)", color="mediumseagreen", linestyle="--")
        ax.set_title("Top-1 accuracy: legacy vs tiered")
        ax.set_ylim(0, 1.05); ax.set_xlabel("Step")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[2, 2]
        _plot(ax, "rank/pos_score_mean", "pos score (legacy)", color="darkgreen")
        _plot(ax, "rank/neg_score_mean", "neg score (legacy)", color="darkred")
        ax.set_title("Legacy pos/neg mean scores")
        ax.set_xlabel("Step"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Row 3: fixed benchmark
        ax = axes[3, 0]
        _plot(ax, "bench/strict_order_acc",
              "bench strict",     color="crimson",     marker="o")
        _plot(ax, "tiered/strict_order_acc",
              "tiered strict",    color="crimson",     marker="s", linestyle="--")
        ax.set_title("Strict order acc: bench vs tiered")
        ax.set_ylim(0, 1.05); ax.set_xlabel("Step")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[3, 1]
        _plot(ax, "bench/acc_gt_temporal_neighbor",
              "s > temporal_nb",  color="#3498db", marker="^")
        _plot(ax, "bench/acc_gt_same_task_hard",
              "s > same_hard",    color="#e67e22", marker="D")
        _plot(ax, "bench/acc_gt_same_task_mismatch",
              "s > same_mismatch", color="#9b59b6", marker="v")
        ax.set_title("Bench: per-source-type accuracy")
        ax.set_ylim(0, 1.05); ax.set_xlabel("Step")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[3, 2]
        _plot(ax, "bench/mean_score_gt",
              "gt",               color="#2ecc71")
        _plot(ax, "bench/mean_score_temporal_neighbor",
              "temporal_nb",      color="#3498db", linestyle="--")
        _plot(ax, "bench/mean_score_same_task_hard",
              "same_hard",        color="#e67e22", linestyle="--")
        _plot(ax, "bench/mean_score_large_noise",
              "large_noise",      color="#e74c3c", linestyle=":")
        ax.set_title("Bench: mean DINO score per source type")
        ax.set_xlabel("Step"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    _finalize(axes)
    fig.tight_layout()
    out = Path(output_dir) / "ranking_eval_curves.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Ranking eval curves saved → %s", out)


# ---------------------------------------------------------------------------
# Training & eval curve plots
# ---------------------------------------------------------------------------

def plot_training_curves(output_dir: str) -> None:
    """Read train_metrics.jsonl and save separate loss/evaluation curve PNGs."""
    if not _HAS_MPL:
        return

    import json as _json
    jsonl_path = Path(output_dir) / "train_metrics.jsonl"
    if not jsonl_path.exists():
        return

    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(_json.loads(line))
                except _json.JSONDecodeError:
                    continue
    if not records:
        return

    def _xy(key):
        pts = [(r["step"], r[key]) for r in records if key in r]
        return ([p[0] for p in pts], [p[1] for p in pts]) if pts else ([], [])

    def _plot(ax, key, label, color=None, marker=None, linestyle="-"):
        xs, ys = _xy(key)
        if xs:
            ax.plot(xs, ys, label=label, color=color, linestyle=linestyle,
                    marker=marker, markersize=3, linewidth=1.2)

    def _finalize_axes(axes):
        for ax in axes.ravel():
            if not ax.lines:
                ax.axis("off")

    # --- Loss curves: only optimization targets used during training ---
    loss_fig, loss_axes = plt.subplots(3, 2, figsize=(12, 12))
    loss_fig.suptitle(f"Loss curves — {Path(output_dir).name}", fontsize=12, fontweight="bold")

    ax = loss_axes[0, 0]
    _plot(ax, "loss", "total", color="black")
    ax.set_title("Total loss"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = loss_axes[0, 1]
    _plot(ax, "recon", "recon", color="steelblue")
    _plot(ax, "dino_feature", "dino_feat", color="orange")
    ax.set_title("Core losses"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = loss_axes[1, 0]
    _plot(ax, "focus_supervision", "focus_sup", color="green")
    _plot(ax, "focus_sparsity", "sparsity", color="gray", linestyle="--")
    ax.set_title("Focus losses"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = loss_axes[1, 1]
    _plot(ax, "fg_recon", "fg_recon", color="#117733")
    _plot(ax, "fg_grad", "fg_grad", color="#44aa99")
    _plot(ax, "bg_residual_penalty", "bg_pen", color="#cc6677", linestyle="--")
    ax.set_title("Optional pixel losses"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = loss_axes[2, 0]
    _plot(ax, "lr", "lr", color="#332288")
    ax.set_title("Learning rate"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = loss_axes[2, 1]
    _plot(ax, "grad_norm", "grad_norm", color="#aa4499")
    ax.set_title("Gradient norm"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    _finalize_axes(loss_axes)
    loss_fig.tight_layout()
    loss_out = Path(output_dir) / "loss_curves.png"
    loss_fig.savefig(str(loss_out), dpi=120, bbox_inches="tight")
    plt.close(loss_fig)
    logger.info("Loss curves saved → %s", loss_out)

    # --- Evaluation / diagnostics curves: ranking + feature/focus/mask metrics ---
    eval_fig, eval_axes = plt.subplots(4, 3, figsize=(15, 16))
    eval_fig.suptitle(f"Evaluation curves — {Path(output_dir).name}", fontsize=12, fontweight="bold")

    ax = eval_axes[0, 0]
    _plot(ax, "rank/pairwise_acc", "pairwise_acc", color="crimson", marker="o")
    _plot(ax, "rank/top1_acc", "top1_acc", color="darkorange", marker="s")
    ax.set_title("Action ranking accuracy"); ax.set_xlabel("Step")
    ax.set_ylim(bottom=0); ax.legend(); ax.grid(alpha=0.3)

    ax = eval_axes[0, 1]
    _plot(ax, "rank/mean_margin", "mean_margin", color="purple", marker="D")
    _plot(ax, "rank/hardest_negative_margin", "hard_margin", color="mediumpurple", marker="^")
    ax.set_title("Ranking margin"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = eval_axes[0, 2]
    _plot(ax, "rank/pos_score_mean", "pos_score", color="seagreen")
    _plot(ax, "rank/neg_score_mean", "neg_score", color="tomato")
    ax.set_title("Ranking scores (pos vs neg)"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = eval_axes[1, 0]
    _plot(ax, "feat/dino_cosine_similarity", "cosine_sim", color="teal")
    _plot(ax, "feat/dino_feature_mse", "feat_mse", color="#88ccee", linestyle="--")
    ax.set_title("DINO feature metrics"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = eval_axes[1, 1]
    _plot(ax, "focus/focus_mean", "focus_mean", color="goldenrod")
    _plot(ax, "focus/focus_entropy", "entropy", color="coral", linestyle="--")
    ax.set_title("Focus map statistics"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = eval_axes[1, 2]
    _plot(ax, "focus/iou_vs_change", "IoU", color="forestgreen", marker="^")
    _plot(ax, "focus/dice_vs_change", "Dice", color="seagreen", marker="v")
    ax.set_title("Focus vs change target"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = eval_axes[2, 0]
    _plot(ax, "context_mask_mean", "ctx_mean", color="#4477aa")
    _plot(ax, "write_mask_mean", "wrt_mean", color="#ee6677")
    _plot(ax, "diag/context_mask_entropy", "ctx_ent", color="#4477aa", linestyle="--")
    _plot(ax, "diag/write_mask_entropy", "wrt_ent", color="#ee6677", linestyle="--")
    ax.set_title("Dual Mask: mean & entropy"); ax.set_xlabel("Step")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = eval_axes[2, 1]
    _plot(ax, "write_context_gap", "wrt-ctx gap", color="#228833")
    _plot(ax, "write_context_ratio", "wrt/ctx ratio", color="#aa3377")
    _plot(ax, "diag/context_write_overlap", "ctx&wrt", color="#ccbb44", linestyle="--")
    ax.set_title("Dual Mask: gap & ratio"); ax.set_xlabel("Step")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = eval_axes[2, 2]
    _plot(ax, "diag/context_iou", "ctx_iou", color="#4477aa", marker="^")
    _plot(ax, "diag/write_iou", "wrt_iou", color="#ee6677", marker="v")
    _plot(ax, "diag/write_minus_context_iou", "delta_iou", color="#228833", linestyle="--")
    ax.set_title("Dual Mask: IoU vs change"); ax.set_xlabel("Step")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = eval_axes[3, 0]
    _plot(ax, "bg_fg_residual_ratio", "bg/fg_res_ctx", color="#999933")
    _plot(ax, "bg_fg_residual_ratio_write", "bg/fg_res_wrt", color="#bb5566")
    _plot(ax, "diag/write_bg_fg_residual_ratio", "wrt_bg/fg_res", color="#bb5566", linestyle=":")
    ax.set_title("Residual locality"); ax.set_xlabel("Step")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = eval_axes[3, 1]
    _plot(ax, "diag/change_magnitude_ratio", "chg_ratio", color="#004488")
    _plot(ax, "diag/grad_l1_gap", "grad_gap", color="#ddaa33")
    ax.set_title("Change & sharpness"); ax.set_xlabel("Step")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = eval_axes[3, 2]
    _plot(ax, "diag/fg_recon_l1_write", "fg_err_wrt", color="#117733")
    _plot(ax, "diag/bg_recon_l1_write", "bg_err_wrt", color="#cc6677")
    _plot(ax, "diag/fg_bg_recon_ratio_write", "bg/fg_err_wrt", color="#882255", linestyle="--")
    ax.set_title("Foreground/background eval"); ax.set_xlabel("Step")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    _finalize_axes(eval_axes)
    eval_fig.tight_layout()
    eval_out = Path(output_dir) / "evaluate_curves.png"
    eval_fig.savefig(str(eval_out), dpi=120, bbox_inches="tight")
    plt.close(eval_fig)
    logger.info("Evaluation curves saved → %s", eval_out)

    # Tiered ranking eval curves (written only when tiered metrics are present)
    plot_ranking_eval_curves(output_dir)


# ---------------------------------------------------------------------------
# Per-task summary CSV
# ---------------------------------------------------------------------------

def save_per_task_summary(
    task_metrics: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    """Save per-task metrics as a CSV.

    task_metrics: {task_name: {metric_name: value, …}, …}
    """
    import csv
    out = Path(output_dir) / "eval_reports" / "libero" / "per_task_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    all_keys = set()
    for v in task_metrics.values():
        all_keys.update(v.keys())
    all_keys = sorted(all_keys)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task"] + all_keys, extrasaction="ignore")
        writer.writeheader()
        for task, metrics in sorted(task_metrics.items()):
            writer.writerow({"task": task, **metrics})

    logger.info("Per-task summary saved: %s", out)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    import torch
    from torch.utils.data import DataLoader, default_collate

    from .focused_config import FocusedWMConfig
    from .focused_model import ActionConditionedFocusedResidualWM
    from ..datasets.libero.data import resolve_dataset_name
    from .train_focused_libero import FocusedWindowDataset

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Offline evaluation for ActionConditionedFocusedResidualWM."
    )
    parser.add_argument("--mode", type=str, default="full_eval",
                        choices=["full_eval"],
                        help="Evaluation mode (currently only full_eval).")
    parser.add_argument("--task-suite", type=str, required=True,
                        help="LIBERO task suite: spatial | object | goal | 10")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing RLDS dataset sub-directories.")
    parser.add_argument("--model-pt", type=str, required=True,
                        help="Path to model.pt (PyTorch state-dict checkpoint).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where metrics and visualizations are saved.")
    parser.add_argument("--max-batches", type=int, default=200,
                        help="Maximum number of batches to evaluate over.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Evaluation batch size per device.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-json", type=str, default="",
                        help="Path to config.json saved alongside the checkpoint. "
                             "If omitted, FocusedWMConfig defaults are used.")
    parser.add_argument("--save-vis", action="store_true", default=False,
                        help="Save casebook and focus-analysis PNGs.")
    parser.add_argument("--save-report", action="store_true", default=True,
                        help="Save metrics.json and per-task CSV (default: True).")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device: auto | cuda | cpu")
    args = parser.parse_args()

    # ---- Device -------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # ---- Config -------------------------------------------------------
    cfg = FocusedWMConfig()
    if args.config_json:
        cfg_path = Path(args.config_json)
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg_dict = json.load(f)
            for k, v in cfg_dict.items():
                if hasattr(cfg, k) and not callable(getattr(cfg, k)):
                    setattr(cfg, k, v)
            logger.info("Config loaded from %s", cfg_path)
        else:
            logger.warning("--config-json not found: %s — using defaults", cfg_path)

    # ---- Model --------------------------------------------------------
    logger.info("Loading model from %s", args.model_pt)
    model = ActionConditionedFocusedResidualWM(cfg)
    state = torch.load(args.model_pt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys in checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded. Parameters: %.1fM", n_params)

    # ---- Dataset ------------------------------------------------------
    dataset_name  = resolve_dataset_name(args.task_suite)
    segment_length = cfg.action_horizon + 1
    dataset = FocusedWindowDataset(
        dataset_name=dataset_name,
        data_dir=args.data_root,
        segment_length=segment_length,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=default_collate,
        num_workers=args.num_workers,
    )

    # ---- Evaluate -----------------------------------------------------
    logger.info(
        "Running %s | task=%s | max_batches=%d",
        args.mode, args.task_suite, args.max_batches,
    )
    summary = run_full_evaluation(
        model=model,
        data_loader=loader,
        device=device,
        cfg=cfg,
        output_dir=args.output_dir,
        task_suite=args.task_suite,
        max_batches=args.max_batches,
    )

    # ---- Print summary ------------------------------------------------
    logger.info("=" * 60)
    logger.info("Evaluation complete — %s", args.output_dir)
    for k in sorted(summary.keys()):
        v = summary[k]
        if isinstance(v, float):
            logger.info("  %-40s %.5f", k, v)
        else:
            logger.info("  %-40s %s", k, v)

    sys.exit(0)
