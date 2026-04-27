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

        # --- Motion-sensitive error ---
        if change is not None:
            ms_recon = _motion_sensitive_recon(pred, fut_f.to(pred.device), change, cfg.patch_hw)
            agg.setdefault("motion_sensitive_recon", []).append(ms_recon)

        # --- Collect casebook sample (first item in batch) ---
        recon_loss = im_met["future_image_smooth_l1"]
        casebook[f"batch_{n_batches:04d}"] = {
            "recon_loss": recon_loss,
            "focus_mean": fm_met.get("focus_mean", 0),
            "batch_idx": n_batches,
            # Store tensors for later rendering
            "cur_t":  cur[0].cpu(),
            "fut_t":  fut[0].cpu(),
            "pred_t": pred[0].cpu(),
            "focus_t": focus[0].cpu(),
            "change_t": change[0].cpu() if change is not None else None,
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
        _save_focus_analysis(casebook, out_viz)
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
# Training & eval curve plots
# ---------------------------------------------------------------------------

def plot_training_curves(output_dir: str) -> None:
    """Read train_metrics.jsonl and save training/eval curve PNGs.

    Outputs:
      {output_dir}/training_curves.png   — loss + ranking + feature/focus metrics
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

    def _xy(key):
        pts = [(r["step"], r[key]) for r in records if key in r]
        return ([p[0] for p in pts], [p[1] for p in pts]) if pts else ([], [])

    def _plot(ax, key, label, color=None, marker=None, linestyle="-"):
        xs, ys = _xy(key)
        if xs:
            ax.plot(xs, ys, label=label, color=color, linestyle=linestyle,
                    marker=marker, markersize=3, linewidth=1.2)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"Training curves — {Path(output_dir).name}", fontsize=12, fontweight="bold")

    # --- Row 0: Losses ---
    ax = axes[0, 0]
    _plot(ax, "loss", "total", color="black")
    ax.set_title("Total loss"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    _plot(ax, "recon", "recon", color="steelblue")
    ax.set_title("Reconstruction loss"); ax.set_xlabel("Step"); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    _plot(ax, "dino_feature",      "dino_feat",  color="orange")
    _plot(ax, "focus_supervision", "focus_sup",  color="green")
    _plot(ax, "focus_sparsity",    "sparsity",   color="gray", linestyle="--")
    ax.set_title("Auxiliary losses"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    # --- Row 1: Action ranking eval ---
    ax = axes[1, 0]
    _plot(ax, "rank/pairwise_acc", "pairwise_acc", color="crimson",    marker="o")
    _plot(ax, "rank/top1_acc",     "top1_acc",     color="darkorange", marker="s")
    ax.set_title("Action ranking accuracy"); ax.set_xlabel("Step")
    ax.set_ylim(bottom=0); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    _plot(ax, "rank/mean_margin",             "mean_margin",  color="purple",      marker="D")
    _plot(ax, "rank/hardest_negative_margin", "hard_margin",  color="mediumpurple", marker="^")
    ax.set_title("Ranking margin"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    _plot(ax, "rank/pos_score_mean", "pos_score", color="seagreen")
    _plot(ax, "rank/neg_score_mean", "neg_score", color="tomato")
    ax.set_title("Ranking scores (pos vs neg)"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    # --- Row 2: Feature & focus metrics ---
    ax = axes[2, 0]
    _plot(ax, "feat/dino_cosine_similarity", "cosine_sim", color="teal")
    ax.set_title("DINO cosine similarity"); ax.set_xlabel("Step"); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    _plot(ax, "focus/focus_mean",    "focus_mean", color="goldenrod")
    _plot(ax, "focus/focus_entropy", "entropy",    color="coral", linestyle="--")
    ax.set_title("Focus map statistics"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2, 2]
    _plot(ax, "focus/iou_vs_change",  "IoU",  color="forestgreen", marker="^")
    _plot(ax, "focus/dice_vs_change", "Dice", color="seagreen",    marker="v")
    ax.set_title("Focus vs change target"); ax.set_xlabel("Step"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(output_dir) / "training_curves.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved → %s", out_path)


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
