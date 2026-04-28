"""3-tier action ranking evaluation: success / near_success / failure.

Evaluation hierarchy
--------------------
  success      (tier=2)  GT action chunk
  near_success (tier=1)  GT + small Gaussian noise
  failure      (tier=0)  mix: large noise / shuffle / roll

Candidate layout (K = 1 + n_near_success + n_failure, default K=6)
  index 0              : success (GT)
  index 1 .. n_ns      : near_success (small_noise × n_ns)
  index n_ns+1 .. K-1  : failure (large_noise, shuffle, roll, cycling)

Metrics
-------
  acc_success_gt_nearsuccess     P(score_s > score_ns) over all (s,ns) pairs
  acc_nearsuccess_gt_failure     P(score_ns > score_f) over all (ns,f) pairs
  acc_success_gt_failure         P(score_s > score_f)  over all (s,f) pairs
  strict_order_acc               P(mean_s > mean_ns > mean_f) per item
  margin_success_minus_nearsuccess
  margin_nearsuccess_minus_failure
  margin_success_minus_failure
  tier_score_success / _nearsuccess / _failure  mean DINO score per tier
  spearman_tier_corr             mean Spearman ρ(score, tier_ordinal) per item

  # legacy  (matches existing run_action_rank_eval)
  pairwise_acc  top1_acc  mean_margin  pos_score_mean  neg_score_mean
  hardest_negative_margin

Outputs (under <output_dir>/rank_eval/)
---------------------------------------
  rank_eval_dataset.json     built once on first eval call, metadata only
  rank_eval_metrics.json     overwritten each eval step
  rank_eval_candidates.csv   1 row per candidate (overwritten)
  rank_eval_items.csv        1 row per item (overwritten)
  rank_eval_summary.csv      metric summary (overwritten)
  tier_score_distribution.png
  tier_margin_hist.png
  strict_order_examples.png
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..eval_roi_utils import append_ranking_jsonl as _shared_append_ranking_jsonl
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Tier constants
# ---------------------------------------------------------------------------

TIER_SUCCESS      = "success"
TIER_NEAR_SUCCESS = "near_success"
TIER_FAILURE      = "failure"

TIER_ORDINAL: Dict[str, int] = {
    TIER_SUCCESS:      2,
    TIER_NEAR_SUCCESS: 1,
    TIER_FAILURE:      0,
}

_FAILURE_MODE_CYCLE = ["large_noise", "shuffle", "roll"]

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class TieredEvalDataset:
    """Fixed evaluation set built from probe batches.

    Each entry in ``batches`` is a triple (cur, fut, candidates):
      cur        : [B, H, W, 3] uint8 CPU tensor
      fut        : [B, H, W, 3] uint8 CPU tensor
      candidates : [B, K, H_act, 7] float32 CPU tensor

    ``tiers[k]`` and ``modes[k]`` describe the k-th candidate slot.
    Candidate order: [success(1), near_success(n_ns), failure(n_f)]
    """

    batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    tiers: List[str]        # length K
    modes: List[str]        # length K
    n_near_success: int
    n_failure: int
    near_noise_std: float
    fail_noise_std: float
    seed: int
    task_name: str = ""

    @property
    def K(self) -> int:
        return len(self.tiers)

    @property
    def n_items(self) -> int:
        return sum(b[0].shape[0] for b in self.batches)

    @property
    def success_idx(self) -> int:
        return 0

    @property
    def near_success_indices(self) -> List[int]:
        return list(range(1, 1 + self.n_near_success))

    @property
    def failure_indices(self) -> List[int]:
        return list(range(1 + self.n_near_success, self.K))


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _build_tiered_candidates(
    acts: torch.Tensor,       # [B, H_act, 7]  — CPU tensor
    n_near_success: int,
    n_failure: int,
    near_noise_std: float,
    fail_noise_std: float,
    gen: torch.Generator,     # CPU generator for reproducibility
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Return (candidates [B, K, H_act, 7], tiers [K], modes [K])."""
    B, H, D = acts.shape

    parts: List[torch.Tensor] = [acts.unsqueeze(1)]  # [B,1,H,7] success
    tiers: List[str] = [TIER_SUCCESS]
    modes: List[str] = ["gt"]

    # near_success: small independent noise realizations
    for _ in range(n_near_success):
        noise = torch.randn(B, H, D, generator=gen) * near_noise_std
        parts.append((acts + noise).unsqueeze(1))
        tiers.append(TIER_NEAR_SUCCESS)
        modes.append("small_noise")

    # failure: cycle through [large_noise, shuffle, roll]
    for i in range(n_failure):
        mode = _FAILURE_MODE_CYCLE[i % len(_FAILURE_MODE_CYCLE)]
        if mode == "large_noise":
            noise = torch.randn(B, H, D, generator=gen) * fail_noise_std
            parts.append((acts + noise).unsqueeze(1))
        elif mode == "shuffle":
            perm = torch.randperm(H, generator=gen)
            parts.append(acts[:, perm, :].unsqueeze(1))
        else:  # roll — use another sample's action within the batch
            parts.append(torch.roll(acts, 1, dims=0).unsqueeze(1))
        tiers.append(TIER_FAILURE)
        modes.append(mode)

    return torch.cat(parts, dim=1), tiers, modes   # [B, K, H, 7]


def build_tiered_eval_dataset(
    probe_batches: List[Dict[str, torch.Tensor]],
    n_near_success: int = 2,
    n_failure: int = 3,
    near_noise_std: float = 0.05,
    fail_noise_std: float = 0.30,
    seed: int = 42,
    task_name: str = "",
) -> TieredEvalDataset:
    """Build a fixed TieredEvalDataset from probe batches.

    The CPU RNG is seeded once and advanced deterministically, so identical
    candidates are produced on every call with the same arguments.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    tiers_template: Optional[List[str]] = None
    modes_template: Optional[List[str]] = None
    built: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for batch in probe_batches:
        cur  = batch["current_pixels"].cpu()   # [B, H, W, 3] uint8
        fut  = batch["future_pixels"].cpu()    # [B, H, W, 3] uint8
        acts = batch["actions"].cpu()          # [B, H_act, 7] float32

        candidates, tiers, modes = _build_tiered_candidates(
            acts, n_near_success, n_failure, near_noise_std, fail_noise_std, gen,
        )
        built.append((cur, fut, candidates))

        if tiers_template is None:
            tiers_template = tiers
            modes_template = modes

    if not built:
        raise ValueError("probe_batches is empty; cannot build TieredEvalDataset.")

    return TieredEvalDataset(
        batches=built,
        tiers=tiers_template,
        modes=modes_template,
        n_near_success=n_near_success,
        n_failure=n_failure,
        near_noise_std=near_noise_std,
        fail_noise_std=fail_noise_std,
        seed=seed,
        task_name=task_name,
    )


# ---------------------------------------------------------------------------
# Ranking utilities
# ---------------------------------------------------------------------------

def _row_ranks(x: torch.Tensor) -> torch.Tensor:
    """0-based rank of each element within its row.  [N, K] → [N, K] float."""
    return x.argsort(dim=1).argsort(dim=1).float()


def _spearman_batch(scores: torch.Tensor, ordinals: torch.Tensor) -> float:
    """Mean Spearman ρ between score rows and ordinal rows. [N,K]×[N,K] → float."""
    sr  = _row_ranks(scores.float())
    ori = _row_ranks(ordinals.float())
    sr  = sr  - sr.mean(dim=1, keepdim=True)
    ori = ori - ori.mean(dim=1, keepdim=True)
    num = (sr * ori).sum(dim=1)
    den = (sr.norm(dim=1) * ori.norm(dim=1)).clamp(min=1e-8)
    return (num / den).mean().item()


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_tiered_metrics(
    all_scores: torch.Tensor,
    dataset: TieredEvalDataset,
) -> dict:
    """Compute all tiered + legacy metrics from per-item score matrix [N, K]."""
    N, K = all_scores.shape
    s_i  = dataset.success_idx
    ns_i = dataset.near_success_indices
    f_i  = dataset.failure_indices

    s_sc  = all_scores[:, s_i]          # [N]
    ns_sc = all_scores[:, ns_i]         # [N, n_ns]
    f_sc  = all_scores[:, f_i]          # [N, n_f]

    s_mean  = s_sc                       # [N]
    ns_mean = ns_sc.mean(dim=1)          # [N]
    f_mean  = f_sc.mean(dim=1)           # [N]

    # tier-wise pairwise accuracy
    acc_s_ns = (s_sc.unsqueeze(1) > ns_sc).float().mean().item()
    acc_ns_f = (ns_sc.unsqueeze(2) > f_sc.unsqueeze(1)).float().mean().item()
    acc_s_f  = (s_sc.unsqueeze(1) > f_sc).float().mean().item()

    # strict 3-way ordering per item
    strict = ((s_mean > ns_mean) & (ns_mean > f_mean)).float().mean().item()

    # tier margins
    margin_s_ns = (s_mean - ns_mean).mean().item()
    margin_ns_f = (ns_mean - f_mean).mean().item()
    margin_s_f  = (s_mean - f_mean).mean().item()

    # tier mean scores
    score_s  = s_sc.mean().item()
    score_ns = ns_mean.mean().item()
    score_f  = f_mean.mean().item()

    # Spearman: rank correlation between model score and tier ordinal
    tier_ord = torch.tensor(
        [TIER_ORDINAL[t] for t in dataset.tiers], dtype=torch.float32,
    )  # [K]
    spearman = _spearman_batch(all_scores, tier_ord.unsqueeze(0).expand(N, K))

    # legacy metrics — success vs all non-success candidates
    all_neg = torch.cat([ns_sc, f_sc], dim=1)   # [N, n_ns + n_f]
    pairwise_acc = (s_sc.unsqueeze(1) > all_neg).float().mean().item()
    top1_acc     = (s_sc > all_neg.max(dim=1).values).float().mean().item()
    mean_margin  = (s_sc.unsqueeze(1) - all_neg).mean().item()
    hard_margin  = (s_sc - all_neg.max(dim=1).values).mean().item()

    return {
        "acc_success_gt_nearsuccess":       acc_s_ns,
        "acc_nearsuccess_gt_failure":       acc_ns_f,
        "acc_success_gt_failure":           acc_s_f,
        "strict_order_acc":                 strict,
        "margin_success_minus_nearsuccess": margin_s_ns,
        "margin_nearsuccess_minus_failure": margin_ns_f,
        "margin_success_minus_failure":     margin_s_f,
        "tier_score_success":               score_s,
        "tier_score_nearsuccess":           score_ns,
        "tier_score_failure":               score_f,
        "spearman_tier_corr":               spearman,
        # legacy keys
        "pairwise_acc":                     pairwise_acc,
        "top1_acc":                         top1_acc,
        "mean_margin":                      mean_margin,
        "pos_score_mean":                   score_s,
        "neg_score_mean":                   all_neg.mean().item(),
        "hardest_negative_margin":          hard_margin,
    }


# ---------------------------------------------------------------------------
# JSON / CSV saving
# ---------------------------------------------------------------------------

def save_tiered_rank_json(
    all_item_scores: List[dict],
    metrics: dict,
    dataset: TieredEvalDataset,
    step: int,
    out_dir: Path,
) -> None:
    """Save rank_eval_dataset.json (once) and rank_eval_metrics.json (each eval)."""

    # dataset.json — written only on first call
    dataset_path = out_dir / "rank_eval_dataset.json"
    if not dataset_path.exists():
        ds_doc = {
            "metadata": {
                "task_name":              dataset.task_name,
                "num_items":              dataset.n_items,
                "n_near_success":         dataset.n_near_success,
                "n_failure":              dataset.n_failure,
                "near_success_noise_std": dataset.near_noise_std,
                "failure_noise_std":      dataset.fail_noise_std,
                "seed":                   dataset.seed,
                "K":                      dataset.K,
                "tiers":                  dataset.tiers,
                "modes":                  dataset.modes,
                "tier_ordinal":           TIER_ORDINAL,
                "created_at":             datetime.datetime.utcnow().isoformat(),
            },
            "items": [
                {
                    "item_id":         it["item_id"],
                    "batch_idx":       it["batch_idx"],
                    "sample_in_batch": it["sample_in_batch"],
                    "candidates": [
                        {
                            "candidate_idx": k,
                            "tier":          dataset.tiers[k],
                            "mode":          dataset.modes[k],
                            "noise_std": (
                                dataset.near_noise_std
                                if dataset.tiers[k] == TIER_NEAR_SUCCESS
                                else dataset.fail_noise_std
                                if (dataset.tiers[k] == TIER_FAILURE
                                    and dataset.modes[k] == "large_noise")
                                else None
                            ),
                        }
                        for k in range(dataset.K)
                    ],
                }
                for it in all_item_scores
            ],
        }
        dataset_path.write_text(json.dumps(ds_doc, indent=2))
        logger.info("Saved rank_eval_dataset.json → %s", dataset_path)

    # metrics.json — overwritten each eval
    metrics_doc = {
        "step":     step,
        "metrics":  metrics,
        "saved_at": datetime.datetime.utcnow().isoformat(),
    }
    metrics_path = out_dir / "rank_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics_doc, indent=2))
    logger.info("Saved rank_eval_metrics.json (step=%d)", step)


def save_tiered_rank_csv(
    all_item_scores: List[dict],
    metrics: dict,
    dataset: TieredEvalDataset,
    step: int,
    out_dir: Path,
) -> None:
    """Save rank_eval_candidates.csv, rank_eval_items.csv, rank_eval_summary.csv."""

    # 1. candidates.csv — 1 row per (item, candidate)
    cand_path = out_dir / "rank_eval_candidates.csv"
    with cand_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "item_id", "batch_idx", "sample_in_batch",
                    "candidate_idx", "tier", "mode", "noise_std", "score"])
        for it in all_item_scores:
            for k, score in enumerate(it["scores"]):
                tier = dataset.tiers[k]
                mode = dataset.modes[k]
                if tier == TIER_NEAR_SUCCESS:
                    ns = dataset.near_noise_std
                elif tier == TIER_FAILURE and mode == "large_noise":
                    ns = dataset.fail_noise_std
                else:
                    ns = ""
                w.writerow([
                    step,
                    it["item_id"], it["batch_idx"], it["sample_in_batch"],
                    k, tier, mode, ns, f"{score:.6f}",
                ])
    logger.info("Saved rank_eval_candidates.csv (%d rows)",
                len(all_item_scores) * dataset.K)

    # 2. items.csv — 1 row per item with tier-aggregated scores
    s_i  = dataset.success_idx
    ns_i = dataset.near_success_indices
    f_i  = dataset.failure_indices

    items_path = out_dir / "rank_eval_items.csv"
    with items_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "item_id", "batch_idx", "sample_in_batch",
                    "score_success", "score_ns_mean", "score_f_mean",
                    "margin_s_ns", "margin_ns_f", "margin_s_f", "strict_order"])
        for it in all_item_scores:
            sc  = it["scores"]
            s   = sc[s_i]
            ns  = (sum(sc[i] for i in ns_i) / len(ns_i)) if ns_i else 0.0
            fm  = (sum(sc[i] for i in f_i)  / len(f_i))  if f_i  else 0.0
            w.writerow([
                step, it["item_id"], it["batch_idx"], it["sample_in_batch"],
                f"{s:.6f}", f"{ns:.6f}", f"{fm:.6f}",
                f"{s - ns:.6f}", f"{ns - fm:.6f}", f"{s - fm:.6f}",
                int(s > ns > fm),
            ])
    logger.info("Saved rank_eval_items.csv (%d rows)", len(all_item_scores))

    # 3. summary.csv — overall metric table
    summary_path = out_dir / "rank_eval_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "metric", "value"])
        for k, v in sorted(metrics.items()):
            w.writerow([step, k, f"{v:.6f}"])
    logger.info("Saved rank_eval_summary.csv")


def append_tiered_rank_jsonl(
    all_item_scores: List[dict],
    metrics: dict,
    dataset: TieredEvalDataset,
    step: int,
    out_dir: Path,
    task_suite: str = "",
) -> None:
    """Append one JSON line to rank_eval_candidates.jsonl.

    Unlike rank_eval_metrics.json (overwritten each step), this file accumulates
    history across the entire training run so that analyze_worldmodel_eval.py
    can plot ranking trajectories and compute step-over-step deltas.

    Format per line::
        {
          "step": 49000,
          "task_name": "libero_spatial",
          "task_suite": "spatial",
          "n_items": 128,
          "K": 6,
          "tiers": ["success", "near_success", ...],
          "modes": ["gt", "small_noise", ...],
          "metrics": { ... },           # same dict as rank_eval_metrics.json
          "per_item": [
              {"item_id": 0, "scores": [0.972, 0.970, ...]},
              ...
          ],
          "saved_at": "2026-04-28T..."
        }
    """
    _shared_append_ranking_jsonl(
        out_path=out_dir / "rank_eval_candidates.jsonl",
        step=step,
        model_type="residual_focused",
        task_name=dataset.task_name,
        task_suite=task_suite,
        n_items=len(all_item_scores),
        K=dataset.K,
        tiers=dataset.tiers,
        modes=dataset.modes,
        metrics=metrics,
        per_item=[{"item_id": it["item_id"], "scores": it["scores"]}
                  for it in all_item_scores],
        score_breakdown=None,
    )


# ---------------------------------------------------------------------------
# PNG visualizations
# ---------------------------------------------------------------------------

def _to_rgb(t: torch.Tensor) -> np.ndarray:
    """[H,W,3] uint8 or float tensor → HWC uint8 numpy."""
    arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else t
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr


def _plot_score_distribution(
    all_item_scores: List[dict],
    dataset: TieredEvalDataset,
    step: int,
    out_dir: Path,
) -> None:
    """tier_score_distribution.png — boxplot + mean-bar per tier."""
    s_i  = dataset.success_idx
    ns_i = dataset.near_success_indices
    f_i  = dataset.failure_indices

    s_sc  = [it["scores"][s_i] for it in all_item_scores]
    ns_sc = [sc for it in all_item_scores for sc in [it["scores"][i] for i in ns_i]]
    f_sc  = [sc for it in all_item_scores for sc in [it["scores"][i] for i in f_i]]

    data   = [s_sc, ns_sc, f_sc]
    labels = ["success", "near_success", "failure"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Tier Score Distribution — step {step:,}", fontsize=11)

    # boxplot
    ax = axes[0]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labels)
    ax.set_title("Score distribution (boxplot)")
    ax.set_ylabel("DINO cosine score"); ax.grid(axis="y", alpha=0.3)

    # mean bar with ±1 std
    ax = axes[1]
    means = [float(np.mean(d)) for d in data]
    stds  = [float(np.std(d))  for d in data]
    ax.bar(labels, means, color=colors, alpha=0.7, edgecolor="black", linewidth=0.8)
    ax.errorbar(labels, means, yerr=stds, fmt="none", color="black", capsize=5)
    ax.set_title("Mean score ± std per tier")
    ax.set_ylabel("DINO cosine score"); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "tier_score_distribution.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved tier_score_distribution.png")


def _plot_margin_histograms(
    all_item_scores: List[dict],
    dataset: TieredEvalDataset,
    step: int,
    out_dir: Path,
) -> None:
    """tier_margin_hist.png — per-item margin histograms for each tier pair."""
    s_i  = dataset.success_idx
    ns_i = dataset.near_success_indices
    f_i  = dataset.failure_indices

    margins_s_ns, margins_ns_f, margins_s_f = [], [], []
    for it in all_item_scores:
        sc = it["scores"]
        s  = sc[s_i]
        ns = (sum(sc[i] for i in ns_i) / len(ns_i)) if ns_i else 0.0
        fm = (sum(sc[i] for i in f_i)  / len(f_i))  if f_i  else 0.0
        margins_s_ns.append(s - ns)
        margins_ns_f.append(ns - fm)
        margins_s_f.append(s - fm)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Tier Margin Histograms — step {step:,}", fontsize=11)

    for ax, data, label, color in [
        (axes[0], margins_s_ns, "success − near_success", "#e67e22"),
        (axes[1], margins_ns_f, "near_success − failure",  "#9b59b6"),
        (axes[2], margins_s_f,  "success − failure",       "#c0392b"),
    ]:
        ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(float(np.mean(data)), color="red", linestyle="-", linewidth=1.5,
                   label=f"mean={float(np.mean(data)):.3f}")
        ax.set_title(label); ax.set_xlabel("Margin"); ax.set_ylabel("Count")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / "tier_margin_hist.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved tier_margin_hist.png")


def _plot_strict_order_examples(
    all_item_scores: List[dict],
    dataset: TieredEvalDataset,
    step: int,
    out_dir: Path,
    n_examples: int = 3,
) -> None:
    """strict_order_examples.png — correct vs incorrect ordering with images."""
    s_i  = dataset.success_idx
    ns_i = dataset.near_success_indices
    f_i  = dataset.failure_indices

    correct_items: List[dict] = []
    wrong_items:   List[dict] = []

    for it in all_item_scores:
        sc = it["scores"]
        s  = sc[s_i]
        ns = (sum(sc[i] for i in ns_i) / len(ns_i)) if ns_i else 0.0
        fm = (sum(sc[i] for i in f_i)  / len(f_i))  if f_i  else 0.0
        record = {**it, "s_score": s, "ns_score": ns, "f_score": fm}
        (correct_items if (s > ns > fm) else wrong_items).append(record)

    n_c = min(n_examples, len(correct_items))
    n_w = min(n_examples, len(wrong_items))
    shown = (
        [(r, True)  for r in correct_items[:n_c]] +
        [(r, False) for r in wrong_items[:n_w]]
    )
    if not shown:
        return

    n_rows = len(shown)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3.0 * n_rows),
                              squeeze=False)
    fig.suptitle(
        f"Strict Order Examples — step {step:,}\n"
        "Columns: current | GT future | tier scores",
        fontsize=10,
    )

    for r, (item_rec, is_correct) in enumerate(shown):
        bid  = item_rec["batch_idx"]
        sid  = item_rec["sample_in_batch"]
        cur_np = _to_rgb(dataset.batches[bid][0][sid])
        fut_np = _to_rgb(dataset.batches[bid][1][sid])

        axes[r, 0].imshow(cur_np); axes[r, 0].axis("off")
        if r == 0:
            axes[r, 0].set_title("current", fontsize=9)

        axes[r, 1].imshow(fut_np); axes[r, 1].axis("off")
        if r == 0:
            axes[r, 1].set_title("GT future", fontsize=9)

        ax_bar = axes[r, 2]
        sc_vals = [item_rec["s_score"], item_rec["ns_score"], item_rec["f_score"]]
        bar_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        ax_bar.barh(["success", "near_success", "failure"],
                    sc_vals, color=bar_colors, alpha=0.8)
        vmin, vmax = min(sc_vals), max(sc_vals)
        margin_v = max((vmax - vmin) * 0.15, 0.02)
        ax_bar.set_xlim(vmin - margin_v, vmax + margin_v)
        marker = "✓" if is_correct else "✗"
        ax_bar.set_title(
            f"{marker}  s={item_rec['s_score']:.3f}  "
            f"ns={item_rec['ns_score']:.3f}  f={item_rec['f_score']:.3f}",
            fontsize=7,
        )
        if r == 0:
            ax_bar.set_xlabel("DINO score", fontsize=8)
        ax_bar.tick_params(labelsize=7); ax_bar.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "strict_order_examples.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved strict_order_examples.png")


def save_tiered_rank_plots(
    all_item_scores: List[dict],
    metrics: dict,
    dataset: TieredEvalDataset,
    step: int,
    out_dir: Path,
) -> None:
    """Save all three per-eval PNG visualizations."""
    if not _HAS_MPL:
        logger.debug("matplotlib unavailable; skipping tiered rank plots.")
        return
    _plot_score_distribution(all_item_scores, dataset, step, out_dir)
    _plot_margin_histograms(all_item_scores, dataset, step, out_dir)
    _plot_strict_order_examples(all_item_scores, dataset, step, out_dir)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_tiered_rank_eval(
    model,
    dataset: TieredEvalDataset,
    device: torch.device,
    cfg,
    output_dir: Optional[str] = None,
    step: int = 0,
    task_name: str = "",
    task_suite: str = "",
) -> dict:
    """Run tiered ranking evaluation; return metric dict prefixed 'tiered/'.

    When output_dir is given, saves JSON / CSV / PNG under
    ``<output_dir>/rank_eval/`` according to cfg.save_rank_eval_*.
    """
    model_inner = model.module if hasattr(model, "module") else model
    model_inner.eval()

    all_item_scores: List[dict] = []
    item_id = 0

    for batch_idx, (cur, fut, candidates) in enumerate(dataset.batches):
        scores = model_inner.rank_action_candidates(
            cur.to(device),
            candidates.to(device),   # [B, K, H_act, 7]
            fut.to(device),
        ).float().cpu()               # [B, K]

        for b in range(cur.shape[0]):
            all_item_scores.append({
                "item_id":         item_id,
                "batch_idx":       batch_idx,
                "sample_in_batch": b,
                "scores":          scores[b].tolist(),
            })
            item_id += 1

    all_scores_t = torch.tensor([it["scores"] for it in all_item_scores])  # [N, K]
    metrics = _compute_tiered_metrics(all_scores_t, dataset)

    if output_dir:
        out_dir = Path(output_dir) / "rank_eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        if getattr(cfg, "save_rank_eval_json", True):
            save_tiered_rank_json(all_item_scores, metrics, dataset, step, out_dir)

        if getattr(cfg, "save_rank_eval_csv", True):
            save_tiered_rank_csv(all_item_scores, metrics, dataset, step, out_dir)

        if getattr(cfg, "save_rank_eval_plots", True):
            save_tiered_rank_plots(all_item_scores, metrics, dataset, step, out_dir)

        # Always append JSONL history (enables trajectory analysis in analyze_worldmodel_eval.py)
        append_tiered_rank_jsonl(all_item_scores, metrics, dataset, step, out_dir,
                                 task_suite=task_suite)

    return {f"tiered/{k}": v for k, v in metrics.items()}
