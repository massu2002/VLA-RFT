"""Fixed ranking benchmark with temporal-neighbor near-success and same-task hard negatives.

Architecture
-----------
  RankingBenchmark   — serialisable fixed evaluation set (saved once to disk as .pt)
  PoolWindow         — lightweight per-window record (no images)
  BenchmarkCandidate — one action candidate with tier / source metadata
  BenchmarkItem      — one evaluation item (anchor frame + K candidates)

Near-success candidates (in preference order per slot)
  1. temporal_neighbor : adjacent window (ws±1, ±2) in the same episode
  2. small_noise       : GT + N(0, near_success_noise_std)   [fallback]

Failure candidates (cycled in order)
  1. same_task_hard      : different episode, |progress_diff| < 0.2
  2. same_task_mismatch  : different episode, |progress_diff| > 0.4
  3. large_noise         : GT + N(0, fail_noise_std)
  4. shuffle             : temporally shuffled GT actions

Per-source-type metrics (appended to standard tiered metrics)
  acc_gt_{source_type}     P(score_success > score_from_source_type)
  mean_score_{source_type} mean DINO score for each source type

Outputs (under <output_dir>/bench/)
  ranking_benchmark.pt             saved once (load on subsequent runs)
  ranking_benchmark_meta.json      metadata sidecar
  benchmark_metrics.json           overwritten each eval step
  benchmark_candidates.csv         1 row per (item, candidate)
  benchmark_summary.csv            metric summary
  source_type_score_distribution.png
  hard_negative_examples.png
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .tiered_rank_eval import (
    TIER_FAILURE,
    TIER_NEAR_SUCCESS,
    TIER_SUCCESS,
    TIER_ORDINAL,
    _compute_tiered_metrics,
)

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
# Source type constants
# ---------------------------------------------------------------------------

SRC_GT                 = "gt"
SRC_TEMPORAL_NEIGHBOR  = "temporal_neighbor"
SRC_SMALL_NOISE        = "small_noise"
SRC_SAME_TASK_HARD     = "same_task_hard"
SRC_SAME_TASK_MISMATCH = "same_task_mismatch"
SRC_LARGE_NOISE        = "large_noise"
SRC_SHUFFLE            = "shuffle"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PoolWindow:
    """Lightweight per-window record (no images) for pool-level lookups."""
    task_name: str
    episode_id: int
    window_start: int
    episode_length: int
    actions: torch.Tensor   # [H_act, 7] float32

    @property
    def progress(self) -> float:
        """Normalised position within the episode [0, 1]."""
        return self.window_start / max(1, self.episode_length - 1)


@dataclass
class BenchmarkCandidate:
    """One action candidate for a BenchmarkItem."""
    tier: str                    # TIER_SUCCESS | TIER_NEAR_SUCCESS | TIER_FAILURE
    source_type: str             # SRC_* constant
    action: torch.Tensor         # [H_act, 7] float32
    source_episode_id: int  = -1
    source_window_start: int = -1
    noise_std: float        = 0.0


@dataclass
class BenchmarkItem:
    """One evaluation item: anchor frame + K ordered candidates."""
    item_id: int
    episode_id: int
    window_start: int
    cur_pixels: torch.Tensor    # [H, W, 3] uint8
    fut_pixels: torch.Tensor    # [H, W, 3] uint8
    candidates: List[BenchmarkCandidate]

    def actions_stacked(self) -> torch.Tensor:
        """Return [K, H_act, 7] float32."""
        return torch.stack([c.action for c in self.candidates])


@dataclass
class RankingBenchmark:
    """Complete fixed ranking benchmark, serialisable to .pt."""
    items: List[BenchmarkItem]
    task_name: str
    dataset_name: str
    segment_length: int
    seed: int
    n_near_success: int
    n_failure: int
    near_noise_std: float
    fail_noise_std: float
    ns_modes: List[str]
    f_modes: List[str]
    created_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )

    @property
    def K(self) -> int:
        return len(self.items[0].candidates) if self.items else 0

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def tiers(self) -> List[str]:
        return [c.tier for c in self.items[0].candidates] if self.items else []

    @property
    def source_types(self) -> List[str]:
        return [c.source_type for c in self.items[0].candidates] if self.items else []


# ---------------------------------------------------------------------------
# _TieredMock — duck-type adapter so _compute_tiered_metrics can be reused
# ---------------------------------------------------------------------------

class _TieredMock:
    """Minimal duck-type of TieredEvalDataset for _compute_tiered_metrics."""

    def __init__(self, tiers: List[str], n_ns: int, n_f: int) -> None:
        self.tiers           = tiers
        self.n_near_success  = n_ns
        self.n_failure       = n_f
        self.near_noise_std  = 0.0
        self.fail_noise_std  = 0.0

    @property
    def K(self) -> int:
        return len(self.tiers)

    @property
    def success_idx(self) -> int:
        return next(i for i, t in enumerate(self.tiers) if t == TIER_SUCCESS)

    @property
    def near_success_indices(self) -> List[int]:
        return [i for i, t in enumerate(self.tiers) if t == TIER_NEAR_SUCCESS]

    @property
    def failure_indices(self) -> List[int]:
        return [i for i, t in enumerate(self.tiers) if t == TIER_FAILURE]


# ---------------------------------------------------------------------------
# Internal anchor frame (only used during pool collection)
# ---------------------------------------------------------------------------

@dataclass
class _AnchorFrame:
    episode_id: int
    window_start: int
    episode_length: int
    cur_pixels: torch.Tensor   # [H, W, 3] uint8
    fut_pixels: torch.Tensor   # [H, W, 3] uint8
    actions: torch.Tensor      # [H_act, 7] float32


# ---------------------------------------------------------------------------
# Episode pool collection
# ---------------------------------------------------------------------------

def _collect_episode_pool(
    dataset_name: str,
    data_dir: str,
    segment_length: int,
    task_name: str,
    max_pool_episodes: int,
    n_anchors_per_episode: int,
    seed: int,
    image_key: str = "image",
) -> Tuple[Dict[int, List[PoolWindow]], List[_AnchorFrame]]:
    """Stream RLDS dataset and collect pool windows + anchor frames.

    Windows are visited in episode order (shuffle_windows=False).
    Episode boundaries are detected when window_start resets to 0.

    Returns
    -------
    pool    : {episode_id: [PoolWindow, ...]}  — lightweight, no images
    anchors : List[_AnchorFrame]               — randomly selected windows with images
    """
    from ..datasets.libero.data import RldsIterableDataset

    rng = random.Random(seed)

    inner = RldsIterableDataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        raw_chunk_length=segment_length,
        seed=seed,
        shuffle_episodes=False,
        shuffle_windows=False,
        window_stride=1,
        image_key=image_key,
        include_episode_metadata=True,
    )

    pool:        Dict[int, List[PoolWindow]] = {}
    all_anchors: List[_AnchorFrame]          = []

    # Per-episode buffer: (window_start, episode_length, cur_pix, fut_pix, actions)
    cur_ep_buf: List[Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    global_ep_id  = 0
    episode_count = 0

    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.clone()
        return torch.as_tensor(np.array(x))

    def _finalize_episode(ep_id: int, buf: list) -> None:
        if not buf:
            return
        pool_wins: List[PoolWindow] = []
        for ws, ep_len, _, _, acts in buf:
            pool_wins.append(PoolWindow(
                task_name=task_name,
                episode_id=ep_id,
                window_start=ws,
                episode_length=ep_len,
                actions=acts.clone(),
            ))
        pool[ep_id] = pool_wins

        interior = [i for i in range(len(buf)) if 1 <= i < len(buf) - 1]
        if not interior:
            interior = list(range(len(buf)))
        n_sel = min(n_anchors_per_episode, len(interior))
        chosen = rng.sample(interior, n_sel)
        for idx in chosen:
            ws, ep_len, cur_pix, fut_pix, acts = buf[idx]
            all_anchors.append(_AnchorFrame(
                episode_id=ep_id,
                window_start=ws,
                episode_length=ep_len,
                cur_pixels=cur_pix,
                fut_pixels=fut_pix,
                actions=acts,
            ))

    for sample in inner:
        ws      = int(sample["window_start"])
        ep_len  = int(sample["episode_length"])
        pixels  = _to_tensor(sample["pixels"])          # [T, H, W, C] uint8
        actions = _to_tensor(sample["actions"]).float() # [H_act, 7]

        cur_pix = pixels[0].clone()
        fut_pix = pixels[-1].clone()

        if ws == 0 and cur_ep_buf:
            _finalize_episode(global_ep_id, cur_ep_buf)
            cur_ep_buf  = []
            global_ep_id  += 1
            episode_count += 1
            if episode_count >= max_pool_episodes:
                break

        cur_ep_buf.append((ws, ep_len, cur_pix, fut_pix, actions))

    if cur_ep_buf and episode_count < max_pool_episodes:
        _finalize_episode(global_ep_id, cur_ep_buf)

    n_pool_wins = sum(len(v) for v in pool.values())
    logger.info(
        "Pool collected: %d episodes  %d pool windows  %d anchors",
        len(pool), n_pool_wins, len(all_anchors),
    )
    return pool, all_anchors


# ---------------------------------------------------------------------------
# Benchmark item construction
# ---------------------------------------------------------------------------

def _build_benchmark_items(
    pool: Dict[int, List[PoolWindow]],
    anchors: List[_AnchorFrame],
    n_ns: int,
    n_f: int,
    ns_modes: List[str],
    f_modes: List[str],
    num_items: int,
    near_noise_std: float,
    fail_noise_std: float,
    seed: int,
) -> List[BenchmarkItem]:
    """Build BenchmarkItem list from pool + anchor frames."""
    rng = random.Random(seed + 1)

    pool_ws_index: Dict[int, Dict[int, PoolWindow]] = {
        ep_id: {pw.window_start: pw for pw in wins}
        for ep_id, wins in pool.items()
    }
    all_pool_windows = [pw for wins in pool.values() for pw in wins]

    shuffled = list(anchors)
    rng.shuffle(shuffled)
    if num_items > 0:
        shuffled = shuffled[:num_items]

    items: List[BenchmarkItem] = []

    for item_id, anchor in enumerate(shuffled):
        ep_id        = anchor.episode_id
        ws           = anchor.window_start
        ep_len       = anchor.episode_length
        anchor_prog  = ws / max(1, ep_len - 1)
        gt_action    = anchor.actions   # [H_act, 7]

        # ── near_success candidates ────────────────────────────────────────
        ns_candidates: List[BenchmarkCandidate] = []
        for i_ns in range(n_ns):
            mode = ns_modes[i_ns % len(ns_modes)] if ns_modes else "small_noise"
            cand: Optional[BenchmarkCandidate] = None

            if mode == "temporal_neighbor":
                for delta in [-1, 1, -2, 2]:
                    nb_pw = pool_ws_index.get(ep_id, {}).get(ws + delta)
                    if nb_pw is not None:
                        cand = BenchmarkCandidate(
                            tier=TIER_NEAR_SUCCESS,
                            source_type=SRC_TEMPORAL_NEIGHBOR,
                            action=nb_pw.actions.clone(),
                            source_episode_id=ep_id,
                            source_window_start=ws + delta,
                        )
                        break
                if cand is None:
                    noise = torch.randn_like(gt_action) * near_noise_std
                    cand = BenchmarkCandidate(
                        tier=TIER_NEAR_SUCCESS,
                        source_type=SRC_SMALL_NOISE,
                        action=gt_action + noise,
                        source_episode_id=ep_id,
                        source_window_start=ws,
                        noise_std=near_noise_std,
                    )
            else:  # small_noise
                noise = torch.randn_like(gt_action) * near_noise_std
                cand = BenchmarkCandidate(
                    tier=TIER_NEAR_SUCCESS,
                    source_type=SRC_SMALL_NOISE,
                    action=gt_action + noise,
                    source_episode_id=ep_id,
                    source_window_start=ws,
                    noise_std=near_noise_std,
                )
            ns_candidates.append(cand)

        # ── failure candidates ─────────────────────────────────────────────
        same_hard_pool = [
            pw for pw in all_pool_windows
            if pw.episode_id != ep_id and abs(pw.progress - anchor_prog) < 0.2
        ]
        same_mismatch_pool = [
            pw for pw in all_pool_windows
            if pw.episode_id != ep_id and abs(pw.progress - anchor_prog) > 0.4
        ]

        f_candidates: List[BenchmarkCandidate] = []
        for i_f in range(n_f):
            mode = f_modes[i_f % len(f_modes)] if f_modes else "large_noise"
            cand = None

            if mode == "same_task_hard" and same_hard_pool:
                chosen = rng.choice(same_hard_pool)
                cand = BenchmarkCandidate(
                    tier=TIER_FAILURE,
                    source_type=SRC_SAME_TASK_HARD,
                    action=chosen.actions.clone(),
                    source_episode_id=chosen.episode_id,
                    source_window_start=chosen.window_start,
                )
            elif mode == "same_task_mismatch" and same_mismatch_pool:
                chosen = rng.choice(same_mismatch_pool)
                cand = BenchmarkCandidate(
                    tier=TIER_FAILURE,
                    source_type=SRC_SAME_TASK_MISMATCH,
                    action=chosen.actions.clone(),
                    source_episode_id=chosen.episode_id,
                    source_window_start=chosen.window_start,
                )
            elif mode == "large_noise":
                noise = torch.randn_like(gt_action) * fail_noise_std
                cand = BenchmarkCandidate(
                    tier=TIER_FAILURE,
                    source_type=SRC_LARGE_NOISE,
                    action=gt_action + noise,
                    noise_std=fail_noise_std,
                )
            elif mode == "shuffle":
                perm = torch.randperm(gt_action.shape[0])
                cand = BenchmarkCandidate(
                    tier=TIER_FAILURE,
                    source_type=SRC_SHUFFLE,
                    action=gt_action[perm],
                )

            if cand is None:  # fallback when pool is too small
                noise = torch.randn_like(gt_action) * fail_noise_std
                cand = BenchmarkCandidate(
                    tier=TIER_FAILURE,
                    source_type=SRC_LARGE_NOISE,
                    action=gt_action + noise,
                    noise_std=fail_noise_std,
                )
            f_candidates.append(cand)

        gt_cand = BenchmarkCandidate(
            tier=TIER_SUCCESS,
            source_type=SRC_GT,
            action=gt_action.clone(),
            source_episode_id=ep_id,
            source_window_start=ws,
        )

        items.append(BenchmarkItem(
            item_id=item_id,
            episode_id=ep_id,
            window_start=ws,
            cur_pixels=anchor.cur_pixels,
            fut_pixels=anchor.fut_pixels,
            candidates=[gt_cand] + ns_candidates + f_candidates,
        ))

    return items


# ---------------------------------------------------------------------------
# High-level build / save / load
# ---------------------------------------------------------------------------

def build_ranking_benchmark(
    dataset_name: str,
    data_dir: str,
    segment_length: int,
    task_name: str,
    cfg,
    seed: int = 1337,
) -> RankingBenchmark:
    """Build a RankingBenchmark by streaming episodes from the dataset."""
    max_pool        = getattr(cfg, "num_benchmark_pool_episodes",       20)
    n_anchors       = getattr(cfg, "num_benchmark_anchors_per_episode",  5)
    n_ns            = getattr(cfg, "num_near_success_candidates",         2)
    n_f             = getattr(cfg, "num_failure_candidates",              3)
    near_noise_std  = getattr(cfg, "near_success_noise_std",           0.05)
    fail_noise_std  = getattr(cfg, "failure_noise_std",                0.30)
    num_items       = getattr(cfg, "num_rank_eval_items",                 0)
    ns_modes_str    = getattr(cfg, "near_success_modes_bench",
                              "temporal_neighbor,small_noise")
    f_modes_str     = getattr(cfg, "failure_modes_bench",
                              "same_task_hard,same_task_mismatch,large_noise,shuffle")

    ns_modes = [m.strip() for m in ns_modes_str.split(",") if m.strip()]
    f_modes  = [m.strip() for m in f_modes_str.split(",")  if m.strip()]

    logger.info(
        "Building RankingBenchmark  pool_eps=%d  anchors/ep=%d  "
        "n_ns=%d  n_f=%d  ns=%s  f=%s",
        max_pool, n_anchors, n_ns, n_f, ns_modes, f_modes,
    )

    pool, anchors = _collect_episode_pool(
        dataset_name=dataset_name,
        data_dir=data_dir,
        segment_length=segment_length,
        task_name=task_name,
        max_pool_episodes=max_pool,
        n_anchors_per_episode=n_anchors,
        seed=seed,
    )

    if not anchors:
        raise RuntimeError(
            "No anchor frames collected. "
            "Check dataset_name, data_dir, and num_benchmark_pool_episodes."
        )

    items = _build_benchmark_items(
        pool=pool,
        anchors=anchors,
        n_ns=n_ns,
        n_f=n_f,
        ns_modes=ns_modes,
        f_modes=f_modes,
        num_items=num_items,
        near_noise_std=near_noise_std,
        fail_noise_std=fail_noise_std,
        seed=seed,
    )

    benchmark = RankingBenchmark(
        items=items,
        task_name=task_name,
        dataset_name=dataset_name,
        segment_length=segment_length,
        seed=seed,
        n_near_success=n_ns,
        n_failure=n_f,
        near_noise_std=near_noise_std,
        fail_noise_std=fail_noise_std,
        ns_modes=ns_modes,
        f_modes=f_modes,
    )
    logger.info(
        "RankingBenchmark ready: %d items  K=%d  (ns=%d  f=%d)",
        len(items), benchmark.K, n_ns, n_f,
    )
    return benchmark


def save_ranking_benchmark(benchmark: RankingBenchmark, out_dir: Path) -> Path:
    """Save benchmark to .pt + metadata JSON. Returns the .pt path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = out_dir / "ranking_benchmark.pt"
    torch.save(benchmark, pt_path)

    meta = {
        "task_name":      benchmark.task_name,
        "dataset_name":   benchmark.dataset_name,
        "segment_length": benchmark.segment_length,
        "seed":           benchmark.seed,
        "n_near_success": benchmark.n_near_success,
        "n_failure":      benchmark.n_failure,
        "K":              benchmark.K,
        "n_items":        benchmark.n_items,
        "tiers":          benchmark.tiers,
        "source_types":   benchmark.source_types,
        "ns_modes":       benchmark.ns_modes,
        "f_modes":        benchmark.f_modes,
        "created_at":     benchmark.created_at,
    }
    (out_dir / "ranking_benchmark_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("RankingBenchmark saved → %s", pt_path)
    return pt_path


def load_ranking_benchmark(pt_path: str) -> RankingBenchmark:
    # weights_only=False required: RankingBenchmark contains custom dataclasses
    benchmark = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    logger.info(
        "Loaded RankingBenchmark: %d items  K=%d  from %s",
        benchmark.n_items, benchmark.K, pt_path,
    )
    return benchmark


def load_or_build_benchmark(
    out_dir: str,
    dataset_name: str,
    data_dir: str,
    segment_length: int,
    task_name: str,
    cfg,
    seed: int = 1337,
    regenerate: bool = False,
) -> RankingBenchmark:
    """Load benchmark from disk if present; build and save it otherwise."""
    bench_dir = Path(out_dir) / "bench"
    pt_path   = bench_dir / "ranking_benchmark.pt"

    custom_path = getattr(cfg, "rank_eval_dataset_path", "")
    if custom_path:
        pt_path = Path(custom_path)

    if pt_path.exists() and not regenerate:
        return load_ranking_benchmark(pt_path)

    benchmark = build_ranking_benchmark(
        dataset_name=dataset_name,
        data_dir=data_dir,
        segment_length=segment_length,
        task_name=task_name,
        cfg=cfg,
        seed=seed,
    )
    save_ranking_benchmark(benchmark, bench_dir)
    return benchmark


# ---------------------------------------------------------------------------
# Per-source-type metrics
# ---------------------------------------------------------------------------

def _compute_source_metrics(
    all_item_scores: List[dict],
    tiers: List[str],
    source_types_per_cand: List[str],
) -> dict:
    """Accuracy and mean DINO score for every non-GT source type."""
    s_idx = next(i for i, t in enumerate(tiers) if t == TIER_SUCCESS)

    src_scores:  Dict[str, List[float]] = {}
    src_correct: Dict[str, List[bool]]  = {}

    for it in all_item_scores:
        scores  = it["scores"]
        s_score = scores[s_idx]
        for k, (tier, src) in enumerate(zip(tiers, source_types_per_cand)):
            if tier == TIER_SUCCESS:
                continue
            src_scores.setdefault(src, []).append(scores[k])
            src_correct.setdefault(src, []).append(bool(s_score > scores[k]))

    metrics: dict = {}
    for src, sc_list in src_scores.items():
        metrics[f"mean_score_{src}"] = float(np.mean(sc_list))
        if src_correct.get(src):
            metrics[f"acc_gt_{src}"] = float(np.mean(src_correct[src]))
    return metrics


# ---------------------------------------------------------------------------
# Artifact saving helpers
# ---------------------------------------------------------------------------

def _save_benchmark_metrics_json(
    all_item_scores: List[dict],
    metrics: dict,
    benchmark: RankingBenchmark,
    step: int,
    out_dir: Path,
) -> None:
    doc = {
        "step":     step,
        "metrics":  metrics,
        "saved_at": datetime.datetime.utcnow().isoformat(),
    }
    (out_dir / "benchmark_metrics.json").write_text(json.dumps(doc, indent=2))
    logger.info("Saved benchmark_metrics.json (step=%d)", step)


def _save_benchmark_csv(
    all_item_scores: List[dict],
    metrics: dict,
    benchmark: RankingBenchmark,
    step: int,
    out_dir: Path,
) -> None:
    tiers = benchmark.tiers
    srcs  = benchmark.source_types

    cand_path = out_dir / "benchmark_candidates.csv"
    with cand_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "item_id", "episode_id", "window_start",
                    "candidate_idx", "tier", "source_type", "score"])
        for it in all_item_scores:
            for k, score in enumerate(it["scores"]):
                w.writerow([
                    step,
                    it["item_id"], it["episode_id"], it["window_start"],
                    k, tiers[k], srcs[k], f"{score:.6f}",
                ])
    logger.info("Saved benchmark_candidates.csv (%d rows)",
                len(all_item_scores) * benchmark.K)

    summary_path = out_dir / "benchmark_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "metric", "value"])
        for k, v in sorted(metrics.items()):
            w.writerow([step, k, f"{v:.6f}"])
    logger.info("Saved benchmark_summary.csv")


def _to_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else t
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# PNG: source_type_score_distribution
# ---------------------------------------------------------------------------

_SRC_COLORS = {
    SRC_GT:                 "#2ecc71",
    SRC_TEMPORAL_NEIGHBOR:  "#3498db",
    SRC_SMALL_NOISE:        "#f39c12",
    SRC_SAME_TASK_HARD:     "#e67e22",
    SRC_SAME_TASK_MISMATCH: "#9b59b6",
    SRC_LARGE_NOISE:        "#e74c3c",
    SRC_SHUFFLE:            "#c0392b",
}

_SRC_ORDER = [
    SRC_GT, SRC_TEMPORAL_NEIGHBOR, SRC_SMALL_NOISE,
    SRC_SAME_TASK_HARD, SRC_SAME_TASK_MISMATCH,
    SRC_LARGE_NOISE, SRC_SHUFFLE,
]


def _save_source_type_distribution(
    all_item_scores: List[dict],
    metrics: dict,
    benchmark: RankingBenchmark,
    step: int,
    out_dir: Path,
) -> None:
    if not _HAS_MPL:
        return

    srcs = benchmark.source_types
    scores_by_src: Dict[str, List[float]] = {}
    for it in all_item_scores:
        for k, score in enumerate(it["scores"]):
            scores_by_src.setdefault(srcs[k], []).append(score)

    present = [s for s in _SRC_ORDER if s in scores_by_src]
    present += [s for s in scores_by_src if s not in present]

    data         = [scores_by_src[s] for s in present]
    plot_colors  = [_SRC_COLORS.get(s, "#7f8c8d") for s in present]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Source Type Score Distribution — step {step:,}", fontsize=11)

    ax = axes[0]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(present, rotation=30, ha="right", fontsize=8)
    ax.set_title("Score distribution by source type")
    ax.set_ylabel("DINO cosine score")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    means = [float(np.mean(d)) for d in data]
    stds  = [float(np.std(d))  for d in data]
    xs    = list(range(len(present)))
    ax.bar(xs, means, color=plot_colors, alpha=0.7, edgecolor="black", linewidth=0.8)
    ax.errorbar(xs, means, yerr=stds, fmt="none", color="black", capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(present, rotation=30, ha="right", fontsize=8)
    ax.set_title("Mean score ± std per source type")
    ax.set_ylabel("DINO cosine score")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "source_type_score_distribution.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved source_type_score_distribution.png")


# ---------------------------------------------------------------------------
# PNG: hard_negative_examples
# ---------------------------------------------------------------------------

def _save_hard_negative_examples(
    all_item_scores: List[dict],
    metrics: dict,
    benchmark: RankingBenchmark,
    step: int,
    out_dir: Path,
    n_examples: int = 4,
) -> None:
    """Show items where hard negatives scored closest to GT."""
    if not _HAS_MPL:
        return

    tiers = benchmark.tiers
    srcs  = benchmark.source_types
    s_idx = next(i for i, t in enumerate(tiers) if t == TIER_SUCCESS)

    _HARD_SRC = {SRC_SAME_TASK_HARD, SRC_SAME_TASK_MISMATCH, SRC_TEMPORAL_NEIGHBOR}
    hard_ks = [k for k, s in enumerate(srcs) if s in _HARD_SRC]

    if not hard_ks:
        return

    records = []
    for it in all_item_scores:
        scores  = it["scores"]
        s_score = scores[s_idx]
        hard_sc = [scores[k] for k in hard_ks if k < len(scores)]
        if not hard_sc:
            continue
        max_hard = max(hard_sc)
        records.append({**it, "s_score": s_score, "max_hard": max_hard,
                        "margin": s_score - max_hard})

    records.sort(key=lambda x: x["margin"])
    shown = records[:n_examples]
    if not shown:
        return

    items_by_id = {it.item_id: it for it in benchmark.items}
    n_rows = len(shown)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3.0 * n_rows), squeeze=False)
    fig.suptitle(
        f"Hard Negative Examples — step {step:,}\n"
        "Items where hard negatives scored closest to GT  (sorted by margin ↑)",
        fontsize=10,
    )

    for r, rec in enumerate(shown):
        item = items_by_id.get(rec["item_id"])
        if item is None:
            continue

        axes[r, 0].imshow(_to_rgb(item.cur_pixels))
        axes[r, 0].axis("off")
        if r == 0:
            axes[r, 0].set_title("current", fontsize=9)

        axes[r, 1].imshow(_to_rgb(item.fut_pixels))
        axes[r, 1].axis("off")
        if r == 0:
            axes[r, 1].set_title("GT future", fontsize=9)

        ax_bar = axes[r, 2]
        bar_labels, bar_vals, bar_clrs = [], [], []
        for k, (tier, src) in enumerate(zip(tiers, srcs)):
            if tier == TIER_SUCCESS or src in _HARD_SRC:
                bar_labels.append(f"{src}\n({tier[:3]})")
                bar_vals.append(rec["scores"][k])
                bar_clrs.append(
                    "#2ecc71" if tier == TIER_SUCCESS
                    else "#e67e22" if src == SRC_SAME_TASK_HARD
                    else "#9b59b6" if src == SRC_SAME_TASK_MISMATCH
                    else "#3498db"
                )

        ax_bar.barh(bar_labels, bar_vals, color=bar_clrs, alpha=0.8)
        vmin_v = min(bar_vals) if bar_vals else 0
        vmax_v = max(bar_vals) if bar_vals else 1
        pad = max((vmax_v - vmin_v) * 0.15, 0.02)
        ax_bar.set_xlim(vmin_v - pad, vmax_v + pad)
        ax_bar.set_title(
            f"margin={rec['margin']:.3f}  gt={rec['s_score']:.3f}  "
            f"hard={rec['max_hard']:.3f}",
            fontsize=7,
        )
        if r == 0:
            ax_bar.set_xlabel("DINO score", fontsize=8)
        ax_bar.tick_params(labelsize=7)
        ax_bar.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "hard_negative_examples.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved hard_negative_examples.png")


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_ranking_benchmark_eval(
    model,
    benchmark: RankingBenchmark,
    device: torch.device,
    cfg,
    output_dir: Optional[str] = None,
    step: int = 0,
) -> dict:
    """Evaluate model on the fixed ranking benchmark.

    Returns metric dict prefixed 'bench/'.
    Saves JSON / CSV / PNG under ``<output_dir>/bench/`` when output_dir is given.
    """
    model_inner = model.module if hasattr(model, "module") else model
    model_inner.eval()

    BATCH_SIZE = 8
    items = benchmark.items
    tiers = benchmark.tiers
    srcs  = benchmark.source_types

    all_item_scores: List[dict] = []

    for start in range(0, len(items), BATCH_SIZE):
        batch_items = items[start : start + BATCH_SIZE]

        cur_batch  = torch.stack([it.cur_pixels         for it in batch_items])
        fut_batch  = torch.stack([it.fut_pixels         for it in batch_items])
        cand_batch = torch.stack([it.actions_stacked()  for it in batch_items])

        scores = model_inner.rank_action_candidates(
            cur_batch.to(device),
            cand_batch.to(device),
            fut_batch.to(device),
        ).float().cpu()  # [B, K]

        for b, item in enumerate(batch_items):
            all_item_scores.append({
                "item_id":      item.item_id,
                "episode_id":   item.episode_id,
                "window_start": item.window_start,
                "scores":       scores[b].tolist(),
            })

    all_scores_t = torch.tensor([it["scores"] for it in all_item_scores])  # [N, K]

    mock    = _TieredMock(tiers=tiers, n_ns=benchmark.n_near_success, n_f=benchmark.n_failure)
    metrics = _compute_tiered_metrics(all_scores_t, mock)

    source_metrics = _compute_source_metrics(all_item_scores, tiers, srcs)
    metrics.update(source_metrics)

    if output_dir:
        out_dir = Path(output_dir) / "bench"
        out_dir.mkdir(parents=True, exist_ok=True)

        if getattr(cfg, "save_rank_eval_json", True):
            _save_benchmark_metrics_json(
                all_item_scores, metrics, benchmark, step, out_dir
            )
        if getattr(cfg, "save_rank_eval_csv", True):
            _save_benchmark_csv(
                all_item_scores, metrics, benchmark, step, out_dir
            )
        if getattr(cfg, "save_rank_eval_plots", True):
            _save_source_type_distribution(
                all_item_scores, metrics, benchmark, step, out_dir
            )
            _save_hard_negative_examples(
                all_item_scores, metrics, benchmark, step, out_dir
            )

    return {f"bench/{k}": v for k, v in metrics.items()}
