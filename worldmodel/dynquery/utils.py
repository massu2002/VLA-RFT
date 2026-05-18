"""Utilities for DynQueryWorldModel evaluation and training."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def motion_center_of_mass(
    current_image: torch.Tensor,   # [B, 3, H, W] float [0, 1]
    future_image: torch.Tensor,    # [B, 3, H, W] float [0, 1]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (cy, cx) in [0, 1] for each batch item."""
    B, C, H, W = current_image.shape
    diff = (future_image.float() - current_image.float()).abs().mean(dim=1)  # [B, H, W]
    total = diff.sum(dim=[1, 2]).clamp(min=1e-6)  # [B]

    device = diff.device
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)

    cy = (diff * ys[None, :, None]).sum(dim=[1, 2]) / total
    cx = (diff * xs[None, None, :]).sum(dim=[1, 2]) / total
    return cy, cx


def get_lpips_fn(device: str = "cpu", net: str = "alex"):
    """Return a cached LPIPS callable that accepts uint8 numpy [H, W, 3]."""
    global _LPIPS_FN
    if _LPIPS_FN is not None:
        return _LPIPS_FN

    try:
        import lpips as _lpips_lib
        _loss_fn = _lpips_lib.LPIPS(net=net, verbose=False).to(device).eval()
    except ImportError:
        logger.warning("lpips not installed; LPIPS metrics will be NaN.")
        _LPIPS_FN = lambda pred, gt: float("nan")  # noqa: E731
        return _LPIPS_FN

    def _lpips(pred_np: np.ndarray, gt_np: np.ndarray) -> float:
        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(arr).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)
            return t.to(device) * 2 - 1

        with torch.no_grad():
            score = _loss_fn(_to_tensor(pred_np), _to_tensor(gt_np))
        return float(score.item())

    _LPIPS_FN = _lpips
    return _LPIPS_FN


_LPIPS_FN = None


def aggregate_phase1_metrics(
    per_window_rows: List[Dict],
    output_dir: str,
    condition_name: str,
) -> Dict:
    """Aggregate per-window metrics into aggregate_metrics.json, csv files."""
    import csv

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    negative_types = set()
    for row in per_window_rows:
        for key in row.keys():
            if key.startswith("rft_reward_gap_"):
                negative_types.add(key[len("rft_reward_gap_"):])
    preferred_neg_order = ["same_phase", "temporal_shift", "action_noise", "mixed"]
    negative_types = sorted(
        negative_types,
        key=lambda x: preferred_neg_order.index(x) if x in preferred_neg_order else 99,
    )

    scalar_keys = [
        # Primary: reconstruction
        "horizon_avg_lpips",
        "horizon_avg_mae",
        "horizon_avg_mse",
        "rft_reward_proxy",
        "copy_current_horizon_avg_mse",
        "horizon_mse_over_copy",
        # Primary: per-step LPIPS / MAE
        "lpips_step1",
        "lpips_step4",
        "lpips_step8",
        "mae_step1",
        "mae_step4",
        "mae_step8",
        # Primary: GT-masked dynamic / static
        "dynamic_region_mse_gt",
        "dynamic_region_mae_gt",
        "dynamic_region_lpips_gt",
        "static_consistency_mse",
        "static_consistency_mae",
        # Primary: ROI
        "roi/gripper_mse",
        "roi/gripper_mae",
        "roi/gripper_lpips",
        "roi/goal_mse",
        "roi/goal_mae",
        "roi/goal_lpips",
        # Secondary: ranking
        "score_gap",
        "rft_reward_gap",
        # Secondary: dynamic mask localisation
        "dynamic_mask_iou_gt",
        "dynamic_mask_precision_gt",
        "dynamic_mask_recall_gt",
        # Debug: model internals
        "fuser_mask_entropy",
        "fuser_mask_overlap",
        "dynamic_mask_entropy",
        "dynamic_mask_overlap",
        "future_dynamic_query_norm",
    ]

    agg: Dict[str, List[float]] = {k: [] for k in scalar_keys}
    pairwise_wins_rft = 0
    num_windows = 0
    num_rft_ranking_windows = 0

    for row in per_window_rows:
        num_windows += 1
        for k in scalar_keys:
            v = row.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                agg[k].append(float(v))
        is_rft_ranked = row.get("rft_reward_gap") is not None
        if is_rft_ranked and not (isinstance(row.get("rft_reward_gap"), float) and np.isnan(row.get("rft_reward_gap"))):
            num_rft_ranking_windows += 1
            if row.get("pairwise_win_rft"):
                pairwise_wins_rft += 1

    def _mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    def _is_valid(v) -> bool:
        if v is None:
            return False
        try:
            return not bool(np.isnan(v))
        except TypeError:
            return True

    def _phase_name(row: Dict) -> str:
        phase = str(row.get("window_phase") or row.get("window_position") or "unknown")
        return "later" if phase == "late" else phase

    def _empty_bucket() -> Dict[str, List[float]]:
        keys = scalar_keys + ["pairwise_win_rft", "pairwise_win_score"]
        for neg in negative_types:
            keys.extend([
                f"rft_reward_gap_{neg}",
                f"score_gap_{neg}",
                f"pairwise_win_rft_{neg}",
                f"pairwise_win_score_{neg}",
            ])
        return {k: [] for k in keys}

    def _add_row(bucket: Dict[str, List[float]], row: Dict) -> None:
        for k in scalar_keys:
            v = row.get(k)
            if _is_valid(v):
                bucket[k].append(float(v))
        rft_gap = row.get("rft_reward_gap")
        if _is_valid(rft_gap):
            bucket["pairwise_win_rft"].append(float(row.get("pairwise_win_rft", 0)))
        shuf_score = row.get("score_shuffle")
        if _is_valid(shuf_score):
            bucket["pairwise_win_score"].append(float(row.get("pairwise_win_score", 0)))
        for neg in negative_types:
            rft_gap_neg = row.get(f"rft_reward_gap_{neg}")
            if _is_valid(rft_gap_neg):
                bucket[f"rft_reward_gap_{neg}"].append(float(rft_gap_neg))
                bucket[f"pairwise_win_rft_{neg}"].append(float(row.get(f"pairwise_win_rft_{neg}", 0)))
            score_gap_neg = row.get(f"score_gap_{neg}")
            if _is_valid(score_gap_neg):
                bucket[f"score_gap_{neg}"].append(float(score_gap_neg))
            score_shuffle_neg = row.get(f"score_shuffle_{neg}")
            if _is_valid(score_shuffle_neg):
                bucket[f"pairwise_win_score_{neg}"].append(float(row.get(f"pairwise_win_score_{neg}", 0)))

    def _bucket_to_metrics(bucket: Dict[str, List[float]]) -> Dict[str, float]:
        metrics = {k: _mean(bucket[k]) for k in scalar_keys}
        metrics["pairwise_acc_score"] = _mean(bucket["pairwise_win_score"])
        metrics["num_windows"] = len(bucket["pairwise_win_rft"])
        for neg in negative_types:
            metrics[f"rft_reward_gap_mean_{neg}"] = _mean(bucket[f"rft_reward_gap_{neg}"])
            metrics[f"score_gap_mean_{neg}"] = _mean(bucket[f"score_gap_{neg}"])
            metrics[f"pairwise_acc_rft_{neg}"] = _mean(bucket[f"pairwise_win_rft_{neg}"])
            metrics[f"pairwise_acc_score_{neg}"] = _mean(bucket[f"pairwise_win_score_{neg}"])
        _per_type_accs = [
            metrics[f"pairwise_acc_rft_{neg}"]
            for neg in negative_types
            if _is_valid(metrics.get(f"pairwise_acc_rft_{neg}"))
        ]
        metrics["pairwise_acc_rft"] = _mean(_per_type_accs) if _per_type_accs else _mean(bucket["pairwise_win_rft"])
        return metrics

    def _phase_sort_key(phase: str) -> int:
        order = {"early": 0, "middle": 1, "later": 2}
        return order.get(phase, 99)

    def _collect_metric(key: str) -> List[float]:
        vals = []
        for row in per_window_rows:
            v = row.get(key)
            if _is_valid(v):
                vals.append(float(v))
        return vals

    def _collect_bool_metric(key: str, valid_key: str) -> List[float]:
        vals = []
        for row in per_window_rows:
            if _is_valid(row.get(valid_key)):
                vals.append(float(row.get(key, 0)))
        return vals

    agg_metrics = {k: _mean(agg[k]) for k in scalar_keys}
    agg_metrics["pairwise_acc_rft"]     = (pairwise_wins_rft / num_rft_ranking_windows
                                            if num_rft_ranking_windows > 0 else float("nan"))
    agg_metrics["num_windows"]          = num_windows
    agg_metrics["num_ranking_windows"]  = num_rft_ranking_windows

    rft_gaps = agg.get("rft_reward_gap", [])
    agg_metrics["rft_reward_gap_min"] = float(np.min(rft_gaps)) if rft_gaps else float("nan")

    # Reward signal stability
    rft_proxies = agg.get("rft_reward_proxy", [])
    agg_metrics["rft_reward_proxy_std"]   = float(np.std(rft_proxies))  if rft_proxies else float("nan")
    agg_metrics["rft_reward_proxy_range"] = float(np.ptp(rft_proxies))  if rft_proxies else float("nan")
    agg_metrics["rft_reward_gap_std"]     = float(np.std(rft_gaps))     if rft_gaps    else float("nan")

    for neg in negative_types:
        rft_gap_key = f"rft_reward_gap_{neg}"
        score_gap_key = f"score_gap_{neg}"
        rft_gaps_neg = _collect_metric(rft_gap_key)
        score_gaps_neg = _collect_metric(score_gap_key)
        agg_metrics[f"rft_reward_gap_mean_{neg}"] = _mean(rft_gaps_neg)
        agg_metrics[f"rft_reward_gap_min_{neg}"] = float(np.min(rft_gaps_neg)) if rft_gaps_neg else float("nan")
        agg_metrics[f"score_gap_mean_{neg}"] = _mean(score_gaps_neg)
        agg_metrics[f"score_gap_min_{neg}"] = float(np.min(score_gaps_neg)) if score_gaps_neg else float("nan")
        agg_metrics[f"pairwise_acc_rft_{neg}"] = _mean(
            _collect_bool_metric(f"pairwise_win_rft_{neg}", rft_gap_key)
        )
        agg_metrics[f"pairwise_acc_score_{neg}"] = _mean(
            _collect_bool_metric(f"pairwise_win_score_{neg}", f"score_shuffle_{neg}")
        )

    # Override pairwise_acc_rft with the mean of per-type accuracies so the overall
    # metric reflects all negative types equally rather than the pooled-equal strategy.
    _per_type_accs = [
        agg_metrics[f"pairwise_acc_rft_{neg}"]
        for neg in negative_types
        if _is_valid(agg_metrics.get(f"pairwise_acc_rft_{neg}"))
    ]
    if _per_type_accs:
        agg_metrics["pairwise_acc_rft"] = _mean(_per_type_accs)

    by_phase: Dict[str, Dict[str, List[float]]] = {}
    for row in per_window_rows:
        phase = _phase_name(row)
        by_phase.setdefault(phase, _empty_bucket())
        _add_row(by_phase[phase], row)

    phase_rows = []
    for phase, bucket in sorted(by_phase.items(), key=lambda kv: _phase_sort_key(kv[0])):
        prow = {"window_phase": phase}
        prow.update(_bucket_to_metrics(bucket))
        phase_rows.append(prow)

    if phase_rows:
        with open(os.path.join(output_dir, "metrics_by_phase.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(phase_rows[0].keys()))
            w.writeheader()
            w.writerows(phase_rows)

    by_task_phase: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for row in per_window_rows:
        t = row.get("task_name", "unknown")
        phase = _phase_name(row)
        by_task_phase.setdefault(t, {})
        by_task_phase[t].setdefault(phase, _empty_bucket())
        _add_row(by_task_phase[t][phase], row)

    task_phase_rows = []
    for t, phase_buckets in sorted(by_task_phase.items()):
        for phase, bucket in sorted(phase_buckets.items(), key=lambda kv: _phase_sort_key(kv[0])):
            row = {"task_name": t, "window_phase": phase}
            row.update(_bucket_to_metrics(bucket))
            task_phase_rows.append(row)

    if task_phase_rows:
        with open(os.path.join(output_dir, "metrics_by_task_by_phase.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(task_phase_rows[0].keys()))
            w.writeheader()
            w.writerows(task_phase_rows)

    task_rows = []
    for t, phase_buckets in sorted(by_task_phase.items()):
        phase_metrics = [_bucket_to_metrics(bucket) for _, bucket in sorted(
            phase_buckets.items(), key=lambda kv: _phase_sort_key(kv[0])
        )]
        trow = {"task_name": t}
        for k in scalar_keys:
            trow[k] = _mean([m[k] for m in phase_metrics if _is_valid(m.get(k))])
        trow["pairwise_acc_rft"] = _mean([
            m["pairwise_acc_rft"] for m in phase_metrics if _is_valid(m.get("pairwise_acc_rft"))
        ])
        trow["pairwise_acc_score"] = _mean([
            m["pairwise_acc_score"] for m in phase_metrics if _is_valid(m.get("pairwise_acc_score"))
        ])
        for neg in negative_types:
            for metric_name in (
                f"rft_reward_gap_mean_{neg}",
                f"score_gap_mean_{neg}",
                f"pairwise_acc_rft_{neg}",
                f"pairwise_acc_score_{neg}",
            ):
                trow[metric_name] = _mean([
                    m[metric_name] for m in phase_metrics if _is_valid(m.get(metric_name))
                ])
        trow["num_windows"] = int(sum(m.get("num_windows", 0) for m in phase_metrics))
        trow["num_phases"] = len(phase_metrics)
        task_rows.append(trow)

    if task_rows:
        with open(os.path.join(output_dir, "metrics_by_task.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
            w.writeheader()
            w.writerows(task_rows)

    ranking_keys = [
        "task_name", "task_index", "window_id", "window_phase",
        "episode_length", "episode_file",
        "frame_indices", "action_indices",
        "score_correct", "score_shuffle", "score_gap", "pairwise_win_score",
        "rft_reward_proxy", "rft_reward_gap", "pairwise_win_rft",
        "horizon_avg_lpips", "horizon_avg_mae", "horizon_avg_mse",
        "horizon_mse_over_copy",
    ]
    for neg in negative_types:
        ranking_keys.extend([
            f"score_shuffle_{neg}",
            f"score_gap_{neg}",
            f"pairwise_win_score_{neg}",
            f"rft_reward_gap_{neg}",
            f"pairwise_win_rft_{neg}",
        ])
    ranking_rows = []
    for row in per_window_rows:
        v = row.get("rft_reward_gap")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ranking_rows.append({k: row.get(k) for k in ranking_keys})
    if ranking_rows:
        with open(os.path.join(output_dir, "ranking_by_window.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ranking_keys)
            w.writeheader()
            w.writerows(ranking_rows)

    with open(os.path.join(output_dir, "config_used.json"), "w") as f:
        json.dump({"condition": condition_name}, f, indent=2)

    logger.info(
        "[phase1_metrics] %s: pairwise_acc_rft=%.4f  lpips=%.4f  dyn_lpips_gt=%.4f  "
        "rft_proxy_std=%.4f  gap_std=%.4f  n=%d",
        condition_name,
        agg_metrics.get("pairwise_acc_rft", float("nan")),
        agg_metrics.get("horizon_avg_lpips", float("nan")),
        agg_metrics.get("dynamic_region_lpips_gt", float("nan")),
        agg_metrics.get("rft_reward_proxy_std", float("nan")),
        agg_metrics.get("rft_reward_gap_std", float("nan")),
        num_windows,
    )

    return agg_metrics
