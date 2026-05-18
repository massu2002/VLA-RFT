#!/usr/bin/env python3
"""Priority-A diagnostic summary for Phase 1 residual WM evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("metrics", data)
    except Exception:
        return {}


def f(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        y = float(x)
        if math.isnan(y):
            return None
        return y
    except Exception:
        return None


def fmt(x: Any, digits: int = 5, pct: bool = False) -> str:
    y = f(x)
    if y is None:
        return "N/A"
    return f"{y * 100:.1f}%" if pct else f"{y:.{digits}f}"


def infer_loss_profile(name: str) -> str:
    if name.startswith("phase0_ar_pixel"):
        return "phase0_ar_pixel_native"
    if "pixel_baseline" in name:
        return "image"
    if "_roi_" in name:
        return "residual+image+dynamic+gripper+static+write"
    if "residual" in name:
        return "residual+image+write" if name.startswith("v3") else "residual+image"
    return ""


def collect(root: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(root.glob("*/aggregate_metrics.json")):
        dirname = p.parent.name
        if dirname.endswith("_test") or dirname.endswith("_smoke") or "smoke" in dirname:
            continue
        metrics = load_json(p)
        name = metrics.get("model_name") or dirname
        protocol = load_json(p.parent / "eval_protocol_config.json")
        if name == "phase0_ar_pixel" and not (metrics.get("metric_source") or protocol.get("metric_source")):
            continue
        model_family = metrics.get("model_family") or protocol.get("model_family") or ("phase0" if name == "phase0_ar_pixel" else "phase1")
        model_generation = metrics.get("model_generation") or protocol.get("model_generation") or ("baseline" if "pixel_baseline" in name else "v3" if name.startswith("v3") else "v1" if name.startswith("v1") else "")
        target_mode = metrics.get("target_mode") or protocol.get("target_mode") or ("ar_pixel" if name == "phase0_ar_pixel" else "")
        metric_source = metrics.get("metric_source") or protocol.get("metric_source") or ""
        window_manifest_hash = metrics.get("window_manifest_hash") or protocol.get("window_manifest_hash") or ""
        strict = (
            bool(window_manifest_hash)
            and metric_source != "converted_phase0_result"
            and protocol.get("negative_type") == "same_task_other_window"
            and protocol.get("lpips_input_range") == "[-1,1]"
            and protocol.get("pairwise_unit") == "window"
            and f(protocol.get("eval_horizon", metrics.get("eval_horizon", 7))) is not None
        )
        comparable_group_id = window_manifest_hash[:12] if strict else ""
        action_summary = load_json(p.parent / "action_ablation" / "action_ablation_summary.json")
        full = f(metrics.get("full_mse"))
        copy = f(metrics.get("copy_current_full_mse", metrics.get("copy_current_mse")))
        ratio = metrics.get("full_mse_over_copy_current_mse")
        if f(ratio) is None and full is not None and copy is not None and copy > 0:
            ratio = full / copy
        warning_copy = full is not None and copy is not None and full >= copy
        warning_write = f(metrics.get("write_mask_mean")) is not None and f(metrics.get("write_mask_mean")) > 0.75
        warning_pairwise = f(metrics.get("pairwise_acc")) is not None and f(metrics.get("pairwise_acc")) < 0.5
        warning_action = bool(action_summary.get("warning_action_insensitive", False))
        rows.append({
            "model_name": name,
            "model_family": model_family,
            "model_generation": model_generation,
            "target_mode": target_mode,
            "metric_source": metric_source,
            "window_manifest_hash": window_manifest_hash,
            "is_strictly_comparable": strict,
            "comparable_group_id": comparable_group_id,
            "loss_profile": infer_loss_profile(name),
            "full_mse": metrics.get("full_mse"),
            "gripper_mse": metrics.get("gripper_mse"),
            "dynamic_mse": metrics.get("dynamic_mse"),
            "copy_current_full_mse": metrics.get("copy_current_full_mse", metrics.get("copy_current_mse")),
            "full_mse_over_copy_current_mse": ratio,
            "pairwise_acc": metrics.get("pairwise_acc"),
            "lpips_gap_mean": metrics.get("lpips_gap", metrics.get("lpips_gap_mean")),
            "lpips_gap_min": metrics.get("lpips_gap_min"),
            "reverse_windows": metrics.get("reverse_windows"),
            "write_mask_mean": metrics.get("write_mask_mean"),
            "write_mask_max": metrics.get("write_mask_max"),
            "residual_abs_mean": metrics.get("residual_abs_mean"),
            "residual_abs_max": metrics.get("residual_abs_max"),
            "num_windows": metrics.get("num_windows"),
            "phase0_compatible": protocol.get("phase0_compatible"),
            "correct_best_rate": action_summary.get("correct_best_rate"),
            "action_correct_best_rate": action_summary.get("correct_best_rate"),
            "correct_vs_shuffle_pred_diff_mse": action_summary.get("correct_vs_shuffle_pred_diff_mse"),
            "correct_vs_zero_pred_diff_mse": action_summary.get("correct_vs_zero_pred_diff_mse"),
            "correct_vs_random_pred_diff_mse": action_summary.get("correct_vs_random_pred_diff_mse"),
            "warning_protocol_mismatch": protocol and not bool(protocol.get("phase0_compatible")),
            "warning_copy_current_collapse": warning_copy,
            "warning_write_everywhere": warning_write,
            "warning_action_insensitive": warning_action,
            "warning_pairwise_below_random": warning_pairwise,
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_root", nargs="?", default="results/phase1/residual_worldmodel")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    root = Path(args.eval_root)
    out = Path(args.out_dir) if args.out_dir else root / "priority_a_summary"
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    write_csv(out / "summary.csv", rows)

    lines = ["# Phase 1 Priority-A Diagnostics Summary", ""]
    lines.append("## 比較ルール")
    lines.append("- `converted_phase0_result` は過去Phase0結果の参照値であり、Phase1モデルとの厳密比較には使いません。")
    lines.append("- 主比較baselineは `direct_eval_on_phase1_manifest` の `phase0_ar_pixel_direct_eval` です。")
    lines.append("- `window_manifest_hash` が一致し、horizon / negative / LPIPS / pairwise単位が一致するモデルだけを strict comparison として扱います。")
    lines.append("")

    strict_rows = [r for r in rows if r.get("is_strictly_comparable")]
    hist_rows = [r for r in rows if r.get("metric_source") == "converted_phase0_result"]

    lines.append("## 表0: Strict comparison group")
    lines.append("| model | source | group | manifest_hash | num_windows | full_mse | gripper_mse | dynamic_mse | pairwise | gap |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    if strict_rows:
        for r in strict_rows:
            lines.append(
                f"| `{r['model_name']}` | {r.get('metric_source','')} | {r.get('comparable_group_id','')} | "
                f"`{str(r.get('window_manifest_hash',''))[:12]}` | {r.get('num_windows','')} | {fmt(r['full_mse'])} | "
                f"{fmt(r['gripper_mse'])} | {fmt(r['dynamic_mse'])} | {fmt(r['pairwise_acc'], pct=True)} | {fmt(r['lpips_gap_mean'])} |"
            )
    else:
        lines.append("| N/A |  |  |  |  |  |  |  |  |  |")
    lines.append("")

    lines.append("## 表0b: Historical reference")
    lines.append("| model | source | num_windows | full_mse | gripper_mse | pairwise | gap | note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    if hist_rows:
        for r in hist_rows:
            lines.append(
                f"| `{r['model_name']}` | {r.get('metric_source','')} | {r.get('num_windows','')} | "
                f"{fmt(r['full_mse'])} | {fmt(r['gripper_mse'])} | {fmt(r['pairwise_acc'], pct=True)} | "
                f"{fmt(r['lpips_gap_mean'])} | strict比較から除外 |"
            )
    else:
        lines.append("| N/A |  |  |  |  |  |  |  |")
    lines.append("")

    lines.append("## 表1: モデル比較")
    lines.append("| model | family | generation | target | source | strict | full_mse | copy_full_mse | full/copy | gripper_mse | dynamic_mse | pairwise | gap | write_mean | residual_mean |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r['model_name']}` | {r['model_family']} | {r['model_generation']} | {r['target_mode']} | "
            f"{r.get('metric_source','')} | {r.get('is_strictly_comparable')} | "
            f"{fmt(r['full_mse'])} | {fmt(r['copy_current_full_mse'])} | {fmt(r['full_mse_over_copy_current_mse'], 3)} | "
            f"{fmt(r['gripper_mse'])} | {fmt(r['dynamic_mse'])} | {fmt(r['pairwise_acc'], pct=True)} | "
            f"{fmt(r['lpips_gap_mean'])} | {fmt(r['write_mask_mean'])} | {fmt(r['residual_abs_mean'])} |"
        )
    lines.append("")
    lines.append("## 表2: Phase0 AR-Pixel vs Phase1 models")
    lines.append("| model | is_phase0_ar_pixel | metric_source | manifest_hash | full_mse | gripper_mse | dynamic_mse | pairwise | gap | num_windows | phase0_compatible |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r['model_name']}` | {str(str(r['model_name']).startswith('phase0_ar_pixel'))} | "
            f"{r.get('metric_source','')} | `{str(r.get('window_manifest_hash',''))[:12]}` | {fmt(r['full_mse'])} | "
            f"{fmt(r['gripper_mse'])} | {fmt(r['dynamic_mse'])} | {fmt(r['pairwise_acc'], pct=True)} | "
            f"{fmt(r['lpips_gap_mean'])} | {r.get('num_windows','')} | {r.get('phase0_compatible','')} |"
        )
    lines.append("")
    lines.append("## 表3: Action ablation")
    lines.append("| model | correct_best_rate | correct-vs-shuffle MSE | correct-vs-zero MSE | correct-vs-random MSE | warning |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r['model_name']}` | {fmt(r['correct_best_rate'], pct=True)} | {fmt(r['correct_vs_shuffle_pred_diff_mse'])} | "
            f"{fmt(r['correct_vs_zero_pred_diff_mse'])} | {fmt(r['correct_vs_random_pred_diff_mse'])} | {r['warning_action_insensitive']} |"
        )
    lines.append("")
    lines.append("## 表4: Warnings")
    for r in rows:
        ws = [k for k in ("warning_protocol_mismatch", "warning_copy_current_collapse", "warning_write_everywhere", "warning_action_insensitive", "warning_pairwise_below_random") if r.get(k)]
        if ws:
            lines.append(f"- `{r['model_name']}`: " + ", ".join(ws))
    if not any(any(r.get(k) for k in ("warning_protocol_mismatch", "warning_copy_current_collapse", "warning_write_everywhere", "warning_action_insensitive", "warning_pairwise_below_random")) for r in rows):
        lines.append("- No warnings.")
    lines.append("")
    lines.append("## 表5: Protocol mismatch warnings")
    any_proto = False
    for r in rows:
        warns = []
        if r.get("metric_source") == "converted_phase0_result":
            warns.append("converted historical result")
        if not r.get("is_strictly_comparable"):
            warns.append("not in strict comparison group")
        if not r.get("window_manifest_hash"):
            warns.append("missing window_manifest_hash")
        if warns:
            any_proto = True
            lines.append(f"- `{r['model_name']}`: " + ", ".join(warns))
    if not any_proto:
        lines.append("- No protocol mismatches.")
    lines.append("")
    lines.append("## Debug visual paths")
    lines.append("- Per-model: `<eval_root>/<model>/debug_visuals/task_<id>/window_<id>/`")
    lines.append("- Action ablation: `<eval_root>/<model>/action_ablation/`")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'summary.md'}")
    print(f"Wrote {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
