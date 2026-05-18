#!/usr/bin/env python3
"""Generate a Japanese Markdown report for Phase 1 world-model evaluation.

The evaluator writes one directory per condition:

    <eval_root>/<condition>/aggregate_metrics.json
    <eval_root>/<condition>/metrics_by_task.csv
    <eval_root>/<condition>/ranking_by_task.csv
    <eval_root>/<condition>/ranking_by_window.csv
    <eval_root>/<condition>/eval_protocol_config.json

This script reads those files and writes:

    <eval_root>/comparison_ja.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


AGG_METRICS = [
    ("full_mse", "Full-image MSE", "lower"),
    ("full_lpips", "Full-image LPIPS", "lower"),
    ("gripper_mse", "Gripper ROI MSE", "lower"),
    ("gripper_lpips", "Gripper ROI LPIPS", "lower"),
    ("goal_mse", "Goal ROI MSE", "lower"),
    ("goal_lpips", "Goal ROI LPIPS", "lower"),
    ("dynamic_mse", "Dynamic region MSE", "lower"),
    ("dynamic_lpips", "Dynamic region LPIPS", "lower"),
    ("static_consistency_mse", "Static consistency MSE", "lower"),
    ("pairwise_acc", "Pairwise Acc", "higher"),
    ("lpips_gap", "LPIPS Gap", "higher"),
    ("correct_lpips", "Correct LPIPS", "lower"),
    ("shuffled_lpips", "Shuffled LPIPS", "lower"),
    ("reverse_windows", "Reverse windows", "lower"),
    ("num_windows", "評価window数", "higher"),
]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def fmt(value: Any, nd: int = 5) -> str:
    val = as_float(value)
    if val is None:
        return "N/A"
    return f"{val:.{nd}f}"


def pct(value: Any) -> str:
    val = as_float(value)
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def metric(metrics: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def condition_sort_key(name: str) -> tuple:
    order = [
        "pixel_baseline",
        "v1_residual",
        "v1_roi",
        "v3_residual",
        "v3_roi",
    ]
    for i, prefix in enumerate(order):
        if name.startswith(prefix):
            return (i, name)
    return (len(order), name)


def load_condition(directory: Path) -> Optional[Dict[str, Any]]:
    agg = load_json(directory / "aggregate_metrics.json")
    if not agg:
        return None
    metrics = agg.get("metrics", agg)
    protocol = load_json(directory / "eval_protocol_config.json")
    if not protocol:
        protocol = load_json(directory / "config_used.json")
    return {
        "name": directory.name,
        "dir": directory,
        "aggregate": agg,
        "metrics": metrics,
        "protocol": protocol.get("eval_protocol", protocol),
        "metrics_by_task": read_csv(directory / "metrics_by_task.csv"),
        "ranking_by_task": read_csv(directory / "ranking_by_task.csv"),
        "metrics_by_phase": read_csv(directory / "metrics_by_phase.csv"),
        "metrics_by_task_phase": read_csv(directory / "metrics_by_task_phase.csv"),
        "ranking_by_window": read_csv(directory / "ranking_by_window.csv"),
    }


def best_condition(
    conditions: Iterable[Dict[str, Any]], key: str, better: str
) -> Optional[Dict[str, Any]]:
    scored = []
    for cond in conditions:
        val = as_float(metric(cond["metrics"], key))
        if val is not None:
            scored.append((val, cond))
    if not scored:
        return None
    return min(scored, key=lambda x: x[0])[1] if better == "lower" else max(scored, key=lambda x: x[0])[1]


def table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def summarize_protocol(conditions: List[Dict[str, Any]]) -> List[str]:
    lines = []
    protocols = [c["protocol"] for c in conditions if c.get("protocol")]
    if not protocols:
        return ["- `eval_protocol_config.json` が見つからない条件があります。評価条件の完全な監査はできません。"]
    first = protocols[0]
    keys = [
        "task_suite",
        "selected_task_indices",
        "num_eval_windows",
        "num_ranking_windows",
        "eval_horizon",
        "segment_length",
        "action_start_offset",
        "negative_type",
        "same_task_shuffle",
        "lpips_input_range",
        "use_terminal_frame_only",
        "ranking_gap_definition",
        "pairwise_unit",
        "roi_crop_size",
        "gripper_roi_method",
        "phase0_compatible",
        "window_position_mode",
        "num_eval_episodes_per_task",
    ]
    for key in keys:
        values = {str(p.get(key, "N/A")) for p in protocols}
        if len(values) == 1:
            lines.append(f"- `{key}`: `{next(iter(values))}`")
        else:
            lines.append(f"- `{key}`: 条件間で不一致 `{', '.join(sorted(values))}`")
    ckpts = []
    for cond in conditions:
        proto = cond.get("protocol", {})
        ckpt = proto.get("checkpoint_path") or proto.get("model_dir") or str(cond["dir"])
        ckpts.append(f"- `{cond['name']}`: `{ckpt}`")
    lines.append("")
    lines.append("評価対象checkpoint:")
    lines.extend(ckpts)
    return lines


def select_candidate(conditions: List[Dict[str, Any]]) -> str:
    candidates = []
    for cond in conditions:
        m = cond["metrics"]
        pairwise = as_float(metric(m, "pairwise_acc"))
        gap = as_float(metric(m, "lpips_gap"))
        gripper = as_float(metric(m, "gripper_mse"))
        dynamic = as_float(metric(m, "dynamic_mse"))
        if pairwise is None:
            continue
        score = pairwise
        if gap is not None:
            score += 10.0 * gap
        if gripper is not None:
            score -= 0.2 * gripper
        if dynamic is not None:
            score -= 0.2 * dynamic
        candidates.append((score, cond))
    if not candidates:
        return "評価可能なranking指標がないため、候補は判断できません。"
    best = max(candidates, key=lambda x: x[0])[1]
    return (
        f"現時点の採用候補は `{best['name']}` です。"
        "選定理由は、RFTに効きやすい `pairwise_acc` と `LPIPS Gap` を主に見つつ、"
        "Gripper/Dynamic領域の誤差が極端に悪化していない条件を優先したためです。"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Phase 1 Residual World Model 単体評価レポート")
    parser.add_argument("--data-root", type=str, default="")
    args = parser.parse_args()

    eval_root = args.eval_root
    conditions = [
        c for c in (load_condition(p) for p in sorted(eval_root.iterdir(), key=lambda p: condition_sort_key(p.name)))
        if c is not None
    ]
    if not conditions:
        raise SystemExit(f"No aggregate_metrics.json found under {eval_root}")

    output = args.output or (eval_root / "comparison_ja.md")

    rows = []
    for cond in conditions:
        m = cond["metrics"]
        rows.append(
            [
                f"`{cond['name']}`",
                fmt(metric(m, "full_mse")),
                fmt(metric(m, "gripper_mse")),
                fmt(metric(m, "dynamic_mse")),
                pct(metric(m, "pairwise_acc")),
                fmt(metric(m, "lpips_gap")),
                fmt(metric(m, "reverse_windows", "num_reverse_windows"), 0),
                fmt(metric(m, "num_windows"), 0),
            ]
        )

    best_lines = []
    for key, label, better in AGG_METRICS:
        cond = best_condition(conditions, key, better)
        if cond is None:
            continue
        best_lines.append(f"- {label}: `{cond['name']}` ({fmt(metric(cond['metrics'], key))})")

    per_metric_rows = []
    for key, label, better in AGG_METRICS:
        row = [label, "低いほど良い" if better == "lower" else "高いほど良い"]
        best = best_condition(conditions, key, better)
        for cond in conditions:
            value = fmt(metric(cond["metrics"], key))
            if best is not None and cond["name"] == best["name"]:
                value = f"**{value}**"
            row.append(value)
        per_metric_rows.append(row)

    task_rows = []
    for cond in conditions:
        for row in cond["ranking_by_task"]:
            task_rows.append(
                [
                    f"`{cond['name']}`",
                    row.get("task_id") or row.get("task_index") or "",
                    row.get("task_name") or row.get("task_name_or_description") or "",
                    pct(row.get("pairwise_acc")),
                    fmt(row.get("lpips_gap") or row.get("lpips_gap_mean")),
                    fmt(row.get("num_windows"), 0),
                ]
            )

    phase_rows = []
    for cond in conditions:
        for row in cond["metrics_by_phase"]:
            phase_rows.append(
                [
                    f"`{cond['name']}`",
                    row.get("window_phase", ""),
                    fmt(row.get("full_mse")),
                    fmt(row.get("gripper_mse")),
                    fmt(row.get("dynamic_mse")),
                    pct(row.get("pairwise_acc")),
                    fmt(row.get("lpips_gap_mean") or row.get("lpips_gap")),
                    fmt(row.get("reverse_windows"), 0),
                    fmt(row.get("num_windows"), 0),
                ]
            )

    lines = [
        f"# {args.title}",
        "",
        f"- 評価ディレクトリ: `{eval_root}`",
        f"- データセットroot: `{args.data_root or 'eval_protocol_config.jsonを参照'}`",
        f"- 評価条件数: `{len(conditions)}`",
        *(
            [
                "- 注意: この実行環境では `/localdata/modified_libero_rlds` ではないデータセットrootを明示指定しています。実機で `/localdata` が見えている場合は、同じ評価モードを `/localdata/modified_libero_rlds` で再実行できます。"
            ]
            if args.data_root and not str(args.data_root).startswith("/localdata")
            else []
        ),
        "",
        "## 評価プロトコル",
        "",
        *summarize_protocol(conditions),
        "",
        "## 全体サマリ",
        "",
        table(
            ["条件", "Full MSE", "Gripper MSE", "Dynamic MSE", "Pairwise Acc", "LPIPS Gap", "Reverse", "Windows"],
            rows,
        ),
        "",
        "## 指標別の最良条件",
        "",
        *best_lines,
        "",
        "## 詳細比較表",
        "",
        table(["指標", "方向"] + [f"`{c['name']}`" for c in conditions], per_metric_rows),
        "",
        "## 解釈",
        "",
        "- `Full-image MSE` は画像全体の再構成性能です。背景のコピーが強いモデルほど良く見えやすいため、Phase 1の主目的である action-sensitive local dynamics の判定では補助指標として扱います。",
        "- `Gripper ROI` と `Dynamic region` は、手先・物体・接触に近い局所変化を見ます。Residual WM の狙いに近い指標です。",
        "- `Pairwise Acc` は GT action の予測が shuffle action より LPIPS 的に近い window の割合です。RFT reward に接続する候補を選ぶうえでは、この値と `LPIPS Gap = shuffled_lpips - correct_lpips` を重視します。",
        "- `LPIPS Gap` が正なら GT action が有利、負なら shuffle action が有利です。Pairwise Acc が高くても gap が極端に小さい場合、ranking signal は弱い可能性があります。",
        "",
        "## 採用候補",
        "",
        select_candidate(conditions),
        "",
    ]
    if task_rows:
        lines.extend(
            [
                "## Task別Ranking",
                "",
                table(["条件", "task_id", "task", "Pairwise Acc", "LPIPS Gap", "Windows"], task_rows),
                "",
            ]
        )

    if phase_rows:
        lines.extend(
            [
                "## Episode位置別サマリ",
                "",
                "各episodeから `early / middle / late` のwindowを取った評価では、ここを見ることで「序盤は見分けられるが終盤は弱い」などの時間位置依存を確認できます。",
                "",
                table(
                    ["条件", "位置", "Full MSE", "Gripper MSE", "Dynamic MSE", "Pairwise Acc", "LPIPS Gap", "Reverse", "Windows"],
                    phase_rows,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 次に確認すべき点",
            "",
            "- `pixel_baseline` の Pairwise Acc が Phase 0 の AR-Pixel baseline と大きくずれる場合は、評価プロトコル差分を再監査してください。",
            "- WM単体指標が改善してもRFT成功率が改善しない場合は、Residual WM自体ではなく reward/value 設計がボトルネックの可能性があります。",
            "- RFTに進む候補は、Full MSE最小の条件ではなく、Pairwise Acc、LPIPS Gap、Gripper/Dynamic誤差のバランスで選ぶのが本筋です。",
            "",
        ]
    )

    output.write_text("\n".join(lines))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
