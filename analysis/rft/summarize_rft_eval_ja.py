#!/usr/bin/env python3
"""Summarize Phase 1 Residual-WM RFT LIBERO eval results in Japanese."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def fmt(value: Any, digits: int = 3, pct: bool = False) -> str:
    x = as_float(value)
    if x is None:
        return "N/A"
    if pct:
        return f"{x * 100:.1f}%"
    return f"{x:.{digits}f}"


def short_exp_name(rft_exp_name: str) -> str:
    name = rft_exp_name
    if name.startswith("phase1_"):
        name = name[len("phase1_") :]
    if name.endswith("_rft"):
        name = name[: -len("_rft")]
    return name


def infer_family(exp_name: str) -> tuple[str, str]:
    short = short_exp_name(exp_name)
    if short == "pixel_baseline":
        return "baseline", "pixel"
    if short.startswith("v1_roi"):
        return "v1", "pixel_residual_roi_dynamic"
    if short.startswith("v1_residual"):
        return "v1", "pixel_residual"
    if short.startswith("v3_roi"):
        return "v3", "pixel_residual_roi_dynamic"
    if short.startswith("v3_residual"):
        return "v3", "pixel_residual"
    return "", ""


def success_from_summary(data: dict[str, Any]) -> float | None:
    for key in ("success_rate", "success", "success_mean", "rft_success_mean"):
        value = as_float(data.get(key))
        if value is not None:
            return value
    successes = as_float(data.get("num_successes"))
    episodes = as_float(data.get("num_episodes"))
    if successes is not None and episodes:
        return successes / episodes
    return None


def task_success_rows(exp_dir: Path) -> list[dict[str, str]]:
    rows = read_csv(exp_dir / "success_by_task.csv")
    if rows:
        return rows
    rows = read_csv(exp_dir / "eval" / "success_by_task.csv")
    return rows


def load_wm_metrics(wm_eval_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not wm_eval_root.exists():
        return out
    for path in wm_eval_root.glob("*/aggregate_metrics.json"):
        data = load_json(path)
        out[path.parent.name] = data.get("metrics", data)
    return out


def collect_eval_rows(rft_eval_root: Path, wm_eval_root: Path) -> list[dict[str, Any]]:
    wm_by_exp = load_wm_metrics(wm_eval_root)
    rows: list[dict[str, Any]] = []
    for exp_dir in sorted(p for p in rft_eval_root.iterdir() if p.is_dir() and p.name != "logs"):
        cfg = load_json(exp_dir / "eval_config_used.json")
        summary = load_json(exp_dir / "success_summary.json")
        short = short_exp_name(exp_dir.name)
        gen, mode = infer_family(exp_dir.name)
        wm = wm_by_exp.get(short, {})
        success = success_from_summary(summary)
        policy_ckpt = cfg.get("policy_ckpt", "")
        step = ""
        for part in Path(policy_ckpt).parts:
            if part.startswith("global_step_"):
                step = part.replace("global_step_", "")
        rows.append(
            {
                "exp_name": exp_dir.name,
                "wm_exp_name": short,
                "model_generation": gen,
                "target_mode": mode,
                "policy_ckpt": policy_ckpt,
                "rft_step": step,
                "task_suite": cfg.get("task_suite", ""),
                "num_trials": cfg.get("num_trials", ""),
                "seed": cfg.get("seed", ""),
                "success_rate": success if success is not None else "",
                "num_successes": summary.get("num_successes", ""),
                "num_episodes": summary.get("num_episodes", ""),
                "num_tasks": summary.get("num_tasks_evaluated", summary.get("num_tasks", "")),
                "wm_full_mse": wm.get("full_mse", ""),
                "wm_gripper_mse": wm.get("gripper_mse", ""),
                "wm_dynamic_mse": wm.get("dynamic_mse", ""),
                "wm_pairwise_acc": wm.get("pairwise_acc", ""),
                "wm_lpips_gap_mean": wm.get("lpips_gap_mean", wm.get("lpips_gap", "")),
                "status": "done" if success is not None else "missing_success_summary",
            }
        )
    return rows


def collect_task_rows(rft_eval_root: Path) -> list[dict[str, Any]]:
    by_exp: dict[str, list[dict[str, str]]] = {}
    for exp_dir in sorted(p for p in rft_eval_root.iterdir() if p.is_dir() and p.name != "logs"):
        rows = task_success_rows(exp_dir)
        if rows:
            by_exp[exp_dir.name] = rows
    task_ids = sorted(
        {
            int(row.get("task_id", -1))
            for rows in by_exp.values()
            for row in rows
            if str(row.get("task_id", "")).isdigit()
        }
    )
    out: list[dict[str, Any]] = []
    for task_id in task_ids:
        row_out: dict[str, Any] = {"task_id": task_id, "task_description": ""}
        scores: dict[str, float] = {}
        for exp, rows in by_exp.items():
            for row in rows:
                if int(row.get("task_id", -1)) != task_id:
                    continue
                row_out["task_description"] = row_out["task_description"] or row.get("task_description", "")
                score = as_float(row.get("success_rate"))
                if score is not None:
                    row_out[short_exp_name(exp)] = score
                    scores[short_exp_name(exp)] = score
        if scores:
            best_name, best_score = max(scores.items(), key=lambda item: item[1])
            row_out["best_model"] = best_name
            row_out["best_success"] = best_score
        out.append(row_out)
    return out


def load_manifest_rows(rft_eval_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(rft_eval_root.glob("rft_eval_manifest_node*_of_*.tsv")):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f, delimiter="\t"))
    return rows


def make_markdown(
    rft_eval_root: Path,
    wm_eval_root: Path,
    rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
) -> str:
    done = [r for r in rows if r["status"] == "done"]
    missing = [r for r in rows if r["status"] != "done"]
    best = max(done, key=lambda r: as_float(r["success_rate"]) or -1.0) if done else None
    baseline = next((r for r in done if r["wm_exp_name"] == "pixel_baseline"), None)
    baseline_success = as_float(baseline["success_rate"]) if baseline else None

    cfg = load_json((rft_eval_root / rows[0]["exp_name"] / "eval_config_used.json")) if rows else {}
    lines: list[str] = []
    lines.append("# Phase 1 Residual WM RFT 評価結果")
    lines.append("")
    lines.append("## 1. 評価設定")
    lines.append("")
    lines.append("- 評価対象: Phase 1 の PixelResidualWM を reward signal として RFT 事後学習した policy actor")
    lines.append(f"- 評価タスク: LIBERO `{cfg.get('task_suite', 'spatial')}`")
    lines.append("- 評価方法: `run_libero_eval.py` による rollout success rate")
    lines.append(f"- Base VLA: `{cfg.get('base_vla_path', 'N/A')}`")
    lines.append("- Policy checkpoint: `PixelResidualWM-RFT/<task>/<exp>/global_step_*/actor`")
    lines.append(f"- Rollout/動画保存先: `{cfg.get('rollout_root', 'rollouts/libero/rft_phase1/<exp>/<task_suite>')}`")
    lines.append("- Rollout保存形式: Phase0 RFTと同様に `task_XX__.../{success,failure}/*.mp4`, `episode_results.json`, `task_results.json`, `overall_results.json` を保存")
    lines.append("- Step選択: デフォルトでは各実験の最新 `global_step`。今回の検出では主に `global_step_400/actor`")
    lines.append(f"- 試行回数: `NUM_TRIALS={cfg.get('num_trials', 'N/A')}` per task")
    lines.append(f"- Seed: `{cfg.get('seed', 'N/A')}`")
    lines.append("- 3台PC分割: `job_id % NUM_NODES == NODE_INDEX` で評価対象を分配")
    lines.append(f"- RFT評価root: `{rft_eval_root}`")
    lines.append(f"- WM単体評価root: `{wm_eval_root}`")
    lines.append("")
    lines.append("この評価は world model 単体のMSE/LPIPSではなく、RFT後 policy を実際にLIBERO環境で動かし、タスク成功率を測ります。したがって Phase 1 の主目的である「Residual WMが post-training success を改善するか」を直接確認する評価です。")
    lines.append("")
    lines.append("## 2. 実験条件一覧")
    lines.append("")
    lines.append("| 実験 | generation | target_mode | actor step | 評価状態 |")
    lines.append("|---|---|---|---:|---|")
    for row in rows:
        lines.append(
            f"| `{row['exp_name']}` | {row['model_generation']} | {row['target_mode']} | "
            f"{row['rft_step'] or 'N/A'} | {row['status']} |"
        )
    lines.append("")
    lines.append("## 3. RFT後 Success 比較")
    lines.append("")
    if not done:
        lines.append("現時点では `success_summary.json` が見つかっていないため、成功率はまだ集計できません。")
        lines.append("3台PCで実評価を完了したあと、同じ集計コマンドを再実行するとこの表が埋まります。")
    lines.append("")
    lines.append("| 実験 | Success | baseline差分 | WM pairwise_acc | WM lpips_gap | WM full_mse | WM gripper_mse | WM dynamic_mse |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        success = as_float(row["success_rate"])
        delta = success - baseline_success if success is not None and baseline_success is not None else None
        lines.append(
            f"| `{row['wm_exp_name']}` | {fmt(success, pct=True)} | {fmt(delta, pct=True)} | "
            f"{fmt(row['wm_pairwise_acc'], pct=True)} | {fmt(row['wm_lpips_gap_mean'], 5)} | "
            f"{fmt(row['wm_full_mse'], 5)} | {fmt(row['wm_gripper_mse'], 5)} | {fmt(row['wm_dynamic_mse'], 5)} |"
        )
    lines.append("")
    lines.append("## 4. Task別 Success")
    lines.append("")
    if task_rows:
        exp_cols = sorted({k for row in task_rows for k in row if k not in {"task_id", "task_description", "best_model", "best_success"}})
        lines.append("| task_id | best_model | best_success | " + " | ".join(exp_cols) + " |")
        lines.append("|---:|---|---:|" + "|".join(["---:" for _ in exp_cols]) + "|")
        for row in task_rows:
            vals = " | ".join(fmt(row.get(col), pct=True) for col in exp_cols)
            lines.append(f"| {row['task_id']} | {row.get('best_model', '')} | {fmt(row.get('best_success'), pct=True)} | {vals} |")
    else:
        lines.append("まだ `success_by_task.csv` が見つかっていないため、task別の比較は未集計です。")
    lines.append("")
    lines.append("## 5. WM単体評価との対応")
    lines.append("")
    if done:
        lines.append("- RFT success が `pixel_baseline` より高いResidual条件があれば、action-sensitive residual dynamics が事後学習に有効だった候補です。")
        lines.append("- WM pairwise_acc や LPIPS gap が高いのに success が伸びない場合、世界モデルの予測改善だけでは reward として不十分で、progress-aware value などの Phase 2 が必要です。")
        if best:
            lines.append(f"- 現時点のSuccess最大は `{best['wm_exp_name']}` です。")
    else:
        lines.append("- WM単体評価では `v3_roi_d4_g2_s05_w02` が pairwise_acc 最大、`v3_residual_w02` がLPIPS系に強い候補でした。")
        lines.append("- RFT後successが未集計なので、現時点では「WM単体の改善が実際の成功率改善に対応したか」はまだ判断できません。")
    lines.append("")
    lines.append("## 6. 3PC分割状況")
    lines.append("")
    if manifest_rows:
        lines.append("| job_id | node | 実験 | actor | status |")
        lines.append("|---:|---:|---|---|---|")
        for row in manifest_rows:
            node = f"{row.get('node_index', '')}/{row.get('num_nodes', '')}"
            lines.append(
                f"| {row.get('job_id', '')} | {node} | `{row.get('exp_name', '')}` | "
                f"`{row.get('policy_ckpt', '')}` | {row.get('status', '')} |"
            )
    else:
        lines.append("manifest TSV が見つかりませんでした。")
    lines.append("")
    lines.append("## 7. 次に見るべき点")
    lines.append("")
    if missing:
        lines.append("- まず3台PCで実評価を完了し、`success_summary.json` と `success_by_task.csv` を生成してください。")
    lines.append("- `v3_roi_d4_g2_s05_w02` が baseline を上回るなら、local action-conditioned residual + dynamic-heavy loss が有効な候補です。")
    lines.append("- `v3_residual_w02` が上回るなら、ROI/Dynamic lossよりも局所write mask構造そのものが効いている可能性があります。")
    lines.append("- 全Residual条件がbaseline以下なら、pixel-level residual rewardだけでは不十分で、DINO patch feature / progress valueへの移行を検討してください。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rft_eval_root",
        nargs="?",
        default="results/phase1/phase1_sweeps/phase1_residual_spatial_rft/rft_eval",
    )
    parser.add_argument(
        "--wm-eval-root",
        default="results/phase1/phase1_sweeps/phase1_residual_spatial/wm_eval_episode_phases_e7",
    )
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    rft_eval_root = Path(args.rft_eval_root)
    wm_eval_root = Path(args.wm_eval_root)
    out_md = Path(args.out_md) if args.out_md else rft_eval_root / "comparison.md"

    rows = collect_eval_rows(rft_eval_root, wm_eval_root)
    task_rows = collect_task_rows(rft_eval_root)
    manifest_rows = load_manifest_rows(rft_eval_root)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(make_markdown(rft_eval_root, wm_eval_root, rows, task_rows, manifest_rows), encoding="utf-8")

    summary_fields = [
        "exp_name",
        "wm_exp_name",
        "model_generation",
        "target_mode",
        "rft_step",
        "task_suite",
        "num_trials",
        "seed",
        "success_rate",
        "num_successes",
        "num_episodes",
        "num_tasks",
        "wm_full_mse",
        "wm_gripper_mse",
        "wm_dynamic_mse",
        "wm_pairwise_acc",
        "wm_lpips_gap_mean",
        "status",
        "policy_ckpt",
    ]
    write_csv(rft_eval_root / "comparison.csv", rows, summary_fields)
    task_fields = sorted({k for row in task_rows for k in row})
    if task_fields:
        preferred = ["task_id", "task_description", "best_model", "best_success"]
        task_fields = preferred + [k for k in task_fields if k not in preferred]
        write_csv(rft_eval_root / "comparison_by_task.csv", task_rows, task_fields)

    print(f"Wrote {out_md}")
    print(f"Wrote {rft_eval_root / 'comparison.csv'}")
    if task_fields:
        print(f"Wrote {rft_eval_root / 'comparison_by_task.csv'}")


if __name__ == "__main__":
    main()
