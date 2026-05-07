#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

ROOT="${ROOT:-${REPO_ROOT}/results/phase1/residual_worldmodel}"
RFT_ROOT="${RFT_ROOT:-${ROOT}/rft}"
OUT_MD="${RFT_ROOT}/summary.md"
OUT_CSV="${RFT_ROOT}/summary.csv"
OUT_TASK_CSV="${RFT_ROOT}/summary_by_task.csv"

mkdir -p "${RFT_ROOT}"
export ROOT RFT_ROOT OUT_MD OUT_CSV OUT_TASK_CSV

"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import csv, json, math, os
from pathlib import Path

root = Path(os.environ.get("ROOT", "results/phase1/residual_worldmodel"))
rft_root = Path(os.environ.get("RFT_ROOT", root / "rft"))
out_csv = Path(os.environ.get("OUT_CSV", rft_root / "summary.csv"))
out_task_csv = Path(os.environ.get("OUT_TASK_CSV", rft_root / "summary_by_task.csv"))
out_md = Path(os.environ.get("OUT_MD", rft_root / "summary.md"))

def load_json(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}

def fnum(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return ""
        return float(x)
    except Exception:
        return ""

wm_files = [p for p in root.glob("**/aggregate_metrics.json") if "/rft/" not in str(p)]
wm_by_key = {}
for p in wm_files:
    data = load_json(p)
    cfg = load_json(p.parent / "config_used.json")
    mode = data.get("target_mode") or cfg.get("target_mode") or p.parent.name
    gen = cfg.get("model_generation") or ("v3" if "v3" in str(p) else "v1" if "v1" in str(p) else "")
    wm_by_key[(gen, mode)] = data
    wm_by_key[("", mode)] = data

rows = []
tasks_by_exp = {}
for exp_dir in sorted(p for p in rft_root.iterdir() if p.is_dir()):
    info = load_json(exp_dir / "world_model_info.json")
    if not info:
        continue
    eval_summary = load_json(exp_dir / "eval" / "success_summary.json")
    exp = exp_dir.name
    gen = info.get("model_generation", "")
    mode = info.get("target_mode", "")
    wm = wm_by_key.get((gen, mode), wm_by_key.get(("", mode), {}))
    sr = eval_summary.get("success_rate")
    n_tasks = eval_summary.get("num_tasks_evaluated")
    n_eps = eval_summary.get("num_episodes")
    trials = ""
    if n_tasks and n_eps:
        trials = int(n_eps) // int(n_tasks)
    row = {
        "exp_name": exp,
        "model_generation": gen,
        "target_mode": mode,
        "world_model_ckpt": info.get("world_model_ckpt", ""),
        "wm_full_mse": fnum(wm.get("full_mse")),
        "wm_full_lpips": fnum(wm.get("full_lpips")),
        "wm_gripper_mse": fnum(wm.get("gripper_mse")),
        "wm_gripper_lpips": fnum(wm.get("gripper_lpips")),
        "wm_dynamic_mse": fnum(wm.get("dynamic_mse")),
        "wm_dynamic_lpips": fnum(wm.get("dynamic_lpips")),
        "wm_pairwise_acc": fnum(wm.get("pairwise_acc")),
        "wm_lpips_gap_mean": fnum(wm.get("lpips_gap_mean")),
        "wm_lpips_gap_min": fnum(wm.get("lpips_gap_min")),
        "wm_reverse_windows": fnum(wm.get("reverse_windows")),
        "rft_success_mean": fnum(sr),
        "rft_success_std": "",
        "delta_over_base_vla": "",
        "delta_over_ar_pixel_rft": "",
        "num_tasks": n_tasks or "",
        "num_trials_per_task": trials,
    }
    rows.append(row)

    task_csv = exp_dir / "eval" / "success_by_task.csv"
    if task_csv.exists():
        with open(task_csv, newline="", encoding="utf-8") as f:
            tasks_by_exp[exp] = list(csv.DictReader(f))

baseline_success = next((r["rft_success_mean"] for r in rows if r["exp_name"] == "baseline" and r["rft_success_mean"] != ""), "")
for r in rows:
    if baseline_success != "" and r["rft_success_mean"] != "":
        r["delta_over_ar_pixel_rft"] = float(r["rft_success_mean"]) - float(baseline_success)

fieldnames = [
    "exp_name", "model_generation", "target_mode", "world_model_ckpt",
    "wm_full_mse", "wm_full_lpips", "wm_gripper_mse", "wm_gripper_lpips",
    "wm_dynamic_mse", "wm_dynamic_lpips", "wm_pairwise_acc",
    "wm_lpips_gap_mean", "wm_lpips_gap_min", "wm_reverse_windows",
    "rft_success_mean", "rft_success_std", "delta_over_base_vla",
    "delta_over_ar_pixel_rft", "num_tasks", "num_trials_per_task",
]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

all_task_ids = sorted({
    int(t.get("task_id", -1))
    for tasks in tasks_by_exp.values()
    for t in tasks
    if str(t.get("task_id", "")).isdigit()
})
task_fields = [
    "task_id", "task_name_or_description", "base_vla_success",
    "ar_pixel_rft_success", "v1_pixel_residual_success",
    "v1_pixel_residual_roi_dynamic_success", "v3_pixel_residual_success",
    "v3_pixel_residual_roi_dynamic_success", "best_model",
    "best_delta_over_ar_pixel_rft",
]
exp_to_col = {
    "baseline": "ar_pixel_rft_success",
    "v1_pixel_residual": "v1_pixel_residual_success",
    "v1_pixel_residual_roi_dynamic": "v1_pixel_residual_roi_dynamic_success",
    "v3_pixel_residual": "v3_pixel_residual_success",
    "v3_pixel_residual_roi_dynamic": "v3_pixel_residual_roi_dynamic_success",
}
task_rows = []
for tid in all_task_ids:
    out = {k: "" for k in task_fields}
    out["task_id"] = tid
    scores = {}
    for exp, tasks in tasks_by_exp.items():
        for t in tasks:
            if int(t.get("task_id", -1)) == tid:
                out["task_name_or_description"] = out["task_name_or_description"] or t.get("task_description", "")
                col = exp_to_col.get(exp)
                if col:
                    val = fnum(t.get("success_rate"))
                    out[col] = val
                    scores[exp] = val
    if scores:
        best = max(scores.items(), key=lambda kv: kv[1])
        out["best_model"] = best[0]
        base = scores.get("baseline")
        if base is not None:
            out["best_delta_over_ar_pixel_rft"] = best[1] - base
    task_rows.append(out)
with open(out_task_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=task_fields)
    w.writeheader()
    w.writerows(task_rows)

def fmt(x):
    return "" if x == "" else f"{float(x):.4f}"

lines = []
lines.append("# Phase 1 Residual WM RFT Summary\n")
lines.append("## 1. 実験条件一覧\n")
for r in rows:
    lines.append(f"- {r['exp_name']}: generation={r['model_generation']}, target_mode={r['target_mode']}")
lines.append("\n## 2. World Model単体指標の比較\n")
lines.append("| exp | full_mse | gripper_mse | dynamic_mse | pairwise_acc | lpips_gap |")
lines.append("|---|---:|---:|---:|---:|---:|")
for r in rows:
    lines.append(f"| {r['exp_name']} | {fmt(r['wm_full_mse'])} | {fmt(r['wm_gripper_mse'])} | {fmt(r['wm_dynamic_mse'])} | {fmt(r['wm_pairwise_acc'])} | {fmt(r['wm_lpips_gap_mean'])} |")
lines.append("\n## 3. RFT後 success の比較\n")
lines.append("| exp | success | delta over AR-Pixel RFT |")
lines.append("|---|---:|---:|")
for r in rows:
    lines.append(f"| {r['exp_name']} | {fmt(r['rft_success_mean'])} | {fmt(r['delta_over_ar_pixel_rft'])} |")
lines.append("\n## 4. task別 success の比較\n")
lines.append(f"- 詳細: `{out_task_csv}`")
lines.append("\n## 5. WM指標と RFT success の対応\n")
lines.append("- WM指標が改善し、RFT success も改善した条件は residual WM が post-training signal として有効な候補です。")
lines.append("- WM指標は改善したが RFT success が改善しない場合は、Phase 2 の progress-aware value / reward設計を検討してください。")
lines.append("\n## 6. 採用候補モデル\n")
valid = [r for r in rows if r["rft_success_mean"] != ""]
if valid:
    best = max(valid, key=lambda r: float(r["rft_success_mean"]))
    lines.append(f"- 現時点の success 最大: **{best['exp_name']}** ({fmt(best['rft_success_mean'])})")
else:
    lines.append("- まだ RFT eval の成功率が見つかっていません。")
lines.append("\n## 7. 次に見るべき課題\n")
lines.append("- v3 pixel_residual_roi_dynamic が AR-Pixel RFT より改善: action-sensitive local residual dynamics が有効な可能性が高い。")
lines.append("- v1 は改善せず v3 が改善: residual target だけでなく local action-conditioned update が重要。")
lines.append("- v3 でも改善しない: pixel-level residual だけでは不十分で、DINO patch feature / focused residual model / progress value へ進む。")

out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
