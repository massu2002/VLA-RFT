"""
Build task-level metrics table from Phase 0 outputs.

Reads:
  results/phase0/<run>/worldmodel/task{i}/eval_report__trained__*.json
  results/phase0/<run>/summary.json

Outputs:
  results/phase0/<run>/correlation/task_level_metrics.csv
  results/phase0/<run>/correlation/task_level_metrics.json
"""

import argparse
import json
import csv
import glob
import os
import sys


def load_eval_report(task_dir: str) -> dict:
    pattern = os.path.join(task_dir, "eval_report__trained__*.json")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No eval_report file in {task_dir}")
    with open(matches[0]) as fp:
        return json.load(fp)


def extract_wm_metrics(report: dict) -> dict:
    rf = report["rollout_fidelity"]["overall"]
    sa = report["action_sensitivity"]
    return {
        "full_image_lpips": rf["lpips"]["mean"],
        "full_image_mse": rf["mse"]["mean"],
        "gripper_lpips": rf["roi/gripper_lpips"]["mean"],
        "gripper_mse": rf["roi/gripper_mse"]["mean"],
        "goal_lpips": rf["roi/goal_lpips"]["mean"],
        "goal_mse": rf["roi/goal_mse"]["mean"],
        "pairwise_acc": sa["per_window_pairwise_acc"],
        "lpips_gap_mean": sa["lpips_gap"]["mean"],
        "correct_lpips_mean": sa["correct_lpips"]["mean"],
        "shuffled_lpips_mean": sa["shuffled_lpips"]["mean"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="results/phase0/20260429_055911_spatial_all",
        help="Phase 0 run directory",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    summary_path = os.path.join(run_dir, "summary.json")

    with open(summary_path) as fp:
        summary = json.load(fp)

    base_vla = summary["base_vla"]
    vla_rft = summary["vla_rft"]

    rows = []
    for task_id in range(10):
        task_dir = os.path.join(run_dir, "worldmodel", f"task{task_id}")
        report = load_eval_report(task_dir)
        wm = extract_wm_metrics(report)

        task_name = base_vla[str(task_id)]["task_name"]
        base_sr = base_vla[str(task_id)]["success_rate"]
        rft_sr = vla_rft[str(task_id)]["success_rate"]
        delta = round(rft_sr - base_sr, 4)

        row = {
            "task_id": task_id,
            "task_name": task_name,
            "base_success": base_sr,
            "rft_success": rft_sr,
            "delta_success": delta,
            **wm,
        }
        rows.append(row)

    # CSV
    out_dir = os.path.join(run_dir, "correlation")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "task_level_metrics.csv")
    json_path = os.path.join(out_dir, "task_level_metrics.json")

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as fp:
        json.dump(rows, fp, indent=2)

    print(f"Written: {csv_path}")
    print(f"Written: {json_path}")

    # Print summary table
    print(
        f"\n{'Task':>4}  {'gripper_lpips':>13}  {'goal_lpips':>10}  "
        f"{'pairwise_acc':>12}  {'base':>5}  {'rft':>5}  {'delta':>6}"
    )
    for r in rows:
        print(
            f"{r['task_id']:>4}  {r['gripper_lpips']:>13.4f}  {r['goal_lpips']:>10.4f}  "
            f"{r['pairwise_acc']:>12.2f}  {r['base_success']:>5.1%}  "
            f"{r['rft_success']:>5.1%}  {r['delta_success']:>+6.0%}"
        )


if __name__ == "__main__":
    main()
