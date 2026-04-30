"""
Compute Pearson + Spearman correlations between WM error metrics and VLA success,
then produce scatter plots with regression lines and a 6-panel overview PNG.

Reads:
  results/phase0/<run>/correlation/task_level_metrics.csv

Outputs:
  results/phase0/<run>/correlation/correlation_summary.csv
  results/phase0/<run>/correlation/correlation_summary.json
  results/phase0/<run>/correlation/plots/*.png            (individual scatter plots)
  results/phase0/<run>/correlation/plots/phase0_correlation_overview.png
  results/phase0/<run>/correlation/correlation_notes.md
"""

import argparse
import csv
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


# ── data loading ─────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as fp:
        reader = csv.DictReader(fp)
        rows = []
        for r in reader:
            rows.append({k: (float(v) if k not in ("task_name",) else v)
                         for k, v in r.items()})
    return rows


# ── correlation helpers ───────────────────────────────────────────────────────

def pearson(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def spearman(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def compute_correlations(rows: list[dict]) -> list[dict]:
    wm_metrics = [
        "gripper_lpips", "gripper_mse",
        "goal_lpips", "goal_mse",
        "pairwise_acc", "lpips_gap_mean",
        "full_image_lpips",
    ]
    target_metrics = ["base_success", "rft_success", "delta_success"]

    results = []
    for wm in wm_metrics:
        x = np.array([r[wm] for r in rows])
        for tgt in target_metrics:
            y = np.array([r[tgt] for r in rows])
            pr, pp = pearson(x, y)
            sr, sp = spearman(x, y)
            results.append({
                "wm_metric": wm,
                "target": tgt,
                "pearson_r": round(pr, 4),
                "pearson_p": round(pp, 4),
                "spearman_r": round(sr, 4),
                "spearman_p": round(sp, 4),
            })
    return results


# ── scatter plot helper ───────────────────────────────────────────────────────

TASK5_COLOR = "#e74c3c"   # red  — zero-success outlier
TASK7_COLOR = "#e67e22"   # orange — low pairwise_acc outlier
DEFAULT_COLOR = "#2980b9"  # blue
NORMAL_ALPHA = 0.85

XLABELS = {
    "gripper_lpips": "Gripper ROI LPIPS ↑ (worse WM)",
    "gripper_mse":   "Gripper ROI MSE ↑ (worse WM)",
    "goal_lpips":    "Goal ROI LPIPS ↑ (worse WM)",
    "goal_mse":      "Goal ROI MSE ↑ (worse WM)",
    "pairwise_acc":  "Ranking pairwise_acc ↑ (better signal)",
    "lpips_gap_mean":"LPIPS gap mean ↑ (better signal)",
    "full_image_lpips": "Full-image LPIPS ↑ (worse WM)",
}
YLABELS = {
    "base_success": "Base VLA success rate",
    "rft_success":  "VLA-RFT success rate",
    "delta_success": "VLA-RFT − Base VLA (Δ success)",
}


def point_color(task_id: int) -> str:
    if task_id == 5:
        return TASK5_COLOR
    if task_id == 7:
        return TASK7_COLOR
    return DEFAULT_COLOR


def scatter_one(
    ax,
    rows: list[dict],
    wm_metric: str,
    tgt_metric: str,
    corr_results: list[dict],
    show_legend: bool = True,
):
    x = np.array([r[wm_metric] for r in rows])
    y = np.array([r[tgt_metric] for r in rows])
    task_ids = [int(r["task_id"]) for r in rows]

    colors = [point_color(t) for t in task_ids]
    ax.scatter(x, y, c=colors, s=60, zorder=3, alpha=NORMAL_ALPHA, edgecolors="white", linewidths=0.5)

    for xi, yi, tid in zip(x, y, task_ids):
        ax.annotate(
            f"T{tid}",
            (xi, yi),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
            color=point_color(tid),
        )

    # Regression line (all points)
    m, b, _, _, _ = stats.linregress(x, y)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, m * xline + b, color="gray", linewidth=1.2, linestyle="--", zorder=2)

    # Correlation annotation
    entry = next(
        (c for c in corr_results if c["wm_metric"] == wm_metric and c["target"] == tgt_metric),
        None,
    )
    if entry:
        pr = entry["pearson_r"]
        sr = entry["spearman_r"]
        pp = entry["pearson_p"]
        ax.text(
            0.97, 0.05,
            f"Pearson r={pr:.2f} (p={pp:.2f})\nSpearman ρ={sr:.2f}",
            transform=ax.transAxes,
            fontsize=7.5,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.9),
        )

    ax.set_xlabel(XLABELS.get(wm_metric, wm_metric), fontsize=8)
    ax.set_ylabel(YLABELS.get(tgt_metric, tgt_metric), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.4, zorder=1)

    if show_legend:
        patches = [
            mpatches.Patch(color=TASK5_COLOR, label="Task5 (0%/0% outlier)"),
            mpatches.Patch(color=TASK7_COLOR, label="Task7 (pairwise=0.40)"),
            mpatches.Patch(color=DEFAULT_COLOR, label="Other tasks"),
        ]
        ax.legend(handles=patches, fontsize=7, loc="upper left")


# ── individual scatter plots ─────────────────────────────────────────────────

INDIVIDUAL_PAIRS = [
    ("gripper_lpips", "base_success"),
    ("gripper_lpips", "rft_success"),
    ("gripper_lpips", "delta_success"),
    ("goal_lpips", "base_success"),
    ("goal_lpips", "rft_success"),
    ("goal_lpips", "delta_success"),
    ("pairwise_acc", "base_success"),
    ("pairwise_acc", "rft_success"),
    ("pairwise_acc", "delta_success"),
    ("lpips_gap_mean", "delta_success"),
    ("full_image_lpips", "base_success"),
    ("full_image_lpips", "delta_success"),
]

OVERVIEW_PAIRS = [
    ("gripper_lpips", "base_success"),
    ("gripper_lpips", "rft_success"),
    ("gripper_lpips", "delta_success"),
    ("goal_lpips", "delta_success"),
    ("pairwise_acc", "delta_success"),
    ("lpips_gap_mean", "delta_success"),
]


def save_individual(rows, pairs, corr_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for wm, tgt in pairs:
        fig, ax = plt.subplots(figsize=(5, 4))
        scatter_one(ax, rows, wm, tgt, corr_results, show_legend=True)
        ax.set_title(f"{wm}  vs  {tgt}", fontsize=9)
        fname = f"{wm}__vs__{tgt}.png"
        path = os.path.join(out_dir, fname)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    return saved


def save_overview(rows, pairs, corr_results, out_path):
    n = len(pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, (wm, tgt) in enumerate(pairs):
        scatter_one(axes_flat[i], rows, wm, tgt, corr_results, show_legend=(i == 0))
        axes_flat[i].set_title(f"{wm}\nvs {tgt}", fontsize=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Phase 0 — WM error metrics vs VLA success  (n=10, LIBERO-Spatial)\n"
        "Red=Task5 (0%/0%), Orange=Task7 (pairwise=0.40)",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── notes markdown ───────────────────────────────────────────────────────────

def strongest(corr_results, target, key="pearson_r"):
    subset = [c for c in corr_results if c["target"] == target]
    subset.sort(key=lambda c: abs(c[key]), reverse=True)
    return subset[:3]


def generate_notes(corr_results: list[dict], rows: list[dict]) -> str:
    lines = [
        "# Phase 0 — Correlation Analysis Notes",
        "",
        "> **Caution**: n=10 tasks. Pearson/Spearman correlations with n=10 are highly",
        "> sensitive to outliers and should be treated as diagnostic indicators only, not",
        "> strong statistical evidence.",
        "",
        "## Outlier flags",
        "",
        "- **Task 5** (`pick up the black bowl on the ramekin`): Base VLA = 0%, VLA-RFT = 0%.",
        "  This task has no policy signal at all. It is included in all computations but its",
        "  outsized influence on regressions involving `base_success` / `rft_success` should",
        "  be kept in mind.",
        "- **Task 7** (`pick up the black bowl on the stove`): pairwise_acc = 0.40 — WorldModel",
        "  ranking signal is *worse than random* for this task. Yet both Base VLA and VLA-RFT",
        "  achieve 100%. The WM error metric is a poor predictor here.",
        "",
        "## Strongest correlations with delta_success (VLA-RFT gain)",
        "",
    ]

    top_delta = strongest(corr_results, "delta_success")
    for c in top_delta:
        lines.append(
            f"- `{c['wm_metric']}`: Pearson r={c['pearson_r']:+.2f} (p={c['pearson_p']:.2f}), "
            f"Spearman ρ={c['spearman_r']:+.2f}"
        )

    lines += [
        "",
        "## Strongest correlations with base_success",
        "",
    ]
    top_base = strongest(corr_results, "base_success")
    for c in top_base:
        lines.append(
            f"- `{c['wm_metric']}`: Pearson r={c['pearson_r']:+.2f} (p={c['pearson_p']:.2f}), "
            f"Spearman ρ={c['spearman_r']:+.2f}"
        )

    lines += [
        "",
        "## Key observations",
        "",
    ]

    # Auto-generate observations from data
    gripper_base = next(c for c in corr_results if c["wm_metric"] == "gripper_lpips" and c["target"] == "base_success")
    gripper_delta = next(c for c in corr_results if c["wm_metric"] == "gripper_lpips" and c["target"] == "delta_success")
    pairwise_delta = next(c for c in corr_results if c["wm_metric"] == "pairwise_acc" and c["target"] == "delta_success")
    lpipsgap_delta = next(c for c in corr_results if c["wm_metric"] == "lpips_gap_mean" and c["target"] == "delta_success")

    def sign_str(r):
        return "positive" if r >= 0 else "negative"

    lines += [
        f"1. **Gripper LPIPS vs base_success**: r={gripper_base['pearson_r']:+.2f} "
        f"({sign_str(gripper_base['pearson_r'])}). "
        "Higher gripper reconstruction error correlates with "
        + ("lower" if gripper_base["pearson_r"] < 0 else "higher")
        + " base policy success.",
        "",
        f"2. **Gripper LPIPS vs delta_success**: r={gripper_delta['pearson_r']:+.2f}. "
        "If negative, tasks where the WM reconstruction is worse tended *not* to benefit "
        "more from RFT — consistent with WM reward signal being noisy on hard tasks.",
        "",
        f"3. **pairwise_acc vs delta_success**: ρ={pairwise_delta['spearman_r']:+.2f}. "
        "Tasks with stronger WM ranking signal "
        + ("tended to benefit more from" if pairwise_delta["spearman_r"] > 0 else "did not reliably benefit more from")
        + " RFT.",
        "",
        f"4. **lpips_gap_mean vs delta_success**: r={lpipsgap_delta['pearson_r']:+.2f}. "
        "A larger action-sensitivity gap (GT LPIPS lower than shuffled LPIPS) "
        + ("correlates with larger RFT gains." if lpipsgap_delta["pearson_r"] > 0 else "does not clearly correlate with RFT gains."),
        "",
        "## Summary verdict",
        "",
        "With n=10 tasks, no single WM metric shows a dominant, statistically significant "
        "correlation with delta_success. The main takeaway is that **Task 5 is an outlier "
        "that has zero policy signal**, and **Task 7 shows WM ranking failure (pairwise_acc=0.40) "
        "despite 100% VLA success** — indicating the WM reward is unreliable on some tasks "
        "independent of policy performance.",
        "",
        "Phase 1 (Residual WM) should be evaluated on whether it can fix the WM signal on "
        "Tasks 3, 7, 8 (low pairwise_acc) while measuring whether improved WM ranking "
        "translates to larger delta_success.",
    ]

    return "\n".join(lines) + "\n"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="results/phase0/20260429_055911_spatial_all",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    corr_dir = os.path.join(run_dir, "correlation")
    plots_dir = os.path.join(corr_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    csv_in = os.path.join(corr_dir, "task_level_metrics.csv")
    rows = load_csv(csv_in)

    corr_results = compute_correlations(rows)

    # Save correlation summary
    csv_out = os.path.join(corr_dir, "correlation_summary.csv")
    json_out = os.path.join(corr_dir, "correlation_summary.json")

    fieldnames = list(corr_results[0].keys())
    with open(csv_out, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corr_results)

    with open(json_out, "w") as fp:
        json.dump(corr_results, fp, indent=2)

    print(f"Written: {csv_out}")
    print(f"Written: {json_out}")

    # Individual scatter plots
    saved = save_individual(rows, INDIVIDUAL_PAIRS, corr_results, plots_dir)
    for p in saved:
        print(f"Written: {p}")

    # 6-panel overview
    overview_path = os.path.join(plots_dir, "phase0_correlation_overview.png")
    save_overview(rows, OVERVIEW_PAIRS, corr_results, overview_path)
    print(f"Written: {overview_path}")

    # Notes markdown
    notes = generate_notes(corr_results, rows)
    notes_path = os.path.join(corr_dir, "correlation_notes.md")
    with open(notes_path, "w") as fp:
        fp.write(notes)
    print(f"Written: {notes_path}")

    # Print correlation table
    print("\n=== Correlation summary (|Pearson r| > 0.30) ===")
    print(f"{'WM metric':20s}  {'target':15s}  {'pearson_r':>9}  {'p':>6}  {'spearman_r':>10}")
    for c in sorted(corr_results, key=lambda x: abs(x["pearson_r"]), reverse=True):
        if abs(c["pearson_r"]) > 0.30:
            print(
                f"{c['wm_metric']:20s}  {c['target']:15s}  "
                f"{c['pearson_r']:>+9.3f}  {c['pearson_p']:>6.3f}  {c['spearman_r']:>+10.3f}"
            )


if __name__ == "__main__":
    main()
