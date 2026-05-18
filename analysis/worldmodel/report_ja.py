#!/usr/bin/env python3
"""Generate a Japanese Markdown evaluation report from aggregate_metrics.json.

Usage (from repo root):
    python scripts/generate_wm_report_ja.py \\
        --metrics-file results/baseline_ar_pixel_wm/spatial/aggregate_metrics.json \\
        --output results/baseline_ar_pixel_wm/spatial/wm_result.md \\
        --ckpt checkpoints/libero/WorldModel/spatial/20260514_baseline_ar_pixel_wm/checkpoint-150000 \\
        --task-suite spatial
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Metric metadata: (display_name, unit, meaning_ja, acceptable_ranges_ja)
# ---------------------------------------------------------------------------

METRIC_META: dict[str, tuple[str, str, str, str]] = {
    # ── 全体フレーム品質 ──────────────────────────────────────────────────
    "horizon_avg_lpips": (
        "全体LPIPS（水平線平均）",
        "低いほど良",
        "予測フレームと正解フレームの知覚的類似度（AlexNetベースのLPIPS）。"
        "人間の知覚に近い距離尺度で、テクスチャや構造の乱れを捉える。",
        "< 0.10: 優秀 ／ < 0.20: 良好 ／ < 0.30: 可 ／ ≥ 0.30: 要改善",
    ),
    "horizon_avg_mae": (
        "全体MAE（水平線平均）",
        "低いほど良",
        "フレーム全体のピクセル平均絶対誤差（Mean Absolute Error）。"
        "RFT報酬プロキシの構成要素の一つ。値域は [0, 1]（正規化後）。",
        "< 0.03: 優秀 ／ < 0.05: 良好 ／ < 0.08: 可 ／ ≥ 0.08: 要改善",
    ),
    "horizon_avg_mse": (
        "全体MSE（水平線平均）",
        "低いほど良",
        "フレーム全体のピクセル平均二乗誤差（Mean Squared Error）。"
        "MAEと比べ外れ値に敏感。",
        "< 0.003: 優秀 ／ < 0.008: 良好 ／ < 0.015: 可 ／ ≥ 0.015: 要改善",
    ),
    # ── ステップ別LPIPS ────────────────────────────────────────────────────
    "lpips_step1": (
        "LPIPS ステップ1",
        "低いほど良",
        "予測ステップ1（1フレーム先）のLPIPS。近傍予測精度を示す。",
        "< 0.08: 優秀 ／ < 0.15: 良好 ／ < 0.25: 可",
    ),
    "lpips_step4": (
        "LPIPS ステップ4",
        "低いほど良",
        "予測ステップ4（4フレーム先）のLPIPS。中期予測精度を示す。",
        "< 0.12: 優秀 ／ < 0.22: 良好 ／ < 0.32: 可",
    ),
    "lpips_step8": (
        "LPIPS ステップ8",
        "低いほど良",
        "予測ステップ8（8フレーム先）のLPIPS。長期予測精度を示す。"
        "ステップが進むほど予測困難になるため数値が大きくなるのは自然。",
        "< 0.18: 優秀 ／ < 0.28: 良好 ／ < 0.38: 可",
    ),
    # ── ステップ別MAE ──────────────────────────────────────────────────────
    "mae_step1": (
        "MAE ステップ1",
        "低いほど良",
        "予測ステップ1のピクセルMAE。",
        "< 0.025: 優秀 ／ < 0.040: 良好 ／ < 0.060: 可",
    ),
    "mae_step4": (
        "MAE ステップ4",
        "低いほど良",
        "予測ステップ4のピクセルMAE。",
        "< 0.035: 優秀 ／ < 0.055: 良好 ／ < 0.075: 可",
    ),
    "mae_step8": (
        "MAE ステップ8",
        "低いほど良",
        "予測ステップ8のピクセルMAE。",
        "< 0.045: 優秀 ／ < 0.065: 良好 ／ < 0.085: 可",
    ),
    # ── GT動的マスク評価 ───────────────────────────────────────────────────
    "dynamic_region_mse_gt": (
        "GT動的領域 MSE",
        "低いほど良",
        "GT動的マスク（|frame_t+1 − frame_t| > 0.05、膨張kernel=7）で抽出した"
        "動的ピクセル領域における予測誤差（MSE）。"
        "グリッパーや操作対象など実際に動く領域への集中評価。",
        "< 0.008: 優秀 ／ < 0.015: 良好 ／ < 0.025: 可",
    ),
    "dynamic_region_mae_gt": (
        "GT動的領域 MAE",
        "低いほど良",
        "GT動的マスク領域における予測誤差（MAE）。",
        "< 0.040: 優秀 ／ < 0.070: 良好 ／ < 0.100: 可",
    ),
    "dynamic_region_lpips_gt": (
        "GT動的領域 LPIPS",
        "低いほど良",
        "GT動的マスク領域における知覚的距離（LPIPS）。"
        "動く部分の見た目の品質を直接評価する。",
        "< 0.12: 優秀 ／ < 0.22: 良好 ／ < 0.32: 可",
    ),
    "static_consistency_mse": (
        "静的領域一貫性 MSE",
        "低いほど良",
        "GT動的マスクで静的と判定された領域（背景など）における、"
        "予測フレームと現在フレームの差分（MSE）。"
        "世界モデルが動かない部分をどれだけ正確に保持できているかを示す。",
        "< 0.003: 優秀 ／ < 0.006: 良好 ／ < 0.010: 可",
    ),
    "static_consistency_mae": (
        "静的領域一貫性 MAE",
        "低いほど良",
        "静的領域における予測フレームと現在フレームのMAE。",
        "< 0.015: 優秀 ／ < 0.025: 良好 ／ < 0.035: 可",
    ),
    # ── ROI評価（グリッパー） ──────────────────────────────────────────────
    "roi/gripper_mse": (
        "グリッパーROI MSE",
        "低いほど良",
        "動き重心（motion center-of-mass）で推定したグリッパー近傍のROI切り抜き"
        "における予測誤差（MSE）。把持動作の予測精度を集中評価。",
        "< 0.010: 優秀 ／ < 0.020: 良好 ／ < 0.030: 可",
    ),
    "roi/gripper_mae": (
        "グリッパーROI MAE",
        "低いほど良",
        "グリッパーROI領域における予測誤差（MAE）。",
        "< 0.050: 優秀 ／ < 0.080: 良好 ／ < 0.110: 可",
    ),
    "roi/gripper_lpips": (
        "グリッパーROI LPIPS",
        "低いほど良",
        "グリッパーROI領域における知覚的距離（LPIPS）。",
        "< 0.15: 優秀 ／ < 0.25: 良好 ／ < 0.35: 可",
    ),
    # ── ROI評価（ゴール） ──────────────────────────────────────────────────
    "roi/goal_mse": (
        "ゴールROI MSE",
        "低いほど良",
        "タスクゴール位置（roi_coords_v1.json から取得）のROI切り抜きにおける"
        "予測誤差（MSE）。操作対象・目標領域の予測精度を評価。",
        "< 0.008: 優秀 ／ < 0.015: 良好 ／ < 0.025: 可",
    ),
    "roi/goal_mae": (
        "ゴールROI MAE",
        "低いほど良",
        "ゴールROI領域における予測誤差（MAE）。",
        "< 0.040: 優秀 ／ < 0.065: 良好 ／ < 0.090: 可",
    ),
    "roi/goal_lpips": (
        "ゴールROI LPIPS",
        "低いほど良",
        "ゴールROI領域における知覚的距離（LPIPS）。",
        "< 0.12: 優秀 ／ < 0.20: 良好 ／ < 0.30: 可",
    ),
    # ── RFT報酬信号 ───────────────────────────────────────────────────────
    "rft_reward_proxy": (
        "RFT報酬プロキシ（平均）",
        "高いほど良",
        "RFT事後学習に使用する報酬代理値 = −(LPIPS + MAE)。"
        "正しいアクション条件下での世界モデル予測品質を示す。"
        "値が高いほど事後学習の学習信号として強い。",
        "> −0.20: 優秀 ／ > −0.30: 良好 ／ > −0.40: 可 ／ ≤ −0.40: 要改善",
    ),
    "rft_reward_gap": (
        "RFT報酬ギャップ（平均）",
        "高いほど良",
        "正しいアクションの報酬プロキシ − 誤ったアクションの報酬プロキシ。"
        "正値であれば世界モデルが正誤アクションを識別できており、"
        "RFT信号として有効。値が大きいほど識別力が強い。",
        "> 0.05: 優秀 ／ > 0.02: 良好 ／ > 0.00: 可（辛うじて識別） ／ ≤ 0.00: 識別失敗",
    ),
    "rft_reward_gap_mean": (
        "RFT報酬ギャップ（mean alias）",
        "高いほど良",
        "rft_reward_gap と同値（一部スクリプトとの互換性エイリアス）。",
        "上記 rft_reward_gap と同じ目安を参照。",
    ),
    "pairwise_acc_rft": (
        "ペアワイズ精度（RFT報酬基準）",
        "高いほど良",
        "正しいアクションのRFT報酬 > 誤ったアクションのRFT報酬となる"
        "ウィンドウの割合。0.50 = ランダム。RFT識別力の頻度指標。",
        "> 0.70: 優秀 ／ > 0.60: 良好 ／ > 0.55: 可 ／ ≤ 0.50: 問題（識別力なし）",
    ),
    "rft_reward_proxy_std": (
        "RFT報酬プロキシ標準偏差",
        "適度な値が望ましい",
        "ウィンドウ間でのRFT報酬プロキシのばらつき（標準偏差）。"
        "大きすぎるとRFT学習が不安定になる可能性がある。",
        "0.02〜0.08: 適度 ／ > 0.10: ばらつき大（注意）",
    ),
    "rft_reward_gap_std": (
        "RFT報酬ギャップ標準偏差",
        "参考値",
        "ウィンドウ間での報酬ギャップのばらつき。"
        "大きい場合、タスク・フェーズによって識別容易度が大きく異なることを示す。",
        "参考値（特定の目安なし）",
    ),
    "rft_reward_gap_min": (
        "RFT報酬ギャップ（最小値）",
        "高いほど良",
        "最も識別が難しいウィンドウでの報酬ギャップ。"
        "負値であれば一部ウィンドウで識別失敗している。",
        "> 0.00: 全ウィンドウで識別成功 ／ < 0.00: 一部識別失敗（注意）",
    ),
    # ── 負例タイプ別ペアワイズ精度 ────────────────────────────────────────
    "pairwise_acc_rft_same_phase": (
        "ペアワイズ精度 — same_phase",
        "高いほど良",
        "同フェーズ別エピソードを負例としたときの"
        "RFT報酬ベースのペアワイズ精度。"
        "アクションに関係なく外見が似たフレームを識別できるかを測る。",
        "> 0.70: 優秀 ／ > 0.60: 良好 ／ > 0.55: 可 ／ ≤ 0.50: 問題",
    ),
    "pairwise_acc_rft_temporal_shift": (
        "ペアワイズ精度 — temporal_shift",
        "高いほど良",
        "時間シフト（最大±3ステップ）した誤アクション系列を負例としたときの"
        "ペアワイズ精度。タイミングのずれに対する識別力を測る。",
        "> 0.70: 優秀 ／ > 0.60: 良好 ／ > 0.55: 可 ／ ≤ 0.50: 問題",
    ),
    "pairwise_acc_rft_action_noise": (
        "ペアワイズ精度 — action_noise",
        "高いほど良",
        "正解アクションにガウスノイズ（std=0.15）を加えた負例に対する"
        "ペアワイズ精度。微妙なアクション差を識別できるかを測る。",
        "> 0.70: 優秀 ／ > 0.60: 良好 ／ > 0.55: 可 ／ ≤ 0.50: 問題",
    ),
    "rft_reward_gap_mean_same_phase": (
        "RFT報酬ギャップ — same_phase",
        "高いほど良",
        "same_phase 負例に対する平均報酬ギャップ（正例 − 負例）。",
        "> 0.05: 優秀 ／ > 0.02: 良好 ／ > 0.00: 可 ／ ≤ 0.00: 識別失敗",
    ),
    "rft_reward_gap_mean_temporal_shift": (
        "RFT報酬ギャップ — temporal_shift",
        "高いほど良",
        "temporal_shift 負例に対する平均報酬ギャップ（正例 − 負例）。",
        "> 0.05: 優秀 ／ > 0.02: 良好 ／ > 0.00: 可 ／ ≤ 0.00: 識別失敗",
    ),
    "rft_reward_gap_mean_action_noise": (
        "RFT報酬ギャップ — action_noise",
        "高いほど良",
        "action_noise 負例に対する平均報酬ギャップ（正例 − 負例）。",
        "> 0.05: 優秀 ／ > 0.02: 良好 ／ > 0.00: 可 ／ ≤ 0.00: 識別失敗",
    ),
    # ── DynQuery専用（AR-pixelでは N/A） ──────────────────────────────────
    "dynamic_mask_iou_gt_mean": (
        "GT動的マスクIoU（DynQuery専用）",
        "高いほど良",
        "DynQueryWorldModelのfuser_masksとGT動的マスクのIoU。"
        "AR-Pixel WMでは計測不可能（N/A = NaN）。",
        "N/A（AR-Pixel WMでは計測外）",
    ),
    "pearson_rft_score_corr": (
        "ピアソン相関（スコア vs RFT報酬、DynQuery専用）",
        "高いほど良",
        "DynQueryのActionFutureScoreとRFT報酬プロキシのピアソン相関。"
        "AR-Pixel WMでは計測不可能（N/A = NaN）。",
        "N/A（AR-Pixel WMでは計測外）",
    ),
    "spearman_rft_score_corr": (
        "スピアマン相関（スコア vs RFT報酬、DynQuery専用）",
        "高いほど良",
        "DynQueryのActionFutureScoreとRFT報酬プロキシのスピアマン順位相関。"
        "AR-Pixel WMでは計測不可能（N/A = NaN）。",
        "N/A（AR-Pixel WMでは計測外）",
    ),
}

# Display order groups
DISPLAY_GROUPS: list[tuple[str, list[str]]] = [
    ("全体フレーム品質", [
        "horizon_avg_lpips", "horizon_avg_mae", "horizon_avg_mse",
    ]),
    ("ステップ別指標（LPIPS）", [
        "lpips_step1", "lpips_step4", "lpips_step8",
    ]),
    ("ステップ別指標（MAE）", [
        "mae_step1", "mae_step4", "mae_step8",
    ]),
    ("GT動的マスク評価", [
        "dynamic_region_mse_gt", "dynamic_region_mae_gt", "dynamic_region_lpips_gt",
        "static_consistency_mse", "static_consistency_mae",
    ]),
    ("ROI評価 — グリッパー", [
        "roi/gripper_mse", "roi/gripper_mae", "roi/gripper_lpips",
    ]),
    ("ROI評価 — ゴール", [
        "roi/goal_mse", "roi/goal_mae", "roi/goal_lpips",
    ]),
    ("RFT報酬信号", [
        "rft_reward_proxy", "rft_reward_gap", "pairwise_acc_rft",
        "rft_reward_proxy_std", "rft_reward_gap_std", "rft_reward_gap_min",
    ]),
    ("負例タイプ別ペアワイズ精度", [
        "pairwise_acc_rft_same_phase",
        "pairwise_acc_rft_temporal_shift",
        "pairwise_acc_rft_action_noise",
        "rft_reward_gap_mean_same_phase",
        "rft_reward_gap_mean_temporal_shift",
        "rft_reward_gap_mean_action_noise",
    ]),
    ("DynQuery専用（AR-Pixel WMでは N/A）", [
        "dynamic_mask_iou_gt_mean", "pearson_rft_score_corr", "spearman_rft_score_corr",
    ]),
]


def _fmt(v: object, std: object = None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        s = f"{v:.5f}"
        if std is not None and isinstance(std, float) and not math.isnan(std):
            s += f" ±{std:.5f}"
        return s
    return str(v)


def _judge(key: str, v: object) -> str:
    """Return a simple pass/warn/fail/na emoji for the metric value."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if not isinstance(v, (int, float)):
        return "—"
    val = float(v)

    # Lower-is-better metrics (most metrics)
    lower_better = {
        "horizon_avg_lpips": [(0.10, "✅"), (0.20, "🟡"), (0.30, "🟠"), (None, "🔴")],
        "horizon_avg_mae":   [(0.03, "✅"), (0.05, "🟡"), (0.08, "🟠"), (None, "🔴")],
        "horizon_avg_mse":   [(0.003, "✅"), (0.008, "🟡"), (0.015, "🟠"), (None, "🔴")],
        "lpips_step1":       [(0.08, "✅"), (0.15, "🟡"), (0.25, "🟠"), (None, "🔴")],
        "lpips_step4":       [(0.12, "✅"), (0.22, "🟡"), (0.32, "🟠"), (None, "🔴")],
        "lpips_step8":       [(0.18, "✅"), (0.28, "🟡"), (0.38, "🟠"), (None, "🔴")],
        "mae_step1":         [(0.025, "✅"), (0.040, "🟡"), (0.060, "🟠"), (None, "🔴")],
        "mae_step4":         [(0.035, "✅"), (0.055, "🟡"), (0.075, "🟠"), (None, "🔴")],
        "mae_step8":         [(0.045, "✅"), (0.065, "🟡"), (0.085, "🟠"), (None, "🔴")],
        "dynamic_region_mse_gt":    [(0.008, "✅"), (0.015, "🟡"), (0.025, "🟠"), (None, "🔴")],
        "dynamic_region_mae_gt":    [(0.040, "✅"), (0.070, "🟡"), (0.100, "🟠"), (None, "🔴")],
        "dynamic_region_lpips_gt":  [(0.12, "✅"), (0.22, "🟡"), (0.32, "🟠"), (None, "🔴")],
        "static_consistency_mse":   [(0.003, "✅"), (0.006, "🟡"), (0.010, "🟠"), (None, "🔴")],
        "static_consistency_mae":   [(0.015, "✅"), (0.025, "🟡"), (0.035, "🟠"), (None, "🔴")],
        "roi/gripper_mse":   [(0.010, "✅"), (0.020, "🟡"), (0.030, "🟠"), (None, "🔴")],
        "roi/gripper_mae":   [(0.050, "✅"), (0.080, "🟡"), (0.110, "🟠"), (None, "🔴")],
        "roi/gripper_lpips": [(0.15, "✅"), (0.25, "🟡"), (0.35, "🟠"), (None, "🔴")],
        "roi/goal_mse":      [(0.008, "✅"), (0.015, "🟡"), (0.025, "🟠"), (None, "🔴")],
        "roi/goal_mae":      [(0.040, "✅"), (0.065, "🟡"), (0.090, "🟠"), (None, "🔴")],
        "roi/goal_lpips":    [(0.12, "✅"), (0.20, "🟡"), (0.30, "🟠"), (None, "🔴")],
    }
    # Higher-is-better metrics
    higher_better = {
        "rft_reward_proxy":  [(-0.20, "✅"), (-0.30, "🟡"), (-0.40, "🟠"), (None, "🔴")],
        "rft_reward_gap":    [(0.05, "✅"), (0.02, "🟡"), (0.00, "🟠"), (None, "🔴")],
        "rft_reward_gap_mean": [(0.05, "✅"), (0.02, "🟡"), (0.00, "🟠"), (None, "🔴")],
        "pairwise_acc_rft":  [(0.70, "✅"), (0.60, "🟡"), (0.55, "🟠"), (None, "🔴")],
        "pairwise_acc":      [(0.70, "✅"), (0.60, "🟡"), (0.55, "🟠"), (None, "🔴")],
        "pairwise_acc_rft_same_phase":      [(0.70, "✅"), (0.60, "🟡"), (0.55, "🟠"), (None, "🔴")],
        "pairwise_acc_rft_temporal_shift":  [(0.70, "✅"), (0.60, "🟡"), (0.55, "🟠"), (None, "🔴")],
        "pairwise_acc_rft_action_noise":    [(0.70, "✅"), (0.60, "🟡"), (0.55, "🟠"), (None, "🔴")],
        "rft_reward_gap_mean_same_phase":      [(0.05, "✅"), (0.02, "🟡"), (0.00, "🟠"), (None, "🔴")],
        "rft_reward_gap_mean_temporal_shift":  [(0.05, "✅"), (0.02, "🟡"), (0.00, "🟠"), (None, "🔴")],
        "rft_reward_gap_mean_action_noise":    [(0.05, "✅"), (0.02, "🟡"), (0.00, "🟠"), (None, "🔴")],
        "rft_reward_gap_min": [(0.00, "✅"), (None, "🟠")],
    }

    if key in lower_better:
        for threshold, emoji in lower_better[key]:
            if threshold is None or val <= threshold:
                return emoji
        return "🔴"
    if key in higher_better:
        for threshold, emoji in higher_better[key]:
            if threshold is None or val >= threshold:
                return emoji
        return "🔴"
    return "—"


def generate_report(metrics: dict, ckpt: str, task_suite: str, out_path: Path) -> None:
    today = datetime.date.today().isoformat()
    n_windows = metrics.get("num_windows", metrics.get("n_windows", "?"))

    lines: list[str] = []
    num_seeds = metrics.get("num_seeds")
    seeds_list = metrics.get("seeds")
    seed_info = ""
    if num_seeds and num_seeds > 1 and seeds_list:
        seed_info = f"  \n**シード**: {seeds_list}（{num_seeds}シード平均 ± 標準偏差）"

    lines += [
        f"# AR-Pixel World Model 評価結果レポート（LIBERO-{task_suite.capitalize()}）",
        "",
        f"**評価日時**: {today}  ",
        f"**モデル**: AR-Pixel World Model (Phase0 baseline)  ",
        f"**チェックポイント**: `{ckpt}`  ",
        f"**評価ウィンドウ数**: {n_windows}{seed_info}  ",
        f"**タスクスイート**: LIBERO-{task_suite.capitalize()}  ",
        "",
        "---",
        "",
    ]

    # ── Section 1: 指標説明テーブル ────────────────────────────────────────
    lines += [
        "## 1. 評価指標の説明と目安値",
        "",
        "> **凡例**: ✅ 優秀 ／ 🟡 良好 ／ 🟠 可 ／ 🔴 要改善 ／ — 適用外",
        "",
    ]

    for group_name, keys in DISPLAY_GROUPS:
        lines += [f"### {group_name}", ""]
        lines += [
            "| 指標名（内部キー） | 表示名 | 向き | 意味 | 目安値 |",
            "|:---|:---|:---:|:---|:---|",
        ]
        for key in keys:
            if key not in METRIC_META:
                continue
            disp, unit, meaning, thresholds = METRIC_META[key]
            lines.append(f"| `{key}` | {disp} | {unit} | {meaning} | {thresholds} |")
        lines.append("")

    # ── Section 2: 評価結果 ────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 2. 評価結果",
        "",
    ]

    for group_name, keys in DISPLAY_GROUPS:
        lines += [f"### {group_name}", ""]
        lines += [
            "| 指標 | 値 | 評価 |",
            "|:---|---:|:---:|",
        ]
        for key in keys:
            v = metrics.get(key)
            std = metrics.get(f"{key}_std")
            disp = METRIC_META.get(key, (key, "", "", ""))[0]
            lines.append(f"| {disp} | {_fmt(v, std)} | {_judge(key, v)} |")
        lines.append("")

    # ── Section 3: RFT適用性サマリー ──────────────────────────────────────
    lpips = metrics.get("horizon_avg_lpips")
    gap = metrics.get("rft_reward_gap", metrics.get("rft_reward_gap_mean"))
    pairwise = metrics.get("pairwise_acc_rft", metrics.get("pairwise_acc"))
    proxy = metrics.get("rft_reward_proxy")
    lpips_std   = metrics.get("horizon_avg_lpips_std")
    gap_std     = metrics.get("rft_reward_gap_std")
    pairwise_std = metrics.get("pairwise_acc_rft_std")

    lines += [
        "---",
        "",
        "## 3. RFT事後学習への適用性評価",
        "",
        "| 評価観点 | 値 | 判定 | コメント |",
        "|:---|---:|:---:|:---|",
    ]

    # Reconstruction quality
    lpips_ok = isinstance(lpips, float) and lpips < 0.25
    lines.append(
        f"| 再構成品質（LPIPS < 0.25） | {_fmt(lpips, lpips_std)} | {'✅' if lpips_ok else '🔴'} | "
        f"{'十分な予測品質' if lpips_ok else 'フレーム品質が低い。チェックポイントや学習設定を確認'} |"
    )

    # RFT reward gap
    gap_ok = isinstance(gap, float) and gap > 0.02
    lines.append(
        f"| 報酬ギャップ（gap > 0.02） | {_fmt(gap, gap_std)} | {'✅' if gap_ok else '🔴'} | "
        f"{'正誤アクションを識別できており、RFT学習信号として有効' if gap_ok else 'ギャップが小さく、RFT信号として弱い'} |"
    )

    # Pairwise accuracy (pooled)
    pairwise_ok = isinstance(pairwise, float) and pairwise > 0.60
    lines.append(
        f"| ペアワイズ精度 — 平均（> 60%） | {_fmt(pairwise, pairwise_std)} | {'✅' if pairwise_ok else '🔴'} | "
        f"{'過半数ウィンドウで正誤を識別' if pairwise_ok else '識別頻度が低い。窓サンプリングや負例生成を再検討'} |"
    )

    # Per-type pairwise accuracy
    _per_type = [
        ("same_phase",     "同フェーズ負例"),
        ("temporal_shift", "時間シフト負例"),
        ("action_noise",   "アクションノイズ負例"),
    ]
    for _neg_type, _neg_label in _per_type:
        _pa_key = f"pairwise_acc_rft_{_neg_type}"
        _gap_key = f"rft_reward_gap_mean_{_neg_type}"
        _pa_v   = metrics.get(_pa_key)
        _gap_v  = metrics.get(_gap_key)
        _pa_ok  = isinstance(_pa_v, float) and not math.isnan(_pa_v) and _pa_v > 0.60
        _pa_str = _fmt(_pa_v)
        _gap_str = _fmt(_gap_v)
        lines.append(
            f"| ペアワイズ精度 — {_neg_label} | {_pa_str} (gap: {_gap_str}) | "
            f"{'✅' if _pa_ok else ('🔴' if isinstance(_pa_v, float) and not math.isnan(_pa_v) else '—')} | "
            f"{'識別良好' if _pa_ok else ('識別不十分' if isinstance(_pa_v, float) and not math.isnan(_pa_v) else 'データなし')} |"
        )

    # Overall judgment
    all_ok = lpips_ok and gap_ok and pairwise_ok
    lines += [
        "",
        f"**総合判定**: {'✅ RFT事後学習への適用を推奨' if all_ok else '🔴 改善が必要。上記の指摘点を確認してください'}",
        "",
    ]

    # ── Section 4: 詳細コメント ────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 4. 評価設定",
        "",
        "| 項目 | 値 |",
        "|:---|:---|",
        f"| タスクスイート | LIBERO-{task_suite.capitalize()} |",
        f"| 評価ウィンドウ数 | {n_windows} |",
        f"| 予測ホライズン | {metrics.get('eval_horizon', 8)} フレーム |",
        "| GT動的マスク閾値 | 0.05 |",
        "| GT動的マスク膨張kernel | 7 |",
        "| ROI半径 | 40 px（roi_coords_v1.json から取得） |",
        "| 負例サンプリング | same_phase / temporal_shift / action_noise / mixed |",
        f"| チェックポイント | `{ckpt}` |",
        f"| 評価日 | {today} |",
        "",
        "---",
        "",
        "> このレポートは `analysis/worldmodel/report_ja.py` で自動生成されました。",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--task-suite", default="spatial")
    args = ap.parse_args()

    data = json.loads(Path(args.metrics_file).read_text(encoding="utf-8"))
    metrics = data.get("metrics", data)

    generate_report(metrics, args.ckpt, args.task_suite, Path(args.output))


if __name__ == "__main__":
    main()
