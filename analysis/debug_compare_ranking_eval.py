#!/usr/bin/env python3
"""Replay-style diagnostic for Phase 0 vs Phase 1 ranking outputs.

This tool aligns saved Phase 0 ranking JSONL records with Phase 1
ranking_by_window.csv rows where possible. It is deliberately conservative:
if the two evaluators were run on different model families or do not expose
shared window ids, it reports the mismatch instead of inventing equality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_phase0_items(root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("ranking_eval/*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for item in rec.get("per_item", []):
                scores = item.get("scores", [])
                correct = -scores[0] if len(scores) > 0 else math.nan
                shuffled = -scores[1] if len(scores) > 1 else math.nan
                items.append({
                    "task_id": rec.get("task_index"),
                    "task_name": rec.get("task_name", ""),
                    "window_id": item.get("item_id"),
                    "phase0_correct_lpips": correct,
                    "phase0_shuffled_lpips": shuffled,
                    "phase0_gap": shuffled - correct if correct == correct and shuffled == shuffled else math.nan,
                    "phase0_win": item.get("win"),
                    "phase0_source": str(p),
                })
    return items


def _read_phase1_rows(root_or_csv: Path) -> List[Dict[str, Any]]:
    p = root_or_csv
    if p.is_dir():
        p = p / "ranking_by_window.csv"
    if not p.exists():
        return []
    with p.open() as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        def fnum(k):
            try:
                return float(r.get(k, "nan"))
            except Exception:
                return math.nan
        win_raw = r.get("pairwise_win", "")
        out.append({
            "task_id": r.get("task_index", ""),
            "task_name": r.get("task_name", ""),
            "window_id": r.get("window_id", ""),
            "frame_indices": r.get("frame_indices", ""),
            "action_indices": r.get("action_indices", ""),
            "phase1_correct_lpips": fnum("correct_lpips"),
            "phase1_shuffled_lpips": fnum("shuffled_lpips"),
            "phase1_gap": fnum("lpips_gap"),
            "phase1_win": str(win_raw).lower() in {"1", "true", "yes"},
        })
    return out


def _phase1_root_from_config(config_path: str) -> Path:
    p = Path(config_path)
    if p.is_file():
        if p.name == "config_used.json" or p.name == "eval_protocol_config.json":
            return p.parent
        return p
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-results", required=True)
    ap.add_argument("--phase1-config", required=True,
                    help="Phase1 result dir, config_used.json, eval_protocol_config.json, or ranking_by_window.csv")
    ap.add_argument("--world-model-ckpt", default="")
    ap.add_argument("--out-dir", default="results/phase1/residual_worldmodel/protocol_audit")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p0_rows = _read_phase0_items(Path(args.phase0_results))
    p1_root = _phase1_root_from_config(args.phase1_config)
    p1_rows = _read_phase1_rows(p1_root)

    n = max(len(p0_rows), len(p1_rows))
    replay_rows = []
    same_win_count = 0
    comparable = 0
    for i in range(n):
        p0 = p0_rows[i] if i < len(p0_rows) else {}
        p1 = p1_rows[i] if i < len(p1_rows) else {}
        p0c = p0.get("phase0_correct_lpips", math.nan)
        p0s = p0.get("phase0_shuffled_lpips", math.nan)
        p1c = p1.get("phase1_correct_lpips", math.nan)
        p1s = p1.get("phase1_shuffled_lpips", math.nan)
        p0w = p0.get("phase0_win")
        p1w = p1.get("phase1_win")
        same = (bool(p0w) == bool(p1w)) if p0 and p1 else False
        if p0 and p1:
            comparable += 1
            same_win_count += int(same)
        replay_rows.append({
            "task_id": p1.get("task_id", p0.get("task_id", "")),
            "window_id": p1.get("window_id", p0.get("window_id", "")),
            "frame_indices": p1.get("frame_indices", ""),
            "action_indices": p1.get("action_indices", ""),
            "phase0_correct_lpips": p0c,
            "phase0_shuffled_lpips": p0s,
            "phase0_gap": p0.get("phase0_gap", math.nan),
            "phase0_win": p0w,
            "phase1_correct_lpips": p1c,
            "phase1_shuffled_lpips": p1s,
            "phase1_gap": p1.get("phase1_gap", math.nan),
            "phase1_win": p1w,
            "abs_diff_correct_lpips": abs(p0c - p1c) if p0c == p0c and p1c == p1c else math.nan,
            "abs_diff_shuffled_lpips": abs(p0s - p1s) if p0s == p0s and p1s == p1s else math.nan,
            "same_win_label": same,
        })

    csv_path = out / "ranking_eval_replay.csv"
    if replay_rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(replay_rows[0].keys()))
            w.writeheader()
            w.writerows(replay_rows)

    summary = {
        "phase0_rows": len(p0_rows),
        "phase1_rows": len(p1_rows),
        "aligned_by": "row order only; exact same-window replay requires evaluator-side shared window manifest",
        "comparable_rows": comparable,
        "same_win_label_fraction": same_win_count / comparable if comparable else None,
        "world_model_ckpt": args.world_model_ckpt,
        "note": (
            "This diagnostic compares persisted ranking outputs. It does not "
            "run the AR-token Phase0 model through the PixelResidual evaluator."
        ),
    }
    (out / "ranking_eval_replay_summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Ranking Eval Replay Diagnostic",
        "",
        f"- Phase0 rows: `{len(p0_rows)}`",
        f"- Phase1 rows: `{len(p1_rows)}`",
        f"- Comparable rows: `{comparable}`",
        f"- Same win-label fraction: `{summary['same_win_label_fraction']}`",
        "",
        "## Caveat",
        summary["note"],
        "",
        "For exact replay, run Phase1 with `PHASE0_COMPATIBLE=1` so `ranking_by_window.csv` contains frame/action indices, then compare against a Phase0 run with the same saved window manifest.",
    ]
    (out / "ranking_eval_replay.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out / 'ranking_eval_replay_summary.json'}")
    print(f"Wrote {out / 'ranking_eval_replay.md'}")


if __name__ == "__main__":
    main()
