#!/usr/bin/env python3
"""Compare Phase 0 and Phase 1 world-model evaluation protocols.

The script intentionally combines saved config/metrics inspection with a small
static protocol table. Older Phase 0 runs did not persist every evaluator arg,
so the static table records the protocol used by the checked-in scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _find_first(root: Path, names: List[str]) -> Optional[Path]:
    if root.is_file():
        return root
    for name in names:
        p = root / name
        if p.exists():
            return p
    for p in sorted(root.rglob("*")):
        if p.name in names:
            return p
    return None


def _phase0_metrics(root: Path) -> Dict[str, Any]:
    summary = _load_json(root / "summary.json")
    wm = summary.get("worldmodel", {})
    ranking = wm.get("ranking", {})
    return {
        "pairwise_acc": ranking.get("pairwise_acc"),
        "num_wins": ranking.get("num_wins"),
        "num_losses": ranking.get("num_losses"),
        "lpips_gap_mean": None,
        "source": str(root / "summary.json") if summary else None,
    }


def _phase1_metrics(root: Path) -> Dict[str, Any]:
    p = _find_first(root, ["aggregate_metrics.json"])
    raw = _load_json(p) if p else {}
    metrics = raw.get("metrics", raw)
    return {
        "pairwise_acc": metrics.get("pairwise_acc"),
        "num_windows": metrics.get("num_windows"),
        "num_ranking_windows": metrics.get("num_ranking_windows"),
        "lpips_gap_mean": metrics.get("lpips_gap"),
        "lpips_gap_min": metrics.get("lpips_gap_min"),
        "source": str(p) if p else None,
    }


def _phase1_protocol(root: Path) -> Dict[str, Any]:
    p = _find_first(root, ["eval_protocol_config.json", "config_used.json"])
    raw = _load_json(p) if p else {}
    return raw.get("eval_protocol", raw)


def _protocol_rows(phase1: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Phase 0 values are from scripts/libero/run_phase0_eval.sh and
    # worldmodel/libero/visualize.py as of this repository state.
    rows = [
        ("model family", "AR-token Pixel WorldModel", "PixelResidualWorldModel", "high",
         "A Phase1 pixel checkpoint is not the same checkpoint/model as Phase0 AR-Pixel."),
        ("checkpoint", "checkpoints/libero/WorldModel/<suite>/<exp>", phase1.get("checkpoint_path"), "high",
         "Must be identical to reproduce Phase0 numbers."),
        ("task filtering", "episode file name matched to selected task", "native: not filtered; compat: filtered by task filename", "high",
         "Unfiltered Phase1 can evaluate the same dataset stream for every task id."),
        ("selected tasks", "parse_task_indices; empty means all tasks", phase1.get("selected_task_indices"), "medium",
         "Use same task list as the Phase0 run being compared."),
        ("window sampling", "random one window per selected episode, round-robin by task", "native: sliding windows from first 50 episodes; compat: random one per matched episode", "high",
         "Changes negative pool and difficulty distribution."),
        ("split/heldout", "fallback_all: heldout hash split then all if empty", phase1.get("split_mode"), "medium",
         "Phase1 still uses train stream; compatible mode records this limitation."),
        ("eval horizon", 7, phase1.get("effective_eval_horizon", phase1.get("eval_horizon")), "medium",
         "Should match exactly."),
        ("action offset", 0, phase1.get("action_start_offset"), "high",
         "Native Phase1 model consumes pixels[1]/actions[1]; compat prepends one frame/action."),
        ("current/target frame", "current=frame[start], target=frames[start+1:start+H+1]", phase1.get("frame_index_alignment"), "high",
         "Frame mismatch directly changes LPIPS and ranking labels."),
        ("negative generation", "same-task other-window action sequence", phase1.get("negative_type"), "high",
         "Temporal permutation inside the same window is not the Phase0 negative."),
        ("ranking LPIPS frame", "average over all horizon frames", "terminal only" if phase1.get("use_terminal_frame_only") else "horizon average", "high",
         "Terminal-only scoring has a different scale and variance."),
        ("gap definition", "shuffled_lpips - correct_lpips", phase1.get("ranking_gap_definition"), "high",
         "Positive should mean GT action is better."),
        ("pairwise unit", "per window", phase1.get("pairwise_unit"), "high",
         "Do not compare after task/global averaging first."),
        ("LPIPS input", "uint8 -> [0,1] -> [-1,1]", phase1.get("lpips_input_range"), "medium",
         "Both paths should feed LPIPS in [-1,1]."),
        ("gripper ROI", "motion COM current to final GT, roi_half from roi config", phase1.get("gripper_roi_method"), "low",
         "Affects ROI metrics more than pairwise_acc."),
    ]
    return [
        {
            "item": item,
            "phase0": p0,
            "phase1": p1,
            "impact": impact,
            "note": note,
            "match": str(p0) == str(p1),
        }
        for item, p0, p1, impact, note in rows
    ]


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-results", required=True)
    ap.add_argument("--phase1-results", required=True)
    ap.add_argument("--out-dir", default="results/phase1/residual_worldmodel/protocol_audit")
    args = ap.parse_args()

    p0_root = Path(args.phase0_results)
    p1_root = Path(args.phase1_results)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p1_protocol = _phase1_protocol(p1_root)
    rows = _protocol_rows(p1_protocol)
    high = [r for r in rows if r["impact"] == "high" and not r["match"]]
    payload = {
        "phase0_results": str(p0_root),
        "phase1_results": str(p1_root),
        "phase0_metrics": _phase0_metrics(p0_root),
        "phase1_metrics": _phase1_metrics(p1_root),
        "phase1_protocol": p1_protocol,
        "diffs": rows,
        "high_impact_diffs": high,
    }
    (out / "eval_protocol_diff.json").write_text(json.dumps(payload, indent=2, default=str))
    _write_csv(rows, out / "eval_protocol_diff.csv")

    lines = [
        "# Phase 0 vs Phase 1 Eval Protocol Diff",
        "",
        "## Phase 0 Metrics",
        f"- pairwise_acc: `{payload['phase0_metrics'].get('pairwise_acc')}`",
        f"- wins/losses: `{payload['phase0_metrics'].get('num_wins')}` / `{payload['phase0_metrics'].get('num_losses')}`",
        "",
        "## Phase 1 Metrics",
        f"- pairwise_acc: `{payload['phase1_metrics'].get('pairwise_acc')}`",
        f"- num_windows: `{payload['phase1_metrics'].get('num_windows')}`",
        f"- num_ranking_windows: `{payload['phase1_metrics'].get('num_ranking_windows')}`",
        f"- lpips_gap_mean: `{payload['phase1_metrics'].get('lpips_gap_mean')}`",
        "",
        "## Diff Table",
        "| Item | Phase 0 | Phase 1 | Impact | Match |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(f"| {r['item']} | {r['phase0']} | {r['phase1']} | {r['impact']} | {r['match']} |")
    lines += [
        "",
        "## Pairwise-Acc Sensitive Diffs",
    ]
    if high:
        lines += [f"- `{r['item']}`: {r['note']}" for r in high]
    else:
        lines.append("- No high-impact mismatch detected from persisted config.")
    lines += [
        "",
        "## 修正方針",
        "- Phase 1 native mode は既存結果の再現用として維持する。",
        "- `PHASE0_COMPATIBLE=1` では same-task other-window negative、horizon 平均 LPIPS、Phase0 互換 frame/action alignment を使う。",
        "- Phase0 の AR-token WorldModel と Phase1 の PixelResidualWorldModel はモデルファミリが違うため、完全再現には Phase0 evaluator/checkpoint を使う必要がある。",
    ]
    (out / "eval_protocol_diff.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out / 'eval_protocol_diff.md'}")
    print(f"Wrote {out / 'eval_protocol_diff.json'}")


if __name__ == "__main__":
    main()
