# LIBERO WorldModel Evaluation Protocol v1

Defines what can and cannot be quantitatively compared between the
**baseline worldmodel** (`worldmodel/libero/visualize.py`, `visualize_base.py`)
and the **residual worldmodel** (`worldmodel/residual_worldmodel/focused_visualize.py`).

---

## 1. Metric Comparison Matrix

### 1-A. Full-image reconstruction (COMPARABLE — identical keys)

| Metric key | Baseline | Residual | Notes |
|---|---|---|---|
| `future_image_smooth_l1` | ✓ | ✓ | Primary recon loss proxy |
| `future_image_l1` | ✓ | ✓ | |
| `future_image_mse` | ✓ | ✓ | Written by visualize_base.py |
| `future_image_lpips` | ✓ | ✓ | |

### 1-B. ROI reconstruction (COMPARABLE — identical keys, asymmetric ROI center source)

| Metric key | Baseline source | Residual source | Comparable? |
|---|---|---|---|
| `roi/gripper_mse` | motion COM of `\|frame_T - frame_0\|` | write_mask COM (focus map) | **Yes** — same metric, different center proxy |
| `roi/gripper_lpips` | same | same | **Yes** |
| `roi/gripper_psnr` | same | same | **Yes** |
| `roi/gripper_ssim` | same | same | **Yes** |
| `roi/goal_mse` | `roi_coords_v1.json` (fixed) | `roi_coords_v1.json` (fixed) | **Yes** — identical |
| `roi/goal_lpips` | same | same | **Yes** |
| `roi/goal_psnr` | same | same | **Yes** |
| `roi/goal_ssim` | same | same | **Yes** |
| `roi/multi_step_gripper_mse` | list, per-frame | list, per-frame | **Yes** — compare drift_ratio |
| `roi/multi_step_goal_mse` | list, per-frame | list, per-frame | **Yes** |

> **Asymmetry note**: The gripper ROI center differs between models.
> Baseline uses a pixel-level motion COM (heuristic); residual uses the learned
> write_mask COM. A larger gripper ROI error for the baseline may reflect both
> poorer reconstruction *and* a less accurate ROI center — interpret with caution.

### 1-C. Ranking metrics (PARTIALLY COMPARABLE)

| Metric key | Baseline | Residual | Notes |
|---|---|---|---|
| `pairwise_acc` | ✓ K=2 | ✓ K=6 | **Directionally comparable** but K differs; tiered is harder |
| `mean_margin` | ✓ | ✓ | Same caveat as pairwise_acc |
| `strict_order_acc` | ✗ null | ✓ | Residual-only (requires 3-tier candidates) |
| `acc_success_gt_nearsuccess` | ✗ null | ✓ | Residual-only |
| `acc_nearsuccess_gt_failure` | ✗ null | ✓ | Residual-only |
| `spearman_tier_corr` | ✗ null | ✓ | Residual-only |
| `margin_success_minus_failure` | ✗ null | ✓ | Residual-only |
| `tier_score_success/nearsuccess/failure` | ✗ null | ✓ | Residual-only |

> `metric_family` column in `worldmodel_eval_ranking_history.csv`:
> - `"pairwise"` → K=2, baseline convention
> - `"tiered"`   → K≥4, residual convention
>
> **Do not average or directly compare `pairwise_acc` between families.**
> Use them as within-family rankings only.

### 1-D. Focus / attention metrics (NOT COMPARABLE)

| Metric key | Baseline | Residual | Notes |
|---|---|---|---|
| `focus_mean` | ✗ | ✓ | Residual-only (write_mask) |
| `focus_entropy` | ✗ | ✓ | Residual-only |
| `iou_vs_change` | ✗ | ✓ | Residual-only |
| `dice_vs_change` | ✗ | ✓ | Residual-only |
| `dino_feature_mse` | ✗ | ✓ | Residual-only (DINO loss component) |
| `dino_cosine_similarity` | ✗ | ✓ | Residual-only |

### 1-E. Score breakdown (NOT COMPARABLE)

| Key | Baseline | Residual |
|---|---|---|
| `score_breakdown/dino_cosine` | null | ✓ |
| `score_breakdown/image_l1` | null | ✓ |
| `score_breakdown/combined` | null | ✓ |

---

## 2. ROI Center Sources

| Model | Gripper center | Goal center |
|---|---|---|
| Baseline | `motion_com_np(frame_0, frame_T)` — pixel-level motion COM | `configs/libero/roi_coords_v1.json` (fixed per task) |
| Residual | `_batch_com(write_mask)` — learned focus map COM | Same `roi_coords_v1.json` + optional `cfg.goal_roi_y` override |

**Debug output**: run with `--enable-roi-metrics --debug-roi` on baseline to save
`task{N}_*_roi_debug.png` — shows gripper (blue) and goal (orange) ROI boxes
overlaid on the first GT frame. Verify coordinates before comparing ROI metrics.

---

## 3. JSONL Ranking Schema

Both models write to `rank_eval/rank_eval_candidates.jsonl` using
`worldmodel/eval_roi_utils.append_ranking_jsonl()`.

Fixed keys (null for unavailable model):
```
RANKING_METRIC_KEYS (17 keys):
  strict_order_acc, pairwise_acc, top1_acc, mean_margin,
  pos_score_mean, neg_score_mean, hardest_negative_margin,
  acc_success_gt_nearsuccess, acc_nearsuccess_gt_failure, acc_success_gt_failure,
  spearman_tier_corr,
  margin_success_minus_nearsuccess, margin_nearsuccess_minus_failure,
  margin_success_minus_failure,
  tier_score_success, tier_score_nearsuccess, tier_score_failure

SCORE_BREAKDOWN_KEYS (3 keys):
  dino_cosine, image_l1, combined
```

`task_suite` field is populated for residual (from training args) and for
baseline via `_emit_baseline_ranking_jsonl`.

---

## 4. Multi-step Drift

List metrics (`roi/multi_step_gripper_mse`, etc.) are expanded to scalars in
`analyze_worldmodel_eval.py`:

| Scalar suffix | Meaning |
|---|---|
| `_first` | Error at frame 1 (shortest horizon) |
| `_last`  | Error at final frame (longest horizon) |
| `_max`   | Peak error across horizon |
| `_drift_ratio` | `last / (first + 1e-8)` — >1.3 indicates transport drift |

Compare `drift_ratio` across models: a residual model with better focus should
show lower gripper drift than baseline.

---

## 5. What to Verify Before Progress/Rerank Implementation

In order of priority:

1. **ROI center sanity** — open `task*_roi_debug.png` for 2–3 tasks; confirm
   the orange (goal) box overlaps the relevant region. If not, update
   `configs/libero/roi_coords_v1.json`.

2. **JSONL schema completeness** — every line must have all 17
   `RANKING_METRIC_KEYS`; null is acceptable, absence is not. Run:
   ```bash
   python -c "
   import json
   from worldmodel.eval_roi_utils import RANKING_METRIC_KEYS
   with open('evals/.../rank_eval/rank_eval_candidates.jsonl') as f:
       for line in f:
           r = json.loads(line)
           missing = [k for k in RANKING_METRIC_KEYS if k not in r['metrics']]
           assert not missing, f'Missing: {missing}'
   print('OK')
   "
   ```

3. **task_suite in JSONL** — confirm `task_suite != ""` in every residual record.

4. **pairwise_acc baseline vs residual** — if baseline ≤ 0.55 (near random),
   the baseline worldmodel has no meaningful ranking signal. This is the key
   diagnostic for why RFT didn't improve.

5. **drift_ratio** — if `roi/multi_step_gripper_mse_drift_ratio > 1.5` for
   baseline but < 1.2 for residual, the focus mechanism genuinely helps
   long-horizon fidelity.

---

## 6. Known Asymmetries (still acceptable)

| Asymmetry | Impact | Mitigation |
|---|---|---|
| Gripper ROI center differs (motion COM vs write_mask COM) | Baseline gripper ROI may be noisier | Interpret directionally; compare drift_ratio |
| Baseline ranking K=2, residual K=6 | pairwise_acc not directly comparable | Use `metric_family` column to separate groups |
| `visualize_base.py` is per-episode (not batch eval) | Baseline ROI metrics are single-episode, not windowed | Accept as approximation; baseline lacks sliding-window eval |
| goal ROI coordinates are heuristic estimates | Absolute goal ROI error values are approximate | Verify via `--debug-roi` PNG; update roi_coords_v1.json if needed |
