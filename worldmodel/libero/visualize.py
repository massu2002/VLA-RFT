from __future__ import annotations

import argparse
import gc
import re
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lpips
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import torch
from safetensors.torch import load_file as load_safetensors
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from libero.libero import benchmark

from ..datasets.libero.data import resolve_dataset_name
from ..core.model import WorldModelTrainer as LiberoWorldModelTrainer, WorldModelRuntimeConfig


# =========================================================
# Parser
# =========================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LIBERO world model quality with held-out token loss, "
            "single-pass rollout fidelity, per-task summaries, and full-episode rollouts."
        )
    )
    parser.add_argument(
        "--task-suite",
        type=str,
        choices=["spatial", "object", "goal", "10"],
        required=True,
    )
    parser.add_argument("--data-root", type=str, default="data/modified_libero_rlds")

    parser.add_argument(
        "--base-model-root",
        type=str,
        default="checkpoints/libero/WorldModel",
    )
    parser.add_argument(
        "--trained-model-root",
        type=str,
        default="checkpoints/libero/WorldModel",
    )
    parser.add_argument("--trained-exp-name", type=str, default="worldmodel_scratch")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument(
        "--trained-model-dir",
        type=str,
        default="",
        help=(
            "Direct path to a fully exported trained model dir. "
            "If set, date/trained-exp-name are ignored."
        ),
    )
    parser.add_argument("--visual-tokenizer", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Main evaluation controls
    parser.add_argument("--num-eval-windows", type=int, default=100)
    parser.add_argument("--eval-horizon", type=int, default=7)
    parser.add_argument("--heldout-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["heldout", "all", "fallback_all"],
        default="heldout",
        help=(
            "Window-evaluation split mode. "
            "'fallback_all' tries heldout first, then falls back to all "
            "if no candidate episodes are found."
        ),
    )
    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=2,
        help="Number of generated future frames to detokenize at once",
    )
    parser.add_argument("--display-frames", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)

    # Task subset selection inside a suite
    parser.add_argument(
        "--task-indices",
        type=str,
        default="",
        help='Comma-separated task indices within the suite, e.g. "0,1,4". Empty means all tasks.',
    )

    # Optional diagnostics
    parser.add_argument("--compare-base", action="store_true", default=False)
    parser.add_argument("--run-action-sensitivity", action="store_true", default=False)
    parser.add_argument("--run-diagnostic-chunk", action="store_true", default=False)
    parser.add_argument("--chunk-future-length", type=int, default=7)

    # Casebook
    parser.add_argument("--save-casebook-count", type=int, default=5)

    # Full-episode rollout controls
    parser.add_argument(
        "--full-episode-split-mode",
        type=str,
        choices=["heldout", "all", "fallback_all"],
        default="fallback_all",
        help=(
            "Split mode for full-episode visualization. "
            "'fallback_all' tries heldout first, then falls back to all."
        ),
    )
    parser.add_argument("--num-full-episodes-per-task", type=int, default=1)
    parser.add_argument("--full-episode-index", type=int, default=0)
    parser.add_argument("--full-episode-display-cols", type=int, default=6)
    parser.add_argument("--save-full-episode-frames", action="store_true", default=False)

    return parser


# =========================================================
# Data containers
# =========================================================


@dataclass
class EvalWindow:
    task_name: str
    task_index: int
    episode_file: str
    start: int
    frames: List[np.ndarray]   # len = horizon + 1
    actions: List[np.ndarray]  # len = horizon


# =========================================================
# Basic helpers
# =========================================================


def _suite_key(task_suite: str) -> str:
    return f"libero_{task_suite}"


def get_task_names(task_suite: str) -> List[str]:
    benchmarks = benchmark.get_benchmark_dict()
    suite_key = _suite_key(task_suite)
    if suite_key not in benchmarks:
        raise ValueError(f"Unsupported task suite: {task_suite}")
    return benchmarks[suite_key]().get_task_names()


def parse_task_indices(args: argparse.Namespace) -> List[int]:
    task_names = get_task_names(args.task_suite)
    if not args.task_indices.strip():
        return list(range(len(task_names)))

    out: List[int] = []
    for x in args.task_indices.split(","):
        x = x.strip()
        if not x:
            continue
        idx = int(x)
        if idx < 0 or idx >= len(task_names):
            raise ValueError(f"task index out of range: {idx}")
        out.append(idx)
    return sorted(set(out))


def _decode_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _experiment_label(args: argparse.Namespace, trained_model_dir: str) -> str:
    if args.trained_model_dir:
        return _slugify(Path(trained_model_dir).name)
    date_prefix = args.date or ""
    exp_name = (
        f"{date_prefix}_{args.trained_exp_name}"
        if date_prefix
        else args.trained_exp_name
    )
    return _slugify(exp_name)


def _device_for_run(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _cleanup_task_memory(*objs) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# =========================================================
# Model loading
# =========================================================


def _load_model(
    model_dir: str,
    tokenizer_dir: str,
    device: torch.device,
) -> LiberoWorldModelTrainer:
    dtype = torch.bfloat16 if device.type == "cuda" else None
    model = LiberoWorldModelTrainer(
        lm_path=model_dir,
        visual_tokenizer_path=tokenizer_dir,
        runtime_cfg=WorldModelRuntimeConfig(
            action_ranges_path="train/verl/ivideogpt/configs/libero_action_ranges.pth",
            tokenizer_micro_batch_size=1,
            context_length=1,
            action_dim=7,
            action_bins=256,
            max_length=1663,
            visual_token_num=-1,
            bos_token_id=9006,
            eos_token_id=9007,
            pad_token_id=9007,
            tokens_per_frame=64,
            processor_type="ctx_msp",
            use_img_gt_ac=False,
            interact=False,
        ),
        torch_dtype=dtype,
        gradient_checkpointing=False,
        load_pretrained_weights=True,
    )
    model.runtime_cfg.visual_token_num = int(model.visual_tokenizer.num_vq_embeddings)
    model.processor.config.visual_token_num = model.runtime_cfg.visual_token_num
    model.to(device)
    model.eval()
    return model


def _extract_lm_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if any(k.startswith("lm.") for k in state_dict.keys()):
        return {k[3:]: v for k, v in state_dict.items() if k.startswith("lm.")}
    if any(k.startswith("model.") for k in state_dict.keys()):
        return {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    return {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("visual_tokenizer.")
    }


def _load_trained_model(
    base_model_dir: str,
    trained_model_dir: str,
    tokenizer_dir: str,
    device: torch.device,
) -> LiberoWorldModelTrainer:
    model = _load_model(base_model_dir, tokenizer_dir, device)

    weight_path = Path(trained_model_dir) / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Trained weight not found: {weight_path}")

    state_dict = load_safetensors(str(weight_path))
    state_dict = _extract_lm_state_dict(state_dict)

    missing, unexpected = model.lm.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[warn] Missing keys when loading trained weights: {len(missing)}")
        print(missing[:20])
    if unexpected:
        print(f"[warn] Unexpected keys when loading trained weights: {len(unexpected)}")
        print(unexpected[:20])

    model.to(device)
    model.eval()
    return model


def _resolve_trained_model_dir(args: argparse.Namespace) -> str:
    if args.trained_model_dir:
        trained_dir = Path(args.trained_model_dir)
    else:
        date_prefix = args.date or ""
        exp_name = (
            f"{date_prefix}_{args.trained_exp_name}"
            if date_prefix
            else args.trained_exp_name
        )
        trained_dir = Path(args.trained_model_root) / args.task_suite / exp_name

    if not trained_dir.exists():
        raise FileNotFoundError(f"Trained model dir not found: {trained_dir}")
    if not trained_dir.is_dir():
        raise NotADirectoryError(
            f"Trained model path is not a directory: {trained_dir}"
        )
    if trained_dir.name.startswith("checkpoint-"):
        raise RuntimeError(
            f"Please point to the experiment root directory, not a checkpoint directory: {trained_dir}\n"
            f"Expected something like: .../{args.task_suite}/{args.date}_{args.trained_exp_name}"
        )

    return str(trained_dir)


# =========================================================
# Inference helpers
# =========================================================


@torch.no_grad()
def _prime_model(model: LiberoWorldModelTrainer, prompt_ids: torch.Tensor):
    outputs = model.lm(input_ids=prompt_ids, use_cache=True)
    return outputs.logits[:, -1, :], outputs.past_key_values


@torch.no_grad()
def _advance_model(
    model: LiberoWorldModelTrainer,
    token_ids: torch.Tensor,
    past_key_values,
):
    outputs = model.lm(
        input_ids=token_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return outputs.logits[:, -1, :], outputs.past_key_values


@torch.no_grad()
def _encode_context_prompt(
    model: LiberoWorldModelTrainer,
    context_frame: np.ndarray,
):
    """
    Current training processor uses:
      prefix_tokens = ctx_tokens
      step_tokens   = [action_tokens, future_dyn_tokens]
    So inference prompt should use ctx_tokens only.
    """
    device = next(model.lm.parameters()).device
    visual_token_num = model.runtime_cfg.visual_token_num

    ctx_pair = np.stack([context_frame, context_frame], axis=0)
    ctx_tensor = (
        torch.from_numpy(ctx_pair)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        .to(device=device)
        .float()
        / 255.0
    )

    ctx_tokens_raw, _ = model.visual_tokenizer.tokenize(ctx_tensor)
    ctx_tokens_raw = ctx_tokens_raw[:, :1]
    ctx_tokens_prompt = ctx_tokens_raw + visual_token_num
    prompt_ids = ctx_tokens_prompt.reshape(1, -1).long()
    return ctx_tokens_raw, prompt_ids


@torch.no_grad()
def _action_ids_from_action(
    model: LiberoWorldModelTrainer,
    action: np.ndarray,
):
    device = next(model.lm.parameters()).device
    visual_token_num = model.runtime_cfg.visual_token_num

    action_tensor = (
        torch.from_numpy(np.asarray(action, dtype=np.float32))
        .to(device=device)
        .view(1, 1, -1)
    )
    action_ids = model.processor._discretize_actions(
        action_tensor, model.runtime_cfg.action_bins
    )[0, 0]
    action_ids = action_ids + visual_token_num * 2
    return action_ids


@torch.no_grad()
def _generate_one_dyn_block(
    model: LiberoWorldModelTrainer,
    logits: torch.Tensor,
    past_key_values,
):
    visual_token_num = model.runtime_cfg.visual_token_num
    frame_tokens = []

    for _ in range(model.runtime_cfg.tokens_per_frame):
        allowed_logits = logits.clone()
        allowed_logits[:, visual_token_num:] = -torch.inf
        next_token = torch.argmax(allowed_logits, dim=-1, keepdim=True)
        frame_tokens.append(next_token)
        logits, past_key_values = _advance_model(
            model, next_token.long(), past_key_values
        )

    dyn_block = torch.cat(frame_tokens, dim=-1)
    return dyn_block, logits, past_key_values


@torch.no_grad()
def _decode_dyn_tokens_in_chunks(
    model: LiberoWorldModelTrainer,
    ctx_tokens_raw: torch.Tensor,
    dyn_tokens: torch.Tensor,
    decode_chunk_size: int,
) -> List[np.ndarray]:
    if decode_chunk_size <= 0:
        raise ValueError("decode_chunk_size must be positive")

    pred_frames: List[np.ndarray] = []
    total_t = dyn_tokens.shape[1]

    for start in range(0, total_t, decode_chunk_size):
        end = min(start + decode_chunk_size, total_t)
        dyn_chunk = dyn_tokens[:, start:end]

        recon = model.visual_tokenizer.detokenize(ctx_tokens_raw, dyn_chunk)
        recon = recon[:, 1:].clamp(0.0, 1.0)

        chunk_frames = [
            np.clip(
                frame.detach().cpu().permute(1, 2, 0).numpy() * 255.0,
                0,
                255,
            ).astype(np.uint8)
            for frame in recon[0]
        ]
        pred_frames.extend(chunk_frames)

        del dyn_chunk, recon, chunk_frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pred_frames


@torch.no_grad()
def rollout_episode(
    model: LiberoWorldModelTrainer,
    frames: List[np.ndarray],
    actions: List[np.ndarray],
    chunk_future_length: int,
    decode_chunk_size: int,
):
    """
    Diagnostic chunked rollout.
    Re-primes at each chunk using the last generated frame.
    """
    generated_frames: List[np.ndarray] = []
    context_frame = frames[0]
    action_cursor = 0

    if chunk_future_length <= 0:
        raise ValueError("chunk_future_length must be positive")

    while action_cursor < len(actions):
        chunk_actions = actions[action_cursor : action_cursor + chunk_future_length]
        if len(chunk_actions) == 0:
            break

        ctx_tokens_raw, prompt_ids = _encode_context_prompt(model, context_frame)
        logits, past = _prime_model(model, prompt_ids)

        dyn_blocks: List[torch.Tensor] = []

        for action in chunk_actions:
            action_ids = _action_ids_from_action(model, action)

            for token in action_ids.reshape(-1):
                token_tensor = token.view(1, 1).long()
                logits, past = _advance_model(model, token_tensor, past)

            dyn_block, logits, past = _generate_one_dyn_block(model, logits, past)
            dyn_blocks.append(dyn_block)

        dyn_tokens = torch.stack(dyn_blocks, dim=1)

        chunk_frames_uint8 = _decode_dyn_tokens_in_chunks(
            model=model,
            ctx_tokens_raw=ctx_tokens_raw,
            dyn_tokens=dyn_tokens,
            decode_chunk_size=decode_chunk_size,
        )
        generated_frames.extend(chunk_frames_uint8)

        if len(chunk_frames_uint8) > 0:
            context_frame = chunk_frames_uint8[-1]

        del (
            ctx_tokens_raw,
            prompt_ids,
            logits,
            past,
            dyn_blocks,
            dyn_tokens,
            chunk_frames_uint8,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        action_cursor += len(chunk_actions)

    gt_frames = frames[1 : 1 + len(generated_frames)]
    return generated_frames, gt_frames


@torch.no_grad()
def rollout_episode_single_pass(
    model: LiberoWorldModelTrainer,
    frames: List[np.ndarray],
    actions: List[np.ndarray],
    decode_chunk_size: int,
):
    """
    Paper-like single-pass rollout:
      - one initial image
      - full action sequence
      - no chunk re-priming
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames")
    if len(actions) != len(frames) - 1:
        raise ValueError(
            f"Expected len(actions) == len(frames)-1, got {len(actions)} and {len(frames)}"
        )

    ctx_tokens_raw, prompt_ids = _encode_context_prompt(model, frames[0])
    logits, past = _prime_model(model, prompt_ids)

    dyn_blocks = []

    for action in actions:
        action_ids = _action_ids_from_action(model, action)

        for token in action_ids.reshape(-1):
            token_tensor = token.view(1, 1).long()
            logits, past = _advance_model(model, token_tensor, past)

        dyn_block, logits, past = _generate_one_dyn_block(model, logits, past)
        dyn_blocks.append(dyn_block)

    dyn_tokens = torch.stack(dyn_blocks, dim=1)

    pred_frames = _decode_dyn_tokens_in_chunks(
        model=model,
        ctx_tokens_raw=ctx_tokens_raw,
        dyn_tokens=dyn_tokens,
        decode_chunk_size=decode_chunk_size,
    )
    gt_frames = frames[1 : 1 + len(pred_frames)]

    del ctx_tokens_raw, prompt_ids, logits, past, dyn_blocks, dyn_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_frames, gt_frames


# =========================================================
# Metrics
# =========================================================


@torch.no_grad()
def compute_sequence_metrics_all(
    gt_frames: List[np.ndarray],
    pred_frames: List[np.ndarray],
    lpips_model,
    device: torch.device,
):
    mses, psnrs, ssims, lpips_scores = [], [], [], []

    for gt, pred in zip(gt_frames, pred_frames):
        gt_f = gt.astype(np.float32) / 255.0
        pred_f = pred.astype(np.float32) / 255.0

        mse = float(np.mean((pred_f - gt_f) ** 2))
        psnr = float(peak_signal_noise_ratio(gt_f, pred_f, data_range=1.0))
        ssim = float(
            structural_similarity(gt_f, pred_f, channel_axis=2, data_range=1.0)
        )

        gt_t = (
            torch.from_numpy(gt)
            .to(device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        pred_t = (
            torch.from_numpy(pred)
            .to(device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        lp = float(lpips_model(pred_t * 2.0 - 1.0, gt_t * 2.0 - 1.0).item())

        mses.append(mse)
        psnrs.append(psnr)
        ssims.append(ssim)
        lpips_scores.append(lp)

    return {
        "mse_per_frame": np.asarray(mses, dtype=np.float32),
        "psnr_per_frame": np.asarray(psnrs, dtype=np.float32),
        "ssim_per_frame": np.asarray(ssims, dtype=np.float32),
        "lpips_per_frame": np.asarray(lpips_scores, dtype=np.float32),
        "avg_mse": float(np.mean(mses)) if len(mses) > 0 else float("nan"),
        "avg_psnr": float(np.mean(psnrs)) if len(psnrs) > 0 else float("nan"),
        "avg_ssim": float(np.mean(ssims)) if len(ssims) > 0 else float("nan"),
        "avg_lpips": float(np.mean(lpips_scores)) if len(lpips_scores) > 0 else float("nan"),
    }


def _summarize_values(values: List[float]) -> Dict:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "worst10_mean": float("nan"),
        }

    sorted_arr = np.sort(arr)
    k = max(1, math.ceil(0.1 * len(arr)))
    worst10 = sorted_arr[-k:]

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "worst10_mean": float(np.mean(worst10)),
    }


# =========================================================
# Window sampling
# =========================================================

def _episode_to_arrays(episode: Dict) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
    steps = list(episode["steps"])
    frames: List[np.ndarray] = []
    actions: List[np.ndarray] = []

    for step in steps:
        frames.append(np.asarray(step["observation"]["image"], dtype=np.uint8))
        actions.append(np.asarray(step["action"], dtype=np.float32))

    episode_file = _decode_bytes(episode["episode_metadata"]["file_path"])
    return frames, actions[:-1], episode_file


def _stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _is_heldout(file_path: str, heldout_ratio: float) -> bool:
    if heldout_ratio <= 0.0:
        return False
    if heldout_ratio >= 1.0:
        return True
    bucket = (_stable_hash_int(file_path) % 10000) / 10000.0
    return bucket < heldout_ratio


def _sample_window_from_episode(
    task_name: str,
    task_index: int,
    frames: List[np.ndarray],
    actions: List[np.ndarray],
    episode_file: str,
    horizon: int,
    rng: random.Random,
) -> Optional[EvalWindow]:
    if len(actions) < horizon:
        return None

    max_start = len(actions) - horizon
    start = rng.randint(0, max_start)
    window_frames = frames[start : start + horizon + 1]
    window_actions = actions[start : start + horizon]

    if len(window_frames) != horizon + 1 or len(window_actions) != horizon:
        return None

    return EvalWindow(
        task_name=task_name,
        task_index=task_index,
        episode_file=episode_file,
        start=start,
        frames=window_frames,
        actions=window_actions,
    )


def collect_eval_windows(args: argparse.Namespace) -> List[EvalWindow]:
    rng = random.Random(args.seed)

    dataset_name = resolve_dataset_name(args.task_suite)
    task_names = get_task_names(args.task_suite)
    normalized_task_names = [_normalize_name(t) for t in task_names]
    selected_task_indices = parse_task_indices(args)
    selected_set = set(selected_task_indices)

    ds = tfds.load(
        dataset_name,
        data_dir=args.data_root,
        split="train",
        shuffle_files=False,
    )

    def collect_candidates(split_mode_for_collect: str):
        candidates_by_task: Dict[int, List[Tuple[str, int, Dict]]] = {
            idx: [] for idx in selected_task_indices
        }
        raw_counts: Dict[int, int] = {idx: 0 for idx in selected_task_indices}

        for episode in tfds.as_numpy(ds):
            file_path = _decode_bytes(episode["episode_metadata"]["file_path"])
            base_name = _normalize_name(os.path.basename(file_path))

            matched_task_index = None
            for idx, task_key in enumerate(normalized_task_names):
                if task_key in base_name:
                    matched_task_index = idx
                    break

            if matched_task_index is None or matched_task_index not in selected_set:
                continue

            raw_counts[matched_task_index] += 1

            use_this = True
            if split_mode_for_collect == "heldout":
                use_this = _is_heldout(file_path, args.heldout_ratio)

            if use_this:
                candidates_by_task[matched_task_index].append(
                    (task_names[matched_task_index], matched_task_index, episode)
                )

        return candidates_by_task, raw_counts

    actual_split_mode = args.split_mode
    candidates_by_task, raw_counts = collect_candidates(
        "heldout" if args.split_mode == "fallback_all" else args.split_mode
    )

    total_candidates = sum(len(v) for v in candidates_by_task.values())

    if args.split_mode == "fallback_all" and total_candidates == 0:
        print(
            "[warn] No heldout evaluation episodes found for window evaluation. "
            "Falling back to split_mode=all."
        )
        candidates_by_task, raw_counts = collect_candidates("all")
        actual_split_mode = "all"
        total_candidates = sum(len(v) for v in candidates_by_task.values())

    print(f"[info] requested split_mode={args.split_mode}, actual_split_mode={actual_split_mode}")
    print(f"[info] using task orders {selected_task_indices}")
    for idx in selected_task_indices:
        print(
            f"[info] task {idx:02d} | {task_names[idx]} | "
            f"raw episodes={raw_counts.get(idx, 0)} | "
            f"selected episodes={len(candidates_by_task.get(idx, []))}"
        )

    if total_candidates == 0:
        raise RuntimeError(
            "No evaluation episodes found. Check heldout split, task indices, or dataset path."
        )

    windows: List[EvalWindow] = []
    per_task_cursor = {idx: 0 for idx in selected_task_indices}

    shuffled_candidates_by_task: Dict[int, List[Tuple[str, int, Dict]]] = {}
    for idx in selected_task_indices:
        arr = list(candidates_by_task[idx])
        rng.shuffle(arr)
        shuffled_candidates_by_task[idx] = arr

    while len(windows) < args.num_eval_windows:
        progressed = False

        for task_index in selected_task_indices:
            arr = shuffled_candidates_by_task[task_index]
            if len(arr) == 0:
                continue

            triple = arr[per_task_cursor[task_index] % len(arr)]
            task_name, task_index2, episode = triple
            frames, actions, episode_file = _episode_to_arrays(episode)

            maybe_window = _sample_window_from_episode(
                task_name=task_name,
                task_index=task_index2,
                frames=frames,
                actions=actions,
                episode_file=episode_file,
                horizon=args.eval_horizon,
                rng=rng,
            )
            per_task_cursor[task_index] += 1

            if maybe_window is not None:
                windows.append(maybe_window)
                progressed = True

            if len(windows) >= args.num_eval_windows:
                break

        if not progressed:
            break

    if len(windows) == 0:
        raise RuntimeError("Failed to build any evaluation windows.")

    return windows


def _stack_batch(windows: Sequence[EvalWindow]) -> Tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.stack(
        [
            torch.from_numpy(np.ascontiguousarray(np.stack(w.frames, axis=0)))
            for w in windows
        ],
        dim=0,
    )
    actions = torch.stack(
        [
            torch.from_numpy(np.ascontiguousarray(np.stack(w.actions, axis=0)))
            for w in windows
        ],
        dim=0,
    )
    return pixels, actions


# =========================================================
# Evaluation
# =========================================================


@torch.no_grad()
def evaluate_token_loss(
    model: LiberoWorldModelTrainer,
    windows: List[EvalWindow],
    batch_size: int,
) -> Dict:
    losses = []

    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        pixels, actions = _stack_batch(batch)
        outputs = model(pixels=pixels, actions=actions)
        losses.append(float(outputs.loss.item()))
        _cleanup_task_memory(pixels, actions, outputs)

    arr = np.asarray(losses, dtype=np.float32)
    return {
        "num_batches": int(len(losses)),
        "num_windows": int(len(windows)),
        "mean_loss": float(np.mean(arr)),
        "std_loss": float(np.std(arr)),
        "median_loss": float(np.median(arr)),
        "min_loss": float(np.min(arr)),
        "max_loss": float(np.max(arr)),
    }


def aggregate_metric_curves(case_records: List[Dict], horizon: int) -> Dict:
    keys = ["mse_per_frame", "psnr_per_frame", "ssim_per_frame", "lpips_per_frame"]
    curves = {k: [[] for _ in range(horizon)] for k in keys}

    for record in case_records:
        metrics = record["metrics_full"]
        for key in keys:
            arr = metrics[key]
            for t in range(min(horizon, len(arr))):
                curves[key][t].append(float(arr[t]))

    out = {}
    for key in keys:
        out[key] = []
        for t in range(horizon):
            vals = curves[key][t]
            out[key].append(_summarize_values(vals))
    return out


@torch.no_grad()
def evaluate_rollout_fidelity(
    model: LiberoWorldModelTrainer,
    windows: List[EvalWindow],
    lpips_model,
    device: torch.device,
    decode_chunk_size: int,
) -> Dict:
    case_records = []

    for idx, w in enumerate(windows):
        pred_frames, gt_frames = rollout_episode_single_pass(
            model=model,
            frames=w.frames,
            actions=w.actions,
            decode_chunk_size=decode_chunk_size,
        )
        metrics = compute_sequence_metrics_all(gt_frames, pred_frames, lpips_model, device)

        case_records.append(
            {
                "case_id": idx,
                "task_name": w.task_name,
                "task_index": w.task_index,
                "episode_file": w.episode_file,
                "start": w.start,
                "frames": w.frames,
                "pred_frames": pred_frames,
                "gt_frames": gt_frames,
                "metrics_full": metrics,
                "avg_mse": float(metrics["avg_mse"]),
                "avg_psnr": float(metrics["avg_psnr"]),
                "avg_ssim": float(metrics["avg_ssim"]),
                "avg_lpips": float(metrics["avg_lpips"]),
            }
        )

    scalar_summary = {
        "mse": _summarize_values([r["avg_mse"] for r in case_records]),
        "psnr": _summarize_values([r["avg_psnr"] for r in case_records]),
        "ssim": _summarize_values([r["avg_ssim"] for r in case_records]),
        "lpips": _summarize_values([r["avg_lpips"] for r in case_records]),
    }

    horizon = len(windows[0].actions) if len(windows) > 0 else 0
    curves = aggregate_metric_curves(case_records, horizon)

    return {
        "num_windows": int(len(case_records)),
        "overall": scalar_summary,
        "by_horizon": curves,
        "case_records": case_records,
    }


@torch.no_grad()
def evaluate_action_sensitivity(
    model: LiberoWorldModelTrainer,
    windows: List[EvalWindow],
    lpips_model,
    device: torch.device,
    decode_chunk_size: int,
    seed: int,
) -> Dict:
    rng = random.Random(seed)
    records = []

    if len(windows) < 2:
        return {"num_windows": 0}

    for idx, w in enumerate(windows):
        other = windows[rng.randrange(len(windows))]
        shuffled_actions = list(other.actions)

        pred_correct, gt_correct = rollout_episode_single_pass(
            model, w.frames, w.actions, decode_chunk_size
        )
        pred_shuffled, gt_shuffled = rollout_episode_single_pass(
            model, w.frames, shuffled_actions, decode_chunk_size
        )

        metrics_correct = compute_sequence_metrics_all(
            gt_correct, pred_correct, lpips_model, device
        )
        metrics_shuffled = compute_sequence_metrics_all(
            gt_shuffled, pred_shuffled, lpips_model, device
        )

        records.append(
            {
                "case_id": idx,
                "correct_avg_lpips": float(metrics_correct["avg_lpips"]),
                "shuffled_avg_lpips": float(metrics_shuffled["avg_lpips"]),
                "correct_avg_mse": float(metrics_correct["avg_mse"]),
                "shuffled_avg_mse": float(metrics_shuffled["avg_mse"]),
            }
        )

    lpips_gap = [r["shuffled_avg_lpips"] - r["correct_avg_lpips"] for r in records]
    mse_gap = [r["shuffled_avg_mse"] - r["correct_avg_mse"] for r in records]

    return {
        "num_windows": int(len(records)),
        "correct_lpips": _summarize_values([r["correct_avg_lpips"] for r in records]),
        "shuffled_lpips": _summarize_values([r["shuffled_avg_lpips"] for r in records]),
        "lpips_gap": _summarize_values(lpips_gap),
        "correct_mse": _summarize_values([r["correct_avg_mse"] for r in records]),
        "shuffled_mse": _summarize_values([r["shuffled_avg_mse"] for r in records]),
        "mse_gap": _summarize_values(mse_gap),
    }


@torch.no_grad()
def evaluate_diagnostic_chunk(
    model: LiberoWorldModelTrainer,
    windows: List[EvalWindow],
    lpips_model,
    device: torch.device,
    chunk_future_length: int,
    decode_chunk_size: int,
) -> Dict:
    values = []

    for w in windows:
        pred_frames, gt_frames = rollout_episode(
            model=model,
            frames=w.frames,
            actions=w.actions,
            chunk_future_length=chunk_future_length,
            decode_chunk_size=decode_chunk_size,
        )
        metrics = compute_sequence_metrics_all(gt_frames, pred_frames, lpips_model, device)
        values.append(float(metrics["avg_lpips"]))

    return {
        "num_windows": int(len(values)),
        "avg_lpips": _summarize_values(values),
    }


def group_windows_by_task(windows: List[EvalWindow]) -> Dict[int, List[EvalWindow]]:
    grouped: Dict[int, List[EvalWindow]] = {}
    for w in windows:
        grouped.setdefault(w.task_index, []).append(w)
    return grouped


def evaluate_per_task(
    model: LiberoWorldModelTrainer,
    windows: List[EvalWindow],
    args: argparse.Namespace,
    lpips_model,
    device: torch.device,
) -> Dict:
    grouped = group_windows_by_task(windows)
    per_task = {}

    for task_index, task_windows in grouped.items():
        token_loss_summary = evaluate_token_loss(
            model=model,
            windows=task_windows,
            batch_size=args.eval_batch_size,
        )
        rollout_summary = evaluate_rollout_fidelity(
            model=model,
            windows=task_windows,
            lpips_model=lpips_model,
            device=device,
            decode_chunk_size=args.decode_chunk_size,
        )

        action_sensitivity_summary = None
        if args.run_action_sensitivity:
            action_sensitivity_summary = evaluate_action_sensitivity(
                model=model,
                windows=task_windows,
                lpips_model=lpips_model,
                device=device,
                decode_chunk_size=args.decode_chunk_size,
                seed=args.seed + task_index,
            )

        per_task[str(task_index)] = {
            "task_name": task_windows[0].task_name,
            "window_count": len(task_windows),
            "token_loss": token_loss_summary,
            "rollout_fidelity": {
                "overall": rollout_summary["overall"],
                "by_horizon": rollout_summary["by_horizon"],
            },
            "action_sensitivity": action_sensitivity_summary,
        }

    return per_task


def save_per_task_summary(
    output_path: Path,
    config: Dict,
    per_task_summary: Dict,
) -> None:
    payload = {
        "config": config,
        "per_task": per_task_summary,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =========================================================
# Casebook rendering
# =========================================================


def rank_cases(
    case_records: List[Dict],
    key: str = "avg_lpips",
    count: int = 5,
) -> Dict[str, List[Dict]]:
    if len(case_records) == 0:
        return {"best": [], "median": [], "worst": []}

    sorted_cases = sorted(case_records, key=lambda x: x[key])
    best = sorted_cases[:count]
    worst = sorted_cases[-count:]

    center = len(sorted_cases) // 2
    half = count // 2
    median = sorted_cases[
        max(0, center - half) : min(len(sorted_cases), center - half + count)
    ]

    return {
        "best": best,
        "median": median,
        "worst": worst,
    }


def _select_frame_indices(num_frames: int, max_frames: int) -> List[int]:
    if num_frames <= max_frames:
        return list(range(num_frames))
    return np.linspace(0, num_frames - 1, num=max_frames, dtype=int).tolist()


def _make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_f = gt.astype(np.float32) / 255.0
    pred_f = pred.astype(np.float32) / 255.0
    err = np.mean(np.abs(gt_f - pred_f), axis=2)
    err = np.clip(err / max(err.max(), 1e-6), 0.0, 1.0)
    err_rgb = np.stack([err, np.zeros_like(err), 1.0 - err], axis=2)
    return (err_rgb * 255.0).astype(np.uint8)


def render_casebook_figure(
    case: Dict,
    out_path: Path,
    display_frames: int,
    title_prefix: str,
) -> None:
    gt_frames = case["gt_frames"]
    pred_frames = case["pred_frames"]
    indices = _select_frame_indices(len(pred_frames), display_frames)

    fig = plt.figure(figsize=(3.0 * len(indices), 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, len(indices))

    rows = [
        ("GT", gt_frames),
        ("Pred", pred_frames),
        ("Error", None),
    ]

    for row_idx, (label, img_list) in enumerate(rows):
        for col_idx, idx in enumerate(indices):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if label == "Error":
                img = _make_error_map(gt_frames[idx], pred_frames[idx])
            else:
                img = img_list[idx]

            ax.imshow(img)
            ax.axis("off")

            if row_idx == 0:
                ax.set_title(f"step={idx + 1}", fontsize=9)
            elif row_idx == 1:
                ax.set_title(
                    f"MSE {case['metrics_full']['mse_per_frame'][idx]:.4f}\n"
                    f"LPIPS {case['metrics_full']['lpips_per_frame'][idx]:.4f}",
                    fontsize=8,
                )

            if col_idx == 0:
                ax.set_ylabel(
                    label,
                    rotation=0,
                    labelpad=28,
                    fontsize=10,
                    va="center",
                )

    fig.suptitle(
        f"{title_prefix}\n"
        f"{case['task_name']} | start={case['start']} | "
        f"avg MSE={case['avg_mse']:.4f} | avg PSNR={case['avg_psnr']:.2f} | "
        f"avg SSIM={case['avg_ssim']:.4f} | avg LPIPS={case['avg_lpips']:.4f}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_casebook(
    ranked_cases: Dict[str, List[Dict]],
    output_dir: Path,
    display_frames: int,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    for split_name, cases in ranked_cases.items():
        saved[split_name] = []
        for i, case in enumerate(cases):
            out_path = output_dir / f"{split_name}_{i:02d}.png"
            render_casebook_figure(
                case=case,
                out_path=out_path,
                display_frames=display_frames,
                title_prefix=split_name.upper(),
            )
            saved[split_name].append(str(out_path))

    return saved


# =========================================================
# Full-episode rollouts
# =========================================================


def load_episode_for_task(
    dataset_name: str,
    data_root: str,
    task_name: str,
    episode_index: int,
    split_mode: str = "all",
    heldout_ratio: float = 0.2,
) -> Tuple[Dict, str, str]:
    """
    Returns:
        episode, file_path, actual_split_used
    """
    if split_mode not in {"heldout", "all", "fallback_all"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    dataset = tfds.load(dataset_name, data_dir=data_root, split="train")
    task_key = _normalize_name(task_name)

    def collect_matches(use_mode: str):
        matched: List[Tuple[Dict, str]] = []
        for episode in tfds.as_numpy(dataset):
            file_path = _decode_bytes(episode["episode_metadata"]["file_path"])
            base_name = os.path.basename(file_path).lower()

            if task_key not in base_name:
                continue

            if use_mode == "heldout" and not _is_heldout(file_path, heldout_ratio):
                continue

            matched.append((episode, file_path))
        return matched

    if split_mode == "fallback_all":
        matched = collect_matches("heldout")
        actual_mode = "heldout"
        if len(matched) == 0:
            print(
                f"[warn] No heldout episode found for task={task_name}. "
                f"Falling back to split_mode=all for full-episode visualization."
            )
            matched = collect_matches("all")
            actual_mode = "all"
    else:
        matched = collect_matches(split_mode)
        actual_mode = split_mode

    if len(matched) == 0:
        raise RuntimeError(
            f"No episodes found for task={task_name}, split_mode={split_mode}"
        )
    if episode_index < 0 or episode_index >= len(matched):
        raise RuntimeError(
            f"Could not find episode_index={episode_index} for task={task_name}, "
            f"available={len(matched)}, split_mode={actual_mode}"
        )

    episode, file_path = matched[episode_index]
    return episode, file_path, actual_mode


def render_full_episode_overview(
    task_suite: str,
    task_index: int,
    task_name: str,
    episode_file: str,
    gt_frames: List[np.ndarray],
    pred_frames: List[np.ndarray],
    metrics: Dict,
    output_path: Path,
    ncols: int = 6,
) -> None:
    num_frames = min(len(gt_frames), len(pred_frames))
    if num_frames == 0:
        raise ValueError("No frames to render.")

    nrows = math.ceil(num_frames / ncols)
    fig = plt.figure(
        figsize=(3.0 * ncols, 3.0 * nrows * 3),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(3 * nrows, ncols)

    for idx in range(num_frames):
        r = idx // ncols
        c = idx % ncols

        gt_ax = fig.add_subplot(gs[r * 3 + 0, c])
        pr_ax = fig.add_subplot(gs[r * 3 + 1, c])
        er_ax = fig.add_subplot(gs[r * 3 + 2, c])

        gt_ax.imshow(gt_frames[idx])
        gt_ax.axis("off")
        gt_ax.set_title(f"step={idx + 1}", fontsize=8)

        pr_ax.imshow(pred_frames[idx])
        pr_ax.axis("off")
        pr_ax.set_title(
            f"MSE {metrics['mse_per_frame'][idx]:.4f}\n"
            f"LPIPS {metrics['lpips_per_frame'][idx]:.4f}",
            fontsize=7,
        )

        err_img = _make_error_map(gt_frames[idx], pred_frames[idx])
        er_ax.imshow(err_img)
        er_ax.axis("off")

        if c == 0:
            gt_ax.set_ylabel("GT", rotation=0, labelpad=20, va="center")
            pr_ax.set_ylabel("Pred", rotation=0, labelpad=20, va="center")
            er_ax.set_ylabel("Err", rotation=0, labelpad=20, va="center")

    fig.suptitle(
        f"{task_suite} / task{task_index + 1:02d} / {task_name}\n"
        f"{episode_file}\n"
        f"avg MSE={metrics['avg_mse']:.4f} | avg PSNR={metrics['avg_psnr']:.2f} | "
        f"avg SSIM={metrics['avg_ssim']:.4f} | avg LPIPS={metrics['avg_lpips']:.4f}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_full_episode_frames(
    output_dir: Path,
    gt_frames: List[np.ndarray],
    pred_frames: List[np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_frames = min(len(gt_frames), len(pred_frames))

    for i in range(num_frames):
        gt_path = output_dir / f"gt_{i:04d}.png"
        pr_path = output_dir / f"pred_{i:04d}.png"
        er_path = output_dir / f"err_{i:04d}.png"

        plt.imsave(gt_path, gt_frames[i])
        plt.imsave(pr_path, pred_frames[i])
        plt.imsave(er_path, _make_error_map(gt_frames[i], pred_frames[i]))


@torch.no_grad()
def evaluate_full_episodes_for_tasks(
    model: LiberoWorldModelTrainer,
    args: argparse.Namespace,
    device: torch.device,
    lpips_model,
    model_label: str,
) -> Dict:
    dataset_name = resolve_dataset_name(args.task_suite)
    task_names = get_task_names(args.task_suite)
    selected_task_indices = parse_task_indices(args)

    summary = {}

    for task_index in selected_task_indices:
        task_name = task_names[task_index]
        per_task_runs = []

        for k in range(args.num_full_episodes_per_task):
            episode_index = args.full_episode_index + k
            episode, episode_file, actual_split_used = load_episode_for_task(
                dataset_name=dataset_name,
                data_root=args.data_root,
                task_name=task_name,
                episode_index=episode_index,
                split_mode=args.full_episode_split_mode,
                heldout_ratio=args.heldout_ratio,
            )
            frames, actions, _ = _episode_to_arrays(episode)
            del episode

            pred_frames, gt_frames = rollout_episode_single_pass(
                model=model,
                frames=frames,
                actions=actions,
                decode_chunk_size=args.decode_chunk_size,
            )
            metrics = compute_sequence_metrics_all(gt_frames, pred_frames, lpips_model, device)

            task_dir = (
                Path(args.output_dir)
                / f"full_episode__{model_label}"
                / _slugify(task_name)
                / f"episode_{episode_index:03d}"
            )
            task_dir.mkdir(parents=True, exist_ok=True)

            overview_png = task_dir / "overview.png"
            metrics_json = task_dir / "metrics.json"

            render_full_episode_overview(
                task_suite=args.task_suite,
                task_index=task_index,
                task_name=task_name,
                episode_file=episode_file,
                gt_frames=gt_frames,
                pred_frames=pred_frames,
                metrics=metrics,
                output_path=overview_png,
                ncols=args.full_episode_display_cols,
            )

            payload = {
                "task_suite": args.task_suite,
                "task_index": task_index,
                "task_name": task_name,
                "episode_index": episode_index,
                "episode_file": episode_file,
                "requested_split_mode": args.full_episode_split_mode,
                "actual_split_used": actual_split_used,
                "frame_count": len(pred_frames),
                "avg_mse": float(metrics["avg_mse"]),
                "avg_psnr": float(metrics["avg_psnr"]),
                "avg_ssim": float(metrics["avg_ssim"]),
                "avg_lpips": float(metrics["avg_lpips"]),
                "mse_per_frame": metrics["mse_per_frame"].tolist(),
                "psnr_per_frame": metrics["psnr_per_frame"].tolist(),
                "ssim_per_frame": metrics["ssim_per_frame"].tolist(),
                "lpips_per_frame": metrics["lpips_per_frame"].tolist(),
                "overview_png": str(overview_png),
            }
            with open(metrics_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            if args.save_full_episode_frames:
                save_full_episode_frames(task_dir / "frames", gt_frames, pred_frames)

            per_task_runs.append(payload)
            _cleanup_task_memory(frames, actions, pred_frames, gt_frames, metrics)

        summary[str(task_index)] = {
            "task_name": task_name,
            "episodes": per_task_runs,
        }

    return summary


# =========================================================
# Report
# =========================================================


def save_eval_report(
    output_path: Path,
    config: Dict,
    token_loss_summary: Dict,
    rollout_summary: Dict,
    action_sensitivity_summary: Optional[Dict],
    casebook_paths: Dict,
    diagnostic_chunk_summary: Optional[Dict],
    per_task_summary_path: Optional[str] = None,
    full_episode_summary_path: Optional[str] = None,
) -> None:
    report = {
        "config": config,
        "token_loss": token_loss_summary,
        "rollout_fidelity": {
            "num_windows": rollout_summary["num_windows"],
            "overall": rollout_summary["overall"],
            "by_horizon": rollout_summary["by_horizon"],
        },
        "action_sensitivity": action_sensitivity_summary,
        "diagnostic_chunk": diagnostic_chunk_summary,
        "casebook": casebook_paths,
        "per_task_summary_path": per_task_summary_path,
        "full_episode_summary_path": full_episode_summary_path,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# =========================================================
# Run one model
# =========================================================


def run_single_model_evaluation(
    model: LiberoWorldModelTrainer,
    windows: List[EvalWindow],
    args: argparse.Namespace,
    lpips_model,
    device: torch.device,
) -> Dict:
    token_loss_summary = evaluate_token_loss(
        model=model,
        windows=windows,
        batch_size=args.eval_batch_size,
    )

    rollout_summary = evaluate_rollout_fidelity(
        model=model,
        windows=windows,
        lpips_model=lpips_model,
        device=device,
        decode_chunk_size=args.decode_chunk_size,
    )

    action_sensitivity_summary = None
    if args.run_action_sensitivity:
        action_sensitivity_summary = evaluate_action_sensitivity(
            model=model,
            windows=windows,
            lpips_model=lpips_model,
            device=device,
            decode_chunk_size=args.decode_chunk_size,
            seed=args.seed + 123,
        )

    diagnostic_chunk_summary = None
    if args.run_diagnostic_chunk:
        diagnostic_chunk_summary = evaluate_diagnostic_chunk(
            model=model,
            windows=windows,
            lpips_model=lpips_model,
            device=device,
            chunk_future_length=args.chunk_future_length,
            decode_chunk_size=args.decode_chunk_size,
        )

    ranked_cases = rank_cases(
        rollout_summary["case_records"],
        key="avg_lpips",
        count=args.save_casebook_count,
    )

    return {
        "token_loss_summary": token_loss_summary,
        "rollout_summary": rollout_summary,
        "action_sensitivity_summary": action_sensitivity_summary,
        "diagnostic_chunk_summary": diagnostic_chunk_summary,
        "ranked_cases": ranked_cases,
    }


# =========================================================
# Main evaluation runner
# =========================================================


def run_evaluation(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = _device_for_run(args.device)
    trained_model_dir = _resolve_trained_model_dir(args)
    exp_label = _experiment_label(args, trained_model_dir)
    selected_task_indices = parse_task_indices(args)

    base_model_dir = str(Path(args.base_model_root) / args.task_suite)
    trained_model = _load_trained_model(
        base_model_dir=base_model_dir,
        trained_model_dir=trained_model_dir,
        tokenizer_dir=args.visual_tokenizer,
        device=device,
    )

    base_model = None
    if args.compare_base:
        base_model = _load_model(base_model_dir, args.visual_tokenizer, device)

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    windows = collect_eval_windows(args)

    trained_eval = run_single_model_evaluation(
        model=trained_model,
        windows=windows,
        args=args,
        lpips_model=lpips_model,
        device=device,
    )

    trained_casebook_dir = output_root / f"casebook__trained__{exp_label}"
    trained_casebook_paths = render_casebook(
        ranked_cases=trained_eval["ranked_cases"],
        output_dir=trained_casebook_dir,
        display_frames=args.display_frames,
    )

    trained_per_task = evaluate_per_task(
        model=trained_model,
        windows=windows,
        args=args,
        lpips_model=lpips_model,
        device=device,
    )
    trained_per_task_json = output_root / f"per_task_summary__trained__{exp_label}.json"
    save_per_task_summary(
        output_path=trained_per_task_json,
        config={
            "task_suite": args.task_suite,
            "selected_task_indices": selected_task_indices,
            "num_eval_windows": args.num_eval_windows,
            "eval_horizon": args.eval_horizon,
            "heldout_ratio": args.heldout_ratio,
            "split_mode": args.split_mode,
        },
        per_task_summary=trained_per_task,
    )

    trained_full_episode = evaluate_full_episodes_for_tasks(
        model=trained_model,
        args=args,
        device=device,
        lpips_model=lpips_model,
        model_label=f"trained__{exp_label}",
    )
    trained_full_episode_json = output_root / f"full_episode_summary__trained__{exp_label}.json"
    with open(trained_full_episode_json, "w", encoding="utf-8") as f:
        json.dump(trained_full_episode, f, indent=2)

    trained_report_path = output_root / f"eval_report__trained__{exp_label}.json"
    save_eval_report(
        output_path=trained_report_path,
        config={
            "task_suite": args.task_suite,
            "selected_task_indices": selected_task_indices,
            "num_eval_windows": args.num_eval_windows,
            "eval_horizon": args.eval_horizon,
            "heldout_ratio": args.heldout_ratio,
            "split_mode": args.split_mode,
            "decode_chunk_size": args.decode_chunk_size,
            "device": str(device),
            "trained_model_dir": trained_model_dir,
            "visual_tokenizer": args.visual_tokenizer,
        },
        token_loss_summary=trained_eval["token_loss_summary"],
        rollout_summary=trained_eval["rollout_summary"],
        action_sensitivity_summary=trained_eval["action_sensitivity_summary"],
        casebook_paths=trained_casebook_paths,
        diagnostic_chunk_summary=trained_eval["diagnostic_chunk_summary"],
        per_task_summary_path=str(trained_per_task_json),
        full_episode_summary_path=str(trained_full_episode_json),
    )

    if args.compare_base and base_model is not None:
        base_eval = run_single_model_evaluation(
            model=base_model,
            windows=windows,
            args=args,
            lpips_model=lpips_model,
            device=device,
        )

        base_casebook_dir = output_root / f"casebook__base__{exp_label}"
        base_casebook_paths = render_casebook(
            ranked_cases=base_eval["ranked_cases"],
            output_dir=base_casebook_dir,
            display_frames=args.display_frames,
        )

        base_per_task = evaluate_per_task(
            model=base_model,
            windows=windows,
            args=args,
            lpips_model=lpips_model,
            device=device,
        )
        base_per_task_json = output_root / f"per_task_summary__base__{exp_label}.json"
        save_per_task_summary(
            output_path=base_per_task_json,
            config={
                "task_suite": args.task_suite,
                "selected_task_indices": selected_task_indices,
                "num_eval_windows": args.num_eval_windows,
                "eval_horizon": args.eval_horizon,
                "heldout_ratio": args.heldout_ratio,
                "split_mode": args.split_mode,
            },
            per_task_summary=base_per_task,
        )

        base_full_episode = evaluate_full_episodes_for_tasks(
            model=base_model,
            args=args,
            device=device,
            lpips_model=lpips_model,
            model_label=f"base__{exp_label}",
        )
        base_full_episode_json = output_root / f"full_episode_summary__base__{exp_label}.json"
        with open(base_full_episode_json, "w", encoding="utf-8") as f:
            json.dump(base_full_episode, f, indent=2)

        base_report_path = output_root / f"eval_report__base__{exp_label}.json"
        save_eval_report(
            output_path=base_report_path,
            config={
                "task_suite": args.task_suite,
                "selected_task_indices": selected_task_indices,
                "num_eval_windows": args.num_eval_windows,
                "eval_horizon": args.eval_horizon,
                "heldout_ratio": args.heldout_ratio,
                "split_mode": args.split_mode,
                "decode_chunk_size": args.decode_chunk_size,
                "device": str(device),
                "base_model_dir": base_model_dir,
                "visual_tokenizer": args.visual_tokenizer,
            },
            token_loss_summary=base_eval["token_loss_summary"],
            rollout_summary=base_eval["rollout_summary"],
            action_sensitivity_summary=base_eval["action_sensitivity_summary"],
            casebook_paths=base_casebook_paths,
            diagnostic_chunk_summary=base_eval["diagnostic_chunk_summary"],
            per_task_summary_path=str(base_per_task_json),
            full_episode_summary_path=str(base_full_episode_json),
        )

    _cleanup_task_memory(trained_model, base_model, lpips_model, windows)
    return trained_report_path


def main():
    args = build_parser().parse_args()
    report_path = run_evaluation(args)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()