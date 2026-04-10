from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import lpips
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import torch
from safetensors.torch import load_file as load_safetensors

from libero.libero import benchmark

from .data import resolve_dataset_name
from .model import LiberoWorldModelTrainer, WorldModelRuntimeConfig


@dataclass
class EpisodeBundle:
    task_index: int
    task_name: str
    file_path: str
    frames: List[np.ndarray]
    actions: List[np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualize LIBERO world-model rollouts against GT.')
    parser.add_argument('--task-suite', type=str, choices=['spatial', 'object', 'goal', '10'], required=True)
    parser.add_argument('--task-index', type=int, required=True)
    parser.add_argument('--data-root', type=str, default='data/modified_libero_rlds')
    parser.add_argument('--base-model-root', type=str, default='checkpoints/libero/WorldModel')
    parser.add_argument('--trained-model-root', type=str, default='checkpoints/libero/WorldModel')
    parser.add_argument('--trained-exp-name', type=str, default='worldmodel_scratch')
    parser.add_argument('--date', type=str, default='')
    parser.add_argument(
        '--trained-model-dir',
        type=str,
        default='',
        help='Direct path to a fully exported trained model dir. If set, date/trained-exp-name are ignored.',
    )
    parser.add_argument('--visual-tokenizer', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--episode-index', type=int, default=0)
    parser.add_argument('--chunk-future-length', type=int, default=8)
    parser.add_argument('--display-frames', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    return parser


def _suite_key(task_suite: str) -> str:
    return f'libero_{task_suite}'


def get_task_names(task_suite: str) -> List[str]:
    benchmarks = benchmark.get_benchmark_dict()
    suite_key = _suite_key(task_suite)
    if suite_key not in benchmarks:
        raise ValueError(f'Unsupported task suite: {task_suite}')
    return benchmarks[suite_key]().get_task_names()


def _decode_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def _normalize_name(name: str) -> str:
    return name.lower().replace(' ', '_')


def load_episode_for_task(dataset_name: str, data_root: str, task_name: str, episode_index: int) -> Tuple[Dict, str]:
    dataset = tfds.load(dataset_name, data_dir=data_root, split='train')
    task_key = _normalize_name(task_name)
    matched = 0
    fallback_episode = None
    fallback_path = ''

    for episode in tfds.as_numpy(dataset):
        file_path = _decode_bytes(episode['episode_metadata']['file_path'])
        base_name = os.path.basename(file_path).lower()
        if fallback_episode is None:
            fallback_episode = episode
            fallback_path = file_path
        if task_key in base_name:
            if matched == episode_index:
                return episode, file_path
            matched += 1

    if fallback_episode is None:
        raise RuntimeError(f'No episodes found for dataset {dataset_name}')
    if episode_index == 0:
        return fallback_episode, fallback_path
    raise RuntimeError(f'Could not find episode_index={episode_index} for task {task_name}')


def _episode_to_arrays(episode: Dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    steps = list(episode['steps'])
    frames: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    for step in steps:
        frames.append(np.asarray(step['observation']['image'], dtype=np.uint8))
        actions.append(np.asarray(step['action'], dtype=np.float32))
    return frames, actions[:-1]


def _select_frame_indices(num_frames: int, max_frames: int) -> List[int]:
    if num_frames <= max_frames:
        return list(range(num_frames))
    return np.linspace(0, num_frames - 1, num=max_frames, dtype=int).tolist()


def _device_for_run(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def _load_model(model_dir: str, tokenizer_dir: str, device: torch.device) -> LiberoWorldModelTrainer:
    dtype = torch.bfloat16 if device.type == 'cuda' else None
    model = LiberoWorldModelTrainer(
        lm_path=model_dir,
        visual_tokenizer_path=tokenizer_dir,
        runtime_cfg=WorldModelRuntimeConfig(
            action_ranges_path='train/verl/ivideogpt/configs/libero_action_ranges.pth',
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
            processor_type='ctx_msp',
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


def _strip_common_prefixes(state_dict):
    if any(k.startswith("lm.") for k in state_dict.keys()):
        return {k[3:] if k.startswith("lm.") else k: v for k, v in state_dict.items()}
    if any(k.startswith("model.") for k in state_dict.keys()):
        return {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    return state_dict


def _load_trained_model(
    base_model_dir: str,
    trained_model_dir: str,
    tokenizer_dir: str,
    device: torch.device,
) -> LiberoWorldModelTrainer:
    # まず valid な base model config からモデル骨格を作る
    model = _load_model(base_model_dir, tokenizer_dir, device)

    weight_path = Path(trained_model_dir) / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Trained weight not found: {weight_path}")

    state_dict = load_safetensors(str(weight_path))
    state_dict = _strip_common_prefixes(state_dict)

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
        date_prefix = args.date or ''
        exp_name = f'{date_prefix}_{args.trained_exp_name}' if date_prefix else args.trained_exp_name
        trained_dir = Path(args.trained_model_root) / args.task_suite / exp_name

    if not trained_dir.exists():
        raise FileNotFoundError(f'Trained model dir not found: {trained_dir}')

    if not trained_dir.is_dir():
        raise NotADirectoryError(f'Trained model path is not a directory: {trained_dir}')

    # checkpoint-* を直接指したときだけ止める
    if trained_dir.name.startswith('checkpoint-'):
        raise RuntimeError(
            f'Please point to the experiment root directory, not a checkpoint directory: {trained_dir}\n'
            f'Expected something like: .../{args.task_suite}/{args.date}_{args.trained_exp_name}'
        )

    return str(trained_dir)


@torch.no_grad()
def _prime_model(model: LiberoWorldModelTrainer, prompt_ids: torch.Tensor):
    outputs = model.lm(input_ids=prompt_ids, use_cache=True)
    return outputs.logits[:, -1, :], outputs.past_key_values


@torch.no_grad()
def _advance_model(model: LiberoWorldModelTrainer, token_ids: torch.Tensor, past_key_values):
    outputs = model.lm(input_ids=token_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.logits[:, -1, :], outputs.past_key_values


@torch.no_grad()
def rollout_episode(model: LiberoWorldModelTrainer, frames: List[np.ndarray], actions: List[np.ndarray], chunk_future_length: int):
    device = next(model.lm.parameters()).device
    visual_token_num = model.runtime_cfg.visual_token_num
    generated_frames: List[np.ndarray] = []

    context_frame = frames[0]
    action_cursor = 0

    while action_cursor < len(actions):
        chunk_actions = actions[action_cursor:action_cursor + chunk_future_length]
        if len(chunk_actions) == 0:
            break

        ctx_pair = np.stack([context_frame, context_frame], axis=0)
        ctx_tensor = torch.from_numpy(ctx_pair).permute(0, 3, 1, 2).to(device=device).unsqueeze(0).float() / 255.0
        ctx_tokens_raw, _ = model.visual_tokenizer.tokenize(ctx_tensor)
        ctx_tokens_raw = ctx_tokens_raw[:, :1]
        ctx_tokens_prompt = ctx_tokens_raw + visual_token_num
        prompt_ids = ctx_tokens_prompt.reshape(1, -1).long()

        logits, past = _prime_model(model, prompt_ids)
        dyn_blocks = []

        for action in chunk_actions:
            frame_tokens = []
            for _ in range(model.runtime_cfg.tokens_per_frame):
                allowed_logits = logits.clone()
                allowed_logits[:, visual_token_num:] = -torch.inf
                next_token = torch.argmax(allowed_logits, dim=-1, keepdim=True)
                frame_tokens.append(next_token)
                logits, past = _advance_model(model, next_token.long(), past)
            dyn_blocks.append(torch.cat(frame_tokens, dim=-1))

            action_tensor = torch.from_numpy(np.asarray(action, dtype=np.float32)).to(device=device).view(1, 1, -1)
            action_ids = model.processor._discretize_actions(action_tensor, model.runtime_cfg.action_bins)[0, 0]
            action_ids = action_ids + visual_token_num * 2
            for token in action_ids.reshape(-1):
                token_tensor = token.view(1, 1).long()
                logits, past = _advance_model(model, token_tensor, past)

        dyn_tokens = torch.stack(dyn_blocks, dim=1)
        recon = model.visual_tokenizer.detokenize(ctx_tokens_raw, dyn_tokens)
        recon = recon[:, 1:].clamp(0.0, 1.0)
        chunk_frames = [frame.detach().cpu().permute(1, 2, 0).numpy() for frame in recon[0]]
        generated_frames.extend([np.clip(frame * 255.0, 0, 255).astype(np.uint8) for frame in chunk_frames])

        if len(chunk_frames) > 0:
            context_frame = generated_frames[-1]
        action_cursor += len(chunk_actions)

    gt_frames = frames[1:1 + len(generated_frames)]
    return generated_frames, gt_frames


@torch.no_grad()
def _compute_metrics(gt_frames: List[np.ndarray], pred_frames: List[np.ndarray], lpips_model, device: torch.device):
    mses: List[float] = []
    lpips_scores: List[float] = []
    for gt, pred in zip(gt_frames, pred_frames):
        gt_t = torch.from_numpy(gt).to(device=device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        pred_t = torch.from_numpy(pred).to(device=device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mses.append(torch.mean((pred_t - gt_t) ** 2).item())
        lpips_scores.append(lpips_model(pred_t * 2.0 - 1.0, gt_t * 2.0 - 1.0).item())
    return np.asarray(mses, dtype=np.float32), np.asarray(lpips_scores, dtype=np.float32)


def _render_episode_figure(
    task_suite: str,
    task_index: int,
    task_name: str,
    file_path: str,
    frames: List[np.ndarray],
    actions: List[np.ndarray],
    base_pred: List[np.ndarray],
    trained_pred: List[np.ndarray],
    base_mse: np.ndarray,
    base_lpips: np.ndarray,
    trained_mse: np.ndarray,
    trained_lpips: np.ndarray,
    output_path: Path,
    display_frames: int,
):
    num_frames = min(len(base_pred), len(trained_pred), len(frames) - 1)
    display_indices = _select_frame_indices(num_frames, display_frames)
    fig = plt.figure(figsize=(3.2 * len(display_indices), 8.8), constrained_layout=True)
    gs = fig.add_gridspec(3, len(display_indices), height_ratios=[1.0, 1.0, 1.0])

    aligned_gt_frames = frames[1: 1 + num_frames]
    aligned_base_pred = base_pred[:num_frames]
    aligned_trained_pred = trained_pred[:num_frames]
    aligned_base_mse = base_mse[:num_frames]
    aligned_base_lpips = base_lpips[:num_frames]
    aligned_trained_mse = trained_mse[:num_frames]
    aligned_trained_lpips = trained_lpips[:num_frames]

    rows = [
        ('GT', aligned_gt_frames, None, None),
        ('Base', aligned_base_pred, aligned_base_mse, aligned_base_lpips),
        ('Trained', aligned_trained_pred, aligned_trained_mse, aligned_trained_lpips),
    ]

    for row_idx, (label, image_list, mse_list, lpips_list) in enumerate(rows):
        for col_idx, frame_idx in enumerate(display_indices):
            step_num = frame_idx + 1
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(image_list[frame_idx])
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(f'step={step_num}', fontsize=9)
            else:
                ax.set_title(
                    f'step={step_num}\nMSE {mse_list[frame_idx]:.4f}\nLPIPS {lpips_list[frame_idx]:.4f}',
                    fontsize=8,
                )
            if col_idx == 0:
                ax.set_ylabel(label, rotation=0, labelpad=32, fontsize=10, va='center')

    base_avg_mse = float(np.mean(base_mse))
    base_avg_lpips = float(np.mean(base_lpips))
    trained_avg_mse = float(np.mean(trained_mse))
    trained_avg_lpips = float(np.mean(trained_lpips))
    fig.suptitle(
        f'{task_suite} / task{task_index + 1:02d} / {task_name}\n'
        f'Base avg MSE {base_avg_mse:.4f} | Base avg LPIPS {base_avg_lpips:.4f} | '
        f'Trained avg MSE {trained_avg_mse:.4f} | Trained avg LPIPS {trained_avg_lpips:.4f}',
        fontsize=12,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def visualize_one_task(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    task_names = get_task_names(args.task_suite)
    if args.task_index < 0 or args.task_index >= len(task_names):
        raise ValueError(f'task_index out of range: {args.task_index}')
    task_name = task_names[args.task_index]
    dataset_name = resolve_dataset_name(args.task_suite)
    episode, file_path = load_episode_for_task(dataset_name, args.data_root, task_name, args.episode_index)
    frames, actions = _episode_to_arrays(episode)

    device = _device_for_run(args.device)
    base_model_dir = str(Path(args.base_model_root) / args.task_suite)
    trained_model_dir = _resolve_trained_model_dir(args)

    base_model = _load_model(base_model_dir, args.visual_tokenizer, device)
    trained_model = _load_trained_model(
        base_model_dir=base_model_dir,
        trained_model_dir=trained_model_dir,
        tokenizer_dir=args.visual_tokenizer,
        device=device,
    )

    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    base_pred, gt_frames = rollout_episode(base_model, frames, actions, args.chunk_future_length)
    trained_pred, _ = rollout_episode(trained_model, frames, actions, args.chunk_future_length)

    if len(base_pred) == 0 or len(trained_pred) == 0:
        raise RuntimeError('No rollout frames were generated')

    gt_frames = gt_frames[:len(base_pred)]
    base_mse, base_lpips = _compute_metrics(gt_frames, base_pred, lpips_model, device)
    trained_mse, trained_lpips = _compute_metrics(gt_frames, trained_pred, lpips_model, device)

    task_out_dir = Path(args.output_dir)
    task_out_dir.mkdir(parents=True, exist_ok=True)
    slug = task_name.lower().replace(' ', '_')
    out_png = task_out_dir / f'task{args.task_index + 1:02d}_{slug}.png'
    out_json = task_out_dir / f'task{args.task_index + 1:02d}_{slug}.json'

    _render_episode_figure(
        args.task_suite,
        args.task_index,
        task_name,
        file_path,
        frames,
        actions,
        base_pred,
        trained_pred,
        base_mse,
        base_lpips,
        trained_mse,
        trained_lpips,
        out_png,
        args.display_frames,
    )

    summary = {
        'task_suite': args.task_suite,
        'task_index': args.task_index,
        'task_name': task_name,
        'episode_index': args.episode_index,
        'episode_file': file_path,
        'base_model_dir': base_model_dir,
        'trained_model_dir': trained_model_dir,
        'base_avg_mse': float(np.mean(base_mse)),
        'base_avg_lpips': float(np.mean(base_lpips)),
        'trained_avg_mse': float(np.mean(trained_mse)),
        'trained_avg_lpips': float(np.mean(trained_lpips)),
        'frame_count': len(base_pred),
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return out_png, out_json


def main():
    args = build_parser().parse_args()
    out_png, out_json = visualize_one_task(args)
    print(f'Wrote {out_png}')
    print(f'Wrote {out_json}')


if __name__ == '__main__':
    main()