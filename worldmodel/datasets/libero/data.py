from __future__ import annotations

import re
import random
import zlib
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset


def _extract_task_id(episode: dict, episode_index: int = 0) -> int:
    """Derive a stable integer task identifier from episode_metadata.file_path.

    Strategy:
      - Strip trailing episode numbers ("_demo_42", "_1") from the filename
      - Combine with parent directory to handle both flat and hierarchical layouts
      - CRC32 for a stable, collision-resistant hash

    If file_path is unavailable, returns a unique negative sentinel so that no
    false same-task matches occur in the collator.
    """
    meta = episode.get("episode_metadata", {})
    fp = meta.get("file_path", b"")
    if isinstance(fp, (bytes, bytearray)):
        fp = fp.decode("utf-8", errors="ignore")
    fp = fp.replace("\\", "/").strip()
    if not fp:
        # Unique per episode → no false same-task matches
        return -(episode_index + 1)

    # Split into parent dir + basename
    if "/" in fp:
        parent, basename_full = fp.rsplit("/", 1)
    else:
        parent, basename_full = "", fp

    # Strip extension
    basename = basename_full.rsplit(".", 1)[0] if "." in basename_full else basename_full
    # Strip trailing episode numbers: "task_demo_42" → "task", "task_1" → "task"
    basename_no_num = re.sub(r"(_demo)?_\d+$", "", basename).rstrip("_")

    # Use parent + stripped basename as task key
    # For hierarchical: parent="task_dir", basename_no_num="" → "task_dir/"
    # For flat: parent="suite_dir", basename_no_num="task_name" → "suite_dir/task_name"
    task_key = parent + "/" + basename_no_num
    return zlib.crc32(task_key.encode("utf-8")) & 0x7FFFFFFF


# Maps LIBERO task-suite short names to TFDS dataset names.
# To add a new dataset, extend this dict or pass --dataset-name directly.
LIBERO_TASK_TO_DATASET = {
    "spatial": "libero_spatial_no_noops",
    "object": "libero_object_no_noops",
    "goal": "libero_goal_no_noops",
    "10": "libero_10_no_noops",
    "long": "libero_10_no_noops",  # "long" is an alias for the LIBERO-10 suite
}


def resolve_dataset_name(task_suite: str) -> str:
    if task_suite in LIBERO_TASK_TO_DATASET:
        return LIBERO_TASK_TO_DATASET[task_suite]
    if task_suite.startswith("libero_") and task_suite.endswith("_no_noops"):
        return task_suite
    raise ValueError(f"Unsupported LIBERO task suite: {task_suite}")


def _dist_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


@dataclass
class SampleWindow:
    pixels: torch.Tensor
    actions: torch.Tensor
    # Optional episode-level metadata (populated when include_episode_metadata=True)
    episode_init_pixels: Optional[torch.Tensor] = None  # [H, W, C] uint8 — episode frame_0
    episode_goal_pixels: Optional[torch.Tensor] = None  # [H, W, C] uint8 — episode last frame
    window_start: Optional[int] = None                  # index of window start in episode
    episode_length: Optional[int] = None                # total number of steps in episode


class RldsIterableDataset(IterableDataset):
    """
    Streams fixed-length windows from any RLDS episode dataset.

    Expects each episode to have a "steps" field where each step has:
      - step["observation"][image_key]: uint8 image array
      - step["action"]: float32 action array

    When include_episode_metadata=True, each yielded sample additionally
    contains episode_init_pixels, episode_goal_pixels, window_start, and
    episode_length for reward-aligned training.  Default is False to preserve
    backward compatibility with existing training pipelines.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        raw_chunk_length: int,
        seed: int = 42,
        shuffle_episodes: bool = True,
        shuffle_windows: bool = True,
        window_stride: int = 1,
        image_key: str = "image",
        include_episode_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.raw_chunk_length = raw_chunk_length
        self.seed = seed
        self.shuffle_episodes = shuffle_episodes
        self.shuffle_windows = shuffle_windows
        self.window_stride = window_stride
        self.image_key = image_key
        self.include_episode_metadata = include_episode_metadata

    def _episode_stream(self) -> Iterator[dict]:
        ds = tfds.load(
            self.dataset_name,
            data_dir=self.data_dir,
            split="train",
            shuffle_files=self.shuffle_episodes,
        )
        if self.shuffle_episodes:
            ds = ds.shuffle(128, seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.repeat()
        yield from tfds.as_numpy(ds)

    def _window_starts(self, episode_length: int, rng: random.Random) -> List[int]:
        if episode_length < self.raw_chunk_length:
            return []
        starts = list(range(0, episode_length - self.raw_chunk_length + 1, self.window_stride))
        if self.shuffle_windows:
            rng.shuffle(starts)
        return starts

    def _build_sample(
        self,
        steps: Sequence[dict],
        start: int,
        episode_init_pixels: Optional[np.ndarray] = None,
        episode_goal_pixels: Optional[np.ndarray] = None,
    ) -> SampleWindow:
        window = steps[start : start + self.raw_chunk_length]
        pixels = np.stack(
            [step["observation"][self.image_key] for step in window],
            axis=0,
        ).astype(np.uint8)
        actions = np.stack(
            [step["action"] for step in window[:-1]],
            axis=0,
        ).astype(np.float32)

        sw = SampleWindow(
            pixels=torch.from_numpy(np.ascontiguousarray(pixels)),
            actions=torch.from_numpy(np.ascontiguousarray(actions)),
        )

        if self.include_episode_metadata:
            sw.episode_init_pixels = torch.from_numpy(
                np.ascontiguousarray(episode_init_pixels)
            )
            sw.episode_goal_pixels = torch.from_numpy(
                np.ascontiguousarray(episode_goal_pixels)
            )
            sw.window_start = start
            sw.episode_length = len(steps)

        return sw

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rank, world_size = _dist_info()

        # DataLoader worker sharding (within each DDP rank).
        # Combines DDP rank and worker ID into a single shard index so that
        # num_workers > 0 does not produce duplicate samples across workers.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            effective_rank  = rank * worker_info.num_workers + worker_info.id
            effective_world = world_size * worker_info.num_workers
        else:
            effective_rank  = rank
            effective_world = world_size

        rng = random.Random(self.seed + effective_rank)

        for episode_index, episode in enumerate(self._episode_stream()):
            if episode_index % effective_world != effective_rank:
                continue

            steps = list(episode["steps"])
            starts = self._window_starts(len(steps), rng)
            if not starts:
                continue

            # Stable task identifier (same for all windows of this episode)
            task_id = _extract_task_id(episode, episode_index)

            # Episode-level metadata (computed once per episode)
            ep_init = None
            ep_goal = None
            if self.include_episode_metadata:
                ep_init = steps[0]["observation"][self.image_key].astype(np.uint8)
                ep_goal = steps[-1]["observation"][self.image_key].astype(np.uint8)

            for start in starts:
                sample = self._build_sample(steps, start, ep_init, ep_goal)
                out: Dict[str, torch.Tensor] = {
                    "pixels":  sample.pixels,
                    "actions": sample.actions,
                    "task_id": torch.tensor(task_id, dtype=torch.long),
                }
                if self.include_episode_metadata:
                    out["episode_init_pixels"] = sample.episode_init_pixels
                    out["episode_goal_pixels"] = sample.episode_goal_pixels
                    out["window_start"] = torch.tensor(sample.window_start, dtype=torch.long)
                    out["episode_length"] = torch.tensor(sample.episode_length, dtype=torch.long)
                yield out


# Backward-compatibility alias used by existing LIBERO training code
LiberoWorldModelIterableDataset = RldsIterableDataset
