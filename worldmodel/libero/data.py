from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence

import numpy as np
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset


LIBERO_TASK_TO_DATASET = {
    "spatial": "libero_spatial_no_noops",
    "object": "libero_object_no_noops",
    "goal": "libero_goal_no_noops",
    "10": "libero_10_no_noops",
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


class LiberoWorldModelIterableDataset(IterableDataset):
    """
    Streams fixed-length windows from RLDS LIBERO episodes.

    Each yielded sample contains a raw chunk of frames/actions. The trainer
    prepends the context frame/action, mirroring the paper's stage-I world
    model setup.
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

    def _build_sample(self, steps: Sequence[dict], start: int) -> SampleWindow:
        window = steps[start : start + self.raw_chunk_length]
        pixels = np.stack(
            [step["observation"][self.image_key] for step in window],
            axis=0,
        ).astype(np.uint8)
        actions = np.stack(
            [step["action"] for step in window[:-1]],
            axis=0,
        ).astype(np.float32)
        return SampleWindow(
            pixels=torch.from_numpy(np.ascontiguousarray(pixels)),
            actions=torch.from_numpy(np.ascontiguousarray(actions)),
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rank, world_size = _dist_info()
        rng = random.Random(self.seed + rank)

        for episode_index, episode in enumerate(self._episode_stream()):
            if episode_index % world_size != rank:
                continue

            steps = list(episode["steps"])
            starts = self._window_starts(len(steps), rng)
            if not starts:
                continue

            for start in starts:
                sample = self._build_sample(steps, start)
                yield {"pixels": sample.pixels, "actions": sample.actions}
