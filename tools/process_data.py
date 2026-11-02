"""Whole-video context beside the unchanged released MSR local sampler.

This intentionally extends the model's input coverage. It does not change the
released subject split, local-window enumeration, or video-score aggregation.
No labels, predictions, or features are used to construct the context.
"""
from __future__ import annotations

import hashlib
from numbers import Integral
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessInput(dict):
    """Collatable input mapping retaining the official runner's tensor helpers."""

    @property
    def shape(self):
        return self["local"].shape

    def to(self, device, non_blocking: bool = False) -> "ProcessInput":
        if not all(isinstance(value, torch.Tensor) for value in self.values()):
            raise TypeError("ProcessInput.to requires collated tensor values")
        return type(self)({key: value.to(device=device, non_blocking=non_blocking)
                           for key, value in self.items()})


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class ProcessDataset(Dataset):
    """Wrap an already constructed, frozen-source ``MSRAction3D`` dataset.

    ``local`` reproduces the released ``__getitem__`` arithmetic and global
    NumPy random-call order, including its repeat/residue point sampling. The
    private per-video context stream never consumes global NumPy/Torch RNG.
    Cached values are raw points, not augmented points or learned features.
    """

    def __init__(self, official_dataset: Dataset, *, context_frames: int = 32,
                 context_points: int = 256, context_seed: int = 0) -> None:
        super().__init__()
        required = ("videos", "video_names", "labels", "index_map",
                    "directory_manifest", "num_classes", "frames_per_clip",
                    "step_between_clips", "num_points", "train")
        for name in required:
            if not hasattr(official_dataset, name):
                raise TypeError(f"official MSR dataset is missing {name}")
        if (official_dataset.frames_per_clip != 24
                or official_dataset.step_between_clips != 1
                or official_dataset.num_points != 2048):
            raise ValueError("this candidate requires released local settings 24/1/2048")
        self.official_dataset = official_dataset
        for name in required:
            setattr(self, name, getattr(official_dataset, name))
        self.context_frames = _positive_integer(context_frames, "context_frames")
        self.context_points = _positive_integer(context_points, "context_points")
        if self.context_frames < 2 or self.context_frames > 32:
            raise ValueError("context_frames must be between 2 and 32")
        if isinstance(context_seed, bool) or not isinstance(context_seed, Integral):
            raise ValueError("context_seed must be an integer")
        self.context_seed = int(context_seed)
        self._context_cache: dict[int, tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.official_dataset)

    def _raw_context(self, video_index: int):
        """Return read-only (raw points, normalized times, validity, indices)."""
        if video_index in self._context_cache:
            return self._context_cache[video_index]
        video = self.videos[video_index]
        length = len(video)
        if length < self.frames_per_clip:
            raise ValueError("a short video cannot produce an official local window")
        count = min(self.context_frames, length)
        indices = np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
        if (len(np.unique(indices)) != count or indices[0] != 0
                or indices[-1] != length - 1):
            raise RuntimeError("context indices must be unique and cover both endpoints")
        seed_payload = (b"pst-small-global-process-context-v1\x00"
                        + str(self.context_seed).encode("ascii") + b"\x00"
                        + str(self.video_names[video_index]).encode("utf-8"))
        seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:4], "little")
        rng = np.random.RandomState(seed)
        raw = np.zeros((self.context_frames, self.context_points, 3), dtype=np.float64)
        times = np.zeros(self.context_frames, dtype=np.float32)
        valid = np.zeros(self.context_frames, dtype=np.bool_)
        for position, frame_index in enumerate(indices):
            points = np.asarray(video[int(frame_index)])
            if (points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0
                    or points.dtype.kind not in "fiu"
                    or not np.isfinite(points).all()):
                raise ValueError("context frames require finite real numeric [N,3] points")
            if points.shape[0] > self.context_points:
                selected = rng.choice(points.shape[0], size=self.context_points, replace=False)
            else:
                repeat, residue = divmod(self.context_points, points.shape[0])
                selected = rng.choice(points.shape[0], size=residue, replace=False)
                selected = np.concatenate([np.arange(points.shape[0]) for _ in range(repeat)]
                                          + [selected], axis=0)
            raw[position] = points[selected, :]
        times[:count] = indices / float(length - 1)
        valid[:count] = True
        for array in (raw, times, valid, indices):
            array.flags.writeable = False
        result = (raw, times, valid, indices)
        self._context_cache[video_index] = result
        return result

    def __getitem__(self, idx: int):
        index, t = self.index_map[idx]
        video = self.videos[index]
        label = self.labels[index]

        # Keep the exact released local sampling and arithmetic order. Do not
        # replace this with vectorized sampling or a separate local generator.
        clip = [video[t + i * self.step_between_clips] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)
        scales = None
        if self.train:
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales
        clip = clip / 300
        local = clip.astype(np.float32)

        raw, times, valid, _ = self._raw_context(index)
        context = raw.copy() if scales is None else raw * scales
        context = (context / 300).astype(np.float32)
        # Return writable copies so default_collate never wraps read-only data.
        return (ProcessInput(local=local, context=context, times=times.copy(), valid=valid.copy()),
                label, index)


__all__ = ["ProcessDataset", "ProcessInput"]
