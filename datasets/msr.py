"""Official MSR-Action3D local sampling with MoTrend whole-video context.

The subject split, stride-one windows, local point draws, scale augmentation,
and /300 conversion follow the released PST MSR loader. Both train and test
local point sets are drawn again on every read from the normal NumPy stream.
Only the added whole-video context uses a private, per-video cached sampler.
"""
from __future__ import annotations

import hashlib
import os
import re
from numbers import Integral
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


ACTIONS = (
    "high_arm_wave", "horizontal_arm_wave", "hammer", "hand_catch",
    "forward_punch", "high_throw", "draw_x", "draw_tick", "draw_circle",
    "hand_clap", "two_hand_wave", "side_boxing", "bend", "forward_kick",
    "side_kick", "jogging", "tennis_swing", "tennis_serve", "golf_swing",
    "pickup_throw",
)
NUM_CLASSES = len(ACTIONS)
_NAME = re.compile(r"a(\d+)_s(\d+)_e(\d+)(?:_sdepth)?\.npz", re.IGNORECASE)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _file_order(root: Path, requested):
    # Preserve the official directory order; do not silently sort or filter
    # individual videos based on their labels, predictions or success rate.
    available = [name for name in os.listdir(root)
                 if name.endswith(".npz") and (root / name).is_file()]
    if requested is None:
        return available
    if not isinstance(requested, (list, tuple)):
        raise ValueError("file_order must be the complete saved filename list")
    names = list(requested)
    if any(not isinstance(name, str) or Path(name).name != name for name in names):
        raise ValueError("file_order entries must be plain filenames")
    if len(names) != len(set(names)):
        raise ValueError("file_order contains duplicate filenames")
    if set(names) != set(available):
        missing = sorted(set(names) - set(available))
        added = sorted(set(available) - set(names))
        raise ValueError(f"Dataset files changed: missing={missing}, added={added}")
    return names


class MSRDataset(Dataset):
    """One item is one official local window plus aligned global context.

    ``seed`` is retained as run metadata only. The caller seeds NumPy/Torch
    once, and DataLoader seeds its workers normally. It is never used to reset
    a per-window RNG. ``context_seed`` controls only the added global branch.

    Short videos stay in ``videos``/``video_names`` but create no windows, just
    as in the official loader. ``metadata`` reports both loaded and scored
    video counts; the evaluator must use the actually observed videos.
    """

    def __init__(self, data_root, split="train", clip_len=24, num_points=2048,
                 context_frames=32, context_points=256, seed=0, context_seed=0,
                 file_order=None):
        super().__init__()
        self.root = Path(data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"MSR dataset directory does not exist: {self.root}")
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        for value, name in ((seed, "seed"), (context_seed, "context_seed")):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        self.seed, self.context_seed = int(seed), int(context_seed)
        self.split, self.train = split, split == "train"
        self.clip_len = self.frames_per_clip = _positive_int(clip_len, "clip_len")
        if self.clip_len < 2:
            raise ValueError("clip_len must be at least 2 for video timestamps")
        self.step_between_clips = 1
        self.num_points = _positive_int(num_points, "num_points")
        self.context_frames = _positive_int(context_frames, "context_frames")
        self.context_points = _positive_int(context_points, "context_points")
        if not 2 <= self.context_frames <= 32:
            raise ValueError("context_frames must be between 2 and 32")
        self.directory_manifest = _file_order(self.root, file_order)
        if not self.directory_manifest:
            raise ValueError(f"No NPZ videos found in {self.root}")

        self.videos, self.video_names, self.labels, self.index_map = [], [], [], []
        for name in self.directory_manifest:
            match = _NAME.fullmatch(name)
            if match is None:
                raise ValueError(f"Unexpected MSR filename: {name}")
            action, subject, _ = map(int, match.groups())
            if not 1 <= action <= NUM_CLASSES or not 1 <= subject <= 10:
                raise ValueError(f"MSR action/subject out of range: {name}")
            if (subject <= 5) != self.train:
                continue
            with np.load(self.root / name, allow_pickle=True) as archive:
                if "point_clouds" not in archive:
                    raise ValueError(f"{name} has no point_clouds array")
                video = archive["point_clouds"]
            # Validate without converting raw coordinates to float32 early.
            # Their original dtype participates in the official /300 math.
            for frame_index, points in enumerate(video):
                points = np.asarray(points)
                if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
                        or points.dtype.kind not in "fiu" or not np.isfinite(points).all()):
                    raise ValueError(f"{name}, frame {frame_index}: expected finite numeric [N,3] points")
            video_index = len(self.videos)
            self.videos.append(video)
            self.video_names.append(name)
            self.labels.append(action - 1)
            for start in range(0, len(video) - (self.frames_per_clip - 1)):
                self.index_map.append((video_index, start))

        self.num_classes = max(self.labels) + 1 if self.labels else NUM_CLASSES
        self._context_cache = {}
        scored_indices = {index for index, _ in self.index_map}
        self.scored_video_names = [name for i, name in enumerate(self.video_names)
                                   if i in scored_indices]
        self.unscored_video_names = [name for i, name in enumerate(self.video_names)
                                     if i not in scored_indices]
        self.metadata = dict(
            dataset="MSRAction3D", split=self.split,
            files=len(self.directory_manifest), videos=len(self.video_names),
            scored_videos=len(scored_indices),
            unscored_videos=len(self.video_names) - len(scored_indices),
            windows=len(self.index_map), clip_len=self.clip_len,
            num_points=self.num_points, step_between_clips=1,
            context_frames=self.context_frames, context_points=self.context_points,
            context_seed=self.context_seed,
        )

    def __len__(self):
        return len(self.index_map)

    def _raw_context(self, video_index):
        """Cached unaugmented context; never consumes the local NumPy stream."""
        if video_index in self._context_cache:
            return self._context_cache[video_index]
        video = self.videos[video_index]
        length = len(video)
        if length < self.frames_per_clip:
            raise ValueError("A short video cannot supply a local window")
        count = min(self.context_frames, length)
        frame_ids = np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
        payload = (b"pst-small-global-process-context-v1\x00"
                   + str(self.context_seed).encode("ascii") + b"\x00"
                   + self.video_names[video_index].encode("utf-8"))
        private_seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
        rng = np.random.RandomState(private_seed)
        raw = np.zeros((self.context_frames, self.context_points, 3), dtype=np.float64)
        times = np.zeros(self.context_frames, dtype=np.float32)
        valid = np.zeros(self.context_frames, dtype=np.bool_)
        for position, frame_index in enumerate(frame_ids):
            points = video[int(frame_index)]
            if len(points) > self.context_points:
                selected = rng.choice(len(points), size=self.context_points, replace=False)
            else:
                repeat, residue = divmod(self.context_points, len(points))
                selected = rng.choice(len(points), size=residue, replace=False)
                selected = np.concatenate([np.arange(len(points)) for _ in range(repeat)]
                                          + [selected], axis=0)
            raw[position] = points[selected, :]
        times[:count] = frame_ids / float(length - 1)
        valid[:count] = True
        for array in (raw, times, valid):
            array.flags.writeable = False
        self._context_cache[video_index] = raw, times, valid
        return self._context_cache[video_index]

    def __getitem__(self, idx):
        index, start = self.index_map[idx]
        video, label = self.videos[index], self.labels[index]

        # Keep the original operations and random-call order, including
        # choice(size=0), which also advances NumPy's state for replace=False.
        clip = [video[start + i] for i in range(self.frames_per_clip)]
        for i, points in enumerate(clip):
            if points.shape[0] > self.num_points:
                selected = np.random.choice(points.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = divmod(self.num_points, points.shape[0])
                selected = np.random.choice(points.shape[0], size=residue, replace=False)
                selected = np.concatenate([np.arange(points.shape[0]) for _ in range(repeat)]
                                          + [selected], axis=0)
            clip[i] = points[selected, :]
        clip = np.array(clip)
        scales = None
        if self.train:
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales
        clip = clip / 300
        local = clip.astype(np.float32)

        raw, times, valid = self._raw_context(index)
        context = raw.copy() if scales is None else raw * scales
        context = (context / 300).astype(np.float32)
        center = context[valid].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        centered = np.zeros_like(context)
        centered[valid] = context[valid] - center
        frame_ids = start + np.arange(self.frames_per_clip, dtype=np.int64)
        inputs = dict(
            local=local, context=centered, times=times.copy(), valid=valid.copy(),
            local_times=(frame_ids / float(len(video) - 1)).astype(np.float32),
            local_to_context_scale=np.asarray(1.0, dtype=np.float32),
            local_to_context_offset=-center,
        )
        return inputs, label, index


__all__ = ["MSRDataset", "ACTIONS", "NUM_CLASSES"]
