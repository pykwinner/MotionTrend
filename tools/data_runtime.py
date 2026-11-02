"""Local and whole-video point sampling for MotionTrend.

Each source window is exactly T consecutive real frames, enumerated at stride1.
No short local clip is padded, temporally subsampled, repeated or interpolated.
Whole-video context uses8/16/24/32/36 slots and the original name-keyed
private point RNG. Only min(G,n) unique real context frames are valid; other
slots stay zero/invalid. The local clip is never padded or interpolated.
"""
from __future__ import annotations

import hashlib
from copy import copy
from numbers import Integral
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CLIP_FRAMES = (8, 12, 16, 24, 32, 36)
LOCAL_POLICIES = ('natural', 'reference24_span')
GLOBAL_FRAME_OPTIONS = (8, 16, 24, 32, 36)
GLOBAL_FRAMES, LOCAL_POINTS, CONTEXT_POINTS = 32, 2048, 256
TRAIN_CONTEXT_SEED, TEST_CONTEXT_SEED = 0, 0


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_clip_frames(value):
    require(type(value) is int and value in CLIP_FRAMES, 'clip_frames must be8/12/16/24/32/36')
    return value


def validate_global_frames(value):
    require(type(value) is int and value in GLOBAL_FRAME_OPTIONS,
            'global_frames must be8/16/24/32/36')
    return value


def validate_local_policy(value, clip_frames):
    validate_clip_frames(clip_frames)
    require(isinstance(value, str) and value in LOCAL_POLICIES,
            'local_policy must be natural or reference24_span')
    require(value != 'reference24_span' or clip_frames in (8, 12, 16, 24),
            'reference24_span only supports8/12/16/24 real selected frames')
    return value


class ProcessInput(dict):
    @property
    def shape(self):
        return self['local'].shape

    def to(self, device, non_blocking=False):
        if not all(isinstance(value, torch.Tensor) for value in self.values()):
            raise TypeError('ProcessInput.to requires collated tensor values')
        return type(self)({key: value.to(device=device, non_blocking=non_blocking)
                           for key, value in self.items()})


def dataset_contract(raw, *, global_frames=None):
    """Natural denominator/window metadata with stable loaded-video indices."""
    frames = validate_clip_frames(int(raw.frames_per_clip))
    global_frames = validate_global_frames(getattr(raw, 'global_frames', GLOBAL_FRAMES) if global_frames is None else global_frames)
    require(raw.step_between_clips == 1 and raw.num_points == LOCAL_POINTS,
            'Natural protocol requires stride1 and2048 local points')
    require(len(raw.videos) == len(raw.video_names) == len(raw.labels), 'Loaded video metadata mismatch')
    require(len(set(raw.video_names)) == len(raw.video_names), 'Loaded video names must be unique')
    per_video, expected_map = [], []
    for index, (name, video, label) in enumerate(zip(raw.video_names, raw.videos, raw.labels)):
        length = len(video)
        windows = max(0, length - frames + 1)
        expected_map.extend((index, start) for start in range(windows))
        per_video.append(dict(name=str(name), video_index=index, label=int(label), frames=length,
                              windows=windows, eligible=windows > 0))
    require(list(raw.index_map) == expected_map, 'Index map must enumerate every natural consecutive window')
    eligible = [row['name'] for row in per_video if row['eligible']]
    unscored = [row['name'] for row in per_video if not row['eligible']]
    return dict(clip_frames=frames, global_frames=global_frames, window_stride=1,
                local_points=LOCAL_POINTS, context_points=CONTEXT_POINTS,
                loaded_videos=len(per_video), eligible_videos=len(eligible), scored_videos=len(eligible),
                windows=len(expected_map), clips=len(expected_map),
                eligible_names=eligible, unscored_names=unscored, per_video=per_video,
                minimum_local_frames=frames, local_padding=False,
                train=bool(raw.train), num_classes=int(raw.num_classes),
                subject_split='subjects1..5 TRAIN; subjects6..10 TEST',
                protocol='Natural T-consecutive-frame windows at stride1; no short-local padding or temporal subset')


class _SourceSampler(Dataset):
    """Original released point sampling/arithmetic with an actual T-frame clip."""

    def __init__(self, official_dataset, *, clip_frames, global_frames, context_seed):
        super().__init__()
        frames = validate_clip_frames(clip_frames)
        global_frames = validate_global_frames(global_frames)
        required = ('videos', 'video_names', 'labels', 'index_map', 'directory_manifest',
                    'num_classes', 'frames_per_clip', 'step_between_clips', 'num_points', 'train')
        for name in required:
            require(hasattr(official_dataset, name), 'Official dataset is missing ' + name)
        require(official_dataset.frames_per_clip == frames and official_dataset.step_between_clips == 1
                and official_dataset.num_points == LOCAL_POINTS, 'Source dataset must use the actual T/stride1/2048 settings')
        require(not isinstance(context_seed, bool) and isinstance(context_seed, Integral), 'context_seed must be an integer')
        self.official_dataset = official_dataset
        for name in required:
            setattr(self, name, getattr(official_dataset, name))
        self.clip_frames, self.local_frames, self.global_frames = frames, frames, global_frames
        self.local_policy = 'natural'
        self.source_clip_frames = frames
        self.context_frames, self.context_points = global_frames, CONTEXT_POINTS
        self.context_seed = int(context_seed)
        self._context_cache = {}
        self._contract = dataset_contract(official_dataset, global_frames=global_frames)

    def __len__(self):
        return len(self.official_dataset)

    def contract(self):
        result = dataset_contract(self.official_dataset, global_frames=self.global_frames)
        result.update(context_seed=self.context_seed,
                      local_policy=self.local_policy,source_clip_frames=self.source_clip_frames,
                      context_frame_policy=f'rint(linspace(0,video_length-1,min({self.global_frames},video_length))); valid-mask padding to{self.global_frames}',
                      context_point_policy='Original name+context_seed private RandomState,256 points per valid frame')
        return result

    @property
    def eligible_video_names(self):
        return list(self._contract['eligible_names'])

    @property
    def video_lengths(self):
        return [row['frames'] for row in self._contract['per_video']]

    @property
    def video_window_counts(self):
        return [row['windows'] for row in self._contract['per_video']]

    def _raw_context(self, video_index):
        if video_index in self._context_cache:
            return self._context_cache[video_index]
        video = self.videos[video_index]
        length = len(video)
        require(length >= self.frames_per_clip and length >= 2,
                'Context is requested only for a video with an eligible real local clip')
        count = min(self.context_frames, length)
        indices = np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
        require(len(np.unique(indices)) == count and indices[0] == 0 and indices[-1] == length - 1,
                'Context indices must be unique and span both video endpoints')
        seed_payload = (b'pst-small-global-process-context-v1\x00'
                        + str(self.context_seed).encode('ascii') + b'\x00'
                        + str(self.video_names[video_index]).encode('utf-8'))
        seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:4], 'little')
        random = np.random.RandomState(seed)
        raw = np.zeros((self.context_frames, self.context_points, 3), dtype=np.float64)
        times = np.zeros(self.context_frames, dtype=np.float32)
        valid = np.zeros(self.context_frames, dtype=np.bool_)
        for position, frame_index in enumerate(indices):
            points = np.asarray(video[int(frame_index)])
            require(points.ndim == 2 and points.shape[1] == 3 and points.shape[0] > 0
                    and points.dtype.kind in 'fiu' and np.isfinite(points).all(),
                    'Context frames require finite real numeric [N,3] points')
            if points.shape[0] > self.context_points:
                selected = random.choice(points.shape[0], size=self.context_points, replace=False)
            else:
                repeat, residue = divmod(self.context_points, points.shape[0])
                selected = random.choice(points.shape[0], size=residue, replace=False)
                selected = np.concatenate([np.arange(points.shape[0]) for _ in range(repeat)] + [selected], axis=0)
            raw[position] = points[selected, :]
        times[:count] = indices / float(length - 1)
        valid[:count] = True
        for array in (raw, times, valid, indices):
            array.flags.writeable = False
        result = raw, times, valid, indices
        self._context_cache[video_index] = result
        return result

    def __getitem__(self, idx):
        index, start = self.index_map[idx]
        video, label = self.videos[index], self.labels[index]
        clip = [video[start + i * self.step_between_clips] for i in range(self.frames_per_clip)]
        for i, points in enumerate(clip):
            if points.shape[0] > self.num_points:
                selected = np.random.choice(points.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // points.shape[0], self.num_points % points.shape[0]
                selected = np.random.choice(points.shape[0], size=residue, replace=False)
                selected = np.concatenate([np.arange(points.shape[0]) for _ in range(repeat)] + [selected], axis=0)
            clip[i] = points[selected, :]
        clip = np.array(clip)
        scales = None
        if self.train:
            scales = np.random.uniform(.9, 1.1, size=3)
            clip = clip * scales
        clip = clip / 300
        local = clip.astype(np.float32)
        raw, times, valid, _ = self._raw_context(index)
        context = raw.copy() if scales is None else raw * scales
        context = (context / 300).astype(np.float32)
        return ProcessInput(local=local, context=context, times=times.copy(), valid=valid.copy()), label, index


class ClipStudyDataset(_SourceSampler):
    """Actual consecutive local clips with original full-video context alignment."""

    def __init__(self, official_dataset, *, clip_frames=None, global_frames=GLOBAL_FRAMES, context_seed=None):
        clip_frames = official_dataset.frames_per_clip if clip_frames is None else clip_frames
        if context_seed is None:
            context_seed = TRAIN_CONTEXT_SEED if official_dataset.train else TEST_CONTEXT_SEED
        super().__init__(official_dataset, clip_frames=clip_frames, global_frames=global_frames, context_seed=context_seed)

    def __getitem__(self, idx):
        inputs, label, index = super().__getitem__(idx)
        video_index, start = self.index_map[idx]
        length = len(self.videos[video_index])
        frame_ids = start + np.arange(self.frames_per_clip, dtype=np.int64) * self.step_between_clips
        require(length >= 2 and frame_ids.min() >= 0 and frame_ids.max() < length,
                'Local timestamps must refer to existing video frames')
        context, valid = inputs['context'], inputs['valid']
        require(bool(valid.any()), 'Each eligible video needs valid whole-video context')
        center = context[valid].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        require(np.isfinite(center).all(), 'Whole-video context center must be finite')
        centered = np.zeros_like(context)
        centered[valid] = context[valid] - center
        inputs['context'] = centered
        inputs['local_times'] = (frame_ids / float(length - 1)).astype(np.float32)
        inputs['local_to_context_scale'] = np.asarray(1., dtype=np.float32)
        inputs['local_to_context_offset'] = -center
        return inputs, label, index


class ReferenceSpanDataset(ClipStudyDataset):
    """Select L unique real frames only after original24 point/scale draws.

    The source still enumerates and samples consecutive24-frame windows. Thus
    TRAIN point RNG and the single anisotropic scale draw match original24
    exactly, independent of the number of returned local frames.
    """
    def __init__(self, official_dataset, *, clip_frames=24, global_frames=GLOBAL_FRAMES, context_seed=None):
        validate_local_policy('reference24_span', clip_frames)
        super().__init__(official_dataset,clip_frames=24,global_frames=global_frames,context_seed=context_seed)
        self.local_policy='reference24_span'
        self.clip_frames=self.local_frames=clip_frames
        self.source_clip_frames=24
        self.span_indices=np.rint(np.linspace(0,23,clip_frames)).astype(np.int64)
        require(len(np.unique(self.span_indices))==clip_frames and self.span_indices[0]==0 and self.span_indices[-1]==23,
                'Reference24 span must contain unique real frame IDs spanning0..23')
        self.span_indices.flags.writeable=False

    def __getitem__(self,idx):
        inputs,label,index=super().__getitem__(idx)
        # Local and true normalized time are subset using the same frame IDs.
        # Context, affine transform and NumPy state stay exactly original24.
        if self.local_frames!=24:
            inputs['local']=inputs['local'][self.span_indices]
            inputs['local_times']=inputs['local_times'][self.span_indices]
        return inputs,label,index

    def contract(self):
        result=super().contract()
        result.update(clip_frames=self.local_frames,source_clip_frames=24,
                      local_policy='reference24_span',local_span_frames=24,
                      local_frame_offsets=self.span_indices.tolist(),
                      point_rng_sampled_frames_per_window=24,minimum_local_frames=24,
                      protocol='Original24 consecutive-window index map; sample original24 points/scale, then select unique rint(linspace(0,23,L)) real frames')
        return result


class WindowDataset(Dataset):
    """Official subject split and natural range(0,nframes-T+1) enumeration."""

    def __init__(self, root, *, train, clip_frames=24, global_frames=GLOBAL_FRAMES, ordered_names=None):
        self.root = Path(root).resolve()
        frames = validate_clip_frames(clip_frames)
        global_frames = validate_global_frames(global_frames)
        require(type(train) is bool, 'train must be Boolean')
        names = list(os.listdir(self.root)) if ordered_names is None else list(ordered_names)
        require(names and len(names) == len(set(names)), 'Require a nonempty unique ordered dataset manifest')
        self.directory_manifest = names
        self.videos, self.video_names, self.labels, self.index_map = [], [], [], []
        for name in names:
            require(isinstance(name, str) and Path(name).name == name and name.endswith('.npz')
                    and not any(c in name for c in ('\\', ':', '\x00')), 'Unsafe NPZ filename')
            subject = int(name.split('_')[1].split('s')[1])
            if (subject <= 5) != train:
                continue
            with np.load(self.root / name, allow_pickle=True) as archive:
                video = archive['point_clouds']
            index = len(self.videos)
            self.videos.append(video)
            self.video_names.append(name)
            self.labels.append(int(name.split('_')[0][1:]) - 1)
            self.index_map.extend((index, start) for start in range(0, len(video) - frames + 1, 1))
        require(self.labels, 'Subject split is empty')
        self.frames_per_clip, self.clip_frames, self.step_between_clips, self.num_points = frames, frames, 1, LOCAL_POINTS
        self.train, self.num_classes = train, max(self.labels) + 1
        self.global_frames = global_frames
        self._contract = dataset_contract(self)

    def __len__(self):
        return len(self.index_map)

    def contract(self):
        return dataset_contract(self)


def make_dataset(data, *, train, clip_frames=24, global_frames=GLOBAL_FRAMES, context_seed=None,
                 ordered_names=None, local_policy='natural'):
    validate_local_policy(local_policy,clip_frames)
    source_frames=24 if local_policy=='reference24_span' else clip_frames
    raw = WindowDataset(data, train=train, clip_frames=source_frames, global_frames=global_frames, ordered_names=ordered_names)
    return from_raw(raw,clip_frames=clip_frames,global_frames=global_frames,context_seed=context_seed,local_policy=local_policy)


def from_raw(official_dataset, *, clip_frames=24, global_frames=GLOBAL_FRAMES, context_seed=None,local_policy='natural'):
    """Share a verified official source while independently indexing natural T clips.

    Video arrays, labels and filename order are shared read-only in use. Only the
    shallow view's clip metadata/index map change; the original24 dataset remains
    usable by its original DataLoader. Sampling is performed by this module and
    never calls the official source's __getitem__.
    """
    frames = validate_clip_frames(clip_frames)
    validate_local_policy(local_policy,frames)
    source_frames=24 if local_policy=='reference24_span' else frames
    global_frames = validate_global_frames(global_frames)
    require(official_dataset.step_between_clips == 1 and official_dataset.num_points == LOCAL_POINTS,
            'Natural raw view requires verified stride1/2048 source settings')
    raw = copy(official_dataset)
    raw.frames_per_clip = raw.clip_frames = source_frames
    raw.global_frames = global_frames
    raw.index_map = [(index, start) for index, video in enumerate(raw.videos)
                     for start in range(max(0, len(video) - source_frames + 1))]
    raw._contract = dataset_contract(raw, global_frames=global_frames)
    cls=ReferenceSpanDataset if local_policy=='reference24_span' else ClipStudyDataset
    return cls(raw, clip_frames=frames, global_frames=global_frames, context_seed=context_seed)


__all__ = ['ClipStudyDataset', 'ReferenceSpanDataset', 'WindowDataset', 'ProcessInput', 'make_dataset', 'from_raw',
           'dataset_contract', 'CLIP_FRAMES', 'GLOBAL_FRAME_OPTIONS', 'LOCAL_POLICIES',
           'validate_clip_frames', 'validate_global_frames', 'validate_local_policy']
