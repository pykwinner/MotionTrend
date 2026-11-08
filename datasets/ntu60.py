"""NTU60 official local sampling plus independently sampled CMTR context.

Both training and evaluation local clips use the ordinary advancing NumPy RNG.
Context never consumes that stream. Whole-video context makes inference offline.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
from pathlib import Path
import re
import numpy as np
from torch.utils.data import Dataset

CROSS_SUBJECT_TRAIN = (1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38)
CROSS_VIEW_TRAIN_CAMERAS = (2, 3)
NAME = re.compile(r'^S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})R(?P<repeat>\d{3})A(?P<action>\d{3})$')


def _context_rng(seed, name):
    digest = hashlib.sha256(f'ntu-ordered-context-v1\0{seed}\0{name}'.encode('utf8')).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], 'little'))


def clip_normalize(clip):
    """Keep the released arithmetic and dtype until the final float32 cast."""
    pc = np.reshape(a=clip, newshape=[-1, 3])
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    radius = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('NTU local clip has zero or nonfinite normalization radius')
    return (clip - centroid) / radius


def _sample(points, count, rng):
    # Preserve even the zero-size remainder choice in the official RNG sequence.
    if len(points) > count:
        selected = rng.choice(len(points), size=count, replace=False)
    else:
        repeat, residue = divmod(count, len(points))
        selected = rng.choice(len(points), size=residue, replace=False)
        selected = np.concatenate([np.arange(len(points)) for _ in range(repeat)] + [selected], axis=0)
    return points[selected, :]


class NTUDataset(Dataset):
    def __init__(self, data_root, meta=None, *, split='xsub', train=True,
                 frames_per_clip=24, step_between_clips=2, num_points=2048,
                 context_frames=32, context_points=256, context_seed=0,
                 cache_size=2, file_order=None):
        if split not in ('xsub', 'xview') or type(train) is not bool:
            raise ValueError('split must be xsub/xview and train must be Boolean')
        for key, value in dict(frames_per_clip=frames_per_clip, step_between_clips=step_between_clips,
                               num_points=num_points, context_frames=context_frames,
                               context_points=context_points).items():
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if context_frames > 32:
            raise ValueError('CMTR supports at most 32 context frames')
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError('cache_size must be a nonnegative integer')
        if type(context_seed) is not int or not 0 <= context_seed < 2**63:
            raise ValueError('context_seed must be an integer in [0, 2**63)')
        supplied = Path(data_root).expanduser().resolve()
        self.root = supplied / 'video' if (supplied / 'video').is_dir() else supplied
        if meta is None:
            candidates = [supplied / 'ntu60.list', supplied.parent / 'ntu60.list']
            meta = next((p for p in candidates if p.is_file()), candidates[0])
        self.meta = Path(meta).expanduser().resolve()
        if not self.root.is_dir() or not self.meta.is_file():
            raise FileNotFoundError('Provide an NTU NPZ directory and its ntu60.list metadata')
        self.split, self.train = split, train
        self.frames_per_clip, self.step_between_clips, self.num_points = frames_per_clip, step_between_clips, num_points
        self.context_frames, self.context_points, self.context_seed = context_frames, context_points, context_seed
        self.cache_size, self.num_classes = cache_size, 60
        entries = OrderedDict()
        for number, line in enumerate(self.meta.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f'Metadata line {number} must contain name and frame count')
            name, count = fields
            match = NAME.fullmatch(name)
            if match is None or name in entries:
                raise ValueError(f'Invalid or duplicate NTU name on line {number}: {name}')
            try:
                count = int(count)
            except ValueError as error:
                raise ValueError(f'Invalid frame count on line {number}') from error
            ids = {key: int(value) for key, value in match.groupdict().items()}
            if count < 1 or not (1 <= ids['action'] <= 60 and 1 <= ids['camera'] <= 3
                                  and 1 <= ids['subject'] <= 40 and ids['setup'] > 0 and ids['repeat'] > 0):
                raise ValueError(f'Invalid NTU60 metadata on line {number}: {name}')
            entries[name] = (count, ids)
        if file_order is not None:
            order = list(file_order)
            if len(order) != len(set(order)) or set(order) != set(entries):
                raise ValueError('Saved metadata order and current NTU video names differ')
            entries = OrderedDict((name, entries[name]) for name in order)
        self.directory_manifest = list(entries)
        self.videos, self.video_names, self.labels, self.lengths, self.index_map = [], [], [], [], []
        for name, (length, ids) in entries.items():
            is_train = ids['subject'] in CROSS_SUBJECT_TRAIN if split == 'xsub' else ids['camera'] in CROSS_VIEW_TRAIN_CAMERAS
            if is_train != train:
                continue
            path = self.root / (name + '.npz')
            if not path.is_file():
                raise FileNotFoundError(f'NTU metadata references missing point cloud: {path}')
            index = len(self.videos)
            self.videos.append(path)
            self.video_names.append(name)
            self.labels.append(ids['action'] - 1)
            self.lengths.append(length)
            # Official rule: same step for frame sampling and window starts.
            # Short videos have no windows, rather than artificial padded clips.
            self.index_map.extend((index, start) for start in range(
                0, length - step_between_clips * (frames_per_clip - 1), step_between_clips))
        if not self.videos:
            raise ValueError(f'No videos for NTU {split}, train={train}')
        usable = {index for index, _ in self.index_map}
        self.metadata = dict(split=split, train=train, listed_videos=len(self.videos),
                             usable_videos=len(usable), windows=len(self.index_map), num_classes=60,
                             frames_per_clip=frames_per_clip, step_between_clips=step_between_clips,
                             num_points=num_points, context_frames=context_frames,
                             context_points=context_points, context_seed=context_seed,
                             cache_videos_per_worker=cache_size,
                             unscored_videos=[name for i, name in enumerate(self.video_names) if i not in usable],
                             local_sampling='ordinary advancing NumPy RNG in both train and evaluation',
                             context_sampling='separate fixed per-video RNG', offline_whole_video_context=True)
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.index_map)

    def load(self, video_index):
        if video_index in self.cache:
            self.cache.move_to_end(video_index)
            return self.cache[video_index]
        path, name = self.videos[video_index], self.video_names[video_index]
        with np.load(path, allow_pickle=True) as archive:
            if 'data' not in archive:
                raise ValueError(f'NTU archive lacks data: {name}')
            raw = archive['data']
        if raw.ndim == 0 or len(raw) != self.lengths[video_index]:
            raise ValueError(f'NTU metadata frame count differs from NPZ: {name}')
        for i, points in enumerate(raw):
            if (not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3
                    or not len(points) or not np.issubdtype(points.dtype, np.floating)):
                raise ValueError(f'Require nonempty floating XYZ frame: {name}, frame {i}')
            if not np.isfinite(points).all():
                raise ValueError(f'Nonfinite XYZ frame: {name}, frame {i}')
        # Integer input is rejected before multiplication to prevent overflow.
        video = raw * 100
        length = len(video)
        ids = np.rint(np.linspace(0, length - 1, min(self.context_frames, length))).astype(np.int64)
        context = np.zeros((self.context_frames, self.context_points, 3), dtype=np.float64)
        times = np.zeros(self.context_frames, dtype=np.float32)
        valid = np.zeros(self.context_frames, dtype=bool)
        rng = _context_rng(self.context_seed, name)
        for j, i in enumerate(ids):
            context[j] = _sample(video[int(i)], self.context_points, rng)
        valid[:len(ids)] = True
        times[:len(ids)] = ids / max(1, length - 1)
        center = context[valid].reshape(-1, 3).mean(axis=0)
        radius = np.sqrt(((context[valid] - center) ** 2).sum(axis=-1)).max()
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f'Degenerate whole-video context: {name}')
        context[valid] = (context[valid] - center) / radius
        item = video, context, times, valid, center, radius
        if self.cache_size:
            self.cache[video_index] = item
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return item

    def __getitem__(self, item):
        video_index, start = self.index_map[item]
        video, context, times, valid, context_center, context_radius = self.load(video_index)
        frame_ids = start + np.arange(self.frames_per_clip) * self.step_between_clips
        clip = np.array([_sample(video[int(i)], self.num_points, np.random) for i in frame_ids])
        flat = clip.reshape(-1, 3)
        center = np.mean(flat, axis=0)
        radius = np.max(np.sqrt(np.sum((flat - center) ** 2, axis=1)))
        clip = clip_normalize(clip)
        scales = np.ones(3, dtype=np.float64)
        if self.train:
            scales = np.random.uniform(.9, 1.1, size=3)
            clip = clip * scales
            context = context * scales
        inputs = dict(local=clip.astype(np.float32), context=context.astype(np.float32),
                      times=times.copy(), valid=valid.copy(),
                      local_times=(frame_ids / max(1, len(video) - 1)).astype(np.float32),
                      local_to_context_scale=np.asarray(radius / context_radius, dtype=np.float32),
                      local_to_context_offset=(((center - context_center) / context_radius) * scales).astype(np.float32))
        return inputs, self.labels[video_index], video_index


__all__ = ['NTUDataset', 'clip_normalize', 'CROSS_SUBJECT_TRAIN', 'CROSS_VIEW_TRAIN_CAMERAS']
