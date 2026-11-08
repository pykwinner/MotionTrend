"""Synthia4D unified dataset (single file).

Merges the baseline SegDataset processing into one class:
  - metadata txt (sequence/point_clouds/frame.ply as first field) -> per-frame npz
    (pc [16384,3] / rgb [16384,3] / semantic / center)
  - frame stack: current + previous (fallback to most recent available)
  - half crop with context=1.0 around center; loss masks per half
  - label conversion via index_to_label; void/unsupported labels -> 0, masked out
  - per-frame shuffle to num_points; train-time flip/scale/rot augmentation
  - label weights: 1/log(1.2+w) normalized (train) / ones (eval)

__getitem__ returns a dict batch (default_collate friendly):
  halves: [pc1, rgb1, label1, mask1, pc2, rgb2, label2, mask2]
    pc   float32 [3, N, 3]      (world coords, augmented about current-frame center)
    rgb  float32 [3, 3, N]      (channel-first, matches baseline)
    label int64   [3, N]
    mask float32 [3, N]
  times  float32 [3]  actual supplied frame ID / sequence maximum ID (always finite)
  valid  bool    [3]

Missing history reuses the most recent available frame and its timestamp;
valid=False excludes that duplicate from CMTR memory while keeping local queries finite.
"""

from __future__ import annotations

from pathlib import Path
from collections import OrderedDict

import numpy as np
from torch.utils.data import Dataset

CLASS_NAMES = [
    "Building", "Road", "Sidewalk", "Fence", "Vegetation", "Pole",
    "Car", "Traffic Sign", "Pedestrian", "Bicycle", "Lanemarking", "Traffic Light",
]
NUM_CLASSES = len(CLASS_NAMES)

index_to_label = np.array([12, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -1, 11], dtype="int32")


def load_labelweights(path, train: bool) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if 'labelweights' not in archive:
            raise ValueError('Label weight archive must contain labelweights')
        w = archive['labelweights'].astype(np.float32)
    if w.shape != (NUM_CLASSES,) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Require 12 finite nonnegative class frequencies')
    if train:
        w = 1.0 / np.log(1.2 + w)
        w = w / w.min()
    else:
        w = np.ones_like(w)
    return w


class Synthia4DDataset(Dataset):
    """One sample = 3-frame clip, evaluated as two half-crops (baseline protocol)."""

    def __init__(self, root, meta, labelweight, frames_per_clip=3,
                 num_points=16384, train=True, context=1.0, cache_size=8,
                 with_metadata=True):
        super().__init__()
        if type(train) is not bool or type(with_metadata) is not bool:
            raise ValueError('train and with_metadata must be Boolean')
        if type(frames_per_clip) is not int or frames_per_clip < 1:
            raise ValueError('frames_per_clip must be a positive integer')
        if type(num_points) is not int or num_points < 1:
            raise ValueError('num_points must be a positive integer')
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError('cache_size must be a nonnegative integer')
        if not np.isfinite(context) or context <= 0:
            raise ValueError('context must be finite and positive')
        self.root = Path(root).expanduser().resolve()
        self.meta_path = Path(meta).expanduser().resolve()
        self.labelweight_path = Path(labelweight).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f'Missing processed point-cloud directory: {self.root}')
        self.train = train
        self.num_classes = NUM_CLASSES
        self.frames_per_clip = frames_per_clip
        self.num_points = num_points
        self.context = float(context)
        self.cache_size, self.with_metadata = cache_size, with_metadata
        self.labelweights = load_labelweights(self.labelweight_path, train)
        self.cache = OrderedDict()

        self.meta = []
        self.data = {}
        with self.meta_path.open('r', encoding='utf-8-sig') as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                fields = line.split()[0].replace('\\', '/').split('/')
                sequence_name = fields[0]
                if len(fields) < 2 or sequence_name in ('', '.', '..'):
                    raise ValueError(f'Invalid sequence path on metadata line {number}')
                try:
                    frame_id = int(fields[-1].split('.')[0])
                except ValueError as error:
                    raise ValueError(f'Invalid frame id on metadata line {number}') from error
                if frame_id < 0:
                    raise ValueError(f'Negative frame id on metadata line {number}')
                key = f"{sequence_name}-{frame_id:06d}"
                if key in self.data:
                    continue
                path = self.root / f'{key}.npz'
                if not path.is_file():
                    raise FileNotFoundError(f'Missing listed point-cloud frame: {path}')
                self.data[key] = path
                self.meta.append((sequence_name, frame_id))
        self.meta.sort()
        if not self.meta:
            raise RuntimeError(f"No frames listed in {meta}")
        self.sequence_max = {}
        for sequence, frame in self.meta:
            self.sequence_max[sequence] = max(self.sequence_max.get(sequence, 0), frame)
        self.metadata_summary = dict(frames=len(self.meta), sequences=len(self.sequence_max),
                                     frames_per_clip=frames_per_clip, num_points=num_points,
                                     context=self.context, cache_size=cache_size,
                                     sampling='ordinary NumPy RNG in train and evaluation',
                                     crop_center='current frame',
                                     ignored_raw_semantic_ids=[0, 1, 13, 14])

    def _load(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        with np.load(self.data[key], allow_pickle=False) as payload:
            required = {'pc', 'rgb', 'semantic', 'center'}
            if not required.issubset(payload.files):
                raise ValueError(f'{key}: missing pc/rgb/semantic/center')
            pc, rgb = payload['pc'].astype(np.float32), payload['rgb'].astype(np.float32)
            semantic, center = payload['semantic'], payload['center'].astype(np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3 or not len(pc) or rgb.shape != pc.shape:
            raise ValueError(f'{key}: invalid or empty pc/rgb')
        if semantic.shape != (len(pc),) or center.shape != (3,):
            raise ValueError(f'{key}: invalid semantic/center shape')
        if not all(np.isfinite(x).all() for x in (pc, rgb, center)):
            raise ValueError(f'{key}: nonfinite pc/rgb/center')
        if not np.issubdtype(semantic.dtype, np.integer) or ((semantic < 0) | (semantic > 15)).any():
            raise ValueError(f'{key}: raw semantic IDs must be integers in [0,15]')
        item = pc, rgb, semantic.astype(np.uint8), center
        if self.cache_size:
            self.cache[key] = item
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return item

    def __len__(self):
        return len(self.meta)

    # ---- frame stack: current + previous frames (fallback to most recent) ----

    def _read_clip(self, index):
        sequence_name, frame_id = self.meta[index]
        pcs, rgbs, sems = [], [], []
        most_recent = -1
        current_center = None
        for diff in range(self.frames_per_clip):
            key = f"{sequence_name}-{frame_id - diff:06d}"
            if key in self.data:
                pc, rgb, sem, center = self._load(key)
                most_recent = frame_id - diff
            else:
                pc, rgb, sem, center = self._load(f"{sequence_name}-{most_recent:06d}")
            if diff == 0:
                current_center = center
            pcs.append(pc)
            rgbs.append(rgb)
            sems.append(sem)
        if len({len(pc) for pc in pcs}) != 1:
            raise ValueError('Preprocessed frames in one clip must have equal point counts')
        return np.stack(pcs), np.stack(rgbs), np.stack(sems), current_center

    # ---- half crop with context (baseline geometry) ----

    def _half_crop(self, half, pc, rgb, sem, center):
        idx_all = np.arange(pc.shape[1])
        keep, loss = [], []
        for f in range(self.frames_per_clip):
            if half == 0:
                sel = idx_all[pc[f, :, 2] > (center[2] - self.context)]
                loss.append(pc[f, sel, 2] > center[2])
            else:
                sel = idx_all[pc[f, :, 2] < (center[2] + self.context)]
                loss.append(pc[f, sel, 2] < center[2])
            if not len(sel):
                raise ValueError(f'Empty half-crop: half={half}, clip_frame={f}; check coordinates/center/context')
            keep.append(sel)
        pc_h = [pc[f, s] for f, s in enumerate(keep)]
        rgb_h = [rgb[f, s] for f, s in enumerate(keep)]
        sem_h = [sem[f, s] for f, s in enumerate(keep)]
        return pc_h, rgb_h, sem_h, loss

    @staticmethod
    def _labels(sem, loss_masks):
        labels, masks = [], []
        for s, lm in zip(sem, loss_masks):
            label = index_to_label[s.astype(np.int64)].astype(np.int64)
            supported = (label >= 0) & (label < NUM_CLASSES)
            mask = (supported & lm).astype(np.float32)
            # Masking happens after CE in the trainer, so every label must be safe.
            label[~supported] = 0
            labels.append(label)
            masks.append(mask)
        return labels, masks

    def _to_num_points(self, pc, rgb, label, mask):
        for f in range(self.frames_per_clip):
            n = pc[f].shape[0]
            if n > self.num_points:
                idx = np.random.choice(n, self.num_points, replace=False)
            else:
                idx = np.concatenate([np.random.choice(n, self.num_points - n),
                                      np.arange(n)])
            pc[f], rgb[f], label[f], mask[f] = pc[f][idx], rgb[f][idx], label[f][idx], mask[f][idx]
        return (np.stack(pc).astype(np.float32), np.stack(rgb).astype(np.float32),
                np.stack(label).astype(np.int64), np.stack(mask).astype(np.float32))

    def _augment(self, pc, center):
        if np.random.uniform() > 0.5:                       # flip around center x
            pc = pc - center
            pc[..., 0] *= -1.0
            pc = pc + center
        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center
        angle = np.random.uniform(0.0, 2.0 * np.pi)         # rotate about y through center
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
        return (pc - center) @ rot.T + center

    def _half(self, half, pc, rgb, sem, center):
        pc_h, rgb_h, sem_h, loss = self._half_crop(half, pc, rgb, sem, center)
        label_h, mask_h = self._labels(sem_h, loss)
        pc_h, rgb_h, label_h, mask_h = self._to_num_points(pc_h, rgb_h, label_h, mask_h)
        if self.train:
            pc_h = self._augment(pc_h, center)
        return pc_h, np.swapaxes(rgb_h, 1, 2), label_h, mask_h

    def __getitem__(self, index):
        pc, rgb, sem, center = self._read_clip(index)
        pc1, rgb1, label1, mask1 = self._half(0, pc, rgb, sem, center)
        pc2, rgb2, label2, mask2 = self._half(1, pc, rgb, sem, center)

        sequence_name, frame_id = self.meta[index]
        denom = max(self.sequence_max[sequence_name], 1)
        times, valid = [], []
        most_recent = frame_id
        for diff in range(self.frames_per_clip):
            key = f"{sequence_name}-{frame_id - diff:06d}"
            ok = key in self.data
            if ok:
                most_recent = frame_id - diff
            valid.append(ok)
            # Match _read_clip's actual fallback source, including sequence starts.
            # Invalid history is masked as memory, but its local query time must
            # remain finite for the shared CMTR and the segmentation backbone.
            times.append(most_recent / float(denom))

        result = dict(halves=[pc1, rgb1, label1, mask1, pc2, rgb2, label2, mask2])
        if self.with_metadata:
            result.update(times=np.asarray(times, dtype=np.float32), valid=np.asarray(valid, dtype=np.bool_))
        return result


__all__ = ["Synthia4DDataset", "CLASS_NAMES", "NUM_CLASSES", "load_labelweights"]
