"""State and RNG operations preserving the recorded native training arithmetic."""
from __future__ import annotations
from bisect import bisect_right
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch

def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state().clone(),
                torch_cuda=([v.clone() for v in torch.cuda.get_rng_state_all()]
                            if torch.cuda.is_initialized() else None))


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda'] is not None:
        require(torch.cuda.is_initialized(), 'CUDA RNG restore requires initialized CUDA')
        torch.cuda.set_rng_state_all([v.cpu() for v in state['torch_cuda']])


def same_tree(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


@contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        require(state['torch_cuda'] is not None or not torch.cuda.is_initialized(),
                'Initialize CUDA before an operation that may use CUDA under preserve_rng')
        restore_rng(state)


def state_sha(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(str((name, tuple(value.shape), str(value.dtype))).encode('utf8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def split_state(model, *, srr=False):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith('prepool_relation.') == srr}


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Released scheduler, kept algebraically identical for the parent stream."""
    def __init__(self, optimizer, milestones, gamma=.1, warmup_factor=1.0 / 3,
                 warmup_iters=5, warmup_method='linear', last_epoch=-1):
        if not milestones == sorted(milestones):
            raise ValueError('Milestones must be sorted')
        if warmup_method not in ('constant', 'linear'):
            raise ValueError('Unsupported warmup method')
        self.milestones, self.gamma = milestones, gamma
        self.warmup_factor, self.warmup_iters = warmup_factor, warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            else:
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]


class InputRecorder:
    def __init__(self, loader):
        self.loader = loader
        self.reset()

    @property
    def dataset(self):
        return self.loader.dataset

    def __len__(self):
        return len(self.loader)

    def reset(self):
        self.digest = hashlib.sha256()
        self.batches = self.clips = 0

    def __iter__(self):
        for inputs, labels, indices in self.loader:
            for name, value in [*sorted(inputs.items()), ('labels', labels), ('indices', indices)]:
                value = value.detach().cpu().contiguous()
                self.digest.update(name.encode() + b'\0' + str((tuple(value.shape), str(value.dtype))).encode() + b'\0')
                self.digest.update(value.numpy().tobytes())
            self.batches += 1
            self.clips += len(labels)
            yield inputs, labels, indices

    def receipt(self):
        return dict(sha256=self.digest.hexdigest(), batches=self.batches, clips=self.clips,
                    protocol='Every CPU input key, label/index, shape/dtype and bytes in original batch order.')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
