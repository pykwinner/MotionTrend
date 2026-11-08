"""One explicit configuration for the single-stage MSR experiment."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config/msr.json'
DEFAULTS = {
    'data': {
        'root': None, 'clip_len': 24, 'num_points': 2048,
        'context_frames': 32, 'context_points': 256, 'context_seed': 0,
        'expected_videos': 287,
    },
    'model': {
        'use_ebte': True, 'use_cmtr': True, 'use_srrf': False,
        'seed': 0, 'num_classes': 20,
        'radius': .3, 'nsamples': 32, 'spatial_stride': 32,
        'temporal_kernel_size': 3, 'temporal_stride': 2,
        'dim': 80, 'depth': 5, 'heads': 2, 'dim_head': 40, 'mlp_dim': 160,
        'dropout1': 0., 'dropout2': 0.,
        'rho': .02, 'epsilon': 1e-6, 'srrf_scale': 1.,
    },
    'train': {
        'seed': 0, 'epochs': 50, 'batch_size': 14, 'workers': 10,
        'lr': .01, 'momentum': .9, 'weight_decay': 1e-4,
        'warmup_epochs': 10, 'warmup_factor': 1e-5,
        'lr_milestones': [20, 30], 'lr_gamma': .1, 'print_freq': 100,
    },
}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def validate_config(config):
    for name in ('clip_len', 'num_points', 'context_frames', 'context_points'):
        _integer(config['data'][name], 'data.' + name)
    _integer(config['data']['context_seed'], 'data.context_seed', 0)
    _integer(config['data']['expected_videos'], 'data.expected_videos', 0)
    train = config['train']
    for name in ('seed', 'workers', 'warmup_epochs'):
        _integer(train[name], 'train.' + name, 0)
    for name in ('epochs', 'batch_size', 'print_freq'):
        _integer(train[name], 'train.' + name)
    for name in ('lr', 'momentum', 'weight_decay', 'warmup_factor', 'lr_gamma'):
        value = train[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'train.{name} must be finite')
    if train['lr'] <= 0 or not 0 <= train['momentum'] < 1 or train['weight_decay'] < 0:
        raise ValueError('Invalid SGD settings')
    if not 0 < train['warmup_factor'] <= 1 or not 0 < train['lr_gamma'] <= 1:
        raise ValueError('warmup_factor and lr_gamma must be in (0,1]')
    milestones = train['lr_milestones']
    if (not isinstance(milestones, list) or any(type(v) is not int or v < 1 for v in milestones)
            or milestones != sorted(set(milestones))):
        raise ValueError('lr_milestones must contain distinct increasing positive epochs')
    if config['model']['seed'] != train['seed']:
        raise ValueError('Use the same model.seed and train.seed; --seed sets both')
    return config


def load_config(path=DEFAULT_CONFIG):
    supplied = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError('Configuration must contain only data, model and train sections')
    result = deepcopy(DEFAULTS)
    for section, values in supplied.items():
        if not isinstance(values, dict):
            raise ValueError(f'{section} must be an object')
        unknown = set(values) - set(DEFAULTS[section])
        if unknown:
            raise ValueError(f'Unsupported {section} configuration: {sorted(unknown)}')
        result[section].update(values)
    return validate_config(result)
