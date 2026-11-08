"""One explicit configuration for the single-stage NTU60 experiment."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config/ntu_xsub.json'
DEFAULTS = {'data': {'root': None,
          'meta': None,
          'split': 'xsub',
          'frames_per_clip': 24,
          'step_between_clips': 2,
          'num_points': 2048,
          'context_frames': 32,
          'context_points': 256,
          'context_seed': 0,
          'cache_size': 2,
          'expected_videos': 16156,
          'expected_listed_videos': 16560,
          'expected_metadata_videos': 56880},
 'model': {'use_ebte': True,
           'use_cmtr': True,
           'use_srrf': False,
           'seed': 0,
           'num_classes': 60,
           'radius': 0.1,
           'nsamples': 32,
           'spatial_stride': 32,
           'temporal_kernel_size': 3,
           'temporal_stride': 2,
           'dim': 80,
           'depth': 5,
           'heads': 2,
           'dim_head': 40,
           'mlp_dim': 160,
           'dropout1': 0.0,
           'dropout2': 0.5,
           'rho': 0.02,
           'epsilon': 1e-06,
           'srrf_scale': 1.0},
 'train': {'seed': 0,
           'epochs': 50,
           'batch_size': 24,
           'workers': 4,
           'lr': 0.01,
           'momentum': 0.9,
           'weight_decay': 0.0001,
           'warmup_epochs': 10,
           'warmup_factor': 1e-05,
           'lr_milestones': [20, 30],
           'lr_gamma': 0.1,
           'print_freq': 100,
           'label_smoothing': 0.0}}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def validate_config(config):
    for name in ('frames_per_clip', 'step_between_clips', 'num_points', 'context_frames', 'context_points'):
        _integer(config['data'][name], 'data.' + name)
    _integer(config['data']['context_seed'], 'data.context_seed', 0)
    _integer(config['data']['expected_videos'], 'data.expected_videos', 0)
    for name in ('cache_size', 'expected_listed_videos', 'expected_metadata_videos'):
        _integer(config['data'][name], 'data.' + name, 0)
    if config['data']['split'] not in ('xsub', 'xview'):
        raise ValueError('data.split must be xsub or xview')
    if config['model']['num_classes'] != 60:
        raise ValueError('This package requires 60 NTU60 classes')
    train = config['train']
    for name in ('seed', 'workers', 'warmup_epochs'):
        _integer(train[name], 'train.' + name, 0)
    for name in ('epochs', 'batch_size', 'print_freq'):
        _integer(train[name], 'train.' + name)
    for name in ('lr', 'momentum', 'weight_decay', 'warmup_factor', 'lr_gamma', 'label_smoothing'):
        value = train[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'train.{name} must be finite')
    if train['lr'] <= 0 or not 0 <= train['momentum'] < 1 or train['weight_decay'] < 0:
        raise ValueError('Invalid SGD settings')
    if not 0 < train['warmup_factor'] <= 1 or not 0 < train['lr_gamma'] <= 1:
        raise ValueError('warmup_factor and lr_gamma must be in (0,1]')
    if not 0 <= train['label_smoothing'] < 1:
        raise ValueError('label_smoothing must be in [0,1)')
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
    if supplied.get("data", {}).get("split") == "xview":
        result["data"].update(split="xview", expected_videos=18224, expected_listed_videos=18960)
    for section, values in supplied.items():
        if not isinstance(values, dict):
            raise ValueError(f'{section} must be an object')
        unknown = set(values) - set(DEFAULTS[section])
        if unknown:
            raise ValueError(f'Unsupported {section} configuration: {sorted(unknown)}')
        result[section].update(values)
    return validate_config(result)
