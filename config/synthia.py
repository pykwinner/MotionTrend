"""Explicit configuration for independent 12-class Synthia segmentation."""
from copy import deepcopy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config/synthia_full.json'

def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return validate_config(cfg)

def validate_config(cfg):
    required = {'data_root', 'metadata_root', 'run_dir', 'seed', 'num_classes', 'frames_per_clip',
                'num_points', 'model', 'train', 'data', 'eval_batch_size'}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError('Synthia configuration must contain the documented top-level fields')
    def integer(v, name, minimum=1):
        if type(v) is not int or v < minimum:
            raise ValueError(name + ' must be an integer >= ' + str(minimum))
    integer(cfg['seed'], 'seed', 0)
    if cfg['seed'] >= 2**32:
        raise ValueError('seed must be below 2**32')
    if cfg['num_classes'] != 12:
        raise ValueError('Synthia requires 12 classes')
    for key in ('frames_per_clip', 'num_points', 'eval_batch_size'):
        integer(cfg[key], key)
    if cfg['frames_per_clip'] != 3:
        raise ValueError('The released Synthia model requires three frames')
    for key in ('data_root', 'metadata_root', 'run_dir'):
        if cfg[key] is not None and not isinstance(cfg[key], str):
            raise ValueError(key + ' must be a string or null')
    if not cfg['metadata_root']:
        raise ValueError('metadata_root is required')
    if not isinstance(cfg['data'], dict) or set(cfg['data']) != {'context', 'cache_size'}:
        raise ValueError('data must contain context and cache_size')
    integer(cfg['data']['cache_size'], 'cache_size', 0)
    context = cfg['data']['context']
    if type(context) not in (int, float) or not math.isfinite(context) or context <= 0:
        raise ValueError('context must be finite and positive')
    model = cfg['model']
    if not isinstance(model, dict):
        raise ValueError('model must be an object')
    for key in ('use_ebte', 'use_cmtr', 'use_srrf'):
        if type(model.get(key)) is not bool:
            raise ValueError(key + ' must be an explicit Boolean')
    train = cfg['train']
    allowed = {'epochs', 'batch_size', 'workers', 'lr', 'momentum', 'weight_decay', 'gamma',
               'milestones', 'warmup_epochs', 'warmup_factor', 'print_freq'}
    if not isinstance(train, dict) or set(train) != allowed:
        raise ValueError('Unknown or missing training configuration field')
    for key in ('epochs', 'batch_size', 'print_freq'):
        integer(train[key], key)
    for key in ('workers', 'warmup_epochs'):
        integer(train[key], key, 0)
    for key in ('lr', 'momentum', 'weight_decay', 'gamma', 'warmup_factor'):
        if type(train[key]) not in (int, float) or not math.isfinite(train[key]):
            raise ValueError(key + ' must be finite')
    if train['lr'] <= 0 or not 0 <= train['momentum'] < 1 or train['weight_decay'] < 0:
        raise ValueError('Invalid SGD parameters')
    if not 0 < train['gamma'] <= 1 or not 0 < train['warmup_factor'] <= 1:
        raise ValueError('gamma/warmup_factor must be in (0,1]')
    steps = train['milestones']
    if not isinstance(steps, list) or any(type(s) is not int or s < 1 for s in steps) or steps != sorted(set(steps)):
        raise ValueError('milestones must be increasing positive integers')
    return cfg

def resume_contract(cfg):
    result = deepcopy(cfg)
    for key in ('data_root', 'metadata_root', 'run_dir'):
        result.pop(key)
    for key in ('epochs', 'print_freq'):
        result['train'].pop(key)
    return result

def resolve_config_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
