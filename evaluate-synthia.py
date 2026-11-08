"""Evaluate one Synthia checkpoint on both half-scenes with global 12-class mIoU."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import pprint
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from utils_synthia import (SCHEMA, setup, atomic_text, build_datasets, data_identity,
                                    make_loader, evaluate, write_metrics, maybe_data_parallel, parallelism)
from config.synthia import validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--metadata-root', type=Path, default=ROOT / 'datasets/synthia_metadata')
    parser.add_argument('--output', '--output-dir', '--run-dir', dest='output', type=Path, required=True)
    parser.add_argument('--seed', type=int, help='One ordinary sampling seed; defaults to the checkpoint seed')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--print-freq', type=int)
    parser.add_argument('--data-parallel', action='store_true', help='Explicit multi-GPU evaluation')
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('schema') != SCHEMA:
        parser.error('This evaluator requires a checkpoint from train-synthia.py')
    cfg = deepcopy(saved['config'])
    cfg['data_root'] = str(args.data_root.expanduser().resolve())
    cfg['metadata_root'] = str(args.metadata_root.expanduser().resolve())
    output = args.output.expanduser().resolve()
    cfg['run_dir'] = str(output)
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.batch_size is not None:
        cfg['eval_batch_size'] = args.batch_size
    if args.workers is not None:
        cfg['train']['workers'] = args.workers
    if args.print_freq is not None:
        cfg['train']['print_freq'] = args.print_freq
    validate_config(cfg)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Evaluation requires a new or empty --output directory')
    if output in (ROOT, Path(cfg['data_root']), Path(cfg['metadata_root'])):
        parser.error('Use a separate output directory')
    if not torch.cuda.is_available():
        parser.error('Evaluation requires a CUDA GPU and the bundled point-cloud extension')
    if args.data_parallel and torch.cuda.device_count() < 2:
        parser.error('--data-parallel requires at least two visible CUDA GPUs')
    setup(cfg['seed'])
    dataset = build_datasets(cfg, train=False)
    if data_identity(dataset) != saved['test_identity']:
        raise ValueError('Test metadata or class weights differ from the checkpoint protocol')
    loader = make_loader(dataset, batch_size=cfg['eval_batch_size'], workers=cfg['train']['workers'])
    from models import build_synthia_model
    # Architecture and all module options always come from the checkpoint.
    # A sampling-seed override must not change module initialization configuration.
    model = build_synthia_model(saved['config']).to('cuda')
    if model.model_config() != saved['model_config']:
        raise ValueError('Checkpoint model configuration is inconsistent')
    model.load_state_dict(saved['model'], strict=True)
    model = maybe_data_parallel(model, args.data_parallel)
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(pprint.pformat(cfg, sort_dicts=True) + '\n', output / 'settings.txt')
    from telemetry import GPUMonitor
    with GPUMonitor(output):
        score = evaluate(model, loader, 'cuda', cfg['train']['print_freq'])
    write_metrics(score, output, 'evaluation')
    atomic_text(f'checkpoint={checkpoint}\nepoch={saved["epoch"] + 1}\n'
        f'test_miou_percent={100 * score["mean_iou"]:.6f}\n'
        f'test_mean_class_accuracy_percent={100 * score["mean_acc"]:.6f}\n'
        f'test_point_accuracy_percent={100 * score["point_accuracy"]:.6f}\n'
        f'valid_points={score["valid_points"]}\nclip_samples={score["samples"]}\n'
        f'evaluation_seconds={score["seconds"]:.6f}\nsampling_seed={cfg["seed"]}\nparallelism={parallelism(model)}\n'
        'Evaluation time includes data loading, both half-scenes, model forward, loss and metrics.\n'
        'This is one ordinary re-evaluation draw; it need not equal the recorded training-epoch draw.\n'
        'Both halves and all three clip frames contribute; repeated frame occurrences are retained.\n',
        output / 'result.txt')
    print((output / 'result.txt').read_text(encoding='utf-8'), flush=True)
    return score


if __name__ == '__main__':
    main()
