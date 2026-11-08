"""Evaluate one checkpoint once with the released MSR video scoring rule."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from utils import ROOT, setup, ordinary_loader, check_population, evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'results/mrs/best.pth')
    parser.add_argument('--output', '--output-dir', type=Path, default=Path('evaluation/best'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--context-seed', type=int, default=None,
                        help='Defaults to the checkpoint context seed, or 0 for a legacy checkpoint')
    parser.add_argument('--batch-size', type=int, default=14)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--expected-videos', type=int, default=None,
                        help='Defaults to checkpoint setting or 287; 0 disables the completeness check')
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Choose a new or empty output directory')
    if not torch.cuda.is_available():
        parser.error('Evaluation needs a CUDA GPU and the packaged pointnet2 extension')

    setup(args.seed)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or 'model_config' not in payload or 'model' not in payload:
        raise ValueError('Use the supplied best checkpoint or a checkpoint produced by train-msr-small.py')
    model_config = dict(payload['model_config'])
    saved_data = payload.get('data_config', {})
    data_config = dict(clip_len=24, num_points=2048, context_frames=32,
                       context_points=256, context_seed=0)
    for key in data_config:
        if key in saved_data:
            data_config[key] = saved_data[key]
    if args.context_seed is not None:
        data_config['context_seed'] = args.context_seed
    expected = (saved_data.get('expected_videos', 287) if args.expected_videos is None
                else args.expected_videos)

    # Match the official sequence: construct data/loaders, then the model, then
    # start the loader. A standalone pass has its own ordinary random point set.
    from datasets.msr import MSRDataset
    ds = MSRDataset(args.data, split='test', seed=args.seed,
                    file_order=payload.get('test_file_order'), **data_config)
    population = check_population(ds, expected)
    loader = ordinary_loader(ds, batch_size=args.batch_size, workers=args.workers)
    from models import MSRPSTModel
    model = MSRPSTModel(**model_config)
    model.load_state_dict(payload['model'], strict=True)
    model.cuda()

    args.output.mkdir(parents=True, exist_ok=True)
    score = evaluate(model, loader, 'cuda', args.output / 'predictions.csv')
    if score['total'] != population['usable_videos'] or score['clips'] != population['windows']:
        raise RuntimeError('Evaluation did not cover every declared test window/video')
    result = (
        f'Checkpoint: {args.checkpoint.resolve()}\nEpoch index: {payload.get("epoch")}\n'
        f'Model: {model_config}\nData: {data_config}\n'
        f'Seed: {args.seed}; batch_size: {args.batch_size}; workers: {args.workers}\n'
        'Protocol: one ordinary random-sampling pass; FP32 window softmax summed per video; '
        'argmax per video; no sampling search or parameter updates.\n'
        f'Video accuracy: {score["correct"]}/{score["total"]} = {score["accuracy"] * 100:.6f}%\n'
        f'Windows: {score["clips"]}; unscored short videos: {score["unscored"]}\n'
        f'Clip accuracy: {score["clip_accuracy"] * 100:.6f}%; mean CE: {score["loss"]:.8f}\n'
        f'Class counts: {score["class_counts"]}\nClass correct: {score["class_correct"]}\n'
        'Standalone inputs need not equal the inputs drawn during the best training epoch.\n'
    )
    (args.output / 'result.txt').write_text(result, encoding='utf8')
    print(result, flush=True)


if __name__ == '__main__':
    main()
