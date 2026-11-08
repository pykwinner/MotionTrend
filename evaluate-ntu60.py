"""Evaluate one NTU60 checkpoint with the official per-video scoring rule."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from utils_ntu import setup, ordinary_loader, check_population, evaluate
from utils_ntu import SCHEMA


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--meta', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', '--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['xsub', 'xview'], help='Optional assertion; must match checkpoint')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--context-seed', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--expected-videos', type=int, default=None)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Choose a new or empty output directory')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
        raise ValueError('Use a checkpoint produced by this NTU60 package')
    data = dict(payload['data_config'])
    split = data['split']
    if args.split is not None and args.split != split:
        parser.error('--split conflicts with the trained checkpoint protocol')
    if payload['model_config']['num_classes'] != 60:
        raise ValueError('Expected an NTU60 classifier with 60 classes')
    if args.context_seed is not None:
        data['context_seed'] = args.context_seed
    expected = data['expected_videos'] if args.expected_videos is None else args.expected_videos
    batch_size = payload['config']['train']['batch_size'] if args.batch_size is None else args.batch_size
    workers = payload['config']['train']['workers'] if args.workers is None else args.workers
    if not torch.cuda.is_available():
        parser.error('Evaluation requires a CUDA GPU and the bundled point-cloud extension')
    setup(args.seed)
    from datasets.ntu60 import NTUDataset
    kwargs = {k: data[k] for k in ('frames_per_clip', 'step_between_clips', 'num_points',
                                  'context_frames', 'context_points', 'context_seed', 'cache_size')}
    ds = NTUDataset(args.data, meta=args.meta, split=split, train=False,
                    file_order=payload['test_file_order'], **kwargs)
    population = check_population(ds, expected,
                                  expected_listed_videos=data['expected_listed_videos'],
                                  expected_metadata_videos=data['expected_metadata_videos'])
    if list(ds.video_names) != payload['test_video_names'] or len(ds) != payload['test_windows']:
        raise ValueError('Checkpoint evaluation population or window counts differ')
    if list(ds.lengths) != payload['test_lengths']:
        raise ValueError('Checkpoint metadata frame counts differ')
    loader = ordinary_loader(ds, batch_size=batch_size, workers=workers)
    from models import NTUPSTModel
    model = NTUPSTModel(**payload['model_config'])
    model.load_state_dict(payload['model'], strict=True)
    model.cuda()
    output.mkdir(parents=True, exist_ok=True)
    from telemetry import GPUMonitor
    with GPUMonitor(output):
        score = evaluate(model, loader, 'cuda', output / 'predictions.csv')
    if score['total'] != population['usable_videos'] or score['clips'] != population['windows']:
        raise RuntimeError('Evaluation did not cover every declared window/video')
    text = (
        f'Checkpoint: {args.checkpoint.resolve()}\nEpoch: {payload["epoch"]+1}\n'
        f'Protocol: NTU60 {split}; listed={len(ds.video_names)}, scored={score["total"]}, '
        f'unscored_short={score["unscored"]}; windows={score["clips"]}\n'
        f'Model: {payload["model_config"]}\nData: {data}\n'
        f'Seed={args.seed}; batch_size={batch_size}; workers={workers}\n'
        'One ordinary local random-sampling pass; no search or parameter updates.\n'
        'FP32 window softmax summed sequentially per video, then video argmax.\n'
        f'Video accuracy: {score["correct"]}/{score["total"]} = {100*score["accuracy"]:.6f}%\n'
        f'Window accuracy: {100*score["clip_accuracy"]:.6f}%; ordinary hard-label CE: {score["loss"]:.8f}\n'
        'Standalone draws need not equal those from the best training epoch.\n')
    (output / 'result.txt').write_text(text, encoding='utf8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
