"""Single-stage, from-scratch NTU60 training with the official video evaluator."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import io
from pathlib import Path
import pprint
import sys
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from utils_ntu import (boolean, setup, capture_rng, restore_rng,
                            ordinary_loader, evaluate, write_predictions)
from config.ntu import DEFAULT_CONFIG, load_config, validate_config
from scheduler import WarmupMultiStepLR

from utils_ntu import SCHEMA
FIELDS = ['epoch', 'train_loss', 'train_hard_ce', 'train_clip_accuracy', 'test_loss', 'correct',
          'total', 'accuracy_percent', 'test_clip_accuracy', 'clips', 'lr', 'train_seconds', 'eval_seconds', 'seconds']


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_text(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value, encoding='utf8')
    temporary.replace(path)


def save_curve(history, path):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(history)
    atomic_text(stream.getvalue(), path)


def score_without_rows(score):
    return {k: v for k, v in score.items() if k != 'rows'}


def check_population(ds, expected, **checks):
    from utils_ntu import check_population as check
    return check(ds, expected, **checks)['usable_videos']


def optimizer_and_scheduler(model, config, batches):
    train = config['train']
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError('Single-stage training requires every model parameter to be trainable')
    optimizer = torch.optim.SGD(model.parameters(), lr=train['lr'], momentum=train['momentum'],
                                weight_decay=train['weight_decay'])
    scheduler = WarmupMultiStepLR(
        optimizer, milestones=[batches * m for m in train['lr_milestones']],
        gamma=train['lr_gamma'], warmup_iters=train['warmup_epochs'] * batches,
        warmup_factor=train['warmup_factor'])
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, print_freq, label_smoothing=0.):
    model.train()
    started = time.monotonic()
    loss_sum = hard_sum = correct = count = 0
    last_lr = optimizer.param_groups[0]['lr']
    for batch, (inputs, target, _) in enumerate(loader):
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        target = target.to(device, non_blocking=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, target, label_smoothing=label_smoothing)
        hard_ce = loss.detach() if label_smoothing == 0 else F.cross_entropy(logits.detach(), target)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.detach()) * len(target)
        hard_sum += float(hard_ce) * len(target)
        correct += int(logits.detach().argmax(1).eq(target).sum())
        count += len(target)
        if batch % print_freq == 0 or batch + 1 == len(loader):
            print(f'Epoch {epoch+1}: batch {batch+1}/{len(loader)}, '
                  f'loss={float(loss.detach()):.5f}, lr={last_lr:.9g}', flush=True)
    if count != len(loader.dataset):
        raise RuntimeError('The training loader must retain every window, including the last batch')
    return dict(loss=loss_sum / count, hard_ce=hard_sum / count, clip_accuracy=correct / count,
                samples=count, lr=last_lr, seconds=time.monotonic() - started)


def resume_contract(config):
    result = deepcopy(config)
    # Relocating the data/output is allowed; population and order are checked separately.
    result['data'].pop('root', None)
    result['data'].pop('meta', None)
    for key in ('epochs', 'print_freq'):
        result['train'].pop(key, None)
    return result


def run_epochs(model, train_loader, test_loader, config, output, device, *, resume_state=None):
    """Persist the post-training state before evaluation, so interrupted evaluation is retried."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    total = check_population(test_loader.dataset, config['data']['expected_videos'])
    optimizer, scheduler = optimizer_and_scheduler(model, config, len(train_loader))
    train_order = list(train_loader.dataset.directory_manifest)
    test_order = list(test_loader.dataset.directory_manifest)
    data_config = {k: v for k, v in config['data'].items() if k not in ('root', 'meta')}
    history, best_correct, best_epoch = [], -1, None
    epoch, pending = 0, False
    if resume_state is not None:
        saved = resume_state
        if saved.get('schema') != SCHEMA:
            raise ValueError('Resume requires this package\'s last.pth, not a legacy/staged run')
        if saved['model_config'] != model.model_config():
            raise ValueError('Resume model configuration differs')
        if resume_contract(saved['config']) != resume_contract(config):
            raise ValueError('Resume training/data settings differ')
        if (saved['batches_per_epoch'] != len(train_loader)
                or saved['train_windows'] != len(train_loader.dataset)
                or saved['test_windows'] != len(test_loader.dataset)
                or saved['data_file_order'] != train_order
                or saved['test_file_order'] != test_order
                or saved['test_total'] != total
                or saved['train_lengths'] != list(train_loader.dataset.lengths)
                or saved['test_lengths'] != list(test_loader.dataset.lengths)):
            raise ValueError('Resume data population/order or window counts differ')
        if saved['status'] not in ('awaiting_evaluation', 'complete'):
            raise ValueError('Unrecognized checkpoint status')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['lr_scheduler'])
        history = deepcopy(saved['history'])
        best_correct, best_epoch = saved['best_correct'], saved['best_epoch']
        pending = saved['status'] == 'awaiting_evaluation'
        epoch = saved['epoch'] if pending else saved['epoch'] + 1
        if saved['global_step'] != (saved['epoch'] + 1) * len(train_loader):
            raise ValueError('Checkpoint optimizer step count is inconsistent')
        if config['train']['epochs'] < saved['epoch'] + 1:
            raise ValueError('Requested total epochs precedes the saved checkpoint')
        if best_correct >= 0 and not (output / 'best.pth').is_file():
            raise ValueError('Resume output directory is missing its best.pth')
        restore_rng(saved['rng'])
        # CSVs are derived outputs. Rebuild them if a crash happened after the
        # atomic checkpoint commit but before the corresponding file write.
        if not pending:
            write_predictions(output / 'last_predictions.csv', saved['predictions'])
        if (output / 'best.pth').is_file():
            best_saved = torch.load(output / 'best.pth', map_location='cpu', weights_only=False)
            if (best_saved.get('schema') != SCHEMA
                    or best_saved['model_config'] != model.model_config()
                    or resume_contract(best_saved['config']) != resume_contract(config)
                    or best_saved['test_video_names'] != list(test_loader.dataset.video_names)
                    or best_saved['test_total'] != total):
                raise ValueError('best.pth does not belong to this run architecture/protocol')
            if not pending and best_saved['epoch'] != best_epoch:
                raise ValueError('best.pth and last.pth disagree on the best epoch')
            write_predictions(output / 'best_predictions.csv', best_saved['predictions'])
    save_curve(history, output / 'metrics.csv')
    atomic_text(pprint.pformat(config, sort_dicts=True) + '\n'
                + pprint.pformat(model.model_config(), sort_dicts=True) + '\n', output / 'settings.txt')
    atomic_text(f'Training in progress: target {config["train"]["epochs"]} epochs; '
                f'{len(history)} epochs have completed evaluation.\n', output / 'result.txt')

    def checkpoint(status, epoch_index, train_stats, *, score=None, rows=None, elapsed=0.):
        return dict(
            schema=SCHEMA, status=status, epoch=epoch_index,
            model=model.state_dict(), model_config=model.model_config(),
            optimizer=optimizer.state_dict(), lr_scheduler=scheduler.state_dict(),
            global_step=(epoch_index + 1) * len(train_loader),
            config=deepcopy(config), data_config=data_config,
            data_file_order=train_order, test_file_order=test_order,
            train_video_names=list(train_loader.dataset.video_names),
            test_video_names=list(test_loader.dataset.video_names),
            train_lengths=list(train_loader.dataset.lengths), test_lengths=list(test_loader.dataset.lengths),
            train_windows=len(train_loader.dataset), test_windows=len(test_loader.dataset),
            test_total=total, batches_per_epoch=len(train_loader),
            best_correct=best_correct, best_epoch=best_epoch, best_total=total,
            history=deepcopy(history), train_stats=train_stats, score=score,
            predictions=rows, elapsed=elapsed, rng=capture_rng())

    while epoch < config['train']['epochs']:
        started = time.monotonic()
        if pending:
            stats = resume_state['train_stats']
            prior_elapsed = resume_state.get('elapsed', 0.)
            print(f'Resuming evaluation of epoch {epoch+1}; no optimizer update repeated.', flush=True)
        else:
            stats = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch,
                                    config['train']['print_freq'], config['train']['label_smoothing'])
            prior_elapsed = 0.
            # This is the exact state/RNG immediately before ordinary evaluation.
            atomic_save(checkpoint('awaiting_evaluation', epoch, stats,
                                   elapsed=time.monotonic() - started), output / 'last.pth')
        eval_started = time.monotonic()
        score = evaluate(model, test_loader, device)
        eval_seconds = time.monotonic() - eval_started
        if score['total'] != total or score['clips'] != len(test_loader.dataset):
            raise RuntimeError('Evaluation must cover every declared test video/window')
        improved = score['correct'] > best_correct
        if improved:
            best_correct, best_epoch = score['correct'], epoch
        row = dict(epoch=epoch+1, train_loss=stats['loss'], train_hard_ce=stats['hard_ce'],
                   train_clip_accuracy=stats['clip_accuracy'], test_loss=score['loss'],
                   correct=score['correct'], total=score['total'],
                   accuracy_percent=100 * score['accuracy'],
                   test_clip_accuracy=score['clip_accuracy'], clips=score['clips'],
                   lr=stats['lr'], train_seconds=stats['seconds'], eval_seconds=eval_seconds, seconds=prior_elapsed + time.monotonic() - started)
        history.append(row)
        state = checkpoint('complete', epoch, stats, score=score_without_rows(score), rows=score['rows'])
        if improved:
            atomic_save(state, output / 'best.pth')
            write_predictions(output / 'best_predictions.csv', score['rows'])
        atomic_save(state, output / 'last.pth')
        write_predictions(output / 'last_predictions.csv', score['rows'])
        save_curve(history, output / 'metrics.csv')
        atomic_text(
            f'Training in progress: {epoch+1}/{config["train"]["epochs"]} epochs evaluated.\n'
            f'Modules: EBTE={config["model"]["use_ebte"]}, '
            f'CMTR={config["model"]["use_cmtr"]}, SRRF={config["model"]["use_srrf"]}.\n'
            f'Current video accuracy: {score["correct"]}/{total}.\n'
            f'Best ordinary video accuracy: {best_correct}/{total}, epoch {best_epoch+1}.\n',
            output / 'result.txt')
        print(f'Epoch {epoch+1}/{config["train"]["epochs"]}: '
              f'video={score["correct"]}/{total} ({100*score["accuracy"]:.4f}%), '
              f'best={best_correct}/{total} at epoch {best_epoch+1}', flush=True)
        pending = False
        epoch += 1
    atomic_text(
        f'Single-stage training completed: {epoch} epochs.\n'
        f'Best ordinary per-epoch video accuracy: {best_correct}/{total} '
        f'({100*best_correct/total:.6f}%), epoch {best_epoch+1}.\n'
        'All enabled modules were jointly optimized with cross-entropy.\n'
        'Selection is maximum video accuracy over epochs; no repeated-sampling search.\n',
        output / 'result.txt')
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--meta', type=Path, help='NTU metadata list; defaults to ntu60.list in the data root')
    parser.add_argument('--output', '--output-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path, help='Resume last.pth in the same output directory')
    for name in ('ebte', 'cmtr', 'srrf'):
        parser.add_argument('--use-' + name, '--use_' + name, type=boolean, default=None)
    for name in ('seed', 'epochs', 'batch-size', 'workers', 'expected-videos', 'context-seed'):
        parser.add_argument('--' + name, type=int, default=None)
    parser.add_argument('--label-smoothing', type=float, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for name in ('ebte', 'cmtr', 'srrf'):
        value = getattr(args, 'use_' + name)
        if value is not None:
            config['model']['use_' + name] = value
    for key in ('seed', 'epochs', 'batch_size', 'workers'):
        if getattr(args, key) is not None:
            config['train'][key] = getattr(args, key)
    if args.label_smoothing is not None:
        config['train']['label_smoothing'] = args.label_smoothing
    if args.seed is not None:
        config['model']['seed'] = args.seed
    for key in ('expected_videos', 'context_seed'):
        if getattr(args, key) is not None:
            config['data'][key] = getattr(args, key)
    data_root = args.data or config['data']['root']
    if data_root is None:
        parser.error('Provide --data pointing to the NTU60 root or video directory')
    data_root = Path(data_root).expanduser().resolve()
    meta = args.meta or config['data']['meta']
    meta = Path(meta).expanduser().resolve() if meta is not None else None
    output = args.output.expanduser().resolve()
    if output in (ROOT, data_root):
        parser.error('Use a separate run directory as --output')
    config['data']['root'] = str(data_root)
    config['data']['meta'] = str(meta) if meta is not None else None
    validate_config(config)
    state = None
    if args.resume:
        resume_path = args.resume.expanduser().resolve()
        if resume_path != output / 'last.pth':
            parser.error('--resume must be last.pth in the existing --output directory')
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        if state.get('schema') != SCHEMA:
            parser.error('Only a single-stage checkpoint produced by this package can resume')
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('From-scratch training requires a new or empty output directory')
    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU and the bundled point-cloud extension')
    setup(config['train']['seed'])
    from datasets.ntu60 import NTUDataset
    from models import build_ntu_model
    data_args = {k: config['data'][k] for k in
                 ('frames_per_clip', 'step_between_clips', 'num_points', 'context_frames',
                  'context_points', 'context_seed', 'cache_size')}
    train_data = NTUDataset(data_root, meta=meta, split=config['data']['split'], train=True, **data_args,
                            file_order=state['data_file_order'] if state else None)
    test_data = NTUDataset(data_root, meta=meta, split=config['data']['split'], train=False, **data_args,
                           file_order=state['test_file_order'] if state else None)
    if set(train_data.video_names) & set(test_data.video_names):
        raise ValueError('Training and evaluation videos overlap')
    if not len(train_data):
        raise ValueError('No training windows')
    check_population(test_data, config['data']['expected_videos'],
                     expected_listed_videos=config['data']['expected_listed_videos'],
                     expected_metadata_videos=config['data']['expected_metadata_videos'])
    train_loader = ordinary_loader(train_data, train=True, batch_size=config['train']['batch_size'],
                                   workers=config['train']['workers'])
    test_loader = ordinary_loader(test_data, batch_size=config['train']['batch_size'],
                                  workers=config['train']['workers'])
    # Official order: seed -> construct datasets/loaders -> construct model -> iterate loaders.
    model = build_ntu_model(config).to('cuda')
    flags = model.model_config()
    print(f'NTU60 protocol={config["data"]["split"]}; label_smoothing={config["train"]["label_smoothing"]}', flush=True)
    print(f'Single-stage modules: EBTE={flags["use_ebte"]}, '
          f'CMTR={flags["use_cmtr"]}, SRRF={flags["use_srrf"]}. '
          'Every enabled module jointly trains from epoch 1.', flush=True)
    print(f'Train windows={len(train_data)}, test windows={len(test_data)}, '
          f'test videos={len({idx for idx, _ in test_data.index_map})}, '
          f'parameters={sum(p.numel() for p in model.parameters())}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    from modules import pointnet2_utils
    extension_path = getattr(pointnet2_utils._ext, '__file__', 'unknown')
    atomic_text(f'torch={torch.__version__}\ncuda={torch.version.cuda}\n'
                f'gpu={torch.cuda.get_device_name()}\npointnet2={extension_path}\n'
                f'train={train_data.metadata}\ntest={test_data.metadata}\n', output / 'runtime.txt')
    from telemetry import GPUMonitor
    with GPUMonitor(output):
        run_epochs(model, train_loader, test_loader, config, output, 'cuda', resume_state=state)


if __name__ == '__main__':
    main()
