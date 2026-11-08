"""Single-stage Synthia training from scratch, with paired-half semantic evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import pprint
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from utils_synthia import (SCHEMA, boolean, setup, capture_rng, restore_rng, atomic_save,
    atomic_text, write_csv, write_metrics, build_datasets, data_identity, make_loader,
    half_inputs, forward_half, masked_loss, evaluate, sync, unwrap_model, parallelism,
    maybe_data_parallel, write_module_monitor, log_message, append_csv_row)
from config.synthia import DEFAULT_CONFIG, load_config, validate_config, resume_contract, resolve_config_path
from scheduler import WarmupMultiStepLR

FIELDS = ['epoch', 'train_loss', 'test_loss', 'test_miou_percent', 'test_mean_accuracy_percent',
          'test_point_accuracy_percent', 'test_valid_points', 'lr', 'optimizer_steps',
          'scheduler_steps', 'train_seconds', 'eval_seconds', 'seconds']
STEP_FIELDS = ['run_id', 'attempt_id', 'epoch', 'batch', 'batches', 'half0_loss', 'half1_loss',
               'loss', 'lr_used', 'lr_next', 'elapsed_seconds']


def optimizer_and_scheduler(model, cfg, batches):
    if batches < 1:
        raise ValueError('Training loader is empty; training drops the final incomplete batch')
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError('Every parameter must be trainable in this single-stage run')
    t = cfg['train']
    optimizer = torch.optim.SGD(model.parameters(), lr=t['lr'], momentum=t['momentum'], weight_decay=t['weight_decay'])
    scheduler = WarmupMultiStepLR(optimizer, milestones=[batches * m for m in t['milestones']],
        gamma=t['gamma'], warmup_iters=t['warmup_epochs'] * batches, warmup_factor=t['warmup_factor'])
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, print_freq, *,
                    output=None, run_id=None, attempt_id=None):
    model.train()
    weight = torch.as_tensor(loader.dataset.labelweights, dtype=torch.float32, device=device)
    sync(device)
    started = time.monotonic()
    loss_sum, count = 0., 0
    step_records = []
    epoch_dir = Path(output) / 'epochs' / f'epoch_{epoch + 1:03d}' if output is not None else None
    if epoch_dir is not None:
        epoch_dir.mkdir(parents=True, exist_ok=True)
        write_csv([], STEP_FIELDS, epoch_dir / 'train_steps.csv')
    last_lr = optimizer.param_groups[0]['lr']
    for step, batch in enumerate(loader):
        losses = []
        for half in (0, 1):
            xyz, rgb, label, mask, times, valid = half_inputs(batch, half, device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_half(model, xyz, rgb, times, valid)
            loss = masked_loss(logits, label, mask, weight)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            last_lr = optimizer.param_groups[0]['lr']
            optimizer.step()
            losses.append(float(loss.detach()))
            del logits, loss, xyz, rgb, label, mask, times, valid
        # Two sequential half-scene updates share one LR; step after the pair.
        scheduler.step()
        loss_sum += sum(losses) / 2
        count += 1
        row = dict(run_id=run_id, attempt_id=attempt_id, epoch=epoch + 1, batch=step + 1,
                   batches=len(loader), half0_loss=losses[0], half1_loss=losses[1],
                   loss=sum(losses) / 2, lr_used=last_lr,
                   lr_next=optimizer.param_groups[0]['lr'], elapsed_seconds=time.monotonic() - started)
        step_records.append(row)
        if epoch_dir is not None:
            append_csv_row(row, STEP_FIELDS, epoch_dir / 'train_steps.csv')
        if step % print_freq == 0 or step + 1 == len(loader):
            message = (f'run={run_id} attempt={attempt_id} epoch={epoch + 1} '
                       f'batch={step + 1}/{len(loader)} half_loss={losses} '
                       f'lr_used={last_lr:.9g} lr_next={row["lr_next"]:.9g} '
                       f'elapsed={row["elapsed_seconds"]:.3f}s')
            if epoch_dir is not None:
                log_message(output, message)
            else:
                print(message, flush=True)
    if not count:
        raise ValueError('No training batches')
    sync(device)
    return dict(loss=loss_sum / count, lr=last_lr, batches=count,
                optimizer_steps=2 * count, seconds=time.monotonic() - started, step_records=step_records)


def run_epochs(model, train_loader, test_loader, cfg, output, device, *, resume_state=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    active_parallelism = parallelism(model)
    optimizer, scheduler = optimizer_and_scheduler(model, cfg, len(train_loader))
    train_identity, test_identity = data_identity(train_loader.dataset), data_identity(test_loader.dataset)
    history, best_miou, best_epoch, epoch, pending = [], -1., None, 0, False
    run_id = uuid.uuid4().hex
    if resume_state is not None:
        saved = resume_state
        if saved.get('schema') != SCHEMA:
            raise ValueError('Resume requires a Synthia segmentation checkpoint from this entry point')
        if saved['model_config'] != raw_model.model_config() or resume_contract(saved['config']) != resume_contract(cfg):
            raise ValueError('Resume architecture or training configuration differs')
        if (saved['train_identity'] != train_identity or saved['test_identity'] != test_identity
                or saved['batches_per_epoch'] != len(train_loader)):
            raise ValueError('Resume metadata, class weights, or loader population differs')
        if saved['status'] not in ('complete', 'awaiting_evaluation'):
            raise ValueError('Invalid checkpoint status')
        expected_steps = (saved['epoch'] + 1) * len(train_loader)
        if (saved['scheduler_steps'] != expected_steps or saved['optimizer_steps'] != 2 * expected_steps
                or saved['lr_scheduler']['last_epoch'] != expected_steps):
            raise ValueError('Checkpoint half-pair step counts disagree')
        if cfg['train']['epochs'] < saved['epoch'] + 1:
            raise ValueError('Requested epochs precede checkpoint')
        if saved.get('parallelism', dict(data_parallel=False, devices=1)) != active_parallelism:
            raise ValueError('Resume requires the same single-/multi-GPU training mode and device count')
        raw_model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['lr_scheduler'])
        history, best_miou, best_epoch = deepcopy(saved['history']), saved['best_miou'], saved['best_epoch']
        pending = saved['status'] == 'awaiting_evaluation'
        run_id = saved['run_id']
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('Checkpoint run identity is missing')
        if len(history) != saved['epoch'] + (not pending):
            raise ValueError('Checkpoint history and status disagree')
        epoch = saved['epoch'] if pending else saved['epoch'] + 1
        best_path = output / 'best.pth'
        if best_miou >= 0 and not best_path.is_file():
            raise ValueError('Resume output is missing its best.pth')
        if best_path.is_file():
            best = torch.load(best_path, map_location='cpu', weights_only=False)
            if (best.get('schema') != SCHEMA or best.get('run_id') != run_id or best['model_config'] != raw_model.model_config()
                    or resume_contract(best['config']) != resume_contract(cfg)
                    or best['train_identity'] != train_identity or best['test_identity'] != test_identity
                    or (not pending and best['epoch'] != best_epoch)):
                raise ValueError('The output best.pth belongs to another run or epoch')
            write_metrics(best['score'], output, 'best')
        if not pending:
            write_metrics(saved['score'], output, 'last')
        restore_rng(saved['rng'])
    attempt_id = uuid.uuid4().hex
    log_message(output, f'BEGIN run={run_id} attempt={attempt_id} mode={active_parallelism} '
                f'resume={resume_state is not None} next_epoch={epoch + 1} awaiting_evaluation={pending}')
    if resume_state is None:
        write_module_monitor(raw_model, output / 'modules_init.csv', run_id=run_id, epoch=0)
    else:
        # Record resume weights explicitly, never label them as fresh initialization.
        write_module_monitor(raw_model, output / f'modules_resume_{attempt_id}.csv',
                             run_id=run_id, epoch=epoch)
    write_csv(history, FIELDS, output / 'metrics.csv')
    atomic_text(pprint.pformat(cfg, sort_dicts=True) + '\n', output / 'settings.txt')
    atomic_text(f'Run in progress: {len(history)} epochs evaluated.\n', output / 'result.txt')

    def state(status, epoch_index, train_stats, score=None):
        steps = (epoch_index + 1) * len(train_loader)
        return dict(schema=SCHEMA, run_id=run_id, status=status, epoch=epoch_index, model=raw_model.state_dict(), parallelism=active_parallelism,
            model_config=raw_model.model_config(), config=deepcopy(cfg), optimizer=optimizer.state_dict(),
            lr_scheduler=scheduler.state_dict(), rng=capture_rng(), history=deepcopy(history),
            train_identity=train_identity, test_identity=test_identity, batches_per_epoch=len(train_loader),
            optimizer_steps=2 * steps, scheduler_steps=steps, best_miou=best_miou,
            best_epoch=best_epoch, train_stats=train_stats, score=score)

    while epoch < cfg['train']['epochs']:
        if pending:
            stats = resume_state['train_stats']
        else:
            stats = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, cfg['train']['print_freq'],
                                    output=output, run_id=run_id, attempt_id=attempt_id)
            # Interruptions during evaluation never silently skip that epoch.
            atomic_save(state('awaiting_evaluation', epoch, stats), output / 'last.pth')
        epoch_dir = output / 'epochs' / f'epoch_{epoch + 1:03d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        if pending and 'step_records' in stats:
            write_csv(stats['step_records'], STEP_FIELDS, epoch_dir / 'train_steps.csv')
        # Match the source: training may use DataParallel, periodic evaluation uses GPU 0.
        score = evaluate(raw_model, test_loader, device, cfg['train']['print_freq'])
        write_metrics(score, epoch_dir, 'evaluation')
        write_module_monitor(raw_model, epoch_dir / 'modules.csv', run_id=run_id, epoch=epoch + 1)
        improved = score['mean_iou'] > best_miou
        if improved:
            best_miou, best_epoch = score['mean_iou'], epoch
        steps = (epoch + 1) * len(train_loader)
        history.append(dict(epoch=epoch + 1, train_loss=stats['loss'], test_loss=score['loss'],
            test_miou_percent=100 * score['mean_iou'], test_mean_accuracy_percent=100 * score['mean_acc'],
            test_point_accuracy_percent=100 * score['point_accuracy'], test_valid_points=score['valid_points'],
            lr=stats['lr'], optimizer_steps=2 * steps, scheduler_steps=steps,
            train_seconds=stats['seconds'], eval_seconds=score['seconds'], seconds=stats['seconds'] + score['seconds']))
        write_csv([history[-1]], FIELDS, epoch_dir / 'summary.csv')
        saved = state('complete', epoch, stats, score)
        if improved:
            atomic_save(saved, output / 'best.pth')
            write_metrics(score, output, 'best')
        atomic_save(saved, output / 'last.pth')
        write_metrics(score, output, 'last')
        write_csv(history, FIELDS, output / 'metrics.csv')
        message = (f'Epoch {epoch + 1}/{cfg["train"]["epochs"]}: test mIoU={100 * score["mean_iou"]:.6f}%; '
                   f'best={100 * best_miou:.6f}% at epoch {best_epoch + 1}.')
        log_message(output, f'run={run_id} attempt={attempt_id} ' + message)
        atomic_text(message + '\nSelection uses per-epoch test mIoU; no independent validation split.\n', output / 'result.txt')
        pending = False
        epoch += 1
    atomic_text(f'Single-stage training complete: {epoch} epochs.\n'
        f'Best recorded test mIoU={100 * best_miou:.6f}%, epoch {best_epoch + 1}.\n'
        'Every enabled module trains jointly from epoch 1; no parent weight is loaded.\n'
        'Per-epoch test mIoU selects the checkpoint; this is not an independent held-out validation selection.\n'
        'Both halves and all clip frames contribute masked points to a global 12-class confusion matrix.\n'
        'Absent-union classes have zero IoU. No final re-sampling replaces the recorded best score.\n', output / 'result.txt')
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--metadata-root', type=Path)
    parser.add_argument('--output', '--output-dir', '--run-dir', dest='output', type=Path, required=True)
    parser.add_argument('--resume', type=Path, help='Explicitly resume last.pth in this output directory')
    for name in ('ebte', 'cmtr', 'srrf'):
        parser.add_argument('--use-' + name, '--use_' + name, type=boolean)
    parser.add_argument('--data-parallel', action='store_true', help='Explicit multi-GPU training; default is one GPU')
    for name in ('seed', 'epochs', 'batch-size', 'eval-batch-size', 'workers', 'print-freq'):
        parser.add_argument('--' + name, type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    for name in ('ebte', 'cmtr', 'srrf'):
        value = getattr(args, 'use_' + name)
        if value is not None:
            cfg['model']['use_' + name] = value
    for key in ('epochs', 'batch_size', 'workers', 'print_freq'):
        if getattr(args, key) is not None:
            cfg['train'][key] = getattr(args, key)
    for key in ('seed', 'eval_batch_size'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    for key in ('data_root', 'metadata_root'):
        supplied = getattr(args, key)
        if supplied is not None:
            cfg[key] = str(supplied.expanduser().resolve())
        elif cfg[key]:
            cfg[key] = str(resolve_config_path(cfg[key]))
    output = args.output.expanduser().resolve()
    cfg['run_dir'] = str(output)
    validate_config(cfg)
    if not cfg['data_root']:
        parser.error('Provide --data-root with preprocessed Synthia NPZ files')
    if output in (ROOT, Path(cfg['data_root']), Path(cfg['metadata_root'])):
        parser.error('Use a separate output directory')
    saved = None
    if args.resume:
        if args.resume.expanduser().resolve() != output / 'last.pth':
            parser.error('--resume must point to last.pth in --output')
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved.get('schema') != SCHEMA:
            parser.error('Resume requires this Synthia entry point\'s checkpoint')
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('From-scratch training requires a new or empty output directory')
    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU and the bundled point-cloud extension')
    if args.data_parallel and torch.cuda.device_count() < 2:
        parser.error('--data-parallel requires at least two visible CUDA GPUs')
    setup(cfg['seed'])
    train_data, test_data = build_datasets(cfg)
    train_loader = make_loader(train_data, train=True, batch_size=cfg['train']['batch_size'], workers=cfg['train']['workers'])
    test_loader = make_loader(test_data, batch_size=cfg['eval_batch_size'], workers=cfg['train']['workers'])
    from models import build_synthia_model
    model = maybe_data_parallel(build_synthia_model(cfg).to('cuda'), args.data_parallel)
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(f'torch={torch.__version__}\ncuda={torch.version.cuda}\ngpu={torch.cuda.get_device_name()}\n'
        f'train={train_data.metadata_summary}\ntest={test_data.metadata_summary}\n'
        f'parameters={sum(p.numel() for p in model.parameters())}\nparallelism={parallelism(model)}\n', output / 'runtime.txt')
    from telemetry import GPUMonitor
    with GPUMonitor(output):
        run_epochs(model, train_loader, test_loader, cfg, output, 'cuda', resume_state=saved)


if __name__ == '__main__':
    main()
