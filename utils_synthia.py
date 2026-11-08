"""Synthia segmentation utilities: paired halves and global 12-class metrics."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Only RNG/argument primitives are shared; classification scoring is not used.
from utils import boolean, setup, capture_rng, restore_rng
from config.synthia import resolve_config_path

SCHEMA = 'motrend-synthia-joint-segmentation-v1'
CLASS_NAMES = ['Building', 'Road', 'Sidewalk', 'Fence', 'Vegetation', 'Pole',
               'Car', 'Traffic Sign', 'Pedestrian', 'Bicycle', 'Lanemarking', 'Traffic Light']


def atomic_text(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(value, encoding='utf-8')
    tmp.replace(path)


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def write_csv(rows, fields, path):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(stream.getvalue(), path)



def append_csv_row(row, fields, path):
    with Path(path).open('a', encoding='utf-8', newline='') as stream:
        csv.DictWriter(stream, fieldnames=fields).writerow(row)

def write_metrics(score, output, prefix):
    rows = [dict(class_id=i, class_name=name, accuracy_percent=100 * score['acc'][i],
                 iou_percent=100 * score['iou'][i], target_points=score['target_points'][i],
                 predicted_points=score['predicted_points'][i]) for i, name in enumerate(CLASS_NAMES)]
    write_csv(rows, list(rows[0]), Path(output) / (prefix + '_classes.csv'))
    write_csv([dict(target_class=name, **{CLASS_NAMES[j]: score['confusion'][i][j]
                                        for j in range(12)}) for i, name in enumerate(CLASS_NAMES)],
              ['target_class'] + CLASS_NAMES, Path(output) / (prefix + '_confusion.csv'))



def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def parallelism(model):
    wrapped = isinstance(model, torch.nn.DataParallel)
    return dict(data_parallel=wrapped, devices=max(1, len(model.device_ids)) if wrapped else 1)


def maybe_data_parallel(model, enabled):
    if not enabled:
        return model
    if torch.cuda.device_count() < 2:
        raise ValueError('--data-parallel requires at least two visible CUDA GPUs')
    return torch.nn.DataParallel(model)


def parameter_summary(model):
    """Disjoint parameter/L2 groups; Conv3 EBTE is never counted as encoder."""
    grouped = {}
    for name, parameter in unwrap_model(model).named_parameters():
        if name.startswith('conv3.adapter.'):
            group = 'ebte_conv3'
        elif name.startswith('bottleneck_ebte.'):
            group = 'ebte_bottleneck'
        elif name.startswith(('conv1.', 'conv2.', 'conv3.', 'conv4.')):
            group = 'encoder'
        elif name.startswith('transformer.'):
            group = 'transformer'
        elif name.startswith('cmtr.'):
            group = 'cmtr'
        elif name.startswith('srrf.'):
            group = 'srrf'
        elif name.startswith(('deconv1.', 'deconv2.', 'deconv3.', 'deconv4.', 'outconv.')):
            group = 'decoder'
        else:
            group = 'other'
        grouped.setdefault(group, []).append(parameter.detach().float())
    return [dict(module=group, parameters=sum(p.numel() for p in values),
                 l2=float(torch.stack([p.square().sum() for p in values]).sum().sqrt().cpu()))
            for group, values in sorted(grouped.items())]


def write_module_monitor(model, path, *, run_id, epoch):
    rows = [dict(run_id=run_id, epoch=epoch, **row) for row in parameter_summary(model)]
    write_csv(rows, ['run_id', 'epoch', 'module', 'parameters', 'l2'], path)
    return rows


def log_message(output, message):
    print(message, flush=True)
    with (Path(output) / 'train.log').open('a', encoding='utf-8') as stream:
        stream.write(message + '\n')

def build_datasets(cfg, *, train=True):
    from datasets.synthia import Synthia4DDataset
    if not cfg['data_root']:
        raise ValueError('Set --data-root to the preprocessed Synthia NPZ directory')
    root = resolve_config_path(cfg['data_root'])
    metadata = resolve_config_path(cfg['metadata_root'])
    kwargs = dict(root=root, labelweight=metadata / 'labelweights.npz',
                  frames_per_clip=cfg['frames_per_clip'], num_points=cfg['num_points'],
                  context=cfg['data']['context'], cache_size=cfg['data']['cache_size'],
                  with_metadata=True)
    test_data = Synthia4DDataset(meta=metadata / 'test_raw.txt', train=False, **kwargs)
    if not train:
        return test_data
    train_data = Synthia4DDataset(meta=metadata / 'trainval_raw.txt', train=True, **kwargs)
    if set(train_data.meta).intersection(test_data.meta):
        raise ValueError('Training and test metadata overlap')
    return train_data, test_data


def data_identity(dataset):
    return dict(meta=[tuple(row) for row in dataset.meta], samples=len(dataset),
                labelweights=np.asarray(dataset.labelweights, dtype=np.float32).tolist())


def make_loader(dataset, *, train=False, batch_size=8, workers=10):
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=workers,
                      pin_memory=True, drop_last=train, persistent_workers=False)


def half_inputs(batch, half, device):
    values = [value.to(device, non_blocking=True) for value in batch['halves'][half * 4:half * 4 + 4]]
    times, valid = batch.get('times'), batch.get('valid')
    times = times.to(device, non_blocking=True) if times is not None else None
    valid = valid.to(device, non_blocking=True) if valid is not None else None
    return (*values, times, valid)


def forward_half(model, xyz, rgb, times, valid):
    if times is not None and valid is not None:
        return model(xyz, rgb, times=times, valid=valid)
    if unwrap_model(model).use_cmtr:
        raise ValueError('CMTR requires times and valid frame metadata')
    return model(xyz, rgb)


def masked_loss(logits, label, mask, weight=None):
    if logits.ndim != 4 or logits.shape[2] != 12 or logits.shape[:2] + logits.shape[3:] != label.shape:
        raise ValueError('Expected logits [B,T,12,N] and labels [B,T,N]')
    if mask.shape != label.shape:
        raise ValueError('Label/mask dimensions differ')
    # Void labels never enter CE; dataset also maps them to zero with mask=0.
    safe_label = torch.where(mask.bool(), label, torch.zeros_like(label))
    losses = F.cross_entropy(logits.transpose(1, 2), safe_label, weight=weight, reduction='none')
    # Preserve the released half-normalization, including the +1 denominator.
    return (losses * mask).sum() / (mask.sum() + 1)


class SegmentationMetrics:
    def __init__(self):
        self.confusion = torch.zeros((12, 12), dtype=torch.int64)

    def update(self, prediction, target, mask):
        active = mask.reshape(-1).detach().cpu().bool()
        prediction = prediction.reshape(-1).detach().cpu()[active].long()
        target = target.reshape(-1).detach().cpu()[active].long()
        if len(target) and (target.min() < 0 or target.max() >= 12 or prediction.min() < 0 or prediction.max() >= 12):
            raise ValueError('An active semantic label/prediction is outside [0,11]')
        self.confusion += torch.bincount(target * 12 + prediction, minlength=144).reshape(12, 12)

    def result(self):
        conf = self.confusion
        target, prediction, tp = conf.sum(1), conf.sum(0), conf.diag()
        total = int(target.sum())
        if not total:
            raise ValueError('No valid semantic points were evaluated')
        acc = tp.double() / target.clamp_min(1)
        iou = tp.double() / (target + prediction - tp).clamp_min(1)
        return dict(mean_acc=float(acc.mean()), mean_iou=float(iou.mean()),
                    point_accuracy=float(tp.sum()) / total, valid_points=total,
                    acc=acc.tolist(), iou=iou.tolist(), target_points=target.tolist(),
                    predicted_points=prediction.tolist(), confusion=conf.tolist())


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, loader, device, print_freq=50):
    model.eval()
    sync(device)
    started = time.monotonic()
    metrics = SegmentationMetrics()
    loss_sum, count, samples = 0., 0, 0
    for step, batch in enumerate(loader):
        for half in (0, 1):
            xyz, rgb, label, mask, times, valid = half_inputs(batch, half, device)
            logits = forward_half(model, xyz, rgb, times, valid)
            loss = masked_loss(logits, label, mask)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite evaluation loss')
            loss_sum += float(loss)
            count += 1
            metrics.update(logits.argmax(2), label, mask)
            if half == 0:
                samples += len(label)
        if step % print_freq == 0 or step + 1 == len(loader):
            print(f'Evaluate batch {step + 1}/{len(loader)}', flush=True)
    if samples != len(loader.dataset):
        raise ValueError('Evaluation must retain the final incomplete batch')
    sync(device)
    return dict(**metrics.result(), loss=loss_sum / count, samples=samples,
                seconds=time.monotonic() - started)
