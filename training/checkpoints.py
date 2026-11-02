"""Closed-epoch checkpoints and restart without repeating completed epochs."""
from __future__ import annotations

from pathlib import Path
import re
import time

import torch

from . import policy as P
from . import state as H
from .transition import validate_selected_provenance


def restore_runtime(model, payload, batches):
    """Restore optimizer clocks before the saved RNG, including at epoch 29."""
    from .engine import new_optimizer, parent_scheduler
    model.load_state_dict(payload['model'], strict=True)
    parent = new_optimizer(model, 0, torch)
    scheduler = parent_scheduler(parent, batches, H)
    relation = None
    if payload['epoch'] >= 0:
        parent.load_state_dict(payload['parent_optimizer'])
        scheduler.load_state_dict(payload['parent_scheduler'])
        if payload['relation_optimizer'] is not None:
            relation = new_optimizer(model, 30, torch)
            relation.load_state_dict(payload['relation_optimizer'])
    model.set_phase(max(0, payload['epoch']))
    model.set_srr_scale(1.)
    H.restore_rng(payload['rng'])
    return parent, scheduler, relation


def validate_optimizer(optimizer, parameters, expected_lr):
    parameters = list(parameters)
    groups = optimizer.get('param_groups')
    H.require(isinstance(groups,list) and len(groups) == 1, 'Resume requires one SGD parameter group')
    group = groups[0]
    keys = group.get('params', [])
    H.require(len(keys) == len(set(keys)) == len(parameters), 'Resume optimizer parameter registry differs')
    for name,value in dict(lr=expected_lr,momentum=.9,dampening=0,weight_decay=1e-4,
                           nesterov=False,maximize=False).items():
        H.require(group.get(name) == value, 'Resume SGD hyperparameter differs: ' + name)
    state = optimizer.get('state', {})
    H.require(set(state) == set(keys), 'Every trained parameter must have its SGD momentum')
    for key,parameter in zip(keys,parameters):
        momentum = state[key].get('momentum_buffer')
        H.require(isinstance(momentum,torch.Tensor) and momentum.shape == parameter.shape
                  and momentum.dtype == parameter.dtype and bool(torch.isfinite(momentum).all()),
                  'Resume SGD momentum is invalid')


def verify_evaluation(directory):
    directory = Path(directory).resolve()
    marker = directory / 'COMPLETED.json'
    H.require(marker.is_file() and not marker.is_symlink(), 'Evaluation was not closed')
    record = H.read(marker)
    H.require(record.get('passed') is True and record.get('input_sha256') == P.INPUT_SHA,
              'Evaluation fixed-input receipt differs')
    for name, digest in record['files_sha256'].items():
        path = directory / name
        H.require(path.name == name and path.resolve().parent == directory and
                  path.is_file() and not path.is_symlink() and H.sha(path) == digest,
                  'Closed evaluation artifact changed: ' + name)
    return H.sha(marker)


def closed_prefix(output, configuration):
    """A metric is published only after both its weight and evaluation close."""
    output = Path(output).resolve()
    rows = []
    for epoch in range(50):
        metric = output / f'epoch_{epoch:03d}_metrics.json'
        if not metric.is_file():
            break
        H.require(not metric.is_symlink(), 'Metric cannot be a symbolic link')
        row = H.read(metric)
        path = output / f'model_{epoch}.pth'
        H.require(row.get('epoch') == epoch and row.get('run_id') == configuration['run_id']
                  and row.get('checkpoint') == path.name and path.is_file()
                  and not path.is_symlink() and H.sha(path) == row['checkpoint_sha256'],
                  'A closed checkpoint or metric changed')
        evaluation_sha = verify_evaluation(output / f'fixed_epoch_{epoch:03d}')
        H.require(evaluation_sha == row['evaluation_completion_sha256'],
                  'Metric no longer identifies its original evaluation')
        rows.append(row)
    P.select_best(rows, run_id=configuration['run_id'])
    H.require(not any((output / f'epoch_{epoch:03d}_metrics.json').exists()
                      for epoch in range(len(rows) + 1, 50)),
              'Metric history has a gap; refusing to skip an unclosed epoch')
    return rows


def quarantine_unclosed(output, start_epoch):
    """Preserve only this run's interrupted next-epoch files before retrying."""
    output = Path(output).resolve()
    candidates = []
    for path in output.iterdir():
        match = re.fullmatch(r'(?:model_(\d+)\.pth(?:\.pending)?|fixed_epoch_(\d{3}))', path.name)
        if match and int(next(value for value in match.groups() if value is not None)) >= start_epoch:
            candidates.append(path)
    if start_epoch <= 30:
        for name in ('fixed_selected_parent', 'PARENT_SELECTED.json', 'TRANSITION.json'):
            path = output / name
            if path.exists():
                candidates.append(path)
    failed = output / 'FAILED.json'
    if failed.exists():
        candidates.append(failed)
    if not candidates:
        return
    target = output / ('interrupted_' + str(time.time_ns()))
    H.require(target.resolve().parent == output and not target.exists(), 'Unsafe quarantine target')
    for path in candidates:
        H.require(path.parent == output and path.resolve().parent == output and not path.is_symlink(),
                  'Refusing to move an artifact outside the owned output')
    target.mkdir()
    for path in candidates:
        path.rename(target / path.name)


def load_resume(output, configuration, model):
    output = Path(output).resolve()
    H.require(H.read(output / 'run_configuration.json') == configuration,
              'Resume configuration differs from this run')
    H.require(not (output / 'COMPLETED.json').exists()
              and not (output / 'PREFLIGHT_ONLY.json').exists(), 'Run is already complete or preflight-only')
    rows = closed_prefix(output, configuration)
    if rows:
        payload = torch.load(output / rows[-1]['checkpoint'], map_location='cpu', weights_only=False)
        epoch = len(rows) - 1
        H.require(payload.get('epoch') == epoch and payload.get('control') == configuration
                  and payload.get('from_scratch') is True, 'Resume checkpoint identity changed')
        expected = P.expected_steps(epoch, configuration['train_batches'])
        H.require(all(payload.get(key) == expected[key] for key in
                      ('global_step', 'parent_optimizer_steps', 'relation_optimizer_steps')),
                  'Resume optimizer update budget differs')
        H.require(payload['parent_scheduler']['last_epoch'] == expected['parent_scheduler_last_epoch'],
                  'Resume parent learning-rate clock differs')
        clock = payload['parent_scheduler']
        batches = configuration['train_batches']
        H.require(clock.get('milestones') == [11*batches,20*batches]
                  and clock.get('warmup_iters') == 10*batches and clock.get('warmup_factor') == 1e-5
                  and clock.get('gamma') == .1 and clock.get('base_lrs') == [.01],
                  'Resume parent scheduler policy differs')
        validate_optimizer(payload['parent_optimizer'], model.parent_parameters(),
                           P.parent_lr_for_step(expected['parent_optimizer_steps'],batches))
        H.require((payload.get('relation_optimizer') is not None) == (epoch >= 30),
                  'Resume SRR optimizer phase differs')
        if epoch >= 30:
            validate_optimizer(payload['relation_optimizer'], model.relation_parameters(),
                               P.relation_lr_for_step(expected['relation_optimizer_steps']-1,batches))
            validate_selected_provenance(payload, rows, output, configuration, H)
    else:
        receipt = H.read(output / 'INITIALIZATION.json')
        path = output / 'initialization.pth'
        H.require(receipt['checkpoint_sha256'] == H.sha(path), 'Owned fresh initialization changed')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        H.require(payload.get('epoch') == -1 and payload.get('from_scratch') is True
                  and payload.get('trained_checkpoint_loaded') is False
                  and payload.get('control') == configuration, 'Wrong restart initialization')
    actual, saved = model.state_dict(), payload['model']
    H.require(set(actual) == set(saved) and all(saved[key].shape == actual[key].shape
              and saved[key].dtype == actual[key].dtype and bool(torch.isfinite(saved[key]).all())
              for key in actual), 'Resume model state has incompatible or nonfinite tensors')
    quarantine_unclosed(output, len(rows))
    return payload, rows
