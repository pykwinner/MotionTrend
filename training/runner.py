"""Portable single-seed training: parent search30, then frozen-parent SRR20.

The fixed 287-video scores select checkpoints. This reproduces the recorded
development protocol; it is not an independently held-out model-selection result.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from . import engine as E
from . import loss as S
from . import policy as P
from . import state as H
from .checkpoints import load_resume, restore_runtime, verify_evaluation
from .runtime import configure_runtime, consume_dev_rng, preflight, training_datasets
from .transition import restore_selected_parent


def validate_recipe(root):
    """Reject configuration edits that the exact released trainer cannot honor."""
    recipe = H.read(root / 'config/training_recipe.json')['recipe']
    expected = dict(total_epochs=50, phase_boundary_epoch=30, from_scratch=True,
        seed=0, batches_per_epoch=320, train_clips=4478, batch_size=14, workers=10,
        optimizer='SGD', lr=.01, momentum=.9, weight_decay=1e-4,
        lr_warmup_epochs=10, lr_warmup_factor=1e-5, parent_lr_milestones=[11,20],
        lr_gamma=.1, relation_lr_milestones=[5,15], relation_lr_warmup_epochs=0,
        keep_kl_weight=.25, evaluation_seed=54, context_seed=14, train_context_seed=0,
        evaluation_input_sha256=P.INPUT_SHA, test_videos=287, test_windows=5356,
        parent_epochs=30, srr_epochs=20, srr_rho=.02, srr_epsilon=1e-6,
        training_srr_scale=1., inference_srr_scale=.5, scale_position='before tanh',
        total_optimizer_steps=16000, joint_training=False)
    H.require(all(recipe.get(key) == value for key,value in expected.items()),
              'training_recipe.json differs from the released exact-training protocol')
    return recipe


def code_identity(root):
    """Bind resumes to the implementation and all protocol configuration."""
    files = [root / 'train.py']
    for directory in ('model', 'training', 'tools', 'config'):
        files.extend(p for p in (root / directory).rglob('*') if p.is_file()
                     and p.suffix in ('.py', '.json') and '__pycache__' not in p.parts)
    values = {p.relative_to(root).as_posix(): H.sha(p) for p in sorted(set(files))}
    return values


def acquire_run_lock(output):
    """The kernel releases this Linux lock on crashes; stale files are harmless."""
    import fcntl
    path = output / 'RUN_LOCK.json'
    H.require(not path.is_symlink(), 'Run lock cannot be a symbolic link')
    stream = path.open('a+', encoding='utf8')
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError('Another process owns this training output; refusing concurrent resume')
    stream.seek(0)
    stream.truncate()
    json.dump(dict(pid=os.getpid(), purpose='Exclusive training output ownership'),stream)
    stream.flush()
    return stream


def validate_output(root, data, output, resume):
    root, data, output = root.resolve(), data.resolve(), output.resolve()
    H.require(output != root and not root.is_relative_to(output)
              and output != data and not data.is_relative_to(output)
              and not output.is_relative_to(data), 'Output overlaps the code root or source data')
    for name in ('model', 'config', 'data', 'training', 'tools', 'results', 'tests'):
        H.require(not output.is_relative_to(root / name), 'Use an output outside release artifacts (for example runs/train)')
    H.require(not output.is_symlink(), 'Output cannot be a symbolic link')
    if resume:
        H.require(output.is_dir() and (output / 'run_configuration.json').is_file(),
                  '--resume requires an existing owned training run')
    else:
        H.require(not output.exists(), 'Preserve existing output; use --resume or a new directory')
        output.mkdir(parents=True)
    return output


def run(args, root):
    root = Path(root).resolve()
    output = validate_output(root, args.data, args.output, args.resume)
    rows = []
    started = time.monotonic()
    lock = None
    try:
        validate_recipe(root)
        configure_runtime()
        lock = acquire_run_lock(output)
        sys.path.insert(0, str(root / 'tools/native'))
        from model.backbone import build_model, load_fresh_initialization, assert_from_scratch_training
        from model.point_ops import bind
        from tools.evaluate import data_identity, native_environment
        from tools.fixed_evaluation import NativeFixedEvaluator

        sources = code_identity(root)
        data_identity(root, args.data)
        reference = H.read(root / 'config/REFERENCE_TRAJECTORY.json')
        train_data, dev_data = training_datasets(root, args.data, reference)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=14, shuffle=True,
                      num_workers=10, pin_memory=True, timeout=120)
        dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=14, shuffle=False,
                    num_workers=10, pin_memory=True, timeout=120)
        training, development = H.InputRecorder(train_loader), H.InputRecorder(dev_loader)
        batches = len(train_loader)
        H.require(batches == 320, 'The declared training recipe requires 320 updates per epoch')
        model, _ = build_model(seed=0, pointnet2_ops=bind(), switch_epoch=30, srr_scale=1.)
        H.require(model.switches.as_dict() == dict(ebte=True, ctr=True, srr=True),
                  'This released training recipe requires all three modules enabled')
        fresh_path = root / 'model/checkpoints/initialization.pth'
        fresh_sha = H.sha(fresh_path)
        initialization = load_fresh_initialization(model,
            torch.load(fresh_path, map_location='cpu', weights_only=False), fresh_sha)
        assert_from_scratch_training(model)
        model.cuda()
        environment = native_environment(model, torch)
        H.require(len(model.state_dict()) == 132 and sum(p.numel() for p in model.parameters()) == 408682,
                  'The published default model must have 408682 parameters and 132 state tensors')
        before = H.capture_rng()
        evaluator = NativeFixedEvaluator(root, args.data, output, 24, 32,
                    local_policy='reference24_span', model_id='MoTrend')
        H.require(H.same_tree(before, H.capture_rng()), 'Evaluator construction consumed TRAIN RNG')
        configuration = dict(schema='motiontrend-public-parent30-srr20-v1', name='MoTrend_Default',
            run_id='MoTrend_Default_seed0', from_scratch=True, single_model=True, ensemble=False,
            seed=0, epochs=50, parent_epochs=30, relation_epochs=20, train_batches=batches,
            train_clips=len(train_data), batch_size=14, workers=10, natural_eval_videos=287,
            common_eval_videos=192, local_frames=24, global_frames=32, local_points=2048,
            context_points=256, parameters=408682, state_tensors=132, rho=.02, epsilon=1e-6,
            keep_kl_weight=.25, training_alpha=1., evaluation_alpha=.5,
            evaluation_alpha_location='before_tanh', evaluation_seed=54, evaluation_context_seed=14,
            fixed_input_sha256=P.INPUT_SHA, source_files_sha256=sources,
            initialization_archive_sha256=fresh_sha, initialization=initialization,
            full_data_manifest_sha256=H.sha(root / 'data/FULL_DATA_MANIFEST.json'),
            training_manifest_sha256=H.sha(root / 'data/TRAINING_MANIFEST.json'),
            trajectory_sha256=H.sha(root / 'config/REFERENCE_TRAJECTORY.json'),
            parent_selection='Own fixed287 maximum among epochs0..29, earliest tie; restore tensors only',
            checkpoint_selection='Own fixed287 maximum among nonzero SRR-trained epochs30..49, earliest tie',
            selection_scope='Evaluation-selected development protocol, not independent held-out selection',
            training_rng_policy='Retain actual end-of-epoch29 RNG when selecting the parent',
            parent_lr=dict(base=.01, momentum=.9, weight_decay=1e-4, warmup_epochs=10,
                           warmup_factor=1e-5, milestones=[11,20]),
            relation_lr=dict(base=.01, momentum=.9, weight_decay=1e-4, milestones=[5,15],
                             clock='Fresh SRR-only optimizer; no second warmup'))
        epoch_start = parent_steps = relation_steps = 0
        parent_sha = selected_parent = None
        if args.resume:
            payload, rows = load_resume(output, configuration, model)
            parent_optimizer, parent_scheduler, relation_optimizer = restore_runtime(model,payload,batches)
            if payload['epoch'] >= 0:
                parent_steps, relation_steps = payload['parent_optimizer_steps'], payload['relation_optimizer_steps']
                parent_sha, selected_parent = payload['frozen_parent_sha256'], payload['selected_parent']
            epoch_start = payload['epoch'] + 1
            H.atomic_json(output / ('RESUME_' + str(time.time_ns()) + '.json'),
                          dict(start_epoch=epoch_start, completed_epochs=len(rows), no_completed_epoch_repeated=True))
        else:
            H.atomic_json(output / 'run_configuration.json', configuration)
            H.atomic_json(output / 'ENVIRONMENT.json', environment)
            H.atomic_json(output / 'PREFLIGHT.json', preflight(model, train_data))
            fresh = dict(model=H.cpu_tree(model.state_dict()), epoch=-1, control=configuration,
                         architecture=model.architecture_config(), model_phase=model.phase_receipt(),
                         rng=H.cpu_tree(H.capture_rng()), from_scratch=True, trained_checkpoint_loaded=False)
            H.save_checkpoint(output / 'initialization.pth', fresh)
            H.atomic_json(output / 'INITIALIZATION.json', dict(checkpoint='initialization.pth',
                checkpoint_sha256=H.sha(output / 'initialization.pth'), epoch=-1,
                from_scratch=True, trained_checkpoint_loaded=False))
            if args.preflight_only:
                fixed = E.fixed_evaluate(model, evaluator, 'fixed_preflight', configuration, H, P.INPUT_SHA)
                H.atomic_json(output / 'PREFLIGHT_ONLY.json', dict(passed=True, retained_optimizer_steps=0,
                    fixed_input_sha256=fixed['input_sha256'], evaluation=fixed['summary']))
                return 0
            parent_optimizer = E.new_optimizer(model, 0, torch)
            parent_scheduler = E.parent_scheduler(parent_optimizer, batches, H)
            relation_optimizer = None
        for epoch in range(epoch_start, 50):
            phase = P.phase_for_epoch(epoch)
            if epoch == 30 and relation_optimizer is None:
                H.require(parent_steps == 30*batches and len(rows) == 30, 'Incomplete parent30 budget')
                selected_parent = restore_selected_parent(model, output, rows, configuration, torch, H)
                result = E.fixed_evaluate(model, evaluator, 'fixed_selected_parent', configuration, H, P.INPUT_SHA)
                H.require(result['summary'] == selected_parent['selected_record']['evaluation'],
                          'Selected parent did not replay its fixed287 score')
                saved = H.read(output / f"fixed_epoch_{selected_parent['selected_parent_epoch']:03d}" / 'HASHES.json')
                H.require(result['record']['output_sha256'] == saved['output_sha256'],
                          'Selected parent per-window probabilities changed')
                selected_parent['fixed_replay_verified'] = True
                selected_parent['fixed_replay_output_sha256'] = result['record']['output_sha256']
                H.atomic_json(output / 'PARENT_SELECTED.json', selected_parent)
                parent_sha = selected_parent['selected_parent_state_sha256']
                relation_optimizer = E.new_optimizer(model, 30, torch)
                H.atomic_json(output / 'TRANSITION.json', dict(epoch=30, selected_parent=selected_parent,
                    frozen_parent_sha256=parent_sha, fresh_relation_optimizer=True, training_rng_rewound=False))
            model.set_phase(epoch)
            model.set_srr_scale(1.)
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            stats = E.train_epoch(model, parent_optimizer if phase == 'parent' else relation_optimizer,
                parent_scheduler if phase == 'parent' else None, training, epoch, batches, torch, H, S)
            if phase == 'parent':
                parent_steps += batches
            else:
                relation_steps += batches
            dev_receipt = consume_dev_rng(development)
            stream = E.parent_stream_gate(parent_optimizer, parent_scheduler, epoch, stats, reference, torch) if epoch < 11 else None
            if phase == 'relation':
                H.require(H.state_sha(H.split_state(model)) == parent_sha, 'Frozen parent tensors changed')
            fixed = E.fixed_evaluate(model, evaluator, f'fixed_epoch_{epoch:03d}', configuration, H, P.INPUT_SHA)
            eligible = phase == 'relation' and relation_steps > 0 and stats['relation_nonzero_exit']
            payload = dict(model=H.cpu_tree(model.state_dict()), epoch=epoch, control=configuration,
                architecture=model.architecture_config(), model_phase=model.phase_receipt(),
                phase=dict(epoch=epoch, phase=phase, relation_active=phase == 'relation'),
                parent_optimizer=H.cpu_tree(parent_optimizer.state_dict()),
                parent_scheduler=H.cpu_tree(parent_scheduler.state_dict()),
                relation_optimizer=H.cpu_tree(relation_optimizer.state_dict()) if relation_optimizer else None,
                global_step=parent_steps+relation_steps, parent_optimizer_steps=parent_steps,
                relation_optimizer_steps=relation_steps, frozen_parent_sha256=parent_sha,
                selected_parent=selected_parent, rng=H.cpu_tree(H.capture_rng()), from_scratch=True,
                single_model=True, ensemble=False, training_alpha=1., evaluation_alpha=.5)
            path = output / f'model_{epoch}.pth'
            H.save_checkpoint(path, payload)
            row = dict(epoch=epoch, run_id=configuration['run_id'], phase=phase, eligible_full=eligible,
                checkpoint=path.name, checkpoint_sha256=H.sha(path), evaluation=fixed['summary'],
                input_sha256=P.INPUT_SHA, evaluation_alpha=.5, parent_optimizer_steps=parent_steps,
                relation_optimizer_steps=relation_steps, train=stats, train_input=training.receipt(),
                dev_rng_iterator=dev_receipt, stream_contract=stream,
                evaluation_completion_sha256=verify_evaluation(output / f'fixed_epoch_{epoch:03d}'),
                total_seconds=time.monotonic()-tick, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            H.atomic_json(output / f'epoch_{epoch:03d}_metrics.json', row)
            rows.append(row)
            H.atomic_json(output / 'CURRENT.json', P.select_best(rows, run_id=configuration['run_id']))
            print('MOTREND_EPOCH', epoch, phase, fixed['summary']['natural'], flush=True)
        H.require(parent_steps == 9600 and relation_steps == 6400 and len(rows) == 50,
                  'Training did not complete the declared 30+20 budget')
        H.require(code_identity(root) == sources, 'Source/configuration files changed during training')
        H.atomic_json(output / 'COMPLETED.json', dict(passed=True, completed_epochs=50,
            from_scratch=True, single_model=True, ensemble=False, parent_optimizer_steps=parent_steps,
            relation_optimizer_steps=relation_steps, total_optimizer_steps=parent_steps+relation_steps,
            selection=P.select_best(rows), fixed_input_sha256=P.INPUT_SHA,
            selected_parent=selected_parent, this_process_seconds=time.monotonic()-started))
        return 0
    except BaseException as error:
        # A rejected second process must not overwrite the owner's status.
        if lock is not None:
            H.atomic_json(output / 'FAILED.json', dict(passed=False, error=repr(error),
                traceback=traceback.format_exc(), closed_epochs=len(rows)))
        raise
    finally:
        if lock is not None:
            lock.close()
