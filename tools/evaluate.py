"""One fixed MoTrend evaluation, without sampling or coefficient search."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import rng
from tools.numerics import (CHECKPOINT_SHA, INPUT_SHA, MODEL, OUTPUT_SHA, RNG_SHA,
                            add_video, base_rows, canonical_sums_sha, isolated_rng,
                            load_arrays, read, recorded_order, require, sha,
                            summarize, update_output_hash, verify_arrays,
                            verify_reference, write)

EXTENSION_SHA = '5bb19112c3ca6e90e06a3eba8fa716692edef3a70cfa20a69688cf9289e30aff'
SEED, CONTEXT = 54, 14


def same_tree(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_tree(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if hasattr(left, 'detach') and hasattr(left, 'dtype'):
        return left.dtype == right.dtype and np.array_equal(left.detach().cpu().numpy(), right.detach().cpu().numpy())
    return left == right


def data_identity(root, data, *, hash_contents=True):
    data = Path(data).resolve()
    manifest = read(Path(root) / 'data/FULL_DATA_MANIFEST.json')
    names = manifest['ordered_names']
    require(len(names) == len(set(names)) == len(manifest['files']) == 567
            and set(names) == set(manifest['files']), 'Wrong full567 manifest')
    require(set(n for n in os.listdir(data) if n.endswith('.npz')) == set(names), 'Require original full567 NPZ population')
    result = {}
    for name in names:
        require(isinstance(name, str) and Path(name).name == name
                and not any(c in name for c in ('\\', ':', '\x00')), 'Unsafe data filename')
        path, expected = data / name, manifest['files'][name]
        require(path.is_file(), 'Missing data: ' + name)
        before, link_before, resolved = path.stat(), path.lstat(), path.resolve()
        require(before.st_size == expected['bytes'], 'Original data size differs: ' + name)
        if hash_contents:
            require(sha(path) == expected['sha256'], 'Original data SHA differs: ' + name)
        after, link_after = path.stat(), path.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(identity(before) == identity(after) and identity(link_before) == identity(link_after)
                and path.resolve() == resolved, 'Data changed during verification: ' + name)
        result[name] = dict(target=str(resolved), file=identity(after), link=identity(link_after))
    return manifest, result


def native_environment(model, torch):
    require(sys.platform.startswith('linux') and sys.version_info[:2] == (3, 10),
            'Exact native replay requires Linux/Python3.10')
    require(torch.__version__.split('+')[0] == '2.1.2' and torch.version.cuda == '11.8'
            and np.__version__ == '1.26.4', 'Use recorded Torch2.1.2/cu118/NumPy1.26.4 environment')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one visible native CUDA GPU required')
    require(torch.get_default_dtype() == torch.float32 and torch.get_num_threads() == 2,
            'Use FP32 and two CPU threads')
    ops = model.tube_embedding.pointnet2_ops
    require(getattr(ops, '_ext', None) is not None and not getattr(ops, 'LOCAL_DIAGNOSTIC_ONLY', False),
            'Compiled native pointnet2 binding is required')
    extension = Path(ops._ext.__file__)
    require(sha(extension) == EXTENSION_SHA, 'Original native pointnet2 binary differs')
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), numpy=np.__version__, gpu=torch.cuda.get_device_name(0),
                extension_sha256=EXTENSION_SHA, threads=torch.get_num_threads(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark)


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(str((name, tuple(value.shape), str(value.dtype))).encode('utf8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fixed_loader(dataset, saved, output, torch):
    """Restore the recorded global, loader and all10 worker states literally."""
    from torch.utils.data import DataLoader
    generator = torch.Generator()
    generator.set_state(saved['loader']['loader_generator_initial_state'])
    require(rng.same_state(rng.snapshot_current(torch, True), saved['global_state'], torch),
            'Fixed reset does not match the actual saved global state')
    rng.restore_current(saved['global_state'], torch)

    def restore_worker(worker_id):
        worker = saved['loader']['worker_initial_states'][worker_id]
        state = dict(python=worker['python_initial_state'], numpy=worker['numpy_initial_state'],
                     torch_cpu=worker['torch_cpu_initial_state'], torch_cuda=None)
        require(rng.same_state(rng.snapshot_current(torch, False), state, torch), 'Worker seed algorithm differs')
        rng.restore_current(state, torch)
        write(output / f'WORKER_{worker_id:02d}_RESTORED.json', dict(passed=True, worker_id=worker_id,
              source_rng_archive_sha256=RNG_SHA))

    loader = DataLoader(dataset, batch_size=14, shuffle=False, num_workers=10,
                        pin_memory=True, drop_last=False, persistent_workers=False, generator=generator,
                        timeout=120, worker_init_fn=restore_worker)
    return loader, generator


def evaluate_once(model, dataset, saved, output, torch):
    input_digest, output_digest, local_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    totals, truth, counts = {}, {}, {}
    labels_parts, index_parts, probability_parts, batch_sizes = [], [], [], []
    clip_correct, begin = 0, time.monotonic()
    with isolated_rng(SEED), torch.no_grad():
        rng.restore_current(saved['global_state'], torch)
        loader, generator = fixed_loader(dataset, saved, output, torch)
        for batch, (inputs, labels, indices) in enumerate(loader):
            for key, value in sorted(inputs.items()):
                input_digest.update(key.encode())
                input_digest.update(value.contiguous().numpy().tobytes())
            local_digest.update(b'local')
            local_digest.update(inputs['local'].contiguous().numpy().tobytes())
            y, ids = labels.numpy(), indices.numpy()
            for label, index in zip(y, ids):
                index, label = int(index), int(label)
                require(index not in truth or truth[index] == label, 'Mixed video labels')
                truth[index] = label
                counts[index] = counts.get(index, 0) + 1
            labels_parts.append(y.copy())
            index_parts.append(ids.copy())
            batch_sizes.append(len(y))
            x = inputs.to('cuda', non_blocking=True)
            versions = {key: value._version for key, value in x.items()}
            logits = model(x)
            require(logits.dtype == torch.float32 and tuple(logits.shape) == (len(y), 20)
                    and bool(torch.isfinite(logits).all()), 'Invalid FP32 model output')
            require(all(value._version == versions[key] for key, value in x.items()), 'Model changed input tensors')
            probabilities = logits.softmax(-1).cpu().numpy()
            update_output_hash(output_digest, probabilities, y, ids)
            add_video(probabilities, ids, totals)
            probability_parts.append(probabilities)
            clip_correct += int((probabilities.argmax(1) == y).sum())
            if batch % 100 == 0:
                print('MoTrend_FIXED_REPLAY', 'batch', batch, 'windows', sum(batch_sizes), flush=True)
        require(torch.equal(generator.get_state(), saved['loader']['loader_generator_after_base_seed_state']),
                'Fixed loader generator consumption differs')
    rows = base_rows(totals, truth, counts, dataset.video_names)
    require(sum(batch_sizes) == 5356 and len(rows) == 287 and len(dataset.video_names) == 297,
            'Full287 coverage differs')
    require(batch_sizes == [14] * 382 + [8], 'Fixed B14 boundaries changed')
    arrays = dict(labels=np.concatenate(labels_parts), indices=np.concatenate(index_parts),
                  batch_sizes=np.asarray(batch_sizes, np.int64), model_ids=np.asarray([MODEL]))
    arrays['probabilities__MoTrend'] = np.concatenate(probability_parts)
    record = dict(model_id=MODEL, evaluation_seed=SEED, context_seed=CONTEXT,
                  input_sha256=input_digest.hexdigest(), local_input_sha256=local_digest.hexdigest(),
                  output_sha256=output_digest.hexdigest(), sums_sha256=canonical_sums_sha(rows),
                  clips=5356, loaded_videos=297, scored_videos=287, clip_correct=clip_correct,
                  unscored=[str(n) for i, n in enumerate(dataset.video_names) if i not in truth],
                  elapsed_seconds=time.monotonic() - begin)
    return record, arrays, rows


def replay(root, data, output):
    """Replay the selected 30+20 checkpoint and compare every saved probability."""
    root, data, output = Path(root).resolve(), Path(data).resolve(), Path(output).resolve()
    require(not os.path.lexists(output) and not output.is_relative_to(root) and not output.is_relative_to(data),
            'Output must be new and outside the package and dataset')
    verify_reference(root)
    manifest, original_data = data_identity(root, data)
    config = read(root / 'config/selected.json')
    require(config['alpha'] == .5 and config['checkpoint_epoch'] == 30 and config['parameters'] == 408682,
            'Wrong selected MotionTrend configuration')
    checkpoint = root / 'model/checkpoints/best.pth'
    reference = root / 'results/selected'
    require(sha(checkpoint) == CHECKPOINT_SHA == config['checkpoint_sha256'], 'Trained checkpoint differs')
    require(sha(reference / 'RNG_STATE.pth') == RNG_SHA, 'Saved sampling states differ')
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(root / 'tools/native'))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    from model.backbone import build_model
    from tools.fixed_evaluation import NativeFixedEvaluator
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, architecture = build_model(srr_scale=.5, switch_epoch=30)
    environment = native_environment(model, torch)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(payload['epoch'] == 30 and payload['relation_optimizer_steps'] == 320
            and len(payload['model']) == 132, 'Wrong selected checkpoint identity')
    model.load_state_dict(payload['model'], strict=True)
    model.set_phase(30)
    model.cuda().eval().requires_grad_(False)
    require(sum(p.numel() for p in model.parameters()) == 408682 and len(model.state_dict()) == 132,
            'MotionTrend registry differs')
    before = state_sha(model)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'ENVIRONMENT.json', environment)
    write(output / 'POLICY.json', dict(model_name=MODEL, alpha=.5, checkpoint_epoch=30,
          checkpoint_sha256=CHECKPOINT_SHA, input_sha256=INPUT_SHA, output_sha256=OUTPUT_SHA,
          single_model=True, ensemble=False, no_training=True, sampling_search=False,
          evaluation_seed=SEED, context_seed=CONTEXT, workers=10, batch_size=14, architecture=architecture))
    statistics = dict(calls=0, raw_nonzero_batches=0, maximum_raw_abs=0.)
    def observe_srr(module, inputs, result):
        require(isinstance(result, torch.Tensor), 'Unexpected SRR output')
        statistics['calls'] += 1
        statistics['raw_nonzero_batches'] += int(bool(torch.count_nonzero(result)))
        statistics['maximum_raw_abs'] = max(statistics['maximum_raw_abs'], float(result.abs().max()))
    hook = model.prepool_relation.register_forward_hook(observe_srr)
    begin = time.monotonic()
    try:
        evaluator = NativeFixedEvaluator(root, data, output, 24, 32,
                                        local_policy='reference24_span', model_id=MODEL)
        result = evaluator.run(model, 'replay', expected_input_sha=INPUT_SHA)
        folder = Path(result['path'])
        native = read(folder / 'HASHES.json')
        record = dict(native)
        for key in ('output_sha256', 'sums_sha256'):
            record[key] = native[key][MODEL]
        rows = read(folder / 'RESULT.json')['natural']
        arrays = load_arrays(folder)
        exact = verify_arrays(record, arrays, rows)
        reference_record, reference_arrays = read(reference / 'HASHES.json'), load_arrays(reference)
        require(record['local_input_sha256'] == reference_record['local_input_sha256'], 'Local points changed')
        for name in reference_arrays:
            require(arrays[name].dtype == reference_arrays[name].dtype
                    and np.array_equal(arrays[name], reference_arrays[name]), 'Saved arrays differ: ' + name)
        require(rows == read(reference / 'RESULT.json')['full287'], 'Saved video rows differ')
        require(statistics['calls'] == statistics['raw_nonzero_batches'] == 383, 'SRR was not active in all batches')
        require(state_sha(model) == before, 'Inference changed model parameters')
        require(data_identity(root, data, hash_contents=False)[1] == original_data, 'Input data changed during replay')
        completion = dict(passed=True, model_name=MODEL, correct=283, total=287, accuracy_percent=100 * 283 / 287,
              alpha=.5, checkpoint_epoch=30, checkpoint_sha256=CHECKPOINT_SHA, exact_replay=exact,
              input_sha256=INPUT_SHA, output_sha256=OUTPUT_SHA, rng_archive_sha256=RNG_SHA,
              srr_statistics=statistics, trained_parameters_unchanged=True,
              single_model=True, ensemble=False, no_training=True, sampling_search=False,
              wall_seconds=time.monotonic() - begin)
        write(output / 'COMPLETED.json', completion)
        print(json.dumps(completion, ensure_ascii=False))
        return completion
    except BaseException as error:
        write(output / 'FAILED.json', dict(passed=False, error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        hook.remove()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    replay(args.root, args.data, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
