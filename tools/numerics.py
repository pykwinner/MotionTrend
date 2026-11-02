"""MoTrend's original window hashing and sequential FP32 video scoring.

This module needs only NumPy. Model identifiers are presentation metadata;
the numerical hash covers identities, labels and probability bytes.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import struct

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SELECTED = json.loads((_ROOT / 'config/selected.json').read_text(encoding='utf8'))
INPUT_SHA = _SELECTED['input_sha256']
OUTPUT_SHA = _SELECTED['output_sha256']
RNG_SHA = _SELECTED['rng_archive_sha256']
CHECKPOINT_SHA = _SELECTED['checkpoint_sha256']
MODEL = 'MoTrend'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def update_output_hash(digest, probabilities, labels, indices):
    arrays = (('indices', np.asarray(indices, dtype=np.int64)),
              ('labels', np.asarray(labels, dtype=np.int64)),
              ('probabilities', np.asarray(probabilities, dtype=np.float32)))
    for name, value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str((name, value.shape, str(value.dtype))).encode('utf-8'))
        digest.update(value.tobytes())


def add_video(probabilities, indices, totals):
    require(probabilities.dtype == np.float32, 'Sequential sums require FP32 probabilities')
    for probability, index in zip(probabilities, indices):
        key = int(index)
        if key not in totals:
            totals[key] = probability.copy()
        else:
            totals[key] += probability


def canonical_sums_sha(rows):
    digest = hashlib.sha256()
    require(len({row['name'] for row in rows}) == len(rows), 'Duplicate video names')
    for row in sorted(rows, key=lambda value: value['name']):
        name = row['name'].encode('utf-8')
        digest.update(struct.pack('<I', len(name)))
        digest.update(name)
        digest.update(np.asarray([row['label'], row['windows']], dtype='<i8').tobytes())
        digest.update(np.asarray(row['probability_sum'], dtype='<f4').tobytes())
    return digest.hexdigest()


def base_rows(totals, truth, counts, names):
    require(set(totals) == set(truth) == set(counts), 'Video population differs')
    result = []
    for index in sorted(truth):
        p = totals[index]
        prediction, label = int(p.argmax()), truth[index]
        result.append(dict(name=str(names[index]), video_index=index, label=label, windows=counts[index],
            prediction=prediction, correct=prediction == label, probability_sum=p.tolist(),
            probabilities=(p.astype(np.float64) / counts[index]).tolist()))
    return result


def summarize(rows):
    require(rows and len({r['name'] for r in rows}) == len(rows), 'Duplicate/empty rows')
    confusion = np.zeros((20, 20), np.int64)
    for row in rows:
        require(row['correct'] == (row['prediction'] == row['label']), 'Wrong correctness flag')
        confusion[row['label'], row['prediction']] += 1
    correct = sum(bool(row['correct']) for row in rows)
    return dict(correct=correct, total=len(rows), accuracy=correct / len(rows),
                confusion_matrix=confusion.tolist(), incorrect_videos=[r['name'] for r in rows if not r['correct']])


def load_arrays(directory):
    with np.load(Path(directory) / 'clips.npz', allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def verify_arrays(record, arrays, rows):
    require(set(arrays) == {'labels', 'indices', 'batch_sizes', 'model_ids', 'probabilities__MoTrend'},
            'Expected one MoTrend output stream')
    require(arrays['model_ids'].tolist() == [MODEL], 'Wrong model identity')
    require(record['input_sha256'] == INPUT_SHA and record['output_sha256'] == OUTPUT_SHA,
            'Wrong fixed numerical identity')
    require(arrays['batch_sizes'].tolist() == [14] * 382 + [8], 'Fixed B14 batch boundaries differ')
    labels, indices, probabilities = arrays['labels'], arrays['indices'], arrays['probabilities__MoTrend']
    require(labels.shape == indices.shape == (5356,) and labels.dtype == indices.dtype == np.int64,
            'Wrong labels/video indices')
    require(probabilities.shape == (5356, 20) and probabilities.dtype == np.float32
            and np.isfinite(probabilities).all() and (probabilities >= 0).all()
            and np.allclose(probabilities.sum(-1), 1, rtol=0, atol=1e-5), 'Invalid FP32 softmax probabilities')
    digest, totals, truth, counts, offset = hashlib.sha256(), {}, {}, {}, 0
    for count in arrays['batch_sizes']:
        end = offset + int(count)
        update_output_hash(digest, probabilities[offset:end], labels[offset:end], indices[offset:end])
        add_video(probabilities[offset:end], indices[offset:end], totals)
        for label, index in zip(labels[offset:end], indices[offset:end]):
            index, label = int(index), int(label)
            require(index not in truth or truth[index] == label, 'Mixed video labels')
            truth[index] = label
            counts[index] = counts.get(index, 0) + 1
        offset = end
    require(digest.hexdigest() == OUTPUT_SHA, 'Window probability bytes differ')
    require(len(rows) == len(totals) == len({row['name'] for row in rows}) == 287, 'Wrong scored population')
    require([row['video_index'] for row in rows] == sorted(totals), 'Video order differs')
    for row in rows:
        index = row['video_index']
        require(row['label'] == truth[index] and row['windows'] == counts[index], 'Video metadata differs')
        require(np.array_equal(totals[index], np.asarray(row['probability_sum'], dtype=np.float32)),
                'Sequential FP32 video sum differs')
        require(int(totals[index].argmax()) == row['prediction']
                and row['correct'] == (row['prediction'] == row['label']), 'Wrong video prediction')
        require(row['probabilities'] == (totals[index].astype(np.float64) / counts[index]).tolist(),
                'Video mean probability differs')
    require(canonical_sums_sha(rows) == record['sums_sha256'], 'Canonical video sums differ')
    require(summarize(rows)['correct'] == 283, 'Expected283/287')
    return dict(passed=True, correct=283, total=287, exact_window_arrays=5356,
                exact_video_rows=287, input_sha256=INPUT_SHA, output_sha256=OUTPUT_SHA)


def verify_reference(root):
    directory = Path(root) / 'results/selected'
    return verify_arrays(read(directory / 'HASHES.json'), load_arrays(directory), read(directory / 'RESULT.json')['full287'])


@contextmanager
def recorded_order(data, names):
    original, target = os.listdir, Path(data).resolve()

    def listing(path='.'):
        if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve() == target:
            return [os.fsencode(n) for n in names] if isinstance(path, bytes) else list(names)
        return original(path)

    os.listdir = listing
    try:
        yield
    finally:
        os.listdir = original


@contextmanager
def isolated_rng(seed):
    import torch
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
