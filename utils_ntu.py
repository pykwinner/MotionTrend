"""Official NTU video scoring, ordinary loaders and training-resume RNG helpers."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent

SCHEMA = 'motrend-ntu-joint-official-v1'
PREDICTION_FIELDS = ('video', 'label', 'prediction', 'correct', 'windows')


def boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ('true', '1', 'yes'):
            return True
        if value.lower() in ('false', '0', 'no'):
            return False
    raise argparse.ArgumentTypeError('Use true or false')


def setup(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng():
    """State for an explicit training resume, never an evaluation-point search."""
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state.get('torch_cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def ordinary_loader(ds, *, train=False, batch_size=24, workers=4):
    if type(batch_size) is not int or batch_size < 1 or type(workers) is not int or workers < 0:
        raise ValueError('batch_size must be positive and workers nonnegative integers')
    # The released loader uses the normal advancing Torch/worker RNG streams.
    # Keep the last, possibly smaller batch in BOTH training and evaluation.
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers,
                      pin_memory=True, drop_last=False, persistent_workers=False)


def check_population(ds, expected_videos=0, *, expected_listed_videos=0, expected_metadata_videos=0):
    """Check data completeness without replacing the measured denominator."""
    if expected_videos is not None and (type(expected_videos) is not int or expected_videos < 0):
        raise ValueError('expected_videos must be a nonnegative integer or None')
    valid = {int(video_index) for video_index, _ in ds.index_map}
    total = len(valid)
    if not total:
        raise ValueError('No test videos have a legal evaluation window')
    if expected_videos and total != expected_videos:
        raise ValueError(
            f'Expected {expected_videos} usable test videos, found {total} '
            f'({len(ds)} windows). Check NTU split, metadata and frame settings. '
            'Use --expected-videos 0 only for an intentionally different population; '
            'the score denominator always remains the actual video count.')
    if expected_videos:
        if expected_listed_videos and len(ds.video_names) != expected_listed_videos:
            raise ValueError('NTU listed-video count differs from the configured protocol')
        if expected_metadata_videos and len(ds.directory_manifest) != expected_metadata_videos:
            raise ValueError('NTU metadata is incomplete; expected all declared videos')
    return dict(videos=len(ds.video_names), usable_videos=total, windows=len(ds),
                unscored=len(ds.video_names) - total)


def add_video(probabilities, indices, totals):
    """Released sequential FP32 probability sum, retaining window order."""
    for probability, index in zip(probabilities, indices):
        index = int(index)
        if index in totals:
            totals[index] += probability
        else:
            totals[index] = probability.copy()


def write_predictions(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model, loader, device, predictions=None):
    """One pass: window softmax -> video sum -> argmax -> video accuracy.

    Local points are drawn by the ordinary dataset/worker RNG. This function
    does not reset, restore or search sampling states. No parameter updates.
    """
    model.eval()
    totals, labels, counts = {}, {}, {}
    clip_correct, clips, loss_sum = 0, 0, 0.0
    num_classes = None
    for inputs, target, indices in loader:
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        target_device = target.to(device, non_blocking=True)
        logits = model(inputs)
        if (logits.dtype != torch.float32 or logits.ndim != 2
                or logits.shape[0] != len(target) or not torch.isfinite(logits).all()):
            raise RuntimeError('Evaluation requires finite FP32 [batch, classes] logits')
        if num_classes is None:
            num_classes = logits.shape[1]
        elif logits.shape[1] != num_classes:
            raise ValueError('Model class count changed during evaluation')
        # Weight batch means by their true sizes, including the final short batch.
        loss_sum += float(F.cross_entropy(logits, target_device)) * len(target)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
        y, ids = target.cpu().numpy(), indices.cpu().numpy()
        clip_correct += int((probabilities.argmax(1) == y).sum())
        clips += len(y)
        add_video(probabilities, ids, totals)
        for label, index in zip(y, ids):
            index, label = int(index), int(label)
            if index in labels and labels[index] != label:
                raise ValueError('Conflicting labels for one video')
            labels[index] = label
            counts[index] = counts.get(index, 0) + 1
    if not totals:
        raise ValueError('Evaluation loader contains no legal windows')
    names = loader.dataset.video_names
    if any(index < 0 or index >= len(names) for index in totals):
        raise ValueError('Video index is outside the dataset name list')
    rows = [dict(video=names[index], label=labels[index], prediction=int(prob.argmax()),
                 correct=int(int(prob.argmax()) == labels[index]), windows=counts[index])
            for index, prob in totals.items()]
    class_counts, class_correct = [0] * num_classes, [0] * num_classes
    for row in rows:
        class_counts[row['label']] += 1
        class_correct[row['label']] += row['correct']
    if predictions is not None:
        write_predictions(predictions, rows)
    correct = sum(row['correct'] for row in rows)
    return dict(correct=correct, total=len(rows), accuracy=correct / len(rows),
                clips=clips, clip_accuracy=clip_correct / clips, loss=loss_sum / clips,
                unscored=len(names) - len(rows), class_counts=class_counts,
                class_correct=class_correct,
                class_accuracy=[good / count if count else None
                                for good, count in zip(class_correct, class_counts)],
                rows=rows)
