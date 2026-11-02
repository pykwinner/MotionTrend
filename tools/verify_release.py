"""CPU-only file, provenance, and sequential-FP32 prediction verification."""
from pathlib import Path
import json

from tools.numerics import (INPUT_SHA, OUTPUT_SHA, CHECKPOINT_SHA, RNG_SHA,
                            read, require, sha, verify_reference)


def verify_manifest(root):
    root = Path(root).resolve()
    entries = read(root / 'SHA256.json')['files_sha256']
    require(isinstance(entries, dict) and entries, 'Missing release manifest')
    for name, expected in entries.items():
        relative = Path(name)
        require(isinstance(name, str) and name and not relative.is_absolute()
                and '..' not in relative.parts and ':' not in name and '\\' not in name,
                'Unsafe release member')
        path = root / relative
        require(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink(),
                'Missing or unsafe release member: ' + name)
        require(sha(path) == expected, 'Release member changed: ' + name)
    return len(entries)


def verify_release(root, *, check_manifest=True):
    root = Path(root).resolve()
    count = verify_manifest(root) if check_manifest else None
    selected = read(root / 'config/selected.json')
    for name, digest in (
        ('model/checkpoints/best.pth', CHECKPOINT_SHA),
        ('model/checkpoints/initialization.pth', selected['initialization_sha256']),
        ('results/selected/RNG_STATE.pth', RNG_SHA),
    ):
        require(sha(root / name) == digest, 'Changed model or RNG archive: ' + name)
    require(selected['checkpoint_epoch'] == 30 and selected['parameters'] == 408682
            and selected['alpha'] == .5 and selected['state_tensors'] == 132,
            'Selected model configuration differs')
    probabilities = verify_reference(root)
    rows = read(root / 'results/training/metrics.json')
    require(len(rows) == 50 and [r['epoch'] for r in rows] == list(range(50)),
            'Expected all 50 recorded epochs')
    for epoch, row in enumerate(rows):
        relation = epoch >= 30
        require(row['phase'] == ('relation' if relation else 'parent')
                and row['input_sha256'] == INPUT_SHA and row['evaluation_alpha'] == .5,
                'Training/evaluation phase or input identity changed')
        require(row['parent_optimizer_steps'] == min(epoch + 1, 30) * 320
                and row['relation_optimizer_steps'] == max(0, epoch + 1 - 30) * 320,
                'Recorded update budget differs')
        require(row['evaluation']['total'] == 287, 'Wrong evaluation denominator')
        if relation:
            require(row['eligible_full'] is True and row['train']['relation_nonzero_exit'] is True
                    and row['train']['frozen_parent_byte_exact'] is True,
                    'Expected a trained SRR and an unchanged parent')
    best = max(rows[30:], key=lambda row: row['evaluation']['correct'])
    require(best['epoch'] == 30 and best['evaluation']['correct'] == 283
            and rows[-1]['evaluation']['correct'] == 282,
            'Best/final results no longer agree with the selected release')
    require(best['checkpoint_sha256'] == selected['original_checkpoint_sha256'],
            'Public checkpoint is not linked to the selected original checkpoint')
    training = read(root / 'results/training/COMPLETED.json')
    require(training['from_scratch'] is True and training['total_epochs'] == 50
            and training['parent_epochs'] == 30 and training['srr_epochs'] == 20
            and training['total_optimizer_steps'] == 16000,
            'Recorded training run was not the complete 30+20 run')
    return dict(passed=True, sealed_files=count, correct=283, total=287,
                accuracy_percent=100*283/287, parameters=408682, checkpoint_epoch=30,
                recorded_total_epochs=50, recorded_final_correct=282,
                input_sha256=INPUT_SHA, output_sha256=OUTPUT_SHA,
                cpu_probability_verification=probabilities, gpu_replay_executed=False)
