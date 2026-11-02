"""Native runtime, dataset ordering, and reversible gradient preflight."""
from __future__ import annotations

import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from . import state as H
from . import loss as S
from .engine import new_optimizer


def configure_runtime():
    """Preserve the numeric settings and initial RNG of the recorded run."""
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    H.require(sys.platform.startswith('linux') and sys.version_info[:2] == (3, 10),
              'Exact training requires Linux and Python 3.10')
    H.require(torch.__version__.split('+')[0] == '2.1.2' and torch.version.cuda == '11.8'
              and np.__version__ == '1.26.4', 'Use Torch 2.1.2+cu118 and NumPy 1.26.4')
    H.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
              'Expose exactly one CUDA GPU')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


def training_datasets(root, full_data, reference):
    """Read the original 566-file stream from the verified 567-file directory.

    The excluded file is a TEST video. It remains in the full fixed evaluation.
    The historical DEV iterator is consumed without inference only to preserve
    the following TRAIN permutation and worker RNG states.
    """
    from tools.official_msr import MSRAction3D
    from tools.stage_data import StageDataset
    from tools.numerics import recorded_order
    from tools.data_runtime import from_raw

    full_data = Path(full_data).resolve()
    names = reference['ordered_filenames']
    manifest = H.read(root / 'data/TRAINING_MANIFEST.json')['files']
    H.require(len(names) == len(set(names)) == 566 and set(names) == set(manifest),
              'Wrong historical 566-file training traversal')
    H.require(set(names).issubset(p.name for p in full_data.glob('*.npz')),
              'The full dataset lacks a historical training/DEV file')
    for name in names:
        H.require(Path(name).name == name and '/' not in name and '\\' not in name,
                  'Unsafe training manifest filename')
        path, expected = full_data / name, manifest[name]
        H.require(path.is_file() and path.stat().st_size == expected['bytes']
                  and H.sha(path) == expected['sha256'], 'Original data differs: ' + name)
    with recorded_order(full_data, names):
        raw_train = MSRAction3D(str(full_data), frames_per_clip=24,
                               step_between_clips=1, num_points=2048, train=True)
        raw_dev = MSRAction3D(str(full_data), frames_per_clip=24,
                             step_between_clips=1, num_points=2048, train=False)
    train = from_raw(raw_train, clip_frames=24, global_frames=32,
                     context_seed=0, local_policy='reference24_span')
    dev = StageDataset(raw_dev, context_seed=0)
    H.require(len(train) == 4478 and len(dev) == 5124,
              'Original TRAIN/DEV traversal changed')
    return train, dev


def consume_dev_rng(loader):
    loader.reset()
    for _ in loader:
        pass
    receipt = loader.receipt()
    H.require(receipt['clips'] == 5124 and receipt['batches'] == 366,
              'Original DEV CPU iterator changed')
    return dict(**receipt, purpose='Historical next-TRAIN RNG compatibility; no DEV inference or score')


def preflight(model, dataset, *, device='cuda'):
    """Discard one parent and two SRR updates, restoring all state and RNG."""
    weights, rng = H.cpu_tree(model.state_dict()), H.capture_rng()
    epoch, mode, alpha = model.phase_epoch, model.training, model.srr_scale
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    gradients = [(p, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    records = []
    try:
        generator = torch.Generator().manual_seed(20261031)
        loader = torch.utils.data.DataLoader(dataset, batch_size=14, shuffle=False,
                                             num_workers=0, generator=generator)
        inputs, labels, _ = next(iter(loader))
        inputs, labels = inputs.to(device), labels.to(device)
        model.set_phase(0)
        model.set_srr_scale(1.)
        model.train()
        optimizer = new_optimizer(model, 0, torch)
        optimizer.zero_grad(set_to_none=True)
        value = torch.nn.functional.cross_entropy(model(inputs), labels)
        value.backward()
        H.require(bool(torch.isfinite(value)) and all(p.grad is not None and
                  bool(torch.isfinite(p.grad).all()) for p in model.parent_parameters()),
                  'Parent preflight gradient failed')
        H.require(all(p.grad is None for p in model.relation_parameters()),
                  'Inactive SRR received parent gradients')
        optimizer.step()
        model.load_state_dict(weights, strict=True)
        model.zero_grad(set_to_none=True)
        H.restore_rng(rng)
        model.set_phase(30)
        model.train()
        optimizer = new_optimizer(model, 30, torch)
        parent_sha = H.state_sha(H.split_state(model))
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = model.training_outputs(inputs)
            value, _ = S.relation_loss(outputs['logits'], outputs['base_logits'], labels)
            value.backward()
            outlet = float(model.prepool_relation.output_projection.weight.grad.norm())
            hidden = float(model.prepool_relation.edge_mlp[0].weight.grad.norm())
            H.require(outlet > 0 and (hidden == 0 if step == 0 else hidden > 0),
                      'Zero-initialized SRR gradient path failed')
            H.require(all(p.grad is None for p in model.parent_parameters()),
                      'Frozen parent received gradients')
            if step == 0:
                H.require(torch.equal(outputs['logits'], outputs['base_logits']),
                          'Fresh zero SRR changed the parent prediction')
            records.append(dict(step=step, output_gradient=outlet, hidden_gradient=hidden))
            optimizer.step()
        H.require(H.state_sha(H.split_state(model)) == parent_sha,
                  'Preflight changed the frozen parent')
    finally:
        model.load_state_dict(weights, strict=True)
        model.zero_grad(set_to_none=True)
        model.set_phase(epoch)
        model.set_srr_scale(alpha)
        model.train(mode)
        for module, old in modes:
            module.training = old
        for parameter, old in flags:
            parameter.requires_grad_(old)
        for parameter, old in gradients:
            parameter.grad = old
        H.restore_rng(rng)
    H.require(H.same_tree(weights, H.cpu_tree(model.state_dict()))
              and H.same_tree(rng, H.capture_rng()), 'Preflight failed to restore state')
    return dict(passed=True, retained_optimizer_steps=0,
                temporary_parent_updates=1, temporary_srr_updates=2,
                restored_all_weights_rng=True, records=records)
