"""Capture actual pre-evaluation RNG tensors and the installed loader seed recipe."""
from __future__ import annotations
import copy
import hashlib
import inspect
import multiprocessing
from pathlib import Path
import platform
import random
import numpy as np

def require(value,message):
    if not value:raise RuntimeError(message)
def digest_bytes(value):return hashlib.sha256(value).hexdigest()
def file_sha(path):return digest_bytes(Path(path).read_bytes())
def tensor_sha(tensor):return digest_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())

def snapshot_current(torch,include_cuda):
    return dict(python=copy.deepcopy(random.getstate()),numpy=copy.deepcopy(np.random.get_state()),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if include_cuda else None)

def restore_current(state,torch):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None:
        require(torch.cuda.device_count()==len(state['torch_cuda']),'CUDA device count differs from state archive')
        torch.cuda.set_rng_state_all(state['torch_cuda'])

def same_state(left,right,torch):
    return (left['python']==right['python'] and left['numpy'][0]==right['numpy'][0]
        and np.array_equal(left['numpy'][1],right['numpy'][1]) and left['numpy'][2:]==right['numpy'][2:]
        and torch.equal(left['torch_cpu'],right['torch_cpu'])
        and ((left['torch_cuda'] is None and right['torch_cuda'] is None)
             or (left['torch_cuda'] is not None and right['torch_cuda'] is not None
                 and len(left['torch_cuda'])==len(right['torch_cuda'])
                 and all(torch.equal(a,b) for a,b in zip(left['torch_cuda'],right['torch_cuda'])))))

def state_hashes(state):
    h=hashlib.sha256();h.update(repr(state['python']).encode())
    h.update(state['numpy'][0].encode());h.update(state['numpy'][1].tobytes());h.update(repr(state['numpy'][2:]).encode())
    h.update(state['torch_cpu'].numpy().tobytes())
    for s in state['torch_cuda'] or []:h.update(s.numpy().tobytes())
    return dict(combined_sha256=h.hexdigest(),python_state_sha256=digest_bytes(repr(state['python']).encode()),
        numpy_state_sha256=digest_bytes(state['numpy'][0].encode()+state['numpy'][1].tobytes()+repr(state['numpy'][2:]).encode()),
        torch_cpu_state_sha256=tensor_sha(state['torch_cpu']),
        torch_cuda_state_sha256=[tensor_sha(s) for s in state['torch_cuda']] if state['torch_cuda'] is not None else None)

def installed_loader_recipe(torch,seed):
    """Use installed private initialization function, not a reimplemented recipe."""
    from torch.utils.data._utils import worker
    from torch.utils.data import dataloader
    generator=torch.Generator().manual_seed(seed);initial=generator.get_state().clone()
    probe=torch.Generator();probe.set_state(initial)
    base_seed=int(torch.empty((),dtype=torch.int64).random_(generator=probe))
    workers=[]
    for worker_id in range(10):
        worker_seed=base_seed+worker_id
        words=list(map(int,worker._generate_state(base_seed,worker_id)))
        python_state=random.Random(worker_seed).getstate()
        numpy_rng=np.random.RandomState();numpy_rng.seed(words)
        cpu=torch.Generator().manual_seed(worker_seed).get_state().clone()
        workers.append(dict(worker_id=worker_id,python_and_torch_seed=worker_seed,
            numpy_seed_words_uint32=words,python_initial_state=python_state,
            numpy_initial_state=copy.deepcopy(numpy_rng.get_state()),torch_cpu_initial_state=cpu))
    return dict(loader_generator_initial_state=initial,loader_generator_after_base_seed_state=probe.get_state().clone(),
        dataloader_base_seed=base_seed,worker_initial_states=workers,
        recipe=dict(torch_version=torch.__version__,torch_cuda_version=torch.version.cuda,numpy_version=np.__version__,
            python_version=platform.python_version(),multiprocessing_start_method=multiprocessing.get_context().get_start_method(),
            batch_size=14,num_workers=10,shuffle=False,drop_last=False,pin_memory=True,persistent_workers=False,prefetch_factor=2,
            worker_seed_formula='base_seed + worker_id',numpy_seed_function='torch.utils.data._utils.worker._generate_state',
            numpy_seed_function_source=inspect.getsource(worker._generate_state),
            numpy_seed_function_source_sha256=digest_bytes(inspect.getsource(worker._generate_state).encode()),
            worker_loop_source_sha256=digest_bytes(inspect.getsource(worker._worker_loop).encode()),
            worker_module_path=str(Path(worker.__file__).resolve()),worker_module_sha256=file_sha(worker.__file__),
            dataloader_module_path=str(Path(dataloader.__file__).resolve()),dataloader_module_sha256=file_sha(dataloader.__file__)))

def capture_pre_eval(torch,seed,include_cuda,context_seed=0):
    """Caller has just reset global RNGs exactly as shared.isolated_rng(seed)."""
    return dict(schema='msr-sampling-pre-eval-state-v1',evaluation_seed=seed,context_seed=context_seed,
        meaning='Actual states after this evaluation seed reset, immediately before DataLoader iterator construction; not old training checkpoint RNG.',
        global_state=snapshot_current(torch,include_cuda),loader=installed_loader_recipe(torch,seed),cuda_included=bool(include_cuda))

def public_receipt(value):
    loader=value['loader'];workers=[]
    for w in loader['worker_initial_states']:
        workers.append(dict(worker_id=w['worker_id'],python_and_torch_seed=w['python_and_torch_seed'],
            numpy_seed_words_uint32=w['numpy_seed_words_uint32'],
            torch_cpu_initial_state_sha256=tensor_sha(w['torch_cpu_initial_state']),
            python_initial_state_sha256=digest_bytes(repr(w['python_initial_state']).encode()),
            numpy_initial_state_sha256=digest_bytes(w['numpy_initial_state'][0].encode()+w['numpy_initial_state'][1].tobytes()+repr(w['numpy_initial_state'][2:]).encode())))
    return dict(schema=value['schema'],evaluation_seed=value['evaluation_seed'],context_seed=value['context_seed'],meaning=value['meaning'],
        cuda_included=value['cuda_included'],global_state_hashes=state_hashes(value['global_state']),
        loader_generator_initial_state_sha256=tensor_sha(loader['loader_generator_initial_state']),
        loader_generator_after_base_seed_state_sha256=tensor_sha(loader['loader_generator_after_base_seed_state']),
        dataloader_base_seed=loader['dataloader_base_seed'],workers=workers,recipe=loader['recipe'])

def context_initial_states(dataset):
    """Seal the private name-keyed context RNGs without consuming global RNG."""
    method=dataset._raw_context.__func__;code=inspect.getsource(method)
    prefix=b'pst-small-global-process-context-v1\x00';records=[]
    for i in sorted({int(i) for i,_ in dataset.index_map}):
        name=str(dataset.video_names[i])
        payload=prefix+str(dataset.context_seed).encode('ascii')+b'\x00'+name.encode('utf8')
        seed=int.from_bytes(hashlib.sha256(payload).digest()[:4],'little')
        private=np.random.RandomState(seed)
        records.append(dict(name=name,video_index=i,private_seed_uint32=seed,initial_numpy_state=copy.deepcopy(private.get_state())))
    return dict(context_seed=dataset.context_seed,source_method='ProcessDataset._raw_context',source=code,
        source_sha256=digest_bytes(code.encode()),videos=records,
        policy='Each worker recreates this per-video RandomState from the sealed name+contextseed recipe; it never consumes global NumPy state.')
