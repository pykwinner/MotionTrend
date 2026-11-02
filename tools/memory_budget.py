"""Copied from verified adaptive50 r2 memory_budget.py, 2026-10-31.

Small streaming resource guard; does not allocate buffers or cache points.
"""
from pathlib import Path


def snapshot():
    current=Path('/sys/fs/cgroup/memory.current');limit=Path('/sys/fs/cgroup/memory.max')
    if not current.is_file():
        current=Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
        limit=Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    out={}
    if current.is_file() and limit.is_file():
        raw=limit.read_text().strip();out['ram_cgroup_bytes']=int(current.read_text())
        if raw!='max':out['ram_limit_bytes']=int(raw)
    return out


def check(values):
    if values.get('ram_limit_bytes',0)>0 and values.get('ram_cgroup_bytes',0)>.85*values['ram_limit_bytes']:
        raise MemoryError('RAM guard85pct: stop owned evaluation before exhausting container budget')
    if values.get('gpu_total_bytes',0)>0 and values.get('gpu_reserved_bytes',0)>.90*values['gpu_total_bytes']:
        raise MemoryError('GPU allocator guard90pct: stop owned evaluation before exhausting VRAM')
    return values


def guard(torch=None):
    out=snapshot()
    if torch is not None and torch.cuda.is_available():
        out.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),gpu_reserved_bytes=torch.cuda.memory_reserved(),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    return check(out)
