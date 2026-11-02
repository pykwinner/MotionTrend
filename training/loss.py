"""TRAIN-only preservation loss for the frozen parent classifier."""
from __future__ import annotations
import math
from typing import Any

def relation_loss(logits: Any, parent_logits: Any, labels: Any, *, kl_weight: float = .25):
    """TRAIN-only CE plus KL on samples the frozen parent predicts correctly.

    Parent logits are detached. Only the student's logits receive gradients.
    KL is averaged over the masked correct-parent subset, not over TEST cases.
    No teacher-correct samples gives a differentiable zero keep term.
    """
    import torch
    import torch.nn.functional as F

    if (type(kl_weight) not in (int, float) or not math.isfinite(kl_weight)
            or kl_weight < 0):
        raise ValueError('kl_weight must be a finite nonnegative real')
    if (not isinstance(logits, torch.Tensor) or not isinstance(parent_logits, torch.Tensor)
            or logits.ndim != 2 or logits.shape != parent_logits.shape or logits.shape[0] < 1
            or logits.shape[1] != 20 or labels.shape != logits.shape[:1]):
        raise ValueError('Expected matching [B,20] logits and [B] training labels')
    if (logits.device != parent_logits.device or labels.device != logits.device
            or logits.dtype != parent_logits.dtype or not logits.is_floating_point()
            or labels.dtype != torch.long or not bool(torch.isfinite(logits).all())
            or not bool(torch.isfinite(parent_logits).all())):
        raise ValueError('Loss inputs must be finite, colocated floating logits and int64 labels')
    teacher = parent_logits.detach()
    mask = teacher.argmax(dim=-1).eq(labels)
    ce = F.cross_entropy(logits, labels)
    per_sample = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(teacher, dim=-1),
                         reduction='none').sum(dim=-1)
    keep = per_sample[mask].mean() if bool(mask.any()) else logits.sum() * 0.0
    total = ce + kl_weight * keep
    metrics = dict(ce=float(ce.detach()), keep_kl=float(keep.detach()),
                   loss=float(total.detach()), keep_samples=int(mask.sum()),
                   batch_samples=labels.numel(), kl_weight=float(kl_weight))
    return total, metrics
