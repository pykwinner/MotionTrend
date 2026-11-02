"""MSR stage-context metadata beside the bit-exact frozen local sampler.

The local clip, whole-video context selection and augmentation come directly
from the frozen ProcessDataset. MSR uses the same anisotropic augmentation and
division by 300 for both paths; unlike NTU there is no separate clip radius.
One center over valid sampled whole-video context points is subtracted from
context only. The affine metadata applies that same shift to query coordinates
AFTER P4D, leaving the official backbone input completely unchanged.

This shared video centering matches the old global encoder's convention. It is
not per-frame centering, anatomical alignment, point tracking or radius scaling.
"""
from __future__ import annotations

import numpy as np

from .process_data import ProcessDataset, ProcessInput


class StageDataset(ProcessDataset):
    """Add true video times and a shared local/context coordinate transform."""

    def __getitem__(self, idx: int):
        inputs, label, index = super().__getitem__(idx)
        video_index, start = self.index_map[idx]
        length = len(self.videos[video_index])
        frame_ids = (start + np.arange(self.frames_per_clip, dtype=np.int64)
                     * self.step_between_clips)
        if length < 2 or frame_ids.min() < 0 or frame_ids.max() >= length:
            raise ValueError("local timestamps must refer to existing video frames")

        # Accumulate in float64, then use the SAME float32 center for the
        # returned context and local-to-context transform. No random calls.
        context, valid = inputs["context"], inputs["valid"]
        if not bool(valid.any()):
            raise ValueError("each video needs valid whole-video context")
        center = context[valid].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        if not np.isfinite(center).all():
            raise ValueError("whole-video context center must be finite")
        centered = np.zeros_like(context)
        centered[valid] = context[valid] - center
        inputs["context"] = centered
        inputs["local_times"] = (frame_ids / float(length - 1)).astype(np.float32)
        inputs["local_to_context_scale"] = np.asarray(1.0, dtype=np.float32)
        inputs["local_to_context_offset"] = -center
        return inputs, label, index


__all__ = ["StageDataset", "ProcessInput"]
