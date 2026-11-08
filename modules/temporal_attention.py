from __future__ import annotations

import copy
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn


class TemporalContractError(ValueError):
    """An unsupported backbone or temporal table configuration was supplied."""


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TemporalContractError(f"{name} must be a positive integer")
    return value


class TemporalAttention(nn.Module):
    """Released Attention with one extra scalar multiplier after softmax.

    Original submodule/state-dict names are preserved. Construction reuses the
    supplied modules; use the model builder's default deep copy for isolation.
    """

    def __init__(self, source: nn.Module, *, max_frames: int = 12) -> None:
        super().__init__()
        self.max_frames = _positive_int(max_frames, "max_frames")
        self.heads = _positive_int(source.heads, "heads")
        expected = {"norm", "to_qkv", "spatial_op", "to_out"}
        if set(source._modules) != expected or list(source.parameters(recurse=False)):
            raise TemporalContractError("expected unmodified released Attention")
        if not isinstance(source.to_qkv, nn.Linear) or source.to_qkv.bias is not None:
            raise TemporalContractError("expected released bias-free QKV projection")
        if source.to_qkv.out_features % (3 * self.heads):
            raise TemporalContractError("QKV width must divide into three sets of heads")
        head_dim = source.to_qkv.out_features // (3 * self.heads)
        if (
            not isinstance(source.spatial_op, nn.Linear)
            or source.spatial_op.in_features != 3
            or source.spatial_op.out_features != head_dim
            or source.spatial_op.bias is not None
        ):
            raise TemporalContractError("expected released bias-free shared spatial_op")
        self.scale = source.scale
        self.norm = source.norm
        self.to_qkv = source.to_qkv
        self.spatial_op = source.spatial_op
        self.to_out = source.to_out
        # Allocate constants directly: do not draw random weights and overwrite.
        anchor = self.to_qkv.weight
        self.w_dt = nn.Parameter(anchor.new_ones(self.heads, 2 * self.max_frames - 1))
        frames = torch.arange(self.max_frames, device=anchor.device)
        self.register_buffer(
            "_lag_index",
            frames[None, :] - frames[:, None] + self.max_frames - 1,
            persistent=False,
        )
        self.training = source.training

    def forward(self, xyzs: Tensor, features: Tensor) -> Tensor:
        if features.ndim != 4 or tuple(xyzs.shape) != (*features.shape[:3], 3):
            raise TemporalContractError("expected features [B,L,N,C] and xyzs [B,L,N,3]")
        b, l, n, _ = features.shape
        if not 1 <= l <= self.max_frames or n < 1:
            raise TemporalContractError(
                f"expected 1 <= Transformer frames <= {self.max_frames} and positive points; "
                f"got L={l}, N={n}"
            )
        norm_features = self.norm(features)
        qkv = self.to_qkv(norm_features).chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, "b l n (h d) -> b h (l n) d", h=self.heads) for t in qkv
        )
        xyzs_flatten = rearrange(xyzs, "b l n d -> b (l n) d")
        delta_xyzs = xyzs_flatten.unsqueeze(1) - xyzs_flatten.unsqueeze(2)
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # Eq. (9). The center is max_frames-1 even for a shorter runtime clip.
        # Do not alter logits, normalize again, or modify softmax output in-place.
        temporal = self.w_dt[:, self._lag_index[:l, :l]].to(dtype=attn.dtype)
        attn = (
            attn.reshape(b, self.heads, l, n, l, n)
            * temporal[None, :, :, None, :, None]
        ).reshape(b, self.heads, l * n, l * n)

        # The SAME modulated weights multiply BOTH terms of the spatial message.
        v = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        delta_xyzs = torch.sum(attn.unsqueeze(4) * delta_xyzs.unsqueeze(1), dim=3)
        displacement_features = self.spatial_op(delta_xyzs)
        out = v + displacement_features
        out = rearrange(out, "b h m d -> b m (h d)")
        out = self.to_out(out)
        out = rearrange(out, "b (l n) d -> b l n d", l=l, n=n)
        return out + features


class TemporalTransformer(nn.Module):
    def __init__(self, source: nn.Module, *, max_frames: int = 12) -> None:
        super().__init__()
        if set(source._modules) != {"layers"} or list(source.parameters(recurse=False)):
            raise TemporalContractError("expected the released backbone Transformer")
        self.layers = nn.ModuleList([
            nn.ModuleList([
                TemporalAttention(attention, max_frames=max_frames), feed_forward
            ])
            for attention, feed_forward in source.layers
        ])
        if not self.layers:
            raise TemporalContractError("Transformer must contain at least one layer")
        self.training = source.training

    def forward(self, xyzs: Tensor, features: Tensor) -> Tensor:
        for attention, feed_forward in self.layers:
            features = feed_forward(attention(xyzs, features))
        return features


def build__temporal_from_official(
    official_model: nn.Module, *, max_frames: int = 12, deepcopy_model: bool = True
) -> nn.Module:
    """Return the released model with ONLY its Transformer attention extended.

    The default deep copy isolates all parameters. With deepcopy_model=False,
    original parameter modules are shared intentionally, but the source model's
    module tree is not changed. Its original forward/pooling/head are retained.
    Construct before creating an optimizer. Load an old Vanilla checkpoint into
    the source model first; new temporal checkpoints require this builder and
    the same max_frames. Strict loading must not hide missing temporal weights.
    """
    _positive_int(max_frames, "max_frames")
    if not hasattr(official_model, "transformer"):
        raise TemporalContractError("source model must expose transformer")
    source = copy.deepcopy(official_model) if deepcopy_model else copy.copy(official_model)
    if not deepcopy_model:
        # nn.Module's shallow copy shares _modules; detach the registry before
        # replacing Transformer so that the source object remains a true Vanilla.
        source._modules = source._modules.copy()
    source.transformer = TemporalTransformer(source.transformer, max_frames=max_frames)
    return source


def specification(*, max_frames: int = 12) -> dict[str, Any]:
    _positive_int(max_frames, "max_frames")
    return {
        "name": "released PST +  Eq9 temporal value modulation",
        "formula": "sum_j softmax(QK)_ij * w[layer,head,t_key-t_query] * (WvF_j+Ws_delta_ij)",
        "max_transformer_frames": max_frames,
        "lag_range": [-(max_frames - 1), max_frames - 1],
        "parameters_per_head_per_layer": 2 * max_frames - 1,
        "initialization": "all ones; explicit implementation choice, not specified in ",
        "constraints": "unconstrained signed scalars, including trainable lag zero",
        "unchanged": ["P4D", "QK logits", "softmax", "GELU", "shared spatial_op", "FFN", "pool", "head"],
        "not_claimed": "full  reproduction or author training initialization",
    }
