"""Spatial Relation Refinement (SRR): same-frame local edge readout.

Same-frame geometric neighborhoods provide a zero-initialized residual before
spatial and temporal max pooling. The backbone applies the recorded per-token
energy bound to this module's raw output. The edges do not track points across
frames or assign anatomical part identities.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_relation_residual(raw: Tensor, parent_tokens: Tensor, *, rho: float, epsilon: float):
    """Limit each residual component with unit derivative at the zero exit.

    tau=rho*stopgrad(RMS(U))+epsilon; delta=tau*tanh(raw/tau).
    A stable RMS avoids overflow from squaring large finite parent features.
    The feature budget carries no gradient into the frozen parent.
    """
    require(type(rho) in (float, int) and math.isfinite(rho) and 0 < rho <= 1,
            'rho must be a finite positive number no greater than1')
    require(type(epsilon) in (float, int) and math.isfinite(epsilon) and epsilon > 0,
            'epsilon must be finite and positive')
    require(raw.shape == parent_tokens.shape and raw.ndim == 4,
            'Relation and parent tokens must have identical [B,L,N,C] shapes')
    require(raw.device == parent_tokens.device and raw.dtype == parent_tokens.dtype
            and raw.is_floating_point(), 'Relation and parent tokens must share floating dtype/device')
    require(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(parent_tokens).all()),
            'Relation and parent features must be finite')
    with torch.autocast(device_type=raw.device.type, enabled=False):
        parent = parent_tokens.detach().float()
        maximum = parent.abs().amax(dim=-1, keepdim=True)
        divisor = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
        rms = maximum * (parent / divisor).square().mean(dim=-1, keepdim=True).sqrt()
        tau = (float(rho) * rms + float(epsilon)).detach()
        bounded = tau * torch.tanh(raw.float() / tau)
    require(bool(torch.isfinite(tau).all()) and bool((tau > 0).all())
            and bool(torch.isfinite(bounded).all()), 'Nonfinite Relation trust region')
    return bounded.to(raw.dtype), tau

def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _valid_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError('init_seed must be an integer in [0,2**63)')


def _validate_inputs(tokens: Tensor, xyz: Tensor, dim: int, weight: Tensor) -> None:
    if not isinstance(tokens, Tensor) or not isinstance(xyz, Tensor):
        raise ValueError('tokens and aligned_xyz must be tensors')
    if (tokens.ndim != 4 or tokens.shape[-1] != dim or
            min(tokens.shape[:3]) < 1 or not tokens.is_floating_point()):
        raise ValueError('tokens must be floating [B,L,N,C] with positive sizes')
    if xyz.shape != (*tokens.shape[:3], 3) or not xyz.is_floating_point():
        raise ValueError('aligned_xyz must be floating [B,L,N,3] matching tokens')
    if xyz.device != tokens.device or xyz.dtype != tokens.dtype:
        raise ValueError('tokens and aligned_xyz must share dtype and device')
    if weight.device != tokens.device or weight.dtype != tokens.dtype:
        raise ValueError('module parameters and inputs must share dtype and device')
    if not bool(torch.isfinite(tokens).all()) or not bool(torch.isfinite(xyz).all()):
        raise ValueError('tokens and aligned_xyz must be finite')


def _same_frame_neighbors(xyz: Tensor, requested_k: int):
    """Select by geometry only, including every tie at the kth distance.

    A fixed top-k would arbitrarily discard equally distant neighbors based on
    point ordering. We instead use the kth distance as a radius, keep all exact
    boundary ties, and pad up to the largest resulting neighborhood in this call.
    Neither the neighborhood identity nor diagnostic distances carry gradients;
    the original coordinate values gathered for the edge MLP remain differentiable.
    """
    b, length, n, _ = xyz.shape
    frames = b * length
    with torch.no_grad():
        if n == 1:
            indices = torch.empty((b, length, n, 0), device=xyz.device, dtype=torch.long)
            valid = torch.empty_like(indices, dtype=torch.bool)
            distances = xyz.new_empty((b, length, n, 0))
            counts = torch.zeros((b, length, n), device=xyz.device, dtype=torch.long)
            return indices, valid, distances, counts
        # Compute half/bfloat16 distances in float32; do not let an autocast
        # matrix product introduce index-dependent cancellation near ties.
        work = xyz.detach().reshape(frames, n, 3)
        if work.dtype in (torch.float16, torch.bfloat16):
            work = work.float()
        delta = work[:, :, None, :] - work[:, None, :, :]
        squared = (delta * delta).sum(dim=-1)
        if not bool(torch.isfinite(squared).all()):
            raise ValueError('coordinate range overflows neighborhood distances')
        self_mask = torch.eye(n, device=xyz.device, dtype=torch.bool)[None]
        squared = squared.masked_fill(self_mask, float('inf'))
        k = min(requested_k, n - 1)
        threshold = squared.kthvalue(k, dim=-1, keepdim=True).values
        included = squared <= threshold
        counts = included.sum(dim=-1)
        padded_k = int(counts.max().item())
        selected_squared, indices = squared.topk(padded_k, dim=-1, largest=False,
                                                 sorted=True)
        valid = selected_squared <= threshold
        # Invalid padded indices are safe to gather and are explicitly masked.
        indices = indices.masked_fill(~valid, 0)
        distances = selected_squared.masked_fill(~valid, 0).sqrt().to(xyz.dtype)
        return (indices.reshape(b, length, n, padded_k),
                valid.reshape(b, length, n, padded_k),
                distances.reshape(b, length, n, padded_k),
                counts.reshape(b, length, n))


class SpatialRelationRefinement(nn.Module):
    """Geometry-conditioned local edge readout, with no cross-frame edges.

    Default parameters: LN(80) + MLP(166,32,32) + bias-free 32->80 = 9,120.
    Inputs use the existing augmented whole-video coordinate system. These
    coordinates are not an anatomical body frame and neighbors are not tracks.
    """
    def __init__(self, dim: int = 80, hidden_dim: int = 32, k: int = 8,
                 init_seed: int = 0):
        super().__init__()
        for name, value in (('dim', dim), ('hidden_dim', hidden_dim), ('k', k)):
            _positive_integer(name, value)
        _valid_seed(init_seed)
        self.dim, self.hidden_dim, self.k = dim, hidden_dim, k
        # CPU-only initialization in an isolated RNG scope preserves the exact
        # original-model RNG state and never initializes CUDA implicitly.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(init_seed)
            self.norm = nn.LayerNorm(dim)
            self.edge_mlp = nn.Sequential(nn.Linear(2 * dim + 6, hidden_dim),
                                          nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
                                          nn.GELU())
            self.output_projection = nn.Linear(hidden_dim, dim, bias=False)
            nn.init.zeros_(self.output_projection.weight)

    def forward(self, tokens: Tensor, aligned_xyz: Tensor, return_trace: bool = False):
        _validate_inputs(tokens, aligned_xyz, self.dim, self.output_projection.weight)
        b, length, n, _ = tokens.shape
        indices, valid, distances, counts = _same_frame_neighbors(aligned_xyz, self.k)
        if n == 1:
            # No self edge: even a trained exit cannot modify a singleton frame.
            pooled_edges = tokens.new_zeros((b, length, n, self.hidden_dim))
        else:
            frames, k = b * length, indices.shape[-1]
            normalized = self.norm(tokens).reshape(frames, n, self.dim)
            xyz = aligned_xyz.reshape(frames, n, 3)
            frame_index = torch.arange(frames, device=tokens.device)[:, None, None]
            flat_indices = indices.reshape(frames, n, k)
            neighbor_features = normalized[frame_index, flat_indices]
            neighbor_xyz = xyz[frame_index, flat_indices]
            center_features = normalized[:, :, None, :].expand(-1, -1, k, -1)
            center_xyz = xyz[:, :, None, :].expand(-1, -1, k, -1)
            edges = torch.cat((center_features, neighbor_features - center_features,
                               center_xyz, neighbor_xyz - center_xyz), dim=-1)
            encoded = self.edge_mlp(edges)
            masked = encoded.masked_fill(~valid.reshape(frames, n, k, 1), 0)
            pooled_edges = (masked.sum(dim=2) /
                            counts.reshape(frames, n, 1).to(encoded.dtype))
            pooled_edges = pooled_edges.reshape(b, length, n, self.hidden_dim)
        residual = self.output_projection(pooled_edges)
        if return_trace:
            return residual, {'neighbor_indices': indices, 'neighbor_valid': valid,
                              'neighbor_distances': distances, 'neighbor_counts': counts}
        return residual


SRR = SpatialRelationRefinement
