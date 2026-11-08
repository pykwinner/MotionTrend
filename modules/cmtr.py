"""CMTR: the selected content retrieval module.

Parameter names, initialization and floating operation order are preserved.
"""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn

def time_features(times: Tensor) -> Tensor:
    """Continuous normalized video time, not an array-position index."""
    angles = times[..., None] * times.new_tensor([1., 2., 4., 8.]) * math.pi
    return torch.cat((times[..., None], angles.sin(), angles.cos()), dim=-1)

class TemporalBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        z = self.norm1(x)
        x = x + self.attention(z, z, z, key_padding_mask=~valid, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return torch.where(valid[..., None], x, torch.zeros_like(x))

class CMTR(nn.Module):
    """Different context residual for each local [time, point] token.

    xyz and context must share augmented whole-video coordinates. Coordinates
    condition queries; keys/values are FRAME features, not body-part identities.
    local_times are actual P4D anchor timestamps in the original video.
    """
    def __init__(self, channels: int, dim: int = 64, heads: int = 4,
                 depth: int = 2, seed: int = 0):
        super().__init__()
        if channels < 1 or dim < 1 or heads < 1 or dim % heads or depth < 1:
            raise ValueError('positive widths/depth and dim divisible by heads required')
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in [0,2**63)')
        self.channels, self.dim, self.heads = channels, dim, heads
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.point_encoder = nn.Sequential(nn.Linear(3, 32), nn.GELU(),
                                               nn.Linear(32, dim), nn.GELU())
            self.frame_projection = nn.Linear(2 * dim, dim)
            self.time_projection = nn.Linear(9, dim, bias=False)
            self.temporal_blocks = nn.ModuleList([TemporalBlock(dim, heads) for _ in range(depth)])
            self.memory_norm = nn.LayerNorm(dim)
            self.local_norm = nn.LayerNorm(channels)
            self.query_projection = nn.Linear(channels, dim, bias=False)
            self.xyz_projection = nn.Sequential(nn.Linear(3, dim), nn.GELU(),
                                                 nn.Linear(dim, dim, bias=False))
            self.key_projection = nn.Linear(dim, dim, bias=False)
            self.value_projection = nn.Linear(dim, dim, bias=False)
            self.relative_time = nn.Linear(3, heads, bias=False)
            self.readout_norm = nn.LayerNorm(dim)
            self.output_projection = nn.Linear(dim, channels, bias=False)
            nn.init.zeros_(self.output_projection.weight)
        # A separate private RNG preserves every parent parameter and the
        # caller's RNG. These parameters are new, not loaded parent weights.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed((seed + 73171) % (2**63))
            self.content_norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.content_query_projection = nn.Linear(channels, dim, bias=False)
            self.content_key_projection = nn.Linear(dim, dim, bias=False)
            self.content_value_projection = nn.Linear(dim, dim, bias=False)
            self.content_relative_time = nn.Linear(3, heads, bias=False)
            self.content_output_projection = nn.Linear(dim, channels, bias=False)
            # One zero exit, no zero gate in series with it.
            nn.init.zeros_(self.content_output_projection.weight)

    def _content_read(self, tokens, local_times, frames, reference_frames,
                      safe_times, valid):
        """Read actual frame evidence without sending a time template as V.

        The reference is evaluated with the current shared encoder and MUST NOT
        be detached. It is zero geometry, not the video's temporal mean, so a
        static but nonzero pose is not deliberately removed. Reference and
        actual frames have identical shapes/operators for exact zero parity.
        """
        b, l, n, _ = tokens.shape
        m, d = frames.shape[1], self.dim // self.heads
        raw = torch.where(valid[..., None], frames - reference_frames,
                          torch.zeros_like(frames))
        content = self.content_norm(raw)
        content = torch.where(valid[..., None], content, torch.zeros_like(content))
        query = self.content_query_projection(self.local_norm(tokens))
        keys = self.content_key_projection(content)
        values = self.content_value_projection(content)
        q = query.reshape(b, l*n, self.heads, d).transpose(1, 2)
        k = keys.reshape(b, m, self.heads, d).transpose(1, 2)
        v = values.reshape(b, m, self.heads, d).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) * (d ** -.5)
        delta = safe_times[:, None, :] - local_times[:, :, None]
        bias = self.content_relative_time(torch.stack((delta, delta.abs(), delta.square()), dim=-1))
        bias = bias.permute(0, 3, 1, 2).unsqueeze(3).expand(b, self.heads, l, n, m)
        logits = logits + bias.reshape(b, self.heads, l*n, m)
        attention = logits.masked_fill(~valid[:, None, None, :], float('-inf')).softmax(-1)
        readout = (attention @ v).transpose(1, 2).reshape(b, l, n, self.dim)
        # No affine normalization, query-conditioned gate, or time residual here.
        residual = self.content_output_projection(readout)
        return residual, dict(content_raw_frames=raw, content_frames=content,
                              content_reference_frame=reference_frames,
                              content_values=values, content_attention=attention,
                              content_readout=readout, content_residual=residual)

    def _validate(self, tokens, xyz, local_times, context, times, valid):
        if (tokens.ndim != 4 or tokens.shape[-1] != self.channels or
                min(tokens.shape[:3]) < 1 or not tokens.is_floating_point()):
            raise ValueError('tokens must be floating [B,L,N,C] with positive sizes')
        b, l, n, _ = tokens.shape
        if xyz.shape != (b, l, n, 3) or local_times.shape != (b, l):
            raise ValueError('xyz/local_times must match local token axes')
        if (context.ndim != 4 or context.shape[0] != b or context.shape[-1] != 3
                or not 1 <= context.shape[1] <= 32 or context.shape[2] < 1):
            raise ValueError('context must be [B,M,P,3], 1<=M<=32 and P>=1')
        if (times.shape != context.shape[:2] or valid.shape != times.shape
                or valid.dtype != torch.bool or valid.device != tokens.device):
            raise ValueError('times/Boolean valid must match [B,M] on the input device')
        if any(x.dtype != tokens.dtype or x.device != tokens.device
               for x in (xyz, local_times, context, times)):
            raise ValueError('all floating inputs must share dtype and device')
        if not bool(valid.any(dim=1).all()):
            raise ValueError('each video needs at least one valid context frame')
        if not all(bool(torch.isfinite(x).all()) for x in
                   (tokens, xyz, local_times, context[valid], times[valid])):
            raise ValueError('valid geometry and times must be finite')
        if bool(((local_times < 0) | (local_times > 1)).any()) or bool(
                ((times[valid] < 0) | (times[valid] > 1)).any()):
            raise ValueError('times must lie in [0,1]')
        ordered = torch.where(valid, times, torch.full_like(times, float('inf'))).sort(dim=1).values
        if bool(((ordered[:, 1:] == ordered[:, :-1]) & torch.isfinite(ordered[:, 1:])).any()):
            raise ValueError('valid frame timestamps must be unique')

    def forward(self, tokens: Tensor, xyz: Tensor, local_times: Tensor,
                context: Tensor, times: Tensor, valid: Tensor,
                return_trace: bool = False):
        self._validate(tokens, xyz, local_times, context, times, valid)
        b, l, n, _ = tokens.shape
        m = context.shape[1]
        # Clear invalid NaNs BEFORE arithmetic, not by multiplying with zero.
        safe_context = torch.where(valid[..., None, None], context, torch.zeros_like(context))
        safe_times = torch.where(valid, times, torch.zeros_like(times))
        points = self.point_encoder(safe_context)
        frames = self.frame_projection(torch.cat((points.mean(dim=2), points.max(dim=2).values), dim=-1))
        memory = frames + self.time_projection(time_features(safe_times))
        memory = torch.where(valid[..., None], memory, torch.zeros_like(memory))
        for block in self.temporal_blocks:
            memory = block(memory, valid)
        memory = torch.where(valid[..., None], self.memory_norm(memory), torch.zeros_like(memory))
        query = (self.query_projection(self.local_norm(tokens)) + self.xyz_projection(xyz)
                 + self.time_projection(time_features(local_times))[:, :, None, :])
        d = self.dim // self.heads
        q = query.reshape(b, l * n, self.heads, d).transpose(1, 2)
        k = self.key_projection(memory).reshape(b, m, self.heads, d).transpose(1, 2)
        v = self.value_projection(memory).reshape(b, m, self.heads, d).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) * (d ** -.5)
        delta = safe_times[:, None, :] - local_times[:, :, None]
        bias = self.relative_time(torch.stack((delta, delta.abs(), delta.square()), dim=-1))
        bias = bias.permute(0, 3, 1, 2).unsqueeze(3).expand(b, self.heads, l, n, m)
        logits = logits + bias.reshape(b, self.heads, l * n, m)
        attention = logits.masked_fill(~valid[:, None, None, :], float('-inf')).softmax(dim=-1)
        readout = (attention @ v).transpose(1, 2).reshape(b, l, n, self.dim)
        readout = self.readout_norm(readout)
        legacy_residual = self.output_projection(readout)
        # Same live encoder, same point count and arithmetic. Keeping gradients
        # through both terms prevents the null reference from becoming a hidden
        # learned content bias. This never modifies the original input tensor.
        zero_points = self.point_encoder(torch.zeros_like(safe_context))
        reference_frames = self.frame_projection(torch.cat(
            (zero_points.mean(dim=2), zero_points.max(dim=2).values), dim=-1))
        content_residual, content_trace = self._content_read(
            tokens, local_times, frames, reference_frames, safe_times, valid)
        residual = legacy_residual + content_residual
        if return_trace:
            return residual, dict(memory=memory, attention=attention, readout=readout,
                                  local_times=local_times, context_times=safe_times,
                                  context_valid=valid, query_xyz=xyz,
                                  legacy_residual=legacy_residual, **content_trace)
        return residual

