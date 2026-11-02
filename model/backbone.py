"""MoTrend backbone. EBTE, CTR and SRR are defined in separate files.

Historical tensor keys stay unchanged so the selected trained checkpoint loads
strictly. PST paper Eq. (9) belongs to the inherited backbone. The release uses
30 parent epochs followed by 20 SRR-only epochs; the training entry point
selects the parent checkpoint and manages the two optimizer clocks.
"""

from __future__ import annotations
from typing import List, Mapping
import math
import torch
from torch import Tensor, nn, einsum
from einops import rearrange
from config.switches import ModuleSwitches, SELECTED_SWITCHES
from .ebte import EnergyBoundedTrendEncoding
from .ctr import ContextAndTrendRetrieval
from .srr import SpatialRelationRefinement, bounded_relation_residual

class P4DConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 spatial_kernel_size: [float, int],
                 spatial_stride: int,
                 temporal_kernel_size: int,
                 temporal_stride: int = 1,
                 temporal_padding: [int, int] = [0, 0],
                 temporal_padding_mode: str = 'replicate',
                 operator: str = 'addition',
                 spatial_pooling: str = 'max',
                 temporal_pooling: str = 'sum',
                 bias: bool = False, pointnet2_ops=None):

        super().__init__()

        self.pointnet2_ops = pointnet2_ops
        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_activation = mlp_activation

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.temporal_padding_mode = temporal_padding_mode

        self.operator = operator
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

        conv_d = [nn.Conv2d(in_channels=4, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
        if mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
        if mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
        self.conv_d = nn.Sequential(*conv_d)

        if in_planes != 0:
            conv_f = [nn.Conv2d(in_channels=in_planes, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
            if mlp_batch_norm[0]:
                conv_f.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
            if mlp_activation[0]:
                conv_f.append(nn.ReLU(inplace=True))
            self.conv_f = nn.Sequential(*conv_f)

        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)


    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            xyzs: torch.Tensor
                 (B, T, N, 3) tensor of sequence of the xyz coordinates
            features: torch.Tensor
                 (B, T, C, N) tensor of sequence of the features
        """
        device = xyzs.device
        if self.pointnet2_ops is None:
            from .point_ops import bind
            self.pointnet2_ops = bind()

        nframes = xyzs.size(1)
        npoints = xyzs.size(2)

        assert (self.temporal_kernel_size % 2 == 1), "P4DConv: Temporal kernel size should be odd!"
        assert ((nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "P4DConv: Temporal length error!"

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        if self.temporal_padding_mode == 'zeros':
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]
        else:
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]

        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

            if self.temporal_padding_mode == 'zeros':
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
            else:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        new_xyzs = []
        new_features = []
        for t in range(self.temporal_kernel_size//2, len(xyzs)-self.temporal_kernel_size//2, self.temporal_stride):                 # temporal anchor frames
            # spatial anchor point subsampling by FPS
            anchor_idx = self.pointnet2_ops.furthest_point_sample(xyzs[t], npoints//self.spatial_stride)                               # (B, N//self.spatial_stride)
            anchor_xyz_flipped = self.pointnet2_ops.gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                 # (B, 3, N//self.spatial_stride)
            anchor_xyz_expanded = torch.unsqueeze(anchor_xyz_flipped, 3)                                                            # (B, 3, N//spatial_stride, 1)
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()                                                            # (B, N//spatial_stride, 3)

            new_feature = []
            for i in range(t-self.temporal_kernel_size//2, t+self.temporal_kernel_size//2+1):
                neighbor_xyz = xyzs[i]

                idx = self.pointnet2_ops.ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)

                neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()                                                    # (B, 3, N)
                neighbor_xyz_grouped = self.pointnet2_ops.grouping_operation(neighbor_xyz_flipped, idx)                                # (B, 3, N//spatial_stride, k)

                xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded                                                       # (B, 3, N//spatial_stride, k)
                t_displacement = torch.ones((xyz_displacement.size()[0], 1, xyz_displacement.size()[2], xyz_displacement.size()[3]), dtype=torch.float32, device=device) * (i-t)
                displacement = torch.cat(tensors=(xyz_displacement, t_displacement), dim=1, out=None)                               # (B, 4, N//spatial_stride, k)
                displacement = self.conv_d(displacement)

                if self.in_planes != 0:
                    neighbor_feature_grouped = self.pointnet2_ops.grouping_operation(features[i], idx)                                 # (B, in_planes, N//spatial_stride, k)
                    feature = self.conv_f(neighbor_feature_grouped)
                    if self.operator == '+':
                        feature = feature + displacement
                    else:
                        feature = feature * displacement
                else:
                    feature = displacement

                feature = self.mlp(feature)
                if self.spatial_pooling == 'max':
                    feature = torch.max(input=feature, dim=-1, keepdim=False)[0]                                                        # (B, out_planes, n)
                elif self.spatial_pooling == 'sum':
                    feature = torch.sum(input=feature, dim=-1, keepdim=False)
                else:
                    feature = torch.mean(input=feature, dim=-1, keepdim=False)

                new_feature.append(feature)
            new_feature = torch.stack(tensors=new_feature, dim=1)
            if self.temporal_pooling == 'max':
                new_feature = torch.max(input=new_feature, dim=1, keepdim=False)[0]
            elif self.temporal_pooling == 'sum':
                new_feature = torch.sum(input=new_feature, dim=1, keepdim=False)
            else:
                new_feature = torch.mean(input=new_feature, dim=1, keepdim=False)
            new_xyzs.append(anchor_xyz)
            new_features.append(new_feature)

        new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x) + x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.spatial_op = nn.Linear(3, dim_head, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, xyzs, features):
        b, l, n, _, h = *features.shape, self.heads

        norm_features = self.norm(features)
        qkv = self.to_qkv(norm_features).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b l n (h d) -> b h (l n) d', h = h), qkv)                             # [b, h, m, d]

        xyzs_flatten = rearrange(xyzs, 'b l n d -> b (l n) d')                                                      # [b, m, 3]

        delta_xyzs = torch.unsqueeze(input=xyzs_flatten, dim=1) - torch.unsqueeze(input=xyzs_flatten, dim=2)        # [b, m, m, 3]

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale                                             # [b, h, m, m]
        attn = dots.softmax(dim=-1)

        v = einsum('b h i j, b h j d -> b h i d', attn, v)                                                          # [b, h, m, d]

        attn = torch.unsqueeze(input=attn, dim=4)                                                                   # [b, h, m, m, 1]
        delta_xyzs = torch.unsqueeze(input=delta_xyzs, dim=1)                                                       # [b, 1, m, m, 3]
        delta_xyzs = torch.sum(input=attn*delta_xyzs, dim=3, keepdim=False)                                         # [b, h, m, 3]

        displacement_features = self.spatial_op(delta_xyzs)                                                         # [b, h, m, d]

        out = v + displacement_features
        out = rearrange(out, 'b h m d -> b m (h d)')
        out =  self.to_out(out)
        out = rearrange(out, 'b (l n) d -> b l n d', l=l, n=n)
        return out + features

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, xyzs, features):
        for attn, ff in self.layers:
            features = attn(xyzs, features)
            features = ff(features)
        return features

class PaperTemporalContractError(ValueError):
    """An unsupported backbone or temporal table configuration was supplied."""

def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PaperTemporalContractError(f"{name} must be a positive integer")
    return value

class PaperTemporalAttention(nn.Module):
    """Released Attention with one extra scalar multiplier after softmax.

    Original submodule/state-dict names are preserved. Construction reuses the
    supplied modules. A new backbone is constructed for every model instance.
    """

    def __init__(self, source: nn.Module, *, max_frames: int = 12) -> None:
        super().__init__()
        self.max_frames = _positive_int(max_frames, "max_frames")
        self.heads = _positive_int(source.heads, "heads")
        expected = {"norm", "to_qkv", "spatial_op", "to_out"}
        if set(source._modules) != expected or list(source.parameters(recurse=False)):
            raise PaperTemporalContractError("expected unmodified released Attention")
        if not isinstance(source.to_qkv, nn.Linear) or source.to_qkv.bias is not None:
            raise PaperTemporalContractError("expected released bias-free QKV projection")
        if source.to_qkv.out_features % (3 * self.heads):
            raise PaperTemporalContractError("QKV width must divide into three sets of heads")
        head_dim = source.to_qkv.out_features // (3 * self.heads)
        if (
            not isinstance(source.spatial_op, nn.Linear)
            or source.spatial_op.in_features != 3
            or source.spatial_op.out_features != head_dim
            or source.spatial_op.bias is not None
        ):
            raise PaperTemporalContractError("expected released bias-free shared spatial_op")
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
            raise PaperTemporalContractError("expected features [B,L,N,C] and xyzs [B,L,N,3]")
        b, l, n, _ = features.shape
        if not 1 <= l <= self.max_frames or n < 1:
            raise PaperTemporalContractError(
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

class PaperTemporalTransformer(nn.Module):
    def __init__(self, source: nn.Module, *, max_frames: int = 12) -> None:
        super().__init__()
        if set(source._modules) != {"layers"} or list(source.parameters(recurse=False)):
            raise PaperTemporalContractError("expected released Transformer, without adapters")
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PaperTemporalAttention(attention, max_frames=max_frames), feed_forward
            ])
            for attention, feed_forward in source.layers
        ])
        if not self.layers:
            raise PaperTemporalContractError("Transformer must contain at least one layer")
        self.training = source.training

    def forward(self, xyzs: Tensor, features: Tensor) -> Tensor:
        for attention, feed_forward in self.layers:
            features = feed_forward(attention(xyzs, features))
        return features

class _PSTBackbone(nn.Module):
    """The released 276460-parameter MSR-small backbone and one classifier."""
    def __init__(self, *, pointnet2_ops=None):
        super().__init__()
        self.tube_embedding = P4DConv(
            in_planes=0, mlp_planes=[80], mlp_batch_norm=[False], mlp_activation=[False],
            spatial_kernel_size=[0.3, 32], spatial_stride=32,
            temporal_kernel_size=3, temporal_stride=2, temporal_padding=[1, 0],
            operator='+', spatial_pooling='max', temporal_pooling='max', pointnet2_ops=pointnet2_ops)
        self.transformer = Transformer(80, 5, 2, 40, 160, dropout=0.)
        self.mlp_head = nn.Sequential(nn.LayerNorm(80), nn.Linear(80, 160),
                                      nn.GELU(), nn.Dropout(0.), nn.Linear(160, 20))

    def forward(self, clips):
        xyz, features = self.tube_embedding(clips)
        tokens = self.transformer(xyz, features.permute(0, 1, 3, 2))
        return self.mlp_head(tokens.max(dim=1)[0].max(dim=1)[0])

PARENT_MODULES = ('tube_embedding', 'transformer', 'mlp_head', 'stage_fusion')
PARENT_PARAMETERS, PARENT_TENSORS = 399562, 125
RELATION_PARAMETERS, RELATION_TENSORS = 9120, 7
PARAMETERS, TENSORS = 408682, 132
TOTAL_EPOCHS, SWITCH_EPOCH = 50, 30
FRESH_INITIALIZATION_SHA = '6febe65dfd2bb23c04035af5e2b56f3385865cbd3844dbafe689be4e0e9cf056'
QR_KEYS = {'tube_embedding.adapter.W_c.weight', 'tube_embedding.adapter.W_o.weight'}

def require(condition, message):
    if not condition:
        raise ValueError(message)

class MoTrend(nn.Module):
    """MoTrend: EBTE -> inherited PST backbone -> CTR -> SRR -> one classifier.

    rho/epsilon/phase are non-tensor configuration, so the historical parent125
    keys plus the new Relation7 keys remain exactly132 state tensors. Every
    checkpoint must save ``architecture_config()`` and ``phase_receipt()``;
    loading only a state_dict is intentionally insufficient for resuming a run.
    """
    def __init__(self, *, seed=0, pointnet2_ops=None, switches=None, rho=.02,
                 epsilon=1e-6, switch_epoch=SWITCH_EPOCH, srr_scale=.5):
        super().__init__()
        self.switches = SELECTED_SWITCHES if switches is None else switches
        require(isinstance(self.switches, ModuleSwitches), 'Use config.switches.ModuleSwitches')
        require(type(seed) is int and 0 <= seed < 2**63, 'seed must be an integer in [0,2**63)')
        require(type(switch_epoch) is int and 1 <= switch_epoch < TOTAL_EPOCHS, 'switch_epoch must be in1..49')
        require(type(rho) in (float,int) and math.isfinite(rho) and 0 < rho <= 1, 'Invalid SRR rho')
        require(type(epsilon) in (float,int) and math.isfinite(epsilon) and epsilon > 0, 'Invalid SRR epsilon')
        self.seed, self.rho, self.epsilon = seed, float(rho), float(epsilon)
        self.switch_epoch, self.phase_epoch = switch_epoch, 0
        self.initialization_role = 'constructor_fresh_unsealed'
        self.set_srr_scale(srr_scale)
        if pointnet2_ops is None:
            from .point_ops import bind
            pointnet2_ops = bind()
        base = _PSTBackbone(pointnet2_ops=pointnet2_ops)
        self.tube_embedding = (EnergyBoundedTrendEncoding(base.tube_embedding,
            backbone_seed=seed, pointnet2_ops=pointnet2_ops, strict=True)
            if self.switches.ebte else base.tube_embedding)
        self.transformer = PaperTemporalTransformer(base.transformer, max_frames=12)
        self.mlp_head = base.mlp_head
        if self.switches.ctr:
            self.stage_fusion = ContextAndTrendRetrieval(80, seed=seed)
        if self.switches.srr:
            before = torch.get_rng_state().clone()
            self.prepool_relation = SpatialRelationRefinement(dim=80, hidden_dim=32, k=8, init_seed=seed)
            require(torch.equal(before, torch.get_rng_state()), 'SRR construction changed the parent CPU stream')
        if self.switches.is_selected_architecture:
            require(len(self.state_dict()) == TENSORS and sum(p.numel() for p in self.parameters()) == PARAMETERS,
                    'Selected MoTrend must have408682 parameters and132 state tensors')
        self.set_phase(0)

    def set_srr_scale(self, value):
        require(type(value) in (float,int) and math.isfinite(value) and 0 < value <= 1,
                'SRR scale must be a finite positive number no greater than1')
        self.srr_scale = float(value)
        return self

    @property
    def relation_active(self):
        return self.switches.srr and self.phase_epoch >= self.switch_epoch

    def parent_parameters(self):
        for name in PARENT_MODULES:
            if hasattr(self, name):
                yield from getattr(self, name).parameters()

    def relation_parameters(self):
        if hasattr(self, 'prepool_relation'):
            yield from self.prepool_relation.parameters()

    def phase_parameters(self):
        yield from (self.relation_parameters() if self.relation_active else self.parent_parameters())

    def set_phase(self, epoch):
        require(type(epoch) is int and 0 <= epoch < TOTAL_EPOCHS, 'epoch must be an integer in0..49')
        self.phase_epoch = epoch
        for p in self.parent_parameters():
            p.requires_grad_(not self.relation_active)
            if self.relation_active:
                p.grad = None
        for p in self.relation_parameters():
            p.requires_grad_(self.relation_active)
            if not self.relation_active:
                p.grad = None
        self.train(self.training)
        return self.phase_receipt()

    def train(self, mode=True):
        require(type(mode) is bool, 'training mode must be a bool')
        super().train(mode)
        for name in PARENT_MODULES:
            if hasattr(self, name):
                getattr(self, name).train(mode and not self.relation_active)
        if hasattr(self, 'prepool_relation'):
            self.prepool_relation.train(mode and self.relation_active)
        return self

    def architecture_config(self):
        return dict(model='MoTrend', schema='motrend-model-v1', seed=self.seed,
                    switches=self.switches.as_dict(), rho=self.rho, epsilon=self.epsilon,
                    srr_scale=self.srr_scale, switch_epoch=self.switch_epoch, total_epochs=TOTAL_EPOCHS,
                    parameters=sum(p.numel() for p in self.parameters()), state_tensors=len(self.state_dict()),
                    training_policy=('parent_only' if not self.switches.srr else
                                     'parent_best30_then_relation20' if self.switch_epoch == 30 else
                                     'staged_parent_then_relation'),
                    parent_epochs=self.switch_epoch if self.switches.srr else TOTAL_EPOCHS,
                    relation_epochs=TOTAL_EPOCHS-self.switch_epoch if self.switches.srr else 0,
                    joint_training=False, parent_freeze=self.switches.srr, content_read=self.switches.ctr,
                    training_srr_scale=1.0, inference_srr_scale=0.5,
                    scale_position='raw SRR before tanh',
                    innovation_modules=['Energy-Bounded Trend Encoding (EBTE)',
                                        'Context and Trend Retrieval (CTR)',
                                        'Spatial Relation Refinement (SRR)'],
                    existing_backbone_methods=['PST-small','PST paper Eq9'], classification_heads=1,
                    ensemble=False)

    def phase_receipt(self):
        active = self.relation_active
        params = list(self.relation_parameters() if active else self.parent_parameters())
        return dict(epoch=self.phase_epoch, phase='relation_only' if active else 'parent_only',
                    relation_active=active, relation_bypassed=not active,
                    parent_trainable=not active, relation_trainable=active,
                    active_parameters=sum(p.numel() for p in params), active_tensors=len(params),
                    initialization_role=self.initialization_role)

    def _features(self, inputs, *, trace):
        require(isinstance(inputs, Mapping) and 'local' in inputs, 'MoTrend requires an input mapping with local clips')
        if self.switches.ctr:
            required = ('context','times','valid','local_times','local_to_context_scale','local_to_context_offset')
            require(all(key in inputs for key in required), 'Missing CTR alignment metadata')
        elif self.switches.srr:
            require(all(key in inputs for key in ('local_to_context_scale','local_to_context_offset')),
                    'Missing SRR coordinate alignment metadata')
        xyz, features = self.tube_embedding(inputs['local'])
        tokens = self.transformer(xyz, features.permute(0, 1, 3, 2))
        aligned_xyz = xyz
        if self.switches.ctr or self.switches.srr:
            batch = tokens.shape[0]
            scale, offset = inputs['local_to_context_scale'], inputs['local_to_context_offset']
            require(scale.shape == (batch,) and offset.shape == (batch,3) and bool((scale > 0).all()),
                    'Invalid local-to-context affine metadata')
            aligned_xyz = xyz * scale[:, None, None, None] + offset[:, None, None, :]
        details = {} if trace else None
        if self.switches.ctr:
            anchor_times = inputs['local_times'][:, ::2]
            require(anchor_times.shape == tokens.shape[:2], 'P4D timestamps do not match token count')
            result = self.stage_fusion(tokens, aligned_xyz, anchor_times, inputs['context'],
                                      inputs['times'], inputs['valid'], return_trace=trace)
            stage, details = result if trace else (result, None)
            parent_tokens = tokens + stage
        else:
            stage = torch.zeros_like(tokens) if trace else None
            parent_tokens = tokens
        if trace:
            details.update(local_tokens=tokens, stage_residual=stage,
                           parent_tokens=parent_tokens, query_xyz=aligned_xyz)
        return parent_tokens, aligned_xyz, details

    def _forward(self, inputs, *, trace=False, teacher=False):
        if self.relation_active:
            # Match the deployed frozen parent's eval arithmetic and do not
            # create an autograd graph through its feature path.
            with torch.no_grad():
                parent_tokens, xyz, details = self._features(inputs, trace=trace)
            raw_result = self.prepool_relation(parent_tokens, xyz, return_trace=trace)
            raw_unscaled, relation_trace = raw_result if trace else (raw_result, None)
            raw = raw_unscaled * self.srr_scale
            delta, tau = bounded_relation_residual(raw, parent_tokens, rho=self.rho, epsilon=self.epsilon)
            fused = parent_tokens + delta
            # Head weights are frozen, but this operation must retain the
            # gradient from the loss to delta/Relation.
            logits = self.mlp_head(fused.max(dim=1)[0].max(dim=1)[0])
            base_logits = None
            if teacher:
                with torch.no_grad():
                    base_logits = self.mlp_head(parent_tokens.max(dim=1)[0].max(dim=1)[0])
        else:
            parent_tokens, xyz, details = self._features(inputs, trace=trace)
            raw = delta = tau = relation_trace = None
            logits = self.mlp_head(parent_tokens.max(dim=1)[0].max(dim=1)[0])
            base_logits = logits.detach() if teacher else None
        if trace:
            details.update(relation_raw=raw, relation_residual=delta, relation_tau=tau,
                           relation_trace=relation_trace, relation_active=self.relation_active,
                           fused_tokens=parent_tokens if delta is None else parent_tokens + delta,
                           parent_pooled=parent_tokens.max(dim=1)[0].max(dim=1)[0])
        return logits, base_logits, details, parent_tokens, raw, delta, tau

    def forward(self, inputs):
        return self._forward(inputs)[0]

    def forward_with_trace(self, inputs):
        logits, _, trace, *_ = self._forward(inputs, trace=True)
        return logits, trace

    def training_outputs(self, inputs):
        logits, base_logits, trace, parent_tokens, raw, delta, tau = self._forward(inputs, teacher=True)
        return dict(logits=logits, base_logits=base_logits, parent_tokens=parent_tokens,
                    relation_raw=raw, relation_residual=delta, relation_tau=tau,
                    relation_active=self.relation_active)

def build_model(*, seed=0, pointnet2_ops=None, switches=None, rho=.02,
                epsilon=1e-6, switch_epoch=SWITCH_EPOCH, srr_scale=.5):
    """Fresh MoTrend. For the recorded training use srr_scale=1; inference uses0.5.

    No weights are loaded and no training is started. The caller controls the
    formal construction RNG; constructor order matches the trained source.
    """
    model = MoTrend(seed=seed, pointnet2_ops=pointnet2_ops, switches=switches,
                    rho=rho, epsilon=epsilon, switch_epoch=switch_epoch, srr_scale=srr_scale)
    return model, model.architecture_config()


def _validate_parent_payload(model, payload):
    require(isinstance(payload, Mapping) and isinstance(payload.get('model'), Mapping), 'Parent payload/model mapping required')
    state = payload['model']
    current = model.state_dict()
    keys = [name for name in current if not name.startswith('prepool_relation.')]
    require(set(state) == set(keys) and len(state) == PARENT_TENSORS,
            'Parent migration requires all and only original125 keys')
    require(sum(v.numel() for v in state.values()) == PARENT_PARAMETERS, 'Wrong parent parameter count')
    for name, value in state.items():
        require(isinstance(value, Tensor) and value.dtype == current[name].dtype == torch.float32
                and value.shape == current[name].shape and bool(torch.isfinite(value).all()),
                'Invalid parent tensor: ' + name)
    require(not bool(torch.count_nonzero(model.prepool_relation.output_projection.weight)),
            'Parent migration requires a fresh zero Relation exit')
    return state

def _merge_parent(model, state):
    current = model.state_dict()
    new = {name: value.detach().clone() for name, value in current.items() if name.startswith('prepool_relation.')}
    before = torch.get_rng_state().clone()
    merged = dict(state)
    merged.update(new)
    model.load_state_dict(merged, strict=True)
    require(torch.equal(before, torch.get_rng_state()), 'Parent migration changed caller RNG')
    after = model.state_dict()
    require(all(torch.equal(after[n].cpu().contiguous().view(torch.uint8), v.cpu().contiguous().view(torch.uint8)) for n, v in state.items()),
            'Parent migration was not byte exact')
    require(all(torch.equal(after[n], v) for n, v in new.items()), 'Parent migration changed Relation initialization')
    return dict(parent_tensors=PARENT_TENSORS, relation_tensors=RELATION_TENSORS,
                parent_byte_exact=True, relation_unchanged=True, strict=True)

def load_fresh_initialization(model, payload, archive_sha256):
    """Restore only the sealed epoch-minus-one parent initialization.

    The caller must compute archive_sha256 from the actual file before loading.
    A trained checkpoint is never installed as fresh training initialization.
    """
    require(isinstance(payload, Mapping), 'Fresh payload must be a mapping')
    require(archive_sha256 == FRESH_INITIALIZATION_SHA, 'Wrong fresh initialization archive SHA')
    require(payload.get('epoch') == -1 and payload.get('from_scratch') is True,
            'Training initialization must be epoch=-1/from_scratch, never a trained checkpoint')
    state = _validate_parent_payload(model, payload)
    current = model.state_dict()
    differences = []
    for name, value in state.items():
        if not torch.equal(current[name].cpu().contiguous().view(torch.uint8), value.cpu().contiguous().view(torch.uint8)):
            error = float((current[name].cpu().double() - value.cpu().double()).abs().max())
            require(name in QR_KEYS and tuple(value.shape) == (10, 80) and error <= 2e-7,
                    'Unexpected fresh constructor state differs: ' + name)
            differences.append(dict(name=name, max_abs=error))
    receipt = _merge_parent(model, state)
    model.initialization_role = 'sealed_epoch_minus_one_fresh'
    return dict(**receipt, archive_sha256=archive_sha256, initialization_role=model.initialization_role,
                constructor_qr_roundoff=differences, trained_checkpoint_loaded=False)

def assert_from_scratch_training(model):
    require(model.initialization_role == 'sealed_epoch_minus_one_fresh',
            'Formal staged50 training requires sealed fresh initialization, not diagnostic trained weights')
    return True
