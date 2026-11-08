"""Sequence classification for MSR-Action3D and NTU60.

Both classifiers retain their original constructor, registration order and
forward arithmetic. All enabled EBTE/CMTR/SRRF modules train jointly. Separate
dataset defaults prevent NTU geometry or class count from changing MSR behavior.
"""


from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from modules.point_ops import bind
from modules.point_4d_convolution import P4DConv
from modules.transformer_v1 import Transformer
from modules.temporal_attention import TemporalTransformer
from modules.ebte import EBTE
from modules.cmtr import CMTR
from modules.srrf import SRRF


MSR_MODEL_DEFAULTS = dict(
    use_ebte=True, use_cmtr=True, use_srrf=False, seed=0,
    num_classes=20, max_frames=12, rho=.02, epsilon=1e-6, srrf_scale=1.,
    radius=.3, nsamples=32, spatial_stride=32,
    temporal_kernel_size=3, temporal_stride=2,
    dim=80, depth=5, heads=2, dim_head=40, mlp_dim=160,
    dropout1=0., dropout2=0.,
)


NTU_MODEL_DEFAULTS = dict(
    use_ebte=True, use_cmtr=True, use_srrf=False, seed=0,
    num_classes=60, max_frames=12, rho=.02, epsilon=1e-6, srrf_scale=1.,
    radius=.1, nsamples=32, spatial_stride=32,
    temporal_kernel_size=3, temporal_stride=2,
    dim=80, depth=5, heads=2, dim_head=40, mlp_dim=160,
    dropout1=0., dropout2=.5,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_srrf_residual(raw: Tensor, parent_tokens: Tensor, *, rho: float, epsilon: float):
    """Limit each residual component with unit derivative at the zero exit.

    tau=rho*stopgrad(RMS(U))+epsilon; delta=tau*tanh(raw/tau).
    The feature budget carries no gradient into the parent tokens.
    """
    require(type(rho) in (float, int) and math.isfinite(rho) and 0 < rho <= 1,
            "rho must be a finite positive number no greater than 1")
    require(type(epsilon) in (float, int) and math.isfinite(epsilon) and epsilon > 0,
            "epsilon must be finite and positive")
    require(raw.shape == parent_tokens.shape and raw.ndim == 4,
            "Relation and parent tokens must have identical [B,L,N,C] shapes")
    require(raw.device == parent_tokens.device and raw.dtype == parent_tokens.dtype
            and raw.is_floating_point(), "Relation and parent tokens must share floating dtype/device")
    require(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(parent_tokens).all()),
            "Relation and parent features must be finite")
    with torch.autocast(device_type=raw.device.type, enabled=False):
        parent = parent_tokens.detach().float()
        maximum = parent.abs().amax(dim=-1, keepdim=True)
        divisor = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
        rms = maximum * (parent / divisor).square().mean(dim=-1, keepdim=True).sqrt()
        tau = (float(rho) * rms + float(epsilon)).detach()
        bounded = tau * torch.tanh(raw.float() / tau)
    require(bool(torch.isfinite(tau).all()) and bool((tau > 0).all())
            and bool(torch.isfinite(bounded).all()), "Nonfinite Relation trust region")
    return bounded.to(raw.dtype), tau


class MSRPSTModel(nn.Module):
    """PST tube with independently optional EBTE/CMTR/SRRF.
    """

    def __init__(self, *, use_ebte=True, use_cmtr=True, use_srrf=False, seed=0,
                 num_classes=20, max_frames=12, rho=.02, epsilon=1e-6, srrf_scale=1.0,
                 radius=.3, nsamples=32, spatial_stride=32,
                 temporal_kernel_size=3, temporal_stride=2,
                 dim=80, depth=5, heads=2, dim_head=40, mlp_dim=160,
                 dropout1=0., dropout2=0., pointnet2_ops=None):
        super().__init__()
        for name, value in [("use_ebte", use_ebte), ("use_cmtr", use_cmtr), ("use_srrf", use_srrf)]:
            require(type(value) is bool, name + " must be a bool")
            setattr(self, name, value)
        require(type(seed) is int and 0 <= seed < 2**63, "seed must be an integer in [0,2**63)")
        require(type(rho) in (float, int) and math.isfinite(rho) and 0 < rho <= 1, "Invalid SRRF rho")
        require(type(epsilon) in (float, int) and math.isfinite(epsilon) and epsilon > 0, "Invalid SRRF epsilon")
        require(type(srrf_scale) in (float, int) and math.isfinite(srrf_scale) and 0 < srrf_scale <= 1,
                "SRRF scale must be a finite positive number no greater than 1")
        for name, value in dict(num_classes=num_classes, max_frames=max_frames,
                                nsamples=nsamples, spatial_stride=spatial_stride,
                                temporal_kernel_size=temporal_kernel_size,
                                temporal_stride=temporal_stride, dim=dim, depth=depth,
                                heads=heads, dim_head=dim_head, mlp_dim=mlp_dim).items():
            require(type(value) is int and value > 0, name + " must be a positive integer")
        require(type(radius) in (int, float) and math.isfinite(radius) and radius > 0,
                "radius must be finite and positive")
        require(temporal_kernel_size >= 3 and temporal_kernel_size % 2 == 1,
                "temporal_kernel_size must be an odd integer at least 3")
        for name, value in (("dropout1", dropout1), ("dropout2", dropout2)):
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value < 1,
                    name + " must be in [0,1)")
        if use_ebte:
            geometry = (dim, radius, nsamples, spatial_stride,
                        temporal_kernel_size, temporal_stride, max_frames)
            require(geometry == (80, .3, 32, 32, 3, 2, 12),
                    "The released EBTE requires dim80, radius0.3, nsamples32, "
                    "spatial_stride32, temporal kernel3/stride2 and 12 output frames. "
                    "Unsupported geometry must not silently change this module.")
        self.seed, self.rho, self.epsilon = seed, float(rho), float(epsilon)
        self.srrf_scale = float(srrf_scale)
        self.temporal_stride = temporal_stride
        self.anchor_start = temporal_kernel_size // 2 - 1
        self._config = dict(use_ebte=use_ebte, use_cmtr=use_cmtr, use_srrf=use_srrf,
                            seed=seed, num_classes=num_classes, max_frames=max_frames,
                            rho=float(rho), epsilon=float(epsilon), srrf_scale=float(srrf_scale),
                            radius=float(radius), nsamples=nsamples, spatial_stride=spatial_stride,
                            temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride,
                            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                            dropout1=float(dropout1), dropout2=float(dropout2))
        ops = pointnet2_ops if pointnet2_ops is not None else bind()

        tube = P4DConv(
            in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
            spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
            temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride,
            temporal_padding=[1, 0], operator="+", spatial_pooling="max", temporal_pooling="max",
            pointnet2_ops=ops)
        self.tube_embedding = (EBTE(
            tube, backbone_seed=seed, pointnet2_ops=ops, strict=True)
            if self.use_ebte else tube)

        self.transformer = TemporalTransformer(
            Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1), max_frames=max_frames)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes))

        if self.use_cmtr:
            self.cmtr = CMTR(dim, seed=seed)
        if self.use_srrf:
            before = torch.get_rng_state().clone()
            self.srrf = SRRF(dim=dim, hidden_dim=32, k=8, init_seed=seed)
            require(torch.equal(before, torch.get_rng_state()),
                    "SRRF construction changed the parent CPU stream")

    def model_config(self):
        """Complete constructor configuration; native operators are runtime dependencies."""
        return dict(self._config)

    def parameter_counts(self):
        named = list(self.named_parameters())
        values = dict(
            total=sum(p.numel() for p in self.parameters()),
            backbone=sum(p.numel() for n, p in named
                         if n.startswith("tube_embedding.") and ".adapter." not in n)
            + sum(p.numel() for n, p in named if n.startswith("transformer.") and not n.endswith("w_dt")),
            eq9=sum(p.numel() for n, p in named if n.endswith("w_dt")),
            head=sum(p.numel() for n, p in named if n.startswith("mlp_head.")),
        )
        if self.use_ebte:
            values["ebte"] = sum(p.numel() for n, p in named if n.startswith("tube_embedding.adapter."))
        if self.use_cmtr:
            values["cmtr"] = sum(p.numel() for n, p in named if n.startswith("cmtr."))
        if self.use_srrf:
            values["srrf"] = sum(p.numel() for n, p in named if n.startswith("srrf."))
        return values

    def forward(self, inputs):
        require(isinstance(inputs, Mapping) and "local" in inputs,
                "model requires an input mapping with local clips")
        if self.use_cmtr:
            required = ("context", "times", "valid", "local_times",
                        "local_to_context_scale", "local_to_context_offset")
            require(all(key in inputs for key in required), "Missing CMTR alignment metadata")
        elif self.use_srrf:
            require(all(key in inputs for key in ("local_to_context_scale", "local_to_context_offset")),
                    "Missing SRRF coordinate alignment metadata")

        xyz, features = self.tube_embedding(inputs["local"])
        tokens = self.transformer(xyz, features.permute(0, 1, 3, 2))

        aligned_xyz = xyz
        if self.use_cmtr or self.use_srrf:
            scale, offset = inputs["local_to_context_scale"], inputs["local_to_context_offset"]
            require(scale.ndim == 1 and offset.shape == (tokens.shape[0], 3) and bool((scale > 0).all()),
                    "Invalid local-to-context affine metadata")
            aligned_xyz = xyz * scale[:, None, None, None] + offset[:, None, None, :]

        if self.use_cmtr:
            stop = self.anchor_start + tokens.shape[1] * self.temporal_stride
            anchor_times = inputs["local_times"][:, self.anchor_start:stop:self.temporal_stride]
            require(anchor_times.shape == tokens.shape[:2], "P4D timestamps do not match token count")
            tokens = tokens + self.cmtr(tokens, aligned_xyz, anchor_times,
                                                inputs["context"], inputs["times"], inputs["valid"])
        if self.use_srrf:
            raw = self.srrf(tokens, aligned_xyz) * self.srrf_scale
            delta, _ = bounded_srrf_residual(raw, tokens, rho=self.rho, epsilon=self.epsilon)
            tokens = tokens + delta

        return self.mlp_head(tokens.max(dim=1)[0].max(dim=1)[0])


class NTUPSTModel(nn.Module):
    """PST tube with independently optional EBTE/CMTR/SRRF.
    """

    def __init__(self, *, use_ebte=True, use_cmtr=True, use_srrf=False, seed=0,
                 num_classes=60, max_frames=12, rho=.02, epsilon=1e-6, srrf_scale=1.0,
                 radius=.1, nsamples=32, spatial_stride=32,
                 temporal_kernel_size=3, temporal_stride=2,
                 dim=80, depth=5, heads=2, dim_head=40, mlp_dim=160,
                 dropout1=0., dropout2=.5, pointnet2_ops=None):
        super().__init__()
        for name, value in [("use_ebte", use_ebte), ("use_cmtr", use_cmtr), ("use_srrf", use_srrf)]:
            require(type(value) is bool, name + " must be a bool")
            setattr(self, name, value)
        require(type(seed) is int and 0 <= seed < 2**63, "seed must be an integer in [0,2**63)")
        require(num_classes == 60, "This package uses the NTU60 60-class label space")
        require(type(rho) in (float, int) and math.isfinite(rho) and 0 < rho <= 1, "Invalid SRRF rho")
        require(type(epsilon) in (float, int) and math.isfinite(epsilon) and epsilon > 0, "Invalid SRRF epsilon")
        require(type(srrf_scale) in (float, int) and math.isfinite(srrf_scale) and 0 < srrf_scale <= 1,
                "SRRF scale must be a finite positive number no greater than 1")
        for name, value in dict(num_classes=num_classes, max_frames=max_frames,
                                nsamples=nsamples, spatial_stride=spatial_stride,
                                temporal_kernel_size=temporal_kernel_size,
                                temporal_stride=temporal_stride, dim=dim, depth=depth,
                                heads=heads, dim_head=dim_head, mlp_dim=mlp_dim).items():
            require(type(value) is int and value > 0, name + " must be a positive integer")
        require(type(radius) in (int, float) and math.isfinite(radius) and radius > 0,
                "radius must be finite and positive")
        require(temporal_kernel_size >= 3 and temporal_kernel_size % 2 == 1,
                "temporal_kernel_size must be an odd integer at least 3")
        for name, value in (("dropout1", dropout1), ("dropout2", dropout2)):
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value < 1,
                    name + " must be in [0,1)")
        if use_ebte:
            geometry = (dim, radius, nsamples, spatial_stride,
                        temporal_kernel_size, temporal_stride, max_frames)
            require(geometry == (80, .1, 32, 32, 3, 2, 12),
                    "The released EBTE requires dim80, NTU radius0.1, nsamples32, "
                    "spatial_stride32, temporal kernel3/stride2 and 12 output frames. "
                    "Unsupported geometry must not silently change this module.")
        self.seed, self.rho, self.epsilon = seed, float(rho), float(epsilon)
        self.srrf_scale = float(srrf_scale)
        self.temporal_stride = temporal_stride
        self.anchor_start = temporal_kernel_size // 2 - 1
        self._config = dict(use_ebte=use_ebte, use_cmtr=use_cmtr, use_srrf=use_srrf,
                            seed=seed, num_classes=num_classes, max_frames=max_frames,
                            rho=float(rho), epsilon=float(epsilon), srrf_scale=float(srrf_scale),
                            radius=float(radius), nsamples=nsamples, spatial_stride=spatial_stride,
                            temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride,
                            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                            dropout1=float(dropout1), dropout2=float(dropout2))
        ops = pointnet2_ops if pointnet2_ops is not None else bind()

        tube = P4DConv(
            in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
            spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
            temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride,
            temporal_padding=[1, 0], operator="+", spatial_pooling="max", temporal_pooling="max",
            pointnet2_ops=ops)
        self.tube_embedding = (EBTE(
            tube, backbone_seed=seed, pointnet2_ops=ops, strict=True, expected_radius=.1)
            if self.use_ebte else tube)

        self.transformer = TemporalTransformer(
            Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1), max_frames=max_frames)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes))

        if self.use_cmtr:
            self.cmtr = CMTR(dim, seed=seed)
        if self.use_srrf:
            before = torch.get_rng_state().clone()
            self.srrf = SRRF(dim=dim, hidden_dim=32, k=8, init_seed=seed)
            require(torch.equal(before, torch.get_rng_state()),
                    "SRRF construction changed the parent CPU stream")

    def model_config(self):
        """Complete constructor configuration; native operators are runtime dependencies."""
        return dict(self._config)

    def parameter_counts(self):
        named = list(self.named_parameters())
        values = dict(
            total=sum(p.numel() for p in self.parameters()),
            backbone=sum(p.numel() for n, p in named
                         if n.startswith("tube_embedding.") and ".adapter." not in n)
            + sum(p.numel() for n, p in named if n.startswith("transformer.") and not n.endswith("w_dt")),
            eq9=sum(p.numel() for n, p in named if n.endswith("w_dt")),
            head=sum(p.numel() for n, p in named if n.startswith("mlp_head.")),
        )
        if self.use_ebte:
            values["ebte"] = sum(p.numel() for n, p in named if n.startswith("tube_embedding.adapter."))
        if self.use_cmtr:
            values["cmtr"] = sum(p.numel() for n, p in named if n.startswith("cmtr."))
        if self.use_srrf:
            values["srrf"] = sum(p.numel() for n, p in named if n.startswith("srrf."))
        return values

    def forward(self, inputs):
        require(isinstance(inputs, Mapping) and "local" in inputs,
                "model requires an input mapping with local clips")
        if self.use_cmtr:
            required = ("context", "times", "valid", "local_times",
                        "local_to_context_scale", "local_to_context_offset")
            require(all(key in inputs for key in required), "Missing CMTR alignment metadata")
        elif self.use_srrf:
            require(all(key in inputs for key in ("local_to_context_scale", "local_to_context_offset")),
                    "Missing SRRF coordinate alignment metadata")

        xyz, features = self.tube_embedding(inputs["local"])
        tokens = self.transformer(xyz, features.permute(0, 1, 3, 2))

        aligned_xyz = xyz
        if self.use_cmtr or self.use_srrf:
            scale, offset = inputs["local_to_context_scale"], inputs["local_to_context_offset"]
            require(scale.ndim == 1 and offset.shape == (tokens.shape[0], 3) and bool((scale > 0).all()),
                    "Invalid local-to-context affine metadata")
            aligned_xyz = xyz * scale[:, None, None, None] + offset[:, None, None, :]

        if self.use_cmtr:
            stop = self.anchor_start + tokens.shape[1] * self.temporal_stride
            anchor_times = inputs["local_times"][:, self.anchor_start:stop:self.temporal_stride]
            require(anchor_times.shape == tokens.shape[:2], "P4D timestamps do not match token count")
            tokens = tokens + self.cmtr(tokens, aligned_xyz, anchor_times,
                                                inputs["context"], inputs["times"], inputs["valid"])
        if self.use_srrf:
            raw = self.srrf(tokens, aligned_xyz) * self.srrf_scale
            delta, _ = bounded_srrf_residual(raw, tokens, rho=self.rho, epsilon=self.epsilon)
            tokens = tokens + delta

        return self.mlp_head(tokens.max(dim=1)[0].max(dim=1)[0])


def _unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def build_model(config="config/msr.json", *, pointnet2_ops=None) -> MSRPSTModel:
    """Build naturally from the effective config, without reseeding or loading weights.

    The training entry point seeds once before construction. Model construction
    must consume that normal stream so subsequent shuffle/worker seeds follow it.
    """
    if isinstance(config, (str, Path)):
        cfg = json.loads(Path(config).read_text(encoding="utf-8-sig"))
    elif isinstance(config, Mapping):
        cfg = dict(config)
    else:
        raise TypeError("config must be a mapping or JSON path")
    require(not cfg.get("pretrained"),
            "Implicit pretrained loading is disabled; use the explicit evaluation or resume entry point")
    mcfg = dict(cfg.get("model", {}))
    data = cfg.get("data", {})
    train = cfg.get("train", {})
    # clip_len belongs to the data definition. Reject contradictory duplicates.
    legacy_clip = mcfg.pop("clip_len", None)
    frames = data.get("clip_len", legacy_clip if legacy_clip is not None else 24)
    require(legacy_clip is None or legacy_clip == frames,
            "model.clip_len and data.clip_len disagree")
    require(type(frames) is int and frames > 0, "data.clip_len must be a positive integer")
    unsupported = set(mcfg) - set(MSR_MODEL_DEFAULTS)
    require(not unsupported, f"Unsupported model configuration keys: {sorted(unsupported)}")
    if "seed" in mcfg and "seed" in train:
        require(mcfg["seed"] == train["seed"], "model.seed and train.seed must agree")
    kwargs = dict(MSR_MODEL_DEFAULTS)
    kwargs.update(mcfg)
    kwargs["seed"] = mcfg.get("seed", train.get("seed", 0))
    kernel, stride = kwargs["temporal_kernel_size"], kwargs["temporal_stride"]
    require(type(kernel) is int and kernel >= 3 and kernel % 2 == 1,
            "temporal_kernel_size must be an odd integer at least 3")
    require(type(stride) is int and stride > 0, "temporal_stride must be a positive integer")
    require(frames + 1 >= kernel and (frames + 1 - kernel) % stride == 0,
            "data.clip_len does not match the configured temporal kernel/stride and padding [1,0]")
    output_frames = (frames + 1 - kernel) // stride + 1
    require("max_frames" not in mcfg or mcfg["max_frames"] == output_frames,
            "model.max_frames must match the actual output frame count")
    kwargs["max_frames"] = output_frames
    points = data.get("num_points", 2048)
    require(type(points) is int and points > 0, "data.num_points must be a positive integer")
    if kwargs["use_ebte"]:
        require(frames == 24 and points == 2048,
                "The released EBTE requires local clips with 24 frames and 2048 points")
    return MSRPSTModel(**kwargs, pointnet2_ops=pointnet2_ops)


def build_ntu_model(config=Path(__file__).resolve().parents[1] / "config/ntu_xsub.json", *, pointnet2_ops=None) -> NTUPSTModel:
    """Build naturally from the effective config, without reseeding or loading weights.

    The training entry point seeds once before construction. Model construction
    must consume that normal stream so subsequent shuffle/worker seeds follow it.
    """
    if isinstance(config, (str, Path)):
        cfg = json.loads(Path(config).read_text(encoding="utf-8-sig"))
    elif isinstance(config, Mapping):
        cfg = dict(config)
    else:
        raise TypeError("config must be a mapping or JSON path")
    require(not cfg.get("pretrained"),
            "Implicit pretrained loading is disabled; use the explicit evaluation or resume entry point")
    mcfg = dict(cfg.get("model", {}))
    data = cfg.get("data", {})
    train = cfg.get("train", {})
    # Frame selection/spacing belongs to the dataset, not the model.
    frames = data.get("frames_per_clip", 24)
    require(type(frames) is int and frames > 0, "data.frames_per_clip must be a positive integer")
    unsupported = set(mcfg) - set(NTU_MODEL_DEFAULTS)
    require(not unsupported, f"Unsupported model configuration keys: {sorted(unsupported)}")
    if "seed" in mcfg and "seed" in train:
        require(mcfg["seed"] == train["seed"], "model.seed and train.seed must agree")
    kwargs = dict(NTU_MODEL_DEFAULTS)
    kwargs.update(mcfg)
    kwargs["seed"] = mcfg.get("seed", train.get("seed", 0))
    kernel, stride = kwargs["temporal_kernel_size"], kwargs["temporal_stride"]
    require(type(kernel) is int and kernel >= 3 and kernel % 2 == 1,
            "temporal_kernel_size must be an odd integer at least 3")
    require(type(stride) is int and stride > 0, "temporal_stride must be a positive integer")
    require(frames + 1 >= kernel and (frames + 1 - kernel) % stride == 0,
            "data.frames_per_clip does not match the configured temporal kernel/stride and padding [1,0]")
    output_frames = (frames + 1 - kernel) // stride + 1
    require("max_frames" not in mcfg or mcfg["max_frames"] == output_frames,
            "model.max_frames must match the actual output frame count")
    kwargs["max_frames"] = output_frames
    points = data.get("num_points", 2048)
    require(type(points) is int and points > 0, "data.num_points must be a positive integer")
    if kwargs["use_ebte"]:
        require(frames == 24 and points == 2048,
                "The released EBTE requires local clips with 24 frames and 2048 points")
    return NTUPSTModel(**kwargs, pointnet2_ops=pointnet2_ops)


def load_checkpoint(model, path):
    """Explicit strict loading; missing modules or extra module weights are errors."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, Mapping) and "model" in ckpt else ckpt
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint must contain a model state mapping")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    _unwrap(model).load_state_dict(state, strict=True)
    return ckpt
