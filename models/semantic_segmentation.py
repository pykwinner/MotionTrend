"""SYNTHIA4D dense semantic segmentation with shared EBTE/CMTR/SRRF modules.

Three input RGB/XYZ frames are encoded and decoded to per-point class logits.
All enabled modules train jointly; construction never loads trained weights.
"""
from __future__ import annotations
import json
import math
from collections.abc import Mapping
from pathlib import Path
import torch
from torch import Tensor, nn
from modules.point_ops import bind
from modules.point_4d_convolution import P4DConv, P4DTransConv
from modules.transformer_v1 import Transformer
from modules.temporal_attention import TemporalTransformer
from modules.ebte import LOCAL_PHASE_ENERGY, Conv3WithEBTE, BottleneckTrendEncoding
from modules.cmtr import CMTR
from modules.srrf import SRRF
from .sequence_classification import bounded_srrf_residual, require

class SynthiaPSTSegmentation(nn.Module):
    """Four-stage P4 encoder + Eq9 bottleneck + optional EBTE/CMTR/SRRF + learned-skip decoder."""

    def __init__(self, conv1, conv2, conv3, conv4, emb_relu,
                 deconv4, deconv3, deconv2, deconv1, outconv, transformer, *,
                 use_ebte=False, use_cmtr=False, use_srrf=False, seed=0,
                 context_points=2048, rho=.02, epsilon=1e-6, srrf_scale=1.0,
                 branch_dim=10, pointnet2_ops=None):
        super().__init__()
        point_ops = pointnet2_ops if pointnet2_ops is not None else bind()
        for name, value in [("use_ebte", use_ebte), ("use_cmtr", use_cmtr), ("use_srrf", use_srrf)]:
            require(type(value) is bool, name + " must be a bool")
            setattr(self, name, value)
        require(type(seed) is int and 0 <= seed < 2**63, "seed must be an integer in [0,2**63)")
        require(type(context_points) is int and context_points >= 8, "context_points >= 8 required")
        require(type(rho) in (float, int) and math.isfinite(rho) and 0 < rho <= 1, "Invalid SRRF rho")
        require(type(epsilon) in (float, int) and math.isfinite(epsilon) and epsilon > 0, "Invalid SRRF epsilon")
        require(type(srrf_scale) in (float, int) and math.isfinite(srrf_scale) and 0 < srrf_scale <= 1,
                "SRRF scale must be a finite positive number no greater than 1")
        self.seed, self.rho, self.epsilon = seed, float(rho), float(epsilon)
        self.srrf_scale, self.context_points = float(srrf_scale), int(context_points)
        self.branch_dim = int(branch_dim)

        self.conv1, self.conv2, self.conv4 = conv1, conv2, conv4
        # conv3: plain P4DConv when EBTE is off, EBTE-wrapped P4DConv when on.
        if self.use_ebte:
            self.conv3 = Conv3WithEBTE(
                conv3, pointnet2_ops=point_ops, seed=seed,
                branch_dim=branch_dim, energy_mode=LOCAL_PHASE_ENERGY)
        else:
            self.conv3 = conv3
        self.emb_relu = emb_relu
        self.deconv4, self.deconv3 = deconv4, deconv3
        self.deconv2, self.deconv1 = deconv2, deconv1
        self.outconv = outconv
        self.transformer = transformer

        dim = self.mlp_head_dim()
        if self.use_ebte:
            self.bottleneck_ebte = BottleneckTrendEncoding(dim, branch_dim=branch_dim, seed=seed)
        if self.use_cmtr:
            self.cmtr = CMTR(dim, seed=seed)
        if self.use_srrf:
            before = torch.get_rng_state().clone()
            self.srrf = SRRF(dim=dim, hidden_dim=32, k=8, init_seed=seed)
            require(torch.equal(before, torch.get_rng_state()),
                    "SRRF construction changed the parent CPU stream")

    def mlp_head_dim(self) -> int:
        attn = self.transformer.layers[0][0]
        return attn.to_qkv.in_features

    def model_config(self):
        return dict(use_ebte=self.use_ebte, use_cmtr=self.use_cmtr, use_srrf=self.use_srrf,
                    seed=self.seed, rho=self.rho, epsilon=self.epsilon,
                    srrf_scale=self.srrf_scale, context_points=self.context_points,
                    branch_dim=self.branch_dim)

    def _context_frames(self, xyzs: Tensor) -> Tensor:
        b, l, n, _ = xyzs.shape
        p = min(self.context_points, n)
        step = max(1, n // p)
        return xyzs[:, :, ::step, :][:, :, :p, :].contiguous()

    def forward(self, xyzs: Tensor, rgbs: Tensor, *, times: Tensor | None = None,
                valid: Tensor | None = None) -> Tensor:
        if not isinstance(xyzs, Tensor) or xyzs.ndim != 4 or xyzs.shape[1] != 3 or xyzs.shape[-1] != 3:
            raise ValueError("Expected XYZ [B,3,N,3]")
        if not isinstance(rgbs, Tensor) or rgbs.shape != (xyzs.shape[0], 3, 3, xyzs.shape[2]):
            raise ValueError("Expected RGB [B,3,3,N]")
        if (xyzs.dtype != torch.float32 or rgbs.dtype != xyzs.dtype
                or rgbs.device != xyzs.device or xyzs.shape[0] < 1):
            raise ValueError("XYZ/RGB must be nonempty float32 tensors on the same device")
        remaining = xyzs.shape[2]
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            remaining //= conv.spatial_stride
        if remaining < 3:
            raise ValueError("The coarsest scale requires at least three points for interpolation")
        if not bool(torch.isfinite(xyzs).all()) or not bool(torch.isfinite(rgbs).all()):
            raise ValueError("XYZ/RGB must be finite")
        if self.use_cmtr:
            if not isinstance(times, Tensor) or not isinstance(valid, Tensor):
                raise ValueError("use_cmtr requires real-source times and validity")
            if times.shape != xyzs.shape[:2] or valid.shape != xyzs.shape[:2]:
                raise ValueError("Expected times/valid [B,3]")
            if times.dtype != xyzs.dtype or valid.dtype != torch.bool:
                raise ValueError("Time dtype must match XYZ; valid must be bool")
            if times.device != xyzs.device or valid.device != xyzs.device:
                raise ValueError("Frame metadata device differs")
            if (not bool(torch.isfinite(times).all())
                    or bool(((times < 0) | (times > 1)).any())
                    or not bool(valid.any(dim=1).all())):
                raise ValueError("Invalid real-source time or empty context")
            ordered = torch.where(valid, times, torch.full_like(times, float("inf"))).sort(dim=1).values
            if bool(((ordered[:, 1:] == ordered[:, :-1]) & torch.isfinite(ordered[:, 1:])).any()):
                raise ValueError("Repeated source times must be masked before encoding")

        new_xyzs1, new_features1 = self.conv1(xyzs, rgbs)
        new_xyzs2, new_features2 = self.conv2(new_xyzs1, new_features1)
        new_xyzs3, new_features3 = self.conv3(new_xyzs2, new_features2)
        new_xyzs4, new_features4 = self.conv4(new_xyzs3, new_features3)

        embedding = self.emb_relu(new_features4.permute(0, 1, 3, 2))     # [B, L, A, C]
        tokens = embedding
        if self.use_ebte:
            tokens = self.bottleneck_ebte(tokens)
        tokens = self.transformer(new_xyzs4, tokens)

        if self.use_cmtr:
            context = self._context_frames(xyzs)
            context_residual = self.cmtr(tokens, new_xyzs4, times,
                                      context, times, valid)
            tokens = tokens + context_residual
        if self.use_srrf:
            raw = self.srrf(tokens, new_xyzs4) * self.srrf_scale
            delta, _ = bounded_srrf_residual(raw, tokens,
                                                 rho=self.rho, epsilon=self.epsilon)
            tokens = tokens + delta
        if tokens.shape != embedding.shape:
            raise ValueError("Bottleneck modules must retain dense token shapes")

        new_features4 = tokens.permute(0, 1, 3, 2)
        new_xyzsd4, new_featuresd4 = self.deconv4(new_xyzs4, new_xyzs3, new_features4, new_features3)
        new_xyzsd3, new_featuresd3 = self.deconv3(new_xyzsd4, new_xyzs2, new_featuresd4, new_features2)
        new_xyzsd2, new_featuresd2 = self.deconv2(new_xyzsd3, new_xyzs1, new_featuresd3, new_features1)
        _, new_featuresd1 = self.deconv1(new_xyzsd2, xyzs, new_featuresd2, rgbs)
        return self.outconv(new_featuresd1.transpose(1, 2)).transpose(1, 2)

    def parameter_counts(self):
        named = list(self.named_parameters())

        def count(*prefixes):
            return sum(p.numel() for n, p in named if n.startswith(prefixes))

        values = dict(
            total=sum(p.numel() for p in self.parameters()),
            encoder=count("conv1.", "conv2.", "conv3.", "conv4.") - count("conv3.adapter."),
            transformer=count("transformer.") - sum(p.numel() for n,p in named if n.endswith("w_dt")),
            eq9=sum(p.numel() for n, p in named if n.endswith("w_dt")),
            decoder=count("deconv4.", "deconv3.", "deconv2.", "deconv1.", "outconv."),
        )
        if self.use_ebte:
            values["ebte_conv3"] = count("conv3.adapter.")
            values["ebte_bottleneck"] = count("bottleneck_ebte.")
        if self.use_cmtr:
            values["cmtr"] = count("cmtr.")
        if self.use_srrf:
            values["srrf"] = count("srrf.")
        return values

    def specification(self):
        return dict(
            switches=self.model_config(),
            input_frames=3, points_per_frame=getattr(self, "points_per_frame", 16384),
            encoder="four-stage P4 RGB/geometric encoder, EBTE conv3",
            bottleneck="Eq9 PST transformer + optional EBTE/CMTR/SRRF",
            decoder="four-stage learned-skip decoder",
        )


def build_synthia_model(config=None, *, pointnet2_ops=None) -> SynthiaPSTSegmentation:
    """Build one jointly trainable segmentation model without loading weights.

    A complete config dictionary or a JSON path is accepted. Module switches
    are explicit booleans under ``model``; no experiment-name aliases apply.
    """
    if config is None:
        config = Path(__file__).resolve().parents[1] / "config" / "synthia_full.json"
    if isinstance(config, (str, Path)):
        cfg = json.loads(Path(config).read_text(encoding="utf-8-sig"))
    elif isinstance(config, Mapping):
        cfg = dict(config)
    else:
        raise TypeError("config must be a mapping or JSON path")
    mcfg = cfg.get("model", {})
    require(isinstance(mcfg, Mapping), "model must be a configuration mapping")
    supported = {
        "dim", "depth", "heads", "dim_head", "mlp_dim", "dropout", "max_frames",
        "radius", "neighbors", "branch_dim", "radius_multipliers", "mlp_planes",
        "spatial_strides", "temporal_kernels", "emb_relu", "context_points",
        "rho", "epsilon", "srrf_scale", "use_ebte", "use_cmtr", "use_srrf",
    }
    require(not set(mcfg) - supported, f"Unknown model keys: {sorted(set(mcfg) - supported)}")
    seed = cfg.get("seed", 0)
    require(type(seed) is int and 0 <= seed < 2**63, "seed must be an integer in [0,2**63)")
    require(type(cfg.get("num_classes", 12)) is int and cfg.get("num_classes", 12) == 12,
            "SYNTHIA4D requires 12 semantic classes")
    require(type(cfg.get("frames_per_clip", 3)) is int and cfg.get("frames_per_clip", 3) == 3,
            "This SYNTHIA4D configuration requires three frames")
    require(type(mcfg.get("max_frames", 3)) is int and mcfg.get("max_frames", 3) == 3,
            "max_frames must equal frames_per_clip=3")
    dims = {k: mcfg.get(k, v) for k, v in dict(dim=1024, depth=2, heads=4,
                dim_head=256, mlp_dim=1024, neighbors=32, branch_dim=10).items()}
    require(all(type(v) is int and v > 0 for v in dims.values()),
            "Channel widths, attention sizes, neighbors and branch_dim must be positive integers")
    radius = mcfg.get("radius", .9)
    require(type(radius) in (int, float) and math.isfinite(radius) and radius > 0,
            "radius must be positive and finite")
    dropout = mcfg.get("dropout", 0.)
    require(type(dropout) in (int, float) and math.isfinite(dropout) and 0 <= dropout < 1,
            "dropout must lie in [0,1)")
    require(type(mcfg.get("emb_relu", False)) is bool, "emb_relu must be a bool")
    radius_multipliers = mcfg.get("radius_multipliers", [1, 2, 4, 8])
    mlp_planes = mcfg.get("mlp_planes", [[32, 64, 128], [128, 128, 256],
                                       [256, 256, 512], [512, 512, 1024]])
    spatial_strides = mcfg.get("spatial_strides", [4, 4, 4, 2])
    temporal_kernels = mcfg.get("temporal_kernels", [1, 1, 3, 1])
    for name, values in (("radius_multipliers", radius_multipliers), ("mlp_planes", mlp_planes),
                         ("spatial_strides", spatial_strides), ("temporal_kernels", temporal_kernels)):
        require(isinstance(values, (list, tuple)) and len(values) == 4,
                name + " must contain four stages")
    require(all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in radius_multipliers),
            "radius_multipliers must be positive and finite")
    require(all(type(v) is int and v > 0 for v in spatial_strides),
            "spatial_strides must be positive integers")
    require(all(type(v) is int and v > 0 and v % 2 == 1 for v in temporal_kernels),
            "temporal_kernels must be positive odd integers")
    require(all(isinstance(v, (list, tuple)) and len(v) > 0 and
                all(type(c) is int and c > 0 for c in v) for v in mlp_planes),
            "mlp_planes must contain positive channel widths")
    require(mlp_planes[-1][-1] == dims["dim"], "Last encoder width must equal transformer dim")
    require(dims["branch_dim"] <= min(mlp_planes[2][-1], dims["dim"]),
            "branch_dim must not exceed Conv3 or bottleneck channels")
    switches = {key: mcfg.get(key, False) for key in ("use_ebte", "use_cmtr", "use_srrf")}
    require(all(type(v) is bool for v in switches.values()), "Module switches must be booleans")
    require(not switches["use_ebte"] or temporal_kernels[2] == 3,
            "Conv3 EBTE requires three temporal slots")
    num_points = cfg.get("num_points", 16384)
    require(type(num_points) is int and num_points >= 3 * math.prod(spatial_strides),
            "num_points must leave at least three coarsest-scale points")
    ops = pointnet2_ops if pointnet2_ops is not None else bind()
    in_planes = [3, mlp_planes[0][-1], mlp_planes[1][-1], mlp_planes[2][-1]]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        convs = []
        for i in range(4):
            tk = temporal_kernels[i]
            convs.append(P4DConv(
                in_planes=in_planes[i], mlp_planes=mlp_planes[i],
                mlp_batch_norm=[True] * len(mlp_planes[i]),
                mlp_activation=[True] * len(mlp_planes[i]),
                spatial_kernel_size=[radius * radius_multipliers[i], dims["neighbors"]],
                spatial_stride=spatial_strides[i], temporal_kernel_size=tk,
                temporal_stride=1, temporal_padding=[(tk - 1) // 2] * 2,
                temporal_padding_mode="replicate", operator="+", spatial_pooling="max",
                temporal_pooling="sum", pointnet2_ops=ops))
        emb_relu = nn.ReLU(inplace=True) if mcfg.get("emb_relu", False) else nn.Identity()
        deconv4 = P4DTransConv(dims["dim"], [256, 256], [True, True], [True, True],
                              original_planes=mlp_planes[2][-1], pointnet2_ops=ops)
        deconv3 = P4DTransConv(256, [256, 256], [True, True], [True, True],
                              original_planes=mlp_planes[1][-1], pointnet2_ops=ops)
        deconv2 = P4DTransConv(256, [128, 128], [True, True], [True, True],
                              original_planes=mlp_planes[0][-1], pointnet2_ops=ops)
        deconv1 = P4DTransConv(128, [128, 128], [True, True], [True, True],
                              original_planes=3, pointnet2_ops=ops)
        outconv = nn.Conv2d(128, 12, 1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 62117)
        transformer = Transformer(dim=dims["dim"], depth=dims["depth"], heads=dims["heads"],
            dim_head=dims["dim_head"], mlp_dim=dims["mlp_dim"], dropout=dropout)
    transformer = TemporalTransformer(transformer, max_frames=3)
    options = {key: mcfg[key] for key in ("context_points", "rho", "epsilon", "srrf_scale") if key in mcfg}
    model = SynthiaPSTSegmentation(*convs, emb_relu, deconv4, deconv3, deconv2, deconv1,
                outconv, transformer, seed=seed, branch_dim=dims["branch_dim"],
                pointnet2_ops=ops, **switches, **options)
    model.points_per_frame = num_points
    return model


_ADAPTATION_PREFIXES = ("conv3.adapter.", "bottleneck_ebte.", "cmtr.", "srrf.")


def _weight_state(path):
    """Read model tensors only; optimizer/RNG restoration belongs to training."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Expected a nonempty model state dictionary")
    if not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items()):
        raise ValueError("A model state dictionary must contain only string-to-tensor entries")
    prefixed = [k.startswith("module.") for k in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Mixed wrapped and unwrapped checkpoint keys")
    return {k.removeprefix("module."): v for k, v in state.items()} if all(prefixed) else dict(state)


def _preflight_weights(expected, incoming):
    missing, extra = sorted(expected.keys() - incoming.keys()), sorted(incoming.keys() - expected.keys())
    if missing or extra:
        raise ValueError(f"Incompatible checkpoint: missing={missing[:8]}, unexpected={extra[:8]}")
    for key, value in incoming.items():
        target = expected[key]
        if value.shape != target.shape or value.dtype != target.dtype or value.layout != target.layout:
            raise ValueError(f"Incompatible tensor {key}: shape, dtype or layout differs")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite checkpoint tensor: {key}")


def _load_validated_weights(model, state, *, strict):
    # Preflight rejects malformed files before any copy. Roll back an unexpected
    # load exception as well, so a rejected file does not leave mixed parameters.
    current = model.state_dict()
    backup = {key: current[key].detach().cpu().clone() for key in state}
    try:
        return model.load_state_dict(state, strict=strict)
    except Exception:
        model.load_state_dict(backup, strict=False)
        raise


def _segmentation_model(model):
    core = model.module if isinstance(model, nn.DataParallel) else model
    if not isinstance(core, SynthiaPSTSegmentation):
        raise TypeError("This weight loader requires a SynthiaPSTSegmentation model")
    return core


def load_checkpoint(model, path, *, logger=None):
    """Explicitly load an exact compatible SYNTHIA model, with no partial matching.

    This helper is never called by from-scratch training. It does not restore
    optimizer/scheduler state or translate another architecture's tensor names.
    """
    core, state = _segmentation_model(model), _weight_state(path)
    _preflight_weights(core.state_dict(), state)
    _load_validated_weights(core, state, strict=True)
    if logger is not None:
        logger.info("Loaded complete compatible SYNTHIA model from %s", path)
    return [], []


def load_parent_only(model, path, *, logger=None):
    """Explicitly load all backbone tensors; leave all three adaptations intact.

    Only known adaptation prefixes may be omitted or ignored. An incomplete
    backbone or an unknown architecture is rejected before modifying the model.
    This optional initialization helper does not create a training stage.
    """
    core, state = _segmentation_model(model), _weight_state(path)
    expected = {k: v for k, v in core.state_dict().items() if not k.startswith(_ADAPTATION_PREFIXES)}
    backbone = {k: v for k, v in state.items() if not k.startswith(_ADAPTATION_PREFIXES)}
    _preflight_weights(expected, backbone)
    result = _load_validated_weights(core, backbone, strict=False)
    if logger is not None:
        logger.info("Loaded complete SYNTHIA backbone from %s; EBTE/CMTR/SRRF unchanged", path)
    return result

