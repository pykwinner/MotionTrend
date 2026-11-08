"""Energy-Bounded Trend Encoding (EBTE).

Trained phase contrast adapter and raw phase-energy bound; names change only.

The released P4D spatial/temporal max ordering and checkpoint submodule keys
are retained. This file is self contained; native point operations are passed
explicitly by the backbone. Optional process-global legacy diagnostics are
excluded; forward_trace still exposes the original tensor trace.
"""
from __future__ import annotations
import importlib
import math
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn

LOCAL_PHASE_ENERGY = "local_phase_energy"
GLOBAL_PHASE_ENERGY = "global_phase_energy"
ENERGY_MODES = (LOCAL_PHASE_ENERGY, GLOBAL_PHASE_ENERGY)

INPUT_FRAMES = 24
INPUT_POINTS = 2_048
RADIUS = 0.30
TUBE_NEIGHBORS = 32
TUBE_FRAMES = 3
TIME_LABELS = (-1.0, 0.0, 1.0)
SPATIAL_STRIDE = 32
TEMPORAL_STRIDE = 2
TEMPORAL_PADDING = (1, 0)
FRAMES = 12
POINTS = 64
TOKEN_DIM = 80
BRANCH_DIM = 10
FUSED_DIM = 20
DEPTH = 5
HEADS = 2
HEAD_DIM = 40
JOINT_LN_EPS = 1.0e-5
VANILLA = "vanilla"
EVENIZED = "evenized"
PRIMARY = "primary"
RAW_TRIPLET = "raw_triplet"
LEARNED_ZERO_SUM_BASIS = "learned_zero_sum_basis"
ADJACENT_DIFF = "adjacent_diff"
FORMAL_ARMS = (VANILLA, EVENIZED, PRIMARY)
OPTIONAL_ABLATION_ARMS = (RAW_TRIPLET, LEARNED_ZERO_SUM_BASIS, ADJACENT_DIFF)
OPCR_ARMS = FORMAL_ARMS + OPTIONAL_ABLATION_ARMS
ADAPTED_ARMS = tuple(arm for arm in OPCR_ARMS if arm != VANILLA)
INIT_SEED_BASE = 20_260_811
INIT_SEED_STRIDE = 1_009
INIT_ODD_SALT = 17
INIT_CURVATURE_SALT = 43
VANILLA_PARAMETER_COUNT = 276_460
EXTRA_PARAMETER_COUNT = (
    TOKEN_DIM * BRANCH_DIM
    + TOKEN_DIM * BRANCH_DIM
    + FUSED_DIM * TOKEN_DIM
)
LEARNED_BASIS_EXTRA_PARAMETER_COUNT = EXTRA_PARAMETER_COUNT + 4
TOTAL_PARAMETER_COUNT = VANILLA_PARAMETER_COUNT + EXTRA_PARAMETER_COUNT


class OPCRContractError(RuntimeError):
    """A frozen OPCR shape, method, or released-model contract was violated."""


@dataclass(frozen=True)
class PhaseComponents:
    """Jointly normalized phase responses and their fixed coordinates.

    Every tensor has shape ``[..., C]`` except ``debiased`` and ``normalized``,
    which have shape ``[..., 3, C]`` in the order ``[-1, 0, +1]``.
    """

    debiased: Tensor
    normalized: Tensor
    dc: Tensor
    odd: Tensor
    curvature: Tensor
    left_difference: Tensor
    right_difference: Tensor
    raw_left: Tensor
    raw_right: Tensor


@dataclass(frozen=True)
class OPCRP4DForward:
    """Optional audit trace around the exact P4D temporal-max insertion."""

    xyz: Tensor
    raw_phase_responses: Tensor
    official_features: Tensor
    residual: Tensor
    adapted_features: Tensor


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _require_arm(arm: str) -> str:
    if arm not in OPCR_ARMS:
        raise OPCRContractError(f"unknown OPCR arm {arm!r}; expected one of {OPCR_ARMS}")
    return arm


def _require_adapted_arm(arm: str) -> str:
    _require_arm(arm)
    if arm == VANILLA:
        raise OPCRContractError("vanilla has no OPCR adapter")
    return arm


def _private_seed(backbone_seed: int, salt: int) -> int:
    if isinstance(backbone_seed, bool) or not isinstance(backbone_seed, int):
        raise OPCRContractError("backbone_seed must be a non-negative integer")
    if backbone_seed < 0:
        raise OPCRContractError("backbone_seed must be non-negative")
    return (
        INIT_SEED_BASE + INIT_SEED_STRIDE * backbone_seed + int(salt)
    ) % (2**63 - 1)


def _orthogonal_rows(rows: int, columns: int, *, seed: int) -> Tensor:
    """Private-RNG row-orthogonal float32 matrix with fixed QR signs."""

    if rows > columns:
        raise OPCRContractError("row-orthogonal initialization requires rows <= columns")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randn(columns, rows, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(raw, mode="reduced")
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q = q * signs.unsqueeze(0)
    return q.transpose(0, 1).contiguous()


def matched_adapter_state(backbone_seed: int) -> Mapping[str, Tensor]:
    """Return matched branch/output bytes without advancing global RNG."""

    return {
        "W_o.weight": _orthogonal_rows(
            BRANCH_DIM,
            TOKEN_DIM,
            seed=_private_seed(backbone_seed, INIT_ODD_SALT),
        ),
        "W_c.weight": _orthogonal_rows(
            BRANCH_DIM,
            TOKEN_DIM,
            seed=_private_seed(backbone_seed, INIT_CURVATURE_SALT),
        ),
        "W_up.weight": torch.zeros(TOKEN_DIM, FUSED_DIM, dtype=torch.float32),
    }


def phase_components(
    raw_phase_responses: Tensor,
    temporal_channel_weight: Tensor,
    *,
    eps: float = JOINT_LN_EPS,
) -> PhaseComponents:
    """Remove the P4D time-label constant and form fixed phase coordinates.

    Parameters
    ----------
    raw_phase_responses:
        Spatially max-pooled P4D responses with shape ``[..., 3, C]`` in
        chronological slot order ``[-1,0,+1]``.  The official temporal max
        must be computed from this *raw* stack, not from the returned values.
    temporal_channel_weight:
        ``conv_d.weight[:,3,0,0]`` with shape ``[C]``.  Because released
        ``conv_d`` is linear and bias-free, subtracting ``q*w_t`` after the
        spatial max exactly removes the constant time-label response.

    One no-affine LayerNorm is evaluated jointly across all ``3*C`` values of
    each anchor.  A shared location/scale therefore preserves inter-slot
    amplitudes, unlike three independent normalizations, and its statistics
    are invariant to slot reversal.
    """

    if not torch.is_floating_point(raw_phase_responses):
        raise OPCRContractError("raw_phase_responses must be floating point")
    if raw_phase_responses.ndim < 2 or raw_phase_responses.shape[-2] != TUBE_FRAMES:
        raise OPCRContractError(
            "raw_phase_responses must end in [3,C], got "
            + str(tuple(raw_phase_responses.shape))
        )
    channels = int(raw_phase_responses.shape[-1])
    if temporal_channel_weight.ndim != 1 or temporal_channel_weight.shape[0] != channels:
        raise OPCRContractError(
            f"temporal_channel_weight must have shape [{channels}], got "
            + str(tuple(temporal_channel_weight.shape))
        )
    if not torch.is_floating_point(temporal_channel_weight):
        raise OPCRContractError("temporal_channel_weight must be floating point")
    if not math.isfinite(float(eps)) or float(eps) <= 0:
        raise OPCRContractError("eps must be finite and positive")

    labels = raw_phase_responses.new_tensor(TIME_LABELS)
    label_shape = (1,) * (raw_phase_responses.ndim - 2) + (TUBE_FRAMES, 1)
    weight_shape = (1,) * (raw_phase_responses.ndim - 1) + (channels,)
    debiased = raw_phase_responses - labels.view(label_shape) * temporal_channel_weight.view(
        weight_shape
    )

    # Accumulate the joint statistics in float32 under AMP, then return the
    # original activation dtype for the released mixed-precision path.
    device_type = raw_phase_responses.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        normalized32 = F.layer_norm(
            debiased.float(),
            normalized_shape=(TUBE_FRAMES, channels),
            weight=None,
            bias=None,
            eps=float(eps),
        )
    normalized = normalized32.to(dtype=raw_phase_responses.dtype)
    minus, center, plus = normalized.unbind(dim=-2)

    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_3 = 1.0 / math.sqrt(3.0)
    inv_sqrt_6 = 1.0 / math.sqrt(6.0)
    dc = (minus + center + plus) * inv_sqrt_3
    odd = (plus - minus) * inv_sqrt_2
    curvature = (plus + minus - 2.0 * center) * inv_sqrt_6
    left_difference = center - minus
    right_difference = plus - center
    # Same-budget raw control: two overlapping half-tubes use all three raw
    # slots without the zero-sum or orthogonal phase prior.
    raw_left = (minus + center) * inv_sqrt_2
    raw_right = (center + plus) * inv_sqrt_2
    return PhaseComponents(
        debiased=debiased,
        normalized=normalized,
        dc=dc,
        odd=odd,
        curvature=curvature,
        left_difference=left_difference,
        right_difference=right_difference,
        raw_left=raw_left,
        raw_right=raw_right,
    )


class OPCRAdapter(nn.Module):
    """Two phase-specific nonlinear branches followed by a zero output map."""

    def __init__(self, *, arm: str, backbone_seed: int) -> None:
        super().__init__()
        self.arm = _require_adapted_arm(arm)

        # nn.Linear constructors initialize eagerly; restore the CPU stream so
        # constructing OPCR is observationally RNG-free to the official run.
        rng_state = torch.random.get_rng_state()
        try:
            self.W_o = nn.Linear(TOKEN_DIM, BRANCH_DIM, bias=False)
            self.W_c = nn.Linear(TOKEN_DIM, BRANCH_DIM, bias=False)
            self.W_up = nn.Linear(FUSED_DIM, TOKEN_DIM, bias=False)
        finally:
            torch.random.set_rng_state(rng_state)

        state = matched_adapter_state(backbone_seed)
        with torch.no_grad():
            self.W_o.weight.copy_(state["W_o.weight"])
            self.W_c.weight.copy_(state["W_c.weight"])
            self.W_up.weight.copy_(state["W_up.weight"])

        if self.arm == LEARNED_ZERO_SUM_BASIS:
            # Any 2x2 mixture of [O,C] remains in the zero-sum phase subspace.
            self.basis_mix = nn.Parameter(torch.eye(2, dtype=torch.float32))
        else:
            self.register_parameter("basis_mix", None)

    def branch_inputs(self, components: PhaseComponents) -> Tuple[Tensor, Tensor]:
        if components.odd.shape[-1] != TOKEN_DIM:
            raise OPCRContractError(
                f"OPCR branch inputs must have {TOKEN_DIM} channels, got "
                f"{components.odd.shape[-1]}"
            )
        if self.arm == PRIMARY:
            return components.odd, components.curvature
        if self.arm == EVENIZED:
            return components.odd.abs(), components.curvature
        if self.arm == RAW_TRIPLET:
            return components.raw_left, components.raw_right
        if self.arm == ADJACENT_DIFF:
            return components.left_difference, components.right_difference
        if self.arm == LEARNED_ZERO_SUM_BASIS:
            assert self.basis_mix is not None
            stacked = torch.stack((components.odd, components.curvature), dim=-2)
            mixed = torch.einsum("ij,...jc->...ic", self.basis_mix, stacked)
            return mixed.unbind(dim=-2)
        raise AssertionError("unreachable OPCR arm")

    def forward(self, components: PhaseComponents) -> Tensor:
        odd_input, curvature_input = self.branch_inputs(components)
        odd_hidden = F.gelu(self.W_o(odd_input))
        curvature_hidden = F.gelu(self.W_c(curvature_input))
        hidden = torch.cat((odd_hidden, curvature_hidden), dim=-1)
        return self.W_up(hidden)


def _resolve_pointnet_ops(source_p4d: nn.Module, explicit_ops: Optional[Any]) -> Any:
    if explicit_ops is not None:
        return explicit_ops
    module_name = source_p4d.__class__.__module__
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)
    ops = getattr(module, "pointnet2_utils", None)
    if ops is None:
        raise OPCRContractError(
            "could not resolve pointnet2_utils from source P4D; pass pointnet2_ops explicitly"
        )
    return ops


def _validate_p4d_identity(source: nn.Module, strict: bool, *, expected_radius: float = RADIUS) -> nn.Conv2d:
    if (type(expected_radius) not in (float, int) or not math.isfinite(expected_radius)
            or expected_radius <= 0):
        raise OPCRContractError("expected_radius must be finite and positive")
    required = (
        "in_planes",
        "r",
        "k",
        "spatial_stride",
        "temporal_kernel_size",
        "temporal_stride",
        "temporal_padding",
        "temporal_padding_mode",
        "operator",
        "spatial_pooling",
        "temporal_pooling",
        "conv_d",
        "mlp",
    )
    missing = [name for name in required if not hasattr(source, name)]
    if missing:
        raise OPCRContractError(f"source P4D module lacks {missing}")
    if int(source.in_planes) != 0:
        raise OPCRContractError("OPCR expects released coordinate-only P4D input")
    essential = (
        int(source.temporal_kernel_size),
        int(source.temporal_stride),
        tuple(int(value) for value in source.temporal_padding),
        str(source.temporal_padding_mode),
        str(source.operator),
        str(source.spatial_pooling),
        str(source.temporal_pooling),
    )
    expected_essential = (3, 2, TEMPORAL_PADDING, "replicate", "+", "max", "max")
    if essential != expected_essential:
        raise OPCRContractError(
            f"source P4D method mismatch: observed {essential}, expected {expected_essential}"
        )
    if not isinstance(source.conv_d, nn.Sequential) or len(source.conv_d) != 1:
        raise OPCRContractError("released conv_d must be a one-layer nn.Sequential")
    conv = source.conv_d[0]
    if not isinstance(conv, nn.Conv2d):
        raise OPCRContractError("released conv_d[0] must be nn.Conv2d")
    if (
        conv.in_channels != 4
        or conv.out_channels != TOKEN_DIM
        or conv.kernel_size != (1, 1)
        or conv.stride != (1, 1)
        or conv.padding != (0, 0)
        or conv.bias is not None
    ):
        raise OPCRContractError("released conv_d must be bias-free 4->80 1x1 Conv2d")
    if not isinstance(source.mlp, nn.Sequential) or len(source.mlp) != 0:
        raise OPCRContractError("released PST-small P4D mlp must be identity")
    if strict:
        observed = (
            float(source.r),
            int(source.k),
            int(source.spatial_stride),
        )
        expected = (float(expected_radius), TUBE_NEIGHBORS, SPATIAL_STRIDE)
        if observed != expected:
            raise OPCRContractError(
                f"source P4D geometry mismatch: observed {observed}, expected {expected}"
            )
    return conv








class _OPCRP4DConvWrapper(nn.Module):
    """Released P4D forward with OPCR inserted after its temporal max."""

    def __init__(
        self,
        source_p4d: nn.Module,
        *,
        arm: str,
        backbone_seed: int,
        pointnet2_ops: Optional[Any] = None,
        strict: bool = True,
        expected_radius: float = RADIUS,
    ) -> None:
        super().__init__()
        _validate_p4d_identity(source_p4d, strict, expected_radius=expected_radius)
        self.strict = bool(strict)
        self.arm = _require_adapted_arm(arm)
        self.in_planes = int(source_p4d.in_planes)
        self.r = float(source_p4d.r)
        self.k = int(source_p4d.k)
        self.spatial_stride = int(source_p4d.spatial_stride)
        self.temporal_kernel_size = int(source_p4d.temporal_kernel_size)
        self.temporal_stride = int(source_p4d.temporal_stride)
        self.temporal_padding = tuple(int(value) for value in source_p4d.temporal_padding)
        self.temporal_padding_mode = str(source_p4d.temporal_padding_mode)
        self.operator = str(source_p4d.operator)
        self.spatial_pooling = str(source_p4d.spatial_pooling)
        self.temporal_pooling = str(source_p4d.temporal_pooling)
        self.conv_d = source_p4d.conv_d
        self.mlp = source_p4d.mlp
        self.adapter = OPCRAdapter(arm=self.arm, backbone_seed=backbone_seed)
        self.pointnet2_ops = _resolve_pointnet_ops(source_p4d, pointnet2_ops)

    @property
    def temporal_channel_weight(self) -> Tensor:
        return self.conv_d[0].weight[:, 3, 0, 0]

    def _forward_impl(
        self,
        xyzs: Tensor,
        features: Optional[Tensor],
        *,
        capture_trace: bool,
    ) -> Tuple[Tensor, Tensor, Optional[OPCRP4DForward]]:
        if xyzs.ndim != 4 or xyzs.shape[-1] != 3 or not torch.is_floating_point(xyzs):
            raise OPCRContractError("P4D input must be floating [B,T,N,3]")
        if features is not None:
            raise OPCRContractError("released coordinate-only PST-small does not accept point features")
        batch, nframes, npoints, _ = xyzs.shape
        if self.strict and (nframes, npoints) != (INPUT_FRAMES, INPUT_POINTS):
            raise OPCRContractError(
                f"official PST-small input must be [B,{INPUT_FRAMES},{INPUT_POINTS},3]"
            )
        if (
            nframes + sum(self.temporal_padding) - self.temporal_kernel_size
        ) % self.temporal_stride != 0:
            raise OPCRContractError("P4D temporal length is incompatible with stride")

        frames = [frame.squeeze(1).contiguous() for frame in torch.split(xyzs, 1, dim=1)]
        frames = [frames[0]] * self.temporal_padding[0] + frames
        frames = frames + [frames[-1]] * self.temporal_padding[1]

        output_xyz = []
        output_official = []
        output_residual = []
        output_adapted = []
        output_raw_phase = []
        half = self.temporal_kernel_size // 2
        for temporal_anchor in range(half, len(frames) - half, self.temporal_stride):
            anchor_idx = self.pointnet2_ops.furthest_point_sample(
                frames[temporal_anchor], npoints // self.spatial_stride
            )
            anchor_flipped = self.pointnet2_ops.gather_operation(
                frames[temporal_anchor].transpose(1, 2).contiguous(), anchor_idx
            )
            anchor_expanded = anchor_flipped.unsqueeze(3)
            anchor_xyz = anchor_flipped.transpose(1, 2).contiguous()

            temporal_features = []
            for neighbor_time in range(temporal_anchor - half, temporal_anchor + half + 1):
                neighbor_xyz = frames[neighbor_time]
                idx = self.pointnet2_ops.ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)
                grouped = self.pointnet2_ops.grouping_operation(
                    neighbor_xyz.transpose(1, 2).contiguous(), idx
                )
                dxyz = grouped - anchor_expanded
                dt = torch.ones(
                    (batch, 1, dxyz.shape[2], dxyz.shape[3]),
                    dtype=torch.float32,
                    device=xyzs.device,
                ) * float(neighbor_time - temporal_anchor)
                response = self.mlp(self.conv_d(torch.cat((dxyz, dt), dim=1)))
                # Frozen released spatial max: [B,C,N_out].
                response = torch.max(response, dim=-1, keepdim=False)[0]
                temporal_features.append(response)

            # Frozen released temporal max remains on the untouched responses.
            raw_stack = torch.stack(temporal_features, dim=1)  # [B,3,C,N]
            official = torch.max(raw_stack, dim=1, keepdim=False)[0]
            canonical_raw = raw_stack.permute(0, 3, 1, 2).contiguous()  # [B,N,3,C]
            components = phase_components(canonical_raw, self.temporal_channel_weight)
            residual = self.adapter(components).permute(0, 2, 1).contiguous()
            adapted = official + residual

            output_xyz.append(anchor_xyz)
            output_official.append(official)
            output_residual.append(residual)
            output_adapted.append(adapted)
            if capture_trace:
                output_raw_phase.append(canonical_raw)

        xyz = torch.stack(output_xyz, dim=1)
        official_features = torch.stack(output_official, dim=1)
        residual = torch.stack(output_residual, dim=1)
        adapted_features = torch.stack(output_adapted, dim=1)

        raw_phase: Optional[Tensor] = None
        if capture_trace:
            raw_phase = torch.stack(output_raw_phase, dim=1)
        trace = None
        if capture_trace:
            assert raw_phase is not None
            trace = OPCRP4DForward(
                xyz=xyz,
                raw_phase_responses=raw_phase,
                official_features=official_features,
                residual=residual,
                adapted_features=adapted_features,
            )
        return xyz, adapted_features, trace

    def forward(
        self, xyzs: Tensor, features: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        xyz, adapted, _ = self._forward_impl(xyzs, features, capture_trace=False)
        return xyz, adapted

    def forward_trace(
        self, xyzs: Tensor, features: Optional[Tensor] = None
    ) -> OPCRP4DForward:
        _, _, trace = self._forward_impl(xyzs, features, capture_trace=True)
        assert trace is not None
        return trace



class EBTEOPCRContractError(RuntimeError):
    """A phase-energy budget, shape, or frozen integration contract failed."""


def _require_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise EBTEOPCRContractError(f"{name} must contain only finite values")


def _stable_radial_scale(residual: Tensor, phase_norm: Tensor) -> Tensor:
    """Evaluate ``phase/hypot(phase, ||residual||)`` without large squares."""

    maximum = residual.abs().amax(dim=-1, keepdim=True)
    nonzero = maximum > 0
    safe_maximum = torch.where(nonzero, maximum, torch.ones_like(maximum))
    normalized = residual / safe_maximum
    normalized_square_sum = normalized.square().sum(dim=-1, keepdim=True)
    normalized_norm = torch.sqrt(
        torch.where(
            nonzero,
            normalized_square_sum,
            torch.ones_like(normalized_square_sum),
        )
    )

    # Choose the ratio whose numerator is no larger than its denominator.
    # Both branches are algebraically identical to the public radial formula,
    # but neither forms maximum*normalized_norm for the unbounded input.
    maximum_dominates = maximum >= phase_norm
    dominant = torch.where(maximum_dominates, safe_maximum, phase_norm)
    phase_ratio = phase_norm / dominant
    residual_ratio = maximum / dominant
    large_scale = phase_ratio / torch.hypot(phase_ratio, normalized_norm)
    small_scale = 1.0 / torch.hypot(
        torch.ones_like(residual_ratio),
        residual_ratio * normalized_norm,
    )
    return torch.where(
        nonzero,
        torch.where(maximum_dominates, large_scale, small_scale),
        torch.ones_like(maximum),
    )


def _stable_vector_norm(value: Tensor) -> Tensor:
    """L2 norm for an already bounded vector without square-sum overflow."""

    maximum = value.abs().amax(dim=-1, keepdim=True)
    nonzero = maximum > 0
    safe_maximum = torch.where(nonzero, maximum, torch.ones_like(maximum))
    normalized = value / safe_maximum
    normalized_square_sum = normalized.square().sum(dim=-1, keepdim=True)
    normalized_norm = torch.sqrt(
        torch.where(
            nonzero,
            normalized_square_sum,
            torch.ones_like(normalized_square_sum),
        )
    )
    return torch.where(
        nonzero,
        maximum * normalized_norm,
        torch.zeros_like(maximum),
    )


@dataclass(frozen=True)
class RawPhaseEvidence:
    """Raw de-biased phase coordinates and their local squared energy."""

    odd: Tensor
    curvature: Tensor
    local_energy: Tensor


@dataclass(frozen=True)
class BoundedResidual:
    """Canonical ``[..., C]`` residual before and after energy bounding."""

    unbounded: Tensor
    bounded: Tensor
    local_energy: Tensor
    applied_budget: Tensor
    scale: Tensor


@dataclass(frozen=True)
class EBTEOPCRP4DForward:
    """Audit trace around the bounded pre-Transformer OPCR insertion."""

    xyz: Tensor
    raw_phase_responses: Tensor
    official_features: Tensor
    unbounded_residual: Tensor
    local_phase_energy: Tensor
    applied_energy_budget: Tensor
    residual_scale: Tensor
    residual: Tensor
    adapted_features: Tensor


def _require_energy_mode(mode: str) -> str:
    if mode not in ENERGY_MODES:
        raise EBTEOPCRContractError(
            f"unknown phase-energy mode {mode!r}; expected one of {ENERGY_MODES}"
        )
    return mode


def raw_phase_evidence(
    raw_phase_responses: Tensor,
    temporal_channel_weight: Tensor,
) -> RawPhaseEvidence:
    """Return float32 raw non-DC coordinates after exact time-label removal.

    ``raw_phase_responses`` ends in ``[3,C]`` in chronological order.  The
    squared energy ``||O_raw||^2 + ||K_raw||^2`` has shape ``[...,1]`` and is
    accumulated in float32 under AMP.  It is deliberately measured before
    OPCR's joint LayerNorm so weak raw evidence cannot acquire an unrestricted
    learned residual merely through normalization.  Its use as a downstream
    budget is stop-gradient; the returned evidence helper itself remains a
    differentiable measurement for isolated analysis.
    """

    if not torch.is_floating_point(raw_phase_responses):
        raise EBTEOPCRContractError("raw_phase_responses must be floating point")
    if raw_phase_responses.ndim < 2 or raw_phase_responses.shape[-2] != TUBE_FRAMES:
        raise EBTEOPCRContractError(
            "raw_phase_responses must end in [3,C], got "
            + str(tuple(raw_phase_responses.shape))
        )
    channels = int(raw_phase_responses.shape[-1])
    if temporal_channel_weight.ndim != 1 or temporal_channel_weight.shape[0] != channels:
        raise EBTEOPCRContractError(
            f"temporal_channel_weight must have shape [{channels}], got "
            + str(tuple(temporal_channel_weight.shape))
        )
    if not torch.is_floating_point(temporal_channel_weight):
        raise EBTEOPCRContractError("temporal_channel_weight must be floating point")
    _require_finite("raw_phase_responses", raw_phase_responses)
    _require_finite("temporal_channel_weight", temporal_channel_weight)

    device_type = raw_phase_responses.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        raw = raw_phase_responses.float()
        weight = temporal_channel_weight.float()
        labels = raw.new_tensor(TIME_LABELS)
        label_shape = (1,) * (raw.ndim - 2) + (TUBE_FRAMES, 1)
        weight_shape = (1,) * (raw.ndim - 1) + (channels,)
        debiased = raw - labels.view(label_shape) * weight.view(weight_shape)
        minus, center, plus = debiased.unbind(dim=-2)
        odd = (plus - minus) * (1.0 / math.sqrt(2.0))
        curvature = (plus + minus - 2.0 * center) * (1.0 / math.sqrt(6.0))
        local_energy = (
            odd.square().sum(dim=-1, keepdim=True)
            + curvature.square().sum(dim=-1, keepdim=True)
        )
    _require_finite("raw odd phase coordinate", odd)
    _require_finite("raw curvature phase coordinate", curvature)
    _require_finite("raw non-DC phase energy", local_energy)

    return RawPhaseEvidence(
        odd=odd,
        curvature=curvature,
        local_energy=local_energy,
    )


def global_same_total_budget(local_energy: Tensor) -> Tensor:
    """Broadcast the per-sample mean with mathematically equal total budget.

    In real arithmetic, summing the returned budget over all token axes equals
    summing ``local_energy`` over those axes.  Finite-precision reductions and
    broadcast summation can differ by rounding error.  The final singleton
    feature axis is retained.
    """

    if not torch.is_floating_point(local_energy):
        raise EBTEOPCRContractError("local_energy must be floating point")
    if local_energy.ndim < 2 or local_energy.shape[-1] != 1:
        raise EBTEOPCRContractError(
            "local_energy must have shape [B,...,1], got "
            + str(tuple(local_energy.shape))
        )
    _require_finite("local_energy", local_energy)
    if bool(torch.any(local_energy < 0).item()):
        raise EBTEOPCRContractError("local_energy must be non-negative")
    token_axes = tuple(range(1, local_energy.ndim - 1))
    if not token_axes:
        return local_energy
    mean = local_energy.mean(dim=token_axes, keepdim=True)
    result = mean.expand_as(local_energy)
    _require_finite("global phase-energy budget", result)
    return result


def bound_residual_with_phase_energy(
    residual: Tensor,
    local_energy: Tensor,
    *,
    mode: str,
) -> BoundedResidual:
    """Radially squash a learned residual into its phase-energy trust region.

    For applied budget ``B`` and residual ``r`` the scale is

    ``sqrt(B / (B + ||r||^2))``.

    A zero budget always returns an exact zero residual, including when the
    residual is so small that squaring it would underflow.  Positive float32
    subnormal budgets are also suppressed: that range cannot carry a reliable
    relative summation margin.  Normal-budget computation is performed in
    float32 and then rechecked after conversion to the requested output dtype.
    Any outward low-precision rounding is conservatively rescaled; a still-
    unsafe token is set to exact zero.
    """

    mode = _require_energy_mode(mode)
    if not torch.is_floating_point(residual):
        raise EBTEOPCRContractError("residual must be floating point")
    if residual.ndim < 2:
        raise EBTEOPCRContractError("residual must have shape [B,...,C]")
    if local_energy.shape != residual.shape[:-1] + (1,):
        raise EBTEOPCRContractError(
            "local_energy must match residual tokens with a singleton channel: "
            f"residual={tuple(residual.shape)}, energy={tuple(local_energy.shape)}"
        )
    if not torch.is_floating_point(local_energy):
        raise EBTEOPCRContractError("local_energy must be floating point")
    _require_finite("residual", residual)
    _require_finite("local_energy", local_energy)
    if bool(torch.any(local_energy < 0).item()):
        raise EBTEOPCRContractError("local_energy must be non-negative")

    device_type = residual.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        residual32 = residual.float()
        # The raw evidence is a trust-region measurement, not an incentive.
        # Stop its gradient so the classification loss cannot enlarge P4D
        # energy merely to purchase a larger residual budget.  The learned
        # residual itself remains fully differentiable.
        local32 = local_energy.float().detach()
        budget = (
            local32
            if mode == LOCAL_PHASE_ENERGY
            else global_same_total_budget(local32)
        )
        _require_finite("applied phase-energy budget", budget)
        reliable_budget = budget >= torch.finfo(torch.float32).tiny
        # Never evaluate sqrt at a zero-ratio active branch: its formal
        # derivative is singular and 0*inf can yield NaNs in backward.  The
        # inactive surrogate is finite, but the public zero-budget result is
        # unconditionally overwritten with exact zeros below.
        safe_budget = torch.where(reliable_budget, budget, torch.ones_like(budget))
        phase_norm = torch.sqrt(safe_budget)
        # This maximum-scaled form is algebraically identical to
        # sqrt(B/(B+||r||^2)), but avoids both ratio underflow and the
        # square-sum overflow that can poison masked backward paths.
        positive_scale = _stable_radial_scale(residual32, phase_norm)
        scale = torch.where(
            reliable_budget, positive_scale, torch.zeros_like(positive_scale)
        )
        bounded = (residual32 * scale).to(dtype=residual.dtype)
        bounded = torch.where(reliable_budget, bounded, torch.zeros_like(bounded))

        # Casting an in-bound float32 vector to fp16/bf16 can round individual
        # components outward.  In addition, a float32 sum of positive squares
        # can round down: an operational result equal to B can therefore have
        # a slightly larger real-valued norm.  Reserve more than twice the
        # standard gamma_n error bound for C positive float32 additions, then
        # re-evaluate the returned dtype against that conservative threshold.
        # This makes the representable output's mathematical squared norm no
        # larger than B, rather than merely passing the same float32 reduction.
        channels = int(residual.shape[-1])
        unit_roundoff = torch.finfo(torch.float32).eps / 2.0
        addition_load = channels * unit_roundoff
        if addition_load >= 0.25:
            raise EBTEOPCRContractError(
                "residual channel count is too large for the float32 sum bound"
            )
        gamma_n = addition_load / (1.0 - addition_load)
        verification_margin = (
            2.0 * gamma_n + 2.0 * torch.finfo(torch.float32).eps
        )
        verification_budget = budget * (1.0 - verification_margin)

        # Two passes handle a second outward target-dtype cast; exact zero is
        # the fail-safe for a token that remains over the conservative budget
        # (notably near the target dtype's subnormal floor).
        # ``nextafter`` has no autograd implementation in PyTorch.  It is a
        # target-dtype safety threshold, not part of the analytical radial
        # map, so compute it from a detached view.  The normal analytical path
        # (and its budget gradient) remains untouched unless a cast violates
        # the public bound.
        detached_budget = verification_budget.detach()
        conservative_budget = torch.nextafter(
            detached_budget, torch.zeros_like(detached_budget)
        )
        conservative_radius = torch.sqrt(
            torch.where(
                reliable_budget,
                conservative_budget,
                torch.zeros_like(conservative_budget),
            )
        )
        target_rounding_guard = max(
            0.0,
            1.0 - 2.0 * torch.finfo(residual.dtype).eps,
        )
        conservative_radius = conservative_radius * target_rounding_guard
        for _ in range(2):
            returned32 = bounded.float()
            returned_energy = returned32.square().sum(dim=-1, keepdim=True)
            over = reliable_budget & (returned_energy > verification_budget)
            returned_norm = _stable_vector_norm(returned32)
            safe_norm = torch.where(over, returned_norm, torch.ones_like(returned_norm))
            correction = torch.where(
                over,
                conservative_radius / safe_norm,
                torch.ones_like(safe_norm),
            )
            bounded = torch.where(
                over,
                (returned32 * correction).to(dtype=residual.dtype),
                bounded,
            )
            scale = scale * correction
        final_energy = bounded.float().square().sum(dim=-1, keepdim=True)
        unsafe = reliable_budget & (final_energy > verification_budget)
        # Halving a finite binary floating-point value is exact away from the
        # subnormal floor and preserves a useful gradient.  It is a final
        # representable projection, not a hard gate for ordinary budgets.
        half = torch.where(
            unsafe,
            torch.full_like(scale, 0.5),
            torch.ones_like(scale),
        )
        bounded = torch.where(
            unsafe,
            (bounded.float() * half).to(dtype=residual.dtype),
            bounded,
        )
        scale = scale * half
        final_energy = bounded.float().square().sum(dim=-1, keepdim=True)
        unrepresentable = reliable_budget & (final_energy > verification_budget)
        bounded = torch.where(unrepresentable, torch.zeros_like(bounded), bounded)
        scale = torch.where(unrepresentable, torch.zeros_like(scale), scale)

    _require_finite("bounded residual", bounded)
    _require_finite("residual scale", scale)

    return BoundedResidual(
        unbounded=residual,
        bounded=bounded,
        local_energy=local32,
        applied_budget=budget,
        scale=scale,
    )


def phase_energy_bounded_residual(
    raw_phase_responses: Tensor,
    temporal_channel_weight: Tensor,
    residual: Tensor,
    *,
    mode: str,
) -> BoundedResidual:
    """Measure raw non-DC evidence and bound a canonical learned residual."""

    evidence = raw_phase_evidence(raw_phase_responses, temporal_channel_weight)
    return bound_residual_with_phase_energy(
        residual,
        evidence.local_energy,
        mode=mode,
    )


class EBTE(_OPCRP4DConvWrapper):
    """Frozen primary OPCR with only its final residual addition bounded."""

    def __init__(
        self,
        source_p4d: nn.Module,
        *,
        energy_mode: str = LOCAL_PHASE_ENERGY,
        backbone_seed: int,
        pointnet2_ops: Optional[Any] = None,
        strict: bool = True,
        expected_radius: float = RADIUS,
    ) -> None:
        self.energy_mode = _require_energy_mode(energy_mode)
        super().__init__(
            source_p4d,
            arm=PRIMARY,
            backbone_seed=backbone_seed,
            pointnet2_ops=pointnet2_ops,
            strict=strict,
            expected_radius=expected_radius,
        )

    def _forward_impl(
        self,
        xyzs: Tensor,
        features: Optional[Tensor],
        *,
        capture_trace: bool,
    ) -> Tuple[Tensor, Tensor, Optional[EBTEOPCRP4DForward]]:
        # Capture the frozen core's exact raw responses and unbounded adapter
        # output, then replace only its final official+residual addition.
        xyz, _, trace = super()._forward_impl(xyzs, features, capture_trace=True)
        assert trace is not None
        canonical_residual = trace.residual.permute(0, 1, 3, 2).contiguous()
        bounded = phase_energy_bounded_residual(
            trace.raw_phase_responses,
            self.temporal_channel_weight,
            canonical_residual,
            mode=self.energy_mode,
        )
        residual = bounded.bounded.permute(0, 1, 3, 2).contiguous()
        adapted = trace.official_features + residual
        ebte_trace = None
        if capture_trace:
            ebte_trace = EBTEOPCRP4DForward(
                xyz=xyz,
                raw_phase_responses=trace.raw_phase_responses,
                official_features=trace.official_features,
                unbounded_residual=trace.residual,
                local_phase_energy=bounded.local_energy,
                applied_energy_budget=bounded.applied_budget,
                residual_scale=bounded.scale,
                residual=residual,
                adapted_features=adapted,
            )
        return xyz, adapted, ebte_trace

    def forward(
        self, xyzs: Tensor, features: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        xyz, adapted, _ = self._forward_impl(xyzs, features, capture_trace=False)
        return xyz, adapted

    def forward_trace(
        self, xyzs: Tensor, features: Optional[Tensor] = None
    ) -> EBTEOPCRP4DForward:
        _, _, trace = self._forward_impl(xyzs, features, capture_trace=True)
        assert trace is not None
        return trace



# SYNTHIA feature-stream EBTE adapters. MSR/NTU coordinate EBTE above is unchanged.
CONV3_TEMPORAL_KERNEL = 3
CONV3_TEMPORAL_STRIDE = 1
CONV3_TEMPORAL_PADDING = (1, 1)
CONV3_TEMPORAL_PADDING_MODE = "replicate"
CONV3_TEMPORAL_POOLING = ("sum", "max")
CONV3_OPERATOR = "+"
CONV3_SPATIAL_POOLING = "max"


def _feature_require(condition, message):
    if not condition:
        raise ValueError(message)


def _feature_orthogonal_rows(rows: int, columns: int, generator: torch.Generator) -> Tensor:
    raw = torch.randn(columns, rows, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(raw, mode="reduced")
    signs = torch.where(torch.diagonal(r) < 0, -torch.ones_like(torch.diagonal(r)),
                        torch.ones_like(torch.diagonal(r)))
    return (q * signs.unsqueeze(0)).transpose(0, 1).contiguous()


class _Conv3EBTEAdapter(nn.Module):
    """Row-orthogonal branch maps + zero-initialized output, mirroring OPCR."""

    def __init__(self, channels: int, branch_dim: int, seed: int):
        super().__init__()
        _feature_require(type(channels) is int and type(branch_dim) is int
                         and 1 <= branch_dim <= channels, "require 1 <= branch_dim <= channels")
        rng_state = torch.random.get_rng_state()
        try:
            self.W_o = nn.Linear(channels, branch_dim, bias=False)
            self.W_c = nn.Linear(channels, branch_dim, bias=False)
            self.W_up = nn.Linear(2 * branch_dim, channels, bias=False)
        finally:
            torch.random.set_rng_state(rng_state)
        generator = torch.Generator(device="cpu")
        generator.manual_seed((int(seed) + 31873) % (2**63 - 1))
        with torch.no_grad():
            self.W_o.weight.copy_(_feature_orthogonal_rows(branch_dim, channels, generator))
            self.W_c.weight.copy_(_feature_orthogonal_rows(branch_dim, channels, generator))
            nn.init.zeros_(self.W_up.weight)

    def forward(self, odd: Tensor, curvature: Tensor) -> Tensor:
        hidden = torch.cat((F.gelu(self.W_o(odd)), F.gelu(self.W_c(curvature))), dim=-1)
        return self.W_up(hidden)


class Conv3WithEBTE(nn.Module):
    """P4DConv conv3 with an energy-bounded phase residual on the temporal stack.

    Reimplements the 3-slot temporal sweep using the source's conv_d/conv_f/mlp
    children so the state-dict keys ``conv3.conv_d.*`` / ``conv3.conv_f.*`` /
    ``conv3.mlp.*`` remain identical to the plain P4DConv. The only added
    parameters are ``conv3.adapter.W_o/W_c/W_up``.

    Phase responses here are the post-BN/ReLU/MLP neighborhood messages. Their
    odd/curvature contrasts use a zero temporal reference of the final output
    width. Subtracting conv_d's input-time column would be both dimensionally
    wrong (256 vs. 512 by default) and invalid after the nonlinear encoder.
    Thus this is feature-phase EBTE, not MSR/NTU's exact linear-time debiasing.
    """

    def __init__(self, source: nn.Module, *, pointnet2_ops, seed: int = 0,
                 branch_dim: int = 10, energy_mode: str = LOCAL_PHASE_ENERGY):
        super().__init__()
        _feature_require(int(source.temporal_kernel_size) == CONV3_TEMPORAL_KERNEL,
                f"Conv3WithEBTE requires temporal_kernel_size={CONV3_TEMPORAL_KERNEL}")
        _feature_require(int(source.temporal_stride) == CONV3_TEMPORAL_STRIDE,
                f"Conv3WithEBTE requires temporal_stride={CONV3_TEMPORAL_STRIDE}")
        _feature_require(tuple(int(v) for v in source.temporal_padding) == CONV3_TEMPORAL_PADDING,
                f"Conv3WithEBTE requires temporal_padding={CONV3_TEMPORAL_PADDING}")
        _feature_require(str(source.temporal_padding_mode) == CONV3_TEMPORAL_PADDING_MODE,
                f"Conv3WithEBTE requires temporal_padding_mode={CONV3_TEMPORAL_PADDING_MODE!r}")
        _feature_require(str(source.operator) == CONV3_OPERATOR,
                f"Conv3WithEBTE requires operator={CONV3_OPERATOR!r}")
        _feature_require(str(source.spatial_pooling) == CONV3_SPATIAL_POOLING,
                f"Conv3WithEBTE requires spatial_pooling={CONV3_SPATIAL_POOLING!r}")
        _feature_require(str(source.temporal_pooling) in CONV3_TEMPORAL_POOLING,
                f"Conv3WithEBTE requires temporal_pooling in {CONV3_TEMPORAL_POOLING}")
        _feature_require(int(source.temporal_kernel_size) % 2 == 1,
                "temporal_kernel_size must be odd for symmetric padding")

        self.r = float(source.r)
        self.k = int(source.k)
        self.spatial_stride = int(source.spatial_stride)
        self.temporal_kernel_size = int(source.temporal_kernel_size)
        self.temporal_stride = int(source.temporal_stride)
        self.temporal_padding = tuple(int(v) for v in source.temporal_padding)
        self.temporal_padding_mode = str(source.temporal_padding_mode)
        self.temporal_pooling = str(source.temporal_pooling)
        self.operator = str(source.operator)
        self.spatial_pooling = str(source.spatial_pooling)
        self.in_planes = int(source.in_planes)

        # Preserve original children so state_dict parent keys are identical.
        self.conv_d = source.conv_d
        self.conv_f = source.conv_f
        self.mlp = source.mlp

        self.pointnet2_ops = pointnet2_ops
        self.energy_mode = energy_mode

        mlp_planes = getattr(source, "mlp_planes", None)
        channels_out = int(mlp_planes[-1]) if mlp_planes else int(self.conv_d[-1].out_channels)
        self.adapter = _Conv3EBTEAdapter(channels_out, branch_dim, seed)
        # Nonlinear BN/ReLU/MLP output has no exact linear temporal-bias vector.
        self.register_buffer("_zero_time_weight", torch.zeros(channels_out), persistent=False)

    @property
    def temporal_channel_weight(self) -> Tensor:
        """Feature-phase reference after the nonlinear encoder, with output width."""
        return self._zero_time_weight

    def _spatial_message(self, neighbor_xyz: Tensor, neighbor_features: Tensor | None,
                         anchor_xyz_expanded: Tensor, temporal_offset: int) -> Tensor:
        point_ops = self.pointnet2_ops
        anchor_xyz = anchor_xyz_expanded.squeeze(-1).transpose(1, 2).contiguous()
        indices = point_ops.ball_query(self.r, self.k, neighbor_xyz.contiguous(), anchor_xyz).int().contiguous()
        neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()
        neighbor_xyz_grouped = point_ops.grouping_operation(neighbor_xyz_flipped, indices)
        xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded

        batch, _, anchors, neighbors = xyz_displacement.shape
        temporal = torch.full((batch, 1, anchors, neighbors), float(temporal_offset),
                              dtype=torch.float32, device=neighbor_xyz.device)
        displacement = torch.cat((xyz_displacement, temporal), dim=1)
        displacement = self.conv_d(displacement)

        if self.in_planes > 0 and neighbor_features is not None:
            feature = point_ops.grouping_operation(neighbor_features.contiguous(), indices)
            feature = self.conv_f(feature) + displacement
        else:
            feature = displacement

        feature = self.mlp(feature)
        return torch.max(feature, dim=-1, keepdim=False)[0]  # spatial max -> [B, C, A]

    def forward(self, xyzs: Tensor, features: Tensor | None = None) -> tuple[Tensor, Tensor]:
        _feature_require(xyzs.ndim == 4 and xyzs.shape[-1] == 3, "Expected XYZ [B,T,N,3]")
        if self.in_planes > 0:
            _feature_require(features is not None and features.ndim == 4,
                    "Feature stream required when in_planes > 0")
            _feature_require(features.shape[0] == xyzs.shape[0]
                    and features.shape[1] == xyzs.shape[1]
                    and features.shape[3] == xyzs.shape[2],
                    "Feature stream shape must be [B,T,C,N] matching XYZ")

        batch, nframes, npoints, _ = xyzs.shape
        half = self.temporal_kernel_size // 2

        frames = [frame.squeeze(1).contiguous() for frame in torch.split(xyzs, 1, dim=1)]
        frames = [frames[0]] * self.temporal_padding[0] + frames \
            + [frames[-1]] * self.temporal_padding[1]
        if self.in_planes > 0:
            values = [frame.squeeze(1).contiguous() for frame in torch.split(features, 1, dim=1)]
            values = [values[0]] * self.temporal_padding[0] + values \
                + [values[-1]] * self.temporal_padding[1]
        else:
            values = [None] * len(frames)

        out_xyzs, out_features = [], []
        for t in range(half, len(frames) - half, self.temporal_stride):
            anchor_idx = self.pointnet2_ops.furthest_point_sample(
                frames[t], npoints // self.spatial_stride).int().contiguous()
            anchor_flipped = self.pointnet2_ops.gather_operation(
                frames[t].transpose(1, 2).contiguous(), anchor_idx)
            anchor_expanded = anchor_flipped.unsqueeze(3)
            anchor_xyz = anchor_flipped.transpose(1, 2).contiguous()

            slot_msgs = [
                self._spatial_message(frames[nt], values[nt], anchor_expanded, nt - t)
                for nt in range(t - half, t + half + 1)
            ]
            raw_stack = torch.stack(slot_msgs, dim=1)                       # [B,3,C,A]
            if self.temporal_pooling == "max":
                official = torch.max(raw_stack, dim=1, keepdim=False)[0]
            else:
                official = torch.sum(raw_stack, dim=1, keepdim=False)

            canonical = raw_stack.permute(0, 3, 1, 2).contiguous()          # [B,A,3,C]
            components = phase_components(canonical, self.temporal_channel_weight)
            residual = self.adapter(components.odd, components.curvature)   # [B,A,C]
            bounded = phase_energy_bounded_residual(
                canonical, self.temporal_channel_weight, residual, mode=self.energy_mode)
            delta = bounded.bounded.permute(0, 2, 1).contiguous()           # [B,C,A]

            out_xyzs.append(anchor_xyz)
            out_features.append(official + delta)

        return torch.stack(out_xyzs, dim=1), torch.stack(out_features, dim=1)


class BottleneckTrendEncoding(nn.Module):
    """Energy-bounded temporal-phase adapter on the L=3 bottleneck tokens.

    Reuses modules/ebte.py's phase_components + phase_energy_bounded_residual on
    the bottleneck token slots; mirrors the OPCR adapter contract.
    Cross-frame slots share FPS rank only: this inherited extension does not
    establish physical point correspondences or tracked temporal trajectories.
    """

    def __init__(self, channels: int, branch_dim: int = 10, seed: int = 0):
        super().__init__()
        _feature_require(type(channels) is int and type(branch_dim) is int
                         and 1 <= branch_dim <= channels, "require 1 <= branch_dim <= channels")
        self.channels, self.branch_dim = channels, branch_dim
        state = torch.random.get_rng_state()
        try:
            self.W_o = nn.Linear(channels, branch_dim, bias=False)
            self.W_c = nn.Linear(channels, branch_dim, bias=False)
            self.W_up = nn.Linear(2 * branch_dim, channels, bias=False)
        finally:
            torch.random.set_rng_state(state)
        generator = torch.Generator(device="cpu")
        generator.manual_seed((int(seed) + 31873) % (2**63 - 1))
        with torch.no_grad():
            self.W_o.weight.copy_(_feature_orthogonal_rows(branch_dim, channels, generator))
            self.W_c.weight.copy_(_feature_orthogonal_rows(branch_dim, channels, generator))
            nn.init.zeros_(self.W_up.weight)
        self.register_buffer("_zero_time_weight", torch.zeros(channels), persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 4 or tokens.shape[1] != 3 or tokens.shape[-1] != self.channels:
            raise ValueError(f"expected bottleneck tokens [B,3,N,{self.channels}], "
                             f"got {tuple(tokens.shape)}")
        # These slots share FPS rank, not tracked physical point identities.
        slots = tokens.permute(0, 2, 1, 3).contiguous()          # [B, N, 3, C]
        components = phase_components(slots, self._zero_time_weight)
        hidden = torch.cat((F.gelu(self.W_o(components.odd)),
                            F.gelu(self.W_c(components.curvature))), dim=-1)
        residual = self.W_up(hidden)                              # [B, N, C]
        bounded = phase_energy_bounded_residual(
            slots, self._zero_time_weight, residual, mode=LOCAL_PHASE_ENERGY)
        delta = bounded.bounded.unsqueeze(1)                      # [B, 1, N, C]
        return tokens + delta.expand_as(tokens)


