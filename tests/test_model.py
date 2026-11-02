"""CPU graph tests. The small deterministic neighborhoods are not accuracy tests."""
from itertools import product
from pathlib import Path

import pytest
import torch

from config.switches import ModuleSwitches
from model.backbone import MoTrend, build_model, bounded_relation_residual
from model.srr import SpatialRelationRefinement, _same_frame_neighbors


class SmallOps:
    """Small deterministic neighborhoods to exercise the full model on CPU."""
    @staticmethod
    def furthest_point_sample(xyz, npoint):
        return torch.arange(4, device=xyz.device, dtype=torch.int32)[None].expand(xyz.shape[0], -1).contiguous()

    @staticmethod
    def gather_operation(features, indices):
        return features.gather(2, indices.long()[:, None, :].expand(-1, features.shape[1], -1))

    @staticmethod
    def ball_query(radius, neighbors, xyz, centers):
        return torch.arange(neighbors, device=xyz.device, dtype=torch.int32)[None, None, :].expand(xyz.shape[0], centers.shape[1], -1).contiguous()

    @staticmethod
    def grouping_operation(features, indices):
        return features[:, :, None, :].expand(-1, -1, indices.shape[1], -1).gather(3, indices.long()[:, None, :, :].expand(-1, features.shape[1], -1, -1))


def inputs():
    gen = torch.Generator().manual_seed(871)
    return dict(local=torch.randn(2, 24, 2048, 3, generator=gen) * .15,
                context=torch.randn(2, 32, 12, 3, generator=gen) * .15,
                times=torch.linspace(0, 1, 32)[None].expand(2, -1),
                valid=torch.ones(2, 32, dtype=torch.bool),
                local_times=torch.linspace(.1, .9, 24)[None].expand(2, -1),
                local_to_context_scale=torch.ones(2),
                local_to_context_offset=torch.tensor([[.2, -.3, .1], [-.2, .1, .3]]))


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("ebte,ctr,srr", list(product([False, True], repeat=3)))
def test_all_three_module_switch_combinations(ebte, ctr, srr):
    switches = ModuleSwitches(ebte=ebte, ctr=ctr, srr=srr)
    model, cfg = build_model(pointnet2_ops=SmallOps, switches=switches)
    assert cfg["switch_epoch"] == 30
    assert hasattr(model, "stage_fusion") is ctr
    assert hasattr(model, "prepool_relation") is srr
    model.set_phase(30)
    model.eval()
    x = inputs()
    versions = {k: v._version for k, v in x.items()}
    with torch.no_grad():
        y, trace = model.forward_with_trace(x)
    assert y.shape == (2, 20) and torch.isfinite(y).all()
    assert all(v._version == versions[k] for k, v in x.items())
    assert trace["relation_active"] is srr
    if ebte and ctr and srr:
        assert cfg["parameters"] == 408682
        assert cfg["state_tensors"] == 132
        assert len(model.stage_fusion.temporal_blocks) == 2
        assert model.stage_fusion.dim == 64
        assert model.prepool_relation.k == 8


def test_phase_boundary_preserves_frozen_parent_and_live_relation_gradients():
    model, _ = build_model(pointnet2_ops=SmallOps, srr_scale=1.)
    assert model.switch_epoch == 30
    model.set_phase(29)
    assert not model.relation_active
    assert all(p.requires_grad for p in model.parent_parameters())
    assert not any(p.requires_grad for p in model.relation_parameters())
    model.set_phase(30)
    model.train()
    assert not model.tube_embedding.training and not model.stage_fusion.training
    assert not model.transformer.training and not model.mlp_head.training
    assert model.prepool_relation.training
    before = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("prepool_relation.")}
    optimizer = torch.optim.SGD(list(model.relation_parameters()), lr=.01)
    result = model.training_outputs(inputs())
    assert result["relation_active"]
    assert not result["base_logits"].requires_grad
    loss = torch.nn.functional.cross_entropy(result["logits"], torch.tensor([0, 1]))
    loss.backward()
    assert all(p.grad is None for p in model.parent_parameters())
    grad = model.prepool_relation.output_projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)
    optimizer.step()
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters() if n in before)


def test_srr_zero_exit_and_ties_are_preserved():
    module = SpatialRelationRefinement(dim=8, hidden_dim=4, k=1)
    xyz = torch.tensor([[[[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]]])
    indices, valid, distances, counts = _same_frame_neighbors(xyz, 1)
    assert int(counts[0, 0, 0]) == 3
    assert int(valid[0, 0, 0].sum()) == 3
    tokens = torch.randn(1, 1, 4, 8)
    assert torch.equal(module(tokens, xyz), torch.zeros_like(tokens))
    with torch.no_grad():
        module.output_projection.weight.fill_(1.)
    singleton = torch.randn(1, 1, 1, 8)
    assert torch.equal(module(singleton, xyz[:, :, :1]), torch.zeros_like(singleton))


def test_bound_applies_alpha_before_tanh_and_stops_budget_gradient():
    parent = torch.ones(1, 1, 2, 4, requires_grad=True)
    raw = torch.full_like(parent, .1, requires_grad=True)
    full, tau = bounded_relation_residual(raw, parent, rho=.02, epsilon=1e-6)
    half, tau_half = bounded_relation_residual(.5 * raw, parent, rho=.02, epsilon=1e-6)
    assert torch.equal(tau, tau_half) and not tau.requires_grad
    assert torch.equal(half, tau * torch.tanh(.5 * raw / tau))
    assert not torch.allclose(half, .5 * full)
    half.sum().backward()
    assert parent.grad is None
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_bad_switches_and_scale_fail_loudly():
    with pytest.raises(TypeError):
        ModuleSwitches(ebte=1)
    model, _ = build_model(pointnet2_ops=SmallOps)
    for scale in (0, -1, float("nan"), 2, True):
        with pytest.raises(ValueError):
            model.set_srr_scale(scale)


def test_selected_checkpoint_loads_strictly_and_uses_a_trained_srr():
    path = Path(__file__).resolve().parents[1] / "model/checkpoints/best.pth"
    assert path.is_file(), "The released selected checkpoint is missing"
    model, _ = build_model(pointnet2_ops=SmallOps)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 30
    model.load_state_dict(payload["model"], strict=True)
    model.set_phase(30)
    model.eval()
    assert len(model.state_dict()) == 132
    assert bool(torch.count_nonzero(model.prepool_relation.output_projection.weight))
    with torch.no_grad():
        y, trace = model.forward_with_trace(inputs())
    assert y.shape == (2, 20) and torch.isfinite(y).all()
    assert bool(torch.count_nonzero(trace["relation_residual"]))


def test_initialization_archive_and_config_identity_match():
    import hashlib
    import json
    from model.backbone import FRESH_INITIALIZATION_SHA
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config/selected.json").read_text(encoding="utf8"))
    path = root / "model/checkpoints/initialization.pth"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FRESH_INITIALIZATION_SHA == config["initialization_sha256"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == -1 and payload["from_scratch"] is True
    assert len(payload["model"]) == 125
    assert sum(x.numel() for x in payload["model"].values()) == 399562
    assert not any(n.startswith("prepool_relation.") for n in payload["model"])
    assert all(torch.isfinite(x).all() for x in payload["model"].values())
