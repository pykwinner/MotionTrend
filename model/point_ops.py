"""The four released pointnet2 operators used by MSR classification.

The extension is bound explicitly. Importing this file on a CPU host does not
load a CUDA binary. Native forward/backward functions preserve the released
operator calls, tensor layouts, and gradient rules.
Copyright (c) Facebook, Inc. and its affiliates. MIT-licensed original wrappers.
"""
import importlib
from pathlib import Path
import sys
import torch
from torch.autograd import Function

_ext = None

def _from_package(module, package):
    origin = getattr(module, '__file__', None)
    return origin is not None and Path(origin).resolve().parent == package


def bind(*, prefer_bundled=True):
    """Bind the recorded package relative to this source file by default.

    ``prefer_bundled=False`` explicitly selects an independently installed
    package for research. Choose a binding in a fresh Python process; an
    already imported package from a different source is never silently reused.
    Importing this module alone still does not load the CUDA extension.
    """
    global _ext
    if type(prefer_bundled) is not bool:
        raise TypeError('prefer_bundled must be a real bool')
    native = Path(__file__).resolve().parents[1] / 'tools' / 'native'
    package = (native / 'pointnet2').resolve()
    if prefer_bundled and not (package / '__init__.py').is_file():
        raise RuntimeError('Missing bundled tools/native/pointnet2 package')
    for name in ('pointnet2', 'pointnet2._ext'):
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        bundled = _from_package(loaded, package)
        if prefer_bundled != bundled:
            requested = 'bundled' if prefer_bundled else 'independently installed'
            raise RuntimeError(
                f'{name} is already imported from a different source; '
                f'start a fresh Python process to select the {requested} binding')
    # Remove only this release's known native directory. Placing it first makes
    # the default independent of cwd and of another globally installed package.
    sys.path[:] = [entry for entry in sys.path
                   if Path(entry or '.').resolve() != native.resolve()]
    if prefer_bundled:
        sys.path.insert(0, str(native))
    actual = importlib.import_module("pointnet2._ext")
    if prefer_bundled != _from_package(actual, package):
        raise RuntimeError('pointnet2._ext resolved to a different source than requested')
    if _ext is not None and _ext is not actual:
        raise RuntimeError("A different pointnet2 native binding is already installed")
    _ext = actual
    return sys.modules[__name__]

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        fps_inds = _ext.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(fps_inds)
        return fps_inds

    @staticmethod
    def backward(xyz, a=None):
        return None, None

class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None

class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None

class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        inds = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

furthest_point_sample = FurthestPointSampling.apply
gather_operation = GatherOperation.apply
grouping_operation = GroupingOperation.apply
ball_query = BallQuery.apply
