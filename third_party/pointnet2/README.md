# Inherited PointNet++ CUDA operators

These C++/CUDA kernels are copied unchanged from the authenticated PST-Transformer
dependency used by this experiment. Their Facebook copyright notices and MIT
license are retained. Source provenance is recorded in `SOURCE_MANIFEST.json`.
The setup script only makes the build independent of the caller's working
directory and retains the `pointnet2._ext` import name.

Build with the CUDA 11.8 toolkit and the pinned PyTorch environment:

```bash
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 python -m pip install --no-build-isolation ./third_party/pointnet2
```

The exact recorded Linux CPython 3.10 binary is also available under
`tools/native/pointnet2/`. It is retained byte-for-byte for reference replay;
its compiled-in paths refer to the original generic Python build environment.
A rebuilt binary has a different file hash even when the source is unchanged.
Source-build compatibility and exact recorded-binary replay are separate checks.
