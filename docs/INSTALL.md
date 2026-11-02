# Reference installation

Use Linux x86-64 and a single visible NVIDIA GPU. The recorded hardware was an RTX 4090 D. Native evaluation is deliberately strict because a changed loader, point order, operator, or floating-point reduction can alter this small dataset's video predictions.

```bash
conda create -n motiontrend python=3.10.8 -y
conda activate motiontrend
python -m pip install -r requirements.txt
python -B reproduce.py --verify-only
```

The reference dependency versions are Python 3.10.8, Torch 2.1.2+cu118, NumPy 1.26.4 and einops 0.8.2. An appropriate NVIDIA driver is required. The bundled `tools/native/pointnet2/_ext.cpython-310-x86_64-linux-gnu.so` was used for the recorded result. `train.py` and `reproduce.py` explicitly make this binding available; a globally installed package named `pointnet2` is not required.

For native numerical parity, the evaluator also verifies the recorded worker-seeding implementation, `fork` start method, FP32, two CPU threads, cuDNN determinism and native binary hash. CPU-only verification does not run inference or imply that a native replay has passed.

## Rebuilding the operators

The complete inherited C++/CUDA sources and license are in `third_party/pointnet2`. Building requires a compatible CUDA toolkit and C++ compiler in addition to PyTorch:

```bash
TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=2 \
python -m pip install --no-build-isolation ./third_party/pointnet2
```

This installs an independently rebuilt `pointnet2` package. The exact-reproduction entries prefer the bundled recorded binary and verify its hash. A rebuilt binary is not automatically byte-identical and does not replace a successful numerical replay. In a new process, use the following explicit API to evaluate a rebuilt operator as a separate environment, and record its source, compiler, binary hash and numerical differences:

```python
from model.point_ops import bind
from model.backbone import build_model

ops = bind(prefer_bundled=False)
model, architecture = build_model(pointnet2_ops=ops)
```

## CPU tests

The unit tests use a CPU PyTorch installation and small deterministic neighborhoods. They cover model assembly, gradients, scheduling and restart logic without loading the native CUDA binding. These test neighborhoods are not the point-cloud operators used for reported accuracy.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Data and output paths

`--data` points to a directory containing the original 567 NPZ files. See [data/README.md](../data/README.md) for the exact schema and population. No local drive, SSH endpoint or private environment path is required by the public code.

Use a new output directory. Replay writes outside the code and dataset directories. Training permits a dedicated `runs/` directory or an external directory and refuses to overwrite an existing run without `--resume`. Completed epochs are verified before resuming, and incomplete next-epoch artifacts are preserved separately.
