# Validation scope

The public layout is derived from the recorded default 30+20 experiment. Validation distinguishes unchanged historical evidence from checks executed on the reorganized release.

- The selected checkpoint's 132 tensors and fresh initialization's 125 parent tensors were compared byte for byte with their originals. Random-state values were preserved while private path metadata was removed.
- Saved probabilities were independently accumulated in sequential FP32 order: all 5,356 windows and 287 video rows agree, with 283 correct.
- The split model was compared with the sealed original implementation on CPU diagnostic neighborhoods. Fresh tensors, constructor RNG, selected-checkpoint forward outputs, parent-phase gradients and SRR-phase gradients agreed byte for byte.
- The CPU model and training tests cover all eight module switch combinations, exact checkpoint loading, tie-inclusive neighborhoods, zero-exit gradient flow, frozen parent weights, the location of the SRR coefficient, learning-rate boundaries, parent selection, interrupted artifacts and optimizer/RNG resume equivalence.

On 2026-11-03 (Asia/Shanghai), the reorganized release completed native replay on an RTX 4090 D with the recorded Linux/Python 3.10.8/PyTorch 2.1.2+cu118 environment. All **5,356 window probability arrays and 287 video rows matched exactly**, giving **283/287 (98.61%)**. All 383 evaluation batches had a nonzero raw SRR output, and inference left model tensors unchanged. The native gradient preflight also passed: one temporary parent update and two temporary SRR updates were discarded, with weights, gradients, modes, phase, and RNG restored.

The same 50 CPU tests passed both in the local CPU test environment and under the reference Linux/PyTorch environment with CUDA hidden. GitHub's CPU integrity workflow also passed for the validated implementation commit. CPU tests themselves do not measure model accuracy.

Receipts and the validated implementation/manifest identities are in `results/validation/native_4090/`. This is an inference and gradient-preflight verification, rather than a newly executed full 50-epoch training run. The recorded full 50-epoch run remains the one described in `results/training/`.

Run `python -B reproduce.py --verify-only` to check release hashes and recompute the saved predictions. Run `python -m pytest -q` for CPU tests. Run the documented native replay to test the actual selected checkpoint on your copy of the original dataset.
