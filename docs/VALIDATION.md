# Validation scope

The public layout is derived from the recorded default 30+20 experiment. Validation distinguishes unchanged historical evidence from checks executed on the reorganized release.

- The selected checkpoint's 132 tensors and fresh initialization's 125 parent tensors were compared byte for byte with their originals. Random-state values were preserved while private path metadata was removed.
- Saved probabilities were independently accumulated in sequential FP32 order: all 5,356 windows and 287 video rows agree, with 283 correct.
- The split model was compared with the sealed original implementation on CPU diagnostic neighborhoods. Fresh tensors, constructor RNG, selected-checkpoint forward outputs, parent-phase gradients and SRR-phase gradients agreed byte for byte.
- The CPU model and training tests cover all eight module switch combinations, exact checkpoint loading, tie-inclusive neighborhoods, zero-exit gradient flow, frozen parent weights, the location of the SRR coefficient, learning-rate boundaries, parent selection, interrupted artifacts and optimizer/RNG resume equivalence.

CPU tests do not measure model accuracy. Native GPU replay status and receipts, when available, are in `results/validation/`. A native preflight retains no optimizer updates and is not a new 50-epoch reproduction run. The recorded full 50-epoch run remains the one described in `results/training/`.

Run `python -B reproduce.py --verify-only` to check release hashes and recompute the saved predictions. Run `python -m pytest -q` for CPU tests. Run the documented native replay to test the actual selected checkpoint on your copy of the original dataset.
