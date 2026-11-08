# Reproducibility and result scope

## Included model and records

The included MSR checkpoint is a 20-class, 399,562-parameter model with EBTE and
CMTR enabled and SRRF disabled. It was saved at training epoch 11 (index 10).
The recorded 283/287 score is the maximum over 1,000 frozen-weight test-input draws.
It is not a standard single-draw estimate or a full three-module result.
The selected draw used local seed 54 and context seed 14; these integers do not
encode the complete worker RNG states or sampled point indices.

The public entry point uses the reference window-to-video scoring rule, but it
performs a new ordinary draw and does not restore the historical selected input.
The point-state archive for an exact replay is not included. Accordingly, this
release supports strict loading and independent evaluation of the checkpoint,
not guaranteed reproduction of its selected historical maximum.

For fair comparisons, report the number and selection rule of input draws,
checkpoint-selection rule, split, effective video count, window count and
module configuration for every method. A sampling maximum should not be
compared as if it were an ordinary single-pass result.

## Training and selection

Current MSR/NTU recipes jointly optimize all enabled modules from scratch for 50
epochs; SYNTHIA uses 150 epochs. None automatically loads the included checkpoint.
Each recipe evaluates the test split once per epoch and selects the highest test
score. These recipes do not use an independent validation split for selection.
That selection rule is part of the experimental protocol and must be reported.

Directory file ordering, data contents, workers, library versions and random
states all affect reproducibility. Fresh runs retain the source loader's file
ordering; resume checks saved ordering and data identity. The portable evaluation
export is intentionally separate from resumable training checkpoints.

## Dataset-specific limits

- MSR: the record contains 287 videos, 5,356 windows and 4 errors. The error list,
  per-class counts and confusion matrix agree with the recorded predictions.
- NTU: the adapted 80-channel code has local functional checks, but this release
  includes no trained NTU checkpoint or benchmark accuracy claim.
- SYNTHIA: current 1024-channel segmentation code has local functional checks,
  but no full-dataset training/evaluation result is supplied. Historical weights
  from a different architecture have been excluded from this release.

No raw dataset is redistributed. The small NTU/SYNTHIA split metadata and
SYNTHIA class-frequency file support the documented preprocessed formats.
Follow the original dataset terms when obtaining or redistributing the data.

## Verification performed for this release

Checks covered Python syntax, six command-line entry points from a different
working directory, module/configuration paths, strict MSR state loading, result
arithmetic, source-file language/private-path scans and mocked CUDA build
configuration. Earlier local checks also covered model output/gradient parity
and interrupted-evaluation resume behavior.

These checks are not a fresh RTX 4090 training run, a native CUDA build on every
environment, or proof of a new accuracy result. Build the extension from source
with the documented toolchain before GPU execution.

## Tool assistance

OpenAI Codex assisted with code implementation, refactoring, debugging,
documentation and local validation during preparation of this code release.
The manuscript's LLM Usage Statement must account for all generative-AI use in
the research and writing, beyond the release-specific activities described here.
