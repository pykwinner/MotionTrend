# SYNTHIA-4D semantic segmentation

This branch predicts 12 semantic classes for each point in a three-frame clip. Its primary metric is mean intersection over union (mIoU), not action-classification accuracy. The complete point-cloud dataset is not included.

## Files and architecture

| Component | Location |
|---|---|
| Dataset, half-scene sampling and label mapping | `datasets/synthia.py` |
| Split lists and class frequencies | `datasets/synthia_metadata/` |
| Shared modules | `modules/ebte.py`, `modules/cmtr.py`, `modules/srrf.py` |
| Encoder, segmentation decoder and module assembly | `models/semantic_segmentation.py` |
| Configuration | `config/synthia_full.json` |
| Training and evaluation | `train-synthia.py`, `evaluate-synthia.py` |

The SYNTHIA configuration explicitly enables `use_ebte`, `use_cmtr` and `use_srrf`. MSR and NTU use their own configurations, where SRRF is disabled by default. No experiment name or legacy switch alias implicitly enables a module.

The network has a four-stage P4 encoder, PST temporal attention and a four-stage interpolation decoder with learned skip connections. XYZ input has shape `[B, 3, N, 3]`, RGB input `[B, 3, 3, N]`, and output logits `[B, 3, 12, N]`, with `N=16384` by default. The bottleneck width is 1,024, distinct from the 80-channel NTU configuration.

| Component | Parameters |
|---|---:|
| Backbone | 14,757,268 |
| EBTE | 61,440 |
| CMTR | 363,224 |
| SRRF | 101,632 |
| Total | 15,283,564 |

EBTE is applied to the Conv3 temporal messages and to the bottleneck. CMTR and the bounded SRRF residual are applied after the transformer and before the decoder. SYNTHIA CMTR reads sparse context from the same three input frames, with at most 2,048 points per frame; it does not read an additional full-video sequence.

## Data format

`--data-root` points to preprocessed files named `<sequence>-<six-digit-frame>.npz`. Each file contains:

| Key | Required content |
|---|---|
| `pc` | XYZ coordinates of shape `[N, 3]`; frames within a clip have equal point counts |
| `rgb` | Floating-point colors of shape `[N, 3]`, conventionally in `[0, 1]` |
| `semantic` | Integer raw semantic IDs in `[0, 15]`, before remapping to the 12 output classes |
| `center` | The current frame's three-dimensional center |

The bundled `trainval_raw.txt` lists 20,703 unique frames and `test_raw.txt` lists 1,886, with no shared frame entries. These are frame counts, not independent-video counts. The original Linux extended-precision class frequencies were converted to the same float32 values consumed by the source loader; the extended-precision values themselves are not retained in the runtime archive.

A sample contains the current frame followed by the previous two frames. Missing history reuses the most recent available frame in that stack. Reused positions have `valid=False` for CMTR memory and retain finite timestamps matching the actual supplied frame. The scene is split about the current frame center into two half-scenes, each with a context margin of 1.0 and its own supervision mask. Raw IDs 0, 1, 13 and 14 are ignored and cannot enter cross-entropy as invalid targets.

Frames are loaded on demand. The default per-worker cache holds at most eight frames rather than the whole dataset. Set `data.cache_size` to `0` to disable caching. Missing files, invalid shapes, nonfinite coordinates and empty half-crops raise errors.

## Train from scratch

Follow the environment and CUDA-extension instructions in the main [README](README.md), then run from the repository root:

```bash
python train-synthia.py --config config/synthia_full.json --data-root /path/to/processed_pc --output-dir runs/synthia_full
```

A fresh clone requires building the extension with `python modules/setup.py build_ext --inplace`. Use the pinned dependencies, the CUDA 11.8 development toolkit (`nvcc`) and a compatible C++ compiler; the PyTorch CUDA wheel alone does not include the compiler.

Use a new or empty output directory. The defaults retain the source SYNTHIA training recipe:

| Setting | Default |
|---|---|
| Seed / epochs | 0 / 150 |
| Training and evaluation batch size | 8 |
| Workers | 10 |
| Optimizer | SGD, momentum 0.9, weight decay 0.0001 |
| Learning rate | 0.01; ten-epoch linear warmup |
| Decay | Multiply by 0.1 at epoch boundaries 30, 40 and 50 |
| Training | All enabled modules jointly optimized from epoch 1 |

The 150-epoch budget belongs to SYNTHIA. The separate MSR and NTU configurations retain 50 epochs. Default training does not load an MSR checkpoint, a teacher or a parent model.

Each loader batch produces two sequential optimizer updates, one per half-scene, followed by one learning-rate scheduler step. Training drops the final incomplete batch; evaluation retains it. The loss is class-weighted masked cross-entropy, normalized by `sum(mask) + 1`. Evaluation loss uses unweighted masked cross-entropy. Training augmentation retains the supplied SYNTHIA implementation's sampling order and flip/scale/rotation operations; it is not a bitwise copy of the older official augmentation implementation.

One ordinary random-sampling evaluation is run on `test_raw.txt` after every epoch. The highest **test mIoU** selects the checkpoint, with the earlier epoch retained on ties. There is no independent validation split for checkpoint selection. Report this rule explicitly in experiments. The entry point does not search multiple point draws or replace the recorded best score with another draw after training.

Module switches accept `--use-ebte true|false`, `--use-cmtr true|false` and `--use-srrf true|false`, as well as underscore spellings. `--output`, `--output-dir` and `--run-dir` are equivalent. `--print-freq` controls console/text logging. Use `--metadata-root` to point to another directory containing the same split filenames and `labelweights.npz`.

## Evaluate and resume

Run one independent evaluation:

```bash
python evaluate-synthia.py --checkpoint runs/synthia_full/best.pth --data-root /path/to/processed_pc --output-dir evaluation/synthia_full
```

The evaluator uses the saved model configuration and loads parameters strictly. It draws points once using the ordinary random stream, so its score may differ from the training epoch's recorded score. An optional `--seed` changes the evaluation draw without changing the checkpoint's model configuration. Historical checkpoints from another architecture are not automatically compatible.

Resume the same experiment with its original configuration and output directory:

```bash
python train-synthia.py --config config/synthia_full.json --data-root /path/to/processed_pc --output-dir runs/synthia_full --resume runs/synthia_full/last.pth
```

Resume restores the model, optimizer, scheduler, random states, completed history and pending-evaluation status. An interrupted evaluation is retried before training continues. An interruption during training restarts from the latest saved epoch boundary; arbitrary-batch resume is not supported. If no checkpoint was saved before an interruption, use a new output directory to restart.

Both entry points default to one GPU. `--data-parallel` explicitly enables multiple visible GPUs and requires at least two. Training resume checks the parallel mode and device count. Per-epoch evaluation during multi-GPU training uses the unwrapped model, retaining the source behavior. CPU wrapper/state checks do not establish real multi-GPU correctness or performance.

When launching from another directory, use absolute paths for the entry script, an explicitly supplied configuration, data and output. Default bundled metadata paths resolve relative to the repository.

## Metrics and outputs

Evaluation accumulates a global 12-by-12 confusion matrix over the masked points from both half-scenes and all three clip positions, then computes per-class IoU and the unweighted mean over all 12 classes. A class with zero union contributes zero IoU. Mean class accuracy and overall valid-point accuracy are reported separately. Historical and filled-in clip positions retain the original half-scene supervision; scoring is not restricted to the current frame.

Outputs include:

- `best.pth`, `last.pth` and `metrics.csv`.
- Best/last per-class metrics and confusion-matrix CSV files, plus `settings.txt`, `runtime.txt`, `result.txt` and `train.log`.
- `modules_init.csv`, with initial module parameter counts and weight norms; Conv3 and bottleneck EBTE are recorded separately.
- `epochs/epoch_001/train_steps.csv` and corresponding directories for later epochs, with both half losses, the learning rate used, the next learning rate and step durations.
- Per-epoch `evaluation_classes.csv`, `evaluation_confusion.csv`, `modules.csv` and `summary.csv`, preserving all epochs rather than only best/last.
- `gpu.csv` and `run_times.txt`, recording GPU utilization, memory, power and run duration.

Step CSV files are organized by epoch, and text logs identify resume attempts. Evaluation duration includes data loading, both half-scene forward passes, loss and metric computation; it is not pure model latency. Device-level telemetry may include other processes on the same GPU.

## Adaptation details and historical records

The integrated dataset fixes interface mismatches, the use of an older frame's center, ignored labels reaching cross-entropy, eager loading of the full corpus and nonfinite timestamps for missing history. Evaluation XYZ, RGB, labels, masks and final NumPy random state match the reference official loader exactly on synthetic cases, including changing frame centers and missing history. This establishes agreement on those cases, not a completed full-dataset evaluation.

The source Conv3 produces 512-channel messages after normalization, activation and multiple projections, while its original adapter used a 256-channel first-layer temporal weight as a reference. The integrated adapter operates on output-message phase differences, uses a matching-width zero-time reference and bounds the energy. The MSR/NTU linear-reference path is retained. Bottleneck EBTE compares tokens at corresponding FPS output indices across frames; this is not physical point tracking.

Historical SYNTHIA checkpoints belong to a different architecture and are excluded from this anonymous release. They used a 256-channel Conv3 adapter input and lacked the current bottleneck EBTE and CMTR content branch. Their recorded scores are not results of the current implementation. No compatible pretrained SYNTHIA checkpoint is supplied here.

`models/semantic_segmentation.py` provides two explicit weight-loading utilities: `load_checkpoint(model, path)` strictly loads a compatible full model, while `load_parent_only(model, path)` loads a completely matching backbone and preserves existing adapter parameters. Both check names and shapes before loading. Default training calls neither utility; there is no hidden pretrained or parent/child schedule. Use `--resume` to continue training with optimizer and random states.

## Validation status

CPU checks cover data processing, all eight module-switch combinations, forward inference with 16,384 points per frame, output/gradient checks, synthetic dataset-to-model updates for both halves, strict checkpoint round trips and numerical MSR/NTU regression checks. With the same Conv3 correction and nonzero adapter exits applied to the source reference, full-width synthetic tests also match initial parameters, outputs and all parameter gradients.

The CPU model checks use reference point operators. No complete SYNTHIA evaluation, formal training run or RTX 4090 native-CUDA validation has been performed for this merged implementation. These checks do not establish a reproducible mIoU, peak GPU memory or throughput result.
