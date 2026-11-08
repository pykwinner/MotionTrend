# NTU RGB+D 60

This branch supports independent cross-subject (X-Sub) and cross-view (X-View) action recognition experiments. The default configuration enables EBTE and CMTR, disables SRRF, and trains all enabled modules jointly from scratch for 50 epochs. There is no parent-model phase or pretrained initialization.

The NTU model uses an 80-channel backbone. It is a different capacity from the earlier 1024-channel NTU experiments. No NTU checkpoint or new NTU accuracy result is provided for this implementation; the bundled MSR checkpoint is not an NTU model.

## Files and module switches

| Component | Location |
|---|---|
| Dataset, sampling, normalization and coordinate alignment | `datasets/ntu60.py` |
| Video names and frame counts | `datasets/ntu60.list` |
| Shared modules | `modules/ebte.py`, `modules/cmtr.py`, `modules/srrf.py` |
| Backbone and 60-class classifier | `models/sequence_classification.py` |
| Configuration loader | `config/ntu.py` |
| Protocol configurations | `config/ntu_xsub.json`, `config/ntu_xview.json` |
| Training and evaluation | `train-ntu60.py`, `evaluate-ntu60.py` |
| GPU monitoring | `telemetry.py` |

The explicit switches are `use_ebte`, `use_cmtr` and `use_srrf`. The default two-module model has 406,002 parameters; enabling SRRF adds 9,120, giving 415,122 parameters. An enabled SRRF uses the same configured scale during training and evaluation.

## Environment and data

The reference environment is Linux, Python 3.10, PyTorch 2.1.2 with CUDA 11.8, and NumPy 1.26.4. See the main [README](README.md) for installation. The intended GPU platform is RTX 4090/4090D; this NTU adaptation has not yet been validated with a complete native CUDA training run.

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python modules/setup.py build_ext --inplace
```

The reference build requires the CUDA 11.8 development toolkit (`nvcc`) and a compatible C++ compiler. The PyTorch CUDA wheel supplies runtime libraries, not the compiler. Install the pinned build dependencies in `requirements.txt`, including setuptools, and build the extension after a fresh clone. A supplied Linux/Python 3.10 extension, when present, can only be used with a compatible environment.

Use preprocessed NTU point clouds in the format expected by the PST-Transformer dataset:

```text
/path/to/ntu60_pst/
  ntu60.list
  video/
    S001C001P001R001A001.npz
    ...
```

Each NPZ contains a `data` array of temporally ordered frames. Each frame must be a nonempty floating-point XYZ array of shape `[N, 3]`; the point count may vary across frames. Retain the original preprocessing dtype. The loader multiplies coordinates by 100, samples local points, and applies clip-level unit-sphere normalization in the released order.

The data are not included. The complete point-cloud collection is approximately 640 GiB. If NPZ files are stored directly in a directory, pass that directory as `--data` and supply `--meta datasets/ntu60.list`. By default, the loader looks for `ntu60.list` in the supplied data directory or its parent. Missing files, empty frames, nonfinite coordinates and discrepancies between listed and actual frame counts raise errors.

Video loading is lazy. The default cache holds at most two complete videos per worker, with four workers. Memory use therefore also depends on video size. Set `data.cache_size` to `0` in the JSON configuration to disable the cache.

## Training

Train the protocols separately, using a new or empty output directory for each run:

```bash
python train-ntu60.py --config config/ntu_xsub.json --data /path/to/ntu60_pst --output-dir runs/ntu_xsub
python train-ntu60.py --config config/ntu_xview.json --data /path/to/ntu60_pst --output-dir runs/ntu_xview
```

To enable all three modules:

```bash
python train-ntu60.py --config config/ntu_xsub.json --data /path/to/ntu60_pst --output-dir runs/ntu_xsub_full --use-ebte true --use-cmtr true --use-srrf true
```

Switches also accept underscore spellings, such as `--use_srrf false`. `--output` and `--output-dir` are equivalent.

| Setting | Default |
|---|---|
| Seed / epochs | 0 / 50 |
| Batch size / workers | 24 / 4 |
| Optimizer | SGD, momentum 0.9, weight decay 0.0001 |
| Learning rate | 0.01; ten-epoch linear warmup |
| Decay | Multiply by 0.1 after epochs 20 and 30 |
| Training loss | Cross-entropy, label smoothing 0 |
| Local input | 24 frames, every second source frame, 2,048 points per frame |
| Whole-video context | Up to 32 frames, 256 points per frame |
| Transformer | Width 80, depth 5, two heads, head dimension 40, MLP width 160 |
| NTU spatial radius / classifier dropout | 0.1 / 0.5 |

Learning-rate scheduling advances per optimizer step. Both training and evaluation retain the final incomplete batch. All enabled modules remain trainable throughout the run; there is no freezing, teacher model, phase switch or restart from an intermediate best epoch.

`--label-smoothing 0.02` enables a fixed alternative from epoch 1. It has not been shown to improve NTU accuracy in this release. Plain cross-entropy is also logged when smoothing is enabled, and evaluation loss remains plain cross-entropy.

## Splits, sampling and evaluation protocol

The local processing and video scoring follow the PST-Transformer NTU loader and training entry point:

- X-Sub uses the released 20 training subjects and tests on the remaining subjects. X-View trains on cameras 2 and 3 and tests on camera 1.
- Local points are sampled using the ordinary advancing NumPy/worker random streams in both training and evaluation. Seeds are not reset within individual windows, and evaluation does not search for a better point sample.
- Local sampling, clip normalization, training-only axis scaling and final float32 conversion retain the released order. Evaluation applies no scaling augmentation.
- The source-frame interval and sliding-window step are both two. A 24-frame clip therefore spans 47 source frames. Shorter videos produce no legal window and are not padded.
- Evaluation uses `model.eval()`, `no_grad()` and FP32. Window softmax probabilities are summed sequentially for each video; the sum's argmax is the video prediction.
- Accuracy is computed over videos that produce at least one legal window. One complete test pass is run after every epoch. The highest test video accuracy selects `best.pth`; ties keep the earlier epoch.

**Checkpoint selection uses the test split after each epoch. There is no independent validation split for model selection in these entry points.** Report this selection rule when reporting results.

With the supplied 56,880-entry metadata and default window settings:

| Split | Training videos: listed / usable | Training windows | Test videos: listed / usable | Test windows |
|---|---:|---:|---:|---:|
| X-Sub | 40,320 / 38,557 | 757,374 | 16,560 / 16,156 | 355,508 |
| X-View | 37,920 / 36,489 | 742,678 | 18,960 / 18,224 | 370,204 |

Startup checks verify metadata and evaluation coverage. Missing or windowless videos are not silently counted as evaluated. Report the window settings and usable-video count rather than claiming evaluation on every listed video. `--expected-videos 0` is available for an intentionally different population, such as a small diagnostic subset; the measured denominator still comes from the data.

CMTR adds whole-video context using a separate per-video random stream. Local and context points share the training scale and a coordinate mapping. This is offline recognition: matching the official local sampling and scoring does not imply that the baseline and MoTrend receive identical inputs.

The 50-epoch schedule and decay milestones 20/30 are this release's joint-training recipe. The referenced official NTU entry point defaults to 15 epochs and milestones 5/10; the training hyperparameters are not all identical to those defaults.

## Evaluation and resume

Run one independent evaluation:

```bash
python evaluate-ntu60.py --data /path/to/ntu60_pst --checkpoint runs/ntu_xsub/best.pth --output-dir evaluation/ntu_xsub
```

The evaluator reads the split, model and input configuration from the checkpoint and loads parameters strictly. An optional `--split xsub` checks the saved protocol and rejects a mismatch. Local points are sampled again, so the result can differ from the draw recorded during the best training epoch.

Resume the same experiment with its original configuration and output directory:

```bash
python train-ntu60.py --config config/ntu_xsub.json --data /path/to/ntu60_pst --output-dir runs/ntu_xsub --resume runs/ntu_xsub/last.pth
```

Resume restores model parameters, optimizer, scheduler, random states and data order, and checks video lengths and splits. If training for an epoch finished but evaluation was interrupted, that evaluation is retried. An interruption within an epoch resumes from the latest saved epoch boundary, not an arbitrary batch. If the first epoch was interrupted before any checkpoint existed, restart in a new output directory.

Outputs include `best.pth`, `last.pth`, `metrics.csv`, best/last video prediction CSV files, `settings.txt`, `runtime.txt` and `result.txt`. GPU monitoring writes `gpu.csv` and `run_times.txt`. Samples are collected approximately once per second per physical GPU, with GPU indices and UUIDs. Utilization may include other processes on the same device. Training and evaluation durations are logged separately; end-to-end duration is not pure model latency.

## Validation status

Local CPU checks cover exact synthetic comparisons with the official local loader and its final random state, both splits and metadata coverage, coordinate/time alignment, all eight module-switch combinations, full-shape forward/backward passes, nonzero branch gradients and strict checkpoint loading. Training-state tests cover normal resume, interrupted evaluation and video aggregation.

These checks do not replace full NTU training or native CUDA validation on an RTX 4090. This release makes no NTU accuracy or peak-memory claim for the adapted model.
