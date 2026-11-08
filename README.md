# MoTrend

Anonymous research-code release for point-cloud sequence modeling, built on the PST-Transformer backbone.

The three optional modules are implemented separately:

- **EBTE**: energy-bounded trend encoding (`modules/ebte.py`).
- **CMTR**: context retrieval for local tokens (`modules/cmtr.py`).
- **SRRF**: bounded spatial relation refinement (`modules/srrf.py`).

This release contains action-recognition pipelines for MSR-Action3D and NTU RGB+D 60, and a semantic-segmentation pipeline for SYNTHIA-4D. It makes no claim of conference acceptance.

## Installation

Use Linux, Python 3.10, PyTorch 2.1.2 with CUDA 11.8, and the dependencies below. Building the point-cloud operators requires the full CUDA 11.8 toolkit (`nvcc`) and a compatible C++ compiler. The PyTorch CUDA runtime wheel alone does not provide these build tools.

From the repository root:

```bash
python -m pip install -r requirements.txt
python modules/setup.py build_ext --inplace
```

The build dependency `setuptools==69.5.1` is pinned because PyTorch 2.1.2 imports `packaging` through `pkg_resources`. Precompiled CUDA binaries are excluded from this anonymous source release. The operator loader imports the locally built extension; it does not fall back to a globally installed copy.

## Repository layout

```text
datasets/                       Dataset loading and split metadata
models/
  sequence_classification.py    MSR and NTU model definitions
  semantic_segmentation.py      SYNTHIA model definition
modules/
  ebte.py / cmtr.py / srrf.py    Optional modules
  point_4d_convolution.py       Inherited point-cloud backbone operators
  transformer_v1.py / temporal_attention.py
  setup.py / _ext_src/          CUDA extension build entry and source
config/                         Dataset configurations and validators
results/mrs/                    MSR checkpoint and labeled historical records
train-msr-small.py / evaluate-msr-small.py
train-ntu60.py / evaluate-ntu60.py
train-synthia.py / evaluate-synthia.py
utils.py / utils_ntu.py / utils_synthia.py
scheduler.py / telemetry.py
```

The source layout follows the [PST-Transformer repository](https://github.com/hehefan/PST-Transformer). `mrs` is retained as the result-directory name; the dataset is MSR-Action3D.

## Configurations

| Dataset | Configuration | Default enabled modules | Epochs |
|---|---|---|---:|
| MSR-Action3D | `config/msr.json` | EBTE + CMTR | 50 |
| NTU60 X-Sub | `config/ntu_xsub.json` | EBTE + CMTR | 50 |
| NTU60 X-View | `config/ntu_xview.json` | EBTE + CMTR | 50 |
| SYNTHIA-4D | `config/synthia_full.json` | EBTE + CMTR + SRRF | 150 |

All enabled modules are optimized jointly from the first epoch. There is no automatic pretrained loading, parent/child schedule or module-freezing stage. The NTU configuration uses the adapted 80-channel backbone. See [NTU instructions](README_NTU.md) and [SYNTHIA instructions](README_SYNTHIA.md) for their data formats and evaluation rules.

The Boolean switches are `use_ebte`, `use_cmtr` and `use_srrf` in each JSON model configuration. Command-line equivalents are `--use-ebte`, `--use-cmtr` and `--use-srrf`, each accepting `true` or `false`.

## MSR-Action3D data

Pass the directory containing preprocessed `*.npz` files through `--data`. Each file must contain `point_clouds`, a sequence of XYZ point sets. Labels and subject IDs are parsed from names such as `a01_s06_e01_sdepth.npz`. Use the original coordinate units expected by the loader: coordinates are divided by 300 before model input.

Subjects 1-5 form the training split; subjects 6-10 form the test split. With the complete 567-file corpus and 24-frame windows, 287 test videos have valid windows, giving 5,356 windows. The completeness check reports missing videos; short videos are not padded into the evaluation denominator. The data itself is not redistributed here.

Local input uses 24 consecutive frames, 2,048 points per frame and sliding-window stride 1. CMTR uses up to 32 uniformly placed context frames and 256 points per frame. Local and context coordinates share the documented augmentation/alignment transform. Context sampling uses a filename-dependent private random stream and does not consume the local sampling stream.

## Train, evaluate and resume

Train the default two-module model from scratch:

```bash
python train-msr-small.py --data /path/to/MSRAction3D --output-dir runs/mrs
```

Enable the full three-module model in a separate run:

```bash
python train-msr-small.py --data /path/to/MSRAction3D --output-dir runs/mrs_full --use-srrf true
```

The MSR defaults are seed 0, batch size 14, 10 workers, SGD with learning rate 0.01, momentum 0.9 and weight decay 0.0001. The schedule uses 10 warmup epochs and decay milestones at 20 and 30 epochs. It runs one ordinary test evaluation per epoch and selects the checkpoint with the highest video accuracy, retaining the earlier epoch on ties. This is test-set checkpoint selection; there is no separate validation-set selection in this configuration.

Evaluate the included checkpoint once:

```bash
python evaluate-msr-small.py --data /path/to/MSRAction3D --output-dir evaluation/mrs
```

The default checkpoint is `results/mrs/best.pth`. To evaluate a new run:

```bash
python evaluate-msr-small.py --data /path/to/MSRAction3D --checkpoint runs/mrs/best.pth --output-dir evaluation/mrs_new
```

Resume a training run from its own output directory:

```bash
python train-msr-small.py --data /path/to/MSRAction3D --output-dir runs/mrs --resume runs/mrs/last.pth
```

The supplied historical checkpoint is an evaluation export, not a resumable training state. Newly generated training checkpoints include optimizer, scheduler, random states, file order and pending-evaluation state. An interrupted evaluation is retried before the next epoch; an interrupted training epoch restarts from the last saved epoch boundary. Use a new or empty output directory for a fresh run.

`--output-dir` and `--output` are aliases. When launching from another working directory, use absolute entry-script, data and output paths. Default bundled configurations and weights resolve relative to the repository.

## Included MSR record and its scope

| Checkpoint | Modules | Training epoch | Parameters | Recorded score | Selection rule |
|---|---|---:|---:|---:|---|
| `results/mrs/best.pth` | EBTE + CMTR; SRRF off | 11 (index 10) | 399,562 | 283/287 (98.6063%) | Highest of 1,000 fixed-weight test-input draws |

The selected historical draw used local seed 54 and context seed 14. Model parameters were fixed during that search. This is a single-model result, but it is a selected sampling maximum, not an ordinary single-draw score, a guarantee for a fresh 50-epoch run, or the score of the full three-module model. Its sampling-selection budget must be disclosed in any comparison.

The included `metrics.csv` is a one-record evaluation summary, not a training curve. Video predictions, four errors, per-class statistics and the confusion matrix agree on 287 videos and 5,356 windows. The checkpoint strictly loads into the current two-module architecture.

The current evaluator does not replay the selected historical input state. It performs one ordinary random draw; the result can differ from 98.6063%. Matching seed numbers alone does not recreate the old worker random states and point indices. The selected point-state archive is not included in this release, so exact reproduction of that recorded draw is not supported by the supplied entry point. See [reproducibility notes](REPRODUCIBILITY.md).

## Evaluation and validation

MSR scoring follows the reference video aggregation: apply FP32 softmax per window, accumulate probabilities per video in window order, then take a video-level argmax. The denominator is the number of videos producing valid windows. Evaluation disables training augmentation and parameter updates.

Local checks cover imports and command-line paths, sampling/aggregation comparisons, module switches, strict checkpoint loading, numerical model comparisons and normal/interrupted resume behavior. CUDA forward/backward was previously checked for the MSR configuration. This release audit did not retrain a model, rerun the full benchmark, or perform a fresh CUDA build. NTU and SYNTHIA documentation states their separate validation limits.

Fresh training follows the loader's dataset file order. Preserve data content, ordering, dependency versions and worker settings for numerical comparisons. Resume validates the saved ordering. A fixed seed alone is not a promise of identical outputs across environments.

## License and attribution

Original project contributions are released under the [MIT License](LICENSE). Required notices for inherited PST-Transformer and PointNet++ code are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset permissions remain separate from the code license.

This anonymous package contains no author affiliations, personal repository links or fabricated publication metadata. Paper-specific citation information can be added after the anonymous review period.
