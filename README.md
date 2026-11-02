# MotionTrend

Reproducible MSR-Action3D implementation of **MoTrend**, with three modules:

- **EBTE** — Energy-Bounded Trend Encoding
- **CTR** — Context and Trend Retrieval
- **SRR** — Spatial Relation Refinement

This release contains the **30-epoch parent + 20-epoch SRR** experiment, its selected single-model checkpoint, and the actual fixed-sampling state. It does not combine checkpoints or predictions.

| Recorded result | Correct / eligible videos | Accuracy | Parameters |
|---|---:|---:|---:|
| Selected full model, epoch 30 (zero-based) | 283 / 287 | 98.61% | 408,682 |
| Final epoch, epoch 49 | 282 / 287 | 98.26% | 408,682 |

**Evaluation protocol:** both parent and full-model checkpoints were selected using this fixed evaluation set. The reported maximum is therefore an evaluation-selected development result, not an independently held-out model-selection estimate. The run completed 50 epochs; the selected checkpoint is an intermediate checkpoint. See [the exact protocol](docs/PROTOCOL.md).

## Layout

```text
MotionTrend/
├── data/                       Dataset manifests and data contract; no raw dataset
├── model/
│   ├── backbone.py             Backbone and model assembly
│   ├── ebte.py                 EBTE
│   ├── ctr.py                  CTR, including the content-reading branch
│   ├── srr.py                  SRR and its bounded residual
│   ├── point_ops.py            Native PointNet++ operator interface
│   └── checkpoints/            Selected trained model and fresh initialization
├── config/
│   ├── switches.py             The three module switches
│   ├── selected.json           Selected checkpoint and exact evaluation identities
│   ├── training_recipe.json    Declared 30+20 training protocol
│   └── environment.json        Reference environment
├── results/
│   ├── selected/               Window probabilities, video predictions, RNG states
│   ├── training/               All 50 recorded epochs and GPU telemetry
│   └── PROVENANCE.json         Original-to-release artifact mapping
├── training/                   Training, phase transition, checkpoint/resume logic
├── tools/                      Input sampling, evaluation, and integrity checks
├── third_party/                Native operator sources and upstream licenses
├── tests/                      CPU tests; these do not measure model accuracy
├── train.py
└── reproduce.py
```

## Installation

The reference run used **Linux x86-64, Python 3.10.8, PyTorch 2.1.2+cu118, NumPy 1.26.4, einops 0.8.2, and an RTX 4090 D**. The bundled PointNet++ extension matches that environment.

```bash
conda create -n motiontrend python=3.10.8 -y
conda activate motiontrend
python -m pip install -r requirements.txt
```

For the exact reference replay, retain the bundled native extension. Source and build instructions are also provided in [third_party/pointnet2](third_party/pointnet2/README.md). CPU tests use small diagnostic neighborhoods; the model's accuracy requires native CUDA operators. See [installation details](docs/INSTALL.md).

## Reproduce the selected checkpoint

Place the **original 567 processed NPZ files** in an external directory. Their order, sizes, and byte hashes are recorded in [data/FULL_DATA_MANIFEST.json](data/FULL_DATA_MANIFEST.json). The dataset is not included, and this release does not contain a verified raw-depth converter.

```bash
# Check the release and recompute the saved 287-video scores on CPU.
python -B reproduce.py --verify-only

# Run the actual selected checkpoint once on the recorded input.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B reproduce.py --data /path/to/full567_npz --output /path/to/new_replay
```

The evaluator restores the saved local/context RNG state, all ten worker states, and file order. It checks **5,356 window probabilities and 287 video predictions**, not just the final accuracy. Batch size 14, ten Linux `fork` workers, local seed 54, context seed 14, and FP32 video-score accumulation are part of the contract. An arbitrary new preprocessing or sampling stream is a different protocol.

## Train from scratch

```bash
# Native input and gradient checks; temporary test updates are discarded.
CUDA_VISIBLE_DEVICES=0 python -B train.py \
  --data /path/to/full567_npz --output /path/to/new_preflight --preflight-only

# A new complete 50-epoch run.
CUDA_VISIBLE_DEVICES=0 python -B train.py \
  --data /path/to/full567_npz --output /path/to/new_training_run

# Resume the same run after its latest closed epoch.
CUDA_VISIBLE_DEVICES=0 python -B train.py \
  --data /path/to/full567_npz --output /path/to/new_training_run --resume
```

Training starts from the saved **epoch −1 random initialization**, not `best.pth`. After 30 parent epochs, the run restores its own selected parent tensors, preserves the epoch-29-end training RNG, and trains a fresh SRR optimizer for 20 epochs while freezing the parent. The inference SRR coefficient is 0.5 and is applied **before** `tanh`; training uses 1.0. The recipe is fixed in advance for this release.

## Model configuration

Edit [config/switches.py](config/switches.py) to construct module variants. EBTE, CTR, and SRR each live in one file; the inherited backbone is not presented as another innovation module. The supplied checkpoint and exact training recipe require all three enabled. Disabling a module creates a different model that needs its own trained weights; it does not retain the 98.61% claim.

```python
from config.switches import ModuleSwitches
from model.backbone import build_model

model, architecture = build_model(
    switches=ModuleSwitches(ebte=True, ctr=True, srr=True),
    srr_scale=0.5,
)
# After loading the complete trained state_dict:
model.set_phase(30)  # Enable the trained SRR branch.
model.eval()
```

See [architecture notes](docs/ARCHITECTURE.md), [result provenance](docs/PROTOCOL.md), and [validation records](docs/VALIDATION.md). Historical state-dictionary names are retained for exact checkpoint loading.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Attribution and licensing

Original contributions are released under the [MIT license](LICENSE). The PST-Transformer backbone and PointNet++ operators retain their upstream notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset and third-party rights are separate. [CITATION.cff](CITATION.cff) describes this software release; no conference acceptance, paper DOI, or author list is asserted here.

[中文使用说明](README_zh.md)
