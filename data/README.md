# MSR-Action3D data contract

The dataset is not redistributed. Supply the processed point-cloud NPZ files
from your licensed MSR-Action3D copy. `FULL_DATA_MANIFEST.json` records the 567
original files, their byte hashes, sizes, and loading order. Each file contains
`point_clouds`, a sequence of finite, nonempty N x 3 point arrays in the original
coordinate units. The loader divides coordinates by 300 after point sampling.
Load NPZ files only from a trusted dataset source: the legacy object-array
format requires NumPy `allow_pickle=True`.

The subject split is train subjects 1--5 / test subjects 6--10. There are 270
loaded training videos and 297 loaded test videos. With 24-frame windows and
stride 1, 257 training videos produce 4,478 windows; 287 test videos produce
5,356 windows. The other 10 test videos are too short for a 24-frame window
and are listed explicitly in `results/selected/CONTRACT.json`.

Local clips contain 24 consecutive frames and 2,048 sampled points per frame.
Whole-video context contains up to 32 uniformly spaced frames and 256 points
per frame, with a validity mask for padding. Context and local inputs share
the training scale augmentation. Test-time local seed 54 and context seed 14
are separate random streams. The saved RNG archive includes the actual
global, DataLoader, worker, and per-video context states; the two seed numbers
alone are not a replacement for that archive and the recorded file order.

`TRAINING_MANIFEST.json` describes the historical 566-file development view,
which excludes `a13_s09_e03_sdepth.npz` (a test-subject video). This does not
remove a training-subject sample. The development iterator is traversed to
preserve the historical next-epoch training RNG stream; it does not supply
training losses or a model-selection score. Final evaluation uses all 567
files and scores all 287 eligible test videos, including that A13 example.

No verified raw-depth-to-NPZ converter is included. Regenerating NPZ archives
can change both point order and archive bytes. Exact reference replay requires
the recorded files and fails if their manifest hashes differ. An independently
preprocessed dataset must be treated as a separate reproduction protocol.
