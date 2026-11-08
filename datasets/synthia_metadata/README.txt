SYNTHIA-4D metadata for the imported semantic-segmentation branch.

trainval_raw.txt: 20703 unique listed frames (training and validation merged).
test_raw.txt: 1886 unique listed frames. No frame overlap with trainval.
The first whitespace-separated field identifies sequence/point_clouds/frame.ply.
Processed input is <data_root>/<sequence>-<six-digit-frame>.npz, containing:
pc[N,3], rgb[N,3], semantic[N], center[3]. All clip frames must have equal N.
The source preprocessing uses 16384 points per frame, RGB scaled to [0,1],
and raw semantic IDs 0..15. The 12 supported semantic classes retain source order.
Raw semantic IDs 0,1,13,14 have no supervised output class and are masked.
Missing preceding frames reuse the latest available frame and its finite
normalized source timestamp; valid=False excludes duplicated memory entries.

The train/test lists retain the reference SYNTHIA point-cloud protocol's
train+validation and test partitions. They contain no author identity or local
filesystem location.

The source class-frequency archive stores Linux x87 extended-precision values.
This copy stores their exact nearest float32 values, using round-to-nearest,
ties-to-even. This is the same dtype conversion performed by the source loader,
and permits reading the archive on Windows as well as Linux. Training still
computes 1/log(1.2+frequency) and normalizes by the minimum; evaluation uses ones.

Full point-cloud NPZ data are not included. The loader reads frames lazily using
a bounded per-worker cache and does not preload the full corpus into RAM.
