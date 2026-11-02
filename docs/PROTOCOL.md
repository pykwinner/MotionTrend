# Recorded experiment and result interpretation

This release contains one default MoTrend model, with EBTE, Content CTR, and bounded SRR. It uses the latest 30+20 experiment and does not mix artifacts from the earlier 11+39 experiment.

## Data and fixed input

The subject split is subjects 1–5 for training and 6–10 for testing. The supplied manifests describe 567 original processed NPZ files. With 24-frame local windows and stride one, 287 of the 297 test videos are eligible, producing 5,356 windows. All 287 eligible videos, including the restored A13 example, are scored.

Each local frame has 2,048 points. Whole-video context uses up to 32 uniformly spaced frames with 256 points per frame. Local seed 54 and context seed 14 control different random streams. Reproduction also requires the saved RNG archive, file order, DataLoader/worker recipe and B14/W10 batch boundaries. The full input SHA256 is recorded in `config/selected.json`.

Video scores are sequential FP32 sums of window softmax probabilities. There is one checkpoint, one set of windows and one classification head. No ensemble or per-video best-sampling selection is used in this release.

## Fifty-epoch training budget

The fixed recipe uses seed 0 and SGD, momentum 0.9, weight decay 0.0001, batch size 14, and 320 updates per epoch.

| Stage | Actual epochs, zero-based | Trainable parameters | Learning-rate schedule |
|---|---|---|---|
| Parent | 0–29 | Backbone + EBTE + CTR | Base 0.01, 10-epoch linear warmup, decay ×0.1 at epochs 11 and 20 |
| SRR | 30–49 | SRR only; parent frozen | Fresh SGD at 0.01, no second warmup, decay ×0.1 after 5 and 15 SRR epochs |

After all 30 parent epochs, the run selects its own highest fixed-evaluation parent checkpoint, with earliest-epoch tie breaking. It restores **model tensors only**. It does not rewind the completed budget, the epoch-29-end RNG stream, or the parent optimizer clock. SRR begins from its zero-initialized exit with a new optimizer.

The SRR loss is cross-entropy plus a weight-0.25 KL preservation term for **training examples** correctly predicted by the frozen parent. Test examples are not used in gradient updates. The historical 566-file development iterator is consumed without inference to preserve the original next-training-epoch RNG sequence; the extra test file remains present in the final fixed evaluation.

Training uses SRR scale 1.0. Evaluation uses a predeclared scale 0.5 **inside** the bounded residual:

\[
Y=U+\tau\tanh(0.5R(U,P)/\tau),\qquad
\tau=0.02\,\operatorname{stopgrad}(\operatorname{RMS}(U))+10^{-6}.
\]

The actual RMS axes and broadcasting are defined in `model/srr.py`. Applying 0.5 after `tanh` would produce a different model.

## Selected and final checkpoints

All 50 epochs completed: 9,600 parent updates and 6,400 SRR updates. The selected parent was epoch 10. The selected full model was epoch 30, after 320 SRR updates, with **283/287 = 98.61%**. The final epoch 49 scored **282/287 = 98.26%**. Because the selected parent tensors are restored at the transition, the selected full checkpoint inherits 3,520 parent updates plus 320 SRR updates; its tensor lineage is not 16,000 sequential updates.

The parent checkpoint also scored 283/287. At the selected full checkpoint, SRR is trained and nonzero, but the per-video predictions preserve the same 283 correct and four incorrect cases. This record does not demonstrate an additional accuracy gain attributable to SRR.

Both parent and full-model checkpoints are selected using the reported evaluation set. The result is therefore an evaluation-selected development maximum. It should not be described as a separately held-out selection result, a multi-seed mean, a final-epoch result, or a conventional all-parameters joint 50-epoch run. The fixed input/scale originated in earlier development; this release freezes those choices instead of presenting them as an untouched benchmark protocol.

## Provenance and efficiency

`results/PROVENANCE.json` links the original checkpoint to the public archive. Only nonessential private path metadata was removed; model tensors and random-state values were preserved. Publication archives have new file hashes, which are recorded separately from the original hashes.

`results/training/metrics.json` contains all 50 epoch records. The accompanying telemetry records device utilization, memory and power. Its full-run timing includes training, data loading, development traversal and repeated fixed evaluation; it is **not** pure model-forward latency. Hardware and sampling conditions must accompany any efficiency comparison.
