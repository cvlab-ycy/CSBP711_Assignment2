# LSNet on small grayscale images

Assignment 2 paper critique and reality check | [Member 1], [Member 2], [Member 3]

**Draft generated from measured experiments. Verify the final three-page layout, fill authors and actual contributions, and confirm paper reservation before submission.**

## Claim and method

LSNet [1] uses large-kernel perception (LKP) to generate spatially adaptive weights and small-kernel aggregation (SKA) to apply them to local features. Its design separates the context used to predict weights from the neighborhood used to aggregate features. This is more specific than stacking a large and a small ordinary convolution. We examine whether the value of a larger perception neighborhood transfers to 28×28 grayscale clothing classification.

Our supporting paper [2] studies transformers on individual pixels. It motivates asking how architectures exploit spatial structure, while retaining positional information. Its results do not imply that every network should tolerate pixel permutation equally.

## Critique grounded in the evidence

The primary paper already provides meaningful controls: Table 5 compares token mixers within a framework, Table 6 compares dynamic convolution, Table 7 studies component and kernel changes, and Table 9 examines other backbones. Therefore, our criticism is not missing ablation. The gap is the reach of its evidence: ImageNet component analysis does not establish that large spatial perception helps much smaller images with different classes and a restricted training budget.

A 7×7 perception kernel occupies a different fraction of a small feature map than of an ImageNet-scale map. Context can help distinguish clothing outlines, but extra receptive field may also be redundant after other layers or blurred by padding near boundaries. These are competing hypotheses, not explanations established merely by a higher score. Holding the adapted network constant and changing only perception convolution tests the incremental benefit more directly than comparing unrelated architectures.

The paper’s biological analogy offers motivation; it does not establish a biological mechanism. Hardware throughput also describes a specific implementation, batch size and device. We cannot infer deployment speed on another device or compare the published throughput directly with our adapted stem. These limits narrow how the claim should be interpreted rather than negating the authors’ reported evidence.

<!-- PAGE BREAK -->

## What our experiment can identify

Comparing LSNet with the small Assignment 1 CNN is useful practically but changes capacity and architecture together. Our principal causal contrasts are within LSNet. Removing only the perception depthwise convolution retains BatchNorm and dynamic SKA; replacing 7×7 with 3×3 tests whether larger perception adds value beyond some spatial perception. These choices isolate narrower questions than removing an entire module with several operations.

The supporting paper’s pixel permutation experiment uses a different model and training regime. Our permutation stress test preserves pixel information and labels but changes neighborhoods for several LSNet operations. A smaller full-minus-control gap after permutation would be consistent with the large kernel benefiting from original spatial organization; it would not prove that LKP alone explains all accuracy loss.

A common learning rate reduces optimization differences between variants, but selecting it with the full model can favor that model. We disclose every pilot validation result. The longer-budget diagnostic addresses one possible alternative explanation—insufficient training—without claiming to exhaust optimizer choices or to reproduce the ImageNet recipe.

## Reality check protocol

We reuse the audited Assignment 1 Fashion-MNIST data [3]: 53,979 training, 5,998 validation and 10,000 official test images; fixed pixels/255 scaling, original split, seeds 42–44 and the same fixed permutation. Single-channel grayscale is not binary. The data offer manageable repeated training and ambiguous clothing categories while removing color cues. The link to our digital-human and deepfake interests is shared CV methodology, not equivalence between clothing and face tasks.

We retain official LSNet-T widths/depths and the official SKA operator, checked against a reference in forward and backward passes. We change the input to one channel, the head to ten classes and stem strides to preserve small-image detail. Six pilots compare learning rates 0.001 and 0.003. The full model selects a shared rate of 0.003 using validation accuracy at its validation-loss-selected checkpoint.

Primary runs use Adam, weight decay 1e-4, batch 256, FP32, no augmentation/pretraining/distillation, at most 40 epochs and validation-loss patience 8. Three variants × two spatial conditions × three seeds give 18 main evaluations. Matching seed-42 pilots are reused, not counted as independent repeats. Three separate seed-42 runs continue to 100 epochs and retain validation-selected checkpoints within caps 40 and 100. All settings and identities are frozen before this study loads test arrays.

<!-- PAGE BREAK -->

## Results and interpretation

| Model | Accuracy % ± seed SD | Macro-F1 | Fit min | Params M | Latency ms |
|---|---:|---:|---:|---:|---:|
| Full 7x7 LKP | 91.94 ± 0.21 | 0.919 | 7.6 | 11.03 | 24.58 |
| No LKP depthwise conv | 91.58 ± 0.16 | 0.916 | 6.3 | 11.00 | 23.96 |
| 3x3 LKP | 91.28 ± 0.16 | 0.913 | 5.9 | 11.01 | 24.16 |

Accuracy is mean ± sample SD over three fitted seeds. Latency is batch-256 FP32, unfused, resident-input inference on NVIDIA A100 80GB PCIe; fit time covers train and validation computation, excluding initialization, I/O and pilot search.

The highest observed mean belongs to Full 7x7 LKP. full - no_lkp: +0.36 pp (95% conditional group interval +0.08 to +0.65); full - small_lkp: +0.65 pp (95% conditional group interval +0.35 to +0.96). These paired intervals resample test duplicate groups after averaging fitted seeds; they do not capture all training uncertainty. A positive contrast supports incremental benefit under this protocol; an interval spanning zero leaves its sign unresolved.

The original-minus-permuted change in the full-control gap is versus no_lkp: +0.25 pp; versus small_lkp: +0.30 pp. The intervention changes spatial organization throughout the network, so its attribution remains broader than LKP.

Budget check, cap40 → cap100, seed42: full: 91.84% → 91.84%; no_lkp: 91.99% → 91.99%; small_lkp: 91.10% → 91.10%. This is a sensitivity diagnostic rather than evidence of convergence across seeds. Learning curves select full: selected epochs 13, 12, 14; no_lkp: selected epochs 11, 10, 9; small_lkp: selected epochs 7, 10, 9. Falling training loss together with rising validation loss is consistent with overfitting; validation behavior and the longer-budget check distinguish that concern from simply running more epochs.

## Limits and next steps

The official test set was seen in Assignment 1; this is a reused benchmark, not a newly blind holdout. Adaptation, common-rate selection and three seeds limit generalization. Contrasts are exploratory, without multiplicity adjustment. Results on small images do not refute results on ImageNet. Next steps are an untouched dataset, a capacity-matched CNN and a multi-seed long-budget check.

**AI and contributions:** AI assisted implementation, source inspection, orchestration and drafting. Members must verify the evidence and understand the code. Actual contributions: [Member 1: …; Member 2: …; Member 3: …].

## References

[1] Wang et al. LSNet: See Large, Focus Small. CVPR 2025. Official code: github.com/THU-MIG/lsnet; pinned commit recorded in upstream.json.

[2] Nguyen et al. An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels. ICLR 2025.

[3] Xiao et al. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. 2017. github.com/zalandoresearch/fashion-mnist, MIT licence; downloaded 2026-09-22.
