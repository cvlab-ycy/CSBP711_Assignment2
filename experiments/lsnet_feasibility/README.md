# LSNet feasibility check — Paper Critiques

This isolated experiment checks whether LSNet can support the follow-up assignment.
It does not replace Assignment 1, evaluate its test set, or establish a model ranking.
All data come from the existing audited train/validation archive; no new dataset is needed.

## Latest verified outcome (3 November 2026)

GPU job **614431 completed successfully in 34 seconds on gn01 (A100 80GB)**.
The official CUDA/Triton SKA passed forward and input/weight gradient comparisons
against the independent reference on three shapes (maximum forward absolute error
1.91e-6). All three `tiny_adapted` variants passed finite-gradient and loss-reduction
checks using 1280 training / 500 validation images, seed 42, three epochs, batch 64.
Final validation accuracies were 68.2% (full), 64.2% (no_lkp), and 62.8% (small_lkp).
These are short feasibility runs, not final ablation evidence or a reliable ranking.

Training took about 1.05–1.07 seconds per 1280-image epoch and peak allocated tensor
memory was approximately 934–942 MiB. This supports continuing with the official
LSNet-T widths/depths plus the documented small-image adaptation; model shrinking
is not currently necessary for GPU resource reasons. Full-data runtime remains an
estimate. The historical CPU small-sample failures are preserved, not overwritten.
See `FEASIBILITY_REPORT.md` and `results/gpu-614431/result.json` for the evidence.
The latest connection uses `UAEUHPC`; it reaches the same project as `UAEUHPCLB`.

## Sources and provenance

- Primary paper: Wang et al., **LSNet: See Large, Focus Small**, CVPR 2025,
  <https://openaccess.thecvf.com/content/CVPR2025/html/Wang_LSNet_See_Large_Focus_Small_CVPR_2025_paper.html>.
- Author implementation: <https://github.com/THU-MIG/lsnet>.
- Pinned commit: `cbe737c92b7c43ecf02d08545a07f03f1010177c`.
- `upstream.json` records SHA-256 hashes of the inspected source files.
- Upstream files in `vendor/lsnet` remain unmodified. `adapter.py` imports the original
  class definitions and applies explicit changes after construction.
- Codex assisted with source inspection, adaptation, checks and this documentation.
  Members must inspect the implementation and disclose actual contributions.

## Why an adaptation is necessary

The official LSNet-T factory hard-codes the ImageNet input geometry and does not
forward arbitrary keyword arguments to the model constructor. The underlying
`LSNet` class accepts single-channel input and a ten-class head, so the adapter
uses it directly, with no pretrained weights or distillation.

With the original three stride-2 stem convolutions, a 28x28 input becomes 4x4,
then 2x2, 1x1, 1x1. Most LS modules then operate on severely reduced spatial
information. A successful forward pass alone would therefore be insufficient.

The candidate changes the three stem strides from 2 to 1, giving stage feature
sizes 28, 14, 7, 4. Input images remain 28x28 single-channel grayscale divided
by 255, exactly as in Assignment 1; there is no enlargement, augmentation or
learned input normalization. Internal BatchNorm/GroupNorm remain part of LSNet.
Attention resolution metadata matches the actual feature sizes.

The primary candidate (`tiny_adapted`) retains official LSNet-T widths
[64,128,256,384], depths [0,2,8,10], attention configuration, LS convolution,
residual blocks, feed-forward blocks and squeeze-excitation. It is a small-image
adaptation, **not** an exact reproduction of the ImageNet training recipe.

A secondary `compact` configuration reduces widths to [32,64,128,192] and depths
to [2,2,2,2]. Its CPU pilot is only an inexpensive implementation sanity check.
It is not the published LSNet-T and must not be reported as such.

## Focused ablation

To avoid an ambiguous reconstruction of the paper's static-aggregation variant,
the initial experiment focuses on the large-kernel perception claim in Table 7:

1. `full`: original 7x7 perception convolution and dynamic 3x3 aggregation.
2. `no_lkp`: remove the perception depthwise convolution, retaining its BatchNorm
   to avoid also changing normalization. The surrounding weight-generation path
   remains. This explicitly documented control is not removal of the entire LKP.
3. `small_lkp`: replace 7x7 perception with 3x3; all other mechanisms remain.

All three use dynamic SKA. The changes affect parameter counts, which are reported;
they are not parameter-matched by construction. A later formal study should address
capacity as a possible alternative explanation if a meaningful gain is observed.
The paper already contains relevant ablations: the critique is about transfer to
low-resolution grayscale data and limited training budgets, not a claim that the
authors omitted ablations. Its Table 7 analysis uses 100 epochs; do not confuse
that table with the paper's longer main training runs.

## Correctness checks

The official SKA implementation is CUDA/Triton-specific. For CPU checks we provide
an equivalent PyTorch unfold implementation, retaining zero padding and the
official channel mapping `channel_index % weight_channels`.

`probe.py` first compares the reference implementation with an independent
shift-and-slice implementation, including gradients for inputs and dynamic weights.
On CUDA it also compares the unmodified official Triton operator against the
reference, including both gradients. The GPU probe fails if parity checks fail;
it does not silently substitute the reference backend.

Feature traces check actual geometry. Short training checks finite loss and
gradients, and whether training loss decreases. These checks establish execution
and learnability, not a generalization claim or statistical significance.

## Data and pilot protocol

- Audited archive SHA-256:
  `71cba7be7a7ae5ffcc4c8cd78f5b00ba1801d736d0640959a10d99cd93958aa1`.
- Samples are drawn independently from the existing train and validation splits,
  balanced by class. Exact indices are saved with every pilot.
- Seed 42; Adam, learning rate 0.001, weight decay 0.0001; FP32.
- CPU pilots: 320 training and 200 validation examples, batch 32, three epochs.
- A follow-up `compact` full-model diagnostic uses 1280 training and 500 validation
  examples, batch 64 and ten epochs, because the initial short pilots did not
  establish useful validation performance. It is not used to rank ablations.
- GPU pilot: 1280 training and 500 validation examples, batch 64, three epochs.
- The batch size and dataset size intentionally differ from final Assignment 1
  runs because this is a resource/correctness probe, not the final reality check.
- No test archive is loaded. The old test set was previously inspected during
  Assignment 1; this feasibility work does not make it a newly blind holdout.
- BatchNorm statistics and overfitting can make these very short pilots' validation
  accuracy poor or unstable. Do not use their rankings to choose a winning method.
- Warm-up is excluded from timing; timing is FP32/unfused. Subset-to-full-epoch
  extrapolations are estimates, not measured full-data training time.
- Different shapes/backends can affect numerical behavior; CPU reference timing
  must never be presented as the official CUDA kernel's performance.

## Running

Dependencies are isolated in this experiment's `deps` (HPC) or `deps_local` (Mac)
directory; the existing Assignment 1 environment has not been upgraded.
HPC uses existing PyTorch 2.7.1+cu126 and adds timm 0.5.4 and torchvision 0.22.1.
The unmodified upstream requirements are not installed wholesale because their
old NumPy pins and unrelated task dependencies are unnecessary here.

From this directory on UAEUHPCLB:

```bash
module load slurm/23.02.08
mkdir -p logs
sbatch run.slurm
```

The scheduler script references the existing `../audited_v2` data, requests one
GPU for up to 20 minutes, and writes job-specific result and log paths. Pending
job 614431 was subsequently shortened to five minutes through `scontrol` to allow
shorter backfill opportunities; this does not change the saved script default. It never
runs GPU training on the login node. GPU task initially submitted: **614431**.
The project remains under `/home/700053286/advanced AI/experiments/lsnet_feasibility`.

For a CPU correctness check, with PyTorch, NumPy and timm available:

```bash
python probe.py --data ../../data/prepared/train_val.npz \
  --output results/new-cpu-check --checks-only
```

Use a fresh output directory for each run. JSON results are saved after each
completed variant, including a traceback if a check fails.

## Decision criteria

1. Input geometry and all three variants run correctly.
2. Official CUDA SKA agrees with the independent reference in outputs and gradients.
3. Short fits reduce training loss without non-finite gradients.
4. Measured GPU memory/time is feasible for the planned three-seed study.

Only after all four are satisfied can we call the intended HPC path validated.
The JSON `loss_reduced` check compares train-subset loss in **eval mode** before
and after fitting. Thus `learning_check_failed` can coexist with falling online
train-mode loss. This intentionally conservative check exposes an unresolved
evaluation/generalization problem; it must not be relabelled as a complete pass.
The final experiment still needs the full train/validation split, a declared tuning
budget, seeds 42/43/44, stable checkpoint selection and a frozen test evaluation.
Nothing in this directory is a final ablation result or a guarantee of assignment marks.
