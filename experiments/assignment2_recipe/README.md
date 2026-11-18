# Assignment 2 supplementary recipe check

Private study. It extends the completed LSNet study in `../assignment2_lsnet` (job 618435) and does not modify it.

## Question

Published Fashion-MNIST CNNs reach about 95.6–96.0% (Zhong et al., AAAI 2020: ResNet-20 95.61 ± 0.08%, WRN-28-10 95.99 ± 0.10%, 5 runs). Our Assignment 1 CNN reaches 91.28% and the adapted LSNet-T 91.94%. This study tests two things:

1. How much of that gap comes from the training recipe (augmentation, schedule, budget)?
2. Does LSNet's large-kernel perception (LKP) benefit survive a literature-style recipe?

## Design (fixed in `protocol.json` before running)

| Fit group | Recipes | Seeds | Purpose |
|---|---|---|---|
| A1 CNN | `a1_aug`, `r2_noaug`, `r2_aug` | 42–44 | 2×2 factorial with the existing A1 checkpoints: augmentation × schedule |
| LSNet full / no_lkp / small_lkp | `r2_aug` | 42–44 | LKP ablation under the literature-style recipe |

- `a1_protocol`: the existing checkpoints. Adam, no augmentation, at most 40 epochs, early stopping with patience 8.
- `a1_aug`: the same schedule plus augmentation.
- `r2_*`: AdamW with weight decay 0.025, 5 warmup epochs then cosine decay, 100 epochs, no early stopping (from LSNet Table 10 and the Table 7 budget). The primary checkpoint is the final epoch.
- Augmentation: random 28×28 crop from 4-pixel zero padding plus horizontal flip (Zhong et al.).
- Learning rates are reused from earlier validation selection (CNN 0.001, LSNet 0.003). There is no new tuning.
- The data split, scaling, seeds, model code and the official SKA operator are unchanged. Checkpoints are hashed and frozen before test loading. The test set was used in A1/A2, so it is a reused benchmark, not a blind holdout.

Remaining differences from the literature:
- 53,979 training images instead of 60,000;
- 100 epochs instead of 300;
- AdamW with cosine decay instead of SGD with step decay;
- none of mixup, CutMix, RandAugment or label smoothing.

## Run

```bash
source /etc/profile; module load slurm/23.02.08
sbatch --time=01:00:00 --export=ALL,SMOKE=1 run.slurm   # smoke test, outputs_smoke/
sbatch run.slurm                                         # formal run, outputs/ and reports/
```

The run resumes from epoch boundaries when resubmitted. Never run two copies at once.

## Outputs

- `reports/recipe_summary.csv`: accuracy mean ± SD for each model, recipe and checkpoint.
- `reports/recipe_contrasts.csv`: recipe effects, architecture gaps and recipe × architecture interactions, with seed differences and paired test-group bootstrap intervals.
- `reports/recipe_per_class.csv`, `reports/recipe_learning_curves.png`.

AI assistance: the code and this README were drafted with an AI assistant. Group members must verify and understand them before submission.
