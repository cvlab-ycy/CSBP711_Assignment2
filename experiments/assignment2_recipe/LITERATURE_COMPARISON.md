# Fashion-MNIST literature comparison (for the report's reality check)

Every literature number was checked against the cited paper's own table. Our numbers are the mean ± sample SD over seeds 42–44 on the official 10,000-image test set. The "literature-style recipe" column comes from job 620078 (`reports/recipe_summary.csv`, primary checkpoints). Those values were independently recomputed from the saved predictions, and all checkpoints were frozen before the test set was loaded.

## Table: our results vs published Fashion-MNIST results

| Published result | Their setup | Published top-1 % | Our closest model | Ours, A1/A2 setup % | Gap, pp | Ours, literature-style recipe % | Remaining gap, pp |
|---|---|---:|---|---:|---:|---:|---:|
| Xiao et al. 2017 [3], sklearn LogisticRegression (L1, C=1) | 60k train, no augmentation | 84.2 | A1 logistic regression | 84.36 ± 0.06 | +0.2 | — | — |
| Xiao et al. 2017 [3], sklearn MLPClassifier (ReLU, 100 hidden) | 60k train, no augmentation | 87.1 | A1 MLP (128-128) | 88.25 ± 0.17 | +1.2 | — | — |
| Zhong et al. 2020 [4], ResNet-20 (≈0.27M params) | 60k train, crop+flip, SGD step decay, 300 epochs, 5 runs | 95.61 ± 0.08 | A1 CNN (0.106M) | 91.28 ± 0.34 | −4.3 | 91.72 ± 0.33 (no aug)<br>90.71 ± 0.23 (with aug) | −3.9 |
| Zhong et al. 2020 [4], WRN-28-10 (≈36.5M params) | same | 95.99 ± 0.10 | Adapted LSNet-T full (11.0M) | 91.94 ± 0.21 | −4.1 | 94.71 ± 0.14 | −1.3 |
| Wang et al. 2025 [1], LSNet-T | ImageNet-1K, 224×224, 300 epochs | 74.9 | — | — | not comparable | — | — |

Notes:
- Zhong et al. report test error. Accuracy here is 100 − error. Their "baseline" column already includes crop+flip.
- Literature-style recipe = crop+flip augmentation, AdamW with weight decay 0.025, 5-epoch warmup then cosine decay, 100 epochs, final-epoch checkpoint. Learning rates are reused from earlier validation selection, with no new tuning.

## What explains each gap

| Candidate source | Finding | Evidence |
|---|---|---|
| Data version | Same 2017 release and same test set. Linear and MLP baselines match the published numbers within 0.2–1.2 pp | Not a cause |
| Split | 53,979 vs 60,000 training images | Not tested. Part of the 1.3 pp LSNet residual |
| Augmentation | Helps the 11M LSNet but **hurts** the 0.1M CNN: −2.45 pp with the A1 schedule [−2.87, −2.04], −1.01 pp with the cosine schedule (difference of means; no interval computed) | Tested (CNN factorial) |
| Training budget and schedule | Cosine schedule without augmentation adds +0.44 pp to the CNN [0.15, 0.73]. The full recipe adds +2.8 to +3.4 pp to every LSNet variant | Tested |
| Checkpoint rule | For LSNet under the new recipe, validation loss bottoms out at about epoch 25–39 while accuracy keeps rising. The validation-loss checkpoint scores 93.42 vs 94.71 at the final epoch (full) | Tested (both checkpoints saved) |
| Metric | Top-1 accuracy in all cases | Not a cause |

Result:
- The recipe closes about 2.8 of the ~4.1 pp gap between LSNet and WRN.
- It does not close the small CNN's gap. That gap is a capacity limit: a 0.1M two-layer CNN cannot absorb crop+flip augmentation, whereas the published 0.27M ResNet-20 is deeper.

## Consequence for the paper's claim (LKP ablation)

| Contrast (test, pp) | A2 protocol (Adam, early stop) | Literature-style recipe | Change (interaction) |
|---|---:|---:|---:|
| full − no_lkp | +0.36 [0.08, 0.65] | −0.02 [−0.24, 0.20] | −0.38 [−0.74, −0.02] |
| full − 3×3 LKP | +0.65 [0.35, 0.96] | −0.00 [−0.22, 0.22] | −0.66 [−1.03, −0.29] |
| full − A1 CNN | +0.66 | +4.00 [3.59, 4.44] | — |

Brackets are paired test-group 95% bootstrap intervals, conditional on the fitted models.
- Seed-level differences, full − no_lkp: [+0.23, −0.14, −0.15].
- Seed-level differences, full − 3×3 LKP: [+0.01, +0.08, −0.10].

Under a stronger, paper-like recipe, the LKP benefit we measured earlier disappears: all three variants reach 94.7%. The earlier gain was an artefact of an undertrained protocol. This matches the paper's own numbers:
- Table 7 ablations use a 100-epoch budget (71.6%), while Table 1 uses 300 epochs (74.9%). The recipe effect (+3.3 pp) is three times the LKP effect (+1.1 pp).
- No table reports variance across runs.
- The adapted LSNet's large advantage over the small CNN (+4.0 pp) remains, but it comes with about 104× the parameters.

## References to add

[4] Z. Zhong, L. Zheng, G. Kang, S. Li, Y. Yang. Random Erasing Data Augmentation. AAAI 2020. arXiv:1708.04896.
