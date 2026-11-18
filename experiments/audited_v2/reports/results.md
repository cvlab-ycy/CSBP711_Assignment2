# Experiment results

All values below come from runs in this project. No leaderboard numbers are used.

## Data

Final split counts: {'test': 10000, 'train': 53979, 'validation': 5998}. All models received 784 pixels scaled by 1/255. See data_audit/audit.json for exclusions and limitations.

## Measured comparison

Accuracy and macro-F1 are means across seeds 42, 43 and 44. Accuracy uncertainty below is sample SD across seeds, not a confidence interval.

| Condition | Model | Accuracy % mean ± SD | Macro-F1 | Fit seconds | Model KiB | Stored coefficients |
|---|---|---:|---:|---:|---:|---:|
| original | Nearest centroid | 67.69 ± 0.00 | 0.6727 | 0.01 | 30.62 | 7840 |
| original | Logistic regression | 84.36 ± 0.06 | 0.8427 | 3.20 | 30.66 | 7850 |
| original | MLP | 88.25 ± 0.17 | 0.8819 | 6.77 | 462.04 | 118282 |
| original | CNN | 91.28 ± 0.34 | 0.9124 | 19.06 | 413.54 | 105866 |
| permuted | Nearest centroid | 67.69 ± 0.00 | 0.6727 | 0.00 | 30.62 | 7840 |
| permuted | Logistic regression | 84.36 ± 0.06 | 0.8427 | 3.18 | 30.66 | 7850 |
| permuted | MLP | 88.02 ± 0.32 | 0.8792 | 5.83 | 462.04 | 118282 |
| permuted | CNN | 87.40 ± 0.41 | 0.8737 | 14.74 | 413.54 | 105866 |

Fit time includes validation checks needed for early stopping, excludes data transfer, startup warmup, checkpoint serialization and hyperparameter search. Each trainable method had two prespecified learning-rate candidates; the centroid has no tuning. Model KiB measures prediction-state tensors, excluding the optimizer. The centroid stores class means and has zero gradient-trained parameters.

Total pilot fit time: 53.22 seconds. Hardware and package versions are recorded per run.

## Explanation and ablation

Original winner: **CNN**. Permuted winner: **MLP**.

CNN minus MLP accuracy: 3.03 percentage points originally and -0.63 after permutation. Relative CNN advantage lost: 3.65 percentage points.

The predicted CNN–MLP rank reversal occurred. This supports the explanation that useful local relationships in these clothing images contribute to the CNN's original advantage. The fixed permutation preserves pixel values and labels while disrupting locality; dense-model initialisation is coordinate-matched as a control.

This experiment isolates a change to spatial arrangement, but does not prove locality is the only cause of a model's performance. Pooling, finite capacity, optimisation and early stopping can also affect sensitivity. CNN and MLP capacity are of similar order; consult learning curves for convergence. Learning rates selected on original validation data are deliberately held fixed across conditions.

Original/permuted prediction agreement controls: {'centroid': 1.0, 'logistic': 1.0, 'mlp': 0.9339333333333333}. The centroid is invariant in exact arithmetic. Coordinate-matched dense models should be close; numerical differences may compound during training.

Paired test-group bootstrap intervals and paired seed results are in ablation_analysis.json and paired_ablation.csv. Test-group intervals are conditional on these trained models; only three training seeds were used.

## Limitations

The scan covers every pair satisfying raw-pixel MAE <= 2 and RMSE <= 6. It does not detect all transformed or semantic duplicates. Official test images remain unchanged, including repeats. This is a corrective rerun after v1 test results were inspected. The hypothesis, models, seeds and learning-rate grid were retained. Corrected-run checkpoints were frozen before corrected-run evaluation, but the official test set is not a newly unseen holdout.

Fashion-MNIST consists of small centred grayscale product images. These findings should not be generalised directly to colour images, unconstrained detection, video, or image generation.

## Reproduction

Run `python src/study.py all --device cuda` in the pinned environment. Completed immutable runs are reused; stale protocol/code hashes are rejected. A frozen checkpoint manifest is written before test evaluation.
