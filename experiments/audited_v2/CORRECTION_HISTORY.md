# Duplicate audit correction

On 22 September 2026 the initial completed experiment (job 549642, protocol 1) used an exact-image hash audit plus a restricted near-image screen. The screen required the same average hash and skipped buckets larger than 200 images. It detected 8 near pairs and excluded 8 development images.

A follow-up exhaustive audit, performed after the initial test results were available, evaluated every pair under the same raw-pixel thresholds: MAE≤2 and RMSE≤6. Block-mean distances safely reject impossible pairs using Jensen's inequality. Full-resolution verification checks every remaining candidate. The expanded search found 31 near pairs and 11 crossings in the initial train/validation/test partition.

Protocol2 changes only the audit search coverage. The same exclusion policy now removes 6 development images in groups touching the official test set and 17 other repeated group members. No conflicting-label development groups were found. The same stratified split seed gives 53,979 training and 5,998 validation images, with 10,000 official test images unchanged. All detected retained groups are disjoint across splits.

The correction necessarily changes the retained data and split membership. All 6 pilot fits and 24 final fits were repeated. The model definitions, hypothesis, seed list, permutation, learning-rate candidate grid and training rules are unchanged. Learning rates are reselected on the corrected validation split using the original rule; the MLP selected 0.003 in v2 versus 0.001 in v1. This is not a test-driven search.

The corrected experiment is job 564725 on node gn07, one NVIDIA A100 80 GB PCIe. It completed with exit 0 in 4m 34s. Five invariant tests passed. Checkpoint hashes were recorded before corrected-run evaluation. This chronology does not make the reused official test set a new blind holdout. The final report states that limitation.

Protocol1 evidence remains archived in the local full package. On UAEUHPC the initial root experiment remains intact and protocol2 is in `experiments/audited_v2/`. Final documents refer to protocol2. No original run is relabelled as corrected evidence.

Results after correction: CNN 91.28% and MLP 88.25% mean accuracy on intact images. After permutation: CNN 87.40% and MLP 88.02%. The CNN–MLP gap changes from +3.03 to −0.63 percentage points, a loss of 3.65 points. The predicted aggregate ranking reversal survives the correction. Consult the CSVs for full precision, seed variation and paired results.
