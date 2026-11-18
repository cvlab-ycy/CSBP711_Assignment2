# Spatial structure in clothing classification

A reproducible CSBP711 computer vision experiment comparing nearest centroid, logistic regression, an MLP and a small CNN on Fashion-MNIST. A fixed pixel permutation tests whether local image structure explains the CNN's advantage.

## Final evidence and correction history

The main results use **protocol v2**, an exhaustive correction to the duplicate screening. The initial experiment is preserved in `archive/original_v1/` in the local full package. On UAEUHPC the original run stays in the project root and the corrected experiment lives in `experiments/audited_v2/`. Do not mix the original outputs with v2 code.

The first screening found eight near pairs but used a hash filter and skipped 37 large buckets. A complete audit found 31 near pairs, including 11 crossings in the original splits. The corrected run removes six development images in groups touching test images and seventeen repeated development-group members. The hypothesis, models, seeds, tuning grid and numeric thresholds were retained. **The v1 test results had already been inspected. The corrected run is not a newly blind holdout.** Corrected checkpoints are frozen before corrected evaluation.

Final evidence:

- `deliverables/CSBP711_Report.docx` and `CSBP711_Slides_Final.pptx` (exactly five slides).
- `reports/comparison.csv`, `all_runs.csv`, `paired_ablation.csv`, `ablation_analysis.json` and `results.md`.
- `reports/verification.json` checks hashes, split separation and metrics independently from saved predictions.
- `reports/figures/` contains learning curves, accuracy comparisons, confusion matrices and fixed-index errors.
- `SUBMISSION_CHECKLIST.md` lists the remaining personal and administrative fields.

Nothing has been published. The private copies are for review and completion.

## Dataset and connection to our work

The group works on different CV topics, including generation and detection. Both use object shape and spatial relationships. This study tests that shared property through classification; it does not claim to measure generation or detection performance.

Source: https://github.com/zalandoresearch/fashion-mnist . Licence: MIT (see the source repository). Citation: Han Xiao, Kashif Rasul and Roland Vollgraf (2017), *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*, https://arxiv.org/abs/1708.07747 . The dataset creators derived the images from Zalando product photographs and reviewed category labels.

The raw data were downloaded and verified on 22 September 2026. Exact source URLs, official MD5 checksums, SHA-256 hashes and verification timestamps are in `data/prepared/audit.json`. The IDX headers, lengths, pixel type and label ranges are validated. The source contains 60,000 development images and 10,000 official test images, ten balanced classes, 784 grayscale pixel features per image and no missing pixels. There are no exact image repeats.

Near pairs satisfy raw-pixel mean absolute difference ≤ 2 AND RMSE ≤ 6 on a 0–255 scale. Block-mean distance gives a safe RMSE lower bound, then full-image distances verify candidates. All pairs are covered under this numeric rule with no hash filter or skipped buckets. This does not detect all semantic or transformed duplicates. Connected groups touching test are removed from development; remaining groups contribute one representative. Conflicting-label development groups would be excluded (none found). A stratified 90/10 split with seed 711 gives 53,979 train and 5,998 validation images. The official 10,000 test images remain unchanged. Group disjointness is checked. Class counts and exclusions are saved.

## Environment and reproduction

The existing UAEUHPC Conda environment is `/home/700053286/miniconda3/envs/advanced-ai`. The experiment uses Python 3.11 and PyTorch 2.7.1+cu126. Exact resolved dependencies are in `reports/environment-lock.txt`. To set up a fresh environment:

```bash
conda create -n advanced-ai python=3.11 -y
conda activate advanced-ai
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

From the corrected project directory:

```bash
python -m pytest -q
python src/study.py all --device cuda
python scripts/verify_results.py
```

For UAEUHPC, submit using the scheduler, not the login node:

```bash
cd '/home/700053286/advanced AI/experiments/audited_v2'
bash scripts/submit.sh
bash scripts/status.sh
```

`src/study.py` stages are `prepare`, `train`, `evaluate`, `report` and `all`. Paths resolve relative to the source, including spaces. `all` resumes complete fits while rejecting source/protocol hash mismatches. An interrupted fit restarts from scratch. For a fresh independent repeat use a new project copy with the same source/config/raw data and empty generated `data/prepared`, `outputs` and `reports` folders. Do not overwrite the preserved evidence.

The initial six validation-only pilots test two learning rates each. Accuracy selects the learning rate, with loss and smaller learning rate breaking ties. Minimum validation loss selects the checkpoint (max 40 epochs, patience 8, minimum improvement 0.0001). Adam uses batch 256 and weight decay 0.0001. Original-condition selected settings stay fixed for both conditions. Final seeds 42/43/44 are shared across all models. No augmentation or pretrained weights. Inputs are pixels/255 for every algorithm. Flattening changes shape only. The baseline is computed directly in PyTorch on the same allocated device.

## Ablation and interpretation

Hypothesis: local spatial relationships in clothing images favour convolution. One shared permutation, seed 20260922, applies to every image in every split. It preserves pixel values and labels while disrupting locality. All four models retrain. The prediction is a smaller CNN–MLP gap, potentially reversing their rank. A ranking reversal is reported only if it occurs.

Logistic and MLP initial first-layer weights are reindexed to match the transformed coordinates, giving equivalent initial functions up to floating-point error. The centroid is invariant in exact arithmetic. Numerical differences can compound during MLP training and early stopping. This data-property ablation directly tests the assignment's data-based explanation. It is not an architecture ablation. Capacity, pooling and optimisation remain possible contributors.

Accuracy and macro-F1 are measured on the saved official test set. Summary tables show mean and sample SD across three seeds. The centroid has deterministic predictions. Paired differences are computed within each seed. A 2,000-resample test-group bootstrap averages the three fitted seeds first, capturing conditional test-sample uncertainty rather than all training randomness.

Fit seconds include the epoch loop and early-stopping validation, with CUDA synchronisation. Loading, transfer, startup warmup, serialization and pilot search are excluded. Pilot cost is reported separately. Tensor bytes measure prediction state without the optimiser. The centroid has 7,840 stored coefficients and zero trained parameters. Hardware and package versions are saved per run. Short baseline timings are sensitive to overhead.

Findings are limited to small centred grayscale product images, one permutation and three seeds. They should not be transferred directly to colour photographs, detection, generation or video.

## Contributions and AI disclosure

| Member | Actual contribution | Genuine commit evidence |
|---|---|---|
| [MEMBER 1 NAME / ID] | [COMPLETE AFTER REVIEW] | [ACTUAL COMMIT IDs] |
| [MEMBER 2 NAME / ID] | [COMPLETE AFTER REVIEW] | [ACTUAL COMMIT IDs] |

Dataset reservation: [CONFIRM WITH INSTRUCTOR]. Public repository URL: [ADD ONLY WHEN AUTHORISED]. No contribution history is fabricated.

OpenAI Codex assisted with implementation, duplicate-audit correction, experiment execution, numerical checks, analysis, report and slides. Members must review and understand the work and accurately state what they themselves contributed. All reported numbers come from saved runs rather than invented or copied leaderboard results.

The project implements the model definitions and study orchestration using library primitives. It uses PyTorch layers/optimisers, scikit-learn metrics/splitting, NumPy, pandas and Matplotlib. No pretrained model weights are used. Cite the dataset above and:

- Paszke et al. (2019), *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. https://arxiv.org/abs/1912.01703
- Pedregosa et al. (2011), *Scikit-learn: Machine Learning in Python*, JMLR 12:2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

The private full package includes raw data, prepared splits, checkpoints, histories, predictions, reports and logs. The source-only package excludes heavy data and weights and reproduces them from the checksum-verified source. Both remain unpublished.
