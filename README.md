# CSBP711 Advanced AI · Assignment 2: LSNet paper critique and reality check

**Does LSNet's 7×7 large-kernel perception (LKP) earn its place?** We re-implemented the comparison on Fashion-MNIST with the authors' official code, under two training recipes and the same data, model and seeds.

- Under our Assignment 1 recipe (Adam, early stopping), the full 7×7 LKP beats both ablations: by +0.36 pp vs no LKP and +0.65 pp vs a 3×3 LKP.
- Under a literature-style recipe (crop + flip, AdamW, cosine schedule, 100 epochs), both gaps are about zero, and all three variants reach about 94.7 %.
- Verdict: locality matters, but the 7×7 kernel itself does not show a reliable advantage on this task.
  - Evidence for locality (Assignment 1): a fixed pixel permutation drops the CNN from 91.28 % to 87.40 %, while the MLP barely moves (88.25 % to 88.02 %).

The final write-up is `deliverables/CSBP711_Assignment2_Report_v5.docx`. The talk is `deliverables/CSBP711_Assignment2_Slides_v3.pptx`, with the PDF beside it.

> **Private repository.** `experiments/assignment2_lsnet/vendor/lsnet/` is the authors' code (github.com/THU-MIG/lsnet, commit cbe737c). That repository has no licence, so this repo must stay private.

## Main results

Fashion-MNIST official test set, mean ± SD over seeds 42–44. `verify_repo.py` recomputes every number below, and every other number in the report.

| Model | A1 protocol (Adam, early stop) | Literature-style recipe |
|---|---:|---:|
| A1 CNN (0.106M params) | 91.28 ± 0.34 | 90.71 ± 0.23 (91.72 ± 0.33 without augmentation) |
| LSNet-T full, 7×7 LKP (11.03M) | 91.94 ± 0.21 | 94.71 ± 0.14 |
| No LKP depthwise conv | 91.58 ± 0.16 | 94.73 ± 0.07 |
| 3×3 LKP | 91.28 ± 0.16 | 94.72 ± 0.17 |

| Paired contrast | Difference, pp, with 95% CI |
|---|---|
| A1 protocol: full minus no LKP | +0.36 [0.08, 0.65] |
| A1 protocol: full minus 3×3 | +0.65 [0.35, 0.96] |
| Literature recipe: full minus no LKP | −0.02 [−0.24, 0.20] |
| Literature recipe: full minus 3×3 | −0.00 [−0.22, 0.22] |
| Interaction (recipe changes the gap), no LKP | −0.38 [−0.74, −0.02] |
| Interaction (recipe changes the gap), 3×3 | −0.66 [−1.03, −0.29] |

The confidence intervals are paired 95% bootstrap intervals over test near-duplicate groups (2000 resamples, seed 711), conditional on the fitted models.

## Who owns what

Each file in this repository was uploaded by exactly one member.

| Member | Work | Folders and files uploaded |
|---|---|---|
| Yingfeng Wang | Assignment 1 baseline study that A2 builds on: data audit and split, four baselines, pixel-shuffling (locality) test. Verification script that recomputes every number in the report | `experiments/audited_v2/`, the A1 copies inside A2 (`experiments/assignment2_lsnet/references/`, `experiments/assignment2_lsnet/outputs/baseline/`), `verify_repo.py`, `README.md`, `.gitignore`, `deliverables/CSBP711_Assignment2_Report_v5.docx` |
| Chenye Yang | Literature-style recipe check (job 620078) and literature comparison. Slides | `experiments/assignment2_recipe/`, `deliverables/CSBP711_Assignment2_Slides_v3.pptx` and `.pdf` |
| Tianyi Wang | Adapting the official LSNet code, SKA kernel checks, first A2 study (job 618435) | `experiments/lsnet_feasibility/`, `experiments/assignment2_lsnet/` (except the A1 copies above) |

Fill in the names above. The Contributions paragraph in the report describes the same split, so keep the two in step.

## Folder map

```
README.md                 this file
verify_repo.py            recomputes every number in the report from saved predictions and logs (CPU, numpy only)
deliverables/             final A2 report and slides
experiments/
  audited_v2/             Assignment 1, protocol v2: data split, A1 baselines (centroid, logistic, MLP, CNN),
                          original vs pixel-permuted images, A1 checkpoints, predictions and reports.
                          Its README.md documents the A1 study in full.
  assignment2_lsnet/      A2 first study (GPU job 618435): official LSNet code (vendor/), adapter,
                          24 fits, predictions, metrics. references/ holds pinned copies of A1 files,
                          which run_study.py checks by SHA-256.
  assignment2_recipe/     A2 recipe check (GPU job 620078): 18 fits under the literature-style recipe,
                          LITERATURE_COMPARISON.md and reports/recipe_*.csv.
  lsnet_feasibility/      Earlier feasibility probe of the LSNet code on the HPC.
```

Keep these relative positions. The A2 scripts find A1 at `../audited_v2`, LSNet at `../assignment2_lsnet/vendor`, and their pinned packages at `../lsnet_feasibility/deps`.

### Which documents are current

- **Final interpretation:** the v5 report and v3 slides in `deliverables/`, plus `experiments/assignment2_recipe/LITERATURE_COMPARISON.md`.
- **Records of the first A2 study only:**
  - `assignment2_lsnet/README.md`, `RESULTS_REVIEW_ZH.md` and `DESIGN_AND_REQUIREMENTS_ZH.md`;
  - `reports/FULL_SUMMARY.md` and `SUBMISSION_DRAFT.md`.

  Their numbers are correct for job 618435. Their conclusion, that LKP gives a modest gain, is superseded by the recipe check.

## Quick check (any computer, about 10 seconds)

```bash
python -m pip install numpy
python verify_repo.py      # 58 checks, one per number or claim in the report
```

Expected last line: `ALL CHECKS PASSED`.

## Reproduce from scratch (CUDA GPU, about 10 hours in total)

All runs used UAEUHPC with one NVIDIA A100 80 GB, Python 3.11.16, PyTorch 2.7.1+cu126 and NumPy 2.2.6. Exact package lists are in `experiments/audited_v2/reports/environment-lock.txt` and `experiments/assignment2_lsnet/reports/base-environment.txt`. The recipe study ran in the same environment.

```bash
conda create -n advanced-ai python=3.11 -y && conda activate advanced-ai
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r experiments/audited_v2/requirements.txt
# The pinned LSNet dependencies are not in git. Install them into the folder the Slurm scripts put on PYTHONPATH:
python -m pip install --no-deps --target experiments/lsnet_feasibility/deps timm==0.5.4 torchvision==0.22.1
export STUDY_PYTHON="$(which python)"
```

Run the studies in this order: A1 (`experiments/audited_v2`), then the A2 first study (`experiments/assignment2_lsnet`), then the recipe check (`experiments/assignment2_recipe`).

Each study refuses to mix new code with old outputs. Before re-running a study, move its `outputs/` and `reports/` aside, and keep them for comparison. Each folder's README and `run.slurm` give the exact commands. Keep `audited_v2/data/prepared`, because A2 pins its SHA-256.

Each study hashes and freezes its checkpoints before it loads the test set. `frozen_experiment.json` and `test_evaluation_started.json` record that order. The test set was already used in Assignment 1, and both A2 studies disclose this.

## Not in this repository

| What | Where it is |
|---|---|
| `lsnet_feasibility/deps/` (third-party timm and torchvision binaries) | Install with the pip command above |
| Trained LSNet weights (`best.pt`, `final.pt`, about 2.1 GB) | On UAEUHPC as `/home/700053286/advanced AI/deliverables/CSBP711_LSNet_checkpoints_20261105.tar`. They are needed only to re-run test evaluation without retraining. `verify_repo.py` does not use them |
| Combined critique + course project presentation | Submitted with the course project |
| The assignment brief | Course page |

The A1 checkpoints (`model.pt`, about 9 MB in total) **are** included, because the A2 baseline evaluation uses them.

AI assistants were used for code, experiment orchestration and drafting. Every number in the deliverables was recomputed from saved predictions.
