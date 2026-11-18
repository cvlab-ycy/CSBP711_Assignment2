# Assignment 2: LSNet small-image reality check

**Completed:** GPU job 618435 finished successfully on 2026-11-04 at 00:13:47 UAE time (3:56:13 elapsed). All 24 fits and numerical verification passed. See `RESULTS_REVIEW_ZH.md` for reviewed interpretation, including the stopping-policy sensitivity that qualifies the main ranking. Local verified evidence is under `received_618435/`; reviewed documents are under `final_materials/output/`. The automatic submission draft in raw reports is superseded by the reviewed three-page Word report.

Private study. No public repository or external publication is created. This is a **controlled adaptation**, not a reproduction of the original ImageNet training recipe. Do not describe feasibility results as final study results.

## Question and papers

Does large-kernel perception help classification on small grayscale images, and how sensitive is its benefit to spatial order and training budget?

- Primary: Wang et al., [LSNet: See Large, Focus Small](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_LSNet_See_Large_Focus_Small_CVPR_2025_paper.html), CVPR 2025.
- Supporting: Nguyen et al., [An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels](https://proceedings.iclr.cc/paper_files/paper/2025/hash/959f70ee50044bed305e48e3484005a7-Abstract-Conference.html), ICLR 2025.
- [Official LSNet implementation](https://github.com/THU-MIG/lsnet), commit `cbe737c92b7c43ecf02d08545a07f03f1010177c`. Vendor sources are unchanged; hashes in `upstream.json`. Retain upstream notices and check upstream redistribution terms before any future publication.

## Dataset and provenance

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), original 2017 release, MIT licence, project download 2026-09-22. Reuse audited Assignment 1 archives (53,979 train / 5,998 validation / 10,000 test) rather than generating a new split. SHA-256 hashes are fixed in `protocol.json`. Images are **grayscale intensity, not binary**. The test set was inspected in Assignment 1; it is a reused benchmark, not a newly blind holdout.

## Run one GPU job

From this project directory on UAEUHPC:

```bash
source /etc/profile
module load slurm/23.02.08
mkdir -p logs
sbatch run.slurm
```

Only submit once. There are no job arrays, child jobs, or separate GPU evaluation jobs. All 24 unique fits, test evaluation and GPU timing run serially in the same allocation. The same job then performs CPU aggregation, verification and evidence packaging. The script requests one GPU, four CPUs, 32 GB host memory and a 24-hour limit; allocation/finish time is not guaranteed.

Existing private dependencies: conda `advanced-ai` (PyTorch 2.7.1+cu126), sibling `../lsnet_feasibility/deps` (timm 0.5.4, torchvision 0.22.1+cu126), and `../audited_v2` data/checkpoints. `STUDY_PYTHON` and `A1_ROOT` may override deployment defaults; no local Mac paths are embedded. The actual runtime versions are recorded in reports and preflight metadata.

For another machine, reproduce those packages plus NumPy, scikit-learn, Matplotlib and Triton 3.3.1; set `PYTHONPATH` for the pinned dependencies, then:

```bash
python run_study.py --a1-root /path/to/audited_assignment1
python aggregate.py
python verify_results.py
python package_results.py
```

CUDA is required for the formal pipeline. CPU architecture/operator checks use the reference SKA only. Official Triton forward and gradients must match that reference on the allocated GPU before any formal fit. `run_study.py` refuses source/config/checkpoint identity changes. Resubmitting the same script after a failure resumes from an epoch boundary; inspect the failure first and never start concurrent copies. No automatic additional job is submitted.

## Planned work and statistical interpretation

1. Six original-image, seed-42 learning-rate pilots: three variants × two rates. Select the full model's validation winner as a single common rate.
2. Eighteen main evaluations: three variants × two spatial conditions × three seeds. Reuse the three matching pilots for original seed 42; they are not independent replicates.
3. Three 100-epoch, seed-42 original-image diagnostic fits; evaluate their validation-selected checkpoints at caps 40 and 100. This is a secondary budget check, not a 100-epoch ImageNet reproduction.
4. Freeze all identities/checkpoints before loading test arrays. Evaluate the 24 study checkpoints and 24 preserved Assignment 1 checkpoints; time all on the same GPU.
5. Recompute metrics from saved predictions, check provenance, report seed SD separately from paired test-group bootstrap intervals. Intervals are conditional on fitted models; exploratory contrasts are not multiplicity-adjusted.

`no_lkp` removes only LKP's depthwise convolution and retains its BatchNorm and dynamic SKA. `small_lkp` changes 7x7 perception to 3x3. Neither removes all spatial processing. Fixed permutation changes neighborhoods throughout the network; it is not an LKP-only intervention. Choosing a common full-selected learning rate helps isolate architectural change but may favor the full model's optimization setting, so all pilot scores are disclosed.

Training time reports include train + validation computation, excluding checkpoint I/O, initialization, warm-up and pilot search; whole-job elapsed time is a separate operational cost. Batch-256 inference latency excludes transfers, uses FP32, resident tensors, unfused models, ten warm-ups and thirty synchronized repeats. Parameters and tensor bytes are both reported; centroid stores buffers but has zero trainable parameters.

## Outputs

- `outputs/progress.json`: current fit/epoch.
- `outputs/fits/`: checkpoints, optimizer/RNG resume state, raw curves and validation evidence.
- `outputs/frozen_experiment.json`: model identities frozen before test loading.
- `outputs/evaluation/`, `outputs/baseline/`: predictions and metrics.
- `reports/analysis.json`, CSVs and figures: measured results, uncertainty and diagnostic curves.
- `reports/FULL_SUMMARY.md`: detailed critique, experiment summary, training walkthrough and limitations, generated only after completion.
- `reports/SUBMISSION_DRAFT.md` and `.html`: concise measured-result draft with three planned page sections. Final rendering must verify the actual page count; this is not a claim of a finished Word/PDF submission.
- `reports/SLIDE_OUTLINE.md`, `presentation_payload.json`: source for a 10-minute talk plus 5-minute Q&A.
- `reports/VERIFICATION.json`: independent numerical audit, required before packaging.
- `deliverables/Assignment2_evidence.zip`: lightweight private evidence package. All large checkpoint files remain in the project.

Final submission still needs a visually checked **maximum-three-page report**, presentation material within **15 minutes**, real names/contributions, and instructor paper reservation confirmation. No result is prefilled before computation. Automated numerical reports are not a claim that human rubric requirements are complete.
