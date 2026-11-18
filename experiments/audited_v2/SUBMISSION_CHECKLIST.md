# CSBP711 requirement and handoff checklist

Evidence is mapped to the supplied assignment and its 10-point rubric. Technical completion does not guarantee a mark. Personal, administrative and publication requirements remain with the group. Nothing is published or submitted by this project.

| Requirement | Evidence / status |
|---|---|
| Real, open dataset relevant to research/work | Fashion-MNIST, source and MIT licence in README/report. Shared CV question about shape and spatial relations. Members should add their specific work connection. |
| Document data collection | Report page 1 and source paper. Product photographs and reviewed category labels, with scope limitations. |
| Size, features, splits, class balance | Complete audit JSON, per-class report table and slide 2. 70,000 source images, 784 pixels, ten categories. |
| Missing values, duplicates and leakage | Full numeric duplicate scan with no skipped buckets, exclusions and disjoint retained groups. Corrected rerun and prior test exposure explicitly disclosed. |
| At least four algorithms including simple baseline | Nearest centroid, logistic regression, MLP and CNN. Same images, scaling, splits and paired seeds. |
| Fair model comparison | Six bounded validation pilots, 24 final fits, fixed training rules, three seeds and source/checkpoint hashes. |
| One table with metric, training time and size or parameters | Report/slide 4 and comparison.csv include accuracy, macro-F1, fit time and parameters or tensor size. |
| Data-based explanation of winner | Local shape relations in clothing images motivate CNN advantage. |
| Small ablation tests explanation and predicted rank change | Shared pixel permutation and retraining of all four models. Actual paired rankings, uncertainty and controls in report/slide 5. Results are retained even if a prediction is not supported. |
| Maximum five slides | Exactly 5: data/story, handling, models, results, explanation/ablation. Speaker notes provide references and detail. |
| README source, licence, version/date and reproduction | README, source checksums, protocol JSON and environment lock. |
| Cite reused code/libraries and disclose AI assistance | README and report. No pretrained weights or copied leaderboard results. |
| Public repository and live link on slide 1 | NOT DONE at the user's instruction. Placeholder only. This explicit assignment requirement remains outstanding until the group authorises and performs publication. |
| Actual contribution and commits from every member | User to complete names, actual work and genuine commit evidence. No fabricated commits. |
| Dataset reserved by email, unique among groups | User to confirm with instructor. No email sent. |
| Submission by 4 October | Group responsibility. No upload or submission performed. |

## Before submission

1. Replace member names/IDs and add a truthful sentence about each member's CV work.
2. Record actual contributions and genuine work/commits under each member's own identity.
3. Confirm that Fashion-MNIST is reserved for this group and accepted as relevant to the group’s topics.
4. Review and practise the five slides using `PRESENTATION_GUIDE.md`. Understand the audit correction and limits of the reused test set.
5. Complete the required public repository and slide 1 link only when you choose to authorise publication. No file here claims it already exists.
6. Check the final files and submit through the instructor's required channel before the deadline.

The public repository/commit criterion is worth 1 point. Its current unfinished status means a promise of full marks would be misleading. Dataset acceptance, presentation quality and instructor judgement can also affect the mark.
