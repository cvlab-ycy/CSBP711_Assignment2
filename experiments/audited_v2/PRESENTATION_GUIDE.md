# Five-slide speaking guide

Target about five to six minutes. Split speaking roles according to actual contributions.

**Slide 1: data and question, about 45 seconds.** Introduce both members and explain their actual generation/detection topics. Say that shape and spatial relationships are the common motivation. The experiment itself is classification on 70,000 small product images. State the source and licence. Complete the repository link before submission when publication is authorised.

**Slide 2: handling, about 75 seconds.** Explain the official 60,000/10,000 separation and the development split after deduplication. State that there are no missing values or exact repeats. The numeric near-duplicate rule is MAE≤2 and RMSE≤6. Explain that the first audit was incomplete and the later audit found crossings, leading to a complete rerun. Be explicit that the final test set had already been seen. This is a documented correction, not a fresh blind evaluation.

**Slide 3: models, about 60 seconds.** Centroid is the simple distance baseline, logistic tests a linear boundary, MLP adds nonlinear capacity and CNN adds local shared filters. Inputs, splits, seed list and scaling are the same. Pilot tuning uses validation only. Explain early stopping and why the permuted condition keeps the selected settings.

**Slide 4: results, about 60 seconds.** Read the measured winner and key comparison from the table. Accuracy ±SD describes variability over three seeds. Give the parameter comparison and timing definition. Avoid treating A100 timings as a universal speed ranking.

**Slide 5: explanation and ablation, about 90 seconds.** State the hypothesis before explaining the intervention. The permutation keeps all values and labels but changes neighbourhoods, applies identically to every split and is followed by retraining. Describe the actual CNN–MLP gap change. The invariant centroid and nearly invariant dense controls check that the result is not simply missing information. Say that the conclusion is conditional on the tested settings and one permutation.

## Likely questions

**Why does this count as an ablation?** The assignment asks for a data-property explanation followed by an intervention testing it. Removing useful spatial neighbourhoods is a data/structure ablation. Architecture ablation, such as removing a convolution, is another type, but it would test a different causal factor.

**Does pixel permutation destroy information?** It is a bijection and preserves every pixel value. A dense model can represent the equivalent function after relabelling coordinates. A small convolutional network cannot generally undo an arbitrary global permutation with a simple reindexing of local kernels.

**Why does the MLP change at all?** Coordinate-matched initial weights make its starting functions equivalent up to floating-point error. Changed accumulation order, minibatch optimisation and early stopping can amplify small numerical differences. We report the observed variation and prediction agreement instead of assuming exact equality.

**Was the test set untouched?** The official test images were consulted for contamination screening. Predictive performance was first sealed until the initial models froze. The later audit caused a corrected rerun after those initial results were known. The same official benchmark is therefore a reused test set. We kept the hypothesis, model family and tuning grid fixed and froze corrected checkpoints before corrected evaluation.

**Why not a more complex dataset?** The assignment asks for a manageable but nontrivial real dataset. Fashion-MNIST supports a controlled comparison and reproducible experiments. It is not evidence about full-size generation or detection systems. Instructor acceptance and dataset reservation must be confirmed.

**What do the uncertainty intervals mean?** Table SD is variability over the three fitted seeds. Bootstrap intervals resample test duplicate groups after averaging those three fitted seeds, so they quantify test-sample uncertainty conditional on the fitted models.

**What is left for the group?** Replace personal fields, record real contributions and commits, confirm reservation and acceptance, complete the public repository/link when authorised, understand the evidence, rehearse and submit.
