# 10-minute talk + 5-minute Q&A

Each member must speak; assign actual speakers before submission.

1. Question and paper claim — shared CV motivation; small-image transfer gap; main/supporting citations.

2. How LSConv works — static large-kernel perception generates dynamic weights; small-kernel aggregation uses them.

3. Critical reading — Tables 5–7 and 9 already provide controls; explain limits of resolution transfer and hardware-specific speed.

4. Reality-check protocol — same A1 split/seeds/scaling, exact adaptation, common LR, architecture controls, test reuse disclosure.

5. Results — original and permuted accuracy, within-LSNet contrasts and uncertainty, same-GPU latency/size. See primary_comparison.csv.

6. Why — interpret actual contrasts and spatial interaction; avoid attributing all permutation damage to LKP.

7. Learning process — actual curves, pilot choices and budget40→100; distinguish generalization from optimization.

8. Verdict, limitations, next steps, AI disclosure and actual contributions. Claims must match analysis.json.
