"""Concise submission draft, produced from measured results, not placeholders."""
import html
from pathlib import Path

def create(reports,a):
    original=[r for r in a['primary'] if r['condition']=='original']
    names={'full':'Full 7x7 LKP','no_lkp':'No LKP depthwise conv','small_lkp':'3x3 LKP'}
    table=['| Model | Accuracy % ± seed SD | Macro-F1 | Fit min | Params M | Latency ms |',
           '|---|---:|---:|---:|---:|---:|']
    for r in original:
        table.append(f"| {names[r['model']]} | {100*r['accuracy_mean']:.2f} ± {100*r['accuracy_sd']:.2f} | {r['macro_f1_mean']:.3f} | {r['fit_seconds_mean']/60:.1f} | {r['parameters_mean']/1e6:.2f} | {r['latency_ms_median_mean']:.2f} |")
    effects=[]
    for r in a['paired']:
        if r['condition']=='original':
            lo,hi=r['conditional_group_ci95_pp'];effects.append(f"{r['comparison']}: {r['difference_pp']:+.2f} pp (95% conditional group interval {lo:+.2f} to {hi:+.2f})")
    spatial='; '.join(f"versus {r['control']}: {r['difference_pp']:+.2f} pp" for r in a['spatial_interactions'])
    budget='; '.join(f"{r['variant']}: {100*r['accuracy_cap40']:.2f}% → {100*r['accuracy_cap100']:.2f}%" for r in a['budget'])
    curve='; '.join(f"{v}: selected epochs "+', '.join(str(r['best_epoch']) for r in a['learning_curves'] if r['id'].startswith(v+'_original')) for v in names)
    pages=[[
      '# LSNet on small grayscale images',
      'Assignment 2 paper critique and reality check | [Member 1], [Member 2], [Member 3]',
      '**Draft generated from measured experiments. Verify the final three-page layout, fill authors and actual contributions, and confirm paper reservation before submission.**',
      '## Claim and method',
      'LSNet [1] uses large-kernel perception (LKP) to generate spatially adaptive weights and small-kernel aggregation (SKA) to apply them to local features. Its design separates the context used to predict weights from the neighborhood used to aggregate features. This is more specific than stacking a large and a small ordinary convolution. We examine whether the value of a larger perception neighborhood transfers to 28×28 grayscale clothing classification.',
      'Our supporting paper [2] studies transformers on individual pixels. It motivates asking how architectures exploit spatial structure, while retaining positional information. Its results do not imply that every network should tolerate pixel permutation equally.',
      '## Critique grounded in the evidence',
      'The primary paper already provides meaningful controls: Table 5 compares token mixers within a framework, Table 6 compares dynamic convolution, Table 7 studies component and kernel changes, and Table 9 examines other backbones. Therefore, our criticism is not missing ablation. The gap is the reach of its evidence: ImageNet component analysis does not establish that large spatial perception helps much smaller images with different classes and a restricted training budget.',
      'A 7×7 perception kernel occupies a different fraction of a small feature map than of an ImageNet-scale map. Context can help distinguish clothing outlines, but extra receptive field may also be redundant after other layers or blurred by padding near boundaries. These are competing hypotheses, not explanations established merely by a higher score. Holding the adapted network constant and changing only perception convolution tests the incremental benefit more directly than comparing unrelated architectures.',
      'The paper’s biological analogy offers motivation; it does not establish a biological mechanism. Hardware throughput also describes a specific implementation, batch size and device. We cannot infer deployment speed on another device or compare the published throughput directly with our adapted stem. These limits narrow how the claim should be interpreted rather than negating the authors’ reported evidence.'
    ],[
      '## What our experiment can identify',
      'Comparing LSNet with the small Assignment 1 CNN is useful practically but changes capacity and architecture together. Our principal causal contrasts are within LSNet. Removing only the perception depthwise convolution retains BatchNorm and dynamic SKA; replacing 7×7 with 3×3 tests whether larger perception adds value beyond some spatial perception. These choices isolate narrower questions than removing an entire module with several operations.',
      'The supporting paper’s pixel permutation experiment uses a different model and training regime. Our permutation stress test preserves pixel information and labels but changes neighborhoods for several LSNet operations. A smaller full-minus-control gap after permutation would be consistent with the large kernel benefiting from original spatial organization; it would not prove that LKP alone explains all accuracy loss.',
      'A common learning rate reduces optimization differences between variants, but selecting it with the full model can favor that model. We disclose every pilot validation result. The longer-budget diagnostic addresses one possible alternative explanation—insufficient training—without claiming to exhaust optimizer choices or to reproduce the ImageNet recipe.',
      '## Reality check protocol',
      'We reuse the audited Assignment 1 Fashion-MNIST data [3]: 53,979 training, 5,998 validation and 10,000 official test images; fixed pixels/255 scaling, original split, seeds 42–44 and the same fixed permutation. Single-channel grayscale is not binary. The data offer manageable repeated training and ambiguous clothing categories while removing color cues. The link to our digital-human and deepfake interests is shared CV methodology, not equivalence between clothing and face tasks.',
      f"We retain official LSNet-T widths/depths and the official SKA operator, checked against a reference in forward and backward passes. We change the input to one channel, the head to ten classes and stem strides to preserve small-image detail. Six pilots compare learning rates 0.001 and 0.003. The full model selects a shared rate of {a['common_learning_rate']:g} using validation accuracy at its validation-loss-selected checkpoint.",
      'Primary runs use Adam, weight decay 1e-4, batch 256, FP32, no augmentation/pretraining/distillation, at most 40 epochs and validation-loss patience 8. Three variants × two spatial conditions × three seeds give 18 main evaluations. Matching seed-42 pilots are reused, not counted as independent repeats. Three separate seed-42 runs continue to 100 epochs and retain validation-selected checkpoints within caps 40 and 100. All settings and identities are frozen before this study loads test arrays.'
    ],[
      '## Results and interpretation',
      '\n'.join(table),
      f"Accuracy is mean ± sample SD over three fitted seeds. Latency is batch-256 FP32, unfused, resident-input inference on {a['hardware']['gpu']}; fit time covers train and validation computation, excluding initialization, I/O and pilot search.",
      f"The highest observed mean belongs to {names[a['observed_winner']]}. "+'; '.join(effects)+'. These paired intervals resample test duplicate groups after averaging fitted seeds; they do not capture all training uncertainty. A positive contrast supports incremental benefit under this protocol; an interval spanning zero leaves its sign unresolved.',
      'The original-minus-permuted change in the full-control gap is '+spatial+'. The intervention changes spatial organization throughout the network, so its attribution remains broader than LKP.',
      'Budget check, cap40 → cap100, seed42: '+budget+'. This is a sensitivity diagnostic rather than evidence of convergence across seeds. Learning curves select '+curve+'. Falling training loss together with rising validation loss is consistent with overfitting; validation behavior and the longer-budget check distinguish that concern from simply running more epochs.',
      '## Limits and next steps',
      'The official test set was seen in Assignment 1; this is a reused benchmark, not a newly blind holdout. Adaptation, common-rate selection and three seeds limit generalization. Contrasts are exploratory, without multiplicity adjustment. Results on small images do not refute results on ImageNet. Next steps are an untouched dataset, a capacity-matched CNN and a multi-seed long-budget check.',
      '**AI and contributions:** AI assisted implementation, source inspection, orchestration and drafting. Members must verify the evidence and understand the code. Actual contributions: [Member 1: …; Member 2: …; Member 3: …].',
      '## References',
      '[1] Wang et al. LSNet: See Large, Focus Small. CVPR 2025. Official code: github.com/THU-MIG/lsnet; pinned commit recorded in upstream.json.',
      '[2] Nguyen et al. An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels. ICLR 2025.',
      '[3] Xiao et al. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. 2017. github.com/zalandoresearch/fashion-mnist, MIT licence; downloaded 2026-09-22.'
    ]]
    report=Path(reports)
    (report/'SUBMISSION_DRAFT.md').write_text('\n\n<!-- PAGE BREAK -->\n\n'.join('\n\n'.join(p) for p in pages)+'\n')
    # Keep explicit page allocation, but do not claim verified PDF page count without rendering.
    chunks=[]
    for page in pages:
        blocks=[]
        for item in page:
            if item.startswith('# '):blocks.append('<h1>'+html.escape(item[2:])+'</h1>')
            elif item.startswith('## '):blocks.append('<h2>'+html.escape(item[3:])+'</h2>')
            elif item.startswith('|'):
                trs=[]
                for i,line in enumerate(item.splitlines()):
                    if i==1:continue
                    tag='th' if i==0 else 'td'
                    trs.append('<tr>'+''.join(f'<{tag}>'+html.escape(c.strip())+f'</{tag}>' for c in line.strip('|').split('|'))+'</tr>')
                blocks.append('<table>'+''.join(trs)+'</table>')
            else:blocks.append('<p>'+html.escape(item.replace('**',''))+'</p>')
        chunks.append('<section>'+''.join(blocks)+'</section>')
    style='@page{size:Letter;margin:.65in}body{font:11pt/1.2 Calibri,Arial,sans-serif;color:#000}section{break-after:page}section:last-child{break-after:auto}h1{font-size:19pt;margin:0 0 12pt}h2{font-size:13pt;margin:12pt 0 6pt}p{margin:0 0 8pt}table{border-collapse:collapse;width:100%;font-size:9.5pt;margin:10pt 0}td,th{border:1px solid #d9d9d9;padding:5pt}th{background:#e6edf5;text-align:left}'
    (report/'SUBMISSION_DRAFT.html').write_text('<!doctype html><meta charset="utf-8"><title>Assignment 2 submission draft</title><style>'+style+'</style>'+''.join(chunks))

