"""Generate reports only from completed, frozen experiment outputs (CPU)."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
from metrics import paired_interval,confusion

ROOT=Path(__file__).resolve().parent
NAMES={'full':'LSNet-T adapted: 7x7 LKP','no_lkp':'No LKP depthwise convolution','small_lkp':'3x3 LKP'}
CLASSES=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def describe(rows,model_key='variant'):
    result=[]
    for condition in ['original','permuted']:
        for model in dict.fromkeys(r[model_key] for r in rows):
            group=[r for r in rows if r['condition']==condition and r[model_key]==model]
            if not group:continue
            row={'condition':condition,'model':model,'n_seeds':len(group)}
            for key in ['accuracy','macro_f1','parameters','tensor_bytes','latency_ms_median','images_per_second','fit_seconds','best_epoch','epochs_run']:
                if all(key in r for r in group):
                    v=[r[key] for r in group];row[key+'_mean']=float(np.mean(v));row[key+'_sd']=float(np.std(v,ddof=1)) if len(v)>1 else 0.
            result.append(row)
    return result

def build(out,reports,cfg):
    read(out/'test_complete.json') # Deliberate fail closed, no placeholder numbers.
    reports.mkdir(parents=True,exist_ok=True)
    rows=read(out/'evaluation_rows.json');baseline=read(out/'baseline_rows.json')
    primary=[r for r in rows if r['analysis']=='primary']
    assert len(primary)==18 and len(rows)==24 and len(baseline)==24
    summary=describe(primary);old=describe(baseline,'model')
    targets=np.load(out/'evaluation_targets.npz');y=targets['labels'];groups=targets['groups']
    pred={r['id']:np.load(out/'evaluation'/r['id']/'predictions.npy') for r in rows}
    paired=[];effects={}
    for condition in cfg['conditions']:
        full=np.array([pred[f'full_{condition}_s{s}']==y for s in cfg['seeds']],float)
        for control in ['no_lkp','small_lkp']:
            alt=np.array([pred[f'{control}_{condition}_s{s}']==y for s in cfg['seeds']],float)
            delta=full-alt;effects[(condition,control)]=delta
            paired.append({'comparison':f'full - {control}','condition':condition,
                           **paired_interval(delta,groups,**{'resamples':cfg['bootstrap']['resamples'],'seed':cfg['bootstrap']['seed']}),
                           'seed_difference_sd_pp':float(delta.mean(1).std(ddof=1)*100),
                           'seed_differences_pp':(delta.mean(1)*100).tolist()})
    interactions=[]
    for control in ['no_lkp','small_lkp']:
        interactions.append({'control':control,'definition':'(full-control) original minus (full-control) permuted',
                             **paired_interval(effects[('original',control)]-effects[('permuted',control)],groups,
                                               cfg['bootstrap']['resamples'],cfg['bootstrap']['seed'])})
    pilots=[]
    for p in read(out/'selection.json')['pilots']:
        pilots.append({k:p[k] for k in ['variant','learning_rate','validation_accuracy','validation_loss','epochs_run','best_epoch','fit_seconds']})
    budget=[]
    for variant in cfg['variants']:
        a=next(r for r in rows if r['variant']==variant and r['analysis']=='budget40')
        b=next(r for r in rows if r['variant']==variant and r['analysis']=='budget100')
        budget.append({'variant':variant,'accuracy_cap40':a['accuracy'],'accuracy_cap100':b['accuracy'],
                       'change_pp':100*(b['accuracy']-a['accuracy']),'best_epoch_cap40':a['best_epoch'],
                       'best_epoch_cap100':b['best_epoch'],'fit_seconds_cap40':a['fit_seconds'],'fit_seconds_cap100':b['fit_seconds']})
    original=[r for r in summary if r['condition']=='original']
    winner=max(original,key=lambda r:r['accuracy_mean'])
    curves=[]
    for r in primary:
        history=read(out/'fits'/r['fit_id']/'history.json');last=history[-1]
        best=history[r['best_epoch']-1]
        curves.append({'id':r['id'],'best_epoch':r['best_epoch'],'epochs_run':r['epochs_run'],
                       'selected_validation_accuracy':best['validation_accuracy'],
                       'last_train_accuracy':last['train_accuracy'],'last_validation_accuracy':last['validation_accuracy'],
                       'last_train_minus_validation_pp':100*(last['train_accuracy']-last['validation_accuracy']),
                       'last_validation_loss_minus_best':last['validation_loss']-best['validation_loss']})
    confusion_rows=[]
    for variant in cfg['variants']:
        matrices=[confusion(y,pred[f'{variant}_original_s{s}']) for s in cfg['seeds']]
        matrix=np.mean(matrices,axis=0);np.savetxt(reports/f'confusion_{variant}.csv',matrix,delimiter=',',fmt='%.6f')
        errors=matrix.copy();np.fill_diagonal(errors,0)
        for idx in np.argsort(errors.ravel())[-3:][::-1]:
            a,b=np.unravel_index(idx,errors.shape)
            confusion_rows.append({'variant':variant,'true_class':CLASSES[a],'predicted_class':CLASSES[b],'mean_count':float(errors[a,b])})
    analysis={'status':'COMPLETED NUMERICAL RESULTS; final document layout and member details require review',
              'common_learning_rate':read(out/'selection.json')['common_learning_rate'],
              'hardware':primary[0]['hardware'],'primary':summary,'baselines':old,'paired':paired,
              'spatial_interactions':interactions,'pilots':pilots,'budget':budget,'learning_curves':curves,
              'confusions':confusion_rows,'observed_winner':winner['model'],
              'uncertainty_note':'Intervals resample paired test groups, conditional on fitted models; three-seed SD is separate. Exploratory contrasts, no multiplicity correction.',
              'test_disclosure':cfg['test_disclosure']}
    write(reports/'analysis.json',analysis)
    for name,data in [('primary_comparison',summary),('baseline_comparison',old),('pilot_results',pilots),('budget_check',budget),('learning_curve_diagnostics',curves),('class_errors',confusion_rows),('paired_contrasts',paired)]:table(reports/f'{name}.csv',data)
    figures(out,reports,primary,summary,budget)
    documents(reports,analysis,cfg)
    from submission_text import create
    create(reports,analysis)
    return analysis

def figures(out,reports,primary,summary,budget):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    fig,axes=plt.subplots(2,3,figsize=(12,6.6),sharex='col')
    for j,variant in enumerate(['full','no_lkp','small_lkp']):
        for row in (r for r in primary if r['variant']==variant and r['condition']=='original'):
            h=read(out/'fits'/row['fit_id']/'history.json');epochs=[x['epoch'] for x in h]
            axes[0,j].plot(epochs,[x['train_loss'] for x in h],alpha=.6,linestyle='--')
            axes[0,j].plot(epochs,[x['validation_loss'] for x in h],label=f"seed {row['seed']}")
            axes[1,j].plot(epochs,[100*x['validation_accuracy'] for x in h])
        axes[0,j].set_title(NAMES[variant]);axes[0,j].legend(fontsize=8)
        axes[1,j].set_xlabel('Epoch');axes[1,j].set_ylim(0,100)
    axes[0,0].set_ylabel('Loss (dashed: training)');axes[1,0].set_ylabel('Validation accuracy (%)')
    fig.tight_layout();fig.savefig(reports/'learning_curves.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4))
    for j,c in enumerate(['original','permuted']):
        rows=[r for r in summary if r['condition']==c];x=np.arange(3)+(j-.5)*.32
        ax.bar(x,[r['accuracy_mean']*100 for r in rows],.32,yerr=[r['accuracy_sd']*100 for r in rows],capsize=4,label=c)
    ax.set_xticks(np.arange(3),['Full 7x7','No DW LKP','3x3 LKP']);ax.set_ylabel('Test accuracy (%)');ax.set_ylim(0,100);ax.legend()
    fig.tight_layout();fig.savefig(reports/'comparison.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4))
    for r in budget:ax.plot([40,100],[r['accuracy_cap40']*100,r['accuracy_cap100']*100],marker='o',label=r['variant'])
    ax.set_xticks([40,100]);ax.set_xlabel('Maximum training epochs (seed 42)');ax.set_ylabel('Accuracy of validation-selected checkpoint (%)');ax.legend()
    fig.tight_layout();fig.savefig(reports/'budget_check.png');plt.close(fig)

def documents(reports,a,cfg):
    primary=[r for r in a['primary'] if r['condition']=='original']
    winner=next(r for r in primary if r['model']==a['observed_winner'])
    full=next(r for r in primary if r['model']=='full')
    cnn=next(r for r in a['baselines'] if r['condition']=='original' and r['model']=='cnn')
    contrasts=[r for r in a['paired'] if r['condition']=='original']
    findings=[]
    for r in contrasts:
        lo,hi=r['conditional_group_ci95_pp']
        conclusion=('positive in this conditional interval' if lo>0 else 'negative in this conditional interval' if hi<0 else 'not resolved by this conditional interval')
        findings.append(f"{r['comparison']}: {r['difference_pp']:+.3f} percentage points (paired test-group 95% interval {lo:+.3f} to {hi:+.3f}); the contrast is {conclusion}. Training-seed difference SD is {r['seed_difference_sd_pp']:.3f} points.")
    lines=['# Assignment 2 — LSNet on small grayscale images','',
           '**Authors:** [Member 1], [Member 2], [Member 3]. Actual contributions and paper reservation: [complete before submission].','',
           '## Claim and method',
           'LSNet proposes large-kernel perception (LKP) to generate spatially adaptive weights, then small-kernel aggregation (SKA) to collect local features. The computational intuition is to obtain context without using a large dynamic aggregation neighborhood. The official implementation and its SKA operator are retained in our adaptation.',
           'Primary: Wang et al., LSNet: See Large, Focus Small, CVPR 2025. Supporting: Nguyen et al., An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels, ICLR 2025.',
           '## Specific critique and testable gap',
           'The paper already tests kernel sizes and component removal (Table 7); absence of ablation is not our criticism. Its ImageNet ablation uses a 100-epoch analysis setting. Tables 5–6 compare token mixers and dynamic convolution, and Table 9 tests transfer to other backbones. These support the reported setting but do not establish that large spatial perception is beneficial for every resolution or task. Our gap is transfer to small, single-channel images under an explicitly limited training budget.',
           'The biological motivation is an analogy, not evidence of a biological mechanism. Throughput is conditional on hardware, implementation, batch and precision. Comparing an 11-million-parameter adaptation with our small Assignment 1 CNN also confounds capacity; therefore the causal contrast is within LSNet, and the old CNN is a practical reference.',
           'The supporting pixel-transformer paper tests pixel tokens and pixel permutation, showing that local patch grouping is not a universal prerequisite. It retains positional information and uses different architectures and training. It motivates a spatial-order stress test; it does not predict that LSNet must be permutation invariant.',
           '## Reality check: data and design',
           'We reuse the audited Fashion-MNIST split: 53,979 training, 5,998 validation and 10,000 official test images; 28x28 grayscale, pixels /255, no augmentation, same seeds 42–44 and fixed permutation as Assignment 1. Grayscale is one intensity channel, not binary-valued images. It removes color cues and keeps repeated experiments feasible; clothing shape/texture still presents class ambiguity. It connects to our shared CV methodology, not to a claim that clothing images represent digital humans or face deepfakes.',
           f"Official LSNet-T widths/depths are retained. We change the stem strides to preserve 28x28 detail, input to one channel and output to ten classes. Training is from scratch with Adam, batch 256, weight decay 1e-4, FP32. Six validation-only pilot fits choose one shared learning rate ({a['common_learning_rate']:g}) using the full model. Main fits use at most 40 epochs and validation-loss patience 8. This is a controlled adaptation, not an ImageNet recipe reproduction.",
           'Full 7x7 LKP is compared with removal of only its depthwise convolution (BatchNorm retained) and replacement by 3x3. Both retain dynamic SKA and other spatial modules. Each variant runs on original and fixed-permuted images. A secondary seed-42, 100-epoch run per variant compares validation-selected checkpoints within caps of 40 and 100. All settings and checkpoints are frozen before Assignment 2 test-array loading.',
           '## Results and explanation','',
           '| Model | Accuracy %, mean ± seed SD | Macro-F1 | Fit min | Parameters | Batch-256 latency ms |',
           '|---|---:|---:|---:|---:|---:|']
    for r in primary:
        lines.append(f"| {NAMES[r['model']]} | {r['accuracy_mean']*100:.3f} ± {r['accuracy_sd']*100:.3f} | {r['macro_f1_mean']:.4f} | {r['fit_seconds_mean']/60:.2f} | {r['parameters_mean']:,.0f} | {r['latency_ms_median_mean']:.3f} |")
    lines+=['',f"The observed highest mean is {NAMES[winner['model']]} ({winner['accuracy_mean']*100:.3f}%). A point-estimate winner alone does not establish a mechanism.",*findings,
            f"The original-image Assignment 1 CNN scores {cnn['accuracy_mean']*100:.3f}% with {cnn['parameters_mean']:,.0f} parameters; adapted full LSNet changes accuracy by {(full['accuracy_mean']-cnn['accuracy_mean'])*100:+.3f} points. Both are re-timed on the same {a['hardware']['gpu']} with resident inputs, FP32 and unfused evaluation. Training times include train and validation, but exclude initialization, checkpoint I/O, JIT warm-up and hyperparameter search; old training times are not presented as a same-hardware comparison.",
            'The within-LSNet interventions test whether added spatial context is useful under this protocol. A positive contrast supports its usefulness here; an unresolved or negative contrast limits that claim. Neither outcome alone identifies shape versus texture as the cause. Confusion matrices show which clothing classes are confused but cannot establish a causal feature explanation.']
    for r in a['spatial_interactions']:
        lo,hi=r['conditional_group_ci95_pp'];lines.append(f"Spatial interaction for {r['control']}: original full-minus-control gap minus permuted gap = {r['difference_pp']:+.3f} points ({lo:+.3f}, {hi:+.3f}). Permutation preserves pixel information but changes neighborhoods throughout the network, so it is not an LKP-only intervention.")
    for r in a['budget']:
        lines.append(f"Budget diagnostic, {r['variant']}: cap-40 {r['accuracy_cap40']*100:.3f}% → cap-100 {r['accuracy_cap100']*100:.3f}% ({r['change_pp']:+.3f} points); selected epochs {r['best_epoch_cap40']} → {r['best_epoch_cap100']}. This is one seed, not a general convergence guarantee.")
    lines+=['## Training walkthrough and learning curves',
            '1. Check archive hashes and disjoint development IDs, then reuse the fixed split and scaling. This prevents accidental data changes from masquerading as a method effect.',
            '2. A minibatch passes through the adapted stem, LSConv/attention blocks and classifier. Cross-entropy measures class-prediction error; backpropagation computes derivatives; Adam updates training parameters. Validation runs in evaluation mode without gradient updates, so it monitors generalization rather than training the network.',
            '3. Save the checkpoint only when validation loss improves by more than 0.0001; stop main training after eight non-improving epochs. Accuracy chooses the common pilot rate, while validation loss selects checkpoints. These are distinct decisions, both made without Assignment 2 test tuning.',
            '4. Inspect training versus validation loss and selected epoch. Falling training loss with rising validation loss is consistent with overfitting; both remaining high suggests optimization or capacity problems. Minibatch training accuracy and evaluation-mode validation accuracy are not directly identical measurements because BatchNorm modes differ.',
            '5. Freeze model identities, evaluate the test once as a planned sweep, compute seed variation and paired group uncertainty, and measure inference on the same GPU. Save raw predictions so every metric can be independently recomputed.']
    for v in cfg['variants']:
        c=[r for r in a['learning_curves'] if r['id'].startswith(v+'_original')]
        lines.append(f"{v}: selected epochs {[r['best_epoch'] for r in c]}; stopped epochs {[r['epochs_run'] for r in c]}; mean last-epoch training minus validation accuracy = {np.mean([r['last_train_minus_validation_pp'] for r in c]):+.2f} points. See learning_curves.png; use validation-loss trajectory, not this gap alone, to assess overfitting.")
    for v in cfg['variants']:
        p=[r for r in a['pilots'] if r['variant']==v]
        lines.append(f"{v} pilot validation accuracies: "+'; '.join(f"LR {r['learning_rate']:g}: {r['validation_accuracy']*100:.3f}% (selected epoch {r['best_epoch']})" for r in p)+'. Non-full pilot optima are sensitivity diagnostics; main contrasts keep the full-selected common rate.')
    lines+=['## Limitations, next steps and disclosure',cfg['test_disclosure'],
            'Three fitted seeds are a small training sample. Paired bootstrap intervals cover test-group resampling conditional on these fits, not all sources of uncertainty; exploratory contrasts are not multiplicity-adjusted. A common learning rate improves attribution but need not optimize each variant. The stem/input/head and dataset changes limit direct comparison with the paper’s numbers. The 100-epoch diagnostic does not reproduce its optimizer, schedule, data or full recipe.',
            'Next: evaluate a reserved new dataset, repeat the budget check across seeds, and compare a capacity-matched convolutional control. These would distinguish transfer limits, budget effects and capacity effects. Do not claim they have been completed.',
            'AI disclosure: AI assisted source inspection, implementation, experiment orchestration and draft writing. Members must verify sources, run/inspect the code, understand the analysis, and replace contribution placeholders with work actually done.',
            'Administrative checks: confirm paper reservation/eligibility with instructor; fill names and actual contributions; every member speaks; prepare a question for another group; review the optional-repository versus rubric ambiguity. No repository is published by this pipeline.',
            '## Sources',
            '- LSNet paper: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_LSNet_See_Large_Focus_Small_CVPR_2025_paper.html',
            '- Official code (pinned commit in upstream.json): https://github.com/THU-MIG/lsnet',
            '- Pixel-transformer paper: https://proceedings.iclr.cc/paper_files/paper/2025/hash/959f70ee50044bed305e48e3484005a7-Abstract-Conference.html',
            '- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist (MIT; original release 2017; project download 2026-09-22).']
    text=''
    for i,line in enumerate(lines):
        text+=line+ ('\n' if line.startswith('|') and i+1<len(lines) and lines[i+1].startswith('|') else '\n\n')
    (reports/'FULL_SUMMARY.md').write_text(text)
    # Machine-readable payload for local polished DOCX/PPTX authoring.
    write(reports/'presentation_payload.json',{'analysis':a,'findings':findings,'full_summary':'FULL_SUMMARY.md'})
    outline=['# 10-minute talk + 5-minute Q&A','Each member must speak; assign actual speakers before submission.',
             '1. Question and paper claim — shared CV motivation; small-image transfer gap; main/supporting citations.',
             '2. How LSConv works — static large-kernel perception generates dynamic weights; small-kernel aggregation uses them.',
             '3. Critical reading — Tables 5–7 and 9 already provide controls; explain limits of resolution transfer and hardware-specific speed.',
             '4. Reality-check protocol — same A1 split/seeds/scaling, exact adaptation, common LR, architecture controls, test reuse disclosure.',
             '5. Results — original and permuted accuracy, within-LSNet contrasts and uncertainty, same-GPU latency/size. See primary_comparison.csv.',
             '6. Why — interpret actual contrasts and spatial interaction; avoid attributing all permutation damage to LKP.',
             '7. Learning process — actual curves, pilot choices and budget40→100; distinguish generalization from optimization.',
             '8. Verdict, limitations, next steps, AI disclosure and actual contributions. Claims must match analysis.json.']
    (reports/'SLIDE_OUTLINE.md').write_text('\n\n'.join(outline)+'\n')
    # A self-contained browser-readable full report; not represented as the 3-page submission.
    body=''.join('<p>'+html.escape(line)+'</p>' for line in lines)
    (reports/'FULL_SUMMARY.html').write_text('<!doctype html><meta charset="utf-8"><title>Assignment 2 results</title><style>body{max-width:1000px;margin:40px auto;font:16px/1.5 system-ui}p{white-space:pre-wrap}</style>'+body)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--a1-root',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'outputs');args=parser.parse_args()
    build(args.output,ROOT/'reports',read(ROOT/'protocol.json'))
    print('Measured summaries, figures, training walkthrough and presentation payload generated.')

if __name__=='__main__':main()
