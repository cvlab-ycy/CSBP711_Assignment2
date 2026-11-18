"""Independent numerical and provenance audit; exits nonzero on a failed check."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def verify(root=ROOT):
    out=root/'outputs';reports=root/'reports';cfg=read(root/'protocol.json')
    complete=read(out/'test_complete.json');freeze=read(out/'frozen_experiment.json')
    assert complete['freeze_sha256']==digest(out/'frozen_experiment.json')
    assert freeze['selection_sha256']==digest(out/'selection.json')
    for name,sha in freeze['source']['files'].items():assert digest(root/name)==sha,name
    fits=list((out/'fits').glob('*/validation.json'))
    assert len(fits)==cfg['fit_count']['unique_total']==24
    assert len(freeze['evaluation'])==24
    expected={(v,c,s) for v in cfg['variants'] for c in cfg['conditions'] for s in cfg['seeds']}
    rows=read(out/'evaluation_rows.json');primary=[r for r in rows if r['analysis']=='primary']
    assert {(r['variant'],r['condition'],r['seed']) for r in primary}==expected
    assert len(primary)==18
    targets=np.load(out/'evaluation_targets.npz');y=targets['labels'];assert len(y)==10000
    checked=0
    for entry in freeze['evaluation']:
        assert digest(out/entry['checkpoint'])==entry['checkpoint_sha256'],entry['id']
        row=next(r for r in rows if r['id']==entry['id'])
        folder=out/'evaluation'/entry['id'];p=np.load(folder/'predictions.npy')
        assert p.shape==y.shape and np.isin(p,np.arange(10)).all()
        # Deliberately do not call the runner's metric implementation.
        acc=sum(int(a==b) for a,b in zip(y,p))/len(y)
        f=[]
        for c in range(10):
            tp=np.sum((y==c)&(p==c));fp=np.sum((y!=c)&(p==c));fn=np.sum((y==c)&(p!=c))
            f.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.)
        assert abs(row['accuracy']-acc)<1e-12
        assert abs(row['macro_f1']-np.mean(f))<1e-12
        assert row['latency_ms_median']>0 and row['fit_seconds']>0
        checked+=1
    analysis=read(reports/'analysis.json')
    for row in analysis['primary']:
        group=[r for r in primary if r['variant']==row['model'] and r['condition']==row['condition']]
        assert len(group)==3
        assert abs(row['accuracy_mean']-np.mean([r['accuracy'] for r in group]))<1e-12
        assert abs(row['accuracy_sd']-np.std([r['accuracy'] for r in group],ddof=1))<1e-12
    baseline=read(out/'baseline_rows.json');assert len(baseline)==24
    for row in baseline:
        p=np.load(out/'baseline'/f"{row['model']}_{row['condition']}_{row['seed']}"/'predictions.npy')
        assert abs(row['accuracy']-np.mean(y==p))<1e-12
    import csv
    with (root/'references/assignment1_comparison.csv').open() as f:old=list(csv.DictReader(f))
    for oldrow in old:
        now=next(r for r in analysis['baselines'] if r['model']==oldrow['model'] and r['condition']==oldrow['condition'])
        # Permit <=1 prediction per run of device-dependent roundoff, otherwise investigate.
        assert abs(now['accuracy_mean']-float(oldrow['accuracy_mean']))<=.000100001,oldrow
    rates={read(p)['learning_rate'] for p in fits if read(p)['kind'] in ['primary','budget']}
    assert rates=={analysis['common_learning_rate']}
    jobs={r['hardware']['job_id'] for r in rows};assert len(jobs)==1 and None not in jobs
    report={'status':'PASS','unique_fits':len(fits),'evaluations_independently_recomputed':checked,
            'baseline_evaluations_checked':24,'source_and_checkpoint_hashes':'PASS',
            'common_rate':'PASS','single_gpu_job':list(jobs)[0],
            'manual_remaining':['Three-page layout and slides visual QA','Names and actual contributions','Paper reservation and instructor repository policy']}
    (reports/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return report

if __name__=='__main__':verify()
