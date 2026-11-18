"""Single-process, resumable Assignment 2 study. All GPU work runs in one job."""
import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from metrics import accuracy_score, f1_score

from adapter import build_model, load_official_ska, reference_ska, shape_trace

ROOT = Path(__file__).resolve().parent


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save_torch(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def fingerprints():
    files = ['protocol.json', 'run_study.py', 'adapter.py', 'metrics.py', 'references/assignment1_study.py',
             'vendor/lsnet/model/lsnet.py', 'vendor/lsnet/model/ska.py']
    hashes = {f: digest(ROOT/f) for f in files}
    return {'files': hashes, 'sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def hardware(device):
    return {'host': platform.node(), 'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'job_id': os.getenv('SLURM_JOB_ID'), 'threads': torch.get_num_threads()}


def model_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def rng_state(generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'generator': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    generator.set_state(state['generator'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


@torch.no_grad()
def predict(model, x, y, batch):
    model.eval()
    predictions, loss = [], 0.
    for i in range(0, len(y), batch):
        logits = model(x[i:i+batch])
        if not torch.isfinite(logits).all():
            raise RuntimeError('Non-finite evaluation logits')
        predictions.append(logits.argmax(1).cpu().numpy())
        loss += nn.functional.cross_entropy(logits, y[i:i+batch], reduction='sum').item()
    return np.concatenate(predictions), loss / len(y)


def tensor_images(x, device, permutation=None):
    array = x.reshape(-1, 784)
    if permutation is not None:
        array = array[:, permutation]
    return torch.tensor(array.copy(), dtype=torch.float32, device=device).reshape(-1,1,28,28)/255.


def get_data(a1, cfg, device):
    path = a1/'data/prepared/train_val.npz'
    if digest(path) != cfg['train_val_sha256']:
        raise RuntimeError('Audited training archive hash mismatch')
    with np.load(path) as z:
        raw = {k:z[k] for k in z.files}
    if set(raw['train_ids']) & set(raw['val_ids']):
        raise RuntimeError('Training and validation IDs overlap')
    if sorted(raw['permutation'].tolist()) != list(range(784)):
        raise RuntimeError('Invalid fixed permutation')
    result = {}
    for condition in cfg['conditions']:
        perm = raw['permutation'] if condition == 'permuted' else None
        result[condition] = (tensor_images(raw['x_train'],device,perm),
                             torch.tensor(raw['y_train'],dtype=torch.long,device=device),
                             tensor_images(raw['x_val'],device,perm),
                             torch.tensor(raw['y_val'],dtype=torch.long,device=device))
    return result, raw['permutation']


def fit(spec, data, cfg, out, signature, device, *, max_new_epochs=None):
    folder = out/'fits'/spec['id']
    folder.mkdir(parents=True,exist_ok=True)
    done = folder/'validation.json'
    expected = {'source':signature,'spec':spec}
    if done.exists():
        saved = read(done)
        if saved['identity'] != expected:
            raise RuntimeError(f'Stale fit: {spec["id"]}')
        if digest(folder/'best.pt') != saved['checkpoint_sha256']:
            raise RuntimeError('Completed checkpoint modified')
        return saved
    seed_all(spec['seed'])
    model = build_model(spec['variant'],cfg['backend'],cfg['scale']).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=spec['lr'],weight_decay=cfg['weight_decay'])
    generator = torch.Generator(device='cpu').manual_seed(spec['seed'])
    x,y,vx,vy = data
    history, best, best_epoch, stale, elapsed = [], float('inf'), 0, 0, 0.
    resume = folder/'resume.pt'
    if resume.exists():
        saved = torch.load(resume,map_location='cpu',weights_only=False)
        if saved['identity'] != expected:
            raise RuntimeError(f'Resume identity changed: {spec["id"]}')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        history,best,best_epoch,stale,elapsed = [saved[k] for k in ['history','best','best_epoch','stale','elapsed']]
        restore_rng(saved['rng'],generator)
    else:
        # Warm eval-only before timing. Do not update BN statistics or optimizer.
        model.eval()
        with torch.no_grad():
            model(x[:min(cfg['batch_size'],len(x))])
        sync(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    new_epochs = 0
    for epoch in range(len(history)+1,spec['epochs']+1):
        if spec['early_stop'] and stale >= cfg['patience']:
            break
        model.train()
        order = torch.randperm(len(x),generator=generator).to(device)
        total, correct = 0., 0
        sync(device)
        begin = time.perf_counter()
        for j in range(0,len(x),cfg['batch_size']):
            idx = order[j:j+cfg['batch_size']]
            if len(idx)<2:
                raise RuntimeError('Singleton batch incompatible with head BatchNorm')
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx])
            loss = nn.functional.cross_entropy(logits,y[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss: {spec["id"]}, epoch {epoch}')
            loss.backward()
            if j==0 and not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError('Non-finite gradients')
            optimizer.step()
            total += loss.item()*len(idx)
            correct += (logits.argmax(1)==y[idx]).sum().item()
        sync(device)
        train_seconds=time.perf_counter()-begin
        vp,vl=predict(model,vx,vy,cfg['evaluation_batch_size'])
        sync(device)
        epoch_seconds=time.perf_counter()-begin
        elapsed+=epoch_seconds
        history.append({'epoch':epoch,'train_loss':total/len(x),'train_accuracy':correct/len(x),
                        'validation_loss':vl,'validation_accuracy':float((vp==vy.cpu().numpy()).mean()),
                        'train_seconds':train_seconds,'train_validation_seconds':epoch_seconds})
        if vl < best-cfg['min_delta']:
            best,best_epoch,stale=vl,epoch,0
            save_torch(folder/'best.pt',model_state(model))
        else:
            stale+=1
        if spec['kind']=='budget' and epoch in cfg['budget_diagnostic']['compare_caps']:
            # Keep each cap's validation-selected checkpoint, not the last epoch.
            state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            save_torch(folder/f'best_cap{epoch}.pt',state)
            write(folder/f'cap{epoch}.json',{'cap':epoch,'best_epoch':best_epoch,'validation_loss':best,
                                          'fit_seconds':elapsed,'source':signature})
        write(folder/'history.json',history)
        save_torch(resume,{'identity':expected,'model':model_state(model),'optimizer':optimizer.state_dict(),
                           'history':history,'best':best,'best_epoch':best_epoch,'stale':stale,
                           'elapsed':elapsed,'rng':rng_state(generator)})
        write(out/'progress.json',{'stage':'training','fit':spec['id'],'epoch':epoch,
                                  'max_epochs':spec['epochs'],'validation':history[-1], 'updated_utc':stamp()})
        if epoch==1 or epoch%5==0 or (spec['early_stop'] and stale>=cfg['patience']):
            print(json.dumps({'fit':spec['id'],**history[-1]}),flush=True)
        new_epochs+=1
        if max_new_epochs and new_epochs>=max_new_epochs and epoch<spec['epochs']:
            return None
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True))
    vp,vl=predict(model,vx,vy,cfg['evaluation_batch_size'])
    np.save(folder/'validation_predictions.npy',vp)
    state=model.state_dict()
    result={'identity':expected,'fit_id':spec['id'],'variant':spec['variant'],'seed':spec['seed'],
            'condition':spec['condition'],'learning_rate':spec['lr'],'kind':spec['kind'],
            'epochs_run':len(history),'best_epoch':best_epoch,'validation_loss':vl,
            'validation_accuracy':float((vp==vy.cpu().numpy()).mean()),
            'fit_seconds':elapsed,'training_seconds':sum(h['train_seconds'] for h in history),
            'parameters':sum(p.numel() for p in model.parameters()),
            'tensor_bytes':sum(v.numel()*v.element_size() for v in state.values()),
            'peak_allocated_mb':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
            'checkpoint_sha256':digest(folder/'best.pt'),'completed_utc':stamp(),'hardware':hardware(device)}
    write(done,result)
    # Resume files intentionally retained for audit and exact restart evidence.
    return result


def make_spec(kind,variant,condition,seed,lr,epochs,early_stop=True):
    return {'id':f'{kind}_{variant}_{condition}_s{seed}_lr{lr:g}', 'kind':kind,
            'variant':variant,'condition':condition,'seed':seed,'lr':lr,
            'epochs':epochs,'early_stop':early_stop}


def official_check(device):
    seed_all(711)
    rows=[]
    for c,wc,h,w in [(8,1,7,5),(16,2,14,14),(24,3,4,4)]:
        x=torch.randn(2,c,h,w,device=device,requires_grad=True)
        weights=torch.randn(2,wc,9,h,w,device=device,requires_grad=True)
        ref=reference_ska(x,weights)
        actual=load_official_ska().SKA()(x,weights)
        grad=torch.randn_like(ref)
        rg=torch.autograd.grad(ref,(x,weights),grad)
        ag=torch.autograd.grad(actual,(x,weights),grad)
        torch.testing.assert_close(actual,ref,rtol=1e-4,atol=2e-5)
        for a,b in zip(ag,rg):
            torch.testing.assert_close(a,b,rtol=1e-4,atol=5e-5)
        rows.append({'shape':[2,c,h,w],'forward_and_backward':'pass',
                     'max_error':float((actual-ref).abs().max())})
    return rows


def train_all(a1,cfg,out,signature,device):
    datasets,permutation=get_data(a1,cfg,device)
    pilots=[]
    for variant in cfg['variants']:
        for lr in cfg['learning_rates']:
            spec=make_spec('pilot',variant,'original',cfg['pilot_seed'],lr,cfg['max_epochs'])
            pilots.append(fit(spec,datasets['original'],cfg,out,signature,device))
    winner=min((p for p in pilots if p['variant']=='full'),
               key=lambda p:(-p['validation_accuracy'],p['validation_loss'],p['learning_rate']))
    lr=winner['learning_rate']
    selection={'source':signature,'common_learning_rate':lr,'rule':cfg['learning_rate_rule'],
               'selected_full_fit':winner['fit_id'],'pilots':pilots,'test_used':False}
    selection_path=out/'selection.json'
    if selection_path.exists() and read(selection_path)!=selection:
        raise RuntimeError('Selection changed on resume')
    write(selection_path,selection)
    evaluation=[]
    for condition in cfg['conditions']:
        for variant in cfg['variants']:
            for seed in cfg['seeds']:
                kind='pilot' if condition=='original' and seed==cfg['pilot_seed'] else 'primary'
                spec=make_spec(kind,variant,condition,seed,lr,cfg['max_epochs'])
                fit(spec,datasets[condition],cfg,out,signature,device)
                evaluation.append({'id':f'{variant}_{condition}_s{seed}','variant':variant,'condition':condition,
                                   'seed':seed,'analysis':'primary','fit_id':spec['id'],
                                   'checkpoint':f'fits/{spec["id"]}/best.pt'})
    for variant in cfg['variants']:
        budget=cfg['budget_diagnostic']
        spec=make_spec('budget',variant,budget['condition'],budget['seed'],lr,budget['epochs'],False)
        fit(spec,datasets[budget['condition']],cfg,out,signature,device)
        for cap in budget['compare_caps']:
            evaluation.append({'id':f'{variant}_budget{cap}_s{budget["seed"]}',
                               'variant':variant,'condition':budget['condition'],'seed':budget['seed'],
                               'analysis':f'budget{cap}','fit_id':spec['id'],
                               'checkpoint':f'fits/{spec["id"]}/best_cap{cap}.pt'})
    frozen={'source':signature,'selection_sha256':digest(selection_path),
            'evaluation':[{**e,'checkpoint_sha256':digest(out/e['checkpoint'])} for e in evaluation],
            'test_data_loaded':False,'test_reuse_disclosure':cfg['test_disclosure']}
    path=out/'frozen_experiment.json'
    if path.exists() and read(path)!=frozen:
        raise RuntimeError('Frozen checkpoints or evaluation identities changed')
    write(path,frozen)
    del datasets
    if device.type=='cuda':
        torch.cuda.empty_cache()
    return frozen,permutation


@torch.no_grad()
def benchmark(model,x,cfg,device):
    settings=cfg['timing']
    model.eval()
    batch=x[:settings['batch']]
    for _ in range(settings['warmup']):model(batch)
    sync(device)
    seconds=[]
    for _ in range(settings['repeats']):
        sync(device); started=time.perf_counter();model(batch);sync(device)
        seconds.append(time.perf_counter()-started)
    return {'benchmark_batch':len(batch),'latency_ms_median':float(np.median(seconds)*1000),
            'latency_ms_p25':float(np.quantile(seconds,.25)*1000),
            'latency_ms_p75':float(np.quantile(seconds,.75)*1000),
            'images_per_second':float(len(batch)/np.median(seconds))}


def evaluate_all(a1,cfg,out,signature,device,frozen,permutation):
    if frozen['source']!=signature or digest(out/'selection.json')!=frozen['selection_sha256']:
        raise RuntimeError('Experiment not frozen under current source')
    for entry in frozen['evaluation']:
        if digest(out/entry['checkpoint'])!=entry['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed before test')
    test=a1/'data/prepared/test.npz'
    if digest(test)!=cfg['test_sha256']:raise RuntimeError('Test archive hash mismatch')
    write(out/'test_evaluation_started.json',{'utc':stamp(),'freeze_sha256':digest(out/'frozen_experiment.json')})
    with np.load(test) as z:
        raw,y,groups=z['x_test'],z['y_test'],z['groups']
    target=torch.tensor(y,dtype=torch.long,device=device)
    tensors={c:tensor_images(raw,device,permutation if c=='permuted' else None) for c in cfg['conditions']}
    rows=[]
    for entry in frozen['evaluation']:
        model=build_model(entry['variant'],cfg['backend'],cfg['scale']).to(device)
        state=torch.load(out/entry['checkpoint'],map_location='cpu',weights_only=True)
        model.load_state_dict(state)
        pred,loss=predict(model,tensors[entry['condition']],target,cfg['evaluation_batch_size'])
        folder=out/'evaluation'/entry['id'];folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'predictions.npy',pred)
        fit_result=read(out/'fits'/entry['fit_id']/'validation.json')
        row={**entry,'accuracy':float(accuracy_score(y,pred)),
             'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
             'test_loss':loss,'parameters':sum(p.numel() for p in model.parameters()),
             'tensor_bytes':sum(v.numel()*v.element_size() for v in state.values()),
             'fit_seconds':fit_result['fit_seconds'],'best_epoch':fit_result['best_epoch'],
             'epochs_run':fit_result['epochs_run'],'hardware':hardware(device),
             **benchmark(model,tensors[entry['condition']],cfg,device)}
        if entry['analysis'].startswith('budget'):
            cap=int(entry['analysis'].replace('budget',''))
            cap_result=read(out/'fits'/entry['fit_id']/f'cap{cap}.json')
            row.update(fit_seconds=cap_result['fit_seconds'],best_epoch=cap_result['best_epoch'],epochs_run=cap)
        write(folder/'metrics.json',row);rows.append(row)
        del model
    # Load the exact preserved Assignment 1 model definitions and checkpoints.
    spec=importlib.util.spec_from_file_location('assignment1_reference',ROOT/'references/assignment1_study.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    baseline=[]
    for condition in cfg['conditions']:
        for name in ['centroid','logistic','mlp','cnn']:
            for seed in cfg['seeds']:
                path=a1/'outputs/runs'/f'{name}_{condition}_{seed}'/'model.pt'
                model=old.make_model(name).to(device)
                state=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(state)
                pred,loss=predict(model,tensors[condition],target,cfg['evaluation_batch_size'])
                row={'model':name,'condition':condition,'seed':seed,
                     'accuracy':float(accuracy_score(y,pred)),
                     'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
                     'parameters':sum(p.numel() for p in model.parameters()),
                     'tensor_bytes':sum(v.numel()*v.element_size() for v in state.values()),
                     'checkpoint_sha256':digest(path),**benchmark(model,tensors[condition],cfg,device)}
                folder=out/'baseline'/f'{name}_{condition}_{seed}';folder.mkdir(parents=True,exist_ok=True)
                np.save(folder/'predictions.npy',pred);write(folder/'metrics.json',row);baseline.append(row)
                del model
    write(out/'evaluation_rows.json',rows);write(out/'baseline_rows.json',baseline)
    # Save labels only after freeze/evaluation; report tools never train from these.
    np.savez_compressed(out/'evaluation_targets.npz',labels=y,groups=groups)
    write(out/'test_complete.json',{'utc':stamp(),'source':signature,'freeze_sha256':digest(out/'frozen_experiment.json'),
                                   'evaluation_rows':len(rows),'baseline_rows':len(baseline)})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--a1-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'outputs')
    args=p.parse_args()
    cfg=read(ROOT/'protocol.json');out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(int(os.getenv('OMP_NUM_THREADS','4')))
    if not torch.cuda.is_available():raise RuntimeError('Formal pipeline requires allocated CUDA GPU')
    for condition in cfg['conditions']:
        for name in ['centroid','logistic','mlp','cnn']:
            for seed in cfg['seeds']:
                path=args.a1_root/'outputs/runs'/f'{name}_{condition}_{seed}'/'model.pt'
                if not path.is_file():raise RuntimeError(f'Missing preserved baseline: {path}')
    upstream=read(ROOT/'upstream.json')
    for name,sha in upstream['files'].items():
        if digest(ROOT/'vendor/lsnet'/name)!=sha:raise RuntimeError(f'Upstream source mismatch: {name}')
    device=torch.device('cuda');sig=fingerprints()
    preflight=out/'preflight.json'
    if preflight.exists() and read(preflight)['source']!=sig:raise RuntimeError('Source changed after preflight')
    write(preflight,{'source':sig,'hardware':hardware(device),'ska_checks':official_check(device)})
    freeze,permutation=train_all(args.a1_root.resolve(),cfg,out,sig,device)
    if (out/'test_complete.json').exists():
        complete=read(out/'test_complete.json')
        if complete['source']!=sig or complete['freeze_sha256']!=digest(out/'frozen_experiment.json'):
            raise RuntimeError('Completed evaluation no longer matches frozen study')
    else:
        evaluate_all(args.a1_root.resolve(),cfg,out,sig,device,freeze,permutation)
    write(out/'progress.json',{'stage':'GPU work complete','updated_utc':stamp(), 'unique_fits':len(list((out/'fits').glob('*/validation.json')))})
    print('All GPU work completed in one scheduler job.',flush=True)


if __name__=='__main__':main()
