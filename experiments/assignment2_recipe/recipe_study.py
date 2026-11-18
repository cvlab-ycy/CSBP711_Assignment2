"""Supplementary Assignment 2 recipe check. Single-process, resumable; all GPU work runs in one job.

Trains the Assignment 1 CNN and the adapted LSNet variants under a literature-style recipe
(crop+flip augmentation, AdamW, warmup+cosine, 100 epochs) on the audited Assignment 1 split.
Every checkpoint is hashed and frozen before the test arrays are loaded.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
A2 = ROOT.parent/'assignment2_lsnet'
sys.path.insert(0, str(A2))
from adapter import build_model  # noqa: E402
from metrics import accuracy_score, f1_score, paired_interval  # noqa: E402
from run_study import (digest, hardware, model_state, official_check, predict, read,  # noqa: E402
                       restore_rng, rng_state, save_torch, seed_all, stamp, sync, tensor_images, write)

SOURCES = {'protocol.json': ROOT/'protocol.json', 'recipe_study.py': ROOT/'recipe_study.py',
           'a2/adapter.py': A2/'adapter.py', 'a2/run_study.py': A2/'run_study.py', 'a2/metrics.py': A2/'metrics.py',
           'a2/references/assignment1_study.py': A2/'references/assignment1_study.py',
           'a2/vendor/lsnet/model/lsnet.py': A2/'vendor/lsnet/model/lsnet.py',
           'a2/vendor/lsnet/model/ska.py': A2/'vendor/lsnet/model/ska.py'}
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def fingerprints():
    hashes = {k: digest(v) for k, v in SOURCES.items()}
    return {'files': hashes, 'sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def load_a1():
    spec = importlib.util.spec_from_file_location('assignment1_reference', A2/'references/assignment1_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(name, cfg, a1):
    return a1.make_model('cnn') if name == 'cnn' else build_model(name, cfg['backend'], cfg['scale'])


def get_data(a1_root, cfg, device):
    path = a1_root/'data/prepared/train_val.npz'
    if digest(path) != cfg['train_val_sha256']:
        raise RuntimeError('Audited training archive hash mismatch')
    with np.load(path) as z:
        x, y, vx, vy = z['x_train'], z['y_train'], z['x_val'], z['y_val']
    return (tensor_images(x, device), torch.tensor(y, dtype=torch.long, device=device),
            tensor_images(vx, device), torch.tensor(vy, dtype=torch.long, device=device))


def augment(batch, generator, pad):
    """Random 28x28 crop from zero padding plus horizontal flip; draws come from the seeded CPU generator."""
    n, device = batch.shape[0], batch.device
    padded = nn.functional.pad(batch[:, 0], (pad, pad, pad, pad))
    oy = torch.randint(0, 2*pad+1, (n,), generator=generator).to(device)
    ox = torch.randint(0, 2*pad+1, (n,), generator=generator).to(device)
    flip = (torch.rand(n, generator=generator) < 0.5).to(device)
    offsets = torch.arange(28, device=device)
    rows = (oy[:, None]+offsets)[:, :, None]
    cols = (ox[:, None]+offsets)[:, None, :]
    crops = padded[torch.arange(n, device=device)[:, None, None], rows, cols]
    return torch.where(flip[:, None, None], crops.flip(-1), crops).unsqueeze(1)


def lr_factor(step, recipe, steps_per_epoch, epochs):
    if recipe['schedule'] == 'constant':
        return 1.0
    warm = recipe['warmup_epochs']*steps_per_epoch
    total = epochs*steps_per_epoch
    if step < warm:
        return (step+1)/warm
    return 0.5*(1+math.cos(math.pi*(step-warm)/max(1, total-warm)))


def make_spec(model, recipe, seed, lr):
    return {'id': f'{model}_{recipe}_s{seed}', 'model': model, 'recipe': recipe, 'seed': seed, 'lr': lr}


def plan(cfg, seeds):
    # CNN fits first (minutes); LSNet seed-major so partial progress stays balanced across variants.
    specs = [make_spec('cnn', r, s, cfg['learning_rates']['cnn']) for s in seeds for r in cfg['cnn_recipes']]
    specs += [make_spec(v, r, s, cfg['learning_rates']['lsnet'])
              for s in seeds for r in cfg['lsnet_recipes'] for v in cfg['lsnet_variants']]
    return specs


def fit(spec, data, cfg, out, signature, device, a1, epochs):
    recipe = cfg['recipes'][spec['recipe']]
    folder = out/'fits'/spec['id']
    folder.mkdir(parents=True, exist_ok=True)
    expected = {'source': signature, 'spec': spec, 'epochs': epochs}
    done = folder/'fit.json'
    if done.exists():
        saved = read(done)
        if saved['identity'] != expected:
            raise RuntimeError(f'Stale fit: {spec["id"]}')
        for kind in ('best', 'final'):
            if digest(folder/f'{kind}.pt') != saved[f'{kind}_sha256']:
                raise RuntimeError(f'Completed checkpoint modified: {spec["id"]}')
        return saved
    seed_all(spec['seed'])
    model = make_model(spec['model'], cfg, a1).to(device)
    optimizer_class = torch.optim.AdamW if recipe['optimizer'] == 'adamw' else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=spec['lr'], weight_decay=recipe['weight_decay'])
    generator = torch.Generator(device='cpu').manual_seed(spec['seed'])
    x, y, vx, vy = data
    batch = cfg['batch_size']
    steps_per_epoch = math.ceil(len(x)/batch)
    if len(x) % batch == 1:
        raise RuntimeError('Singleton batch incompatible with head BatchNorm')
    history, best, best_epoch, stale, elapsed = [], float('inf'), 0, 0, 0.
    resume = folder/'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['identity'] != expected:
            raise RuntimeError(f'Resume identity changed: {spec["id"]}')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        history, best, best_epoch, stale, elapsed = [saved[k] for k in ['history', 'best', 'best_epoch', 'stale', 'elapsed']]
        restore_rng(saved['rng'], generator)
    else:
        model.eval()
        with torch.no_grad():
            model(x[:batch])
        sync(device)
    for epoch in range(len(history)+1, epochs+1):
        if recipe['early_stop'] and stale >= cfg['patience']:
            break
        model.train()
        order = torch.randperm(len(x), generator=generator).to(device)
        total, correct = 0., 0
        sync(device)
        begin = time.perf_counter()
        for step, j in enumerate(range(0, len(x), batch)):
            idx = order[j:j+batch]
            factor = lr_factor((epoch-1)*steps_per_epoch+step, recipe, steps_per_epoch, epochs)
            for group in optimizer.param_groups:
                group['lr'] = spec['lr']*factor
            inputs = augment(x[idx], generator, cfg['augmentation']['pad']) if recipe['augment'] else x[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, y[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss: {spec["id"]}, epoch {epoch}')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(idx)
            correct += (logits.argmax(1) == y[idx]).sum().item()
        sync(device)
        train_seconds = time.perf_counter()-begin
        vp, vl = predict(model, vx, vy, cfg['evaluation_batch_size'])
        sync(device)
        epoch_seconds = time.perf_counter()-begin
        elapsed += epoch_seconds
        history.append({'epoch': epoch, 'learning_rate_end': spec['lr']*factor,
                        'train_loss': total/len(x), 'train_accuracy': correct/len(x),
                        'validation_loss': vl, 'validation_accuracy': float((vp == vy.cpu().numpy()).mean()),
                        'train_seconds': train_seconds, 'train_validation_seconds': epoch_seconds})
        if vl < best-cfg['min_delta']:
            best, best_epoch, stale = vl, epoch, 0
            save_torch(folder/'best.pt', model_state(model))
        else:
            stale += 1
        write(folder/'history.json', history)
        save_torch(resume, {'identity': expected, 'model': model_state(model), 'optimizer': optimizer.state_dict(),
                            'history': history, 'best': best, 'best_epoch': best_epoch, 'stale': stale,
                            'elapsed': elapsed, 'rng': rng_state(generator)})
        write(out/'progress.json', {'stage': 'training', 'fit': spec['id'], 'epoch': epoch, 'max_epochs': epochs,
                                    'validation': history[-1], 'updated_utc': stamp()})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(json.dumps({'fit': spec['id'], **history[-1]}), flush=True)
    save_torch(folder/'final.pt', model_state(model))
    result = {'identity': expected, 'fit_id': spec['id'], 'model': spec['model'], 'recipe': spec['recipe'],
              'seed': spec['seed'], 'learning_rate': spec['lr'], 'epochs_run': len(history), 'best_epoch': best_epoch,
              'fit_seconds': elapsed, 'training_seconds': sum(h['train_seconds'] for h in history),
              'parameters': sum(p.numel() for p in model.parameters()),
              'peak_allocated_mb': torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
              'completed_utc': stamp(), 'hardware': hardware(device)}
    for kind in ('best', 'final'):
        model.load_state_dict(torch.load(folder/f'{kind}.pt', map_location='cpu', weights_only=True))
        vp, vl = predict(model, vx, vy, cfg['evaluation_batch_size'])
        result.update({f'{kind}_validation_loss': vl, f'{kind}_validation_accuracy': float((vp == vy.cpu().numpy()).mean()),
                       f'{kind}_sha256': digest(folder/f'{kind}.pt')})
    write(done, result)
    return result


def train_all(a1_root, cfg, out, signature, device, a1, seeds, epochs_override):
    data = get_data(a1_root, cfg, device)
    fits = [fit(spec, data, cfg, out, signature, device, a1, epochs_override or cfg['recipes'][spec['recipe']]['epochs'])
            for spec in plan(cfg, seeds)]
    entries = []
    for f in fits:
        primary = cfg['recipes'][f['recipe']]['primary_checkpoint']
        for kind in ('best', 'final'):
            entries.append({'id': f'{f["fit_id"]}_{kind}', 'fit_id': f['fit_id'], 'model': f['model'], 'recipe': f['recipe'],
                            'seed': f['seed'], 'checkpoint_kind': kind, 'primary': kind == primary,
                            'checkpoint': f'fits/{f["fit_id"]}/{kind}.pt', 'checkpoint_sha256': f[f'{kind}_sha256']})
    frozen = {'source': signature, 'evaluation': entries, 'test_data_loaded': False,
              'test_reuse_disclosure': cfg['test_disclosure']}
    path = out/'frozen_experiment.json'
    if path.exists() and read(path) != frozen:
        raise RuntimeError('Frozen checkpoints or evaluation identities changed')
    write(path, frozen)
    del data
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return frozen


def evaluate_all(a1_root, cfg, out, signature, device, a1, frozen):
    if frozen['source'] != signature:
        raise RuntimeError('Experiment not frozen under current source')
    for entry in frozen['evaluation']:
        if digest(out/entry['checkpoint']) != entry['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed before test')
    test = a1_root/'data/prepared/test.npz'
    if digest(test) != cfg['test_sha256']:
        raise RuntimeError('Test archive hash mismatch')
    write(out/'test_evaluation_started.json', {'utc': stamp(), 'freeze_sha256': digest(out/'frozen_experiment.json')})
    with np.load(test) as z:
        raw, y = z['x_test'], z['y_test']
    images = tensor_images(raw, device)
    target = torch.tensor(y, dtype=torch.long, device=device)
    rows = []
    for entry in frozen['evaluation']:
        model = make_model(entry['model'], cfg, a1).to(device)
        model.load_state_dict(torch.load(out/entry['checkpoint'], map_location='cpu', weights_only=True))
        pred, loss = predict(model, images, target, cfg['evaluation_batch_size'])
        folder = out/'evaluation'/entry['id']
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder/'predictions.npy', pred)
        fit_result = read(out/'fits'/entry['fit_id']/'fit.json')
        row = {**entry, 'accuracy': accuracy_score(y, pred), 'macro_f1': f1_score(y, pred), 'test_loss': loss,
               'parameters': fit_result['parameters'], 'best_epoch': fit_result['best_epoch'],
               'epochs_run': fit_result['epochs_run'], 'fit_seconds': fit_result['fit_seconds']}
        write(folder/'metrics.json', row)
        rows.append(row)
        del model
    write(out/'evaluation_rows.json', rows)
    write(out/'test_complete.json', {'utc': stamp(), 'source': signature,
                                     'freeze_sha256': digest(out/'frozen_experiment.json'), 'evaluation_rows': len(rows)})


def reference_predictions(model, seed):
    """Existing early-stopped checkpoints evaluated in Assignment 2 job 618435 (a1_protocol)."""
    if model == 'cnn':
        return A2/'outputs/baseline'/f'cnn_original_{seed}'/'predictions.npy'
    return A2/'outputs/evaluation'/f'{model}_original_s{seed}'/'predictions.npy'


def analyze(a1_root, cfg, out, reports, seeds):
    reports.mkdir(parents=True, exist_ok=True)
    with np.load(a1_root/'data/prepared/test.npz') as z:
        y, groups = z['y_test'], z['groups']
    rows = read(out/'evaluation_rows.json')
    pred_paths = {}
    for r in rows:
        pred_paths[(r['model'], r['recipe'], r['checkpoint_kind'], r['seed'])] = out/'evaluation'/r['id']/'predictions.npy'
    models = ['cnn']+cfg['lsnet_variants']
    for m in models:
        for s in seeds:
            pred_paths[(m, 'a1_protocol', 'best', s)] = reference_predictions(m, s)

    def correct(m, recipe, kind):
        return np.stack([np.load(pred_paths[(m, recipe, kind, s)]) == y for s in seeds]).astype(float)

    def primary_kind(recipe):
        return 'best' if recipe == 'a1_protocol' else cfg['recipes'][recipe]['primary_checkpoint']

    summary = []
    cells = [(m, 'a1_protocol', 'best') for m in models]
    cells += [(m, r, k) for m in models for r in (cfg['cnn_recipes'] if m == 'cnn' else cfg['lsnet_recipes'])
              for k in ('best', 'final')]
    for m, r, k in cells:
        c = correct(m, r, k)
        acc = c.mean(1)*100
        preds = [np.load(pred_paths[(m, r, k, s)]) for s in seeds]
        meta = [x for x in rows if x['model'] == m and x['recipe'] == r and x['checkpoint_kind'] == k]
        summary.append({'model': m, 'recipe': r, 'checkpoint': k, 'primary': k == primary_kind(r), 'n_seeds': len(seeds),
                        'accuracy_mean': float(acc.mean()), 'accuracy_sd': float(acc.std(ddof=1)) if len(seeds) > 1 else 0.,
                        'macro_f1_mean': float(np.mean([f1_score(y, p) for p in preds])),
                        'seed_accuracies': [round(float(a), 4) for a in acc],
                        'best_epoch_mean': float(np.mean([x['best_epoch'] for x in meta])) if meta else None,
                        'fit_minutes_mean': float(np.mean([x['fit_seconds'] for x in meta])/60) if meta else None})

    def contrast(label, a, b, sign_pairs=None):
        """a and b are (model, recipe); difference a-b, or (a1-b1)-(a2-b2) when sign_pairs is given."""
        if sign_pairs is None:
            d = correct(*a, primary_kind(a[1]))-correct(*b, primary_kind(b[1]))
        else:
            (a1, b1), (a2, b2) = sign_pairs
            d = ((correct(*a1, primary_kind(a1[1]))-correct(*b1, primary_kind(b1[1])))
                 - (correct(*a2, primary_kind(a2[1]))-correct(*b2, primary_kind(b2[1]))))
        interval = paired_interval(d, groups, cfg['bootstrap']['resamples'], cfg['bootstrap']['seed'])
        per_seed = d.mean(1)*100
        return {'comparison': label, **interval,
                'seed_differences_pp': [round(float(v), 4) for v in per_seed],
                'seed_difference_sd_pp': float(per_seed.std(ddof=1)) if len(seeds) > 1 else 0.}

    contrasts = []
    for r in cfg['cnn_recipes']:
        contrasts.append(contrast(f'cnn {r} - cnn a1_protocol', ('cnn', r), ('cnn', 'a1_protocol')))
    for v in cfg['lsnet_variants']:
        contrasts.append(contrast(f'{v} r2_aug - {v} a1_protocol', (v, 'r2_aug'), (v, 'a1_protocol')))
    for ctrl in ('no_lkp', 'small_lkp', 'cnn'):
        contrasts.append(contrast(f'full - {ctrl} (r2_aug)', ('full', 'r2_aug'), (ctrl, 'r2_aug')))
    for ctrl in ('no_lkp', 'small_lkp'):
        contrasts.append(contrast(f'interaction: (full - {ctrl}) r2_aug minus a1_protocol', None, None,
                                  [(('full', 'r2_aug'), (ctrl, 'r2_aug')), (('full', 'a1_protocol'), (ctrl, 'a1_protocol'))]))

    per_class = []
    for m in models:
        for r in ('a1_protocol', 'r2_aug'):
            preds = [np.load(pred_paths[(m, r, primary_kind(r), s)]) for s in seeds]
            for c, name in enumerate(CLASSES):
                mask = y == c
                per_class.append({'model': m, 'recipe': r, 'class': name,
                                  'recall_mean': float(np.mean([(p[mask] == c).mean() for p in preds])*100)})

    for name, table in (('recipe_summary.csv', summary), ('recipe_contrasts.csv', contrasts), ('recipe_per_class.csv', per_class)):
        with (reports/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    write(reports/'recipe_analysis.json', {'summary': summary, 'contrasts': contrasts, 'per_class': per_class,
                                           'seeds': seeds, 'bootstrap': cfg['bootstrap'], 'generated_utc': stamp()})
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for m in models:
            h = read(out/'fits'/f'{m}_r2_aug_s{seeds[0]}'/'history.json')
            axes[0].plot([e['epoch'] for e in h], [e['validation_accuracy']*100 for e in h], label=m)
            axes[1].plot([e['epoch'] for e in h], [e['validation_loss'] for e in h], label=m)
        axes[0].set(xlabel='epoch', ylabel='validation accuracy %', title=f'r2_aug, seed {seeds[0]}')
        axes[1].set(xlabel='epoch', ylabel='validation loss', yscale='log')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(reports/'recipe_learning_curves.png', dpi=160)
    except Exception as error:  # the figure is optional; numbers are already written
        print(f'Figure skipped: {error}', flush=True)
    print(json.dumps({'summary': [{k: s[k] for k in ('model', 'recipe', 'checkpoint', 'accuracy_mean', 'accuracy_sd')}
                                  for s in summary if s['primary']],
                      'contrasts': [{k: c[k] for k in ('comparison', 'difference_pp', 'conditional_group_ci95_pp')}
                                    for c in contrasts]}, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a1-root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='2 epochs, seed 42 only, separate output folders')
    args = p.parse_args()
    cfg = read(ROOT/'protocol.json')
    seeds = [42] if args.smoke else cfg['seeds']
    epochs_override = 2 if args.smoke else None
    if args.smoke:
        cfg['bootstrap'] = {**cfg['bootstrap'], 'resamples': 50}
    out = ROOT/('outputs_smoke' if args.smoke else 'outputs')
    reports = ROOT/('reports_smoke' if args.smoke else 'reports')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(int(os.getenv('OMP_NUM_THREADS', '4')))
    if not torch.cuda.is_available():
        raise RuntimeError('Formal pipeline requires allocated CUDA GPU')
    a1_root = args.a1_root.resolve()
    for m in ['cnn']+cfg['lsnet_variants']:
        for s in seeds:
            if not reference_predictions(m, s).is_file():
                raise RuntimeError(f'Missing Assignment 2 reference predictions: {m} seed {s}')
    upstream = read(A2/'upstream.json')
    for name, sha in upstream['files'].items():
        if digest(A2/'vendor/lsnet'/name) != sha:
            raise RuntimeError(f'Upstream source mismatch: {name}')
    device = torch.device('cuda')
    signature = fingerprints()
    signature['mode'] = 'smoke' if args.smoke else 'formal'
    preflight = out/'preflight.json'
    if preflight.exists() and read(preflight)['source'] != signature:
        raise RuntimeError('Source changed after preflight')
    write(preflight, {'source': signature, 'hardware': hardware(device), 'ska_checks': official_check(device)})
    a1 = load_a1()
    frozen = train_all(a1_root, cfg, out, signature, device, a1, seeds, epochs_override)
    if (out/'test_complete.json').exists():
        complete = read(out/'test_complete.json')
        if complete['source'] != signature or complete['freeze_sha256'] != digest(out/'frozen_experiment.json'):
            raise RuntimeError('Completed evaluation no longer matches frozen study')
    else:
        evaluate_all(a1_root, cfg, out, signature, device, a1, frozen)
    analyze(a1_root, cfg, out, reports, seeds)
    write(out/'progress.json', {'stage': 'complete', 'updated_utc': stamp(),
                                'fits': len(list((out/'fits').glob('*/fit.json')))})
    print('Recipe check completed in one scheduler job.', flush=True)


if __name__ == '__main__':
    main()
