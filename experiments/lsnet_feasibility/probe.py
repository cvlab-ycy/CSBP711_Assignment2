"""Validation-only feasibility checks, never an assignment test-set evaluation."""
import argparse
import hashlib
import json
import os
import platform
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from adapter import build_model, load_official_ska, reference_ska, shape_trace

HERE = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def check_ska(device):
    results = []
    # Rectangular and odd-sized features catch border and channel-group errors.
    for c, wc, h, w in [(8, 1, 7, 5), (16, 2, 14, 14), (24, 3, 4, 4)]:
        x = torch.randn(2, c, h, w, device=device, requires_grad=True)
        weights = torch.randn(2, wc, 9, h, w, device=device, requires_grad=True)
        # Independent shift/slice implementation; does not use unfold.
        padded = nn.functional.pad(x, (1, 1, 1, 1))
        slow = torch.zeros_like(x)
        for kh in range(3):
            for kw in range(3):
                slow = slow + padded[:, :, kh:kh+h, kw:kw+w] * weights[:, torch.arange(c, device=device) % wc, kh*3+kw]
        ref = reference_ska(x, weights)
        torch.testing.assert_close(ref, slow, rtol=1e-5, atol=1e-5)
        upstream = torch.randn_like(ref)
        grads = torch.autograd.grad(ref, (x, weights), upstream, retain_graph=True)
        slow_grads = torch.autograd.grad(slow, (x, weights), upstream)
        for a, b in zip(grads, slow_grads):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-5)
        row = {'shape': [2, c, h, w], 'weight_channels': wc, 'reference_forward_backward': 'pass'}
        if device.type == 'cuda':
            actual = load_official_ska().SKA()(x, weights)
            official_grads = torch.autograd.grad(actual, (x, weights), upstream)
            torch.testing.assert_close(actual, ref, rtol=1e-4, atol=2e-5)
            for a, b in zip(official_grads, grads):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=5e-5)
            row.update(official_forward_backward='pass', max_output_error=float((actual-ref).abs().max()))
        results.append(row)
    return results


def balanced_indices(labels, per_class, seed):
    rng = np.random.default_rng(seed)
    ids = np.concatenate([rng.choice(np.flatnonzero(labels == k), per_class, replace=False) for k in range(10)])
    rng.shuffle(ids)
    return ids


@torch.no_grad()
def evaluate(model, x, y, batch):
    model.eval()
    loss, correct = 0., 0
    for i in range(0, len(y), batch):
        logits = model(x[i:i+batch])
        loss += nn.functional.cross_entropy(logits, y[i:i+batch], reduction='sum').item()
        correct += (logits.argmax(1) == y[i:i+batch]).sum().item()
    return {'loss': loss / len(y), 'accuracy': correct / len(y)}


def run_variant(variant, args, data, device):
    seed_all(args.seed)
    model = build_model(variant, args.backend, args.scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    x, y, vx, vy = data
    row = {'variant': variant, 'parameters': sum(p.numel() for p in model.parameters()),
           'shapes': shape_trace(model, x[:2]), 'initial_train': evaluate(model, x, y, args.batch), 'epochs': []}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    # Warm-up includes actual backward kernels before timing steady-state epochs.
    initial_state = {k: v.detach().clone() for k,v in model.state_dict().items()}
    warm_start = time.perf_counter()
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x[:args.batch]), y[:args.batch]).backward()
        optimizer.step()
    sync(device)
    row['warmup_seconds'] = time.perf_counter() - warm_start
    model.load_state_dict(initial_state)
    del initial_state
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    seed_all(args.seed)
    all_gradients_finite = True
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(y), device='cpu').to(device)
        total, correct = 0., 0
        sync(device)
        started = time.perf_counter()
        for i in range(0, len(y), args.batch):
            ids = order[i:i+args.batch]
            if len(ids) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[ids])
            loss = nn.functional.cross_entropy(logits, y[ids])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward()
            # Validate once per epoch; avoid claiming speed unaffected by checking.
            if i == 0:
                all_gradients_finite &= all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                if not all_gradients_finite:
                    raise RuntimeError('Non-finite gradients')
            optimizer.step()
            total += loss.item() * len(ids)
            correct += (logits.argmax(1) == y[ids]).sum().item()
        sync(device)
        seconds = time.perf_counter() - started
        validation = evaluate(model, vx, vy, args.batch)
        record = {'epoch': epoch+1, 'train_loss': total/len(y), 'train_accuracy': correct/len(y),
                  'validation': validation, 'train_seconds': seconds}
        row['epochs'].append(record)
        print(json.dumps({'variant': variant, **record}), flush=True)
    row['final_train'] = evaluate(model, x, y, args.batch)
    row['finite_gradients'] = all_gradients_finite
    row['loss_reduced'] = row['final_train']['loss'] < row['initial_train']['loss']
    if device.type == 'cuda':
        row['peak_allocated_mb'] = torch.cuda.max_memory_allocated(device) / 2**20
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(vx[:args.batch])
        sync(device)
        started = time.perf_counter()
        for _ in range(10):
            model(vx[:args.batch])
        sync(device)
    row['inference_ms_per_batch'] = (time.perf_counter()-started)*100
    row['inference_batch_size'] = min(args.batch, len(vy))
    durations = [e['train_seconds'] for e in row['epochs']]
    row['rough_53979_image_epoch_seconds'] = float(np.median(durations)*53979/len(y))
    row['timing_note'] = 'FP32, unfused, no compilation; warm-up excluded. Estimate scales subset time linearly, excludes validation and is NOT a measured full epoch.'
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--backend', choices=['reference','official'], default='reference')
    p.add_argument('--scale', choices=['tiny_adapted','compact'], default='tiny_adapted')
    p.add_argument('--variants', nargs='+', default=['full','no_lkp','small_lkp'])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--train-per-class', type=int, default=128)
    p.add_argument('--val-per-class', type=int, default=50)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--checks-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output/'result.json'
    if result_path.exists():
        raise RuntimeError('Use a new output directory to preserve earlier evidence')
    report = {'status':'running', 'kind':'validation-only feasibility, not final results',
              'args': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'environment': {'torch':torch.__version__, 'numpy':np.__version__, 'python':platform.python_version(),
                              'hostname':platform.node(), 'slurm_job':os.getenv('SLURM_JOB_ID')},
              'variants':[]}
    save(result_path,report)
    try:
        torch.set_num_threads(int(os.getenv('OMP_NUM_THREADS','4')))
        device = torch.device(args.device)
        if device.type == 'cuda':
            report['environment']['gpu'] = torch.cuda.get_device_name(device)
            report['environment']['cuda'] = torch.version.cuda
        seed_all(args.seed)
        report['ska_checks'] = check_ska(device)
        for native in (True,False):
            m = build_model('full', args.backend, args.scale, native=native).to(device)
            report['native_shapes' if native else 'adapted_shapes'] = shape_trace(m, torch.zeros(2,1,28,28,device=device))
            del m
        save(result_path,report)
        if not args.checks_only:
            if args.data.name != 'train_val.npz':
                raise ValueError('Only the prepared train/validation archive is accepted')
            report['data_sha256'] = hashlib.sha256(args.data.read_bytes()).hexdigest()
            if report['data_sha256'] != '71cba7be7a7ae5ffcc4c8cd78f5b00ba1801d736d0640959a10d99cd93958aa1':
                raise ValueError('Prepared archive differs from audited Assignment 1 data')
            with np.load(args.data) as data:
                ti = balanced_indices(data['y_train'],args.train_per_class,args.seed)
                vi = balanced_indices(data['y_val'],args.val_per_class,args.seed)
                np.savez(args.output/'subset_indices.npz',train=ti,validation=vi)
                x=torch.tensor(data['x_train'][ti],device=device,dtype=torch.float32).reshape(-1,1,28,28)/255.
                y=torch.tensor(data['y_train'][ti],device=device,dtype=torch.long)
                vx=torch.tensor(data['x_val'][vi],device=device,dtype=torch.float32).reshape(-1,1,28,28)/255.
                vy=torch.tensor(data['y_val'][vi],device=device,dtype=torch.long)
            for variant in args.variants:
                report['variants'].append(run_variant(variant,args,(x,y,vx,vy),device))
                save(result_path,report)
        report['status']='passed' if all(v['loss_reduced'] for v in report['variants']) else 'learning_check_failed'
    except Exception:
        report['status']='failed'
        report['error']=traceback.format_exc()
        raise
    finally:
        save(result_path,report)


if __name__ == '__main__':
    main()
