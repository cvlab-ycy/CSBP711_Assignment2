"""Recompute every number in the A2 report from the files in this repository. CPU only, needs numpy.

Usage (from the repository root):  python verify_repo.py
Each line shows the value as printed in the report and where it comes from. Exit code 0 means every check passed.
Published values from other papers cannot be recomputed. They are checked against the literature table instead.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT/'experiments'
A1, A2, R, F = EXP/'audited_v2', EXP/'assignment2_lsnet', EXP/'assignment2_recipe', EXP/'lsnet_feasibility'
SEEDS = [42, 43, 44]
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
failures = []


def check(label, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + label + (f'  {detail}' if detail else ''))
    if not ok:
        failures.append(label)


def same(label, value, printed, digits):
    """Does `value` round to the number printed in the report?"""
    check(f'{label}: {printed}', f'{value:.{digits}f}' == printed, f'(recomputed {value:.4f})')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def bootstrap(diff, groups, resamples=2000, seed=711):
    """Same method as the studies: average seeds, then resample test near-duplicate groups."""
    values = diff.mean(0)
    _, idx = np.unique(groups, return_inverse=True)
    counts, sums = np.bincount(idx), np.bincount(idx, weights=values)
    rng = np.random.default_rng(seed)
    draws = [sums[s].sum()/counts[s].sum() for s in (rng.integers(0, len(counts), len(counts)) for _ in range(resamples))]
    return values.mean()*100, np.quantile(draws, [.025, .975])*100


def ci_text(diff):
    point, (lo, hi) = bootstrap(diff, groups)
    fmt = lambda v: f'{0.0 if abs(v) < .005 else v:.2f}'
    return f'{fmt(point)} [{fmt(lo)}, {fmt(hi)}]'


# ------------------------------------------------------------------ 0. repository and data
need = [A1/'data/prepared/test.npz', A1/'reports/comparison.csv', A2/'reports/primary_comparison.csv',
        R/'outputs/evaluation_rows.json', R/'reports/recipe_summary.csv', R/'reports/recipe_contrasts.csv']
missing = [str(p.relative_to(ROOT)) for p in need if not p.is_file()]
check('repository: all three study folders present', not missing, f'missing {missing}' if missing else '')
if missing:
    sys.exit('Pull the latest commits from all members first.')

with np.load(A1/'data/prepared/test.npz') as z:
    y, groups = z['y_test'], z['groups']
with np.load(A1/'data/prepared/train_val.npz') as z:
    n_train, n_val, xmax = len(z['y_train']), len(z['y_val']), int(z['x_train'].max())
check('split: 53,979 train, 5,998 validation, 10,000 test', (n_train, n_val, len(y)) == (53979, 5998, 10000))
check('pixels stored as 0..255 and scaled by 1/255 in code', xmax == 255)
excl = json.loads((A1/'data/prepared/exclusions.json').read_text())
removed = sum(len(v) for v in excl.values())
a1proto = json.loads((A1/'config/protocol.json').read_text())
check(f'audit removed 23 near-duplicates ({removed}) and held out 10% for validation',
      removed == 23 and 60000 - removed == n_train + n_val and a1proto['validation_fraction'] == 0.1)

# ------------------------------------------------------------------ predictions
A1P = lambda m, c, s: A1/f'outputs/runs/{m}_{c}_{s}/test_predictions.npy'
A2P = lambda m, c, s: A2/f'outputs/evaluation/{m}_{c}_s{s}/predictions.npy'
evaluation = {(e['model'], e['recipe'], e['checkpoint_kind'], e['seed']): e['id']
              for e in json.loads((R/'outputs/evaluation_rows.json').read_text())}


def correct(model, recipe, kind='primary', condition='original'):
    """Seeds x test images, 1 where the prediction is right."""
    if recipe == 'a1':  # Assignment 1 protocol: A1 models from audited_v2, LSNet from the first A2 study
        paths = [(A1P if model in ('logistic', 'mlp', 'cnn') else A2P)(model, condition, s) for s in SEEDS]
    else:
        kind = {'a1_aug': 'best', 'r2_noaug': 'final', 'r2_aug': 'final'}[recipe] if kind == 'primary' else kind
        paths = [R/f'outputs/evaluation/{evaluation[(model, recipe, kind, s)]}/predictions.npy' for s in SEEDS]
    preds = np.stack([np.load(p) for p in paths])
    return (preds == y).astype(float), preds


def acc(*a, **k):
    c, _ = correct(*a, **k)
    per_seed = c.mean(1)*100
    return per_seed.mean(), per_seed.std(ddof=1)


def acc_text(*a, **k):
    m, s = acc(*a, **k)
    return f'{m:.2f} ± {s:.2f}'


def recall(model, recipe, cls):
    c, _ = correct(model, recipe)
    return c[:, y == CLASSES.index(cls)].mean()*100


# ------------------------------------------------------------------ 1. Table 1
print('\n# Table 1')
for label, args, printed in [
        ('A1 logistic regression', ('logistic', 'a1'), '84.36 ± 0.06'),
        ('A1 MLP', ('mlp', 'a1'), '88.25 ± 0.17'),
        ('A1 CNN, A1 protocol', ('cnn', 'a1'), '91.28 ± 0.34'),
        ('A1 CNN, literature-style recipe', ('cnn', 'r2_aug'), '90.71 ± 0.23'),
        ('A1 CNN, literature-style recipe without augmentation', ('cnn', 'r2_noaug'), '91.72 ± 0.33'),
        ('LSNet full, A1 protocol', ('full', 'a1'), '91.94 ± 0.21'),
        ('LSNet without LKP, A1 protocol', ('no_lkp', 'a1'), '91.58 ± 0.16'),
        ('LSNet 3x3 LKP, A1 protocol', ('small_lkp', 'a1'), '91.28 ± 0.16'),
        ('LSNet full, literature-style recipe', ('full', 'r2_aug'), '94.71 ± 0.14'),
        ('LSNet without LKP, literature-style recipe', ('no_lkp', 'r2_aug'), '94.73 ± 0.07'),
        ('LSNet 3x3 LKP, literature-style recipe', ('small_lkp', 'r2_aug'), '94.72 ± 0.17')]:
    got = acc_text(*args)
    check(f'{label}: {printed}', got == printed, '' if got == printed else f'(recomputed {got})')

params = {r['model']: float(r['parameters_mean']) for r in rows(A2/'reports/primary_comparison.csv') if r['condition'] == 'original'}
cnn_params = {r['model']: float(r['parameters_mean']) for r in rows(A2/'reports/baseline_comparison.csv')}['cnn']
same('LSNet-T parameters (M)', params['full']/1e6, '11.0', 1)
same('A1 CNN parameters (M)', cnn_params/1e6, '0.11', 2)
lit = (R/'LITERATURE_COMPARISON.md').read_text()
check('published values match LITERATURE_COMPARISON.md: 84.2, 87.1, 95.61 ± 0.08, 95.99 ± 0.10',
      all(v in lit for v in ['| 84.2 |', '| 87.1 |', '95.61 ± 0.08', '95.99 ± 0.10']))

# ------------------------------------------------------------------ 2. Critical analysis
print('\n# Section 2')
gaps = (correct('full', 'a1')[0].mean(1) - correct('no_lkp', 'a1')[0].mean(1))*100
check('per-seed gap full minus no LKP ranges 0.05 to 0.61', f'{gaps.min():.2f} to {gaps.max():.2f}' == '0.05 to 0.61',
      f'(recomputed {np.round(gaps, 2)})')

# ------------------------------------------------------------------ 3. Reality check
print('\n# Section 3: setup')
n_fits_a2 = len([p for p in (A2/'outputs/fits').iterdir() if p.is_dir()])
n_fits_r = len([p for p in (R/'outputs/fits').iterdir() if p.is_dir()])
check(f'42 training runs ({n_fits_a2} in job 618435 + {n_fits_r} in job 620078)', n_fits_a2 + n_fits_r == 42)
for name, d in [('A2 first study', A2), ('recipe check', R)]:
    frozen, started = d/'outputs/frozen_experiment.json', json.loads((d/'outputs/test_evaluation_started.json').read_text())
    ok = json.loads(frozen.read_text())['test_data_loaded'] is False and started['freeze_sha256'] == sha256(frozen)
    check(f'{name}: checkpoints frozen and hashed before test data were loaded', ok)
a1_frozen = json.loads((A1/'outputs/frozen_experiment.json').read_text())['created_utc']
check('A1 study: frozen before test evaluation completed',
      a1_frozen < json.loads((A1/'outputs/test_complete.json').read_text())['completed_utc'])
proto, rproto = json.loads((A2/'protocol.json').read_text()), json.loads((R/'protocol.json').read_text())
check('A1 protocol: Adam, at most 40 epochs, early stopping on validation loss',
      proto['optimizer'] == 'Adam' and proto['max_epochs'] == 40 and 'Validation loss' in proto['checkpoint_selection'])
r2 = rproto['recipes']['r2_aug']
check('literature recipe: pad 4 crop + flip, AdamW 0.025, 5 warm-up, cosine, 100 epochs, final checkpoint',
      rproto['augmentation']['pad'] == 4 and rproto['augmentation']['horizontal_flip_probability'] == 0.5
      and (r2['optimizer'], r2['weight_decay'], r2['warmup_epochs'], r2['schedule'], r2['epochs'], r2['primary_checkpoint'])
      == ('adamw', 0.025, 5, 'warmup_cosine', 100, 'final'))
check('learning rates CNN 0.001 and LSNet 0.003, LSNet grid was {0.001, 0.003}',
      rproto['learning_rates'] == {'cnn': 0.001, 'lsnet': 0.003} and proto['learning_rates'] == [0.001, 0.003])
ska = json.loads((F/'results/gpu-614431/result.json').read_text())['ska_checks']
check('official SKA kernel matches the reference in forward and backward (GPU job 614431)',
      all(c['official_forward_backward'] == 'pass' for c in ska))
same('largest SKA output difference (x1e-6)', max(c['max_output_error'] for c in ska)*1e6, '2', 0)
check('A1 protocol uses no augmentation, the recipe uses crop + flip',
      proto['augmentation'] is False and rproto['recipes']['r2_aug']['augment'] is True)

print('\n# Section 3: explaining the gaps')
full_a1, full_r2 = acc('full', 'a1')[0], acc('full', 'r2_aug')[0]
same('logistic regression (rounded in text)', acc('logistic', 'a1')[0], '84.4', 1)
same('MLP (rounded in text)', acc('mlp', 'a1')[0], '88.3', 1)
same("LSNet's gap to WRN-28-10 under the A1 protocol", 95.99 - full_a1, '4.1', 1)
same('gap closed by the recipe', full_r2 - full_a1, '2.8', 1)
same('gap left', 95.99 - full_r2, '1.3', 1)
cnn_a1 = correct('cnn', 'a1')[0]
check('CNN, crop and flip under the A1 schedule: -2.45 [-2.87, -2.04]', ci_text(correct('cnn', 'a1_aug')[0] - cnn_a1) == '-2.45 [-2.87, -2.04]')
check('CNN, cosine schedule alone: 0.44 [0.15, 0.73]', ci_text(correct('cnn', 'r2_noaug')[0] - cnn_a1) == '0.44 [0.15, 0.73]')

print("\n# Section 3: the paper's method on our data")
C = {(m, r): correct(m, r)[0] for m in ('full', 'no_lkp', 'small_lkp') for r in ('a1', 'r2_aug')}
for label, diff, printed in [
        ('A1 protocol, full minus no LKP', C['full', 'a1'] - C['no_lkp', 'a1'], '0.36 [0.08, 0.65]'),
        ('A1 protocol, full minus 3x3', C['full', 'a1'] - C['small_lkp', 'a1'], '0.65 [0.35, 0.96]'),
        ('recipe, full minus no LKP', C['full', 'r2_aug'] - C['no_lkp', 'r2_aug'], '-0.02 [-0.24, 0.20]'),
        ('recipe, full minus 3x3', C['full', 'r2_aug'] - C['small_lkp', 'r2_aug'], '0.00 [-0.22, 0.22]'),
        ('shrinking of the no-LKP gap', (C['full', 'a1'] - C['no_lkp', 'a1']) - (C['full', 'r2_aug'] - C['no_lkp', 'r2_aug']), '0.38 [0.02, 0.74]'),
        ('shrinking of the 3x3 gap', (C['full', 'a1'] - C['small_lkp', 'a1']) - (C['full', 'r2_aug'] - C['small_lkp', 'r2_aug']), '0.66 [0.29, 1.03]')]:
    got = ci_text(diff)
    check(f'{label}: {printed}', got == printed, '' if got == printed else f'(recomputed {got})')
gains = [acc(m, 'r2_aug')[0] - acc(m, 'a1')[0] for m in ('full', 'no_lkp', 'small_lkp')]
check('recipe adds 2.8 to 3.4 points to every LSNet version', f'{min(gains):.1f} to {max(gains):.1f}' == '2.8 to 3.4')

print('\n# Section 3: why the gain disappears')
for label, args, printed in [
        ('full LSNet Shirt recall, A1 protocol', ('full', 'a1', 'Shirt'), '73.9'),
        ('full LSNet Shirt recall, recipe', ('full', 'r2_aug', 'Shirt'), '83.8'),
        ('full LSNet Coat recall, A1 protocol', ('full', 'a1', 'Coat'), '87.6'),
        ('full LSNet Coat recall, recipe', ('full', 'r2_aug', 'Coat'), '93.9'),
        ('T-shirt recall, full, A1 protocol', ('full', 'a1', 'T-shirt/top'), '87.2'),
        ('T-shirt recall, 3x3, A1 protocol', ('small_lkp', 'a1', 'T-shirt/top'), '82.9'),
        ('T-shirt recall, full, recipe', ('full', 'r2_aug', 'T-shirt/top'), '90.2'),
        ('T-shirt recall, 3x3, recipe', ('small_lkp', 'r2_aug', 'T-shirt/top'), '90.3')]:
    same(label, recall(*args), printed, 1)
lowest, rising = [], []
for m in ('full', 'no_lkp', 'small_lkp'):
    for s in SEEDS:
        h = json.loads((R/f'outputs/fits/{m}_r2_aug_s{s}/history.json').read_text())
        i = int(np.argmin([e['validation_loss'] for e in h]))
        lowest.append(h[i]['epoch'])
        rising.append(h[-1]['validation_accuracy'] > h[i]['validation_accuracy'])
check('LSNet validation loss lowest at epochs 25 to 39, validation accuracy still higher at the end',
      (min(lowest), max(lowest)) == (25, 39) and all(rising), f'(epochs {sorted(lowest)})')
same('full LSNet at the loss-selected checkpoint', acc('full', 'r2_aug', kind='best')[0], '93.42', 2)

print('\n# Section 3: locality')
perm = {m: acc(m, 'a1', condition='permuted')[0] for m in ('full', 'cnn', 'mlp')}
same('full LSNet after pixel shuffling', perm['full'], '87.78', 2)
same('full LSNet cost', full_a1 - perm['full'], '4.2', 1)
same('CNN cost', acc('cnn', 'a1')[0] - perm['cnn'], '3.9', 1)
same('MLP after pixel shuffling', perm['mlp'], '88.02', 2)
same('MLP cost', acc('mlp', 'a1')[0] - perm['mlp'], '0.2', 1)

print()
print('ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED')
sys.exit(1 if failures else 0)
