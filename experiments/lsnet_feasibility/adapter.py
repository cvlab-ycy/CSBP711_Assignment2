"""Explicit small-image adaptation of the pinned official LSNet implementation.

No upstream files are modified. Reference SKA is for correctness/CPU checks;
official Triton remains the intended CUDA backend and must pass parity checks.
"""
import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / 'vendor' / 'lsnet'


def reference_ska(x, weights):
    n, c, h, w = x.shape
    wc, k2 = weights.shape[1:3]
    k = math.isqrt(k2)
    assert k * k == k2 and k % 2 == 1
    assert weights.shape == (n, wc, k2, h, w)
    patches = F.unfold(x, k, padding=k // 2).reshape(n, c, k2, h, w)
    # Official CUDA kernel uses ci % wc, not contiguous groups.
    selected = weights[:, torch.arange(c, device=x.device) % wc]
    return (patches * selected).sum(2)


class ReferenceSKA(nn.Module):
    def forward(self, x, weights):
        return reference_ska(x, weights)


def load_official_ska():
    name = 'lsnet_official_ska'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, UPSTREAM / 'model' / 'ska.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_official_model():
    # Import the unmodified network while avoiding a CUDA-only import on CPU.
    # Each LSConv is assigned the requested backend explicitly in build_model.
    name = 'feasibility_upstream'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(UPSTREAM / 'model')]
        sys.modules[name] = package
        ska = types.ModuleType(name + '.ska')
        ska.SKA = ReferenceSKA
        sys.modules[name + '.ska'] = ska
        spec = importlib.util.spec_from_file_location(name + '.lsnet', UPSTREAM / 'model' / 'lsnet.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[name + '.lsnet']


def build_model(variant='full', backend='reference', scale='tiny_adapted', native=False):
    if variant not in ('full', 'no_lkp', 'small_lkp'):
        raise ValueError(variant)
    source = load_official_model()
    # The primary candidate retains all official LSNet-T widths/depths.
    # A compact secondary configuration is explicitly labelled as an approximation.
    if scale == 'tiny_adapted':
        dims, depths, heads = [64, 128, 256, 384], [0, 2, 8, 10], [3, 3, 3, 4]
    elif scale == 'compact':
        dims, depths, heads = [32, 64, 128, 192], [2, 2, 2, 2], [2, 2, 2, 3]
    else:
        raise ValueError(scale)
    # Stem outputs 28x28 after adaptation; subsequent stage sizes are 14, 7, 4.
    # Native stem outputs 4x4, followed by 2, 1, 1. Resolution metadata is
    # calculated from actual convolution sizes instead of floor(28/8).
    model = source.LSNet(img_size=4 if native else 28, patch_size=1,
                        in_chans=1, num_classes=10, embed_dim=dims,
                        key_dim=[16]*4, depth=depths, num_heads=heads,
                        distillation=False)
    if not native:
        for layer in model.patch_embed:
            if isinstance(layer, source.Conv2d_BN):
                layer.c.stride = (1, 1)
    for layer in model.modules():
        if isinstance(layer, source.LSConv):
            layer.ska = load_official_ska().SKA() if backend == 'official' else ReferenceSKA()
            if variant == 'no_lkp':
                # Remove only the depthwise convolution. Retain BN so that the
                # comparison does not also remove a normalization operation.
                layer.lkp.cv2.c = nn.Identity()
            elif variant == 'small_lkp':
                old = layer.lkp.cv2.c
                layer.lkp.cv2.c = nn.Conv2d(old.in_channels, old.out_channels, 3,
                                          padding=1, groups=old.groups, bias=False)
    return model


def shape_trace(model, x):
    rows, handles = [], []
    for name in ('patch_embed', 'blocks1', 'blocks2', 'blocks3', 'blocks4', 'head'):
        def capture(module, args, output, key=name):
            rows.append({'name': key, 'shape': list(output.shape)})
        handles.append(getattr(model, name).register_forward_hook(capture))
    previous = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous)
    return rows
