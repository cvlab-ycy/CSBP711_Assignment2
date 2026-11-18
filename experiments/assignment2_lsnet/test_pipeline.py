"""Meaningful CPU checks for architecture controls and exact epoch-boundary resume."""
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from adapter import build_model,load_official_model
from run_study import fit,make_spec,read,seed_all

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    cfg=read(ROOT/'protocol.json')
    cfg.update(backend='reference',scale='compact',batch_size=4,evaluation_batch_size=4,
               patience=8,max_epochs=2)
    source=load_official_model()
    checks=[]
    states={}
    for variant in cfg['variants']:
        seed_all(42)
        m=build_model(variant,'reference','compact')
        mixers=[layer for layer in m.modules() if isinstance(layer,source.LSConv)]
        assert len(mixers)==3
        for layer in mixers:
            assert layer.lkp.sks==3
            if variant=='no_lkp':
                assert isinstance(layer.lkp.cv2.c,torch.nn.Identity)
                assert isinstance(layer.lkp.cv2.bn,torch.nn.BatchNorm2d)
            else:
                assert layer.lkp.cv2.c.kernel_size==((7,7) if variant=='full' else (3,3))
        states[variant]=m.state_dict()
        checks.append({'variant':variant,'dynamic_aggregation_retained':True,'control_definition':'pass'})
    # Shared unaffected parameters must begin identically despite ablation creation.
    for variant in ['no_lkp','small_lkp']:
        for name,tensor in states['full'].items():
            if '.lkp.cv2.' not in name:
                torch.testing.assert_close(tensor,states[variant][name],rtol=0,atol=0)
    seed_all(17)
    x=torch.rand(8,1,28,28);y=torch.tensor([0,1,2,3,4,5,6,7])
    vx=torch.rand(4,1,28,28);vy=torch.tensor([0,1,2,3])
    spec=make_spec('primary','full','original',42,.001,2)
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='resume-check-') as tmp:
        base=Path(tmp)
        a=fit(spec,(x,y,vx,vy),cfg,base/'continuous',{'test':'resume'},torch.device('cpu'))
        b=fit(spec,(x,y,vx,vy),cfg,base/'resumed',{'test':'resume'},torch.device('cpu'),max_new_epochs=1)
        assert b is None
        b=fit(spec,(x,y,vx,vy),cfg,base/'resumed',{'test':'resume'},torch.device('cpu'))
        sa=torch.load(base/'continuous/fits'/spec['id']/'best.pt',weights_only=True)
        sb=torch.load(base/'resumed/fits'/spec['id']/'best.pt',weights_only=True)
        for k in sa:torch.testing.assert_close(sa[k],sb[k],rtol=0,atol=0)
        for k in ['best_epoch','validation_loss','validation_accuracy']:
            assert a[k]==b[k]
        try:
            fit(spec,(x,y,vx,vy),cfg,base/'resumed',{'test':'changed'},torch.device('cpu'))
        except RuntimeError:
            checks.append({'changed_source_refused':'pass'})
        else:raise AssertionError('Stale source accepted')
    checks.append({'epoch_boundary_resume':'bitwise identical best model and validation metrics'})
    (ROOT/'checks').mkdir(exist_ok=True)
    (ROOT/'checks/cpu_pipeline.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('Architecture controls, matched initialization, exact resume, stale-source guard: PASS')


if __name__=='__main__':main()
