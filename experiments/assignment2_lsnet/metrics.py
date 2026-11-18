"""Small, dependency-light classification and paired uncertainty calculations."""
import numpy as np

def accuracy_score(y,p):
    return float(np.mean(np.asarray(y)==np.asarray(p)))

def confusion(y,p):
    return np.bincount(np.asarray(y)*10+np.asarray(p),minlength=100).reshape(10,10)

def f1_score(y,p,average='macro',zero_division=0):
    assert average=='macro' and zero_division==0
    c=confusion(y,p);d=c.sum(0)+c.sum(1)
    return float(np.divide(2*c.diagonal(),d,out=np.zeros(10,dtype=float),where=d!=0).mean())

def paired_interval(differences,groups,resamples=2000,seed=711):
    """Average seeds first; resample duplicate groups with paired observations."""
    values=np.asarray(differences,dtype=float)
    if values.ndim==2:values=values.mean(0)
    _,idx=np.unique(groups,return_inverse=True)
    counts=np.bincount(idx);sums=np.bincount(idx,weights=values)
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(resamples):
        sample=rng.integers(0,len(counts),len(counts))
        draws.append(sums[sample].sum()/counts[sample].sum())
    return {'difference_pp':float(values.mean()*100),
            'conditional_group_ci95_pp':(np.quantile(draws,[.025,.975])*100).tolist()}
