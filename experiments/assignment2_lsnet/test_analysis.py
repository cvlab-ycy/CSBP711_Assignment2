"""Numerical tests with known labels, not study results."""
import numpy as np
from metrics import accuracy_score,f1_score,paired_interval

def main():
    y=np.arange(10);p=y.copy();p[0]=1
    assert accuracy_score(y,p)==.9
    assert abs(f1_score(y,p)-(8+2/3)/10)<1e-12
    assert f1_score(y,y)==1.
    result=paired_interval(np.ones((3,10))*.1,np.array([0,0,1,1,2,3,4,5,6,7]),100)
    np.testing.assert_allclose(result['conditional_group_ci95_pp'],[10,10])
    zero=paired_interval(np.zeros((3,10)),np.arange(10),100)
    assert zero['difference_pp']==0 and zero['conditional_group_ci95_pp']==[0,0]
    # A group of size two must retain both observations when selected.
    d=np.array([1.,1.,-1.]);r=paired_interval(d,np.array([0,0,1]),1000)
    assert abs(r['difference_pp']-100/3)<1e-12
    print('Accuracy, macro-F1 and paired group resampling: PASS')

if __name__=='__main__':main()
