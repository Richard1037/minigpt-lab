import numpy as np
from learner_core import Parameter, Module, initialize
if __name__ != "student":
    from t_softmax import Softmax

class AttentionCore(Module):

    def __init__(self, causal=True):
        self.causal=causal
        self.softmax=Softmax()
        # raise NotImplementedError('请完成 __init__')

    def forward(self, q, k, v):
        self._q,self._k,self._v=q.copy(),k.copy(),v.copy()
        self._d=d=q.shape[-1]
        s=q@k.swapaxes(-1,-2)/np.sqrt(d)
        if self.causal:
            if s.shape[-2]!=s.shape[-1]: raise ValueError('wrong format with Q & K')
            self._mask=np.triu(np.ones(s.shape[-2:],dtype=bool),1)
            s=np.where(self._mask,-np.inf,s)
        self._a=a=self.softmax(s)
        return a@v
        raise NotImplementedError('请完成 forward')

    def backward(self, g):
        d=self._d
        a,q,k,v=self._a,self._q,self._k,self._v
        dv=a.swapaxes(-1,-2)@g
        da=g@v.swapaxes(-1,-2)
        ds=self.softmax.backward(da)
        if self.causal: ds=np.where(self._mask,0.,ds)
        dq=ds@k/np.sqrt(d)
        dk=ds.swapaxes(-1,-2)@q/np.sqrt(d)
        return dq,dk,dv
        raise NotImplementedError('请完成 backward')


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_attention", globals())
