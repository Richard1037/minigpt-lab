import numpy as np
from learner_core import Parameter, Module, initialize

class RMSNorm(Module):

    def __init__(self, C, eps=1e-05):
        self.w = Parameter(np.ones(C))
        self.eps = eps

    def forward(self, x):
        """TODO：按网页公式实现此方法。"""
        self._r=1/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+self.eps)
        self._xh=x*self._r
        return self._xh*self.w.data
        raise NotImplementedError('请实现 forward')

    def backward(self, g):
        """TODO：按网页公式实现此方法。"""
        u=g*self.w.data
        xh=self._xh
        self.w.grad+=(g*xh).reshape(-1,g.shape[-1]).sum(axis=0)
        return self._r*(u-xh*np.mean(u*xh,axis=-1,keepdims=True))
        raise NotImplementedError('请实现 backward')

if __name__ == "__main__":
    from _foundation_checks import run_local
    run_local("layer_rmsnorm", globals())
