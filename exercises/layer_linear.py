import numpy as np
from learner_core import Parameter, Module, initialize

class Linear(Module):

    def __init__(self, C, D, use_bias=True,*,rng=None,init_std=None,init_scale=1.0):
        if init_std is None:
            values=initialize((C,D),rng)
        else:
            rng=np.random.default_rng(0) if rng is None else rng
            values=rng.normal(0,init_std,(C,D))
            
        self.W = Parameter(values*init_scale)
        self.b = Parameter(np.zeros(D)) if use_bias else None

    def forward(self, x):
        """TODO：按网页公式实现此方法。"""
        self._shape=x.shape
        C,D=x.shape[-1],self.W.data.shape[-1]
        self._x=x.reshape(-1,C).copy()
        y=self._x@self.W.data
        if self.b is not None:
            y=y+self.b.data
        return y.reshape(*x.shape[:-1],D)
        raise NotImplementedError('请实现 forward')

    def backward(self, g):
        """TODO：按网页公式实现此方法。"""
        g2=g.reshape(-1,self.W.data.shape[1])
        self.W.grad+=self._x.T@g2
        if self.b is not None:
            self.b.grad+=g2.sum(axis=0)
        return (g2@self.W.data.T).reshape(self._shape)
        raise NotImplementedError('请实现 backward')

if __name__ == "__main__":
    from _foundation_checks import run_local
    run_local("layer_linear", globals())
