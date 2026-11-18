import numpy as np
from learner_core import Parameter, Module, initialize

class Scale(Module):

    def __init__(self, value=2.0):
        self.w = Parameter(np.array(value))

    def forward(self, x):
        self._x=x.copy()
        return x*self.w.data
        """TODO：按网页公式实现此方法。"""
        raise NotImplementedError('请实现 forward')

    def backward(self, g):
        self.w.grad+=np.sum(g*self._x)
        return g*self.w.data
        """TODO：按网页公式实现此方法。"""
        raise NotImplementedError('请实现 backward')

if __name__ == "__main__":
    from _foundation_checks import run_local
    run_local("layer_scale", globals())
