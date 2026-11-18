import numpy as np
from learner_core import Parameter, Module, initialize

class Embedding(Module):

    def __init__(self, V, C,*,rng=None,init_std=0.02):
        rng=np.random.default_rng(0) if rng is None else rng
        self.weight = Parameter(rng.normal(0, init_std, (V, C)))

    def forward(self, ids):
        """TODO：按网页公式实现此方法。"""
        self._ids=np.asarray(ids).copy()
        return self.weight.data[ids]
        raise NotImplementedError('请实现 forward')

    def backward(self, g):
        """TODO：按网页公式实现此方法。"""
        np.add.at(self.weight.grad,self._ids.reshape(-1),g.reshape(-1,g.shape[-1]))
        return None
        raise NotImplementedError('请实现 backward')

if __name__ == "__main__":
    from _foundation_checks import run_local
    run_local("layer_embedding", globals())
