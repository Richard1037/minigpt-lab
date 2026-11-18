import numpy as np
from learner_core import Parameter, Module, initialize

if __name__ != "student":
    from layer_linear import Linear
    from layer_gelu import GELU

class MLP(Module):

    def __init__(self, C, hidden,*,rng=None,init_std=None,init_scale=1.0):
        self.fc1 = Linear(C, hidden,rng=rng,init_std=init_std)
        self.act = GELU()
        self.fc2 = Linear(hidden, C,rng=rng,init_std=init_std,init_scale=init_scale)

    def forward(self, x):
        """TODO：按网页公式实现此方法。"""
        return self.fc2(self.act(self.fc1(x)))
        raise NotImplementedError('请实现 forward')

    def backward(self, g):
        """TODO：按网页公式实现此方法。"""
        return self.fc1.backward(self.act.backward(self.fc2.backward(g)))
        raise NotImplementedError('请实现 backward')

if __name__ == "__main__":
    from _foundation_checks import run_local
    run_local("layer_mlp", globals())
