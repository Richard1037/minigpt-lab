import numpy as np
from learner_core import Parameter, Module, initialize

class LayerNorm(Module):

    def __init__(self, C, eps=1e-05):
        raise NotImplementedError('请完成 __init__')

    def forward(self, x):
        raise NotImplementedError('请完成 forward')

    def backward(self, g):
        raise NotImplementedError('请完成 backward')


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_layernorm", globals())
