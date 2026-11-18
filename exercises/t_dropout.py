import numpy as np
from learner_core import Parameter, Module, initialize

class Dropout(Module):

    def __init__(self, p=0.2, seed=0):
        raise NotImplementedError('请完成 __init__')

    def forward(self, x, training=True):
        raise NotImplementedError('请完成 forward')

    def backward(self, g):
        raise NotImplementedError('请完成 backward')


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_dropout", globals())
