import numpy as np
from learner_core import Parameter, Module, initialize

# 网页测评已依次加载个人前置模块；本地运行从原文件导入。
if __name__ != "student":
    from layer_scale import Scale
    from layer_linear import Linear
    from layer_embedding import Embedding
    from layer_gelu import GELU
    from layer_rmsnorm import RMSNorm
    from layer_mlp import MLP

class MLPBranch(Module):
    def __init__(self, C, hidden, eps=1e-5,*,rng=None,init_std=None,init_scale=1.0):
        self.norm=RMSNorm(C,eps)
        self.mlp=MLP(C,hidden,rng=rng,init_std=init_std,init_scale=init_scale)
        # raise NotImplementedError("请创建两个已有子层")

    def forward(self, x):
        return self.mlp.forward(self.norm.forward(x))
        raise NotImplementedError("请串联子层返回值")

    def backward(self, g):
        return self.norm.backward(self.mlp.backward(g))
        raise NotImplementedError("请逆序传递梯度")


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _foundation_checks import run_local
    run_local("foundation_capstone", globals())
