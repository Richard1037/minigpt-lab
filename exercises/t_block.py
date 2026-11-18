import numpy as np
from learner_core import Parameter, Module, initialize
if __name__ != "student":
    from layer_rmsnorm import RMSNorm
    from t_mha import MultiHeadAttention
    from foundation_capstone import MLPBranch

class Block(Module):

    def __init__(self, C, H, hidden, rng=None,init_std=None,init_scale=1.0):
        self.norm1 = RMSNorm(C)
        self.attn = MultiHeadAttention(C, H, rng=rng, init_std=init_std, init_scale=init_scale)
        self.ffn = MLPBranch(C, hidden, rng=rng, init_std=init_std, init_scale=init_scale)

    def forward(self, x):
        x1 = x + self.attn(self.norm1(x))
        return x1 + self.ffn(x1)

    def backward(self, g):
        g1 = g + self.ffn.backward(g)
        return g1 + self.norm1.backward(self.attn.backward(g1))


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_block", globals())
