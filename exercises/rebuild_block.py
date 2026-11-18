import numpy as np
from learner_core import Parameter, Module, initialize
if __name__ != "student":
    from layer_rmsnorm import RMSNorm
    from layer_linear import Linear
    from layer_gelu import GELU
    from layer_mlp import MLP
    from t_softmax import Softmax
    from t_attention import AttentionCore
    from t_mha import MultiHeadAttention

class Block(Module):

    def __init__(self, C, H, hidden):
        raise NotImplementedError('TODO：独立实现 __init__')

    def forward(self, x):
        raise NotImplementedError('TODO：独立实现 forward')

    def backward(self, g):
        raise NotImplementedError('TODO：独立实现 backward')

if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("rebuild_block", globals())
