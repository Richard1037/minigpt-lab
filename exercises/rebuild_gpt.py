import numpy as np
from learner_core import Parameter, Module, initialize
if __name__ != "student":
    from t_loss import CrossEntropy
    from layer_embedding import Embedding
    from layer_rmsnorm import RMSNorm
    from layer_linear import Linear
    from layer_gelu import GELU
    from layer_mlp import MLP
    from t_softmax import Softmax
    from t_attention import AttentionCore
    from t_mha import MultiHeadAttention
    from t_optim import clip_grad, AdamW
    from t_data import tokenize, make_batch
    from t_train import train
    from t_sample import sample
    from t_generate import generate
    from t_checkpoint import save_checkpoint, load_checkpoint
    from t_project import project_step
    from t_block import Block

class GPT(Module):

    def __init__(self, V, C, H, hidden, layers, context, seed=0):
        raise NotImplementedError('TODO：独立实现 __init__')

    def forward(self, ids):
        raise NotImplementedError('TODO：独立实现 forward')

    def backward(self, g):
        raise NotImplementedError('TODO：独立实现 backward')

if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("rebuild_gpt", globals())
