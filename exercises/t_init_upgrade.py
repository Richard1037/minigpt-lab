import numpy as np
from learner_core import Parameter, Module, initialize
if __name__ != "student":
    from layer_linear import Linear
    from layer_embedding import Embedding
    from layer_mlp import MLP
    from foundation_capstone import MLPBranch
    from t_mha import MultiHeadAttention
    from t_block import Block
    from layer_rmsnorm import RMSNorm
    from layer_gelu import GELU
    from t_attention import AttentionCore

if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_init_upgrade", globals())
