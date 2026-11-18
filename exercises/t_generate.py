import numpy as np
from learner_core import Parameter, Module, initialize
if __name__ != "student":
    from layer_embedding import Embedding
    from layer_rmsnorm import RMSNorm
    from layer_linear import Linear
    from layer_gelu import GELU
    from layer_mlp import MLP
    from t_softmax import Softmax
    from t_attention import AttentionCore
    from t_mha import MultiHeadAttention
    from t_block import Block
    from t_gpt import GPT
    from t_sample import sample

def generate(model, ids, steps, temperature=1.0, top_k=0, seed=0):
    result = ids.copy()
    rng = np.random.default_rng(seed)
    for i in range(steps):
        logits = model(result[:,-model.context:])[:,-1,:]
        token = sample(logits,temperature,top_k,rng)
        result = np.concatenate([result,token[:,None]],axis=1)
    return result

if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_generate", globals())
