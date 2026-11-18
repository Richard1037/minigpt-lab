import numpy as np
from learner_core import Parameter, Module, initialize
if __name__ != "student":
    from initialize_weights import initialize
    from t_loss import CrossEntropy
    from layer_embedding import Embedding
    from layer_rmsnorm import RMSNorm
    from layer_linear import Linear
    from layer_gelu import GELU
    from layer_mlp import MLP
    from t_softmax import Softmax
    from t_attention import AttentionCore
    from t_mha import MultiHeadAttention
    from layer_scale import Scale
    from foundation_capstone import MLPBranch
    from t_block import Block
    from t_init_upgrade import Linear, Embedding, MLP, MLPBranch, MultiHeadAttention, Block
    from t_gpt import GPT
    from t_optim import clip_grad, AdamW
    from t_data import tokenize, make_batch

def train(model, x, y, steps, lr):
    loss_fn=CrossEntropy()
    opt=AdamW(model.parameters(),lr=lr,decay=0.01)
    losses=[]
    for i in range(steps):
        model.zero_grad()
        logits=model(x)
        loss=loss_fn(logits,y)
        losses.append(loss)
        model.backward(loss_fn.backward())
        clip_grad(model.parameters(),1.0)
        opt.step()
    return np.array(losses)
    raise NotImplementedError('请完成 train')

# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_train", globals())
