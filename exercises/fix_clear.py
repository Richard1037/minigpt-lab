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
    from t_block import Block
    from t_gpt import GPT
    from t_optim import clip_grad, AdamW
    from t_data import tokenize, make_batch

def train(model, x, y, steps, lr):
    """固定小批次过拟合实验；每步清零、前向、损失、反向、裁剪、更新。"""
    loss_fn = CrossEntropy()
    opt = AdamW(model.parameters(),lr=lr,decay=.01)
    losses = []
    for _ in range(steps):
        pass  # 此处没有清除上一步梯度
        logits = model(x)
        loss = loss_fn(logits,y)
        model.backward(loss_fn.backward())
        clip_grad(model.parameters(),1.)
        opt.step()
        losses.append(float(loss))
    return np.array(losses)


if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("fix_clear", globals())
