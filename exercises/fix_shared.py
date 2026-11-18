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

class GPT(Module):
    def __init__(self, V, C, H, hidden, layers, context, seed=0):
        self.context = context
        self.token, self.position = Embedding(V,C), Embedding(context,C)
        self.blocks = [Block(C,H,hidden) for _ in range(layers)]
        self.norm = RMSNorm(C)
        rng = np.random.default_rng(seed)
        # 初始化每个独立参数一次，不能让同形状矩阵都使用同一个随机序列。
        for p in self.parameters():
            if p.data.ndim == 2: p.data[...] = rng.normal(0,.02,p.data.shape)
        for block in self.blocks:
            block.attn.proj.W.data /= np.sqrt(2*layers)
            block.mlp.fc2.W.data /= np.sqrt(2*layers)

    def forward(self, ids):
        if ids.shape[1] > self.context: raise ValueError('序列超过 context')
        x = self.token(ids) + self.position(np.arange(ids.shape[1]))[None,:,:]
        for block in self.blocks: x = block(x)
        self._h = self.norm(x)
        return self._h @ self.token.weight.data.T

    def backward(self, g):
        C = self._h.shape[-1]
        pass  # 此处漏掉一条共享参数梯度路径
        dx = self.norm.backward(g @ self.token.weight.data)
        for block in reversed(self.blocks): dx = block.backward(dx)
        self.position.backward(dx.sum(axis=0))
        self.token.backward(dx)
        return None


if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("fix_shared", globals())
