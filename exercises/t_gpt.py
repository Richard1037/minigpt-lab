import numpy as np
from learner_core import Parameter, Module, initialize
if __name__ != "student":
    from layer_embedding import Embedding
    from layer_rmsnorm import RMSNorm
    from t_mha import MultiHeadAttention
    from foundation_capstone import MLPBranch
    from t_block import Block

class GPT(Module):

    def __init__(self, V, C, H, hidden, layers, context, seed=0):
        if min(V,C,H,hidden,layers,context)<=0 or C%H: raise ValueError('Fuck You')
        self.context=context
        rng=np.random.default_rng(seed)
        scale=1/np.sqrt(2*layers)
        self.token=Embedding(V,C,rng=rng)
        self.position=Embedding(context,C,rng=rng)
        self.blocks=[Block(C,H,hidden,rng=rng,init_std=0.02,init_scale=scale) for i in range(layers)]
        self.norm=RMSNorm(C)
        # raise NotImplementedError('请完成 __init__')

    def forward(self, ids):
        if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('ids 须为二维整数数组')
        if not 0 < ids.shape[1] <= self.context:
            raise ValueError('序列长度须在 context 内')
        x=self.token(ids)+self.position(np.arange(ids.shape[1]))[None,:,:]
        for block in self.blocks:
            x=block(x)
        z=self._z=self.norm(x)
        return z@self.token.weight.data.T
        raise NotImplementedError('请完成 forward')

    def backward(self, g):
        z=self._z
        C=z.shape[-1]
        self.token.weight.grad+=g.reshape(-1,g.shape[-1]).T @ z.reshape(-1,C)
        dx=self.norm.backward(g @ self.token.weight.data)
        for block in reversed(self.blocks):
            dx=block.backward(dx)
        self.position.backward(dx.sum(axis=0))
        self.token.backward(dx)
        # raise NotImplementedError('请完成 backward')


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_gpt", globals())
