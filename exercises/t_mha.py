import numpy as np
from learner_core import Parameter, Module, initialize
if __name__ != "student":
    from layer_linear import Linear
    from t_attention import AttentionCore

class MultiHeadAttention(Module):

    def __init__(self, C, H,*,rng=None,init_std=None,init_scale=1.0):
        if C%H!=0: raise ValueError('Fuck You')
        self.C,self.H=C,H
        self.qkv=Linear(C,C*3,rng=rng,init_std=init_std)
        self.core=AttentionCore(True)
        self.proj=Linear(C,C,rng=rng,init_std=init_std,init_scale=init_scale)
        return None
        raise NotImplementedError('请完成 __init__')

    def forward(self, x):
        B,T,C=x.shape
        self._shape=x.shape
        # Linear projection
        q,k,v=np.split(self.qkv(x),3,axis=-1) 
        # Split heads
        trans= lambda a:a.reshape(B,T,self.H,C//self.H).transpose(0,2,1,3) 
        # Scaled dot-product attention
        y=self.core(trans(q),trans(k),trans(v)) 
        # Merge heads & Concatenate heads 
        y=y.transpose(0,2,1,3).reshape(B,T,C)
        # Output projection
        return self.proj(y) 
        raise NotImplementedError('请完成 forward') #

    def backward(self, g):
        B,T,C=self._shape
        dy=self.proj.backward(g).reshape(B,T,self.H,C//self.H).transpose(0,2,1,3)
        dq,dk,dv=self.core.backward(dy)
        joined=[a.transpose(0,2,1,3).reshape(B,T,C) for a in (dq,dk,dv)]
        return self.qkv.backward(np.concatenate(joined,axis=-1))
        raise NotImplementedError('请完成 backward')


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_mha", globals())
