import numpy as np
from learner_core import Module

class CrossEntropy(Module):
    def forward(self, logits, targets):
        z=logits-logits.max(axis=-1,keepdims=True)
        logp=z-np.log(np.exp(z).sum(axis=-1,keepdims=True))
        self._p=np.exp(logp)
        self._targets=targets.copy()
        return np.array(-np.take_along_axis(logp,targets[...,None],axis=-1).mean())
        #raise NotImplementedError("请实现前向与缓存")

    def backward(self, g=1.0):
        dx=self._p.copy().reshape(-1,self._p.shape[-1])
        dx[np.arange(dx.shape[0]),self._targets.reshape(-1)]-=1
        return (dx/dx.shape[0]*g).reshape(self._p.shape)
        #raise NotImplementedError("请实现反向")


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_loss", globals())
