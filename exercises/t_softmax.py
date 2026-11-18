import numpy as np
from learner_core import Module

class Softmax(Module):
    def forward(self, x):
        z=np.exp(x-np.max(x,axis=-1,keepdims=True))
        self._p=z/np.sum(z,axis=-1,keepdims=True)
        return self._p.copy()
        raise NotImplementedError("请实现前向与缓存")

    def backward(self, g):
        return self._p*(g-np.sum(g*self._p,axis=-1,keepdims=True))
        raise NotImplementedError("请实现反向")


# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_softmax", globals())
