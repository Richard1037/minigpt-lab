import numpy as np
from learner_core import Parameter, Module, initialize

def clip_grad(parameters, limit):
    norm=float(np.sqrt(sum(np.sum(p.grad**2) for p in parameters)))
    scale=min(1.0,limit/(norm+1e-12))
    for p in parameters: p.grad*=scale
    return np.array(norm)
    #raise NotImplementedError('请完成 clip_grad')

class AdamW:

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-08, decay=0.01):
        self.params=parameters
        self.lr,self.beta1,self.beta2,self.eps,self.decay=lr,beta1,beta2,eps,decay
        self.m=[np.zeros_like(p.data) for p in self.params]
        self.v=[np.zeros_like(p.data) for p in self.params]
        self.t=0
        return None
        raise NotImplementedError('请完成 __init__')

    def step(self):
        self.t+=1
        for p,m,v in zip(self.params,self.m,self.v):
            m*=self.beta1
            m+=(1-self.beta1)*p.grad
            v*=self.beta2
            v+=(1-self.beta2)*p.grad**2
            mh=m/(1-self.beta1**self.t)
            vh=v/(1-self.beta2**self.t)
            if p.data.ndim>=2: p.data*=(1-self.lr*self.decay)
            p.data-=self.lr*mh/(np.sqrt(vh)+self.eps)
        return None
        raise NotImplementedError('请完成 step')

# 直接运行执行本节公开测评；作为模块导入时跳过。
if __name__ == "__main__":
    from _transformer_checks import run_local
    run_local("t_optim", globals())
