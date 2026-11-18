import numpy as np
from learner_core import Parameter, Module, initialize

class GELU(Module):

    def forward(self, x):
        """TODO：按网页公式实现此方法。"""
        self._x=x.copy()
        self._a,self._k=0.044715,np.sqrt(2/np.pi)
        self._t=np.tanh(self._k*(x+self._a*x**3))
        return 1/2*x*(1+self._t)
        raise NotImplementedError('请实现 forward')

    def backward(self, g):
        """TODO：按网页公式实现此方法。"""
        x,a,k,t=self._x,self._a,self._k,self._t
        du=k*(1+3*a*x**2)
        return g*(0.5*(1+t)+0.5*x*(1-t**2)*du)
        raise NotImplementedError('请实现 backward')

if __name__ == "__main__":
    from _foundation_checks import run_local
    run_local("layer_gelu", globals())
