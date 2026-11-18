import numpy as np
from foundation_core import Parameter, Module, initialize

class Parameter:

    def __init__(self, data):
        """TODO：按网页公式实现此方法。"""
        self.data=np.array(data,dtype=np.float64,copy=True)
        self.grad=np.zeros_like(data)
        # raise NotImplementedError('请实现 __init__')

    def zero_grad(self):
        """TODO：按网页公式实现此方法。"""
        self.grad[...]=0
        # raise NotImplementedError('请实现 zero_grad')

class Module:

    # module(X) 自动调用这个方法，相当于 module.forward(X)。
    # 定义中的 *args 将位置参数收集成元组。
    def __call__(self, *args):
        # 调用中的 *args 拆开元组，传给 forward，再返回结果。
        return self.forward(*args)

    def parameters(self):
        """TODO：按网页公式实现此方法。"""
        found,seen=[],set()
        
        def visit(value):
            if isinstance(value,Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    found.append(value)
            elif isinstance(value,Module):
                for name,child in vars(value).items():
                    if not name.startswith('_'):
                        visit(child)
            elif isinstance(value,(list,tuple,dict)):
                for child in value.values() if isinstance(value,dict) else value:
                    visit(child)
        visit(self)
        return found
        raise NotImplementedError('请实现 parameters')

    def zero_grad(self):
        """TODO：按网页公式实现此方法。"""
        for value in self.parameters():
            value.zero_grad()
        # raise NotImplementedError('请实现 zero_grad')

if __name__ == "__main__":
    from _foundation_checks import run_local
    run_local("infrastructure", globals())
