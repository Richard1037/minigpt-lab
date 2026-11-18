"""数组网络的公共接口；不依赖 PyTorch。

前向缓存只服务紧接着的一次反向；更新参数必须在反向之后。
层的 backward 返回输入梯度，把参数梯度累加到 Parameter.grad。
"""
import numpy as np


class Parameter:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        self.grad[...] = 0


class Module:
    def __call__(self, *args):
        return self.forward(*args)

    def parameters(self):
        found, seen = [], set()

        def visit(value):
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    found.append(value)
            elif isinstance(value, Module):
                for name, child in vars(value).items():
                    if not name.startswith('_'):
                        visit(child)
            elif isinstance(value, (list, tuple, dict)):
                for child in (value.values() if isinstance(value, dict) else value):
                    visit(child)

        visit(self)
        return found

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()


def initialize(shape, rng=None):
    """Linear 的 shape=(C,D)，fan-in=C。"""
    rng = rng if rng is not None else np.random.default_rng(0)
    return rng.standard_normal(shape) * np.sqrt(2.0 / shape[0])


def scale_residual_outputs(parameters, layers):
    """只传入残差出口权重；原地缩放一次，不能每次 forward 调用。"""
    for parameter in parameters:
        parameter.data *= 1 / np.sqrt(2 * layers)
