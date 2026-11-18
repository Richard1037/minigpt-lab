"""第四部分参考实现。只供对照，不会自动写入个人练习。W 统一为 (C,D)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exercises'))
import numpy as np
from foundation_core import Parameter, Module, initialize


class Scale(Module):
    def __init__(self, value=2.):
        self.w = Parameter(np.array(value))

    def forward(self, x):
        self._x = x.copy()
        return x * self.w.data

    def backward(self, g):
        self.w.grad += np.sum(g * self._x)
        return g * self.w.data


class Linear(Module):
    def __init__(self, C, D, use_bias=True):
        self.W = Parameter(initialize((C, D)))
        self.b = Parameter(np.zeros(D)) if use_bias else None

    def forward(self, x):
        self._shape = x.shape
        self._x2 = x.reshape(-1, self.W.data.shape[0]).copy()
        y2 = self._x2 @ self.W.data
        if self.b is not None:
            y2 = y2 + self.b.data
        return y2.reshape(*x.shape[:-1], self.W.data.shape[1])

    def backward(self, g):
        g2 = g.reshape(-1, self.W.data.shape[1])
        self.W.grad += self._x2.T @ g2
        if self.b is not None:
            self.b.grad += g2.sum(axis=0)
        return (g2 @ self.W.data.T).reshape(self._shape)


class Embedding(Module):
    def __init__(self, V, C):
        self.weight = Parameter(np.random.default_rng(0).normal(0, .02, (V,C)))

    def forward(self, ids):
        self._ids = np.asarray(ids).copy()
        return self.weight.data[self._ids]

    def backward(self, g):
        np.add.at(self.weight.grad, self._ids.reshape(-1), g.reshape(-1, g.shape[-1]))
        return None


class GELU(Module):
    def forward(self, x):
        self._x = x.copy()
        self._t = np.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x**3))
        return .5 * x * (1 + self._t)

    def backward(self, g):
        x, t = self._x, self._t
        du = np.sqrt(2 / np.pi) * (1 + 3 * .044715 * x**2)
        return g * (.5 * (1 + t) + .5 * x * (1 - t**2) * du)


class RMSNorm(Module):
    def __init__(self, C, eps=1e-5):
        self.w = Parameter(np.ones(C))
        self.eps = eps

    def forward(self, x):
        self._r = 1 / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + self.eps)
        self._xh = x * self._r
        return self._xh * self.w.data

    def backward(self, g):
        u = g * self.w.data
        self.w.grad += (g * self._xh).reshape(-1, g.shape[-1]).sum(axis=0)
        return self._r * (u - self._xh * np.mean(u * self._xh, axis=-1, keepdims=True))


class MLP(Module):
    def __init__(self, C, hidden):
        self.fc1 = Linear(C, hidden)
        self.act = GELU()
        self.fc2 = Linear(hidden, C)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def backward(self, g):
        return self.fc1.backward(self.act.backward(self.fc2.backward(g)))
