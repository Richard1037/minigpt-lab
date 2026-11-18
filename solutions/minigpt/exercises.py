"""src/minigpt/exercises.py 的答案版（ MGPT_ANSWER=1 时启用）。

判据文件（stages/ 和 tests/）对两份实现完全一样，所以这份存在的意义只有一个：
你在自己写完、或者卡了很久之后，能拿到一个"确实通过了所有判据"的参照。
建议用法：先自己写完跑判据，通过后再来这里比对差异 —— 差异处往往就是
你有但没被测出来的坏习惯（或者我有而判据没覆盖的取巧）。
"""

from __future__ import annotations

import numpy as np

from .scalar import Value
from .toolkit import print_progress


def fit_by_autograd(x, y, iters=3000, lr=0.2, log_every=500):
    """用 Value 计算图拟合 y = a·x + b。"""
    a = Value(0.0, label="a")
    b = Value(0.0, label="b")
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    curve = []
    for step in range(iters):
        # 每一步都重新建一张新图（旧图用完就扔，这是 scalar 引擎的简单代价）
        for v in (a, b):
            v.grad = 0.0
        loss = Value(0.0, label="loss")
        for xi, yi in zip(xs, ys):
            e = a * xi + b - yi
            loss = loss + e * e
        loss = loss / len(xs)
        loss.backward()
        curve.append(loss.data)
        a.data -= lr * a.grad
        b.data -= lr * b.grad
        if log_every:
            print_progress(step + 1, iters, every=log_every, prefix="自动微分")
    return a.data, b.data, curve


class ScalarMLP:
    def __init__(self, n_in=1, n_hidden=6, n_out=1, activate=True, seed=0):
        rng = np.random.default_rng(seed)
        self.n_hidden, self.activate = n_hidden, activate
        scale1 = 1.0 / n_in ** 0.5
        scale2 = 1.0 / n_hidden ** 0.5
        self.W1 = [[Value(float(rng.normal(0, scale1)), label=f"W1[{j}][{k}]")
                    for k in range(n_in)] for j in range(n_hidden)]
        self.b1 = [Value(0.0, label=f"b1[{j}]") for j in range(n_hidden)]
        self.W2 = [Value(float(rng.normal(0, scale2)), label=f"W2[{j}]") for j in range(n_hidden)]
        self.b2 = Value(0.0, label="b2")

    def forward(self, x: float) -> Value:
        vals = [x] if not isinstance(x, (list, tuple)) else list(x)
        hidden = []
        for j in range(self.n_hidden):
            z = self.b1[j]
            for k in range(len(vals)):
                z = z + self.W1[j][k] * vals[k]
            hidden.append(z.tanh() if self.activate else z)
        out = self.b2
        for j in range(self.n_hidden):
            out = out + self.W2[j] * hidden[j]
        return out

    def loss(self, xs, ys) -> Value:
        acc = Value(0.0, label="loss")
        for xi, yi in zip(xs, ys):
            e = self.forward(xi) - yi
            acc = acc + e * e
        return acc / len(xs)

    def parameters(self) -> list[Value]:
        return [v for row in self.W1 for v in row] + self.b1 + self.W2 + [self.b2]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


def train_mlp(model: ScalarMLP, xs, ys, iters=4000, lr=0.1, decay=0.0,
              log_every=1000, prefix="训练"):
    curve = []
    for it in range(iters):
        model.zero_grad()
        loss = model.loss(xs, ys)
        loss.backward()
        curve.append(loss.data)
        lr_t = lr / (1.0 + decay * it / iters)
        for p in model.parameters():
            p.data -= lr_t * p.grad
        if log_every:
            print_progress(it + 1, iters, every=log_every, prefix=prefix)
    return curve


def predict(model: ScalarMLP, xs):
    return [model.forward(float(x)).data for x in xs]
