"""你要写的练习都集中在这个文件里（阶段 1 和阶段 2 的动手部分）。

 TODO-1A  fit_by_autograd   阶段 1 · 讲义 §7
 TODO-2A  ScalarMLP.forward 阶段 2 · 讲义 §4
 TODO-2B  train_mlp         阶段 2 · 讲义 §5

做完就跑：
    python stages/s01_fit_line.py
    python stages/s02_fit_curve.py

为什么练习在包里而不是 stage 脚本里、MGPT_ANSWER 开关怎么用：见 README §3。
"""

from __future__ import annotations

import numpy as np

from .scalar import Value
from .toolkit import print_progress


# ---------------------------------------------------------------- 阶段 1 · TODO-1A
def fit_by_autograd(x, y, iters=3000, lr=0.2, log_every=500):
    """用 Value 计算图拟合 y = a·x + b。

    返回 (a, b, curve)，前两个是 **float**（a.data / b.data），不是 Value。
    参数含义、五步循环、手算验证、坑，全在 docs/manual/s01.html#sec-7。
    """
    a = Value(0.0, label="a")
    b = Value(0.0, label="b")
    xs=[float(v) for v in x]
    ys=[float(v) for v in y]
    curve = []

    for i in range(iters):
        for v in (a,b): v.grad=0.0
        loss=Value(0.0,label="loss")

        for xi,yi in zip(xs,ys):
            e=a*xi+b-yi
            loss=loss+e*e
        loss=loss/len(xs)

        curve.append(loss.data)
        loss.backward()

        a.data-=lr*a.grad
        b.data-=lr*b.grad
        if log_every:
            print_progress(i + 1, iters, every=log_every, prefix="自动微分")

    return a.data,b.data,curve

    raise NotImplementedError("TODO-1A: 写 exercises.fit_by_autograd（见 docs/manual/s01.html#sec-7）")


# ---------------------------------------------------------------- 阶段 2 · TODO-2A
class ScalarMLP:
    """一个纯 Python 的小网络：输入 → 隐藏层(tanh) → 输出，参数全是 Value。

    __init__ 已给你写好（只造参数、起名字），要写的是 forward（TODO-2A）。
    `activate=False` 是对照组：两层线性叠起来还是一条直线，墙一点没变矮。
    """

    def __init__(self, n_in=1, n_hidden=6, n_out=1, activate=True, seed=0):
        rng = np.random.default_rng(seed)
        self.n_hidden, self.activate = n_hidden, activate
        scale1 = 1.0 / n_in ** 0.5
        scale2 = 1.0 / n_hidden ** 0.5
        # W1[j][k]: 第 j 个隐藏神经元，第 k 个输入的权重；b1[j]: 它的偏置
        self.W1 = [[Value(float(rng.normal(0, scale1)), label=f"W1[{j}][{k}]")
                    for k in range(n_in)] for j in range(n_hidden)]
        self.b1 = [Value(0.0, label=f"b1[{j}]") for j in range(n_hidden)]
        self.W2 = [Value(float(rng.normal(0, scale2)), label=f"W2[{j}]") for j in range(n_hidden)]
        self.b2 = Value(0.0, label="b2")

    def forward(self, x: float) -> Value:
        """把一个数推到底，返回预测值（Value，不是 float —— predict() 要取 .data）。

        参数表、两条累加式子、三个坑，在 docs/manual/s02.html#sec-4。
        """

        vals=[x] if not isinstance(x,(tuple,list)) else list(x)
        h=[]
        for j in range(self.n_hidden):#枚举所有函数
            z=self.b1[j]
            for k in range(len(vals)):
                z=z+self.W1[j][k]*vals[k]
            h.append(z.tanh() if self.activate else z)
        out=self.b2
        for j in range(self.n_hidden):
            out=out+self.W2[j]*h[j]
        return out

        raise NotImplementedError("TODO-2A: 写 ScalarMLP.forward（见 docs/manual/s02.html#sec-4）")

    def loss(self, xs, ys) -> Value:
        """MSE：每个点算一次预测、减目标、平方，再对点数取平均。"""
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


# ---------------------------------------------------------------- 阶段 2 · TODO-2B
def train_mlp(model: ScalarMLP, xs, ys, iters=4000, lr=0.1, decay=0.0,
              log_every=1000, prefix="训练"):
    """完整批次梯度下降，返回 list[float]：每轮一个 loss，长度必须等于 iters。

    和 fit_by_autograd 是同一个循环，只是参数从 2 个变成 19 个。
    decay 那条 lr 公式、四步顺序和坑，在 docs/manual/s02.html#sec-5。
    """
    curve=[]
    for step in range(iters):
        model.zero_grad()
        loss=model.loss(xs,ys)
        curve.append(loss.data)
        loss.backward()
        lr_t=lr/(1.0+decay*step/iters)
        # lr_t=lr
        for p in model.parameters():
            p.data-=lr_t*p.grad
        if log_every:print_progress(step+1,iters,every=log_every,prefix=prefix)

    return curve

    raise NotImplementedError("TODO-2B: 写 exercises.train_mlp（见 docs/manual/s02.html#sec-5）")


def predict(model: ScalarMLP, xs):
    """给一批 x 返回预测值（普通 float 列表），用来画 ASCII 散点图。"""
    return [model.forward(float(x)).data for x in xs]
