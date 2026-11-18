"""阶段 1-2 用的标量自动微分引擎（micrograd 风格）。

为什么先写标量引擎：反向传播的数学在"一个数对一个数"时最容易看清 ——
每个 Value 记住自己是怎么算出来的（`_op` + `_prev`），
backward() 把这些局部导数按链式法则一路乘下去。

阶段 3 之后换成 numpy 整块数组：数学一模一样，只是每个"数"变成"一摞数"，
因为逐元素建 Python 对象训不动真模型。两套东西的对应关系：

    本文件 Value.grad         →  nn.py 里 Parameter.grad
    本文件 backward() 的链式   →  nn.py 里每个 Module.backward()
    本文件 _prev/_op          →  nn.py 里 forward() 保存的中间量

这个文件不追求快，追求的是你能一眼看懂、并且确信它对。
"""

from __future__ import annotations

import math


class Value:
    """计算图上的一个数，带着能算出它的一切：数值、梯度、父节点、算子名。"""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data: float, _children: tuple = (), _op: str = "", label: str = ""):
        self.data = float(data)
        self.grad = 0.0                    # ∂loss/∂self，backward() 之后才有意义
        self._backward = lambda: None      # 本节点的局部反向规则
        self._prev = set(_children)        # 是由哪些 Value 算出来的
        self._op = _op                     # 用什么算的，只为打印/调试
        self.label = label

    # ------------------------------------------------------------- 四则运算
    # 每个算子都是同一个套路，三行缺一不可：
    #   1) 正向：算出结果 out
    #   2) 写一条规则：out 的梯度如何摊回给操作数
    #   3) 规则里读的必须是 out.grad（"我上游传下来多少"），不是操作数自己的 grad
    # 第 3 点写错的话，所有梯度都会是 0，loss 纹丝不动 —— 而且不会报错。
    # 这就是为什么每个阶段末尾都要做一次 gradient_check_graph。
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __pow__(self, k):
        if not isinstance(k, (int, float)):
            raise TypeError("指数只支持 int/float")
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        denom = other if isinstance(other, Value) else Value(other)
        return self * denom ** -1

    def __lt__(self, other):
        return self.data < (other.data if isinstance(other, Value) else other)

    def __repr__(self):
        tag = f"[{self.label}]" if self.label else ""
        return f"Value{tag}={self.data:.5f}, grad={self.grad:.5f})"

    # ------------------------------------------------------------- 非线性
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def relu(self):
        mask = 1.0 if self.data > 0 else 0.0
        out = Value(self.data if mask else 0.0, (self,), "relu")

        def _backward():
            self.grad += mask * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------- 反向传播
    def backward(self) -> None:
        """从 self（通常是 loss）出发，按逆拓扑序把梯度分发到每个节点。"""
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for c in v._prev:
                build(c)
            topo.append(v)          # 后序：父节点排在孩子后面

        build(self)
        self.grad = 1.0              # ∂loss/∂loss = 1，整条链的起点
        for v in reversed(topo):     # 逆拓扑序 = 从 loss 往参数方向走
            v._backward()

    def parameters(self) -> list["Value"]:
        """图上所有叶子（没有父节点的 = 我们能改的数）。"""
        out, seen = [], set()

        def walk(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            if not v._prev:
                out.append(v)
            for c in v._prev:
                walk(c)

        walk(self)
        return out


def gradient_check_graph(make_graph, params=None, eps: float = 1e-5, tol: float = 1e-4) -> tuple[bool, float, str]:
    """用有限差分验证计算图的反向传播。

    make_graph: 无参数函数，每次调用返回一个全新的 loss Value（可重复构建才能做数值对比）。
    params:     要检查哪些 Value。建议显式传 —— 否则每个数据点也会被当成叶子一起查，
                上百个点会让校验慢几十倍，而且查它们没有意义。
    返回 (是否全部通过, 最大相对误差, 逐参数明细文本)。

    原理：∂f/∂p ≈ (f(p+eps) − f(p−eps)) / 2eps。把它和自动微分给的解析梯度对比。
    相对误差 < 1e-4 基本可以认为反向写对了。
    """
    loss = make_graph()
    loss.backward()
    targets = list(params) if params is not None else loss.parameters()
    analytic = {id(p): p.grad for p in targets}

    worst, lines, ok = 0.0, [], True
    for p in targets:
        ana = analytic[id(p)]
        saved = p.data
        p.data = saved + eps
        hi = make_graph().data
        p.data = saved - eps
        lo = make_graph().data
        p.data = saved
        numeric = (hi - lo) / (2 * eps)
        rel = abs(numeric - ana) / max(1e-8, abs(numeric), abs(ana))
        worst = max(worst, rel)
        if rel > tol:
            ok = False
        lines.append(f"  {p.label or '?':>12}: 解析 {ana:+.7f}  数值 {numeric:+.7f}  相对误差 {rel:.2e}"
                     + ("" if rel <= tol else "   ← 对不上"))
    return ok, worst, "\n".join(lines)
