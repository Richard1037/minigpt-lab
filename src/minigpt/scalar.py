"""阶段 1-2 用的标量自动微分引擎（micrograd 风格）。

为什么先写标量引擎：反向传播的数学在"一个数对一个数"时最容易看清 ——
每个 Value 记住自己是怎么算出来的（`_op` + `_prev`），
backward() 把这些局部导数按链式法则一路乘下去。

阶段 3 之后换成 numpy 整块数组：数学一模一样，只是每个"数"变成"一摞数"，
因为逐元素建 Python 对象训不动真模型。两套东西的对应关系：

    本文件 Value.grad         →  nn.py 里 Parameter.grad
    本文件 backward() 的链式   →  nn.py 里每个 Module.backward()
    本文件 _prev/_op          →  nn.py 里 forward() 保存的中间量

============================ 这个文件里哪些是给你的，哪些要你写 ============
已给你（不用改，读懂就行）：Value 的容器、__add__ 的完整写法、减法/除法/负号的
组合写法、exp/relu/sigmoid、parameters()、gradient_check_graph()。
**要你写**（搜 `TODO` 就能全部找到，共四处）：
    TODO-1  __mul__      阶段 1 · 讲义 §4
    TODO-2  __pow__      阶段 1 · 讲义 §5
    TODO-3  backward()   阶段 1 · 讲义 §6
    TODO-4  tanh()       阶段 2 · 讲义 §6
写完就跑 `python stages/s01_fit_line.py`，它自己会告诉你哪里不对。
别先去看 solutions/ —— 那一份是卡住 30 分钟以上才用的答案。
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
    #
    # ---------------- 已经写好的样例：加法，照着它做乘法和幂 ----------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # ∂(a+b)/∂a = 1，所以 a 收到的量 = 1 · (上游传给 out 的量)
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    __radd__ = __add__      # 2 + Value 也走加法

    def __mul__(self, other):
        # ============ TODO-1（阶段 1 · 讲义 §4）============
        # 结构照抄 __add__，只有两处不同：
        #   正向：out.data = ?
        #   反向：∂(a·b)/∂a = b，∂(a·b)/∂b = a  →  各自乘上 out.grad
        # 做完的判据：python stages/s01_fit_line.py 里"两种方法 a 之差"那条通过。
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),"*")

        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad

        out._backward=_backward
        return out

        raise NotImplementedError("TODO-1: 写 scalar.py 的 __mul__（见 docs/manual/s01.html#sec-4）")

    __rmul__ = __mul__      # 2 * Value 也走乘法

    def __pow__(self, k):
        # ============ TODO-2（阶段 1 · 讲义 §5）============
        # 指数只可能是普通数字（int/float），不是 Value，所以只有一个操作数。
        #   正向：out.data = self.data ** k，_op 用 f"**{k}"（打印时好看）
        #   反向：∂(x^k)/∂x = k·x^(k-1)，乘上 out.grad
        # 注意 __truediv__ 靠的是 `self * denom ** -1`，所以你写完这条除法就自动通了。
        out=Value(self.data**k,(self,),f"**{k}")

        def _backward():
            self.grad+=k*(self.data**(k-1))*out.grad

        out._backward=_backward
        return out
    
        raise NotImplementedError("TODO-2: 写 scalar.py 的 __pow__（见 docs/manual/s01.html#sec-5）")

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
    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def relu(self):
        mask = 1.0 if self.data > 0 else 0.0
        out = Value(self.data * mask, (self,), "relu")

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

    def tanh(self):
        # ============ TODO-4（阶段 2 · 讲义 §6）============
        # 照 exp/relu/sigmoid 的写法，三行：算出 t、建 out、写规则。
        #   正向：t = math.tanh(self.data)
        #   反向：∂tanh/∂x = 1 - t²   （注意用的是 t 自己，不用 x）
        # 阶段 2 全靠它，写完跑 python stages/s02_fit_curve.py。
        t=math.tanh(self.data)
        out=Value(t,(self,),"tanh")

        def _backward():
            self.grad+=(1-t**2)*out.grad

        out._backward=_backward
        return out
        raise NotImplementedError("TODO-4: 写 scalar.py 的 tanh()（见 docs/manual/s02.html#sec-6）")

    # ------------------------------------------------------------- 反向传播
    def backward(self):
        # ============ TODO-3（阶段 1 · 讲义 §6）============
        # 这一步是整份手册里最需要你想清楚的三行，别背，自己推一遍：
        #   1) 从 self 出发递归走完所有 _prev，把节点按"孩子在前、父亲在后"排进一个列表
        #      （叫它 topo）。为什么要去重：一个 Value 可以被多个下游用到。
        #   2) 起点是自己：self.grad = 1.0（∂loss/∂loss 当然等于 1）。
        #      这一行漏了，全部梯度都是 0，而且不会报错。
        #   3) 逆着 topo 从后往前调用每个节点的 _backward()。
        #      顺序错了的话，节点在还没收到上游梯度时就被摊派，结果偏小。
        # 判据：s01 的"梯度校验(有限差分)"与"两种方法 a 之差"两条同时通过。
        topo=[]
        visited=set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad=1.0

        for node in reversed(topo):
            node._backward()
            
        # raise NotImplementedError("TODO-3: 写 scalar.py 的 backward()（见 docs/manual/s01.html#sec-6）")

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

    这个函数是给你的"测谎仪"，不用改。它的价值在于：你写 backward 时那种
    "形状对、能跑、loss 也在降"的自信，只有这里能验证。
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
