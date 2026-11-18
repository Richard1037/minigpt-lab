"""数组版网络层：阶段 3 之后的主力。

和 scalar.py 的关系：数学完全一样，只是每个"数"换成了一整块 numpy 数组。
代价是反向传播要你自己写出"对整块数组的梯度"，好处是快一万倍。

============================================================ 统一约定（重要）
1. 激活值形状一律 (B, T, C)：B=batch，T=序列长度，C=通道数(d_model)。
   Linear 接受任意前面维度、最后一维必须是 in_f，所以 (B,T,C) 直接可用。
2. 所有数组都是 **float64、行主序、连续内存**。
   用 float64 是为了让"有限差分梯度校验"干净可信（float32 差分噪声大，
   会让你误以为代码有 bug）。存检查点时再转 float32，那时精度损失是可以接受的。
   （以后要提速就是在这一行改成 float32 + 内存复用，见 docs/advanced-cpp-roadmap.md）
3. Parameter 同时有 .data 和 .grad，形状相同，训练循环负责清零和更新。
4. Module 协议只有两个方法：
       forward(*inputs)      → 返回输出数组，并把反向要用的中间量存到 self._缓存
       backward(grad_out)    → 返回对输入的梯度（没有可导输入就返回 None），
                               并把参数梯度**累加**进 p.grad
   就这两条，任何层都能被 gradcheck 自动验证。
5. 层内不藏随机性：forward 必须是 (参数, 输入) 的纯函数，否则梯度校验做不了。
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------- 参数
class Parameter:
    """一个有权重的张量：数值 + 梯度，形状永远相同。"""

    __slots__ = ("data", "grad", "name", "wd_exempt")

    def __init__(self, data, name: str = "", wd_exempt: bool = False):
        self.data = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
        self.grad = np.zeros_like(self.data)
        self.name = name
        self.wd_exempt = wd_exempt      # norm 权重/bias 不吃 weight decay

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad[...] = 0.0

    def __repr__(self):
        return f"Parameter({self.name or '?'}{self.shape})"


def randn(shape, std: float, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(shape) * std


# ---------------------------------------------------------------- 模块基类
class Module:
    """所有层的基类。子模块/参数通过普通属性赋值挂载，自动被收集。"""

    def __call__(self, *inputs, **kw):
        return self.forward(*inputs, **kw)

    def forward(self, *inputs, **kw):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError

    # ---- 参数收集（checkpoint、优化器、gradcheck 都靠它）----
    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        """递归收集本模块及子模块的参数。

        按 id 去重：weight tying 时输出头"借用"了词嵌入的同一个 Parameter 对象，
        如果不去重，同一个参数会被登记两次，优化器就会对它做两次更新（等效 lr 翻倍），
        检查点里也会存两份。
        """
        return self._collect(prefix, set())

    def _collect(self, prefix: str, seen: set[int]) -> list[tuple[str, Parameter]]:
        found: list[tuple[str, Parameter]] = []
        for key, val in vars(self).items():
            if key.startswith("_"):
                continue
            if isinstance(val, Parameter):
                if id(val) in seen:
                    continue
                seen.add(id(val))
                if val.name == "":
                    val.name = prefix + key
                found.append((prefix + key, val))
            elif isinstance(val, Module):
                found.extend(val._collect(prefix + key + ".", seen))
            elif isinstance(val, (list, tuple)):
                for i, item in enumerate(val):
                    if isinstance(item, Module):
                        found.extend(item._collect(f"{prefix}{key}.{i}.", seen))
        return found

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.named_parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def n_params(self) -> int:
        return sum(p.data.size for p in self.parameters())


# ---------------------------------------------------------------- 具体层
class Scale(Module):
    """y = w · x（答案实现）—— 热身层，见 src/ 那份的注释和讲义 s03a §3。"""

    def __init__(self, w_f: float = 1.0):
        self.w = Parameter(np.array(float(w_f)), name="w")

    def forward(self, x):
        self._x = np.asarray(x, dtype=np.float64)
        return self._x * self.w.data

    def backward(self, g):
        self.w.grad += np.sum(g * self._x)      # 累加，不是覆盖
        return g * self.w.data                  # dx/dw 都不依赖 y 本身，只要 x


class Linear(Module):
    """y = x @ Wᵀ + b。  W 形状 (out_f, in_f)。

    为什么 W 存成 (out, in)：这样 W 的每一**行**就是一个输出神经元的系数，
    梯度 gW = gᵀ @ x 自然得到 (out, in)，和 W 同形，不需要任何转置技巧。
    """

    def __init__(self, in_f: int, out_f: int, use_bias: bool = True, std: float = 0.02,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        # 按 fan-in 缩放的初始化：比纯 0.02 更稳，层深时不会让激活值爆炸
        self.W = Parameter(randn((out_f, in_f), std * math.sqrt(2.0 / in_f), rng))
        self.b = Parameter(np.zeros(out_f), wd_exempt=True) if use_bias else None

    def forward(self, x):
        in_f = self.W.data.shape[1]
        self._in_shape = x.shape
        self._out_shape = x.shape[:-1] + (self.W.data.shape[0],)
        flat = x.reshape(-1, in_f)
        self._x_flat = flat
        y = flat @ self.W.data.T
        if self.b is not None:
            y = y + self.b.data
        return y.reshape(self._out_shape)

    def backward(self, g):
        g2 = g.reshape(-1, self.W.data.shape[0])
        self.W.grad += g2.T @ self._x_flat
        if self.b is not None:
            self.b.grad += g2.sum(axis=0)
        return (g2 @ self.W.data).reshape(self._in_shape)


class Embedding(Module):
    """查表层：token 序号 → 向量。权重形状 (vocab, C)。

    反向是"把梯度散射回对应行"，所以必须用 np.add.at（同一个 id 出现多次要累加）。
    """

    def __init__(self, vocab: int, C: int, std: float = 0.02, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.weight = Parameter(randn((vocab, C), std, rng))

    def forward(self, ids):
        self._ids = np.asarray(ids)
        return self.weight.data[self._ids]

    def backward(self, g):
        gW = np.zeros_like(self.weight.data)
        np.add.at(gW, self._ids.reshape(-1), np.asarray(g).reshape(-1, g.shape[-1]))
        self.weight.grad += gW
        return None                     # 输入是整数序号，不可导


class RMSNorm(Module):
    """y = w · x / sqrt(mean(x²) + eps)。比 LayerNorm 少一次减均值，效果相当。

    反向（推导写在注释里，别背）：
      记 u = g·w，x̂ = r·x，r = 1/sqrt(mean(x²)+eps)，s = mean_lastdim(u·x̂)
      gx = r·u − r·x̂·s        （第二项是"归一化把整体长度锁住"产生的约束）
      gw = Σ_{b,t} g·x̂
    """

    def __init__(self, C: int, eps: float = 1e-5):
        self.w = Parameter(np.ones(C), wd_exempt=True)
        self.eps = eps

    def forward(self, x):
        ms = np.mean(x * x, axis=-1, keepdims=True)
        self._r = 1.0 / np.sqrt(ms + self.eps)
        self._xh = x * self._r
        return self._xh * self.w.data

    def backward(self, g):
        u = g * self.w.data
        s = np.mean(u * self._xh, axis=-1, keepdims=True)
        gx = self._r * u - self._r * self._xh * s
        self.w.grad += (g * self._xh).reshape(-1, g.shape[-1]).sum(axis=0)
        return gx


class GELU(Module):
    """GELU 的 tanh 近似：y = 0.5x(1+tanh(√(2/π)(x + 0.044715x³)))

    用近似式而不是精确 erf 式，是因为它求导后仍然只用乘加，好写也好验证；
    两者最大差异约 1e-3，对训练没有可测影响。
    """

    K = math.sqrt(2.0 / math.pi)
    A = 0.044715

    def forward(self, x):
        inner = self.K * (x + self.A * x ** 3)
        t = np.tanh(inner)
        self._cache = (x, t, inner)
        return 0.5 * x * (1.0 + t)

    def backward(self, g):
        x, t, inner = self._cache
        du_dx = self.K * (1.0 + 3.0 * self.A * x * x)
        dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
        return g * dy_dx


class MLP(Module):
    """逐位置前馈网络：Linear(C→H) → GELU → Linear(H→C)。

    注意它是"对每个 token 独立地"作用的：形状 (B,T,C) 进去，(B,T,C) 出来，
    token 之间不交换信息。交换信息是注意力干的活，这两件事分工明确。
    """

    def __init__(self, C: int, hidden: int, rng: np.random.Generator | None = None):
        self.fc1 = Linear(C, hidden, rng=rng)
        self.act = GELU()
        self.fc2 = Linear(hidden, C, rng=rng)

    def forward(self, x):
        h = self.act(self.fc1(x))
        self._out = h
        return self.fc2(h)

    def backward(self, g):
        g_h = self.fc2.backward(g)
        g_h = self.act.backward(g_h)
        return self.fc1.backward(g_h)


class CrossEntropy(Module):
    """softmax + 负对数似然，融合在一起写。

    为什么要融合而不是分开两层：
      1) 数值稳定 —— 先减去每行最大值再取指数，避免 exp(大数)=inf；
      2) 梯度极其干净 —— logits 的梯度就是 (概率 − one-hot)/样本数，
         单独写 softmax 层反向要处理 jacobi，分开写更容易错。
    反向返回 (B,T,V) 的梯度；targets 是整数，不可导。
    """

    def forward(self, logits, targets):
        targets = np.asarray(targets)
        m = np.max(logits, axis=-1, keepdims=True)
        e = np.exp(logits - m)
        Z = np.sum(e, axis=-1, keepdims=True)
        log_probs = logits - m - np.log(Z)
        self._probs = e / Z
        self._targets = targets
        B, T, _ = logits.shape
        nll = -log_probs.reshape(-1, logits.shape[-1])[np.arange(targets.size), targets.reshape(-1)]
        self._count = targets.size
        return float(np.mean(nll))

    def backward(self, grad_out=1.0):
        B, T, V = self._probs.shape
        onehot = np.zeros_like(self._probs)
        onehot[np.arange(B)[:, None], np.arange(T)[None, :], self._targets] = 1.0
        return (self._probs - onehot) * (float(grad_out) / self._count)


def init_resid_scale(module: Module, n_layer: int) -> None:
    """把每层"残差出口"的投影按 1/sqrt(2·n_layer) 缩小。

    残差是把子层输出直接加回主干，层数一深，主干的方差就一层层累加变大，
    最后训练不稳。这个缩放让"每层加进来的量"和层数无关。
    """
    scale = 1.0 / math.sqrt(2.0 * n_layer)
    for name, p in module.named_parameters():
        if name.endswith("fc2.W") or name.endswith("attn.proj.W"):
            p.data *= scale
