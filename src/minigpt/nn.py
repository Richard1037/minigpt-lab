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

============================================================ 你要写的部分
阶段 3 拆成了三章，每章末尾有自己的判据。按下面的顺序写，每写完一个就跑对应那一卷 ——
**别一次全写完再测**，七个层里有一个错，报错信息会指向后面所有层。

    TODO-5A Scale.forward/backward        阶段 3a · 讲义 §3   ← 先跑绿这个，流程走通
    TODO-5  Linear.forward                阶段 3a · 讲义 §6
    TODO-6  Linear.backward               阶段 3a · 讲义 §6   ← 本阶段最重要的一条
    TODO-9  GELU.forward/backward         阶段 3b · 讲义 §3
    TODO-8  RMSNorm.forward/backward      阶段 3b · 讲义 §4
    TODO-7  Embedding.forward/backward    阶段 3b · 讲义 §5
    TODO-10 MLP.forward/backward          阶段 3b · 讲义 §6   （就是三条串联）
    TODO-11 CrossEntropy.forward/backward 阶段 3c · 讲义 §3

判据一卷一卷来：tests/test_layers_a.py（8 条）→ _b.py（6 条）→ _c.py（5 条）。

已经给你写好、不需要动的：Parameter、Module（含参数收集）、randn、init_resid_scale。
"""

from __future__ import annotations

import math

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
        """递归收集本模块及子模块的参数，按 id 去重。

        去重是为了 weight tying：输出头"借用"词嵌入的同一个 Parameter 对象，
        如果登记两次，优化器会对它做两次更新（等效 lr 翻倍），检查点里也会存两份。
        """
        return self._collect(prefix, set())

    def _collect(self, prefix: str, seen: set[int]) -> list[tuple[str, Parameter]]:
        found: list[tuple[str, Parameter]] = []
        for key, val in vars(self).items():
            if key.startswith("_"):            # 下划线开头的是缓存，不是子模块
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
    """y = w · x —— 整个手册最小的一个层：一个参数、没有矩阵乘、没有形状变化。

    它存在的唯一理由是**热身**（讲义 s03a §3）。写 Linear 的时候你要同时想三件事：
    摊平成二维、矩阵乘的方向、把形状还原回去 —— 一次学三样很容易崩。Scale 把后两样
    拿掉，只留"一次 forward 配一次 backward、梯度累加、gradcheck 变绿"这一整套流程，
    让你先完整走通一遍，再带着这个手感去写 Linear。

    它不是占位糊弄人的玩具：`tests/test_layers_a.py` 第一组就是在真测它。
    """

    def __init__(self, w_f: float = 1.0):
        self.w = Parameter(np.array(float(w_f)), name="w")

    def forward(self, x):
        # ============ TODO-5A（阶段 3a · 讲义 §3）============
        # 两件事：
        #   1) 把 x 存进 self._x —— 反向要用它。约定：缓存一律下划线开头（s03a §4.1），
        #      否则 Module 的参数收集会把它当成子模块找。
        #   2) return self._x * self.w.data
        # self.w.data 是个 0 维数组，和任何形状相乘都不需要 reshape —— 这就是"没有形状问题"。
        self._x = np.asarray(x, dtype=np.float64)
        return self._x * self.w.data
        raise NotImplementedError("TODO-5A: 写 Scale.forward（见 docs/manual/s03a.html#sec-3）")

    def backward(self, g):
        # ============ TODO-5A（续）============
        # Linear 的"三件产出"在这里缩成两件：
        #   dx = w · g        → 作为返回值，形状和 x 一模一样
        #   dw = Σ(g · x)     → **累加**进 self.w.grad（是 +=，不是 =）
        # 没有 bias，所以第三件不存在。写对了 `python tests/test_layers_a.py` 的第一条就绿。
        self.w.grad += np.sum(g * self._x)      # 累加，不是覆盖
        return g * self.w.data  
        raise NotImplementedError("TODO-5A: 写 Scale.backward（见 docs/manual/s03a.html#sec-3）")


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
        # ============ TODO-5 ============
        # 要做三件事：
        #   1) 记住反向要用的东西（形状、摊平成 (N, in_f) 的输入本身）
        #   2) 算 y = x·Wᵀ (+ b)。(N, in_f) @ (in_f, out_f) → (N, out_f)
        #   3) 把结果 reshape 回"除了最后一维用 out_f 替换"的原形状
        # 坑：(B,T,C) 里 B·T 个位置共享同一份 W，所以要先摊平成二维再做矩阵乘。
        #     另外 W 是 (out,in)，乘之前想清楚要不要 .T。
        self._in_shape=x.shape
        self._out_shape=x.shape[:-1]+(self.W.data.shape[0],)
        flat=x.reshape(-1,x.shape[-1])
        self._flat=flat
        y=flat@self.W.data.T
        if self.b is not None:
            y=y+self.b.data
        return y.reshape(self._out_shape)
        raise NotImplementedError("TODO-5: 写 Linear.forward（见 docs/manual/s03a.html#sec-6）")

    def backward(self, g):
        # ============ TODO-6：本文件最重要的一条 ============
        # 已知 dL/dy（g，形状和 forward 输出相同），要交出三样东西：
        #   dL/dW   → 累加进 self.W.grad，形状必须和 W 一样 (out_f, in_f)
        #   dL/db   → 累加进 self.b.grad（如果 use_bias=False 就没有这个参数）
        #   dL/dx   → 作为返回值，形状必须和 forward 的输入一样
        # 推的时候别背公式，用形状当线索：想让 (out,in) 出现，只能是 gᵀ @ x 这个次序。
        # 如果这一步还是"不知道从哪儿下手"，先读讲义 §2.6：它把标量时代的三条法则
        # 翻译成数组世界的机械步骤（逐元素写出来 → 数下标 → 收回形状），全程不用矩阵微分。
        # 三个高频错：梯度写成覆盖(=)而不是累加(+=)；忘了把 g 摊平；返回值的形状没还原。
        g2=g.reshape(-1,self._out_shape[-1])
        self.W.grad+=g2.T@self._flat
        if self.b is not None:
            self.b.grad+=g2.sum(axis=0)
        return (g2@self.W.data).reshape(self._in_shape)
        raise NotImplementedError("TODO-6: 写 Linear.backward（见 docs/manual/s03a.html#sec-2-6 和 #sec-6）")


class Embedding(Module):
    """查表层：token 序号 → 向量。权重形状 (vocab, C)。

    反向是"把梯度散射回对应行"，所以必须用 np.add.at（同一个 id 出现多次要累加）。
    """

    def __init__(self, vocab: int, C: int, std: float = 0.02, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.weight = Parameter(randn((vocab, C), std, rng))

    def forward(self, ids):
        # ============ TODO-7a ============
        # ids 是整数数组，形状 (B,T)。输出 = 按 id 从权重里取行，形状 (B,T,C)。
        # numpy 一行就够，但要记得把 ids 存下来（反向要用）。
        raise NotImplementedError("TODO-7: 写 Embedding.forward（见 docs/manual/s03b.html#sec-5）")

    def backward(self, g):
        # ============ TODO-7b ============
        # 只有被用到的行有梯度，其余行全 0（这正常，不是 bug）。
        # 关键：同一个 id 在一段里出现 3 次，它那行的梯度是 3 份之和。
        # 用 fancy indexing 赋值 `gW[ids] = ...` 会**覆盖**而不是累加，
        # 必须用 np.add.at(gW, 索引, 值)。这条判据在 test_layers 里专门有一条。
        # 返回 None：输入是整数序号，不可导。
        raise NotImplementedError("TODO-7: 写 Embedding.backward（见 docs/manual/s03b.html#sec-5）")


class RMSNorm(Module):
    """y = w · x / sqrt(mean(x²) + eps)。比 LayerNorm 少一次减均值，效果相当。

    反向要处理"归一化把向量长度锁住"产生的耦合：x 的某个分量变了，
    整行的均方也跟着变，所以每一项都受影响。做法见讲义 §4，推导见 §11.2，结论是：
      记 u = g·w，x̂ = r·x，r = 1/sqrt(mean(x²)+eps)，s = mean_lastdim(u·x̂)
      gx = r·u − r·x̂·s
      gw = Σ_{b,t} g·x̂
    """

    def __init__(self, C: int, eps: float = 1e-5):
        self.w = Parameter(np.ones(C), wd_exempt=True)
        self.eps = eps

    def forward(self, x):
        # ============ TODO-8a ============
        # 缓存反向需要的 r 和 x̂（都带着 (B,T,1)/(B,T,C) 的形状）。
        raise NotImplementedError("TODO-8: 写 RMSNorm.forward（见 docs/manual/s03b.html#sec-4）")

    def backward(self, g):
        # ============ TODO-8b ============
        # 上面那句 gx 的式子里，减掉的第二项就是"长度被锁住"的代价。
        # 如果你只写 r*u，梯度校验会失败 —— 这是新手最常见的一处"看起来挺合理"的错。
        raise NotImplementedError("TODO-8: 写 RMSNorm.backward（见 docs/manual/s03b.html#sec-4）")


class GELU(Module):
    """GELU 的 tanh 近似：y = 0.5x(1+tanh(√(2/π)(x + 0.044715x³)))

    用近似式而不是精确 erf 式，是因为它求导后仍然只用乘加，好写也好验证；
    两者最大差异约 1e-3，对训练没有可测影响。
    """

    K = math.sqrt(2.0 / math.pi)
    A = 0.044715

    def forward(self, x):
        # ============ TODO-9a ============
        # 缓存 (x, tanh(inner), inner) 三样，backward 全要用。
        inner = self.K * (x + self.A * x ** 3)
        t = np.tanh(inner)
        self._cache = (x, t, inner)
        return 0.5 * x * (1.0 + t)

        raise NotImplementedError("TODO-9: 写 GELU.forward（见 docs/manual/s03b.html#sec-3）")

    def backward(self, g):
        # ============ TODO-9b ============
        # 乘积求导 + 链式，两项：0.5(1+t) + 0.5x·(1-t²)·dinner/dx，
        # dinner/dx = K(1 + 3A x²)。没有参数，所以只 return 不累加 grad。
        x, t, inner = self._cache
        du_dx = self.K * (1.0 + 3.0 * self.A * x * x)
        dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
        return g * dy_dx
        raise NotImplementedError("TODO-9: 写 GELU.backward（见 docs/manual/s03b.html#sec-3）")


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
        # ============ TODO-10a ============
        # 三条串联，别自己加残差（残差是 Block 的事）。
        raise NotImplementedError("TODO-10: 写 MLP.forward（见 docs/manual/s03b.html#sec-6）")

    def backward(self, g):
        # ============ TODO-10b ============
        # 反着走一遍：先 fc2，再 act，再 fc1。每步的返回值是下一步的入参。
        # 你会看到"反向 = 正向的镜像"这个规律在整份手册里到处成立。
        raise NotImplementedError("TODO-10: 写 MLP.backward（见 docs/manual/s03b.html#sec-6）")


class CrossEntropy(Module):
    """softmax + 负对数似然，融合在一起写。

    为什么要融合而不是分开两层：
      1) 数值稳定 —— 先减去每行最大值再取指数，避免 exp(大数)=inf；
      2) 梯度极其干净 —— logits 的梯度就是 (概率 − one-hot)/样本数，
         单独写 softmax 层反向要处理 jacobi，分开写更容易错。
    反向返回 (B,T,V) 的梯度；targets 是整数，不可导。
    """

    def forward(self, logits, targets):
        # ============ TODO-11a ============
        # logits (B,T,V)，targets (B,T) 整数。返回一个标量 float：平均负对数似然。
        # 三个坑：
        #   · 忘记减最大值 → exp 溢出成 inf，loss 变 nan
        #   · 用 log(sum(exp)) 而不是 log(Z) 的稳定写法（减完最大值后 log(Z) 是安全的）
        #   · 取"正确词的 log 概率"时索引方式错了：把 (B,T,V) 摊成 (B*T, V) 再用
        #     arange 配合 targets 摊平，是最不容易错的写法
        # 自查：logits 全 0 时 loss 必须正好等于 ln(V)（test_layers 有这条）。
        raise NotImplementedError("TODO-11: 写 CrossEntropy.forward（见 docs/manual/s03c.html#sec-3）")

    def backward(self, grad_out=1.0):
        # ============ TODO-11b ============
        # 缓存 forward 里算出的 probs 和 targets，返回 (probs − one_hot) / 样本数。
        # 为什么除以样本数：forward 用的是 mean，不是 sum。这一步漏了就等于
        # 悄悄把 lr 放大了 batch·T 倍，症状是"训练一开始就炸"。
        raise NotImplementedError("TODO-11: 写 CrossEntropy.backward（见 docs/manual/s03c.html#sec-3）")


def init_resid_scale(module: Module, n_layer: int) -> None:
    """把每层"残差出口"的投影按 1/sqrt(2·n_layer) 缩小。

    残差是把子层输出直接加回主干，层数一深，主干的方差就一层层累加变大，
    最后训练不稳。这个缩放让"每层加进来的量"和层数无关。
    （已给你写好，阶段 6 的讲义会讲它为什么长这样。）
    """
    scale = 1.0 / math.sqrt(2.0 * n_layer)
    for name, p in module.named_parameters():
        if name.endswith("fc2.W") or name.endswith("attn.proj.W"):
            p.data *= scale
