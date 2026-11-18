"""优化器与学习率调度：把梯度变成参数更新。

为什么默认用 AdamW 而不是 SGD：transformer 的每一层尺度差别很大（注意力分数、
MLP 激活、嵌入），SGD 一个全局步长很难同时照顾好；Adam 给每个参数自己估一个
步长，对学习率的容忍度高得多，小模型上也更稳。

weight decay 为什么单独拎出来讲：把它从"梯度里加一项 λw"改成"更新时直接
w *= (1 - lr·λ)"才是 AdamW，两者在 Adam 里**不等价**（前者会被二阶矩缩放掉）。
而且 bias 和 norm 的缩放系数不该被衰减 —— 它们不是"权重矩阵"，衰减它们只会
让模型变笨。这个豁免标记在 Parameter.wd_exempt 里。
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .nn import Module, Parameter


def _as_params(target) -> list[Parameter]:
    if isinstance(target, Module):
        return target.parameters()
    if isinstance(target, Parameter):
        return [target]
    params = list(target)
    if any(not isinstance(p, Parameter) for p in params):
        raise TypeError("优化器只接受 Module / Parameter / Parameter 列表")
    return params


def _zero_bias(params: Iterable[Parameter]):
    return [np.zeros_like(p.data) for p in params]


class Optimizer:
    """共同接口：step() 更新参数，zero_grad() 清梯度。

    lr 既能在构造时给定，也能每次 step 传入（配合 lr_at 做 warmup+cosine）。
    """

    def __init__(self, target, lr: float = 3e-4, weight_decay: float = 0.1):
        self.params = _as_params(target)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.step_count = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def _decay(self, p: Parameter, lr: float) -> None:
        if self.weight_decay and not p.wd_exempt:
            p.data *= (1.0 - lr * self.weight_decay)

    def step(self, lr: float | None = None) -> float:
        raise NotImplementedError


class SGD(Optimizer):
    """带动量的随机梯度下降。小实验里比 Adam 更好观察，因为每步就是"沿梯度走"。"""

    def __init__(self, target, lr: float = 1e-2, weight_decay: float = 0.0,
                 momentum: float = 0.9):
        super().__init__(target, lr, weight_decay)
        self.momentum = momentum
        self.buf = [np.zeros_like(p.data) for p in self.params]

    def step(self, lr: float | None = None) -> float:
        lr = self.lr if lr is None else float(lr)
        self.step_count += 1
        for p, b in zip(self.params, self.buf):
            self._decay(p, lr)
            b *= self.momentum
            b += p.grad
            p.data -= lr * b
        return lr


class AdamW(Optimizer):
    """Adam + 解耦权重衰减。β2=0.95 而不是默认 0.999：小模型步数少，
    0.999 的二阶矩"记忆"太长，前期估计不准，0.9~0.95 收敛更快。
    """

    def __init__(self, target, lr: float = 3e-4, weight_decay: float = 0.1,
                 beta1: float = 0.9, beta2: float = 0.95, eps: float = 1e-8):
        super().__init__(target, lr, weight_decay)
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = _zero_bias(self.params)
        self.v = _zero_bias(self.params)

    def step(self, lr: float | None = None) -> float:
        lr = self.lr if lr is None else float(lr)
        self.step_count += 1
        t = self.step_count
        # 偏差修正：初始 m、v 都是 0，前几步的均值被严重低估，要除以 (1-β^t)
        c1 = 1.0 - self.b1 ** t
        c2 = 1.0 - self.b2 ** t
        for p, mi, vi in zip(self.params, self.m, self.v):
            self._decay(p, lr)
            g = p.grad
            mi[...] = self.b1 * mi + (1.0 - self.b1) * g
            vi[...] = self.b2 * vi + (1.0 - self.b2) * g * g
            p.data -= lr * (mi / c1) / (np.sqrt(vi / c2) + self.eps)
        return lr


def clip_grad_norm(target, max_norm: float) -> float:
    """按全局 L2 范数裁剪梯度，返回裁剪前的范数。

    为什么要：语言模型偶发"某一批数据特别难"，梯度瞬间变大，一步把模型踹进
    坏区域，loss 曲线出现一根很高的尖刺。裁剪不改变方向，只限制步幅。
    注意顺序：**先算梯度、裁剪，再 optimizer.step()**；反过来等于没裁。
    """
    params = _as_params(target)
    total = math.sqrt(float(sum(np.sum(p.grad * p.grad) for p in params)))
    if total > max_norm and total > 0.0:
        scale = max_norm / total
        for p in params:
            p.grad *= scale
    return total


def lr_at(step: int, total_steps: int, peak: float, warmup: int = 400,
          final_frac: float = 0.1) -> float:
    """线性 warmup → cosine 衰减到 peak·final_frac。

    warmup 的作用：Adam 的二阶矩在最初几十步还没估准，这时用大 lr 容易把
    第一层的权重打飞；先用小 lr 走几步"热身"，稳定性提升非常明显。
    cosine 尾端不归零而是留 10%，是因为后期需要一点步长继续微调，
    归零会让模型卡在离最优点还有一小段的地方。
    """
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if total_steps <= warmup:
        return peak
    frac = (step - warmup) / max(1, total_steps - warmup)
    frac = min(1.0, max(0.0, frac))
    cos = 0.5 * (1.0 + math.cos(math.pi * frac))
    return peak * (final_frac + (1.0 - final_frac) * cos)
